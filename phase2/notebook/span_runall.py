"""Embedded, self-contained Kaggle run for a Phase-1-base-03 span extractor.

This file is embedded into CS781_Phase2_Span_Extractor_RunAll.ipynb by its builder.
"""

# %% 1 - runtime and file contracts
from __future__ import annotations

import bisect
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

WORKING = Path('/kaggle/working')
PREVIOUS = WORKING / 'CS781_Capstone' / 'phase2_improved'
ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v1'
BASE_DIR = PREVIOUS / 'stages' / 'final_slot03' / 'phase1_bases' / '03'
BASE_CHECKPOINT = BASE_DIR / 'checkpoints' / 'selected_model.pt'
MODEL_SOURCE = BASE_DIR / 'model' / 'prefix_hybrid_lm.py'
BPE_SOURCE = BASE_DIR / 'model' / 'course_bpe.py'
SOURCE_DIR = PREVIOUS / 'official_phase2'
REPORT_ZIP = WORKING / 'CS781_Phase2_Span_Extractor_Report.zip'
SEED = 781
VERSION = 'span-start-end-pair-v1'
RUN_STARTED = time.time()
SESSION_GUARD_SECONDS = 31_200
MIN_FREE_GIB = 2.0

# This is an additional experiment. The previous six distinct methods remain
# available and are not overwritten. Only your Phase-1 base-03 weights are used.
MAX_EPOCHS = 24
MIN_EPOCHS = 8
PATIENCE = 5
VALIDATE_EVERY = 2
MIN_FINAL_CV_F1 = 0.34  # only fit the official test when the old local baseline is beaten


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def free_gib():
    return shutil.disk_usage(WORKING).free / 2**30


def load_source(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


needed = [BASE_CHECKPOINT, MODEL_SOURCE, BPE_SOURCE,
          BASE_DIR / 'tokenizer' / 'tokenizer.json',
          BASE_DIR / 'tokenizer' / 'tokenizer_config.json',
          BASE_DIR / 'tokenizer' / 'tokenizer_metadata.json',
          PREVIOUS / 'cv_folds.json', PREVIOUS / 'improvement_status.json',
          SOURCE_DIR / 'labels.txt', SOURCE_DIR / 'fold3-text.json']
for part in (1, 2):
    needed.extend([SOURCE_DIR / f'fold{part}-text.json',
                   SOURCE_DIR / f'fold{part}-gold.json'])
missing = [str(path) for path in needed if not path.is_file()]
if missing:
    raise FileNotFoundError('The saved Kaggle session is incomplete:\n' + '\n'.join(missing))
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError('Select the Kaggle two-T4 accelerator before running this notebook.')
if free_gib() < MIN_FREE_GIB:
    raise RuntimeError(f'Only {free_gib():.2f} GiB free; stop before writing checkpoints.')
ROOT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
print('Saved run:', PREVIOUS)
print('New run:', ROOT)
print('Disk free:', f'{free_gib():.2f} GiB')
for device_id in (0, 1):
    available, total = torch.cuda.mem_get_info(device_id)
    print(f'cuda:{device_id}: {torch.cuda.get_device_name(device_id)} '
          f'{available/2**30:.2f}/{total/2**30:.2f} GiB free')
print('FP32 training; both GPUs via DataParallel. Old artifacts are read only.')

# %% 2 - official data and Phase-1 model source
fold_manifest = read_json(PREVIOUS / 'cv_folds.json')
old_status = read_json(PREVIOUS / 'improvement_status.json')
OLD_OOF = float(old_status['completed']['3']['cv']['baseline_oof_macro_f1'])
LABELS = tuple(line.strip() for line in (SOURCE_DIR / 'labels.txt').read_text(encoding='utf-8').splitlines()
               if line.strip())
if len(LABELS) != 12 or len(set(LABELS)) != 12:
    raise RuntimeError('The 12 official labels were not found')
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
texts, gold = {}, {}
for part in (1, 2):
    part_text = read_json(SOURCE_DIR / f'fold{part}-text.json')
    part_gold = read_json(SOURCE_DIR / f'fold{part}-gold.json')
    if set(part_text) != set(part_gold) or set(texts) & set(part_text):
        raise RuntimeError('Official labeled document IDs are inconsistent')
    texts.update(part_text)
    gold.update({doc_id: [(int(row[0]), int(row[1]), row[2]) for row in rows]
                 for doc_id, rows in part_gold.items()})
test_texts = read_json(SOURCE_DIR / 'fold3-text.json')
FOLDS = [tuple(str(doc_id) for doc_id in group) for group in fold_manifest['folds']]
if len(FOLDS) != 3 or set(x for group in FOLDS for x in group) != set(texts) or sum(map(len, FOLDS)) != len(texts):
    raise RuntimeError('The saved document folds do not partition the official training data')
if set(test_texts) & set(texts):
    raise RuntimeError('Private test document IDs overlap public labeled training IDs')
for doc_id, rows in gold.items():
    text = texts[doc_id]
    for a, b, name in rows:
        if name not in LABEL_ID or not 0 <= a < b <= len(text):
            raise RuntimeError(f'Invalid gold span in {doc_id}: {(a,b,name)}')

model_sha = file_sha(MODEL_SOURCE)
bpe_sha = file_sha(BPE_SOURCE)
base_sha = file_sha(BASE_CHECKPOINT)
model_module = load_source(MODEL_SOURCE, 'cs781_span_base_' + model_sha[:12])
bpe_module = load_source(BPE_SOURCE, 'cs781_span_bpe_' + bpe_sha[:12])
tokenizer = bpe_module.CourseBPETokenizer.from_pretrained(BASE_DIR / 'tokenizer')
tokenizer_meta = read_json(BASE_DIR / 'tokenizer' / 'tokenizer_metadata.json')
base_record = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
if base_record.get('model_source_sha256') != model_sha:
    raise RuntimeError('Base-03 checkpoint/model source SHA256 mismatch')
if base_record.get('tokenizer_fingerprint') != tokenizer_meta.get('fingerprint'):
    raise RuntimeError('Base-03 checkpoint/tokenizer fingerprint mismatch')
if len(tokenizer) != int(base_record['model_config']['vocab_size']):
    raise RuntimeError('Base-03 tokenizer vocabulary mismatch')
BASE_CONFIG = base_record['model_config']
del base_record

data_hash = hashlib.sha256(json.dumps({
    'folds': fold_manifest, 'labels': LABELS,
    'sources': {path.name: file_sha(path) for path in needed if path.parent == SOURCE_DIR}
}, sort_keys=True).encode()).hexdigest()

CONFIG = {
    'version': VERSION, 'base_id': '03', 'adapter': 'LoRA query/value',
    'window_tokens': 384, 'window_stride': 288,
    'max_span_tokens': 64, 'lora_rank': 8, 'lora_alpha': 16.0,
    'batch_size': 16, 'epochs': MAX_EPOCHS, 'min_epochs': MIN_EPOCHS,
    'patience': PATIENCE, 'validate_every': VALIDATE_EVERY,
    'lora_lr': 1.5e-4, 'head_lr': 4e-4, 'weight_decay': 0.01,
    'warmup_fraction': 0.05, 'gradient_clip_norm': 1.0,
    'head_dropout': 0.15, 'pair_hidden': 128,
    'start_topk': 64, 'end_topk': 64, 'pair_topk_per_label': 384,
    'keep_per_label_per_window': 48,
    'inference_window_batch_size': 8,
    'default_threshold': 0.45,
    'min_final_cv_f1': MIN_FINAL_CV_F1,
    'seed': SEED, 'precision': 'float32', 'activation_checkpointing': True,
    'base_checkpoint_sha256': base_sha, 'model_source_sha256': model_sha,
    'bpe_source_sha256': bpe_sha, 'tokenizer_fingerprint': tokenizer_meta['fingerprint'],
    'data_hash': data_hash,
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
config_path = ROOT / 'experiment_config.json'
if config_path.exists() and read_json(config_path) != CONFIG:
    raise RuntimeError('Existing span-run configuration differs; refusing unsafe resume')
atomic_json(config_path, CONFIG)
print('Documents:', len(texts), 'test:', len(test_texts), 'folds:', [len(group) for group in FOLDS])
print('Old slot-03 default OOF F1:', f'{OLD_OOF:.6f}')
print('New experiment SHA:', CONFIG_HASH)

# %% 3 - exact token-span alignment and exhaustive windows
ENCODINGS = {}
ALIGNED = {}
alignment_counts = Counter()


def encode_doc(doc_id, text):
    if doc_id not in ENCODINGS:
        encoded = tokenizer.backend.encode(text, add_special_tokens=False)
        ids = tuple(int(x) for x in encoded.ids)
        offsets = tuple((int(a), int(b)) for a, b in encoded.offsets)
        if (not ids and text) or len(ids) != len(offsets) or any(a >= b for a, b in offsets):
            raise RuntimeError(f'Invalid token offsets in document {doc_id}')
        ENCODINGS[doc_id] = (ids, offsets)
    return ENCODINGS[doc_id]


def align_gold(doc_id):
    if doc_id in ALIGNED:
        return ALIGNED[doc_id]
    ids, offsets = encode_doc(doc_id, texts[doc_id])
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]
    aligned = []
    for a, b, name in gold[doc_id]:
        i = bisect.bisect_right(ends, a)
        j = bisect.bisect_left(starts, b) - 1
        if not (0 <= i <= j < len(ids) and starts[i] <= a and b <= ends[j]):
            raise RuntimeError(f'Gold span has no token alignment in {doc_id}: {(a,b,name)}')
        if j - i + 1 > CONFIG['max_span_tokens']:
            raise RuntimeError(f'Gold span exceeds max token width in {doc_id}: {(a,b,name)}')
        left, right = starts[i], ends[j]
        while left < right and texts[doc_id][left].isspace():
            left += 1
        while right > left and texts[doc_id][right - 1].isspace():
            right -= 1
        alignment_counts['total'] += 1
        alignment_counts['trim_exact'] += int((left, right) == (a, b))
        aligned.append((i, j, LABEL_ID[name], a, b))
    ALIGNED[doc_id] = tuple(aligned)
    return ALIGNED[doc_id]


def window_bounds(length, size, stride):
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(start, min(start + size, length)) for start in starts]


def make_windows(doc_ids, training):
    rows, rare_rows, uncovered = [], [], []
    rare = {LABEL_ID[name] for name in ('APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT')}
    for doc_id in sorted(doc_ids):
        ids, offsets = encode_doc(doc_id, texts[doc_id] if doc_id in texts else test_texts[doc_id])
        spans = align_gold(doc_id) if training else ()
        bounds = window_bounds(len(ids), CONFIG['window_tokens'], CONFIG['window_stride'])
        covered = [False] * len(spans)
        seen_bounds = set()
        def add_window(lo, hi):
            if (lo, hi) in seen_bounds:
                return
            seen_bounds.add((lo, hi))
            targets = []
            for position, (i, j, label, a, b) in enumerate(spans):
                if lo <= i and j < hi:
                    targets.append((i - lo, j - lo, label, a, b))
                    covered[position] = True
            row = {'id': f'{doc_id}__{lo:06d}', 'doc_id': doc_id, 'token_lo': lo,
                   'ids': ids[lo:hi], 'offsets': offsets[lo:hi], 'targets': tuple(targets)}
            rows.append(row)
            if any(t[2] in rare for t in targets):
                rare_rows.append(row)
        for lo, hi in bounds:
            add_window(lo, hi)
        if training:
            for position, (i, j, label, a, b) in enumerate(spans):
                if covered[position]:
                    continue
                lo = max(0, min(i, len(ids) - CONFIG['window_tokens']))
                add_window(lo, min(lo + CONFIG['window_tokens'], len(ids)))
            for position, flag in enumerate(covered):
                if not flag:
                    uncovered.append((doc_id, spans[position]))
    if uncovered:
        raise RuntimeError('Gold spans missing from training windows: ' + repr(uncovered[:5]))
    if training and rare_rows:
        # One extra visit for at most a quarter of the original windows.
        sample_count = min(len(rare_rows), len(rows) // 4)
        rng = random.Random(SEED + len(rows))
        for index, row in enumerate(rng.sample(rare_rows, sample_count)):
            rows.append(dict(row, id=row['id'] + f'__rare{index:04d}'))
    return rows


for doc_id in texts:
    align_gold(doc_id)
if alignment_counts['total'] != sum(map(len, gold.values())):
    raise RuntimeError('Gold alignment audit is incomplete')
print('Gold token alignment:', alignment_counts['total'], 'of', alignment_counts['total'])
print('Whitespace-trim token boundary exact rate:',
      f"{alignment_counts['trim_exact']/alignment_counts['total']:.4f}")
print('Longest public document tokens:', max(len(encode_doc(doc_id, text)[0]) for doc_id, text in texts.items()))

# %% 4 - LoRA base and directly supervised span head
class AdditiveLoRA(nn.Module):
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.base = original
        self.a = nn.Linear(original.in_features, rank, bias=False)
        self.b = nn.Linear(rank, original.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class SpanExtractor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        changed = 0
        for block in base.blocks:
            for name in ('query', 'value'):
                layer = getattr(block.attention, name)
                if not isinstance(layer, nn.Linear):
                    raise RuntimeError(f'Unexpected base-03 attention projection {name}')
                setattr(block.attention, name,
                        AdditiveLoRA(layer, CONFIG['lora_rank'], CONFIG['lora_alpha']))
                changed += 1
        if changed != 2 * len(base.blocks):
            raise RuntimeError('LoRA was not applied to every base-03 attention block')
        hidden = base.hidden_size
        self.dropout = nn.Dropout(CONFIG['head_dropout'])
        self.start_head = nn.Linear(hidden, len(LABELS))
        self.end_head = nn.Linear(hidden, len(LABELS))
        self.width_embedding = nn.Embedding(CONFIG['max_span_tokens'] + 1, 16)
        self.pair_head = nn.Sequential(
            nn.Linear(3 * hidden + 16, CONFIG['pair_hidden']),
            nn.GELU(), nn.Dropout(CONFIG['head_dropout']),
            nn.Linear(CONFIG['pair_hidden'], len(LABELS)))

    def features(self, input_ids, attention_mask):
        base = self.base
        length = input_ids.shape[1]
        visibility = base._prefix_hybrid_visibility(attention_mask, attention_mask.sum(1))
        states = base.embedding_dropout(base.token_embedding(input_ids) * math.sqrt(base.hidden_size))
        cosine, sine = base.rotary(length, states.dtype)
        for block in base.blocks:
            if self.training and torch.is_grad_enabled() and CONFIG['activation_checkpointing']:
                states = activation_checkpoint(block, states, visibility, cosine, sine,
                                               use_reentrant=False)
            else:
                states = block(states, visibility, cosine, sine)
        return self.dropout(base.final_norm(states))

    def score_pairs(self, hidden, pair_index):
        width = hidden.shape[-1]
        left = pair_index[..., 0].unsqueeze(-1).expand(-1, -1, width)
        right = pair_index[..., 1].unsqueeze(-1).expand(-1, -1, width)
        start = hidden.gather(1, left)
        end = hidden.gather(1, right)
        distance = (pair_index[..., 1] - pair_index[..., 0]).clamp(
            min=0, max=CONFIG['max_span_tokens'])
        features = torch.cat((start, end, start * end,
                              self.width_embedding(distance)), dim=-1)
        return self.pair_head(features)

    def forward(self, input_ids, attention_mask, pair_index=None, return_hidden=False):
        hidden = self.features(input_ids, attention_mask)
        output = {'start': self.start_head(hidden), 'end': self.end_head(hidden)}
        if pair_index is not None:
            output['pair'] = self.score_pairs(hidden, pair_index)
        if return_hidden:
            output['hidden'] = hidden
        return output


def fresh_model():
    record = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
    if record.get('model_source_sha256') != model_sha or record.get('tokenizer_fingerprint') != tokenizer_meta['fingerprint']:
        raise RuntimeError('Phase-1 base changed while this notebook was running')
    base = model_module.build_model(record['model_config'])
    base.load_state_dict(record['model_state'], strict=True)
    del record
    model = SpanExtractor(base)
    adapted = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LoRA + span-head trainable parameters: {adapted:,} / '
          f'{sum(p.numel() for p in model.parameters()):,}')
    return model


def trainable_state(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name in names}


def load_trainable_state(model, saved):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(saved) != names:
        raise RuntimeError('Adapter/head state keys differ from this model source')
    current = model.state_dict()
    for name, value in saved.items():
        if current[name].shape != value.shape:
            raise RuntimeError(f'Adapter/head shape mismatch: {name}')
        current[name] = value
    model.load_state_dict(current, strict=True)


def label_weights(train_ids):
    counts = Counter(name for doc_id in train_ids for _, _, name in gold[doc_id])
    positive = sorted(counts.values())
    median = positive[len(positive)//2]
    return torch.tensor([min(3.0, max(0.6, math.sqrt(median / max(1, counts[name]))))
                         for name in LABELS], dtype=torch.float32)


def make_pair_examples(row, epoch):
    length = len(row['ids'])
    positives = {(int(i), int(j), int(label)) for i, j, label, a, b in row['targets']}
    pairs = {(i, j) for i, j, _ in positives}
    for i, j, _ in positives:
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                x, y = i + di, j + dj
                if 0 <= x <= y < length and y - x < CONFIG['max_span_tokens']:
                    pairs.add((x, y))
    stable = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
    rng = random.Random(SEED + stable + epoch * 100_003)
    for _ in range(max(50, 3 * len(positives))):
        x = rng.randrange(length)
        y = min(length - 1, x + rng.randrange(CONFIG['max_span_tokens']))
        pairs.add((x, y))
    if len(pairs) > 320:
        # Keep every positive and close boundary perturbation first. Random
        # negatives are then capped to keep DataParallel batches predictable.
        priority = sorted(pairs, key=lambda pair: min(
            (abs(pair[0]-i) + abs(pair[1]-j) for i, j, _ in positives),
            default=10_000))
        pairs = set(priority[:320])
    ordered = sorted(pairs)
    targets = [[int((i, j, label) in positives) for label in range(len(LABELS))]
               for i, j in ordered]
    return ordered, targets


def collate_windows(rows, epoch):
    batch_size = len(rows)
    length = max(len(row['ids']) for row in rows)
    input_ids = torch.full((batch_size, length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, length), dtype=torch.bool)
    starts = torch.zeros((batch_size, length, len(LABELS)), dtype=torch.float32)
    ends = torch.zeros_like(starts)
    examples = [make_pair_examples(row, epoch) for row in rows]
    max_pairs = max(len(item[0]) for item in examples)
    pair_index = torch.zeros((batch_size, max_pairs, 2), dtype=torch.long)
    pair_targets = torch.zeros((batch_size, max_pairs, len(LABELS)), dtype=torch.float32)
    pair_mask = torch.zeros((batch_size, max_pairs), dtype=torch.bool)
    for batch_index, row in enumerate(rows):
        size = len(row['ids'])
        input_ids[batch_index, :size] = torch.tensor(row['ids'], dtype=torch.long)
        attention_mask[batch_index, :size] = True
        for i, j, label, a, b in row['targets']:
            starts[batch_index, i, label] = 1
            ends[batch_index, j, label] = 1
        pairs, targets = examples[batch_index]
        pair_index[batch_index, :len(pairs)] = torch.tensor(pairs, dtype=torch.long)
        pair_targets[batch_index, :len(pairs)] = torch.tensor(targets, dtype=torch.float32)
        pair_mask[batch_index, :len(pairs)] = True
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
            'start_targets': starts, 'end_targets': ends,
            'pair_index': pair_index, 'pair_targets': pair_targets,
            'pair_mask': pair_mask}


def balanced_loss(logits, targets, valid_mask, class_weights):
    valid = valid_mask.unsqueeze(-1).expand_as(targets)
    positive = valid & targets.bool()
    negative = valid & ~targets.bool()
    weight = class_weights.view(*([1] * (logits.ndim - 1)), -1)
    positive_loss = (F.softplus(-logits.float()) * positive * weight).sum() / (
        (positive * weight).sum().clamp_min(1))
    negative_values = F.softplus(logits.float())[negative]
    if negative_values.numel():
        k = min(negative_values.numel(), max(64, int(positive.sum().item()) * 5))
        negative_loss = negative_values.topk(k).values.mean()
    else:
        negative_loss = logits.new_zeros(())
    return positive_loss + 0.35 * negative_loss


def task_loss(output, batch, class_weights):
    start_loss = balanced_loss(output['start'], batch['start_targets'],
                               batch['attention_mask'], class_weights)
    end_loss = balanced_loss(output['end'], batch['end_targets'],
                             batch['attention_mask'], class_weights)
    pair_loss = balanced_loss(output['pair'], batch['pair_targets'],
                              batch['pair_mask'], class_weights)
    return (start_loss + end_loss + pair_loss) / 3

# %% 5 - fast two-GPU inference and strict-span diagnostics
def inference_batch(rows):
    width = max(len(row['ids']) for row in rows)
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    masks = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids[index, :len(row['ids'])] = torch.tensor(row['ids'], dtype=torch.long)
        masks[index, :len(row['ids'])] = True
    return ids.to('cuda:0', non_blocking=True), masks.to('cuda:0', non_blocking=True)


def candidate_pairs(start_logits, end_logits, length):
    """Prune the cross product before the more expensive learned pair head."""
    by_label = {}
    unique_pairs = set()
    start_logits, end_logits = start_logits[:length].float(), end_logits[:length].float()
    for label in range(len(LABELS)):
        starts = torch.topk(start_logits[:, label], min(CONFIG['start_topk'], length)).indices
        ends = torch.topk(end_logits[:, label], min(CONFIG['end_topk'], length)).indices
        span_widths = ends[None, :] - starts[:, None]
        allowed = (span_widths >= 0) & (span_widths < CONFIG['max_span_tokens'])
        pair_count = min(CONFIG['pair_topk_per_label'], int(allowed.sum()))
        if pair_count:
            scores = start_logits[starts, label][:, None] + end_logits[ends, label][None, :]
            scores = scores.masked_fill(~allowed, float('-inf'))
            values, indexes = torch.topk(scores.flatten(), pair_count)
            start_positions = starts[indexes // len(ends)]
            end_positions = ends[indexes % len(ends)]
            kept = [(float(v), int(i), int(j))
                    for v, i, j in zip(values, start_positions, end_positions)]
        else:
            kept = []
        by_label[label] = kept
        unique_pairs.update((i, j) for _, i, j in kept)
    return by_label, sorted(unique_pairs)


def score_window_candidates(model, hidden, start_logits, end_logits, row, source_text):
    length = len(row['ids'])
    by_label, pairs = candidate_pairs(start_logits, end_logits, length)
    if not pairs:
        return []
    pair_scores = []
    for begin in range(0, len(pairs), 1024):
        group = pairs[begin:begin + 1024]
        indexes = torch.tensor(group, dtype=torch.long, device='cuda:0').unsqueeze(0)
        pair_scores.append(model.score_pairs(hidden.unsqueeze(0), indexes)[0].float().cpu())
    pair_scores = torch.cat(pair_scores, dim=0)
    pair_positions = {pair: index for index, pair in enumerate(pairs)}
    scored = []
    offsets = row['offsets']
    for label, proposals in by_label.items():
        ranked = []
        for _, i, j in proposals:
            pair_logit = float(pair_scores[pair_positions[(i, j)], label])
            combined_logit = (float(start_logits[i, label])
                              + float(end_logits[j, label]) + pair_logit) / 3
            confidence = 1 / (1 + math.exp(-max(-50, min(50, combined_logit))))
            ranked.append((confidence, i, j))
        ranked.sort(reverse=True)
        for confidence, i, j in ranked[:CONFIG['keep_per_label_per_window']]:
            a, b = offsets[i][0], offsets[j][1]
            while a < b and source_text[a].isspace():
                a += 1
            while b > a and source_text[b - 1].isspace():
                b -= 1
            if 0 <= a < b <= len(source_text):
                scored.append((a, b, LABELS[label], confidence))
    return scored


def predict_documents(model, parallel, document_ids, document_texts,
                      existing=None, on_document=None):
    was_training = model.training
    model.eval()
    predictions = dict(existing or {})
    try:
        with torch.inference_mode():
            for doc_id in sorted(document_ids):
                if doc_id in predictions:
                    continue
                text = document_texts[doc_id]
                ids, offsets = encode_doc(doc_id, text)
                if not ids:
                    predictions[doc_id] = []
                    if on_document is not None:
                        on_document(predictions)
                    continue
                bounds = window_bounds(len(ids), CONFIG['window_tokens'], CONFIG['window_stride'])
                rows = [{'ids': ids[lo:hi], 'offsets': offsets[lo:hi]}
                        for lo, hi in bounds]
                found = {}
                batch_size = CONFIG['inference_window_batch_size']
                cursor = 0
                while cursor < len(rows):
                    group = rows[cursor:cursor + batch_size]
                    input_ids, masks = inference_batch(group)
                    try:
                        # DataParallel cannot scatter a one-row batch across two GPUs.
                        outputs = (parallel(input_ids, masks, return_hidden=True)
                                   if len(group) > 1 else
                                   model(input_ids, masks, return_hidden=True))
                    except RuntimeError as exc:
                        if 'out of memory' not in str(exc).lower() or batch_size == 1:
                            raise
                        del input_ids, masks
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        continue
                    for row_number, row in enumerate(group):
                        candidates = score_window_candidates(
                            model, outputs['hidden'][row_number],
                            outputs['start'][row_number].detach().cpu(),
                            outputs['end'][row_number].detach().cpu(), row, text)
                        for a, b, name, confidence in candidates:
                            key = (a, b, name)
                            found[key] = max(found.get(key, 0.0), confidence)
                    cursor += len(group)
                predictions[doc_id] = [[a, b, name, float(score)]
                                       for (a, b, name), score in sorted(
                                           found.items(), key=lambda item: -item[1])]
                if on_document is not None:
                    on_document(predictions)
    finally:
        if was_training:
            model.train()
    return predictions


def decode(candidate_map, thresholds=None):
    thresholds = thresholds or {}
    results = {}
    for doc_id, candidates in candidate_map.items():
        accepted, starts = [], []
        for a, b, name, score in candidates:
            if float(score) < float(thresholds.get(name, CONFIG['default_threshold'])):
                continue
            index = bisect.bisect_left(starts, a)
            if ((index and accepted[index - 1][1] > a)
                    or (index < len(accepted) and accepted[index][0] < b)):
                continue
            starts.insert(index, int(a))
            accepted.insert(index, (int(a), int(b), name))
        results[doc_id] = [list(item) for item in accepted]
    return results


def strict_macro(candidate_map, reference_ids, thresholds=None):
    if set(candidate_map) != set(reference_ids):
        raise RuntimeError('Candidate IDs differ from evaluation document IDs')
    decoded = decode(candidate_map, thresholds)
    per_label = {}
    for name in LABELS:
        predicted = {(doc_id, a, b) for doc_id, rows in decoded.items()
                     for a, b, label in rows if label == name}
        expected = {(doc_id, a, b) for doc_id in reference_ids
                    for a, b, label in gold[doc_id] if label == name}
        raw = {(doc_id, int(a), int(b)) for doc_id, rows in candidate_map.items()
               for a, b, label, score in rows if label == name}
        tp = len(predicted & expected)
        p = tp / len(predicted) if predicted else 0.0
        r = tp / len(expected) if expected else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        hits = len(raw & expected)
        per_label[name] = {'gold': len(expected), 'predicted': len(predicted),
                           'tp': tp, 'precision': p, 'recall': r, 'f1': f1,
                           'exact_candidate': hits,
                           'exact_candidate_recall': hits / len(expected) if expected else 0.0}
    return {'macro_f1': sum(row['f1'] for row in per_label.values()) / len(LABELS),
            'per_label': per_label}

# %% 6 - one fold at a time, selected adapters and resumable optimizer
def verify_best(folder, fold_index):
    manifest_path = folder / 'best.json'
    if not manifest_path.is_file():
        return None
    best = read_json(manifest_path)
    if best.get('config_hash') != CONFIG_HASH or best.get('fold') != fold_index:
        raise RuntimeError(f'Fold {fold_index}: selected checkpoint provenance changed')
    adapter_path = folder / best['adapter_file']
    candidates_path = folder / best['candidates_file']
    if (not adapter_path.is_file() or not candidates_path.is_file()
            or file_sha(adapter_path) != best['adapter_sha256']
            or file_sha(candidates_path) != best['candidates_sha256']):
        raise RuntimeError(f'Fold {fold_index}: selected adapter/candidates failed SHA256 checks')
    payload = read_json(candidates_path)
    if payload.get('config_hash') != CONFIG_HASH or set(payload['candidates']) != set(FOLDS[fold_index]):
        raise RuntimeError(f'Fold {fold_index}: selected candidate provenance failed')
    measured = strict_macro(payload['candidates'], FOLDS[fold_index])
    if abs(measured['macro_f1'] - best['default_macro_f1']) > 1e-8:
        raise RuntimeError(f'Fold {fold_index}: selected score does not reproduce')
    return best


def fold_recovery_zip(fold_index, summary):
    folder = ROOT / f'fold{fold_index}'
    best = verify_best(folder, fold_index)
    recovery = WORKING / f'CS781_Phase2_Span_Fold{fold_index}_Recovery.zip'
    members = [(folder / best['adapter_file'], 'selected_adapter.pt'),
               (folder / 'best.json', 'best.json'),
               (folder / 'summary.json', 'summary.json'),
               (ROOT / 'experiment_config.json', 'experiment_config.json'),
               (MODEL_SOURCE, 'model/prefix_hybrid_lm.py'),
               (BPE_SOURCE, 'model/course_bpe.py'),
               (BASE_DIR / 'tokenizer' / 'tokenizer.json', 'tokenizer/tokenizer.json'),
               (BASE_DIR / 'tokenizer' / 'tokenizer_config.json', 'tokenizer/tokenizer_config.json'),
               (BASE_DIR / 'tokenizer' / 'tokenizer_metadata.json', 'tokenizer/tokenizer_metadata.json')]
    if not recovery.is_file():
        temporary = recovery.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path, name in members:
                archive.write(path, name)
        os.replace(temporary, recovery)
    with zipfile.ZipFile(recovery) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {name for _, name in members}:
            raise RuntimeError(f'Fold {fold_index}: recovery ZIP CRC/member audit failed')
        if hashlib.sha256(archive.read('selected_adapter.pt')).hexdigest() != best['adapter_sha256']:
            raise RuntimeError(f'Fold {fold_index}: recovery ZIP has the wrong selected adapter')
    return recovery


def train_fold(fold_index):
    folder = ROOT / f'fold{fold_index}'
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / 'summary.json'
    latest_path = folder / 'checkpoint_latest.pt'
    if summary_path.is_file():
        saved = read_json(summary_path)
        if saved.get('config_hash') != CONFIG_HASH:
            raise RuntimeError(f'Fold {fold_index}: summary configuration changed')
        if saved.get('status') == 'complete':
            verify_best(folder, fold_index)
            print(f'Fold {fold_index}: verified complete, reusing selected adapter')
            fold_recovery_zip(fold_index, saved)
            return saved
        if saved.get('status') != 'paused' or not latest_path.is_file():
            raise RuntimeError(f'Fold {fold_index}: interrupted work lacks a resume checkpoint')
    if free_gib() < MIN_FREE_GIB:
        raise RuntimeError(f'Fold {fold_index}: only {free_gib():.2f} GiB free')
    heldout = FOLDS[fold_index]
    train_ids = set(texts) - set(heldout)
    rows = make_windows(train_ids, training=True)
    print(f'Fold {fold_index}: {len(train_ids)} training documents, '
          f'{len(heldout)} held-out documents, {len(rows)} training windows')
    model = fresh_model().to('cuda:0')
    parallel = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    lora_parameters = [p for name, p in model.named_parameters()
                       if name.startswith('base.') and p.requires_grad]
    head_parameters = [p for name, p in model.named_parameters()
                       if not name.startswith('base.') and p.requires_grad]
    if not lora_parameters or not head_parameters:
        raise RuntimeError('Trainable LoRA or span-head parameters are missing')
    optimizer = torch.optim.AdamW([
        {'params': lora_parameters, 'lr': CONFIG['lora_lr'], 'initial_lr': CONFIG['lora_lr']},
        {'params': head_parameters, 'lr': CONFIG['head_lr'], 'initial_lr': CONFIG['head_lr']},
    ], weight_decay=CONFIG['weight_decay'], foreach=False)
    steps_per_epoch = math.ceil(len(rows) / CONFIG['batch_size'])
    total_steps = MAX_EPOCHS * steps_per_epoch
    warmup = max(1, int(total_steps * CONFIG['warmup_fraction']))
    def learning_rate_factor(step):
        if step < warmup:
            return max(0.001, step / warmup)
        return max(0.05, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    weights = label_weights(train_ids).to('cuda:0')
    epoch_start, stale, history = 0, 0, []
    best = verify_best(folder, fold_index)
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location='cpu', weights_only=False)
        if saved.get('config_hash') != CONFIG_HASH or saved.get('fold') != fold_index or saved.get('window_count') != len(rows):
            raise RuntimeError(f'Fold {fold_index}: optimizer checkpoint provenance failed')
        load_trainable_state(model, saved['trainable_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        random.setstate(saved['python_rng_state'])
        epoch_start, stale, history = saved['epoch'], saved['stale'], saved['history']
        print(f'Fold {fold_index}: resuming after epoch {epoch_start}')
    paused = False
    try:
        for epoch in range(epoch_start + 1, MAX_EPOCHS + 1):
            if time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
                paused = True
                print('Session time guard reached before the next epoch. Run All again to resume.')
                break
            order = list(range(len(rows)))
            random.Random(SEED + fold_index * 1000 + epoch).shuffle(order)
            model.train()
            losses = []
            for begin in range(0, len(order), CONFIG['batch_size']):
                group = [rows[i] for i in order[begin:begin + CONFIG['batch_size']]]
                batch = {name: value.to('cuda:0', non_blocking=True)
                         for name, value in collate_windows(group, epoch).items()}
                optimizer.zero_grad(set_to_none=True)
                output = (parallel(batch['input_ids'], batch['attention_mask'], batch['pair_index'])
                          if len(group) > 1 else
                          model(batch['input_ids'], batch['attention_mask'], batch['pair_index']))
                loss = task_loss(output, batch, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Fold {fold_index} epoch {epoch}: non-finite FP32 loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip_norm'])
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            entry = {'epoch': epoch, 'train_loss_mean': sum(losses) / len(losses),
                     'elapsed_seconds': round(time.time() - RUN_STARTED, 1)}
            if epoch % VALIDATE_EVERY == 0 or epoch == MAX_EPOCHS:
                candidates = predict_documents(model, parallel, heldout, texts)
                measured = strict_macro(candidates, heldout)
                score = measured['macro_f1']
                entry['heldout_default_macro_f1'] = score
                entry['raw_exact_candidate_recall_macro'] = sum(
                    row['exact_candidate_recall'] for row in measured['per_label'].values()) / len(LABELS)
                if best is None or score > best['default_macro_f1'] + 1e-8:
                    adapter_name = f'selected_epoch{epoch:02d}.pt'
                    candidates_name = f'candidates_epoch{epoch:02d}.json'
                    atomic_torch(folder / adapter_name,
                                 {'trainable_state': trainable_state(model),
                                  'config_hash': CONFIG_HASH, 'fold': fold_index,
                                  'epoch': epoch, 'labels': LABELS})
                    atomic_json(folder / candidates_name,
                                {'config_hash': CONFIG_HASH, 'fold': fold_index,
                                 'epoch': epoch, 'candidates': candidates})
                    proposed = {'config_hash': CONFIG_HASH, 'fold': fold_index,
                                'epoch': epoch, 'default_macro_f1': score,
                                'adapter_file': adapter_name,
                                'adapter_sha256': file_sha(folder / adapter_name),
                                'candidates_file': candidates_name,
                                'candidates_sha256': file_sha(folder / candidates_name)}
                    previous = best
                    atomic_json(folder / 'best.json', proposed)
                    best = verify_best(folder, fold_index)
                    if previous is not None:
                        (folder / previous['adapter_file']).unlink(missing_ok=True)
                        (folder / previous['candidates_file']).unlink(missing_ok=True)
                    stale = 0
                else:
                    stale += 1
                print(f'Fold {fold_index} epoch {epoch}: loss {entry["train_loss_mean"]:.4f}, '
                      f'held-out F1 {score:.4f}, raw recall {entry["raw_exact_candidate_recall_macro"]:.4f}, '
                      f'best {best["default_macro_f1"]:.4f} @ {best["epoch"]}, '
                      f'disk {free_gib():.2f} GiB free', flush=True)
            else:
                print(f'Fold {fold_index} epoch {epoch}: loss {entry["train_loss_mean"]:.4f}', flush=True)
            history.append(entry)
            atomic_json(folder / 'history.json', history)
            if free_gib() < MIN_FREE_GIB:
                raise RuntimeError('Disk free fell below the checkpoint reserve')
            atomic_torch(latest_path, {
                'config_hash': CONFIG_HASH, 'fold': fold_index, 'window_count': len(rows),
                'epoch': epoch, 'stale': stale, 'history': history,
                'trainable_state': trainable_state(model),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'python_rng_state': random.getstate()})
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(f'Fold {fold_index}: stopping after {PATIENCE} non-improving validations')
                break
        if best is None:
            raise RuntimeError(f'Fold {fold_index}: no validation checkpoint was selected')
        best = verify_best(folder, fold_index)
        selected_candidates = read_json(folder / best['candidates_file'])['candidates']
        selected_score = strict_macro(selected_candidates, heldout)
        result = {'status': 'paused' if paused else 'complete', 'fold': fold_index,
                  'config_hash': CONFIG_HASH, 'training_documents': len(train_ids),
                  'heldout_documents': len(heldout), 'training_windows': len(rows),
                  'best_epoch': best['epoch'], 'default_macro_f1': best['default_macro_f1'],
                  'per_label': selected_score['per_label'], 'history': history,
                  'selected_adapter_sha256': best['adapter_sha256'],
                  'selected_candidates_sha256': best['candidates_sha256']}
        atomic_json(summary_path, result)
        if not paused:
            fold_recovery_zip(fold_index, result)
            latest_path.unlink(missing_ok=True)
        return result
    finally:
        del model, parallel, optimizer, scheduler
        torch.cuda.empty_cache()

# %% 7 - complete out-of-fold evaluation and threshold selection
fold_results = {}
for fold_index in (0, 1, 2):
    if time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
        print('Session time guard reached before another fold. Rerun Run All to resume.')
        break
    fold_results[fold_index] = train_fold(fold_index)
    if fold_results[fold_index]['status'] != 'complete':
        break

oof_candidates = {}
for fold_index in (0, 1, 2):
    folder = ROOT / f'fold{fold_index}'
    if not (folder / 'summary.json').is_file():
        continue
    summary = read_json(folder / 'summary.json')
    if summary.get('status') != 'complete' or summary.get('config_hash') != CONFIG_HASH:
        continue
    best = verify_best(folder, fold_index)
    group = read_json(folder / best['candidates_file'])['candidates']
    if set(group) != set(FOLDS[fold_index]) or set(group) & set(oof_candidates):
        raise RuntimeError('OOF candidate document IDs are incomplete or overlapping')
    oof_candidates.update(group)


def fast_macro(candidate_map, thresholds):
    predictions = decode(candidate_map, thresholds)
    values = []
    for name in LABELS:
        predicted = {(doc_id, a, b) for doc_id, rows in predictions.items()
                     for a, b, label in rows if label == name}
        expected = {(doc_id, a, b) for doc_id in predictions
                    for a, b, label in gold[doc_id] if label == name}
        tp = len(predicted & expected)
        p = tp / len(predicted) if predicted else 0.0
        r = tp / len(expected) if expected else 0.0
        values.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(values) / len(LABELS)


oof_report = None
threshold_report = None
all_cv_complete = set(oof_candidates) == set(texts)
if all_cv_complete:
    default = strict_macro(oof_candidates, texts)
    oracle = sum(2 * row['exact_candidate_recall'] / (1 + row['exact_candidate_recall'])
                 for row in default['per_label'].values()) / len(LABELS)
    thresholds = {name: CONFIG['default_threshold'] for name in LABELS}
    threshold_grid = (0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80)
    current = default['macro_f1']
    for name in LABELS:
        chosen, chosen_score = thresholds[name], current
        for trial in threshold_grid:
            proposed = dict(thresholds, **{name: trial})
            measured = fast_macro(oof_candidates, proposed)
            if measured > chosen_score + 1e-8:
                chosen, chosen_score = trial, measured
        thresholds[name], current = chosen, chosen_score
    fold_comparison = []
    for fold_index in (0, 1, 2):
        subset = {doc_id: oof_candidates[doc_id] for doc_id in FOLDS[fold_index]}
        fold_comparison.append({'fold': fold_index,
                                'default': fast_macro(subset, {}),
                                'calibrated': fast_macro(subset, thresholds)})
    hurts = sum(row['calibrated'] + 1e-8 < row['default'] for row in fold_comparison)
    if current <= default['macro_f1'] + 1e-8 or hurts >= 2:
        thresholds = {name: CONFIG['default_threshold'] for name in LABELS}
        current = default['macro_f1']
    tuned = strict_macro(oof_candidates, texts, thresholds)
    if abs(tuned['macro_f1'] - current) > 1e-8:
        raise RuntimeError('Calibrated OOF macro F1 is inconsistent')
    oof_report = {'documents': len(texts),
                  'default_macro_f1': default['macro_f1'],
                  'calibrated_macro_f1': tuned['macro_f1'],
                  'exact_candidate_oracle_macro_f1': oracle,
                  'default_per_label': default['per_label'],
                  'calibrated_per_label': tuned['per_label'],
                  'fold_comparison': fold_comparison,
                  'old_slot03_default_oof': OLD_OOF,
                  'old_slot03_calibrated_oof': old_status['completed']['3']['cv']['calibrated_oof_macro_f1'],
                  'selection_caveat': 'Best epoch and thresholds were selected on the same public folds; this OOF metric is selection-biased.'}
    threshold_report = {'thresholds': thresholds, 'default_macro_f1': default['macro_f1'],
                        'calibrated_macro_f1': tuned['macro_f1'],
                        'fold_comparison': fold_comparison}
    atomic_json(ROOT / 'oof_report.json', oof_report)
    atomic_json(ROOT / 'thresholds.json', threshold_report)
    print('Three-fold default OOF F1:', f"{default['macro_f1']:.6f}")
    print('Three-fold calibrated OOF F1:', f"{tuned['macro_f1']:.6f}")
    print('Raw exact candidate oracle ceiling:', f'{oracle:.6f}')
    print('Old slot-03 default OOF F1:', f'{OLD_OOF:.6f}')
else:
    print('OOF incomplete: completed', len(oof_candidates), 'of', len(texts),
          'documents. Rerun Run All in this Kaggle session to continue.')

# %% 8 - full-data fit, private-test inference, exact output verification
def final_recovery_zip(summary):
    final_dir = ROOT / 'final'
    adapter_path = final_dir / 'selected_adapter.pt'
    recovery = WORKING / 'CS781_Phase2_Span_Final_Recovery.zip'
    members = [(adapter_path, 'selected_adapter.pt'),
               (final_dir / 'summary.json', 'summary.json'),
               (ROOT / 'experiment_config.json', 'experiment_config.json'),
               (MODEL_SOURCE, 'model/prefix_hybrid_lm.py'),
               (BPE_SOURCE, 'model/course_bpe.py'),
               (BASE_DIR / 'tokenizer' / 'tokenizer.json', 'tokenizer/tokenizer.json'),
               (BASE_DIR / 'tokenizer' / 'tokenizer_config.json', 'tokenizer/tokenizer_config.json'),
               (BASE_DIR / 'tokenizer' / 'tokenizer_metadata.json', 'tokenizer/tokenizer_metadata.json')]
    if not recovery.is_file():
        temporary = recovery.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path, name in members:
                archive.write(path, name)
        os.replace(temporary, recovery)
    with zipfile.ZipFile(recovery) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {name for _, name in members}:
            raise RuntimeError('Final recovery ZIP CRC/member audit failed')
        if hashlib.sha256(archive.read('selected_adapter.pt')).hexdigest() != summary['adapter_sha256']:
            raise RuntimeError('Final recovery ZIP contains the wrong adapter')
    return recovery


def train_final_model():
    final_dir = ROOT / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    summary_path = final_dir / 'summary.json'
    adapter_path = final_dir / 'selected_adapter.pt'
    latest_path = final_dir / 'checkpoint_latest.pt'
    epochs = max(2, min(MAX_EPOCHS,
                 sorted(read_json(ROOT / f'fold{i}' / 'summary.json')['best_epoch']
                        for i in (0, 1, 2))[1]))
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get('config_hash') != CONFIG_HASH or summary.get('epochs') != epochs:
            raise RuntimeError('Existing final-fit provenance changed')
        if summary.get('status') == 'complete':
            if not adapter_path.is_file() or file_sha(adapter_path) != summary['adapter_sha256']:
                raise RuntimeError('Final selected adapter SHA256 mismatch')
            print('Final full-data fit verified and reused')
            final_recovery_zip(summary)
            return summary
        if summary.get('status') != 'paused' or not latest_path.is_file():
            raise RuntimeError('Final fit cannot resume from its saved state')
    rows = make_windows(texts, training=True)
    model = fresh_model().to('cuda:0')
    parallel = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    lora_params = [p for name, p in model.named_parameters()
                   if name.startswith('base.') and p.requires_grad]
    head_params = [p for name, p in model.named_parameters()
                   if not name.startswith('base.') and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': CONFIG['lora_lr'], 'initial_lr': CONFIG['lora_lr']},
        {'params': head_params, 'lr': CONFIG['head_lr'], 'initial_lr': CONFIG['head_lr']},
    ], weight_decay=CONFIG['weight_decay'], foreach=False)
    total_steps = epochs * math.ceil(len(rows) / CONFIG['batch_size'])
    warmup = max(1, int(total_steps * CONFIG['warmup_fraction']))
    def factor(step):
        if step < warmup:
            return max(0.001, step / warmup)
        return max(0.05, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    weights = label_weights(texts).to('cuda:0')
    start_epoch, history = 0, []
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location='cpu', weights_only=False)
        if saved.get('config_hash') != CONFIG_HASH or saved.get('epochs') != epochs or saved.get('window_count') != len(rows):
            raise RuntimeError('Final optimizer checkpoint provenance changed')
        load_trainable_state(model, saved['trainable_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        random.setstate(saved['python_rng_state'])
        start_epoch, history = saved['epoch'], saved['history']
        print(f'Final fit: resuming after epoch {start_epoch}')
    print(f'Final fit: {len(rows)} windows from all {len(texts)} public labeled documents; '
          f'{epochs} CV-selected epochs')
    paused = False
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            if time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
                paused = True
                print('Session time guard reached before next final-fit epoch')
                break
            order = list(range(len(rows)))
            random.Random(SEED + 10_000 + epoch).shuffle(order)
            model.train()
            losses = []
            for begin in range(0, len(order), CONFIG['batch_size']):
                group = [rows[i] for i in order[begin:begin + CONFIG['batch_size']]]
                batch = {name: value.to('cuda:0', non_blocking=True)
                         for name, value in collate_windows(group, epoch).items()}
                optimizer.zero_grad(set_to_none=True)
                output = (parallel(batch['input_ids'], batch['attention_mask'], batch['pair_index'])
                          if len(group) > 1 else
                          model(batch['input_ids'], batch['attention_mask'], batch['pair_index']))
                loss = task_loss(output, batch, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Final fit epoch {epoch}: non-finite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip_norm'])
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            history.append({'epoch': epoch, 'train_loss_mean': sum(losses) / len(losses),
                            'elapsed_seconds': round(time.time() - RUN_STARTED, 1)})
            atomic_json(final_dir / 'history.json', history)
            if free_gib() < MIN_FREE_GIB:
                raise RuntimeError('Disk free fell below the checkpoint reserve')
            atomic_torch(latest_path, {
                'config_hash': CONFIG_HASH, 'epochs': epochs, 'window_count': len(rows),
                'epoch': epoch, 'history': history,
                'trainable_state': trainable_state(model),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'python_rng_state': random.getstate()})
            print(f'Final fit epoch {epoch}/{epochs}: loss {history[-1]["train_loss_mean"]:.4f}, '
                  f'disk {free_gib():.2f} GiB free', flush=True)
        if not paused:
            atomic_torch(adapter_path, {'config_hash': CONFIG_HASH, 'epochs': epochs,
                                        'trainable_state': trainable_state(model),
                                        'labels': LABELS})
        result = {'status': 'paused' if paused else 'complete',
                  'config_hash': CONFIG_HASH, 'epochs': epochs,
                  'training_documents': len(texts), 'training_windows': len(rows),
                  'history': history,
                  'adapter_sha256': file_sha(adapter_path) if not paused else None,
                  'selection_note': 'Epoch count is the median selected across the three public CV folds; no test labels were used.'}
        atomic_json(summary_path, result)
        if not paused:
            final_recovery_zip(result)
            latest_path.unlink(missing_ok=True)
        return result
    finally:
        del model, parallel, optimizer, scheduler
        torch.cuda.empty_cache()


def make_submission(final_summary):
    final_dir = ROOT / 'final'
    adapter_path = final_dir / 'selected_adapter.pt'
    if file_sha(adapter_path) != final_summary['adapter_sha256']:
        raise RuntimeError('Final adapter changed before inference')
    meta_path = final_dir / 'test_candidates_provenance.json'
    partial_path = final_dir / 'test_candidates.partial.json'
    test_source_hash = file_sha(SOURCE_DIR / 'fold3-text.json')
    thresholds = read_json(ROOT / 'thresholds.json')['thresholds']
    provenance = {'config_hash': CONFIG_HASH,
                  'adapter_sha256': final_summary['adapter_sha256'],
                  'test_file_sha256': test_source_hash,
                  'thresholds': thresholds}
    if meta_path.is_file() and read_json(meta_path) != provenance:
        raise RuntimeError('Stale private-test inference cache provenance')
    atomic_json(meta_path, provenance)
    cached = read_json(partial_path) if partial_path.is_file() else {}
    if not set(cached) <= set(test_texts):
        raise RuntimeError('Partial inference contains unexpected test IDs')
    model = fresh_model().to('cuda:0')
    selected = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if selected.get('config_hash') != CONFIG_HASH:
        raise RuntimeError('Final adapter configuration mismatch')
    load_trainable_state(model, selected['trainable_state'])
    parallel = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    try:
        def progress(predictions):
            if len(predictions) % 5 == 0:
                atomic_json(partial_path, predictions)
                print(f'Test inference: {len(predictions)}/{len(test_texts)} documents', flush=True)
        raw = predict_documents(model, parallel, test_texts, test_texts,
                                existing=cached, on_document=progress)
    finally:
        del model, parallel
        torch.cuda.empty_cache()
    if set(raw) != set(test_texts):
        raise RuntimeError('Test inference is incomplete')
    atomic_json(partial_path, raw)
    for doc_id, rows in raw.items():
        for a, b, name, confidence in rows:
            if (name not in LABEL_ID or not 0 <= int(a) < int(b) <= len(test_texts[doc_id])
                    or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1):
                raise RuntimeError(f'Invalid candidate in test document {doc_id}')
    predictions = decode(raw, thresholds)
    if set(predictions) != set(test_texts):
        raise RuntimeError('Submission document IDs do not match official test IDs')
    for doc_id, rows in predictions.items():
        if any(not 0 <= a < b <= len(test_texts[doc_id]) or label not in LABEL_ID
               for a, b, label in rows):
            raise RuntimeError(f'Invalid final offset/label in {doc_id}')
        if any(rows[i][1] > rows[i+1][0] for i in range(len(rows)-1)):
            raise RuntimeError(f'Overlapping predictions in {doc_id}')
    pred_path = final_dir / 'prediction.json'
    atomic_json(pred_path, predictions)
    zip_path = WORKING / 'CS781_Phase2_Span_Extractor_Submission.zip'
    temporary = zip_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(pred_path, arcname='prediction.json')
    os.replace(temporary, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None or archive.namelist() != ['prediction.json']:
            raise RuntimeError('Final submission ZIP format/CRC verification failed')
        if json.loads(archive.read('prediction.json')) != predictions:
            raise RuntimeError('Final submission ZIP contents differ from the verified predictions')
    package = {'status': 'READY_FOR_MANUAL_CODABENCH_UPLOAD',
               'submission_zip': str(zip_path), 'zip_sha256': file_sha(zip_path),
               'test_documents': len(test_texts),
               'predicted_spans': sum(map(len, predictions.values())),
               'selected_adapter_sha256': final_summary['adapter_sha256'],
               'thresholds': thresholds}
    atomic_json(final_dir / 'package.json', package)
    print('Verified submission:', zip_path)
    return package


final_result = None
package_result = None
if all_cv_complete and oof_report['default_macro_f1'] >= MIN_FINAL_CV_F1:
    if time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
        print('Time guard reached before full-data training. Rerun Run All to resume.')
    else:
        final_result = train_final_model()
        if final_result['status'] == 'complete':
            package_result = make_submission(final_result)
else:
    print('Final fit skipped: finish all CV folds and exceed the saved local baseline first.')

# %% 9 - compact diagnostic report and download locations
saved_folds = {}
for fold_index in (0, 1, 2):
    path = ROOT / f'fold{fold_index}' / 'summary.json'
    if path.is_file():
        saved = read_json(path)
        if saved.get('config_hash') != CONFIG_HASH:
            raise RuntimeError(f'Fold {fold_index}: report config hash mismatch')
        saved_folds[str(fold_index)] = saved

if package_result is None:
    existing_package = ROOT / 'final' / 'package.json'
    if existing_package.is_file():
        proposed = read_json(existing_package)
        candidate_zip = Path(proposed['submission_zip'])
        if candidate_zip.is_file() and file_sha(candidate_zip) == proposed['zip_sha256']:
            package_result = proposed

if package_result:
    run_status = 'READY_FOR_MANUAL_CODABENCH_UPLOAD'
elif all_cv_complete and oof_report['default_macro_f1'] < MIN_FINAL_CV_F1:
    run_status = 'CV_BELOW_FINAL_FIT_GATE'
elif final_result is not None and final_result['status'] == 'paused':
    run_status = 'FINAL_FIT_PAUSED_RESUME_RUN_ALL'
else:
    run_status = 'CV_PAUSED_RESUME_RUN_ALL'

report = {
    'version': VERSION, 'status': run_status, 'config_hash': CONFIG_HASH,
    'source_dir': str(PREVIOUS), 'output_dir': str(ROOT),
    'fold_summaries': saved_folds, 'oof': oof_report,
    'threshold_selection': threshold_report,
    'final_fit': read_json(ROOT / 'final' / 'summary.json') if (ROOT / 'final' / 'summary.json').is_file() else None,
    'package': package_result,
    'gold_alignment': {'total': alignment_counts['total'],
                       'whitespace_trim_exact': alignment_counts['trim_exact'],
                       'whitespace_trim_exact_rate': alignment_counts['trim_exact'] / alignment_counts['total']},
    'elapsed_seconds': round(time.time() - RUN_STARTED, 1),
    'disk_free_gib': round(free_gib(), 2),
    'caveats': [
        'Local strict macro-F1 approximates the official private evaluator.',
        'Best epoch and calibrated thresholds were selected on public validation folds; those F1 estimates are optimistic.',
        'A candidate oracle is a mathematical upper bound, not a likely score.',
        'No private test annotations or external pretrained weights were used.',
        'Submission ZIP, if produced, must be uploaded manually; no private score is known here.',
    ],
}
atomic_json(ROOT / 'report.json', report)
if oof_report:
    rows = []
    for name in LABELS:
        default = oof_report['default_per_label'][name]
        calibrated = oof_report['calibrated_per_label'][name]
        rows.append({'label': name, 'gold': default['gold'],
                     'default_f1': default['f1'], 'calibrated_f1': calibrated['f1'],
                     'precision': default['precision'], 'recall': default['recall'],
                     'exact_candidate_recall': default['exact_candidate_recall'],
                     'true_positive': default['tp'], 'predicted': default['predicted']})
    with (ROOT / 'per_label_oof.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

readme = '''# CS781 Phase-2 span-extractor report

Run All again in the same Kaggle working session if `status` says PAUSED.

`report.json` summarizes all completed validation folds, OOF macro-F1,
per-label candidate recall, threshold selection, and final packaging status.
`per_label_oof.csv` is available once all three folds finish.
Each `foldN/best.json`, `foldN/summary.json`, and `foldN/candidates.json`
records the selected public held-out predictions and their provenance.

The submission ZIP is a separate download named
`CS781_Phase2_Span_Extractor_Submission.zip`. Upload it to Codabench only
after checking the report. Fold recovery ZIPs contain trainable LoRA and span
head weights, model source, and tokenizer, but require the already saved
Phase-1 base-03 checkpoint. The notebook never uses private test labels.
'''
(ROOT / 'README.md').write_text(readme, encoding='utf-8')

report_files = [ROOT / 'report.json', ROOT / 'experiment_config.json', ROOT / 'README.md']
for optional in ('oof_report.json', 'thresholds.json', 'per_label_oof.csv'):
    path = ROOT / optional
    if path.is_file():
        report_files.append(path)
for fold_index in (0, 1, 2):
    folder = ROOT / f'fold{fold_index}'
    for filename in ('summary.json', 'best.json', 'history.json'):
        path = folder / filename
        if path.is_file():
            report_files.append(path)
    if (folder / 'best.json').is_file():
        best = verify_best(folder, fold_index)
        report_files.append(folder / best['candidates_file'])
for filename in ('summary.json', 'history.json', 'package.json'):
    path = ROOT / 'final' / filename
    if path.is_file():
        report_files.append(path)

temporary_report = REPORT_ZIP.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary_report, 'w', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as archive:
    for path in report_files:
        archive.write(path, arcname=path.relative_to(ROOT).as_posix())
os.replace(temporary_report, REPORT_ZIP)
with zipfile.ZipFile(REPORT_ZIP) as archive:
    if archive.testzip() is not None or set(archive.namelist()) != {
            path.relative_to(ROOT).as_posix() for path in report_files}:
        raise RuntimeError('Diagnostic report ZIP failed verification')

print('Run status:', run_status)
print('Download report:', REPORT_ZIP,
      f'({REPORT_ZIP.stat().st_size / 2**20:.2f} MiB)')
if package_result:
    print('Download and manually submit:', package_result['submission_zip'])
print('Disk free:', f'{free_gib():.2f} GiB')
