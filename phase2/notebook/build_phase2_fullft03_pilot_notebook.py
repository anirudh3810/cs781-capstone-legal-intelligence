"""Build the independent Kaggle base-03 full-fine-tune pilot notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "CS781_Phase2_Base03_FullFT_Pilot_and_CV.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


cells = [
    markdown("""# CS781 Phase 2 — base 03 full fine-tune pilot

Run **All** in the **same Kaggle session** as `capstone-project-phase2-notebook (1).ipynb` and `CS781_Phase2_Diagnostics_Steps_1_2.ipynb`. Enable **two T4 GPUs**. This notebook only needs the files already under `/kaggle/working/CS781_Capstone/phase2_improved`; it does not use the previous Python kernel state, download models, or copy the large artifact ZIPs.

The saved diagnostic run showed that slot 03 often failed to generate exact APP, RESP and WIT spans at any threshold. The next controlled experiment is therefore a **full fine-tune of the existing Phase-1 base 03**, retaining the same 384-token training windows, tagger, validation folds, candidate generator and 0.45 decoder as the improved LoRA slot 03. A lower backbone learning rate is used because all base weights now move. The pilot compares held-out fold 0 against the saved LoRA candidates. If it clears the prespecified gate, it runs the other two folds sequentially and reports out-of-fold (OOF) metrics. The best epoch is selected on each validation fold, so these metrics are optimistic. It **does not** make a Codabench submission or claim that 0.6 F1 is guaranteed.

Checkpoints, candidate files and reports are written to `/kaggle/working/CS781_Capstone/phase2_fullft03_pilot/`. A small diagnostic ZIP and a separate best-model recovery ZIP are produced. Only one fold and one optimizer checkpoint are live at a time. A rerun resumes at the last complete epoch. The previous `phase2_improved` directory and its submissions are read only.

**Run All once, then download** `/kaggle/working/CS781_Phase2_FullFT03_Report.zip` and share it. If interrupted, Run All again in the same session. The notebook stops before the Kaggle session time limit. If fold 0 is weak, the remaining folds are skipped so we can change the model/head based on measured errors.

This code is AI assisted; follow your course's disclosure rules.
"""),
    code(r'''from __future__ import annotations

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
from torch.utils.checkpoint import checkpoint as activation_checkpoint

WORKING = Path('/kaggle/working')
PREVIOUS = WORKING / 'CS781_Capstone' / 'phase2_improved'
ROOT = WORKING / 'CS781_Capstone' / 'phase2_fullft03_pilot'
REPORT_ZIP = WORKING / 'CS781_Phase2_FullFT03_Report.zip'
BASE_DIR = PREVIOUS / 'stages' / 'final_slot03' / 'phase1_bases' / '03'
COMMON_PATH = PREVIOUS / 'method_source' / 'phase2_method_common.py'
SEED = 781
VERSION = 'base03-fullft-pilot-v1'
MAX_EPOCHS = 12
MIN_EPOCHS = 5
PATIENCE = 3
BATCH_SIZE = 16  # nn.DataParallel splits this over the two T4 GPUs
MAX_RUNTIME_SECONDS = 31_200  # leave about 80 minutes in a 10-hour Kaggle session
MIN_DISK_FREE_GIB = 2.0
RUN_EXTRA_FOLDS_IF_GATE_PASSES = True
START_TIME = time.time()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

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
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def disk_free_gib():
    return shutil.disk_usage(WORKING).free / 2**30

def import_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

required = [
    BASE_DIR / 'model' / 'prefix_hybrid_lm.py',
    BASE_DIR / 'model' / 'course_bpe.py',
    BASE_DIR / 'checkpoints' / 'selected_model.pt',
    BASE_DIR / 'tokenizer' / 'tokenizer.json',
    BASE_DIR / 'tokenizer' / 'tokenizer_config.json',
    BASE_DIR / 'tokenizer' / 'tokenizer_metadata.json',
    COMMON_PATH,
    PREVIOUS / 'cv_folds.json',
    PREVIOUS / 'improvement_status.json',
    PREVIOUS / 'cv_candidates' / 'cv0_slot03.json',
    PREVIOUS / 'official_phase2' / 'labels.txt',
]
for fold in (1, 2):
    required.extend([PREVIOUS / 'official_phase2' / f'fold{fold}-text.json',
                     PREVIOUS / 'official_phase2' / f'fold{fold}-gold.json'])
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError('Same-session Phase-2 files missing: ' + '\n'.join(missing))
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError('Choose the Kaggle two-T4 GPU accelerator. This experiment requires both GPUs.')
if disk_free_gib() < MIN_DISK_FREE_GIB:
    raise RuntimeError(f'Only {disk_free_gib():.2f} GiB free; keep at least {MIN_DISK_FREE_GIB} GiB for safe checkpoint saves.')
ROOT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
print('Previous artifacts:', PREVIOUS)
print('New output:', ROOT)
print('Disk free:', f'{disk_free_gib():.2f} GiB')
for device_id in (0, 1):
    free, total = torch.cuda.mem_get_info(device_id)
    print(f'cuda:{device_id}: {torch.cuda.get_device_name(device_id)} | {free/2**30:.2f}/{total/2**30:.2f} GiB free')
print('Precision: FP32, to avoid the FP16 overflow seen in the earlier T4 run.')
'''),
    code(r'''def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

manifest = read_json(PREVIOUS / 'cv_folds.json')
status = read_json(PREVIOUS / 'improvement_status.json')
labels = tuple(line.strip() for line in (PREVIOUS / 'official_phase2' / 'labels.txt').read_text(encoding='utf-8').splitlines() if line.strip())
if len(labels) != 12 or len(set(labels)) != 12:
    raise RuntimeError('Expected the exact 12 official entity labels')
texts, gold = {}, {}
source_groups = []
for number in (1, 2):
    group_text = read_json(PREVIOUS / 'official_phase2' / f'fold{number}-text.json')
    group_gold = read_json(PREVIOUS / 'official_phase2' / f'fold{number}-gold.json')
    if set(group_text) != set(group_gold) or set(texts) & set(group_text):
        raise RuntimeError('Official train ID mismatch or overlap')
    source_groups.append(set(group_text))
    texts.update(group_text)
    gold.update(group_gold)
folds = [tuple(str(identifier) for identifier in group) for group in manifest['folds']]
if len(folds) != 3 or len(set(x for group in folds for x in group)) != len(texts) or set(x for group in folds for x in group) != set(texts):
    raise RuntimeError('Saved CV folds do not partition all official training documents')
if any(not isinstance(text, str) for text in texts.values()):
    raise RuntimeError('Invalid official text schema')
for doc_id, rows in gold.items():
    for row in rows:
        a, b, name = int(row[0]), int(row[1]), row[2]
        if name not in labels or not 0 <= a < b <= len(texts[doc_id]):
            raise RuntimeError(f'Invalid gold annotation in {doc_id}: {row}')
gold = {doc_id: [(int(a), int(b), name) for a, b, name, *rest in rows] for doc_id, rows in gold.items()}
old_pilot_payload = read_json(PREVIOUS / 'cv_candidates' / 'cv0_slot03.json')
old_pilot = old_pilot_payload['candidates']
if set(old_pilot) != set(folds[0]):
    raise RuntimeError('Saved LoRA fold-0 candidate IDs differ from the CV manifest')
baseline_record = status['completed']['3']['cv']
old_fold0_default_f1 = float(baseline_record['fold_scores'][0]['baseline'])
old_fold0_tuned_f1 = float(baseline_record['fold_scores'][0]['tuned'])
data_hash = hashlib.sha256(json.dumps({
    'folds': manifest, 'labels': labels,
    'sources': {path.name: sha256_file(path) for path in required if path.parent.name == 'official_phase2'}
}, sort_keys=True).encode()).hexdigest()
print('Official train documents:', len(texts), '| folds:', [len(x) for x in folds])
print('Old LoRA fold-0 F1 at 0.45:', round(old_fold0_default_f1, 6))
print('Old LoRA fold-0 F1 with OOF-tuned thresholds:', round(old_fold0_tuned_f1, 6))
print('Data/fold fingerprint:', data_hash)
'''),
    code(r'''MODEL_SOURCE = BASE_DIR / 'model' / 'prefix_hybrid_lm.py'
BPE_SOURCE = BASE_DIR / 'model' / 'course_bpe.py'
BASE_CHECKPOINT = BASE_DIR / 'checkpoints' / 'selected_model.pt'
source_sha = sha256_file(MODEL_SOURCE)
bpe_sha = sha256_file(BPE_SOURCE)
common_sha = sha256_file(COMMON_PATH)
base_sha = sha256_file(BASE_CHECKPOINT)
model_module = import_source(MODEL_SOURCE, 'cs781_pilot_base03_' + source_sha[:12])
bpe_module = import_source(BPE_SOURCE, 'cs781_pilot_bpe03_' + bpe_sha[:12])
common = import_source(COMMON_PATH, 'cs781_pilot_common_' + common_sha[:12])
tokenizer = bpe_module.CourseBPETokenizer.from_pretrained(BASE_DIR / 'tokenizer')
tokenizer_meta = read_json(BASE_DIR / 'tokenizer' / 'tokenizer_metadata.json')
base_record = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
if base_record.get('model_source_sha256') != source_sha:
    raise RuntimeError('Phase-1 checkpoint/model source binding failed')
if base_record.get('tokenizer_fingerprint') != tokenizer_meta.get('fingerprint'):
    raise RuntimeError('Phase-1 checkpoint/tokenizer binding failed')
if int(base_record['model_config']['vocab_size']) != len(tokenizer):
    raise RuntimeError('Phase-1 tokenizer vocabulary mismatch')
model_config = base_record['model_config']
del base_record

class CheckpointedFullFTTagger(common.EntityTagger):
    """Same tagger and base-03 features; activation recomputation saves T4 memory."""
    def features(self, input_ids, attention_mask):
        base = self.base
        length = input_ids.shape[1]
        visibility = base._prefix_hybrid_visibility(attention_mask, attention_mask.sum(1))
        hidden = base.embedding_dropout(base.token_embedding(input_ids) * math.sqrt(base.hidden_size))
        cosine, sine = base.rotary(length, hidden.dtype)
        for block in base.blocks:
            if self.training and torch.is_grad_enabled():
                hidden = activation_checkpoint(block, hidden, visibility, cosine, sine, use_reentrant=False)
            else:
                hidden = block(hidden, visibility, cosine, sine)
        return base.final_norm(hidden)

def fresh_model(config):
    checkpoint = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
    if checkpoint.get('model_source_sha256') != source_sha or checkpoint.get('tokenizer_fingerprint') != tokenizer_meta['fingerprint']:
        raise RuntimeError('Phase-1 base changed during notebook execution')
    base = model_module.build_model(checkpoint['model_config'])
    base.load_state_dict(checkpoint['model_state'], strict=True)
    del checkpoint
    tagger = CheckpointedFullFTTagger(base, '03', labels, 'full_finetune', config)
    trained = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
    total = sum(p.numel() for p in tagger.parameters())
    if trained != total:
        raise RuntimeError(f'Full fine-tune expected all parameters trainable: {trained}/{total}')
    return tagger

CONFIG = {
    'version': VERSION,
    'base_id': '03', 'method': 'full_finetune',
    'max_tokens': 384, 'window_stride': 288,
    'batch_size': BATCH_SIZE, 'max_epochs': MAX_EPOCHS,
    'min_epochs': MIN_EPOCHS, 'patience': PATIENCE,
    'backbone_lr': 2e-5, 'head_lr': 2e-4,
    'weight_decay': 0.01, 'warmup_fraction': 0.05,
    'gradient_clip_norm': 1.0,
    'head_dropout': 0.1, 'outside_class_weight': 0.6,
    'max_char_shift': 16, 'boundary_loss_weight': 0.2,
    'inference_window_batch_size': 8, 'confidence_threshold': 0.45,
    'seed': SEED, 'precision': 'float32',
    'activation_checkpointing': True,
    'data_hash': data_hash, 'phase1_checkpoint_sha256': base_sha,
    'phase1_model_source_sha256': source_sha,
    'phase1_bpe_source_sha256': bpe_sha,
    'phase2_common_source_sha256': common_sha,
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
config_path = ROOT / 'experiment_config.json'
if config_path.exists() and read_json(config_path) != CONFIG:
    raise RuntimeError('Existing pilot configuration differs; use its original notebook/settings to resume')
atomic_json(config_path, CONFIG)
print('Full FT configuration SHA256:', CONFIG_HASH)
print('Both T4s used for training through nn.DataParallel; activation checkpointing enabled.')
'''),
    code(r'''def window_bounds(length, size, stride):
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(start, min(start + size, length)) for start in starts]

def make_training_windows(train_ids, fold_index):
    # This reproduces the improved slot-03 exhaustive 384/288 window protocol.
    train_text, train_gold, window_ids, rare_ids, uncovered = {}, {}, [], [], []
    rare = {'APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT'}
    core_size = CONFIG['max_tokens'] - 8
    for doc_id in sorted(train_ids):
        text = texts[doc_id]
        offsets = [(int(a), int(b)) for a, b in tokenizer.backend.encode(text, add_special_tokens=False).offsets]
        if not offsets:
            continue
        spans = gold[doc_id]
        covered = [False] * len(spans)
        for window_index, (lo, hi) in enumerate(window_bounds(len(offsets), core_size, min(CONFIG['window_stride'], core_size))):
            char_lo, char_hi = offsets[lo][0], offsets[hi - 1][1]
            if char_hi <= char_lo:
                continue
            contained = []
            for gold_index, (a, b, name) in enumerate(spans):
                if char_lo <= a and b <= char_hi:
                    contained.append((a - char_lo, b - char_lo, name))
                    covered[gold_index] = True
            window_id = f'{doc_id}__{window_index:04d}'
            train_text[window_id] = text[char_lo:char_hi]
            train_gold[window_id] = contained
            window_ids.append(window_id)
            if any(name in rare for _, _, name in contained):
                rare_ids.append(window_id)
        for gold_index, okay in enumerate(covered):
            if okay:
                continue
            a, b, name = spans[gold_index]
            touching = [i for i, (x, y) in enumerate(offsets) if y > a and x < b]
            if not touching or touching[-1] - touching[0] + 1 > core_size:
                uncovered.append((doc_id, spans[gold_index]))
                continue
            lo = max(0, min(touching[0], len(offsets) - core_size))
            hi = min(len(offsets), lo + core_size)
            char_lo, char_hi = offsets[lo][0], offsets[hi - 1][1]
            if not char_lo <= a or not b <= char_hi:
                uncovered.append((doc_id, spans[gold_index]))
                continue
            window_id = f'{doc_id}__entity{gold_index:04d}'
            train_text[window_id] = text[char_lo:char_hi]
            train_gold[window_id] = [(x - char_lo, y - char_lo, label)
                                     for x, y, label in spans if char_lo <= x and y <= char_hi]
            window_ids.append(window_id)
            if name in rare:
                rare_ids.append(window_id)
    extra_count = min(len(rare_ids), max(1, len(window_ids) // 6))
    for index, source_id in enumerate(random.Random(SEED + len(window_ids)).sample(rare_ids, extra_count) if extra_count else []):
        duplicate = f'{source_id}__rare{index:05d}'
        train_text[duplicate], train_gold[duplicate] = train_text[source_id], train_gold[source_id]
        window_ids.append(duplicate)
    if not window_ids:
        raise RuntimeError('No legal NER training windows were built')
    if uncovered:
        atomic_json(ROOT / f'fold{fold_index}_uncovered_spans.json', uncovered)
    return train_text, train_gold, window_ids, uncovered

def decode(candidate_map, threshold=0.45):
    predictions = {}
    for doc_id, rows in candidate_map.items():
        accepted = []
        for a, b, name, score in sorted(rows, key=lambda row: -float(row[3])):
            a, b = int(a), int(b)
            if float(score) < threshold:
                continue
            if any(a < other_b and other_a < b for other_a, other_b, _ in accepted):
                continue
            accepted.append((a, b, name))
        predictions[doc_id] = sorted(accepted)
    return predictions

def strict_macro(candidate_map, fold_ids):
    if set(candidate_map) != set(fold_ids):
        raise RuntimeError('Candidate IDs differ from held-out fold IDs')
    predicted = decode(candidate_map)
    per_label = {}
    for name in labels:
        p = {(doc_id, a, b) for doc_id, rows in predicted.items() for a, b, label in rows if label == name}
        g = {(doc_id, a, b) for doc_id in fold_ids for a, b, label in gold[doc_id] if label == name}
        raw = {(doc_id, int(a), int(b)) for doc_id, rows in candidate_map.items()
               for a, b, label, score in rows if label == name}
        tp = len(p & g)
        precision = tp / len(p) if p else 0.0
        recall = tp / len(g) if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[name] = {'gold': len(g), 'predicted': len(p), 'tp': tp,
                           'precision': precision, 'recall': recall, 'f1': f1,
                           'exact_candidate': len(raw & g),
                           'exact_candidate_recall': len(raw & g) / len(g) if g else 0.0}
    return {'macro_f1': sum(v['f1'] for v in per_label.values()) / len(labels),
            'per_label': per_label}

old_check = strict_macro(old_pilot, folds[0])
if abs(old_check['macro_f1'] - old_fold0_default_f1) > 1e-8:
    raise RuntimeError('Could not reproduce the original LoRA fold-0 local score')
print('Reproduced saved LoRA fold-0 F1:', f'{old_check["macro_f1"]:.6f}')
print('Old weak-label exact candidate recall:',
      {name: round(old_check['per_label'][name]['exact_candidate_recall'], 3) for name in ('APP', 'RESP', 'WIT')})
'''),
    code(r'''def train_fold(fold_index):
    folder = ROOT / f'fold{fold_index}'
    folder.mkdir(parents=True, exist_ok=True)
    selected_path = folder / 'selected_model.pt'
    latest_path = folder / 'checkpoint_latest.pt'
    candidates_path = folder / 'candidates.json'
    summary_path = folder / 'summary.json'
    if summary_path.is_file():
        saved = read_json(summary_path)
        if (saved.get('status') == 'complete' and saved.get('config_hash') == CONFIG_HASH
                and selected_path.is_file() and candidates_path.is_file()
                and sha256_file(selected_path) == saved.get('selected_sha256')
                and sha256_file(candidates_path) == saved.get('candidates_sha256')):
            print(f'Fold {fold_index}: verified complete, reusing saved checkpoint and candidates')
            return saved
        if not (saved.get('status') == 'paused' and saved.get('config_hash') == CONFIG_HASH
                and latest_path.is_file() and selected_path.is_file() and candidates_path.is_file()
                and sha256_file(selected_path) == saved.get('selected_sha256')
                and sha256_file(candidates_path) == saved.get('candidates_sha256')):
            raise RuntimeError(f'Fold {fold_index}: saved summary/artifact integrity failed')
        print(f'Fold {fold_index}: verified paused report; resuming its optimizer checkpoint')
    if disk_free_gib() < MIN_DISK_FREE_GIB:
        raise RuntimeError(f'Fold {fold_index}: only {disk_free_gib():.2f} GiB free')
    val_ids = folds[fold_index]
    train_ids = set(texts) - set(val_ids)
    train_text, train_gold, window_ids, uncovered = make_training_windows(train_ids, fold_index)
    steps_per_epoch = math.ceil(len(window_ids) / BATCH_SIZE)
    total_steps = MAX_EPOCHS * steps_per_epoch
    print(f'Fold {fold_index}: {len(train_ids)} training docs, {len(val_ids)} held-out docs, '
          f'{len(window_ids)} windows, {steps_per_epoch} optimizer steps/epoch, {len(uncovered)} uncovered spans')
    model = fresh_model(CONFIG).to('cuda:0')
    parallel = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.base.parameters() if p.requires_grad], 'lr': CONFIG['backbone_lr'],
         'initial_lr': CONFIG['backbone_lr']},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('base.') and p.requires_grad],
         'lr': CONFIG['head_lr'], 'initial_lr': CONFIG['head_lr']},
    ], weight_decay=CONFIG['weight_decay'], foreach=False)
    warmup_steps = max(1, int(total_steps * CONFIG['warmup_fraction']))
    def lr_factor(step):
        if step < warmup_steps:
            return max(1e-3, step / warmup_steps)
        return max(0.05, (total_steps - step) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch, best_f1, best_epoch, stale, history = 0, -1.0, 0, 0, []
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location='cpu', weights_only=False)
        if latest.get('config_hash') != CONFIG_HASH or latest.get('fold') != fold_index or latest.get('windows') != len(window_ids):
            raise RuntimeError(f'Fold {fold_index}: unsafe resume checkpoint provenance')
        model.load_state_dict(latest['model_state'], strict=True)
        optimizer.load_state_dict(latest['optimizer_state'])
        scheduler.load_state_dict(latest['scheduler_state'])
        torch.set_rng_state(latest['torch_rng_state'])
        torch.cuda.set_rng_state_all(latest['cuda_rng_states'])
        random.setstate(latest['python_rng_state'])
        start_epoch, best_f1, best_epoch = latest['epoch'], latest['best_f1'], latest['best_epoch']
        stale, history = latest['stale'], latest['history']
        if best_epoch and not selected_path.is_file():
            raise RuntimeError('Resume checkpoint references missing selected model')
        print(f'Fold {fold_index}: resuming after epoch {start_epoch}; best {best_f1:.6f} at epoch {best_epoch}')

    def validation_candidates():
        model.eval()
        return {doc_id: common.predict(model, texts[doc_id], tokenizer, labels,
                                       dict(CONFIG, return_candidate_scores=True))
                for doc_id in val_ids}

    paused = False
    try:
        for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
            if time.time() - START_TIME > MAX_RUNTIME_SECONDS:
                paused = True
                print('Time guard reached before next epoch; rerun Run All to resume.')
                break
            order = list(window_ids)
            random.Random(SEED + fold_index * 1000 + epoch).shuffle(order)
            model.train()
            losses = []
            for begin in range(0, len(order), BATCH_SIZE):
                batch_ids = order[begin:begin + BATCH_SIZE]
                rows = [{'text': train_text[window_id], 'spans': train_gold[window_id]}
                        for window_id in batch_ids]
                batch = {key: value.to('cuda:0', non_blocking=True)
                         for key, value in common.collate(rows, tokenizer, labels, CONFIG).items()}
                optimizer.zero_grad(set_to_none=True)
                output = parallel(**batch)
                loss = output['loss'].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite FP32 loss in fold {fold_index}, epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip_norm'])
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            candidate_map = validation_candidates()
            measured = strict_macro(candidate_map, val_ids)
            current_f1 = measured['macro_f1']
            improved = current_f1 > best_f1 + 1e-8
            if improved:
                best_f1, best_epoch, stale = current_f1, epoch, 0
                atomic_torch(selected_path, {'model_state': model.state_dict(), 'model_config': model_config,
                                             'training_config': CONFIG, 'config_hash': CONFIG_HASH,
                                             'fold': fold_index, 'epoch': epoch, 'validation_macro_f1': current_f1,
                                             'labels': labels, 'tokenizer_fingerprint': tokenizer_meta['fingerprint']})
                atomic_json(candidates_path, {'fold': fold_index, 'epoch': epoch,
                                              'selected_sha256': sha256_file(selected_path),
                                              'config_hash': CONFIG_HASH, 'candidates': candidate_map})
            else:
                stale += 1
            item = {'epoch': epoch, 'train_loss_mean': sum(losses) / len(losses),
                    'val_macro_f1': current_f1, 'best_epoch': best_epoch, 'best_f1': best_f1,
                    'elapsed_seconds': round(time.time() - START_TIME, 1)}
            history.append(item)
            atomic_json(folder / 'history.json', history)
            print(f'Fold {fold_index} epoch {epoch}: train loss {item["train_loss_mean"]:.4f}, '
                  f'held-out F1 {current_f1:.4f}, best {best_f1:.4f} @ {best_epoch}, '
                  f'disk {disk_free_gib():.2f} GiB free', flush=True)
            if disk_free_gib() < MIN_DISK_FREE_GIB:
                raise RuntimeError('Disk free fell below checkpoint safety reserve')
            atomic_torch(latest_path, {
                'config_hash': CONFIG_HASH, 'fold': fold_index, 'windows': len(window_ids),
                'epoch': epoch, 'best_f1': best_f1, 'best_epoch': best_epoch, 'stale': stale,
                'history': history, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'python_rng_state': random.getstate()})
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(f'Fold {fold_index}: stopped after {PATIENCE} epochs without held-out improvement')
                break
        if not selected_path.is_file() or not candidates_path.is_file():
            raise RuntimeError(f'Fold {fold_index}: no selected checkpoint/candidates')
        candidate_record = read_json(candidates_path)
        if candidate_record['selected_sha256'] != sha256_file(selected_path):
            raise RuntimeError('Best checkpoint/candidate pairing failed integrity check')
        selected_candidates = candidate_record['candidates']
        score = strict_macro(selected_candidates, val_ids)
        if abs(score['macro_f1'] - best_f1) > 1e-8:
            raise RuntimeError('Best checkpoint candidate score differs from recorded F1')
        result = {'status': 'paused' if paused else 'complete', 'fold': fold_index,
                  'config_hash': CONFIG_HASH, 'validation_documents': len(val_ids),
                  'training_windows': len(window_ids), 'uncovered_gold_spans': len(uncovered),
                  'best_epoch': best_epoch, 'best_macro_f1': best_f1,
                  'per_label': score['per_label'], 'selected_sha256': sha256_file(selected_path),
                  'candidates_sha256': sha256_file(candidates_path),
                  'history': history}
        atomic_json(summary_path, result)
        if not paused:
            # Package selected weights before starting another fold, so the
            # useful model is downloadable even if a later fold is interrupted.
            export_recovery(fold_index, result)
            if summary_path.is_file() and selected_path.is_file() and candidates_path.is_file():
                latest_path.unlink(missing_ok=True)
        return result
    finally:
        del parallel, model, optimizer, scheduler
        torch.cuda.empty_cache()

def export_recovery(fold_index, summary):
    fold_dir = ROOT / f'fold{fold_index}'
    recovery = WORKING / f'CS781_Phase2_FullFT03_Fold{fold_index}_Recovery.zip'
    members = [(fold_dir / 'selected_model.pt', 'selected_model.pt'),
               (fold_dir / 'summary.json', 'summary.json'),
               (ROOT / 'experiment_config.json', 'experiment_config.json'),
               (MODEL_SOURCE, 'model/prefix_hybrid_lm.py'),
               (BPE_SOURCE, 'model/course_bpe.py'),
               (COMMON_PATH, 'model/phase2_method_common.py'),
               (BASE_DIR / 'tokenizer' / 'tokenizer.json', 'tokenizer/tokenizer.json'),
               (BASE_DIR / 'tokenizer' / 'tokenizer_config.json', 'tokenizer/tokenizer_config.json'),
               (BASE_DIR / 'tokenizer' / 'tokenizer_metadata.json', 'tokenizer/tokenizer_metadata.json')]
    if not recovery.is_file():
        with zipfile.ZipFile(recovery, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for source, member in members:
                archive.write(source, member)
    with zipfile.ZipFile(recovery) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {name for _, name in members}:
            raise RuntimeError(f'Fold {fold_index} recovery ZIP failed verification')
        if hashlib.sha256(archive.read('selected_model.pt')).hexdigest() != summary['selected_sha256']:
            raise RuntimeError(f'Fold {fold_index} recovery ZIP contains the wrong selected model')
    return recovery
'''),
    code(r'''def pilot_gate(pilot_summary):
    new_f1 = pilot_summary['best_macro_f1']
    new_weak = {name: pilot_summary['per_label'][name]['exact_candidate_recall']
                for name in ('APP', 'RESP', 'WIT')}
    old_weak = {name: old_check['per_label'][name]['exact_candidate_recall']
                for name in ('APP', 'RESP', 'WIT')}
    weak_gain = sum(new_weak[x] - old_weak[x] for x in new_weak) / len(new_weak)
    # Additional full folds are costly. Proceed for a clear F1 gain over the
    # old fold-0 tuned score, or a substantial raw-candidate recall gain without
    # sacrificing the untuned F1. Thresholds stay fixed for this comparison.
    passed = (new_f1 >= old_fold0_tuned_f1 + 0.03 or
              (weak_gain >= 0.10 and new_f1 >= old_fold0_default_f1 - 0.01))
    return {'passed': bool(passed), 'new_f1': new_f1,
            'old_default_f1': old_fold0_default_f1, 'old_tuned_f1': old_fold0_tuned_f1,
            'weak_label_candidate_recall_old': old_weak,
            'weak_label_candidate_recall_new': new_weak,
            'weak_label_recall_mean_gain': weak_gain,
            'rule': 'F1 >= old tuned + 0.03, OR weak candidate recall gain >= 0.10 with no material default-F1 loss'}

results = {}
results[0] = train_fold(0)
gate = pilot_gate(results[0])
atomic_json(ROOT / 'pilot_gate.json', gate)
print('Pilot gate:', json.dumps(gate, indent=2))
if results[0]['status'] == 'complete' and gate['passed'] and RUN_EXTRA_FOLDS_IF_GATE_PASSES:
    for fold_index in (1, 2):
        if time.time() - START_TIME > MAX_RUNTIME_SECONDS:
            print('Time guard: rerun Run All for remaining CV fold(s).')
            break
        results[fold_index] = train_fold(fold_index)
        if results[fold_index]['status'] != 'complete':
            break
else:
    print('Additional folds skipped; use the pilot report to choose the next model change.')
'''),
    code(r'''# The small report can be downloaded even if a time guard stopped after a fold.
completed = {}
all_candidates = {}
for fold_index in (0, 1, 2):
    path = ROOT / f'fold{fold_index}' / 'summary.json'
    candidate_path = ROOT / f'fold{fold_index}' / 'candidates.json'
    if not path.is_file() or not candidate_path.is_file():
        continue
    row = read_json(path)
    if row.get('config_hash') != CONFIG_HASH or sha256_file(candidate_path) != row.get('candidates_sha256'):
        raise RuntimeError(f'Fold {fold_index} report integrity failed')
    completed[fold_index] = row
    if row['status'] == 'complete':
        candidate_map = read_json(candidate_path)['candidates']
        if set(candidate_map) != set(folds[fold_index]):
            raise RuntimeError(f'Fold {fold_index} candidate IDs do not match held-out documents')
        all_candidates.update(candidate_map)

oof = strict_macro(all_candidates, tuple(all_candidates)) if len(completed) == 3 and all(
    completed[i]['status'] == 'complete' for i in (0, 1, 2)) and set(all_candidates) == set(texts) else None
report = {'version': VERSION, 'status': 'three_fold_complete' if oof else
          ('pilot_complete' if completed.get(0, {}).get('status') == 'complete' else 'paused'),
          'base_checkpoint_sha256': base_sha, 'config_hash': CONFIG_HASH,
          'pilot_gate': gate, 'fold_summaries': completed,
          'three_fold_default_threshold_oof': oof,
          'old_slot03_default_threshold_oof': baseline_record['baseline_oof_macro_f1'],
          'old_slot03_calibrated_oof': baseline_record['calibrated_oof_macro_f1'],
          'note': 'Scores are local strict-span estimates on public labeled folds. Best-epoch selection uses each held-out fold and is optimistic; a new untouched holdout is needed before claiming generalization.'}
atomic_json(ROOT / 'report.json', report)
rows = []
for name in labels:
    old = old_check['per_label'][name]
    new = completed[0]['per_label'][name]
    rows.append({'label': name, 'gold': new['gold'],
                 'old_lora_f1': old['f1'], 'new_fullft_f1': new['f1'],
                 'old_exact_candidate_recall': old['exact_candidate_recall'],
                 'new_exact_candidate_recall': new['exact_candidate_recall'],
                 'new_tp': new['tp'], 'new_predicted': new['predicted']})
with (ROOT / 'pilot_per_label_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

report_files = [ROOT / 'report.json', ROOT / 'pilot_gate.json', ROOT / 'experiment_config.json',
                ROOT / 'pilot_per_label_comparison.csv']
for fold_index in completed:
    report_files.extend([ROOT / f'fold{fold_index}' / 'summary.json',
                         ROOT / f'fold{fold_index}' / 'history.json',
                         ROOT / f'fold{fold_index}' / 'candidates.json'])
with zipfile.ZipFile(REPORT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in report_files:
        archive.write(path, arcname=path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(REPORT_ZIP) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('Report ZIP CRC verification failed')

# Recovery ZIPs are created as each fold completes; verify them again here.
# Full fine-tuned weights include the Phase-1 base; no second base checkpoint is copied.
recovery_paths = []
for fold_index, summary in completed.items():
    if summary['status'] != 'complete':
        continue
    recovery_paths.append(str(export_recovery(fold_index, summary)))
print('Report:', REPORT_ZIP, f'({REPORT_ZIP.stat().st_size / 2**20:.2f} MiB)')
print('Recovery ZIPs:', recovery_paths)
print('Status:', report['status'])
print('Disk free:', f'{disk_free_gib():.2f} GiB')
if oof:
    print('Three-fold default-threshold local OOF F1:', f'{oof["macro_f1"]:.6f}')
else:
    print('Pilot held-out F1:', f'{completed[0]["best_macro_f1"]:.6f}')
'''),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"},
                 "cs781": {"ai_assisted": True, "experiment": "base03-full-finetune-pilot"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
