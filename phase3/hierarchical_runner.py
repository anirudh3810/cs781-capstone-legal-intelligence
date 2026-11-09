"""CS781 Phase 3: document-held-out, target-aware hierarchical run.

Embedded verbatim in the Run All notebook. Uses only the user's Phase 1 model
03, official Phase 3 data, and the already trained Phase 3 slot 06 fallback.
"""
from __future__ import annotations

import argparse
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
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
PRIOR = WORK / 'CS781_Capstone/phase3'
ROOT = WORK / 'CS781_Capstone/phase3_hier'
ARTIFACT = WORK / 'CS781_Phase3_Hierarchical_All_Artifacts.zip'
SUBMISSION = WORK / 'CS781_Phase3_Hierarchical_Submission.zip'
OLD_ARCHIVES = ('CS781_Phase3_All_Artifacts.zip', 'CS781_Phase3_One_All_Artifacts.zip')
FOLDS = 5
SEED = 78123
MAX_TOKENS = 192
BATCH = 32
TOKEN_BUDGET = 6144
MAX_BATCH = 96
ACCUM = 2
SENT_EPOCHS = 10
HEAD_EPOCHS = 20
WINDOW = 128
STRIDE = 96


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf8', delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def atomic_torch(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(obj, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_name(name):
    p = Path(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name


def locate(name):
    direct = WORK / name
    if direct.is_file():
        return direct
    hits = sorted(p for p in INPUT.rglob(name) if p.is_file())
    if len(hits) > 1:
        raise RuntimeError(f'Multiple {name} inputs: {hits}')
    return hits[0] if hits else None


def required_prior(name):
    return (name == 'runner.py' or name.startswith('official/') or
            name.startswith('bases/03/') or name == 'slot_06/best.pt')


def extract_manifest_zip(source, prefix, destination, selector):
    """Selectively extract a manifest archive; verify each selected member."""
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or 'integrity.json' not in names:
            raise RuntimeError(f'Bad recovery member list: {source}')
        manifest = json.loads(z.read('integrity.json'))['files']
        if set(manifest) != set(names) - {'integrity.json'}:
            raise RuntimeError(f'Bad recovery manifest: {source}')
        selected = 0
        for name in names:
            if name == 'integrity.json':
                continue
            if not safe_name(name) or z.getinfo(name).is_dir():
                raise RuntimeError(f'Unsafe recovery path: {name}')
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            if not selector(relative):
                continue
            dest = destination / relative
            info = manifest[name]
            if dest.is_file() and dest.stat().st_size == info['bytes'] and digest(dest) == info['sha256']:
                selected += 1; continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
                temp = Path(out.name); h = hashlib.sha256()
                for block in iter(lambda: src.read(4 << 20), b''):
                    h.update(block); out.write(block)
            if temp.stat().st_size != info['bytes'] or h.hexdigest() != info['sha256']:
                temp.unlink(missing_ok=True)
                raise RuntimeError(f'Corrupt recovery member {name}')
            os.replace(temp, dest); selected += 1
    return selected


def prior_ready():
    needs = ('runner.py', 'official/CL_train.json', 'official/IT_test.json',
             'official/label_vocab.json', 'bases/03/model/prefix_hybrid_lm.py',
             'bases/03/model/course_bpe.py', 'bases/03/checkpoints/selected_model.pt',
             'bases/03/tokenizer/tokenizer.json', 'slot_06/best.pt')
    return all((PRIOR / x).is_file() for x in needs)


def restore():
    own = locate(ARTIFACT.name)
    if own is not None and not (ROOT / 'config.json').is_file():
        extract_manifest_zip(own, 'prior/', PRIOR, required_prior)
        extract_manifest_zip(own, 'hier/', ROOT, lambda _: True)
    if not prior_ready():
        for name in OLD_ARCHIVES:
            source = locate(name)
            if source is None:
                continue
            prefix = '' if name == OLD_ARCHIVES[0] else 'prior/'
            count = extract_manifest_zip(source, prefix, PRIOR, required_prior)
            print('Restored', count, 'prior files from', source, flush=True)
            if prior_ready():
                break
    if not prior_ready():
        raise FileNotFoundError('Need the live six-run Kaggle session or attach CS781_Phase3_All_Artifacts.zip')


def snapshot():
    """Portable archive with model states, optimizer checkpoints, data and code."""
    if not prior_ready():
        raise RuntimeError('Prior run is not staged')
    sources = [(p, 'prior/' + p.relative_to(PRIOR).as_posix())
               for p in PRIOR.rglob('*') if p.is_file() and required_prior(p.relative_to(PRIOR).as_posix())
               and '__pycache__' not in p.parts]
    sources += [(p, 'hier/' + p.relative_to(ROOT).as_posix())
                for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.tmp')]
    sources.sort(key=lambda item: item[1])
    manifest = {'version': 2, 'files': {}}
    temp = ARTIFACT.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', allowZip64=True) as z:
        for path, name in sources:
            manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
            z.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix in {'.pt', '.zip'}
                    else zipfile.ZIP_DEFLATED)
        z.writestr('integrity.json', json.dumps(manifest, sort_keys=True))
    os.replace(temp, ARTIFACT)
    print('Recovery archive:', ARTIFACT, round(ARTIFACT.stat().st_size / 2**30, 2), 'GiB', flush=True)


def legacy():
    spec = importlib.util.spec_from_file_location('cs781_phase3_legacy', PRIOR / 'runner.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sets():
    old = legacy()
    train, dev, test = old.all_data()
    vocab = old.labels()
    if len(train) != 80 or len(dev) != 10 or len(test) != 10:
        raise RuntimeError('Unexpected official Phase 3 document counts')
    return old, train, dev, test, vocab


def fold_map(train):
    rng = random.Random(SEED)
    mapping = {}
    for domain in ('CL', 'IT'):
        rows = [d for d in train if d['domain'] == domain]
        rng.shuffle(rows)
        # Greedy length balance after a seeded shuffle to avoid one giant fold.
        groups = [[] for _ in range(FOLDS)]
        totals = [0] * FOLDS
        for doc in sorted(rows, key=lambda d: len(d['text']), reverse=True):
            candidates = [i for i in range(FOLDS) if len(groups[i]) < 8]
            f = min(candidates, key=lambda i: (totals[i], len(groups[i]), i))
            groups[f].append(doc); totals[f] += len(doc['text'])
        for i, group in enumerate(groups):
            if len(group) != 8:
                raise RuntimeError('Fold must hold out eight documents per domain')
            for doc in group:
                mapping[doc['id']] = i
    return mapping


def target_ids(tokenizer, text):
    seq = tokenizer.backend.encode(text, add_special_tokens=False).ids
    if len(seq) > MAX_TOKENS:
        front = MAX_TOKENS * 3 // 4
        seq = seq[:front] + seq[-(MAX_TOKENS-front):]
    return seq or [int(tokenizer.pad_token_id)]


def prepare():
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Enable two Kaggle GPUs before Run All')
    restore()
    old, train, dev, test, vocab = sets()
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {'folds': FOLDS, 'seed': SEED, 'max_tokens': MAX_TOKENS,
              'batch': BATCH, 'token_budget': TOKEN_BUDGET, 'max_batch': MAX_BATCH,
              'accum': ACCUM, 'window': WINDOW, 'stride': STRIDE,
              'sent_epochs': SENT_EPOCHS, 'head_epochs': HEAD_EPOCHS,
              'prior_source_sha256': digest(PRIOR / 'runner.py'),
              'source_sha256': digest(__file__), 'vocab': vocab,
              'fold_map': fold_map(train), 'train_ids': [d['id'] for d in train],
              'dev_ids': [d['id'] for d in dev], 'test_ids': [d['id'] for d in test]}
    path = ROOT / 'config.json'
    if path.exists() and read_json(path) != config:
        raise RuntimeError('Existing recovery has different data or code; resume with its original notebook')
    atomic_json(path, config)
    shutil.copyfile(__file__, ROOT / 'hierarchical_runner.py')
    cache = ROOT / 'tokens.pt'
    if not cache.exists():
        _, tokenizer = old.load_base('03')
        data = {d['id']: [target_ids(tokenizer, s) for s in d['text']]
                for d in train + dev + test}
        atomic_torch(cache, data)
        del tokenizer
    data = load_tokens()
    lengths = sorted(len(seq) for rows in data.values() for seq in rows)
    atomic_json(ROOT / 'data_audit.json', {
        'documents': {'train': len(train), 'dev': len(dev), 'test': len(test)},
        'sentences': {'train': sum(len(d['text']) for d in train),
                      'dev': sum(len(d['text']) for d in dev),
                      'test': sum(len(d['text']) for d in test)},
        'train_label_counts': dict(Counter(y for d in train for y in d['labels'])),
        'dev_label_counts': dict(Counter(y for d in dev for y in d['labels'])),
        'token_length_p50': lengths[len(lengths)//2],
        'token_length_p99': lengths[int(.99*(len(lengths)-1))],
        'max_target_tokens': MAX_TOKENS,
        'truncated_target_count_at_limit': sum(x == MAX_TOKENS for x in lengths),
        'five_document_held_out_folds': True,
        'fold_training_documents': 60, 'fold_inner_validation_documents': 4,
        'fold_outer_holdout_documents': 16,
        'private_score_reference_slot06': .5058})
    print('Prepared', len(train), 'train documents;', len(dev), 'dev;', len(test), 'test.', flush=True)
    snapshot()


def load_tokens():
    return torch.load(ROOT / 'tokens.pt', map_location='cpu', weights_only=False)


def init_dist():
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    if world != 2 or torch.cuda.device_count() < 2:
        raise RuntimeError('Run distributed stages with torchrun and two GPUs')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    return rank, torch.device('cuda', int(os.environ['LOCAL_RANK']))


def is_final(fold):
    return fold == 'final'


def split_for(fold, train, dev, config):
    if is_final(fold):
        return train, dev
    number = int(fold)
    if not 0 <= number < FOLDS:
        raise ValueError(fold)
    mapping = config['fold_map']
    eligible = [d for d in train if mapping[d['id']] != number]
    inner_ids = set()
    for domain in ('CL', 'IT'):
        rows = [d for d in eligible if d['domain'] == domain]
        ordered = sorted(rows, key=lambda d: hashlib.sha256(
            f'{SEED}:{number}:{d["id"]}'.encode()).hexdigest())
        inner_ids.update(d['id'] for d in ordered[:2])
    fitting = [d for d in eligible if d['id'] not in inner_ids]
    checking = [d for d in eligible if d['id'] in inner_ids]
    if len(fitting) != 60 or len(checking) != 4:
        raise RuntimeError('Expected 60 fitting, four inner-validation documents')
    return fitting, checking


def outer_for(fold, train, config):
    return [d for d in train if config['fold_map'][d['id']] == int(fold)]


def folder(fold):
    return ROOT / ('final' if is_final(fold) else f'fold_{int(fold):02d}')


def collate(rows, tokens, pad, device):
    seqs = [tokens[did][i] for did, i, _, _ in rows]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for j, seq in enumerate(seqs):
        ids[j, :len(seq)] = torch.as_tensor(seq, dtype=torch.long)
        mask[j, :len(seq)] = True
    y = torch.tensor([r[2] for r in rows], dtype=torch.long)
    domain = torch.tensor([r[3] for r in rows], dtype=torch.long)
    if device.type == 'cuda':
        return (ids.pin_memory().to(device, non_blocking=True),
                mask.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
                domain.pin_memory().to(device, non_blocking=True))
    return ids.to(device), mask.to(device), y.to(device), domain.to(device)


def sentence_rows(docs, vocab):
    to_id = {v: i for i, v in enumerate(vocab)}
    return [(d['id'], i, to_id[y] if d['labels'] is not None else -1,
             0 if d['domain'] == 'CL' else 1)
            for d in docs for i, y in enumerate(d['labels'] or [None] * len(d['text']))]


def metrics(prob, docs, vocab):
    old = legacy()
    pred = {d['id']: [vocab[int(i)] for i in prob[d['id']].argmax(-1)] for d in docs}
    truth = {d['id']: d['labels'] for d in docs}
    present = [y for y in vocab if any(y in d['labels'] for d in docs)]
    all13 = old.prf(pred, truth, vocab)
    supported_f1 = sum(all13['per_label'][y]['f1'] for y in present) / len(present)
    domains = {}
    for domain in ('CL', 'IT'):
        subset = [d for d in docs if d['domain'] == domain]
        ps = {d['id']: pred[d['id']] for d in subset}
        ys = {d['id']: truth[d['id']] for d in subset}
        # Legacy PRF expects every predicted label in vocabulary.
        domains[domain] = old.prf(ps, ys, vocab)['macro_f1']
    return {'macro_all13': all13['macro_f1'], 'macro_supported': supported_f1,
            'domain_all13': domains, 'per_label': all13['per_label']}


class TargetEncoder(nn.Module):
    def __init__(self, base, labels_count):
        super().__init__()
        self.base = base
        width = base.hidden_size
        self.attention = nn.Linear(width, 1)
        self.pool = nn.Sequential(nn.Linear(width * 3, 512), nn.GELU(), nn.LayerNorm(512),
                                  nn.Dropout(.25))
        self.classifier = nn.Linear(512, labels_count)
        self.domain_bias = nn.Embedding(2, labels_count)
        nn.init.zeros_(self.domain_bias.weight)

    def forward(self, ids, mask, domain):
        base = self.base
        x = base.embedding_dropout(base.token_embedding(ids) * math.sqrt(base.hidden_size))
        visible = base._prefix_hybrid_visibility(mask, mask.sum(1))
        cosine, sine = base.rotary(ids.size(1), x.dtype)
        for block in base.blocks:
            x = checkpoint(block, x, visible, cosine, sine, use_reentrant=False) if self.training else block(x, visible, cosine, sine)
        x = base.final_norm(x)
        weights = self.attention(x).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        attention = torch.sum(x * weights.unsqueeze(-1), dim=1)
        mean = torch.sum(x * mask.unsqueeze(-1), dim=1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        emb = self.pool(torch.cat((attention, mean, maximum), dim=-1))
        logits = self.classifier(emb) + self.domain_bias(domain)
        return emb, logits


def new_encoder(old, device, vocab):
    base, tokenizer = old.load_base('03')
    base.output_bias.requires_grad_(False)  # LM output bias is unused for classification.
    return TargetEncoder(base, len(vocab)).to(device), tokenizer


def class_weights(docs, vocab, device):
    counts = Counter(y for d in docs for y in d['labels'])
    typical = sum(counts.values()) / len(vocab)
    values = [min(3., max(.6, math.sqrt(typical / max(1, counts[y])))) for y in vocab]
    scale = sum(values) / len(values)
    return torch.tensor([v / scale for v in values], device=device)


def gather_dict(local):
    received = [None, None]
    dist.all_gather_object(received, local)
    combined = {}
    for part in received:
        for did, value in part.items():
            if did in combined:
                raise RuntimeError('Duplicate distributed document: ' + did)
            combined[did] = value
    return combined


def autocast():
    return torch.autocast('cuda', dtype=torch.float16)


def finite_or_fail(loss):
    if not torch.isfinite(loss).all():
        raise RuntimeError('Non-finite loss; inspect checkpoint and batch')


@torch.no_grad()
def distributed_sentence_eval(model, docs, tokens, pad, vocab, rank, device, features=False):
    model.eval()
    own = {}
    for doc in docs[rank::2]:
        chunks = []
        logits = []
        for start in range(0, len(doc['text']), BATCH * 2):
            rows = [(doc['id'], i, -1, 0 if doc['domain'] == 'CL' else 1)
                    for i in range(start, min(start + BATCH * 2, len(doc['text'])))]
            ids, mask, _, domain = collate(rows, tokens, pad, device)
            with autocast():
                emb, score = model(ids, mask, domain)
            chunks.append(emb.float().cpu().half())
            logits.append(score.float().cpu())
        own[doc['id']] = {'features': torch.cat(chunks) if features else None,
                          'prob': torch.cat(logits).softmax(-1)}
    return own


def sentence_optimizer(model, steps):
    bottom, top, head = [], [], []
    half = len(model.base.blocks) // 2
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith('base.'):
            head.append(param)
        elif name.startswith('base.blocks.'):
            number = int(name.split('.')[2])
            (bottom if number < half else top).append(param)
        else:
            bottom.append(param)
    opt = torch.optim.AdamW([
        {'params': bottom, 'lr': 4e-6, 'initial_lr': 4e-6},
        {'params': top, 'lr': 1.2e-5, 'initial_lr': 1.2e-5},
        {'params': head, 'lr': 1.5e-4, 'initial_lr': 1.5e-4}], weight_decay=.04)
    warm = max(20, steps // 15)
    def schedule(step):
        if step < warm:
            return .1 + .9 * step / warm
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step-warm) / max(1, steps-warm))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def token_budget_batches(order, rows, tokens):
    batches = []
    current = []; widest = 0
    for index in order:
        row = rows[index]
        length = len(tokens[row[0]][row[1]])
        next_width = max(widest, length)
        if current and ((len(current)+1)*next_width > TOKEN_BUDGET or len(current) >= MAX_BATCH):
            batches.append(current); current = []; widest = 0
        current.append(index); widest = max(widest, length)
    if current:
        batches.append(current)
    return batches


def sentence_batches(rows, tokens, epoch, fold, rank):
    generator = torch.Generator().manual_seed(SEED + epoch * 101 +
                                              (0 if is_final(fold) else int(fold))*1009)
    counts = Counter((r[2], r[3]) for r in rows)
    sampling = torch.tensor([min(3., math.sqrt(len(rows) / (len(counts)*counts[(r[2],r[3])])))
                             for r in rows], dtype=torch.float32)
    selected = torch.multinomial(sampling, len(rows), replacement=True,
                                 generator=generator).tolist()
    selected.sort(key=lambda i: len(tokens[rows[i][0]][rows[i][1]]) // 32)
    buckets = [selected[i:i+BATCH*20] for i in range(0, len(selected), BATCH*20)]
    random.Random(SEED + epoch).shuffle(buckets)
    global_batches = [batch for bucket in buckets
                      for batch in token_budget_batches(bucket, rows, tokens)]
    if len(global_batches) % 2:
        global_batches.append(global_batches[-1])
    return global_batches[rank::2]


def train_sentence(fold):
    rank, device = init_dist()
    old, train, dev, _, vocab = sets()
    config = read_json(ROOT / 'config.json')
    fitting, checking = split_for(fold, train, dev, config)
    tokens = load_tokens()
    model, tokenizer = new_encoder(old, device, vocab)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    rows = sentence_rows(fitting, vocab)
    val_rows = sentence_rows(checking, vocab)
    steps_per_epoch = math.ceil(len(sentence_batches(rows, tokens, 1, fold, rank)) / ACCUM)
    fixed_epochs = SENT_EPOCHS
    if is_final(fold):
        fold_epochs = [read_json(folder(str(f)) / 'sentence_status.json')['best_epoch']
                       for f in range(FOLDS) if (folder(str(f)) / 'sentence_status.json').exists()]
        fixed_epochs = max(3, min(SENT_EPOCHS, round(sum(fold_epochs) / len(fold_epochs)))) if fold_epochs else 6
    opt, scheduler = sentence_optimizer(model, steps_per_epoch * fixed_epochs)
    scaler = torch.cuda.amp.GradScaler()
    weights = class_weights(fitting, vocab, device)
    path = folder(fold); path.mkdir(parents=True, exist_ok=True)
    latest = path / 'sentence_latest.pt'; best_path = path / 'sentence_best.pt'
    first_epoch, first_batch, step, best, best_epoch, bad = 1, 0, 0, -1., 0, 0
    if latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        first_epoch, first_batch, step = ck['epoch'], ck['next_batch'], ck['step']
        best, best_epoch, bad = ck['best'], ck['best_epoch'], ck['bad']
        if ck.get('rng') is not None:
            torch.set_rng_state(ck['rng'])
    start_time = time.time()
    last_snapshot = start_time
    deadline = float(os.environ.get('CS781_HIER_DEADLINE', time.time() + 20*3600))

    def save_latest(epoch, next_batch):
        if rank == 0:
            atomic_torch(latest, {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                                  'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                                  'rng': torch.get_rng_state(), 'epoch': epoch,
                                  'next_batch': next_batch, 'step': step, 'best': best,
                                  'best_epoch': best_epoch, 'bad': bad})

    for epoch in range(first_epoch, fixed_epochs + 1):
        model.train()
        batches = sentence_batches(rows, tokens, epoch, fold, rank)
        opt.zero_grad(set_to_none=True)
        running = 0.
        for bi in range(first_batch if epoch == first_epoch else 0, len(batches)):
            group = [rows[i] for i in batches[bi]]
            ids, mask, y, domain = collate(group, tokens, int(tokenizer.pad_token_id), device)
            update = (bi+1) % ACCUM == 0 or bi+1 == len(batches)
            context = nullcontext() if update else ddp.no_sync()
            with context:
                with autocast():
                    _, logits = ddp(ids, mask, domain)
                    loss = F.cross_entropy(logits.float(), y, weight=weights,
                                           label_smoothing=.025) / ACCUM
                finite_or_fail(loss)
                scaler.scale(loss).backward()
            running += float(loss.detach()) * ACCUM
            if update:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(opt); scaler.update(); scheduler.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 150 == 0:
                    save_latest(epoch, bi+1)
            if rank == 0 and bi % 80 == 0:
                print(f'sentence {fold} epoch {epoch} batch {bi}/{len(batches)} loss {running/(bi+1):.4f} '
                      f'GPU {torch.cuda.memory_allocated(device)/2**30:.2f}GiB', flush=True)
        dist.barrier()
        own = distributed_sentence_eval(model, checking, tokens, int(tokenizer.pad_token_id),
                                         vocab, rank, device)
        prob = gather_dict({did: value['prob'] for did, value in own.items()})
        if rank == 0:
            report = metrics(prob, checking, vocab)
            score = report['macro_supported']
            improved = is_final(fold) or score > best + .002
            if improved:
                best, best_epoch, bad = score, epoch, 0
                atomic_torch(best_path, {'model': model.state_dict(), 'epoch': epoch,
                                         'supported_f1': best, 'all13_f1': report['macro_all13']})
            else:
                bad += 1
            atomic_json(path / 'sentence_status.json', {'best_epoch': best_epoch,
                        'best_supported_f1': best, 'best_path': str(best_path),
                        'current_epoch': epoch, 'validation': report,
                        'complete': False, 'elapsed_seconds': time.time()-start_time})
            print(f'sentence {fold} epoch {epoch} val supported macro F1={score:.4f}, '
                  f'all13={report["macro_all13"]:.4f}, best={best:.4f}', flush=True)
        dist.barrier()
        state = read_json(path / 'sentence_status.json')
        best, best_epoch = state['best_supported_f1'], state['best_epoch']
        bad = epoch - best_epoch
        save_latest(epoch+1, 0)
        dist.barrier()
        if rank == 0 and time.time()-last_snapshot >= 35*60:
            snapshot(); last_snapshot = time.time()
        dist.barrier()
        cutoff = (not is_final(fold) and epoch >= 4 and bad >= 3) or epoch >= fixed_epochs
        cutoff |= (time.time()-start_time > 55*60 and epoch >= 3)
        cutoff |= time.time() > deadline-3600
        if cutoff:
            break
    if rank == 0:
        state = read_json(path / 'sentence_status.json')
        state['complete'] = True
        atomic_json(path / 'sentence_status.json', state)
        # A finished fold resumes from its best model; the much larger Adam
        # state is only needed while the stage is unfinished.
        latest.unlink(missing_ok=True)
    dist.barrier(); dist.destroy_process_group()


def extract_features(fold):
    rank, device = init_dist()
    old, train, dev, test, vocab = sets()
    tokens = load_tokens()
    path = folder(fold)
    model, tokenizer = new_encoder(old, device, vocab)
    checkpoint_data = torch.load(path / 'sentence_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint_data['model'])
    # Each GPU owns whole documents; rank files avoid a large NCCL object gather.
    for split, docs in (('train', train), ('dev', dev), ('test', test)):
        target = path / f'features_{split}.pt'
        if target.exists():
            continue
        own = distributed_sentence_eval(model, docs, tokens, int(tokenizer.pad_token_id),
                                         vocab, rank, device, features=True)
        shard = path / f'features_{split}_rank{rank}.pt'
        atomic_torch(shard, own)
        dist.barrier()
        if rank == 0:
            combined = {}
            for r in (0, 1):
                part = torch.load(path / f'features_{split}_rank{r}.pt', map_location='cpu', weights_only=False)
                combined.update(part)
            if set(combined) != {d['id'] for d in docs}:
                raise RuntimeError('Feature extraction missed documents')
            atomic_torch(target, combined)
            for r in (0, 1):
                (path / f'features_{split}_rank{r}.pt').unlink()
        dist.barrier()
    dist.destroy_process_group()


class DocumentModel(nn.Module):
    """Overlapping local attention plus a document-level sequence of window summaries."""
    def __init__(self, labels_count):
        super().__init__()
        self.position = nn.Embedding(32, 512)
        self.domain = nn.Embedding(2, 512)
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1536,
                                           dropout=.18, activation='gelu', batch_first=True,
                                           norm_first=True)
        self.local = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)
        self.global_context = nn.GRU(512, 256, num_layers=2, bidirectional=True,
                                     batch_first=True, dropout=.15)
        self.normal = nn.LayerNorm(512)
        self.head_cl = nn.Sequential(nn.Dropout(.25), nn.Linear(512, 512), nn.GELU(),
                                      nn.Dropout(.2), nn.Linear(512, labels_count))
        self.head_it = nn.Sequential(nn.Dropout(.25), nn.Linear(512, 512), nn.GELU(),
                                      nn.Dropout(.2), nn.Linear(512, labels_count))
        self.boundary = nn.Linear(1024, 1)

    def forward(self, embeddings, domain):
        n = embeddings.size(0)
        if n == 0:
            raise RuntimeError('Empty document')
        indices = torch.arange(n, device=embeddings.device)
        pos = (indices * 32 // n).clamp_max(31)
        x = embeddings + self.position(pos) + self.domain.weight[domain][None, :]
        starts = list(range(0, n, STRIDE))
        windows = []
        key_masks = []
        for start in starts:
            part = x[start:start+WINDOW]
            length = len(part)
            windows.append(F.pad(part, (0, 0, 0, WINDOW-length)))
            key_masks.append(torch.arange(WINDOW, device=x.device) >= length)
        bank = torch.stack(windows)
        key_mask = torch.stack(key_masks)
        local = self.local(bank, src_key_padding_mask=key_mask)
        lengths = (~key_mask).sum(-1).clamp_min(1)
        summaries = (local * (~key_mask).unsqueeze(-1)).sum(1) / lengths[:, None]
        global_states, _ = self.global_context(summaries[None])
        local = local + global_states[0, :, None, :]
        combined = torch.zeros_like(x)
        counts = x.new_zeros(n, 1)
        for row, start in enumerate(starts):
            ix = torch.arange(start, min(start+WINDOW, n), device=x.device)
            combined.index_add_(0, ix, local[row, :len(ix)])
            counts.index_add_(0, ix, torch.ones(len(ix), 1, device=x.device))
        combined = self.normal(combined / counts.clamp_min(1))
        logits = (self.head_cl if domain == 0 else self.head_it)(combined)
        boundary = self.boundary(torch.cat((combined[:-1], combined[1:]), -1)).squeeze(-1)
        return logits, boundary


def document_training_order(docs, epoch, rank, fold):
    rng = random.Random(SEED + epoch*53 + (0 if is_final(fold) else int(fold))*1000)
    cl = [d for d in docs if d['domain'] == 'CL']
    it = [d for d in docs if d['domain'] == 'IT']
    rng.shuffle(cl); rng.shuffle(it)
    mixed = []
    for i, pair in enumerate(zip(cl, it)):
        mixed.extend(pair if i % 2 == 0 else pair[::-1])
    if len(mixed) != len(docs):
        raise RuntimeError('Training domains are imbalanced unexpectedly')
    return mixed[rank::2]


@torch.no_grad()
def distributed_document_eval(model, docs, features, rank, device):
    model.eval(); own = {}
    for doc in docs[rank::2]:
        emb = features[doc['id']]['features'].float().to(device)
        domain = 0 if doc['domain'] == 'CL' else 1
        with autocast():
            logits, _ = model(emb, domain)
        own[doc['id']] = logits.float().softmax(-1).cpu()
    return gather_dict(own)


def train_document(fold):
    rank, device = init_dist()
    _, train, dev, _, vocab = sets()
    config = read_json(ROOT / 'config.json')
    fitting, checking = split_for(fold, train, dev, config)
    path = folder(fold)
    feature_train = torch.load(path / 'features_train.pt', map_location='cpu', weights_only=False)
    feature_check = (torch.load(path / 'features_dev.pt', map_location='cpu', weights_only=False)
                     if is_final(fold) else feature_train)
    model = DocumentModel(len(vocab)).to(device)
    ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True,
              broadcast_buffers=False)
    weights = class_weights(fitting, vocab, device)
    to_id = {y: i for i, y in enumerate(vocab)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=.05)
    scaler = torch.cuda.amp.GradScaler()
    fixed_epochs = HEAD_EPOCHS
    if is_final(fold):
        fold_epochs = [read_json(folder(str(f)) / 'head_status.json')['best_epoch']
                       for f in range(FOLDS) if (folder(str(f)) / 'head_status.json').exists()]
        fixed_epochs = max(5, min(HEAD_EPOCHS, round(sum(fold_epochs)/len(fold_epochs)))) if fold_epochs else 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fixed_epochs)
    latest = path / 'head_latest.pt'; best_path = path / 'head_best.pt'
    first_epoch, best, best_epoch = 1, -1., 0
    if latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        first_epoch, best, best_epoch = ck['epoch'], ck['best'], ck['best_epoch']
    start = time.time()
    last_snapshot = start
    deadline = float(os.environ.get('CS781_HIER_DEADLINE', time.time()+20*3600))
    for epoch in range(first_epoch, fixed_epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        order = document_training_order(fitting, epoch, rank, fold)
        total = 0.
        for j, doc in enumerate(order):
            emb = feature_train[doc['id']]['features'].float().to(device)
            gold = torch.tensor([to_id[y] for y in doc['labels']], dtype=torch.long, device=device)
            domain = 0 if doc['domain'] == 'CL' else 1
            update = (j+1) % ACCUM == 0 or j+1 == len(order)
            context = nullcontext() if update else ddp.no_sync()
            with context:
                with autocast():
                    logits, boundary = ddp(emb, domain)
                    ce = F.cross_entropy(logits.float(), gold, weight=weights,
                                         label_smoothing=.015)
                    if len(gold) > 1:
                        changes = (gold[1:] != gold[:-1]).float()
                        positive = torch.tensor(min(8., max(1., (len(changes)-changes.sum().item()) /
                                                          max(1., changes.sum().item()))), device=device)
                        aux = F.binary_cross_entropy_with_logits(boundary.float(), changes,
                                                                 pos_weight=positive)
                    else:
                        aux = ce.new_zeros(())
                    loss = (ce + .12*aux) / ACCUM
                finite_or_fail(loss)
                scaler.scale(loss).backward()
            total += float(loss.detach())*ACCUM
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step(); dist.barrier()
        prob = distributed_document_eval(model, checking, feature_check, rank, device)
        if rank == 0:
            report = metrics(prob, checking, vocab)
            score = report['macro_supported']
            improved = (is_final(fold) or score > best+.002)
            if improved:
                best, best_epoch = score, epoch
                atomic_torch(best_path, {'model': model.state_dict(), 'epoch': epoch,
                                         'supported_f1': score, 'all13_f1': report['macro_all13']})
            atomic_json(path / 'head_status.json', {'best_epoch': best_epoch,
                        'best_supported_f1': best, 'current_epoch': epoch,
                        'validation': report, 'complete': False,
                        'elapsed_seconds': time.time()-start})
            atomic_torch(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                  'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                                  'epoch': epoch+1, 'best': best, 'best_epoch': best_epoch})
            print(f'document {fold} epoch {epoch}: train loss {total/len(order):.4f}, '
                  f'val supported macro F1 {score:.4f}, all13 {report["macro_all13"]:.4f}, '
                  f'GPU {torch.cuda.memory_allocated(device)/2**30:.2f}GiB', flush=True)
        dist.barrier()
        if rank == 0 and time.time()-last_snapshot >= 35*60:
            snapshot(); last_snapshot = time.time()
        dist.barrier()
        best_epoch = read_json(path / 'head_status.json')['best_epoch']
        stop = not is_final(fold) and epoch >= 5 and epoch-best_epoch >= 4
        stop |= epoch >= fixed_epochs or (time.time()-start > 25*60 and epoch >= 4)
        stop |= time.time() > deadline-3600
        if stop:
            break
    if rank == 0:
        state = read_json(path / 'head_status.json')
        state['complete'] = True
        atomic_json(path / 'head_status.json', state)
        latest.unlink(missing_ok=True)
    dist.barrier(); dist.destroy_process_group()


def predict_head(fold, split):
    rank, device = init_dist()
    _, train, dev, test, vocab = sets()
    docs = {'train': train, 'dev': dev, 'test': test}[split]
    path = folder(fold)
    features = torch.load(path / f'features_{split}.pt', map_location='cpu', weights_only=False)
    model = DocumentModel(len(vocab)).to(device)
    ck = torch.load(path / 'head_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    prob = distributed_document_eval(model, docs, features, rank, device)
    if rank == 0:
        atomic_torch(path / f'head_prob_{split}.pt', prob)
        if split in ('dev', 'train'):
            labelled = docs if split == 'dev' else (
                docs if is_final(fold) else outer_for(fold, train, read_json(ROOT / 'config.json')))
            report = metrics(prob, labelled, vocab)
            atomic_json(path / f'head_{split}_metrics.json', report)
    dist.barrier(); dist.destroy_process_group()


def prior_prob(split):
    target = ROOT / f'prior6_prob_{split}.pt'
    if target.exists():
        return torch.load(target, map_location='cpu', weights_only=False)
    old, _, dev, test, vocab = sets()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = old.load_selected(6, device)
    docs = dev if split == 'dev' else test
    prob = old.probabilities(model, docs, tokenizer, vocab, device)
    atomic_torch(target, prob)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return prob


def finished_models():
    names = [str(f) for f in range(FOLDS) if (folder(str(f)) / 'head_best.pt').is_file()]
    if (folder('final') / 'head_best.pt').is_file():
        names.append('final')
    return names


def combine_new(split, names, docs):
    chosen = [f for f in names if (folder(f) / f'head_prob_{split}.pt').is_file()]
    if set(chosen) != set(names):
        raise FileNotFoundError(f'Missing {split} probabilities: {set(names)-set(chosen)}')
    parts = {f: torch.load(folder(f) / f'head_prob_{split}.pt', map_location='cpu', weights_only=False)
             for f in chosen}
    fold_names = [f for f in names if f != 'final']
    if not fold_names:
        return {d['id']: parts['final'][d['id']] for d in docs}, {'final': 1.0}
    weights = {f: (0.7 / len(fold_names) if 'final' in names else 1. / len(fold_names))
               for f in fold_names}
    if 'final' in names:
        weights['final'] = .3
    return {d['id']: sum(weights[f] * parts[f][d['id']] for f in names) for d in docs}, weights


def select():
    _, train, dev, _, vocab = sets()
    prior = prior_prob('dev')
    prior_report = metrics(prior, dev, vocab)
    names = [f for f in finished_models() if (folder(f) / 'head_prob_dev.pt').is_file()]
    folds = [f for f in names if f != 'final' and (folder(f) / 'head_prob_train.pt').is_file()]
    oof = {}
    oof_sentence = {}
    config = read_json(ROOT / 'config.json')
    for f in folds:
        part = torch.load(folder(f) / 'head_prob_train.pt', map_location='cpu', weights_only=False)
        encoded = torch.load(folder(f) / 'features_train.pt', map_location='cpu', weights_only=False)
        for doc in train:
            if config['fold_map'][doc['id']] == int(f):
                oof[doc['id']] = part[doc['id']]
                oof_sentence[doc['id']] = encoded[doc['id']]['prob']
    oof_docs = [d for d in train if d['id'] in oof]
    oof_report = metrics(oof, oof_docs, vocab) if oof_docs else None
    sentence_oof_report = metrics(oof_sentence, oof_docs, vocab) if oof_docs else None
    selected = 'prior6'; all_reports = {'prior6': prior_report}
    weights = {}
    if names:
        new, weights = combine_new('dev', names, dev)
        blend = {d['id']: .75*new[d['id']] + .25*prior[d['id']] for d in dev}
        all_reports['new'] = metrics(new, dev, vocab)
        all_reports['blend'] = metrics(blend, dev, vocab)
        # A fixed, predeclared gate limits overfitting to the ten-document dev
        # set. No weight/transition grid is searched on development labels.
        oof_gain = (oof_report is not None and len(folds) >= 3 and
                    oof_report['macro_supported'] >= sentence_oof_report['macro_supported'] + .01)
        domain_ok = all(all_reports['new']['domain_all13'][domain] >=
                        prior_report['domain_all13'][domain] - .035 for domain in ('CL', 'IT'))
        candidates = []
        if oof_gain and domain_ok:
            if all_reports['new']['macro_supported'] >= prior_report['macro_supported'] + .025:
                candidates.append('new')
            if all_reports['blend']['macro_supported'] >= prior_report['macro_supported'] + .015:
                candidates.append('blend')
        if candidates:
            selected = max(candidates, key=lambda c: all_reports[c]['macro_supported'])
    record = {'selected': selected, 'models': names, 'folds_with_oof': folds,
              'new_model_weights': weights, 'development': all_reports,
              'oof': oof_report, 'sentence_oof': sentence_oof_report,
              'private_score_reference_prior6': .5058,
              'selection_rule': 'Fixed OOF gain, both-domain audit, and dev gain thresholds; no dev grid search.'}
    atomic_json(ROOT / 'selection.json', record)
    print('Selection:', selected, 'prior dev supported', prior_report['macro_supported'],
          'new dev supported', all_reports.get('new', {}).get('macro_supported'),
          'OOF supported', None if oof_report is None else oof_report['macro_supported'], flush=True)


def finalize():
    _, _, _, test, vocab = sets()
    selection = read_json(ROOT / 'selection.json')
    chosen = selection['selected']
    prior = prior_prob('test') if chosen in ('prior6', 'blend') else None
    if chosen in ('new', 'blend'):
        new, _ = combine_new('test', selection['models'], test)
        prob = (new if chosen == 'new' else
                {d['id']: .75*new[d['id']] + .25*prior[d['id']] for d in test})
    else:
        prob = prior
    prediction = {d['id']: [vocab[int(i)] for i in prob[d['id']].argmax(-1)] for d in test}
    if set(prediction) != {d['id'] for d in test} or any(
            len(prediction[d['id']]) != len(d['text']) for d in test):
        raise RuntimeError('Submission IDs or sentence counts do not match official test data')
    if any(y not in vocab for values in prediction.values() for y in values):
        raise RuntimeError('Unknown submission label')
    atomic_json(ROOT / 'prediction.json', prediction)
    temp = SUBMISSION.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / 'prediction.json', 'prediction.json')
    os.replace(temp, SUBMISSION)
    with zipfile.ZipFile(SUBMISSION) as z:
        if z.namelist() != ['prediction.json'] or json.loads(z.read('prediction.json')) != prediction:
            raise RuntimeError('Invalid submission ZIP')
    shutil.copyfile(SUBMISSION, WORK / 'submission.zip')
    shutil.copyfile(SUBMISSION, ROOT / 'submission.zip')
    atomic_json(ROOT / 'report.json', {'status': 'READY_FOR_MANUAL_UPLOAD',
                'submission': str(SUBMISSION), 'submission_sha256': digest(SUBMISSION),
                'selected': chosen, 'test_documents': len(test),
                'test_sentences': sum(len(d['text']) for d in test),
                'development_supported_macro_f1':
                selection['development'][chosen]['macro_supported'],
                'development_all13_macro_f1': selection['development'][chosen]['macro_all13'],
                'private_score': None,
                'note': '0.8 is a target, not a guaranteed or measured private score.'})
    snapshot()
    print('Submission ready:', SUBMISSION, flush=True)
    print('Recovery ready:', ARTIFACT, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'sentence', 'extract', 'head',
                                           'predict-head', 'select', 'finalize', 'snapshot'))
    parser.add_argument('--fold', default='0')
    parser.add_argument('--split', choices=('train', 'dev', 'test'), default='dev')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'sentence': train_sentence(args.fold)
    elif args.action == 'extract': extract_features(args.fold)
    elif args.action == 'head': train_document(args.fold)
    elif args.action == 'predict-head': predict_head(args.fold, args.split)
    elif args.action == 'select': select()
    elif args.action == 'finalize': finalize()
    else: snapshot()


if __name__ == '__main__':
    main()

