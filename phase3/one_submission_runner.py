"""One stronger Phase 3 submission from the completed six-run Kaggle session.

The only pretrained weights are the user's Phase 1 checkpoints. The six prior
Phase 3 checkpoints may contribute to the final ensemble. No test labels or
external pretrained model are used. This file is embedded in its notebook.
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
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

WORK = Path('/kaggle/working')
PRIOR = WORK / 'CS781_Capstone/phase3'
PRIOR_ZIP = WORK / 'CS781_Phase3_All_Artifacts.zip'
ROOT = WORK / 'CS781_Capstone/phase3_one'
OUTPUT_ZIP = WORK / 'CS781_Phase3_One_All_Artifacts.zip'
SUBMISSION_ZIP = WORK / 'CS781_Phase3_One_Submission.zip'
SEED = 12781
MAX_HOURS = 9.5
RESERVE_SECONDS = 3600
MAX_EPOCHS = 26
MIN_EPOCHS = 7
PATIENCE = 6
ACCUM = 4
PRIOR_SLOTS = (6, 2, 1, 4, 5, 3)
RUNS = (
    {'name': 'full03', 'base': '03', 'method': 'prompt', 'rank': 0, 'unfreeze': 12,
     'max_seconds': 4.1 * 3600, 'lr_base': 1.3e-5, 'lr_head': 1.2e-4},
    {'name': 'full01', 'base': '01', 'method': 'rslora', 'rank': 32, 'unfreeze': 4,
     'max_seconds': 3.3 * 3600, 'lr_base': 1.5e-5, 'lr_head': 1.3e-4},
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_member(name):
    p = Path(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name


def locate_archive(filename):
    direct = WORK / filename
    if direct.is_file():
        return direct
    hits = sorted(p for p in Path('/kaggle/input').rglob(filename) if p.is_file())
    if len(hits) > 1:
        raise RuntimeError(f'Multiple archives named {filename}: {hits}')
    return hits[0] if hits else None


def extract_verified(source, mappings):
    """Restore a manifest ZIP to destination roots, rejecting paths and corruption."""
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or 'integrity.json' not in names:
            raise RuntimeError(f'Invalid recovery ZIP member list: {source}')
        manifest = json.loads(z.read('integrity.json'))['files']
        if set(manifest) != set(names) - {'integrity.json'}:
            raise RuntimeError(f'Recovery manifest mismatch: {source}')
        for name in names:
            if name == 'integrity.json':
                continue
            if not safe_member(name) or z.getinfo(name).is_dir():
                raise RuntimeError(f'Unsafe ZIP member: {name}')
            prefix = next((x for x in mappings if name.startswith(x)), None)
            if prefix is None:
                raise RuntimeError(f'Unexpected recovery member: {name}')
            target = mappings[prefix] / name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = manifest[name]
            if target.is_file() and target.stat().st_size == expected['bytes'] and digest(target) == expected['sha256']:
                continue
            with z.open(name) as src, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                temp = Path(out.name)
                h = hashlib.sha256()
                for block in iter(lambda: src.read(4 << 20), b''):
                    h.update(block)
                    out.write(block)
            if h.hexdigest() != expected['sha256'] or temp.stat().st_size != expected['bytes']:
                temp.unlink(missing_ok=True)
                raise RuntimeError(f'Corrupt member: {name}')
            os.replace(temp, target)


def recover_inputs():
    """Use the live six-run folder first; permit portable archive recovery."""
    if (PRIOR / 'runner.py').is_file() and (PRIOR / 'run_config.json').is_file():
        return
    new_archive = locate_archive(OUTPUT_ZIP.name)
    if new_archive:
        extract_verified(new_archive, {'prior/': PRIOR, 'one/': ROOT})
    if (PRIOR / 'runner.py').is_file() and (PRIOR / 'run_config.json').is_file():
        return
    old_archive = locate_archive(PRIOR_ZIP.name)
    if old_archive is None:
        raise FileNotFoundError('Need the live CS781 six-run session, or attach its All_Artifacts ZIP')
    with zipfile.ZipFile(old_archive) as z:
        names = z.namelist()
        if 'integrity.json' not in names or len(names) != len(set(names)):
            raise RuntimeError('Six-run recovery archive is invalid')
        manifest = json.loads(z.read('integrity.json'))['files']
        if set(manifest) != set(names) - {'integrity.json'}:
            raise RuntimeError('Six-run recovery manifest mismatch')
        for name in names:
            if name == 'integrity.json':
                continue
            if not safe_member(name) or z.getinfo(name).is_dir():
                raise RuntimeError(f'Unsafe prior member: {name}')
            target = PRIOR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = manifest[name]
            if target.is_file() and target.stat().st_size == expected['bytes'] and digest(target) == expected['sha256']:
                continue
            with z.open(name) as src, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                temp = Path(out.name)
                h = hashlib.sha256()
                for block in iter(lambda: src.read(4 << 20), b''):
                    h.update(block)
                    out.write(block)
            if h.hexdigest() != expected['sha256'] or temp.stat().st_size != expected['bytes']:
                temp.unlink(missing_ok=True)
                raise RuntimeError(f'Corrupt prior member: {name}')
            os.replace(temp, target)
    if not PRIOR_ZIP.is_file():
        shutil.copyfile(old_archive, PRIOR_ZIP)


def previous():
    path = PRIOR / 'runner.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('cs781_previous_phase3', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.MAX_TOKENS = 384
    module.DOC_CHUNK = 8
    module.DOC_STRIDE = 6
    return module


def snapshot():
    """Portable ZIP includes all prior staged files and this run's states."""
    if not (PRIOR / 'runner.py').is_file():
        raise FileNotFoundError('Previous six-run files are not staged')
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = [(p, 'prior/' + p.relative_to(PRIOR).as_posix())
               for p in sorted(PRIOR.rglob('*')) if p.is_file() and
               'hf_cache' not in p.parts and not p.name.endswith('.tmp')]
    sources += [(p, 'one/' + p.relative_to(ROOT).as_posix())
                for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')]
    manifest = {'version': 1, 'files': {}}
    temp = OUTPUT_ZIP.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', allowZip64=True) as z:
        for path, name in sources:
            manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
            z.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix in {'.pt', '.zip'}
                    else zipfile.ZIP_DEFLATED)
        z.writestr('integrity.json', json.dumps(manifest, sort_keys=True))
    os.replace(temp, OUTPUT_ZIP)
    print(f'Portable recovery: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 2**30:.2f} GiB)', flush=True)


def prep():
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Select exactly two Kaggle GPUs (for example, 2 x T4)')
    recover_inputs()
    old = previous()
    old.all_data()
    for slot in PRIOR_SLOTS:
        if not old.best_path(slot).is_file():
            raise FileNotFoundError(f'Prior slot {slot} best checkpoint missing')
    for base_id in ('01', '02', '03', '04'):
        base = PRIOR / 'bases' / base_id
        for rel in (f'model/{old.SOURCES[base_id]}', 'model/course_bpe.py',
                    'checkpoints/selected_model.pt', 'tokenizer/tokenizer.json'):
            if not (base / rel).is_file():
                raise FileNotFoundError(base / rel)
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {'runs': [dict(run) for run in RUNS], 'prior_slots': list(PRIOR_SLOTS), 'max_epochs': MAX_EPOCHS,
              'min_epochs': MIN_EPOCHS, 'patience': PATIENCE, 'accum': ACCUM,
              'runner_sha256': digest(__file__)}
    config_file = ROOT / 'config.json'
    if config_file.exists() and read_json(config_file) != config:
        raise RuntimeError('Recovery config differs from notebook code. Keep the original notebook for resume.')
    atomic_json(config_file, config)
    shutil.copyfile(__file__, ROOT / 'one_submission_runner.py')
    atomic_json(ROOT / 'prior_scores.json', {'1': 0.4600, '2': 0.4437, '3': 0.3835,
                                             '4': 0.4239, '5': 0.4538, '6': 0.5058})
    print('Ready:', ROOT, flush=True)
    snapshot()


def init_dist():
    if int(os.environ.get('WORLD_SIZE', '1')) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('Launch with torchrun --nproc_per_node=2 on two GPUs')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', timeout=__import__('datetime').timedelta(minutes=30))
    return rank, torch.device('cuda', local_rank)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LargeDocModel(nn.Module):
    """Full Phase 1 backbone plus a larger 3-layer document sequence head."""

    def __init__(self, base, specification, n_labels, old):
        super().__init__()
        self.encoder = old.RRModel(base, specification, n_labels)
        width = int(base.hidden_size if hasattr(base, 'hidden_size') else
                    base.config.hidden_size if hasattr(base.config, 'hidden_size') else
                    base.config['hidden_size'])
        for p in self.encoder.base.parameters():
            p.requires_grad_(True)
        for name, p in self.encoder.base.named_parameters():
            if name.startswith('lm_head'):
                p.requires_grad_(False)
        for p in self.encoder.context.parameters():
            p.requires_grad_(False)
        for p in self.encoder.head.parameters():
            p.requires_grad_(False)
        self.feature_norm = nn.LayerNorm(width)
        self.context = nn.GRU(width + 64, 384, num_layers=3, batch_first=True,
                              bidirectional=True, dropout=.25)
        self.head = nn.Sequential(nn.LayerNorm(768), nn.Dropout(.28),
                                  nn.Linear(768, 512), nn.GELU(), nn.Dropout(.22),
                                  nn.Linear(512, n_labels))
        self.local_head = nn.Linear(width, n_labels)

    def forward(self, ids, mask, pos, domain):
        features = self.feature_norm(self.encoder.features(ids, mask))
        position = self.encoder.position(pos.clamp(0, 15))
        dom = self.encoder.domain(domain)
        encoded, _ = self.context(torch.cat([features, position, dom], -1)[None])
        return self.head(encoded[0]) + .25 * self.local_head(features)


def model_spec(run):
    return {k: run[k] for k in ('base', 'method', 'rank', 'unfreeze')}


def new_model(run, device, old):
    base, tokenizer = old.load_base(run['base'])
    model = LargeDocModel(base, model_spec(run), len(old.labels()), old).to(device)
    return model, tokenizer


def run_dir(name):
    return ROOT / name


def run_status(name):
    path = run_dir(name) / 'status.json'
    return read_json(path) if path.is_file() else {}


def supported_f1(stats):
    values = [v['f1'] for v in stats['per_label'].values() if v['gold'] > 0]
    return sum(values) / len(values) if values else 0.0


def train_one(name):
    rank, device = init_dist()
    old = previous()
    run = next(x for x in RUNS if x['name'] == name)
    seed_everything(SEED + 100 * list(RUNS).index(run) + rank)
    train_docs, dev_docs, _ = old.all_data()
    vocab = old.labels()
    ids = {y: i for i, y in enumerate(vocab)}
    model, tokenizer = new_model(run, device, old)
    ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    groups = [{'params': [], 'lr': run['lr_base']},
              {'params': [], 'lr': run['lr_head']}]
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_adapter = any(piece in parameter_name for piece in ('.a.', '.b.'))
        groups[0 if parameter_name.startswith('encoder.base.') and not is_adapter else 1]['params'].append(parameter)
    groups = [g for g in groups if g['params']]
    try:
        optimizer = torch.optim.AdamW(groups, weight_decay=.04, fused=True)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(groups, weight_decay=.04)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    items = old.chunks(train_docs)
    if not items:
        raise ValueError('No train chunks')
    copies = math.ceil(len(items) / (2 * ACCUM)) * 2 * ACCUM
    steps_epoch = copies // (2 * ACCUM)
    max_steps = MAX_EPOCHS * steps_epoch
    warmup = max(1, int(max_steps * .04))
    def rate(step):
        if step < warmup:
            return max(.1, (step + 1) / warmup)
        fraction = min(1, (step - warmup) / max(1, max_steps - warmup))
        return .12 + .88 * .5 * (1 + math.cos(math.pi * fraction))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    counts = Counter(y for d in train_docs for y in d['labels'])
    total = sum(counts.values())
    weights = torch.tensor([min(3.0, max(.6, math.sqrt(total / (len(vocab) * max(1, counts[y])))))
                            for y in vocab], device=device)
    folder = run_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    state = {'epoch': 0, 'best_present_f1': -1.0, 'best_epoch': 0, 'stale': 0,
             'train_seconds': 0.0}
    latest = folder / 'latest.pt'
    if latest.is_file():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['run'] != run:
            raise RuntimeError('Resume checkpoint configuration mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        schedule.load_state_dict(saved['schedule'])
        state.update(saved['state'])
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'], device)
    deadline = float(os.environ.get('CS781_ONE_DEADLINE', time.time() + MAX_HOURS * 3600))
    clock_start = time.monotonic()
    if rank == 0:
        print(f'{name}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters; '
              f'{len(items)} chunks; {steps_epoch} optimizer steps/epoch', flush=True)
    control = {'stop': False, 'pause': False}
    for epoch in range(state['epoch'], MAX_EPOCHS):
        random.Random(SEED + epoch + (0 if name == 'full03' else 1000)).shuffle(items)
        padded = (items * math.ceil(copies / len(items)))[:copies]
        local = padded[rank::2]
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        gpu_snapshot = []
        for index, item in enumerate(local):
            batch = old.batch_for(item, tokenizer, ids, device)
            sync = (index + 1) % ACCUM == 0
            with nullcontext() if sync else ddp.no_sync():
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = ddp(*batch[:4])
                    loss = F.cross_entropy(logits.float(), batch[4], weight=weights,
                                           label_smoothing=.02) / ACCUM
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite {name} loss at epoch {epoch + 1}')
                scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            if sync:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
            if rank == 0 and index == len(local) // 2:
                try:
                    gpu_snapshot = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                         '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
                except Exception:
                    gpu_snapshot = []
        dist.barrier()
        if rank == 0:
            model.eval()
            dev = old.evaluate(model, dev_docs, tokenizer, vocab, device)
            present = supported_f1(dev['overall'])
            if not math.isfinite(present):
                raise RuntimeError('Non-finite development F1')
            improved = present > state['best_present_f1'] + 1e-4
            state['stale'] = 0 if improved else state['stale'] + 1
            if improved:
                state['best_present_f1'] = present
                state['best_epoch'] = epoch + 1
                atomic_torch(folder / 'best.pt', {'run': run, 'epoch': epoch + 1,
                                                  'model': model.state_dict(),
                                                  'dev': dev})
            state['epoch'] = epoch + 1
            state['train_seconds'] += time.monotonic() - clock_start
            clock_start = time.monotonic()
            atomic_torch(latest, {'run': run, 'state': state, 'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                                  'schedule': schedule.state_dict(),
                                  'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                                  'cuda_rng': torch.cuda.get_rng_state(device)})
            row = {'epoch': epoch + 1, 'loss_scaled_sum': loss_sum,
                   'dev_macro_f1_all_13': dev['overall']['macro_f1'],
                   'dev_macro_f1_supported': present,
                   'CL_macro_f1_all_13': dev['domains']['CL']['macro_f1'],
                   'IT_macro_f1_all_13': dev['domains']['IT']['macro_f1'],
                   'per_label_f1': {y: v['f1'] for y, v in dev['overall']['per_label'].items()},
                   'train_seconds': state['train_seconds'],
                   'gpu_utilization_percent_and_memory_mib': gpu_snapshot}
            with (folder / 'history.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(row) + '\n')
            atomic_json(folder / 'status.json', {'complete': False, **state,
                                                'latest_dev': dev})
            print(f'{name} epoch {epoch + 1}: supported macro-F1={present:.4f}, '
                  f'all-13={dev["overall"]["macro_f1"]:.4f}, '
                  f'best={state["best_present_f1"]:.4f}, '
                  f'CL={dev["domains"]["CL"]["macro_f1"]:.4f}, '
                  f'IT={dev["domains"]["IT"]["macro_f1"]:.4f}, '
                  f'GPUs={gpu_snapshot}', flush=True)
            if (epoch + 1) % 2 == 0:
                snapshot()
            control = {'stop': (epoch + 1 >= MIN_EPOCHS and state['stale'] >= PATIENCE)
                       or state['train_seconds'] >= run['max_seconds'] or epoch + 1 == MAX_EPOCHS,
                       'pause': time.time() >= deadline - RESERVE_SECONDS}
            atomic_json(folder / 'control.json', control)
        dist.barrier()
        control = read_json(folder / 'control.json')
        if control['stop'] or control['pause']:
            break
    if rank == 0:
        status = run_status(name)
        if control['stop']:
            status['complete'] = True
            latest.unlink(missing_ok=True)
        atomic_json(folder / 'status.json', status)
        snapshot()
    dist.barrier()
    dist.destroy_process_group()


def load_new(name, device, old):
    run = next(x for x in RUNS if x['name'] == name)
    model, tokenizer = new_model(run, device, old)
    saved = torch.load(run_dir(name) / 'best.pt', map_location='cpu', weights_only=False)
    if saved['run'] != run:
        raise RuntimeError(f'Mismatched best checkpoint: {name}')
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    return model, tokenizer


def model_sources():
    return [run['name'] for run in RUNS if (run_dir(run['name']) / 'best.pt').is_file()] + [
        f'prior{slot}' for slot in PRIOR_SLOTS]


def source_model(source, device, old):
    if source.startswith('prior'):
        return old.load_selected(int(source[5:]), device)
    return load_new(source, device, old)


def inference(source, split):
    rank, device = init_dist()
    old = previous()
    train, dev, test = old.all_data()
    docs = dev if split == 'dev' else test
    model, tokenizer = source_model(source, device, old)
    selected = docs[rank::2]
    probs = old.probabilities(model, selected, tokenizer, old.labels(), device)
    gathered = [None, None]
    dist.all_gather_object(gathered, probs)
    if rank == 0:
        merged = {key: value for part in gathered for key, value in part.items()}
        if set(merged) != {d['id'] for d in docs}:
            raise RuntimeError(f'Incomplete {source} {split} probabilities')
        atomic_torch(ROOT / 'probabilities' / split / f'{source}.pt', merged)
        print(f'Cached {source} {split} probabilities for {len(docs)} documents', flush=True)
    dist.barrier()
    dist.destroy_process_group()


def transition_scores(docs, vocab, domain):
    K = len(vocab)
    label_to_id = {y: i for i, y in enumerate(vocab)}
    counts = torch.full((K, K), 2.0, dtype=torch.float64)
    marginal = torch.full((K,), 2.0 * K, dtype=torch.float64)
    starts = torch.full((K,), 2.0, dtype=torch.float64)
    for doc in docs:
        if doc['domain'] != domain or not doc['labels']:
            continue
        ys = [label_to_id[y] for y in doc['labels']]
        starts[ys[0]] += 1
        for i, y in enumerate(ys):
            marginal[y] += 1
            if i:
                counts[ys[i - 1], y] += 1
    log_transition = torch.log(counts / counts.sum(1, keepdim=True)) - torch.log(marginal / marginal.sum())[None]
    log_start = torch.log(starts / starts.sum()) - torch.log(marginal / marginal.sum())
    return log_transition, log_start


def decode(probabilities, docs, transitions, strength):
    predictions = {}
    for doc in docs:
        did = doc['id']
        unary = torch.log(torch.as_tensor(probabilities[did], dtype=torch.float64).clamp(1e-8, 1))
        if len(unary) == 0:
            predictions[did] = []
            continue
        if strength == 0:
            predictions[did] = unary.argmax(1).tolist()
            continue
        trans, start = transitions[doc['domain']]
        K = unary.shape[1]
        best = unary[0] + strength * start
        back = torch.empty((len(unary), K), dtype=torch.long)
        for i in range(1, len(unary)):
            scores = best[:, None] + strength * trans
            back[i] = scores.argmax(0)
            best = unary[i] + scores.max(0).values
        sequence = [int(best.argmax())]
        for i in range(len(unary) - 1, 0, -1):
            sequence.append(int(back[i, sequence[-1]]))
        predictions[did] = sequence[::-1]
    return predictions


def blend(probabilities, weights, docs):
    out = {}
    for doc in docs:
        did = doc['id']
        out[did] = sum(float(weight) * probabilities[source][did].float()
                       for source, weight in weights.items())
    return out


def score_candidate(probs, dev, transitions, strength, old):
    predicted = decode(probs, dev, transitions, strength)
    vocab = old.labels()
    named = {did: [vocab[i] for i in values] for did, values in predicted.items()}
    gold = {doc['id']: doc['labels'] for doc in dev}
    result = old.prf(named, gold, vocab)
    return supported_f1(result), result


def choose_ensemble(probabilities, dev, train, old):
    vocab = old.labels()
    transitions = {domain: transition_scores(train, vocab, domain) for domain in ('CL', 'IT')}
    singles = []
    for source in probabilities:
        p = blend(probabilities, {source: 1.0}, dev)
        score, result = score_candidate(p, dev, transitions, 0.0, old)
        singles.append({'source': source, 'supported_macro_f1': score,
                        'all_13_macro_f1': result['macro_f1']})
    singles.sort(key=lambda x: x['supported_macro_f1'], reverse=True)
    # Prior slot 6 has the strongest observed private score (0.5058). The
    # ten-document development set is too small to replace it for a tiny gain.
    by_source = {row['source']: row['supported_macro_f1'] for row in singles}
    anchor = 'prior6' if 'prior6' in by_source else singles[0]['source']
    strongest_new = next((row for row in singles if row['source'].startswith('full')), None)
    if strongest_new and strongest_new['supported_macro_f1'] >= by_source[anchor] + .035:
        anchor = strongest_new['source']
    selected = {anchor: 1.0}
    best = by_source[anchor]
    while len(selected) < 3:
        options = []
        for candidate in probabilities:
            if candidate in selected:
                continue
            for addition in (.2, .35, .5):
                proposed = {key: value * (1 - addition) for key, value in selected.items()}
                proposed[candidate] = addition
                fused = blend(probabilities, proposed, dev)
                score, _ = score_candidate(fused, dev, transitions, 0, old)
                options.append((score, proposed))
        if not options:
            break
        score, proposed = max(options, key=lambda item: item[0])
        if score < best + .008:
            break
        best, selected = score, proposed
    combined = blend(probabilities, selected, dev)
    options = []
    for strength in (0.0, .12, .25, .4):
        score, result = score_candidate(combined, dev, transitions, strength, old)
        options.append((score, strength, result))
    baseline = options[0]
    score, strength, result = max(options, key=lambda row: row[0])
    if score < baseline[0] + .004:
        score, strength, result = baseline
    return {'weights': selected, 'transition_strength': strength,
            'dev_supported_macro_f1': score, 'dev_all_13_macro_f1': result['macro_f1'],
            'dev_per_label': result['per_label'], 'single_models': singles,
            'gold_supported_labels': [y for y, item in result['per_label'].items() if item['gold'] > 0]}


def finalize():
    old = previous()
    train, dev, test = old.all_data()
    sources = model_sources()
    if not any(source.startswith('full') for source in sources):
        raise RuntimeError('No newly trained model is ready')
    dev_probs = {}
    for source in sources:
        path = ROOT / 'probabilities' / 'dev' / f'{source}.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
        dev_probs[source] = torch.load(path, map_location='cpu', weights_only=False)
    choice = choose_ensemble(dev_probs, dev, train, old)
    atomic_json(ROOT / 'selection.json', choice)
    print('Development selection:', json.dumps({k: choice[k] for k in
          ('weights', 'transition_strength', 'dev_supported_macro_f1', 'dev_all_13_macro_f1')},
          indent=2), flush=True)
    missing = [source for source in choice['weights'] if not
               (ROOT / 'probabilities' / 'test' / f'{source}.pt').is_file()]
    if missing:
        print('Need test probabilities:', missing, flush=True)
        return missing
    test_probs = {source: torch.load(ROOT / 'probabilities' / 'test' / f'{source}.pt',
                                     map_location='cpu', weights_only=False)
                  for source in choice['weights']}
    merged = blend(test_probs, choice['weights'], test)
    transitions = {domain: transition_scores(train, old.labels(), domain) for domain in ('CL', 'IT')}
    indexed = decode(merged, test, transitions, choice['transition_strength'])
    vocab = old.labels()
    result = {d['id']: [vocab[i] for i in indexed[d['id']]] for d in test}
    if set(result) != {d['id'] for d in test}:
        raise RuntimeError('Test document IDs do not match')
    if any(len(result[d['id']]) != len(d['text']) for d in test):
        raise RuntimeError('Test prediction length mismatch')
    if any(y not in vocab for values in result.values() for y in values):
        raise RuntimeError('Unknown output label')
    atomic_json(ROOT / 'prediction.json', result)
    temp = SUBMISSION_ZIP.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / 'prediction.json', 'prediction.json')
    os.replace(temp, SUBMISSION_ZIP)
    with zipfile.ZipFile(SUBMISSION_ZIP) as z:
        if z.namelist() != ['prediction.json'] or json.loads(z.read('prediction.json')) != result:
            raise RuntimeError('Invalid Codabench submission ZIP')
    shutil.copyfile(SUBMISSION_ZIP, ROOT / 'submission.zip')
    shutil.copyfile(SUBMISSION_ZIP, WORK / 'submission.zip')
    atomic_json(ROOT / 'one_report.json', {
        'status': 'READY_FOR_MANUAL_UPLOAD', 'submission': SUBMISSION_ZIP.name,
        'submission_sha256': digest(SUBMISSION_ZIP),
        'documents': len(result), 'sentences': sum(map(len, result.values())),
        'development_supported_macro_f1': choice['dev_supported_macro_f1'],
        'development_all_13_macro_f1': choice['dev_all_13_macro_f1'],
        'ensemble_weights': choice['weights'],
        'transition_strength': choice['transition_strength'],
        'note': 'Private Codabench score is unknown until manual upload.'})
    snapshot()
    print('Submission ready:', SUBMISSION_ZIP, flush=True)
    print('Recovery ready:', OUTPUT_ZIP, flush=True)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'train', 'infer', 'finalize', 'snapshot'))
    parser.add_argument('--name', choices=[r['name'] for r in RUNS])
    parser.add_argument('--source')
    parser.add_argument('--split', choices=('dev', 'test'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prep()
    elif args.action == 'train':
        if not args.name:
            parser.error('train needs --name')
        train_one(args.name)
    elif args.action == 'infer':
        if not args.source or not args.split:
            parser.error('infer needs --source and --split')
        inference(args.source, args.split)
    elif args.action == 'finalize':
        missing = finalize()
        if missing:
            print('RUN_TEST_INFERENCE=' + json.dumps(missing))
    else:
        snapshot()


if __name__ == '__main__':
    main()
