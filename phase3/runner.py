"""Self-contained CS781 Phase 3 runner, embedded in the Run All notebook.

AI-assisted reference implementation. Review it and disclose assistance under
the course rules. Only the four user's Phase 1 checkpoints and official Phase 3
data are used. Private test labels are never accessed.
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
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as activation_checkpoint

ROOT = Path('/kaggle/working/CS781_Capstone/phase3')
INPUT = Path('/kaggle/input')
ARTIFACT = Path('/kaggle/working/CS781_Phase3_All_Artifacts.zip')
REPO = 'Exploration-Lab/CS-781-Capstone'
SEED = 781
DATA_FILES = ('CL_train.json', 'CL_dev.json', 'IT_train.json', 'IT_dev.json',
              'label_vocab.json', 'CL_test.json', 'IT_test.json')
BASE_FILES = {f'{i:02d}': f'phase1_submission{i:02d}_recovery.zip' for i in range(1, 5)}
SOURCES = {'01': 'decoder_lm.py', '02': 'encoder_decoder_lm.py',
           '03': 'prefix_hybrid_lm.py', '04': 'efficient_causal_lm.py'}
EXPECTED_LABELS = {'Fact', 'Issue', 'ArgumentPetitioner', 'ArgumentRespondent',
                   'Statute', 'Dissent', 'PrecedentReliedUpon',
                   'PrecedentNotReliedUpon', 'PrecedentOverruled',
                   'RulingByLowerCourt', 'RatioOfTheDecision',
                   'RulingByPresentCourt', 'None'}
PLAN = [
    {'slot': 1, 'base': '03', 'method': 'lora', 'rank': 24, 'unfreeze': 12},
    {'slot': 2, 'base': '01', 'method': 'rslora', 'rank': 32, 'unfreeze': 4},
    {'slot': 3, 'base': '02', 'method': 'ia3', 'rank': 0, 'unfreeze': 3},
    {'slot': 4, 'base': '04', 'method': 'lorma', 'rank': 24, 'unfreeze': 7},
    {'slot': 5, 'base': '03', 'method': 'prompt', 'rank': 0, 'unfreeze': 12},
    {'slot': 6, 'base': '03', 'method': 'distillation', 'rank': 16, 'unfreeze': 12},
]
MAX_EPOCHS = 32
MIN_EPOCHS = 10
PATIENCE = 7
MAX_SLOT_SECONDS = 10 * 3600
MAX_TOKENS = 384
DOC_CHUNK = 8
DOC_STRIDE = 6
GRAD_ACCUM = 4
SNAPSHOT_EVERY = 2
SESSION_SECONDS = 8.5 * 3600


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=path.parent,
                                     delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def find_one(name, required=True):
    hits = sorted({p.resolve() for p in INPUT.rglob(name) if p.is_file()})
    if len(hits) > 1 or (required and len(hits) != 1):
        raise RuntimeError(f'Expected exactly one {name} under /kaggle/input; found {hits}')
    return hits[0] if hits else None


def safe_name(name):
    p = Path(name)
    return name and not p.is_absolute() and '..' not in p.parts and '\\' not in name


def restore_artifact():
    if (ROOT / 'run_config.json').is_file():
        return  # A running Kaggle session may be newer than its attached ZIP.
    source = find_one(ARTIFACT.name, required=False)
    if source is None:
        return
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if 'integrity.json' not in names or len(names) != len(set(names)):
            raise RuntimeError('Recovery archive lacks a unique integrity manifest')
        manifest = json.loads(z.read('integrity.json'))
        recorded = manifest['files']
        if set(recorded) != set(names) - {'integrity.json'}:
            raise RuntimeError('Recovery archive manifest mismatch')
        for name in names:
            if name == 'integrity.json':
                continue
            if not safe_name(name) or z.getinfo(name).is_dir():
                raise RuntimeError(f'Unsafe recovery member: {name}')
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
                tmp = Path(out.name)
                h = hashlib.sha256()
                for b in iter(lambda: src.read(1 << 20), b''):
                    h.update(b); out.write(b)
            if h.hexdigest() != recorded[name]['sha256'] or tmp.stat().st_size != recorded[name]['bytes']:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f'Corrupt recovery member: {name}')
            if dest.exists():
                if sha(dest) != h.hexdigest():
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f'Recovery conflicts with existing file: {name}')
                tmp.unlink()
            else:
                os.replace(tmp, dest)
    print('Restored self-contained Phase 3 state:', source)


def snapshot():
    """One ZIP contains data, source, staged Phase 1 weights, all model states and outputs."""
    ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and 'hf_cache' not in p.parts)
    manifest = {'version': 1, 'files': {}}
    tmp = ARTIFACT.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', allowZip64=True) as z:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            if rel.endswith('.tmp'):
                continue
            manifest['files'][rel] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
            z.write(path, rel, compress_type=zipfile.ZIP_STORED if path.suffix == '.pt' else zipfile.ZIP_DEFLATED)
        z.writestr('integrity.json', json.dumps(manifest, sort_keys=True))
    os.replace(tmp, ARTIFACT)
    print(f'Updated {ARTIFACT} ({ARTIFACT.stat().st_size / 2**30:.2f} GiB)')


def stage_base(base_id):
    dest = ROOT / 'bases' / base_id
    needed = [f'model/{SOURCES[base_id]}', 'model/course_bpe.py',
              'checkpoints/selected_model.pt']
    if all((dest / n).is_file() for n in needed) and (dest / 'tokenizer/tokenizer.json').is_file():
        return dest
    source = find_one(BASE_FILES[base_id])
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(not safe_name(n) for n in names):
            raise RuntimeError(f'Unsafe base archive: {source}')
        wanted = needed + [n for n in names if n.startswith('tokenizer/') and not n.endswith('/')]
        if not all(n in names for n in needed) or 'tokenizer/tokenizer.json' not in wanted:
            raise RuntimeError(f'Incomplete Phase 1 archive: {source}')
        for n in wanted:
            target = dest / n
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                tmp = Path(out.name)
                shutil.copyfileobj(src, out)
            os.replace(tmp, target)
    write_json(dest / 'provenance.json', {'archive_name': source.name, 'archive_sha256': sha(source),
                                           'source_sha256': sha(dest / needed[0]),
                                           'checkpoint_sha256': sha(dest / needed[2])})
    return dest


def stage_data():
    dest = ROOT / 'official'
    if all((dest / n).is_file() for n in DATA_FILES):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
        if not token:
            raise ValueError('empty token')
    except Exception as e:
        raise RuntimeError('Enable Kaggle Internet and the HF_TOKEN Kaggle secret, or attach a complete recovery artifact') from e
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=0.24,<2'])
        from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = api.repo_info(REPO, repo_type='dataset', token=token).sha
    paths = api.list_repo_files(REPO, repo_type='dataset', revision=revision, token=token)
    selected = {}
    for name in DATA_FILES:
        hits = [p for p in paths if p.split('/')[-1] == name and 'Phase_3_RR/' in p]
        if len(hits) != 1:
            raise RuntimeError(f'Expected one official Phase 3 {name}; found {hits}')
        selected[name] = hits[0]
    dest.mkdir(parents=True, exist_ok=True)
    for name, remote in selected.items():
        cached = Path(hf_hub_download(repo_id=REPO, repo_type='dataset', filename=remote,
                                      revision=revision, token=token, cache_dir=str(ROOT / 'hf_cache')))
        shutil.copyfile(cached, dest / name)
    write_json(dest / 'provenance.json', {'repo': REPO, 'revision': revision,
                                         'files': {n: {'path': selected[n], 'sha256': sha(dest / n)}
                                                   for n in DATA_FILES}})
    del token


def labels():
    obj = read_json(ROOT / 'official/label_vocab.json')
    if isinstance(obj, list):
        out = obj
    elif isinstance(obj, dict):
        if 'labels' in obj:
            out = obj['labels']
        elif 'label2id' in obj or 'label_to_id' in obj:
            mapping = obj.get('label2id', obj.get('label_to_id'))
            out = [k for k, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]
        elif 'id2label' in obj or 'id_to_label' in obj:
            mapping = obj.get('id2label', obj.get('id_to_label'))
            out = [v for _, v in sorted(mapping.items(), key=lambda kv: int(kv[0]))]
        elif all(isinstance(v, int) for v in obj.values()):
            out = [k for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]
        elif all(k.isdigit() for k in obj):
            out = [v for _, v in sorted(obj.items(), key=lambda kv: int(kv[0]))]
        else:
            raise ValueError('Unknown label_vocab.json schema')
    else:
        raise ValueError('Unknown label_vocab.json schema')
    if isinstance(out, dict):
        out = [k for k, _ in sorted(out.items(), key=lambda kv: kv[1])]
    if len(out) != 13 or set(out) != EXPECTED_LABELS:
        raise ValueError(f'Unexpected official label vocabulary: {out}')
    return out


def read_docs(name, labelled, domain):
    obj = read_json(ROOT / 'official' / name)
    if isinstance(obj, dict):
        nested = next((obj[k] for k in ('data', 'documents', 'examples')
                       if k in obj and isinstance(obj[k], list)), None)
        if nested is not None:
            rows = nested
        else:
            rows = []
            for key, value in obj.items():
                if isinstance(value, list):
                    rows.append({'id': key, 'text': value})
                elif isinstance(value, dict):
                    rows.append({'id': key, **value})
                else:
                    raise ValueError(f'Unknown document map in {name}')
    elif isinstance(obj, list):
        rows = obj
    else:
        raise ValueError(f'Unknown document format in {name}')
    out = []
    allowed = set(labels())
    for row in rows:
        did = row.get('id')
        text = row.get('text')
        gold = row.get('labels')
        if not isinstance(did, str) or not isinstance(text, list) or any(not isinstance(s, str) for s in text):
            raise ValueError(f'Invalid id/text in {name}: {did}')
        if labelled and (not isinstance(gold, list) or len(gold) != len(text) or any(y not in allowed for y in gold)):
            raise ValueError(f'Invalid labels in {name}: {did}')
        out.append({'id': did, 'text': text, 'labels': gold if labelled else None, 'domain': domain})
    ids = [r['id'] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate document ids in {name}')
    return out


def all_data():
    train = read_docs('CL_train.json', True, 'CL') + read_docs('IT_train.json', True, 'IT')
    dev = read_docs('CL_dev.json', True, 'CL') + read_docs('IT_dev.json', True, 'IT')
    test = read_docs('CL_test.json', False, 'CL') + read_docs('IT_test.json', False, 'IT')
    groups = [set(r['id'] for r in rows) for rows in (train, dev, test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Official document IDs overlap across train/dev/test')
    return train, dev, test


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base(base_id):
    directory = ROOT / 'bases' / base_id
    model_path = directory / 'model' / SOURCES[base_id]
    model_module = source_module(model_path, f'phase3_model_{base_id}_{os.getpid()}')
    bpe_module = source_module(directory / 'model/course_bpe.py', f'phase3_bpe_{base_id}_{os.getpid()}')
    tokenizer = bpe_module.CourseBPETokenizer.from_pretrained(directory / 'tokenizer')
    try:
        ck = torch.load(directory / 'checkpoints/selected_model.pt', map_location='cpu', weights_only=True)
    except TypeError:
        ck = torch.load(directory / 'checkpoints/selected_model.pt', map_location='cpu')
    source_hash = ck.get('model_source_sha256')
    if source_hash and source_hash != sha(model_path):
        raise RuntimeError(f'Phase 1 {base_id} source hash mismatch')
    meta = read_json(directory / 'tokenizer/tokenizer_metadata.json')
    if ck.get('tokenizer_fingerprint') != meta.get('fingerprint'):
        raise RuntimeError(f'Phase 1 {base_id} tokenizer mismatch')
    if int(ck['model_config']['vocab_size']) != len(tokenizer):
        raise RuntimeError(f'Phase 1 {base_id} vocabulary mismatch')
    base = model_module.build_model(ck['model_config'])
    base.load_state_dict(ck['model_state'], strict=True)
    return base, tokenizer


class LoRA(nn.Module):
    def __init__(self, base, rank, alpha, stabilized=False):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / (math.sqrt(rank) if stabilized else rank)

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class IA3(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.scale


class LoRMA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.out_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        y = self.base(x)
        return y + self.b(self.a(F.linear(x, self.base.weight))) * self.scale


def replace_adapters(base, spec):
    method, rank = spec['method'], spec['rank']
    if method in {'prompt'}:
        return 0
    if method == 'distillation':
        method = 'lora'
    count = 0
    for name, module in list(base.named_modules()):
        if not isinstance(module, nn.Linear) or not name or name.startswith('lm_head'):
            continue
        leaf = name.rsplit('.', 1)[-1]
        if method in {'lora', 'rslora'}:
            eligible = leaf in {'query', 'value', 'qkv', 'q_proj', 'v_proj', 'gate', 'up', 'down'}
        elif method == 'ia3':
            eligible = leaf in {'value', 'output', 'input'} and ('attention' in name or 'feed_forward' in name)
        elif method == 'lorma':
            eligible = leaf in {'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj'}
        else:
            eligible = False
        if not eligible:
            continue
        parent = base.get_submodule(name.rsplit('.', 1)[0])
        if method in {'lora', 'rslora'}:
            wrapped = LoRA(module, rank, 16 if method == 'rslora' else rank * 2,
                           method == 'rslora')
        elif method == 'ia3':
            wrapped = IA3(module)
        else:
            wrapped = LoRMA(module, rank, rank * 2)
        setattr(parent, leaf, wrapped)
        count += 1
    if count == 0:
        raise RuntimeError(f'No eligible {method} projections found on Phase 1 model {spec["base"]}')
    return count


def unfreeze_top(base, spec):
    base_id, n = spec['base'], spec['unfreeze']
    if base_id == '02':
        stacks = [base.encoder_blocks, base.decoder_blocks]
        for stack in stacks:
            for block in stack[-n:]:
                for p in block.parameters(): p.requires_grad_(True)
        for norm_name in ('encoder_final_norm', 'decoder_final_norm'):
            for p in getattr(base, norm_name).parameters(): p.requires_grad_(True)
    else:
        for block in base.blocks[-n:]:
            for p in block.parameters(): p.requires_grad_(True)
        for p in base.final_norm.parameters(): p.requires_grad_(True)


class RRModel(nn.Module):
    def __init__(self, base, spec, n_labels):
        super().__init__()
        self.base = base
        self.spec = spec
        for p in base.parameters(): p.requires_grad_(False)
        self.adapter_count = replace_adapters(base, spec)
        unfreeze_top(base, spec)
        width = int(base.hidden_size if hasattr(base, 'hidden_size') else
                    base.config.hidden_size if hasattr(base.config, 'hidden_size') else
                    base.config['hidden_size'])
        self.prompt = nn.Parameter(torch.randn(16, width) * .02) if spec['method'] == 'prompt' else None
        self.pool = nn.Linear(width * 2, width)
        self.position = nn.Embedding(16, 32)
        self.domain = nn.Embedding(2, 32)
        context_width = width + 64
        self.context = nn.GRU(context_width, 256, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=.15)
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(.2),
                                  nn.Linear(512, 384), nn.GELU(), nn.Dropout(.15),
                                  nn.Linear(384, n_labels))
        if spec['base'] == '02':
            base.gradient_checkpointing = True

    def features(self, ids, mask):
        base, bid = self.base, self.spec['base']
        if bid == '02':
            enc, enc_mask = base.encode(ids, mask.bool())
            query = ids[:, :1]
            dec = base._embed(query, base.decoder_position_embedding, base.max_decoder_length)
            for block in base.decoder_blocks:
                if self.training:
                    dec = activation_checkpoint(block, dec, enc, enc_mask, use_reentrant=False)
                else:
                    dec = block(dec, enc, enc_mask)
            hidden = base.decoder_final_norm(dec)[:, 0]
            mean = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            return torch.tanh(self.pool(torch.cat([hidden, mean], -1)))
        if bid == '01':
            length = ids.size(1)
            pos = torch.arange(length, device=ids.device)
            x = base.embedding_dropout(base.token_embedding(ids) + base.position_embedding(pos)[None])
            for block in base.blocks:
                x = activation_checkpoint(block, x, use_reentrant=False) if self.training else block(x)
            x = base.final_norm(x)
        elif bid == '03':
            x = base.embedding_dropout(base.token_embedding(ids) * math.sqrt(base.hidden_size))
            if self.prompt is not None:
                x = torch.cat([self.prompt[None].expand(ids.size(0), -1, -1), x], 1)
                mask = torch.cat([mask.new_ones(ids.size(0), self.prompt.size(0)), mask], 1)
            length = x.size(1)
            visibility = base._prefix_hybrid_visibility(mask.bool(), mask.sum(1))
            cosine, sine = base.rotary(length, x.dtype)
            for block in base.blocks:
                x = activation_checkpoint(block, x, visibility, cosine, sine,
                                          use_reentrant=False) if self.training else block(x, visibility, cosine, sine)
            x = base.final_norm(x)
        elif bid == '04':
            x = base.embedding(ids)
            for block in base.blocks:
                x = activation_checkpoint(block, x, mask.bool(), use_reentrant=False) if self.training else block(x, mask.bool())
            x = base.final_norm(x)
        else:
            raise ValueError(bid)
        end = mask.sum(1).long().clamp_min(1) - 1
        last = x[torch.arange(x.size(0), device=x.device), end]
        mean = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return torch.tanh(self.pool(torch.cat([last, mean], -1)))

    def forward(self, ids, mask, pos, domain):
        emb = self.features(ids, mask)
        p = self.position(pos.clamp(0, 15))
        d = self.domain(domain)
        states = torch.cat([emb, p, d], -1)[None]
        states, _ = self.context(states)
        return self.head(states[0])


TOKEN_CACHE = {}


def encode_sentence(tokenizer, doc, index):
    key = (id(tokenizer.backend), doc['id'], index)
    if key in TOKEN_CACHE:
        return TOKEN_CACHE[key]
    sentences = doc['text']
    before = ' '.join(sentences[max(0, index - 2):index])
    after = ' '.join(sentences[index + 1:index + 3])
    current = sentences[index]
    parts = ['Earlier context: ', before, '\nLater context: ', after,
             '\nTarget sentence: ', current]
    pieces = [tokenizer.backend.encode(p, add_special_tokens=False).ids for p in parts]
    # Keep the full target when possible; cut other context first.
    target = pieces[-1][-MAX_TOKENS + 16:]
    prefix = [int(x) for group in pieces[:-1] for x in group]
    ids = (prefix[-max(0, MAX_TOKENS - len(target)):] + target)[-MAX_TOKENS:]
    if not ids:
        ids = [0]
    TOKEN_CACHE[key] = ids
    return ids


def chunks(docs):
    out = []
    for doc in docs:
        n = len(doc['text'])
        if not n:
            continue
        starts = list(range(0, n, DOC_STRIDE))
        for start in starts:
            out.append((doc, list(range(start, min(start + DOC_CHUNK, n)))))
            if start + DOC_CHUNK >= n:
                break
    return out


def batch_for(item, tokenizer, label_to_id, device, teacher=None):
    doc, indexes = item
    seqs = [encode_sentence(tokenizer, doc, i) for i in indexes]
    width = max(len(s) for s in seqs)
    pad = int(tokenizer.pad_token_id)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
    labels_tensor = None
    if doc['labels'] is not None:
        labels_tensor = torch.tensor([label_to_id[doc['labels'][i]] for i in indexes], dtype=torch.long)
    domain = 0 if doc['domain'] == 'CL' else 1
    pos = [min(15, int(16 * i / max(1, len(doc['text'])))) for i in indexes]
    soft = None
    if teacher is not None:
        soft = torch.tensor([teacher[doc['id']][i] for i in indexes], dtype=torch.float32)
    return (ids.to(device), mask.to(device), torch.tensor(pos, device=device),
            torch.full((len(indexes),), domain, device=device, dtype=torch.long),
            labels_tensor.to(device) if labels_tensor is not None else None,
            soft.to(device) if soft is not None else None)


def prf(pred, truth, vocab):
    counts = {y: {'tp': 0, 'pred': 0, 'gold': 0} for y in vocab}
    for did in truth:
        for p, g in zip(pred[did], truth[did]):
            counts[p]['pred'] += 1
            counts[g]['gold'] += 1
            if p == g: counts[g]['tp'] += 1
    out = {}
    for y, c in counts.items():
        precision = c['tp'] / c['pred'] if c['pred'] else 0.0
        recall = c['tp'] / c['gold'] if c['gold'] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[y] = {'precision': precision, 'recall': recall, 'f1': f1, **c}
    return {'macro_f1': sum(v['f1'] for v in out.values()) / len(vocab), 'per_label': out}


@torch.no_grad()
def probabilities(model, docs, tokenizer, vocab, device):
    model.eval()
    mapping = {doc['id']: torch.zeros((len(doc['text']), len(vocab)), dtype=torch.float32)
               for doc in docs}
    counts = {doc['id']: torch.zeros(len(doc['text']), dtype=torch.float32) for doc in docs}
    ids = {y: i for i, y in enumerate(vocab)}
    for item in chunks(docs):
        doc, ix = item
        batch = batch_for(item, tokenizer, ids, device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(*batch[:4]).float().cpu()
        if not torch.isfinite(logits).all():
            raise RuntimeError(f'Non-finite inference logits for document {doc["id"]}')
        for row, sentence in enumerate(ix):
            mapping[doc['id']][sentence] += logits[row]
            counts[doc['id']][sentence] += 1
    for did in mapping:
        if (counts[did] == 0).any(): raise RuntimeError(f'Uncovered sentence in {did}')
        mapping[did] = F.softmax(mapping[did] / counts[did][:, None], -1)
    return mapping


def evaluate(model, docs, tokenizer, vocab, device):
    scores = probabilities(model, docs, tokenizer, vocab, device)
    predicted = {did: [vocab[int(i)] for i in scores[did].argmax(-1)] for did in scores}
    gold = {doc['id']: doc['labels'] for doc in docs}
    total = prf(predicted, gold, vocab)
    domains = {domain: prf({d['id']: predicted[d['id']] for d in docs if d['domain'] == domain},
                           {d['id']: d['labels'] for d in docs if d['domain'] == domain}, vocab)
               for domain in ('CL', 'IT')}
    return {'overall': total, 'domains': domains}


def init_dist():
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 2 or torch.cuda.device_count() < 2:
        raise RuntimeError('Training and inference require the Kaggle two-GPU accelerator')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=__import__('datetime').timedelta(minutes=30))
    return rank, world, torch.device('cuda', int(os.environ['LOCAL_RANK']))


def make_model(spec, device):
    base, tokenizer = load_base(spec['base'])
    model = RRModel(base, spec, len(labels())).to(device)
    return model, tokenizer


def best_path(slot):
    return ROOT / f'slot_{slot:02d}' / 'best.pt'


def checkpoint_path(slot):
    return ROOT / f'slot_{slot:02d}' / 'latest.pt'


def slot_status(slot):
    path = ROOT / f'slot_{slot:02d}' / 'status.json'
    return read_json(path) if path.exists() else {}


def load_selected(slot, device):
    spec = PLAN[slot - 1]
    model, tok = make_model(spec, device)
    ck = torch.load(best_path(slot), map_location='cpu', weights_only=False)
    if ck['spec'] != spec:
        raise RuntimeError(f'Slot {slot} checkpoint does not match plan')
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return model, tok


def teacher_data(train, dev, device):
    path = ROOT / 'teacher_soft.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    candidates = [s for s in range(1, 6) if slot_status(s).get('complete')]
    if not candidates:
        raise RuntimeError('Distillation needs a completed stronger Phase 1 teacher')
    ordered = sorted(candidates, key=lambda s: slot_status(s)['best_macro_f1'], reverse=True)
    selected = ordered[:1]
    if len(ordered) > 1:
        dev_probs = []
        for candidate in ordered[:2]:
            teacher, tok = load_selected(candidate, device)
            dev_probs.append(probabilities(teacher, dev, tok, labels(), device))
            del teacher
            torch.cuda.empty_cache()
        ensemble = {did: (dev_probs[0][did] + dev_probs[1][did]) / 2 for did in dev_probs[0]}
        vocab = labels()
        guessed = {did: [vocab[int(i)] for i in value.argmax(-1)] for did, value in ensemble.items()}
        truth = {d['id']: d['labels'] for d in dev}
        ensemble_f1 = prf(guessed, truth, vocab)['macro_f1']
        if ensemble_f1 > slot_status(ordered[0])['best_macro_f1']:
            selected = ordered[:2]
    probs = None
    for teacher_slot in selected:
        teacher, tok = load_selected(teacher_slot, device)
        part = probabilities(teacher, train, tok, labels(), device)
        probs = part if probs is None else {did: probs[did] + part[did] for did in probs}
        del teacher
        torch.cuda.empty_cache()
    probs = {did: value / len(selected) for did, value in probs.items()}
    result = {did: value.tolist() for did, value in probs.items()}
    saved = {'teacher_slots': selected, 'probabilities': result}
    save_torch(path, saved)
    return saved


def train(slot):
    rank, world, device = init_dist()
    spec = PLAN[slot - 1]
    train_docs, dev_docs, _ = all_data()
    vocab = labels(); to_id = {y: i for i, y in enumerate(vocab)}
    teacher = None
    if slot == 6:
        if rank == 0: teacher_data(train_docs, dev_docs, device)
        dist.barrier()
        teacher = torch.load(ROOT / 'teacher_soft.pt', map_location='cpu', weights_only=False)['probabilities']
    model, tok = make_model(spec, device)
    ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    groups = [{'params': [], 'lr': 3e-5}, {'params': [], 'lr': 1.5e-4}]
    for name, p in model.named_parameters():
        if p.requires_grad:
            adapter_parameter = any(t in name for t in ('.a.', '.b.', '.gate.')) or name.endswith('.scale')
            groups[0 if name.startswith('base.') and not adapter_parameter else 1]['params'].append(p)
    groups = [g for g in groups if g['params']]
    try:
        optimizer = torch.optim.AdamW(groups, weight_decay=.01, fused=True)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    folder = ROOT / f'slot_{slot:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    counts = Counter(y for d in train_docs for y in d['labels'])
    n = sum(counts.values())
    weights = torch.tensor([min(3.0, max(.5, math.sqrt(n / (len(vocab) * max(1, counts[y])))))
                            for y in vocab], device=device)
    state = {'epoch': 0, 'best': -1.0, 'best_epoch': 0,
             'patience': 0, 'train_seconds': 0.0}
    if checkpoint_path(slot).exists():
        ck = torch.load(checkpoint_path(slot), map_location='cpu', weights_only=False)
        if ck['spec'] != spec:
            raise RuntimeError('Resume checkpoint plan mismatch')
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        state.update(ck['state'])
    items = chunks(train_docs)
    if not items: raise ValueError('No training sentence chunks')
    copies = math.ceil(len(items) / (world * GRAD_ACCUM)) * world * GRAD_ACCUM
    steps_per_epoch = copies // (world * GRAD_ACCUM)
    total_steps = MAX_EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(.05 * total_steps))
    def lr_multiplier(step):
        if step < warmup_steps:
            return max(.05, (step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    if checkpoint_path(slot).exists():
        ck = torch.load(checkpoint_path(slot), map_location='cpu', weights_only=False)
        if 'scheduler' in ck:
            scheduler.load_state_dict(ck['scheduler'])
    started = time.monotonic()
    session_deadline = float(os.environ.get('PHASE3_SESSION_DEADLINE', time.time() + SESSION_SECONDS))
    for epoch in range(state['epoch'], MAX_EPOCHS):
        random.Random(SEED + slot * 100 + epoch).shuffle(items)
        # Equal number of synchronized DDP steps, with deterministic repeated tail.
        padded = (items * math.ceil(copies / len(items)))[:copies]
        local = padded[rank::world]
        ddp.train(); optimizer.zero_grad(set_to_none=True)
        running = 0.0
        gpu_csv = []
        for ix, item in enumerate(local):
            batch = batch_for(item, tok, to_id, device, teacher)
            sync = (ix + 1) % GRAD_ACCUM == 0
            with nullcontext() if sync else ddp.no_sync():
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = ddp(*batch[:4])
                    loss = F.cross_entropy(logits.float(), batch[4], weight=weights,
                                           label_smoothing=.03)
                    if batch[5] is not None:
                        temperature = 2.0
                        kd = F.kl_div(F.log_softmax(logits.float() / temperature, -1),
                                      batch[5], reduction='batchmean') * temperature**2
                        loss = .65 * loss + .35 * kd
                    loss = loss / GRAD_ACCUM
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite training loss in slot {slot}, epoch {epoch+1}; '
                                       'saved checkpoint remains available for diagnosis')
                scaler.scale(loss).backward()
            running += float(loss.detach())
            if sync:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if rank == 0 and ix == len(local) // 2:
                try:
                    gpu_csv = subprocess.check_output(['nvidia-smi',
                        '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                        text=True, timeout=10).strip().splitlines()
                except Exception:
                    pass
        dist.barrier()
        if rank == 0:
            stats = evaluate(model, dev_docs, tok, vocab, device)
            score = stats['overall']['macro_f1']
            if not math.isfinite(score):
                raise RuntimeError(f'Non-finite development macro-F1 in slot {slot}')
            improved = score > state['best'] + 1e-5
            state['patience'] = 0 if improved else state['patience'] + 1
            if improved:
                state['best'] = score
                state['best_epoch'] = epoch + 1
                save_torch(best_path(slot), {'spec': spec, 'model': model.state_dict(),
                                             'epoch': epoch + 1, 'macro_f1': score})
            state['epoch'] = epoch + 1
            state['train_seconds'] += time.monotonic() - started
            started = time.monotonic()
            save_torch(checkpoint_path(slot), {'spec': spec, 'model': model.state_dict(),
                                                'optimizer': optimizer.state_dict(),
                                                'scaler': scaler.state_dict(),
                                                'scheduler': scheduler.state_dict(), 'state': state})
            row = {'epoch': epoch + 1, 'train_loss_scaled_sum': running,
                   'dev_macro_f1': score, 'best_macro_f1': state['best'],
                   'cl_macro_f1': stats['domains']['CL']['macro_f1'],
                   'it_macro_f1': stats['domains']['IT']['macro_f1'],
                   'per_label': {y: v['f1'] for y, v in stats['overall']['per_label'].items()},
                   'elapsed_train_seconds': state['train_seconds'],
                   'gpu_utilization_memory_snapshot': gpu_csv}
            with (folder / 'history.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(row) + '\n')
            write_json(folder / 'status.json', {'complete': False, 'slot': slot, 'epoch': epoch + 1,
                                                 'best_macro_f1': state['best'],
                                                 'best_epoch': state['best_epoch'],
                                                 'train_seconds': state['train_seconds'],
                                                 'latest_dev': stats})
            print(f'Slot {slot} epoch {epoch+1}: macro-F1={score:.4f}, best={state["best"]:.4f}, '
                  f'CL={row["cl_macro_f1"]:.4f}, IT={row["it_macro_f1"]:.4f}; '
                  f'GPU utilization%,memoryMiB={gpu_csv}', flush=True)
            if (epoch + 1) % SNAPSHOT_EVERY == 0:
                snapshot()
            stop = ((epoch + 1 >= MIN_EPOCHS and state['patience'] >= PATIENCE) or
                    state['train_seconds'] >= MAX_SLOT_SECONDS or epoch + 1 == MAX_EPOCHS)
            pause = time.time() >= session_deadline - 900
            write_json(folder / 'control.json', {'stop': stop, 'pause': pause})
        dist.barrier()
        control = read_json(folder / 'control.json')
        if control['stop'] or control['pause']:
            break
    if rank == 0:
        status = slot_status(slot)
        if control['stop']:
            status['complete'] = True
            checkpoint_path(slot).unlink(missing_ok=True)
        write_json(folder / 'status.json', status)
        snapshot()
    dist.barrier(); dist.destroy_process_group()


def predict(slot):
    rank, world, device = init_dist()
    if not slot_status(slot).get('complete'):
        raise RuntimeError(f'Slot {slot} not complete')
    _, _, test = all_data()
    model, tok = load_selected(slot, device)
    vocab = labels()
    selected = test[rank::world]
    probs = probabilities(model, selected, tok, vocab, device)
    local = {did: [vocab[int(i)] for i in p.argmax(-1)] for did, p in probs.items()}
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        result = {k: v for part in gathered for k, v in part.items()}
        expected = {d['id']: len(d['text']) for d in test}
        if set(result) != set(expected): raise RuntimeError('Test document ID mismatch')
        if any(len(result[k]) != n for k, n in expected.items()):
            raise RuntimeError('Test sentence count mismatch')
        if any(y not in vocab for values in result.values() for y in values):
            raise RuntimeError('Invalid predicted label')
        folder = ROOT / f'slot_{slot:02d}'
        write_json(folder / 'prediction.json', result)
        package = folder / 'submission.zip'
        with zipfile.ZipFile(package.with_suffix('.zip.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(folder / 'prediction.json', 'prediction.json')
        os.replace(package.with_suffix('.zip.tmp'), package)
        with zipfile.ZipFile(package) as z:
            if z.namelist() != ['prediction.json'] or json.loads(z.read('prediction.json')) != result:
                raise RuntimeError('Submission ZIP verification failed')
        shutil.copyfile(package, Path('/kaggle/working') / f'CS781_Phase3_Submission_{slot:02d}.zip')
        write_json(folder / 'package.json', {'slot': slot, 'status': 'READY_FOR_MANUAL_UPLOAD',
                                              'zip_sha256': sha(package),
                                              'document_count': len(result),
                                              'sentence_count': sum(map(len, result.values()))})
        snapshot()
    dist.barrier(); dist.destroy_process_group()


def prepare():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Select the Kaggle two-GPU accelerator (two T4 GPUs)')
    ROOT.mkdir(parents=True, exist_ok=True)
    restore_artifact()
    for base_id in BASE_FILES:
        stage_base(base_id)
    stage_data()
    train_docs, dev_docs, test_docs = all_data()
    config = {'plan': PLAN, 'max_epochs': MAX_EPOCHS,
              'max_slot_seconds': MAX_SLOT_SECONDS, 'max_tokens': MAX_TOKENS,
              'doc_chunk': DOC_CHUNK, 'doc_stride': DOC_STRIDE,
              'grad_accum': GRAD_ACCUM, 'seed': SEED,
              'runner_sha256': sha(__file__)}
    config_path = ROOT / 'run_config.json'
    if config_path.exists() and read_json(config_path) != config:
        raise RuntimeError('Notebook code/configuration changed from the recovery artifact; refusing unsafe resume')
    write_json(config_path, config)
    write_json(ROOT / 'data_audit.json', {'train_documents': len(train_docs),
                                          'dev_documents': len(dev_docs),
                                          'test_documents': len(test_docs),
                                          'train_labels': dict(Counter(y for d in train_docs for y in d['labels'])),
                                          'dev_labels': dict(Counter(y for d in dev_docs for y in d['labels'])),
                                          'test_sentences': sum(len(d['text']) for d in test_docs),
                                          'labels': labels()})
    shutil.copyfile(__file__, ROOT / 'runner.py')
    print('Phase 3 prepared:', ROOT)
    print(read_json(ROOT / 'data_audit.json'))
    snapshot()


def finish():
    statuses = [slot_status(i) for i in range(1, 7)]
    packages = [ROOT / f'slot_{i:02d}' / 'submission.zip' for i in range(1, 7)]
    for i, package in enumerate(packages, 1):
        if package.is_file():
            shutil.copyfile(package, Path('/kaggle/working') / f'CS781_Phase3_Submission_{i:02d}.zip')
    if not all(s.get('complete') for s in statuses) or not all(p.is_file() for p in packages):
        print('Incomplete run. Reattach the all-artifacts ZIP and Run All again.')
        snapshot(); return
    hashes = []
    for slot in range(1, 7):
        pred = read_json(ROOT / f'slot_{slot:02d}' / 'prediction.json')
        hashes.append(hashlib.sha256(json.dumps(pred, sort_keys=True).encode()).hexdigest())
    if len(set(hashes)) < 6:
        raise RuntimeError('Two experiments produced identical predictions; six distinct submissions are required')
    write_json(ROOT / 'six_submission_report.json',
               {'status': 'READY_FOR_MANUAL_UPLOAD', 'slots': [
                   {'slot': i, 'method': PLAN[i-1]['method'], 'base': PLAN[i-1]['base'],
                    'best_dev_macro_f1': statuses[i-1]['best_macro_f1'],
                    'package_sha256': sha(packages[i-1])} for i in range(1, 7)]})
    snapshot()
    print('All six submissions ready. Scores above are local dev macro-F1, not Codabench scores.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'predict', 'finish', 'snapshot'])
    parser.add_argument('--slot', type=int, choices=range(1, 7))
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'train': train(args.slot)
    elif args.action == 'predict': predict(args.slot)
    elif args.action == 'finish': finish()
    else: snapshot()
