"""Phase 3 context-aware joint encoder/document experiment.

This file and hierarchical_runner.py are embedded in the Run All notebook.
Only the supplied Phase 1 checkpoint and official Phase 3 data are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

import cs781_phase3_hierarchical_runner as u

WORK = Path('/kaggle/working')
ROOT = WORK / 'CS781_Capstone/phase3_joint'
ARTIFACT = WORK / 'CS781_Phase3_Context_Joint_All_Artifacts.zip'
SUBMISSION = WORK / 'CS781_Phase3_Context_Joint_Submission.zip'
u.ROOT = ROOT
u.ARTIFACT = ARTIFACT
u.OLD_ARCHIVES = ('CS781_Phase3_All_Artifacts.zip',
                  'CS781_Phase3_Hierarchical_All_Artifacts.zip',
                  'CS781_Phase3_One_All_Artifacts.zip')

SEED = 781230
MAX_TOKENS = 448
TARGET_LIMIT = 256
BASE_WINDOW, BASE_STRIDE = 8, 6
JOINT_WINDOW, JOINT_STRIDE = 16, 12
ACCUM = 2
MAX_HOURS = 9.45
ARCHIVE_INTERVAL = 38 * 60
SAVE_STEPS = 100
ROLE_GROUPS = {
    'Fact': 0, 'Issue': 1, 'ArgumentPetitioner': 2,
    'ArgumentRespondent': 2, 'Statute': 3,
    'PrecedentReliedUpon': 3, 'PrecedentNotReliedUpon': 3,
    'PrecedentOverruled': 3, 'RatioOfTheDecision': 4,
    'RulingByLowerCourt': 4, 'RulingByPresentCourt': 4,
    'Dissent': 4, 'None': 5,
}


def log(*args):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *args, flush=True)


def setting():
    return {'version': 1, 'seed': SEED, 'max_tokens': MAX_TOKENS,
            'target_limit': TARGET_LIMIT, 'base_window': BASE_WINDOW,
            'joint_window': JOINT_WINDOW, 'accum': ACCUM,
            'phase1_base': '03', 'prior_reference': 'slot_06',
            'runner_sha256': u.digest(__file__)}


def stable_seed(value):
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    u.restore()
    old, train, dev, test, vocab = u.sets()
    if set(vocab) != set(ROLE_GROUPS) or len(vocab) != 13:
        raise RuntimeError('Unexpected Phase 3 label vocabulary')
    config_path = ROOT / 'config.json'
    config = setting()
    config['fold_map'] = u.fold_map(train)
    if config_path.exists():
        existing = u.read_json(config_path)
        if existing != config:
            raise RuntimeError('Recovery configuration differs from this notebook')
    else:
        u.atomic_json(config_path, config)
    shutil.copy2(Path(u.__file__), ROOT / 'recovery_utilities.py')
    shutil.copy2(Path(__file__), ROOT / 'context_joint_runner.py')
    token_path = ROOT / 'tokens.pt'
    baseline_path = ROOT / 'baseline_tokens.pt'
    tokenizer = None
    if not token_path.exists() or not baseline_path.exists():
        _, tokenizer = old.load_base('03')
    if not token_path.exists():
        tokens = make_tokens(train + dev + test, tokenizer)
        u.atomic_torch(token_path, tokens)
        lengths = [len(row[0]) for rows in tokens.values() for row in rows]
        repeated = {}
        for d in train:
            for text, label in zip(d['text'], d['labels']):
                repeated.setdefault(' '.join(text.lower().split()), set()).add(label)
        labelled_docs = {label: sum(label in d['labels'] for d in train) for label in vocab}
        u.atomic_json(ROOT / 'input_audit.json', {
            'train_docs': len(train), 'dev_docs': len(dev), 'test_docs': len(test),
            'train_sentences': sum(len(d['text']) for d in train),
            'dev_sentences': sum(len(d['text']) for d in dev),
            'test_sentences': sum(len(d['text']) for d in test),
            'token_length_percentiles': np.percentile(lengths, [50, 90, 99, 100]).tolist(),
            'training_label_counts': dict(Counter(y for d in train for y in d['labels'])),
            'training_documents_per_label': labelled_docs,
            'sentence_strings_with_conflicting_training_roles':
                sum(len(roles) > 1 for roles in repeated.values()),
            'vocabulary': vocab,
        })
    if not baseline_path.exists():
        baseline = {}
        for doc in train + dev + test:
            baseline[doc['id']] = []
            for i in range(len(doc['text'])):
                ids = old.encode_sentence(tokenizer, doc, i)
                baseline[doc['id']].append((ids, max(0, len(ids) - 1), len(ids)))
        u.atomic_torch(baseline_path, baseline)
    log('Prepared official documents and target-aware token cache:', token_path)
    if not ARTIFACT.exists():
        u.snapshot()


def make_tokens(docs, tokenizer):
    encode = lambda s: tokenizer.backend.encode(s, add_special_tokens=False).ids
    marker = encode('\n[TARGET SENTENCE] ')
    separator = encode(' [SENTENCE] ')
    if not marker:
        raise RuntimeError('Tokenizer produced no target marker tokens')
    out = {}
    for doc in docs:
        texts = doc['text']
        full = [encode(t) for t in texts]
        rows = []
        for i, target in enumerate(full):
            if len(target) > TARGET_LIMIT:
                front = TARGET_LIMIT * 3 // 4
                target = target[:front] + target[-(TARGET_LIMIT - front):]
            if not target:
                target = [int(tokenizer.pad_token_id)]
            earlier = full[max(0, i-2):i]
            later = full[i+1:i+3]
            left = encode(' [EARLIER CONTEXT] ')
            for segment in earlier:
                left += separator + segment
            right = encode(' [LATER CONTEXT] ')
            for segment in later:
                right += separator + segment
            budget = MAX_TOKENS - len(marker) - len(target)
            left_budget = min(len(left), budget // 2)
            right_budget = min(len(right), budget - left_budget)
            left_budget = min(len(left), budget - right_budget)
            left_part = left[-left_budget:] if left_budget else []
            right_part = right[:right_budget]
            start = len(left_part) + len(marker)
            ids = left_part + marker + target + right_part
            if not 0 < start < start + len(target) <= len(ids) <= MAX_TOKENS:
                raise RuntimeError('Bad target span encoding')
            rows.append((ids, start, start + len(target)))
        out[doc['id']] = rows
    return out


def windows(doc, size, stride):
    n = len(doc['text'])
    if n == 0:
        raise RuntimeError('Empty official document')
    start = 0
    while True:
        end = min(n, start + size)
        yield (doc['id'], start, end)
        if end == n:
            break
        start += stride


def batch_window(item, docs_by_id, tokens, pad, label_to_id, device):
    did, start, end = item
    doc = docs_by_id[did]
    rows = tokens[did][start:end]
    width = max(len(r[0]) for r in rows)
    n = len(rows)
    ids = torch.full((n, width), pad, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.bool)
    target = torch.zeros((n, width), dtype=torch.bool)
    for j, (seq, a, b) in enumerate(rows):
        ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[j, :len(seq)] = True
        target[j, a:b] = True
    y = torch.tensor([label_to_id[x] for x in doc['labels'][start:end]], dtype=torch.long) if doc['labels'] else None
    pos = torch.tensor([min(31, int(32 * i / max(1, len(doc['text']))))
                        for i in range(start, end)], dtype=torch.long)
    def gpu(t):
        return t.pin_memory().to(device, non_blocking=True) if device.type == 'cuda' else t.to(device)
    return (gpu(ids), gpu(mask), gpu(target), gpu(pos),
            0 if doc['domain'] == 'CL' else 1, gpu(y) if y is not None else None)


class ContextJoint(nn.Module):
    def __init__(self, base, n_labels, mode):
        super().__init__()
        if mode not in {'baseline', 'joint'}:
            raise ValueError(mode)
        self.base, self.mode = base, mode
        self.base.output_bias.requires_grad_(False)
        width = base.hidden_size
        self.attention = nn.Linear(width, 1) if mode == 'joint' else None
        self.project = nn.Sequential(nn.Linear(width * (3 if mode == 'joint' else 2), 512),
                                     nn.LayerNorm(512), nn.GELU(), nn.Dropout(.16))
        self.position = nn.Embedding(32, 512)
        self.domain = nn.Embedding(2, 512)
        self.gru = nn.GRU(512, 256, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=.15)
        self.gru_norm = nn.LayerNorm(512)
        self.document_attention = nn.TransformerEncoderLayer(
            512, 8, dim_feedforward=1024, dropout=.12,
            activation='gelu', batch_first=True, norm_first=True) if mode == 'joint' else None
        self.document_norm = nn.LayerNorm(512) if mode == 'joint' else None
        self.cl_head = nn.Sequential(nn.Dropout(.2), nn.Linear(512, n_labels))
        self.it_head = nn.Sequential(nn.Dropout(.2), nn.Linear(512, n_labels))
        if mode == 'joint':
            self.coarse = nn.Linear(512, 6)
            self.argument = nn.Linear(512, 2)
            self.boundary = nn.Linear(1024, 1)

    def forward(self, ids, mask, target, pos, domain):
        base = self.base
        x = base.embedding_dropout(base.token_embedding(ids) * math.sqrt(base.hidden_size))
        visible = base._prefix_hybrid_visibility(mask, mask.sum(1))
        cosine, sine = base.rotary(ids.size(1), x.dtype)
        for block in base.blocks:
            x = checkpoint(block, x, visible, cosine, sine, use_reentrant=False) if self.training else block(x, visible, cosine, sine)
        x = base.final_norm(x)
        if self.mode == 'joint':
            att = self.attention(x).squeeze(-1).masked_fill(~target, -1e4).softmax(-1)
            pooled = (x * att.unsqueeze(-1)).sum(1)
            mean = (x * target.unsqueeze(-1)).sum(1) / target.sum(1, keepdim=True).clamp_min(1)
            maximum = x.masked_fill(~target.unsqueeze(-1), -1e4).max(1).values
            vectors = torch.cat((pooled, mean, maximum), dim=-1)
        else:
            last = x[torch.arange(x.size(0), device=x.device), mask.sum(1).long() - 1]
            mean = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            vectors = torch.cat((last, mean), dim=-1)
        h = self.project(vectors)
        h = h + self.position(pos) + self.domain(torch.tensor(domain, device=ids.device))
        recurrent, _ = self.gru(h[None])
        h = self.gru_norm(h + recurrent[0])
        if self.mode == 'joint':
            h = self.document_norm(h + self.document_attention(h[None])[0])
        fine = self.cl_head(h) if domain == 0 else self.it_head(h)
        if self.mode == 'baseline':
            return fine, None, None, None
        coarse = self.coarse(h)
        argument = self.argument(h)
        boundaries = self.boundary(torch.cat((h[:-1], h[1:]), -1)).squeeze(-1)
        return fine, coarse, argument, boundaries


def loss_weights(train, vocab, device):
    counts = Counter(y for d in train for y in d['labels'])
    typical = sum(counts.values()) / len(vocab)
    weights = [min(2.5, max(.65, math.sqrt(typical / max(1, counts[y])))) for y in vocab]
    scale = sum(weights) / len(weights)
    return torch.tensor([v / scale for v in weights], dtype=torch.float32, device=device)


def objective(output, gold, vocab, weights, mode):
    fine, coarse, arg, boundary = output
    loss = F.cross_entropy(fine.float(), gold, weight=weights)
    if mode == 'baseline':
        return loss
    groups = torch.tensor([ROLE_GROUPS[x] for x in vocab], device=gold.device)
    loss = loss + .16 * F.cross_entropy(coarse.float(), groups[gold])
    ids = {x: vocab.index(x) for x in ('ArgumentPetitioner', 'ArgumentRespondent')}
    is_arg = (gold == ids['ArgumentPetitioner']) | (gold == ids['ArgumentRespondent'])
    if is_arg.any():
        arg_gold = (gold[is_arg] == ids['ArgumentRespondent']).long()
        loss = loss + .12 * F.cross_entropy(arg[is_arg].float(), arg_gold)
    if gold.numel() > 1:
        changed = (gold[:-1] != gold[1:]).float()
        loss = loss + .08 * F.binary_cross_entropy_with_logits(boundary.float(), changed)
    return loss


def optimizer_for(model, total_updates):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('base.blocks.'):
            layer = int(name.split('.')[2])
            lr = 5e-6 + (2.0e-5 - 5e-6) * layer / max(1, len(model.base.blocks) - 1)
        elif name.startswith('base.'):
            lr = 5e-6
        else:
            lr = 1.2e-4
        decay = 0. if name.endswith('bias') or 'norm' in name.lower() else .02
        key = (lr, decay)
        groups.setdefault(key, []).append(param)
    params = [{'params': ps, 'lr': lr, 'weight_decay': wd} for (lr, wd), ps in groups.items()]
    try:
        opt = torch.optim.AdamW(params, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(params)
    warmup = max(10, int(.06 * total_updates))
    def schedule(step):
        if step < warmup:
            return max(.05, (step + 1) / warmup)
        fraction = min(1., (step - warmup) / max(1, total_updates - warmup))
        return .08 + .92 * .5 * (1 + math.cos(math.pi * fraction))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def sample_gpus(stage, epoch, step):
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
        rows = []
        for line in result.stdout.strip().splitlines():
            values = [x.strip() for x in line.split(',')]
            rows.append({'index': int(values[0]), 'utilization_percent': int(values[1]),
                         'memory_mib': int(values[2]), 'total_mib': int(values[3]),
                         'power_watts': float(values[4])})
    except Exception as exc:
        rows = [{'sampling_error': str(exc)}]
    sample = {'timestamp': time.time(), 'stage': stage, 'epoch': epoch,
              'optimizer_step': step, 'gpus': rows}
    with (ROOT / 'gpu_samples.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(sample) + '\n')


@torch.no_grad()
def evaluate(ddp, docs, docs_by_id, tokens, pad, vocab, mode, rank, device):
    ddp.eval()
    label_to_id = {x: i for i, x in enumerate(vocab)}
    size, stride = (BASE_WINDOW, BASE_STRIDE) if mode == 'baseline' else (JOINT_WINDOW, JOINT_STRIDE)
    local = {}
    for doc in docs[rank::2]:
        n = len(doc['text'])
        summed = torch.zeros((n, len(vocab)), dtype=torch.float32)
        counts = torch.zeros(n, dtype=torch.float32)
        for item in windows(doc, size, stride):
            ids, mask, target, pos, domain, _ = batch_window(
                item, docs_by_id, tokens, pad, label_to_id, device)
            with u.autocast():
                logits = ddp.module(ids, mask, target, pos, domain)[0]
            probability = logits.float().softmax(-1).cpu()
            summed[item[1]:item[2]] += probability
            counts[item[1]:item[2]] += 1
        if not torch.all(counts > 0):
            raise RuntimeError('Uncovered sentence in evaluation')
        local[doc['id']] = summed / counts[:, None]
    received = [None, None]
    dist.all_gather_object(received, local)
    merged = {did: p for part in received for did, p in part.items()}
    if set(merged) != {d['id'] for d in docs}:
        raise RuntimeError('Distributed evaluation missing a document')
    return merged


def report_predictions(prob, docs, vocab):
    report = u.metrics(prob, docs, vocab)
    pred = {did: [vocab[int(i)] for i in p.argmax(-1)] for did, p in prob.items()}
    report['predicted_label_counts'] = dict(Counter(y for labels in pred.values() for y in labels))
    report['gold_label_counts'] = dict(Counter(y for d in docs for y in d['labels']))
    confusions = Counter((gold, guess) for d in docs for gold, guess in zip(d['labels'], pred[d['id']]))
    report['confusion'] = {gold: {guess: confusions[(gold, guess)] for guess in vocab}
                           for gold in vocab}
    selected = ['ArgumentRespondent', 'PrecedentNotReliedUpon',
                'RulingByLowerCourt', 'RatioOfTheDecision']
    report['focus'] = {name: report['per_label'][name] for name in selected}
    return report


def stage_path(stage):
    return ROOT / stage


def stage_spec(stage, train, config):
    mode = 'baseline' if stage == 'pilot_baseline' else 'joint'
    fold = 0 if stage in {'pilot_baseline', 'pilot_joint'} else (
        1 if stage == 'confirm_1' else 2 if stage == 'confirm_2' else None)
    fitting = train if fold is None else [d for d in train if config['fold_map'][d['id']] != fold]
    heldout = [] if fold is None else [d for d in train if config['fold_map'][d['id']] == fold]
    if fold is None and stage != 'final':
        raise ValueError(stage)
    if fold is not None and (len(fitting) != 64 or len(heldout) != 16):
        raise RuntimeError('Fold split is not 64/16 documents')
    if stage == 'pilot_baseline':
        return mode, fitting, heldout, 10, 5, 3, 1.4
    if stage == 'pilot_joint':
        return mode, fitting, heldout, 22, 8, 5, 2.5
    pilot = u.read_json(stage_path('pilot_joint') / 'status.json')
    fixed = max(8, min(20, int(pilot['best_epoch'])))
    if stage == 'final':
        fixed = min(24, fixed + 2)
        return mode, fitting, heldout, fixed, fixed, 999, 2.0
    return mode, fitting, heldout, fixed, fixed, 999, 1.55


def run_stage(stage, deadline):
    rank, device = u.init_dist()
    try:
        stable_seed(SEED + rank + sum(ord(c) for c in stage))
        old, train, dev, _, vocab = u.sets()
        config = u.read_json(ROOT / 'config.json')
        mode, fitting, heldout, max_epochs, min_epochs, patience, cap_hours = stage_spec(stage, train, config)
        folder = stage_path(stage)
        folder.mkdir(parents=True, exist_ok=True)
        status_path = folder / 'status.json'
        if status_path.exists() and u.read_json(status_path).get('complete'):
            log(stage, 'already complete')
            return
        token_file = 'baseline_tokens.pt' if mode == 'baseline' else 'tokens.pt'
        tokens = torch.load(ROOT / token_file, map_location='cpu', weights_only=False)
        docs_by_id = {d['id']: d for d in train + dev}
        base, tokenizer = old.load_base('03')
        model = ContextJoint(base, len(vocab), mode).to(device)
        ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True,
                  broadcast_buffers=False, gradient_as_bucket_view=True)
        pad = int(tokenizer.pad_token_id)
        size, stride = (BASE_WINDOW, BASE_STRIDE) if mode == 'baseline' else (JOINT_WINDOW, JOINT_STRIDE)
        items = [item for doc in fitting for item in windows(doc, size, stride)]
        updates_per_epoch = math.ceil(len(items) / (2 * ACCUM))
        optimizer, scheduler = optimizer_for(model, updates_per_epoch * max_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        weights = loss_weights(fitting, vocab, device)
        label_to_id = {x: i for i, x in enumerate(vocab)}
        latest = folder / 'latest.pt'
        best = folder / 'best.pt'
        state = {'epoch': 1, 'next_batch': 0, 'step': 0, 'best_f1': -1.,
                 'best_epoch': 0, 'bad_epochs': 0, 'complete': False}
        if latest.exists():
            ck = torch.load(latest, map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model'], strict=True)
            optimizer.load_state_dict(ck['optimizer'])
            scheduler.load_state_dict(ck['scheduler'])
            scaler.load_state_dict(ck['scaler'])
            state.update(ck['state'])
            log(stage, 'resumed epoch', state['epoch'], 'batch', state['next_batch'])
        local_end = min(deadline - 22 * 60, time.time() + cap_hours * 3600)
        last_archive = time.time()

        def save_checkpoint():
            nonlocal last_archive
            dist.barrier()
            if rank == 0:
                u.atomic_torch(latest, {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'state': state})
                if time.time() - last_archive > ARCHIVE_INTERVAL:
                    u.snapshot()
                    last_archive = time.time()
            dist.barrier()

        last_epoch = int(state['epoch']) - 1
        for epoch in range(int(state['epoch']), max_epochs + 1):
            rng = random.Random(SEED * 7 + epoch)
            order = list(items)
            rng.shuffle(order)
            while len(order) % (2 * ACCUM):
                order.append(order[len(order) % len(items)])
            assigned = order[rank::2]
            ddp.train()
            optimizer.zero_grad(set_to_none=True)
            start_batch = int(state['next_batch']) if epoch == state['epoch'] else 0
            epoch_start = time.time()
            loss_total, token_count = 0., 0
            for batch_i in range(start_batch, len(assigned)):
                ids, mask, target, pos, domain, gold = batch_window(
                    assigned[batch_i], docs_by_id, tokens, pad, label_to_id, device)
                sync = ((batch_i + 1) % ACCUM == 0)
                with (nullcontext() if sync else ddp.no_sync()):
                    with u.autocast():
                        outputs = ddp(ids, mask, target, pos, domain)
                        loss = objective(outputs, gold, vocab, weights, mode) / ACCUM
                    u.finite_or_fail(loss)
                    scaler.scale(loss).backward()
                loss_total += float(loss.detach()) * ACCUM
                token_count += int(mask.sum())
                if sync:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    state['step'] += 1
                    state['next_batch'] = batch_i + 1
                    if state['step'] % 25 == 0 and rank == 0:
                        sample_gpus(stage, epoch, state['step'])
                    if state['step'] % SAVE_STEPS == 0:
                        save_checkpoint()
            dist.barrier()
            dev_prob = evaluate(ddp, dev, docs_by_id, tokens, pad, vocab, mode, rank, device)
            dev_report = report_predictions(dev_prob, dev, vocab)
            score = float(dev_report['macro_supported'])
            if rank == 0:
                u.atomic_json(folder / f'dev_epoch_{epoch:02d}.json', dev_report)
                if score > state['best_f1'] + 1e-5:
                    state['best_f1'] = score
                    state['best_epoch'] = epoch
                    state['bad_epochs'] = 0
                    u.atomic_torch(best, {'model': model.state_dict(), 'epoch': epoch,
                                          'dev_report': dev_report, 'mode': mode})
                else:
                    state['bad_epochs'] += 1
                entry = {'epoch': epoch, 'optimizer_steps': state['step'],
                         'train_loss_rank0': loss_total / max(1, len(assigned) - start_batch),
                         'rank0_tokens_per_second': token_count / max(1, time.time() - epoch_start),
                         'epoch_seconds': time.time() - epoch_start,
                         'dev_macro_supported': score,
                         'dev_macro_all13': dev_report['macro_all13'],
                         'dev_domain_all13': dev_report['domain_all13'],
                         'dev_focus_f1': {name: value['f1']
                                          for name, value in dev_report['focus'].items()},
                         'gpu_allocated_mib_rank0': torch.cuda.memory_allocated(device) / 2**20,
                         'gpu_reserved_mib_rank0': torch.cuda.memory_reserved(device) / 2**20}
                with (folder / 'history.jsonl').open('a', encoding='utf8') as f:
                    f.write(json.dumps(entry) + '\n')
                log(stage, json.dumps(entry))
                sample_gpus(stage, epoch, state['step'])
            state_list = [state if rank == 0 else None]
            dist.broadcast_object_list(state_list, src=0)
            state = state_list[0]
            state['epoch'] = epoch + 1
            state['next_batch'] = 0
            save_checkpoint()
            if rank == 0 and time.time() - last_archive > ARCHIVE_INTERVAL:
                u.snapshot()
                last_archive = time.time()
            dist.barrier()
            early = epoch >= min_epochs and state['bad_epochs'] >= patience
            required_before_time_stop = min_epochs if stage.startswith('pilot_') else 3
            cap = time.time() >= local_end and epoch >= required_before_time_stop
            if early or cap:
                if rank == 0:
                    log(stage, 'stopping:', 'patience' if early else 'time cap')
                last_epoch = epoch
                break
            last_epoch = epoch

        dist.barrier()
        chosen = best if stage.startswith('pilot_') else latest
        ck = torch.load(chosen, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=True)
        if heldout:
            outer = evaluate(ddp, heldout, docs_by_id, tokens, pad, vocab, mode, rank, device)
            outer_report = report_predictions(outer, heldout, vocab)
        else:
            outer, outer_report = None, None
        dev_best = evaluate(ddp, dev, docs_by_id, tokens, pad, vocab, mode, rank, device)
        if rank == 0:
            if outer is not None:
                u.atomic_torch(folder / 'outer_prob.pt', outer)
                u.atomic_json(folder / 'outer_metrics.json', outer_report)
            u.atomic_torch(folder / 'dev_prob.pt', dev_best)
            u.atomic_json(folder / 'selected_dev_metrics.json',
                          report_predictions(dev_best, dev, vocab))
            u.atomic_torch(folder / 'selected.pt', {'model': model.state_dict(),
                           'epoch': last_epoch, 'mode': mode})
            state['complete'] = True
            state['selected_checkpoint'] = 'selected.pt'
            state['time_limited'] = bool(time.time() >= local_end and last_epoch < max_epochs)
            state['trained_epochs'] = last_epoch
            state['outer_supported'] = outer_report['macro_supported'] if outer_report else None
            u.atomic_json(status_path, state)
            latest.unlink(missing_ok=True)
            best.unlink(missing_ok=True)
            u.snapshot()
            log(stage, 'complete; outer supported F1', state['outer_supported'])
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def gate():
    _, train, dev, _, vocab = u.sets()
    baseline_status = u.read_json(stage_path('pilot_baseline') / 'status.json')
    joint_status = u.read_json(stage_path('pilot_joint') / 'status.json')
    base = u.read_json(stage_path('pilot_baseline') / 'outer_metrics.json')
    joint = u.read_json(stage_path('pilot_joint') / 'outer_metrics.json')
    gain = joint['macro_supported'] - base['macro_supported']
    per_domain = {d: joint['domain_all13'][d] - base['domain_all13'][d]
                  for d in ('CL', 'IT')}
    focus = {name: joint['per_label'][name]['f1'] - base['per_label'][name]['f1']
             for name in ('ArgumentRespondent', 'PrecedentNotReliedUpon',
                          'RulingByLowerCourt', 'RatioOfTheDecision')}
    enough_training = (baseline_status['trained_epochs'] >= 5 and
                       joint_status['trained_epochs'] >= 8)
    passed = enough_training and gain >= .03 and min(per_domain.values()) >= -.025 and (
        sum(1 for delta in focus.values() if delta > 0) >= 2)
    decision = {'pass': passed, 'required_supported_macro_gain': .03,
                'minimum_training_met': enough_training,
                'baseline_epochs': baseline_status['trained_epochs'],
                'joint_epochs': joint_status['trained_epochs'],
                'measured_gain': gain, 'per_domain_all13_delta': per_domain,
                'focus_label_f1_delta': focus, 'baseline_outer': base,
                'joint_outer': joint,
                'note': 'This pilot uses exactly the same 16 held-out training documents for both models.'}
    u.atomic_json(ROOT / 'pilot_gate.json', decision)
    log('Pilot gate', 'PASS' if passed else 'STOP', 'supported F1 gain', round(gain, 4))
    u.snapshot()
    return passed


def infer_stage(stage, split):
    rank, device = u.init_dist()
    try:
        stable_seed(SEED + rank)
        old, train, dev, test, vocab = u.sets()
        docs = dev if split == 'dev' else test
        token_file = 'baseline_tokens.pt' if mode == 'baseline' else 'tokens.pt'
        tokens = torch.load(ROOT / token_file, map_location='cpu', weights_only=False)
        base, tokenizer = old.load_base('03')
        mode = 'baseline' if stage == 'pilot_baseline' else 'joint'
        model = ContextJoint(base, len(vocab), mode).to(device)
        ck = torch.load(stage_path(stage) / 'selected.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=True)
        ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True,
                  broadcast_buffers=False)
        docs_by_id = {d['id']: d for d in train + dev + test}
        prob = evaluate(ddp, docs, docs_by_id, tokens, int(tokenizer.pad_token_id),
                        vocab, mode, rank, device)
        if rank == 0:
            u.atomic_torch(stage_path(stage) / f'{split}_prob.pt', prob)
            if split == 'dev':
                u.atomic_json(stage_path(stage) / 'selected_dev_metrics.json',
                              report_predictions(prob, docs, vocab))
            log(stage, split, 'inference complete')
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def combine(prob_a, prob_b, weight_a):
    if set(prob_a) != set(prob_b):
        raise RuntimeError('Probability document IDs differ')
    return {did: weight_a * prob_a[did] + (1 - weight_a) * prob_b[did]
            for did in prob_a}


def choose():
    _, train, dev, _, vocab = u.sets()
    prior = u.prior_prob('dev')
    prior_report = report_predictions(prior, dev, vocab)
    result = {'prior6_dev': prior_report, 'pilot_gate': u.read_json(ROOT / 'pilot_gate.json')}
    stages = ['pilot_joint', 'confirm_1', 'confirm_2']
    complete = all((stage_path(s) / 'status.json').is_file() and
                   u.read_json(stage_path(s) / 'status.json').get('complete') for s in stages)
    final_status_path = stage_path('final') / 'status.json'
    final_ready = (final_status_path.is_file() and
                   u.read_json(final_status_path).get('complete') and
                   u.read_json(final_status_path).get('trained_epochs', 0) >= 8)
    if not result['pilot_gate']['pass'] or not complete or not final_ready:
        result.update({'selected': 'prior6', 'reason': 'Pilot gate failed or full fit incomplete',
                       'private_score': None})
        u.atomic_json(ROOT / 'selection.json', result)
        return result

    # Pool the disjoint held-out fold predictions once. Folds 3-4 remain unused.
    mapping = u.read_json(ROOT / 'config.json')['fold_map']
    oof = {}
    for number, stage in enumerate(stages):
        part = torch.load(stage_path(stage) / 'outer_prob.pt', map_location='cpu', weights_only=False)
        expected = {d['id'] for d in train if mapping[d['id']] == number}
        if set(part) != expected or set(part) & set(oof):
            raise RuntimeError('Outer fold leakage or missing predictions')
        oof.update(part)
    covered = [d for d in train if d['id'] in oof]
    oof_report = report_predictions(oof, covered, vocab)
    result['joint_heldout_48docs'] = oof_report
    result['confirm_folds'] = {
        s: u.read_json(stage_path(s) / 'outer_metrics.json')['macro_supported']
        for s in stages}
    joint = torch.load(stage_path('final') / 'dev_prob.pt', map_location='cpu', weights_only=False)
    joint_report = report_predictions(joint, dev, vocab)
    blend = combine(joint, prior, .8)
    blend_report = report_predictions(blend, dev, vocab)
    result['joint_dev'] = joint_report
    result['fixed_80joint_20prior_dev'] = blend_report

    def domain_safe(candidate, margin):
        return all(candidate['domain_all13'][d] >= prior_report['domain_all13'][d] - margin
                   for d in ('CL', 'IT'))

    # Only two predeclared choices; the official dev set is too small for a grid search.
    if joint_report['macro_supported'] >= prior_report['macro_supported'] + .01 and domain_safe(joint_report, .03):
        selected, reason = 'joint', 'Joint final model beat the prior on dev and passed the domain guard'
    elif blend_report['macro_supported'] >= prior_report['macro_supported'] + .015 and domain_safe(blend_report, .025):
        selected, reason = 'fixed_blend', 'Predeclared blend beat the prior on dev and passed the domain guard'
    elif (oof_report['macro_supported'] >= .48 and
          joint_report['macro_supported'] >= prior_report['macro_supported'] - .005 and
          domain_safe(joint_report, .02)):
        selected, reason = 'joint', 'Held-out evidence passed and dev was approximately tied'
    else:
        selected, reason = 'prior6', 'New model did not clear the conservative dev guard'
    result.update({'selected': selected, 'reason': reason, 'private_score': None})
    u.atomic_json(ROOT / 'selection.json', result)
    u.snapshot()
    log('Selection', selected, reason)
    return result


def package():
    _, _, dev, test, vocab = u.sets()
    decision = u.read_json(ROOT / 'selection.json')
    selected = decision['selected']
    old_prob = u.prior_prob('test') if selected in {'prior6', 'fixed_blend'} else None
    if selected == 'prior6':
        prob = old_prob
    else:
        new_prob = torch.load(stage_path('final') / 'test_prob.pt', map_location='cpu', weights_only=False)
        prob = new_prob if selected == 'joint' else combine(new_prob, old_prob, .8)
    prediction = {d['id']: [vocab[int(i)] for i in prob[d['id']].argmax(-1)] for d in test}
    expected = {d['id']: len(d['text']) for d in test}
    if set(prediction) != set(expected) or any(len(prediction[did]) != n for did, n in expected.items()):
        raise RuntimeError('Submission document IDs or sentence counts are wrong')
    if sum(map(len, prediction.values())) != 2197:
        raise RuntimeError('Unexpected Phase 3 test sentence count')
    if any(y not in vocab for labels in prediction.values() for y in labels):
        raise RuntimeError('Submission has an unknown role label')
    u.atomic_json(ROOT / 'prediction.json', prediction)
    temp = SUBMISSION.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / 'prediction.json', 'prediction.json')
    os.replace(temp, SUBMISSION)
    with zipfile.ZipFile(SUBMISSION) as z:
        if z.namelist() != ['prediction.json'] or json.loads(z.read('prediction.json')) != prediction:
            raise RuntimeError('ZIP verification failed')
    shutil.copy2(SUBMISSION, WORK / 'submission.zip')
    shutil.copy2(SUBMISSION, ROOT / 'submission.zip')
    prior_prediction = {d['id']: [vocab[int(i)] for i in u.prior_prob('test')[d['id']].argmax(-1)]
                        for d in test}
    changes = sum(a != b for did in prediction for a, b in zip(prediction[did], prior_prediction[did]))
    report = {'status': 'READY_FOR_MANUAL_UPLOAD', 'selected': selected,
              'selection_reason': decision['reason'], 'submission': str(SUBMISSION),
              'submission_sha256': u.digest(SUBMISSION), 'test_documents': len(test),
              'test_sentences': sum(expected.values()), 'labels_changed_vs_slot06': changes,
              'prior_private_macro_f1': .5058, 'current_private_macro_f1': None,
              'target_private_macro_f1': .8,
              'note': 'The private 0.8 target cannot be verified or guaranteed before upload.'}
    u.atomic_json(ROOT / 'report.json', report)
    u.snapshot()
    log('Submission', SUBMISSION, 'Recovery', ARTIFACT, 'Changed labels', changes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'train', 'infer', 'gate',
                                           'choose', 'package', 'snapshot'))
    parser.add_argument('--stage', default='pilot_joint')
    parser.add_argument('--split', default='test', choices=('dev', 'test'))
    parser.add_argument('--deadline', type=float, default=time.time() + MAX_HOURS * 3600)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'train':
        run_stage(args.stage, args.deadline)
    elif args.action == 'infer':
        infer_stage(args.stage, args.split)
    elif args.action == 'gate':
        gate()
    elif args.action == 'choose':
        choose()
    elif args.action == 'package':
        package()
    elif args.action == 'snapshot':
        u.snapshot()


if __name__ == '__main__':
    main()


