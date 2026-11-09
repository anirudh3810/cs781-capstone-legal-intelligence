"""Phase 3 full-document section and source tracker.

Embedded verbatim in the Run All notebook. All learned language-model weights
originate in the user's Phase 1 checkpoint 03. The old slot 06 is audited only;
it is never substituted for this notebook's submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

import cs781_phase3_hierarchical_runner as u
import cs781_phase3_context_joint_runner as c

WORK = Path('/kaggle/working')
ROOT = WORK / 'CS781_Capstone/phase3_sections'
ARTIFACT = WORK / 'CS781_Phase3_Sections_All_Artifacts.zip'
SUBMISSION = WORK / 'CS781_Phase3_Sections_Submission.zip'
u.ROOT = ROOT
u.ARTIFACT = ARTIFACT
u.OLD_ARCHIVES = ('CS781_Phase3_All_Artifacts.zip',
                  'CS781_Phase3_Context_Joint_All_Artifacts.zip',
                  'CS781_Phase3_Hierarchical_All_Artifacts.zip')


def required_source(name):
    return name=='runner.py' or name.startswith('official/') or name.startswith('bases/03/')


def source_ready():
    needed=('runner.py','official/CL_train.json','official/IT_train.json',
            'official/CL_dev.json','official/IT_dev.json','official/CL_test.json',
            'official/IT_test.json','official/label_vocab.json',
            'bases/03/model/prefix_hybrid_lm.py','bases/03/model/course_bpe.py',
            'bases/03/checkpoints/selected_model.pt','bases/03/tokenizer/tokenizer.json',
            'bases/03/tokenizer/tokenizer_metadata.json')
    return all((u.PRIOR/name).is_file() for name in needed)


# Use the older runner solely as an archive/data utility. Its slot-06 fallback
# is never called and is not required to reconstruct this experiment.
u.required_prior=required_source
u.prior_ready=source_ready

SEED = 781362
FOLDS = 5
LOCAL_MAX_EPOCHS = 7
LOCAL_INNER_EPOCHS = 5
GLOBAL_EPOCHS = 35
TOKEN_BUDGET = 8192
MAX_SENT_BATCH = 24
ACCUM = 2
MAX_SPAN = 64
ARCHIVE_SECONDS = 35 * 60
START_TIME_FILE = ROOT / 'run_start.json'
FAILED_RUN_SOURCE_SHA256 = 'd3653fac38eefb223f07a8deb5e1b0fb835afefaa6ff519b72305dea21da9c05'
SNAPSHOT_HANG_SOURCE_SHA256 = '14dfb977c41041eb4d9cf39056c8cb7b0dd90116f0b8c77fb602f3b9b44e1361'


def log(*parts):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *parts, flush=True)


def sample_gpus(stage,epoch):
    try:
        query=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,
                             check=True,timeout=15)
        values=[[v.strip() for v in row.split(',')] for row in query.stdout.strip().splitlines()]
        gpu=[{'index':int(v[0]),'utilization_percent':int(v[1]),'memory_mib':int(v[2]),
              'total_mib':int(v[3]),'power_watts':float(v[4])} for v in values]
    except Exception as exc:gpu=[{'error':repr(exc)}]
    with (ROOT/'gpu_samples.jsonl').open('a',encoding='utf8') as out:
        out.write(json.dumps({'time':time.time(),'stage':stage,'epoch':epoch,'gpus':gpu})+'\n')


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32-1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairs(docs):
    return {d['id']: d for d in docs}


def balanced_assignment(docs):
    groups=[[],[]];loads=[0,0]
    for doc in sorted(docs,key=lambda d:len(d['text']),reverse=True):
        rank=min(range(2),key=lambda i:loads[i])
        groups[rank].append(doc)
        loads[rank]+=len(doc['text'])
    return groups


def inner_split(docs, seed):
    mapping = {}
    for domain in ('CL', 'IT'):
        rows = [d for d in docs if d['domain'] == domain]
        rows.sort(key=lambda d: hashlib.sha256(f'{seed}:{d["id"]}'.encode()).hexdigest())
        for i, doc in enumerate(rows):
            mapping[doc['id']] = i % 2
    return mapping


FLAG_NAMES = (
    'heading', 'numbered_heading', 'argument_heading', 'analysis_heading',
    'fact_heading', 'dissent_heading', 'petitioner_word', 'respondent_word',
    'counsel', 'appeared_for', 'court_voice', 'lower_court', 'commission',
    'citation', 'quotation', 'statute', 'precedent', 'negative_precedent',
    'positive_precedent', 'overruled', 'submitted', 'held', 'therefore',
    'sentence_length', 'document_position', 'near_start', 'near_end', 'domain_it',
)


def lexical_flags(doc):
    rows = []
    n = len(doc['text'])
    for i, sentence in enumerate(doc['text']):
        t = sentence.lower().strip()
        short = len(t.split()) <= 12
        heading = short and (t.endswith(':') or sentence.strip().isupper() or
                             bool(re.search(r'\b(analysis|conclusion|arguments|submissions|facts|issues|order|dissent)\b', t)))
        numbered = bool(re.match(r'^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)\s]', t))
        values = [
            heading, numbered, short and bool(re.search(r'argument|submission', t)),
            short and bool(re.search(r'analysis|conclusion|reasoning', t)),
            short and bool(re.search(r'fact|background|history', t)),
            bool(re.search(r'\bdissent(?:ing)?\b', t)),
            bool(re.search(r'petitioners?|appellants?', t)),
            bool(re.search(r'respondents?|opposite part(?:y|ies)', t)),
            bool(re.search(r'counsel|advocate|solicitor', t)),
            'appeared for' in t or 'on behalf of' in t,
            bool(re.search(r'\bwe\b|\bour view\b|this court|we therefore', t)),
            bool(re.search(r'lower court|trial court|tribunal|below court', t)),
            bool(re.search(r'commission|\bcci\b|\bdg\b', t)),
            bool(re.search(r'\b\d{4}\b.*(?:scc|scr|air|plr|itr)|\bsupra\b|\bv\.\s', t)),
            '"' in sentence or "'" in sentence or 'extracted below' in t,
            bool(re.search(r'\bsection\b|\bact\b|\bstatute\b|\brule\b', t)),
            bool(re.search(r'precedent|judg(?:e)?ment|decision|case cited', t)),
            bool(re.search(r'not binding|distinguish(?:ed|able)|not applicable|inapplicable|no assistance|not relied|cannot be relied|rejected', t)),
            bool(re.search(r'relied upon|binding precedent|followed|squarely applies', t)),
            bool(re.search(r'overrul(?:ed|ing)', t)),
            bool(re.search(r'submitted|argued|contended|urged|pleaded', t)),
            bool(re.search(r'held|observed|concluded|ruled|ordered', t)),
            bool(re.search(r'therefore|accordingly|hence|for these reasons', t)),
            min(1., len(sentence.split()) / 100), i / max(1, n-1),
            i < 5, i >= n-5, doc['domain']=='IT',
        ]
        rows.append([float(v) for v in values])
    return torch.tensor(rows, dtype=torch.float32)


def prepare():
    if torch.cuda.device_count()!=2:
        raise RuntimeError('Enable exactly two Kaggle GPUs before Run All')
    ROOT.mkdir(parents=True, exist_ok=True)
    u.restore()
    old, train, dev, test, vocab = u.sets()
    if len(train) != 80 or len(dev) != 10 or len(test) != 10 or len(vocab) != 13:
        raise RuntimeError('Unexpected official Phase 3 data')
    conf = {'version': 1, 'seed': SEED, 'folds': u.fold_map(train),
            'local_max_epochs': LOCAL_MAX_EPOCHS, 'local_inner_epochs': LOCAL_INNER_EPOCHS,
            'max_span': MAX_SPAN, 'source_sha256': u.digest(__file__)}
    path = ROOT / 'config.json'
    if path.exists():
        previous=u.read_json(path)
        if previous!=conf:
            earlier=previous.get('source_sha256')
            expected=dict(conf,source_sha256=earlier)
            saved_source=ROOT/'section_span_runner.py'
            if (earlier not in {FAILED_RUN_SOURCE_SHA256,SNAPSHOT_HANG_SOURCE_SHA256}
                    or previous!=expected or not saved_source.is_file()
                    or u.digest(saved_source)!=earlier):
                raise RuntimeError('Recovery code/configuration mismatch')
            u.atomic_json(ROOT/'migration.json',{
                'from_source_sha256':earlier,
                'to_source_sha256':conf['source_sha256'],
                'reason':'Repair two-GPU auxiliary losses and move recovery snapshots outside DDP',
                'local_checkpoints_reused':True,'features_reused':True,
                'migrated_unix':time.time()})
            u.atomic_json(path,conf)
            log('Resuming saved models and features after runner repair')
    else:
        u.atomic_json(path, conf)
    shutil.copy2(Path(__file__), ROOT / 'section_span_runner.py')
    shutil.copy2(Path(u.__file__), ROOT / 'recovery_utilities.py')
    shutil.copy2(Path(c.__file__), ROOT / 'context_tokenizer.py')
    # Completed stages need only their selected weights, not large optimizer
    # states. Keep latest.pt for every unfinished stage so an interrupted
    # training stage can resume from its last epoch.
    for family in ('local','global'):
        for done in (ROOT/family).glob('*/done.json'):
            (done.parent/'latest.pt').unlink(missing_ok=True)
    if not START_TIME_FILE.exists():
        u.atomic_json(START_TIME_FILE, {'started_unix': time.time()})

    token_path = ROOT / 'tokens.pt'
    if not token_path.exists():
        live = WORK / 'CS781_Capstone/phase3_joint/tokens.pt'
        if live.is_file():
            shutil.copy2(live, token_path)
        else:
            prior_zip = u.locate('CS781_Phase3_Context_Joint_All_Artifacts.zip')
            if prior_zip:
                u.extract_manifest_zip(prior_zip, 'hier/', ROOT, lambda n: n == 'tokens.pt')
    if not token_path.exists():
        _, tokenizer = old.load_base('03')
        u.atomic_torch(token_path, c.make_tokens(train + dev + test, tokenizer))

    flags_path = ROOT / 'lexical_flags.pt'
    if not flags_path.exists():
        flags = {d['id']: lexical_flags(d) for d in train + dev + test}
        u.atomic_torch(flags_path, flags)
    runs = Counter()
    for d in train:
        y = d['labels']
        for i in range(len(y)):
            if i == 0 or y[i] != y[i-1]:
                runs[y[i]] += 1
    u.atomic_json(ROOT / 'audit.json', {
        'training_documents': len(train), 'development_documents': len(dev),
        'test_documents': len(test),
        'train_sentences': sum(len(d['text']) for d in train),
        'dev_sentences': sum(len(d['text']) for d in dev),
        'test_sentences': sum(len(d['text']) for d in test),
        'gold_segments_by_role': dict(runs),
        'feature_names': FLAG_NAMES,
        'source': 'Only official Phase 3 JSON and user Phase 1 base 03; no external weights.',
    })
    if not ARTIFACT.exists():
        u.snapshot()
    log('Prepared', ROOT)


def local_stage_docs(stage, train, config):
    if stage == 'full':
        return train, []
    if stage.startswith('final_inner_'):
        side = int(stage[-1])
        imap = inner_split(train, SEED + 900)
        return ([d for d in train if imap[d['id']] != side],
                [d for d in train if imap[d['id']] == side])
    kind, f, *tail = stage.split('_')
    fold = int(f)
    fitting = [d for d in train if config['folds'][d['id']] != fold]
    outer = [d for d in train if config['folds'][d['id']] == fold]
    if kind == 'outer':
        return fitting, outer
    if kind == 'inner':
        side = int(tail[0])
        imap = inner_split(fitting, SEED + fold)
        return ([d for d in fitting if imap[d['id']] != side],
                [d for d in fitting if imap[d['id']] == side])
    raise ValueError(stage)


class LocalEncoder(nn.Module):
    def __init__(self, base, classes=13):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.unfreeze_from = max(0, len(base.blocks)-8)
        for block in base.blocks[self.unfreeze_from:]:
            for p in block.parameters():p.requires_grad_(True)
        for p in base.final_norm.parameters():p.requires_grad_(True)
        width = base.hidden_size
        self.attention = nn.Linear(width, 1)
        self.project = nn.Sequential(nn.Linear(width*3, 512), nn.LayerNorm(512),
                                     nn.GELU(), nn.Dropout(.30))
        self.position = nn.Embedding(32, 512)
        self.domain = nn.Embedding(2, 512)
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Dropout(.30),
                                        nn.Linear(512, classes))

    def forward(self, ids, mask, target, pos, domain):
        base = self.base
        x = base.embedding_dropout(base.token_embedding(ids) * math.sqrt(base.hidden_size))
        visible = base._prefix_hybrid_visibility(mask, mask.sum(1))
        cosine, sine = base.rotary(ids.size(1), x.dtype)
        fixed=x
        for i, block in enumerate(base.blocks):
            if i < self.unfreeze_from:
                with torch.no_grad():
                    x = block(x, visible, cosine, sine)
                fixed=x
            else:
                x = checkpoint(block, x, visible, cosine, sine, use_reentrant=False) if self.training else block(x, visible, cosine, sine)
        x = base.final_norm(x)
        a = self.attention(x).squeeze(-1).masked_fill(~target, -1e4).softmax(-1)
        pooled = (x*a.unsqueeze(-1)).sum(1)
        mean = (x*target.unsqueeze(-1)).sum(1)/target.sum(1,keepdim=True).clamp_min(1)
        maximum = x.masked_fill(~target.unsqueeze(-1),-1e4).max(1).values
        h = self.project(torch.cat((pooled,mean,maximum),-1))
        h = h + self.position(pos) + self.domain(domain)
        # The learned projection differs across cross-fit encoders. Shared
        # Phase 1 hidden coordinates keep document-head features aligned.
        stable=(fixed*target.unsqueeze(-1)).sum(1)/target.sum(1,keepdim=True).clamp_min(1)
        feature=torch.cat((stable.detach(),mean),-1)
        return feature, self.classifier(h)


def trainable_state(model):
    names = {name for name,p in model.named_parameters() if p.requires_grad}
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()
            if key in names or key.startswith('project.') or key.startswith('classifier.')
            or key.startswith('attention.') or key.startswith('position.') or key.startswith('domain.')}


def load_trainable(model, state):
    unexpected = set(state) - set(model.state_dict())
    if unexpected:
        raise RuntimeError(f'Unknown adapter keys: {unexpected}')
    model.load_state_dict(state, strict=False)


def sample_rows(docs, epoch):
    rng = random.Random(SEED + epoch*1009)
    rows=[]
    for d in docs:
        n=len(d['text'])
        budget=min(n,max(48,round(6*math.sqrt(n))))
        anchors=[i for i,y in enumerate(d['labels']) if i==0 or y!=d['labels'][i-1]]
        rng.shuffle(anchors)
        chosen=set(anchors[:min(len(anchors),budget//3)])
        remaining=list(set(range(n))-chosen)
        rng.shuffle(remaining)
        chosen.update(remaining[:max(0,budget-len(chosen))])
        rows.extend((d['id'],i) for i in sorted(chosen))
    rng.shuffle(rows)
    return rows


def sentence_batches(rows, tokens, epoch):
    rng=random.Random(SEED+epoch*2039)
    pool=list(rows)
    rng.shuffle(pool)
    batches=[]
    for start in range(0,len(pool),384):
        group=pool[start:start+384]
        group.sort(key=lambda r:len(tokens[r[0]][r[1]][0]))
        j=0
        while j<len(group):
            width=len(tokens[group[j][0]][group[j][1]][0])
            size=min(MAX_SENT_BATCH,max(1,TOKEN_BUDGET//max(1,width)))
            batch=group[j:j+size]
            batches.append(batch)
            j+=size
    rng.shuffle(batches)
    while len(batches)%(2*ACCUM):
        batches.append(batches[len(batches)%max(1,len(batches))])
    return batches


def collate_sentences(rows, docs_by_id, tokens, pad, vocab, device):
    label_to_id={label:i for i,label in enumerate(vocab)}
    seqs=[tokens[did][i] for did,i in rows]
    width=max(len(seq[0]) for seq in seqs)
    n=len(rows)
    ids=torch.full((n,width),pad,dtype=torch.long)
    mask=torch.zeros((n,width),dtype=torch.bool)
    target=torch.zeros((n,width),dtype=torch.bool)
    pos=torch.zeros(n,dtype=torch.long)
    domain=torch.zeros(n,dtype=torch.long)
    labels=torch.full((n,),-1,dtype=torch.long)
    for j,((did,i),(seq,a,b)) in enumerate(zip(rows,seqs)):
        d=docs_by_id[did]
        ids[j,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        mask[j,:len(seq)]=True
        target[j,a:b]=True
        pos[j]=min(31,int(32*i/max(1,len(d['text']))))
        domain[j]=0 if d['domain']=='CL' else 1
        if d['labels'] is not None:
            labels[j]=label_to_id[d['labels'][i]]
    def move(x):return x.pin_memory().to(device,non_blocking=True) if device.type=='cuda' else x
    return tuple(move(x) for x in (ids,mask,target,pos,domain,labels))


def local_optimizer(model, total_steps):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('base.blocks.'):
            layer=int(name.split('.')[2]);lr=6e-6+(1.8e-5-6e-6)*(layer-model.unfreeze_from)/max(1,len(model.base.blocks)-1-model.unfreeze_from)
        elif name.startswith('base.'):
            lr=1.8e-5
        else:lr=1.5e-4
        decay=0. if name.endswith('bias') or 'norm' in name.lower() else .03
        groups.setdefault((lr,decay),[]).append(p)
    try:opt=torch.optim.AdamW([{'params':ps,'lr':lr,'weight_decay':wd} for (lr,wd),ps in groups.items()],fused=True)
    except (TypeError,RuntimeError):opt=torch.optim.AdamW([{'params':ps,'lr':lr,'weight_decay':wd} for (lr,wd),ps in groups.items()])
    warm=max(8,int(.08*total_steps))
    def factor(step):
        if step<warm:return max(.1,(step+1)/warm)
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total_steps-warm))))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,factor)


def local_model(old, device, vocab):
    base, tokenizer = old.load_base('03')
    model = LocalEncoder(base, len(vocab)).to(device)
    return model, tokenizer


@torch.no_grad()
def local_evaluate(model, docs, tokens, pad, vocab, rank, device):
    model.eval()
    own = {}
    for d in balanced_assignment(docs)[rank]:
        logits=[]
        for start in range(0,len(d['text']),MAX_SENT_BATCH):
            rows=[(d['id'],i) for i in range(start,min(len(d['text']),start+MAX_SENT_BATCH))]
            ids,mask,target,pos,domain,_=collate_sentences(rows,pairs([d]),tokens,pad,vocab,device)
            with torch.autocast('cuda',dtype=torch.float16):
                _,score=model(ids,mask,target,pos,domain)
            logits.append(score.float().cpu())
        own[d['id']]=torch.cat(logits).softmax(-1)
    result=u.gather_dict(own)
    if rank==0:
        return u.metrics(result,docs,vocab)
    return None


def train_local(stage):
    rank,device=u.init_dist()
    try:
        random_seed(SEED+sum(map(ord,stage))+rank)
        old,train,dev,_,vocab=u.sets()
        conf=u.read_json(ROOT/'config.json')
        fit,held=local_stage_docs(stage,train,conf)
        is_outer=stage.startswith('outer_')
        epochs=LOCAL_MAX_EPOCHS if is_outer or stage=='full' else LOCAL_INNER_EPOCHS
        validation=dev if is_outer or stage=='full' else []
        stage_dir=ROOT/'local'/stage
        stage_dir.mkdir(parents=True,exist_ok=True)
        if (stage_dir/'done.json').exists():
            if rank==0:
                latest=(stage_dir/'latest.pt')
                latest.unlink(missing_ok=True)
                log('Local stage already complete',stage)
            dist.barrier();return
        tokens=torch.load(ROOT/'tokens.pt',map_location='cpu',weights_only=False)
        model,tok=local_model(old,device,vocab)
        pad=int(tok.pad_token_id)
        count=len(sentence_batches(sample_rows(fit,1),tokens,1))//2//ACCUM
        opt,scheduler=local_optimizer(model,max(1,count*epochs))
        scaler=torch.amp.GradScaler('cuda')
        latest=stage_dir/'latest.pt'
        start=1;best=-1.;best_epoch=0
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            load_trainable(model,state['model']);opt.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            start=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch']
        ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,
                find_unused_parameters=False,static_graph=True)
        class_w=u.class_weights(fit,vocab,device)
        docs_by_id=pairs(fit)
        history=u.read_json(stage_dir/'history.json') if (stage_dir/'history.json').exists() else []
        for epoch in range(start,epochs+1):
            ddp.train();opt.zero_grad(set_to_none=True)
            batches=sentence_batches(sample_rows(fit,epoch),tokens,epoch)
            rank_batches=batches[rank::2]
            total_loss=0.
            for j,rows in enumerate(rank_batches):
                ids,mask,target,pos,domain,y=collate_sentences(rows,docs_by_id,tokens,pad,vocab,device)
                with torch.autocast('cuda',dtype=torch.float16):
                    _,score=ddp(ids,mask,target,pos,domain)
                    loss=F.cross_entropy(score.float(),y,weight=class_w,label_smoothing=.06)/ACCUM
                u.finite_or_fail(loss)
                scaler.scale(loss).backward()
                total_loss+=float(loss.detach())*ACCUM
                if (j+1)%ACCUM==0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(opt);scaler.update();scheduler.step()
                    opt.zero_grad(set_to_none=True)
            dist.barrier()
            evaluated=local_evaluate(ddp.module,validation,tokens,pad,vocab,rank,device) if validation else None
            if rank==0:
                metric=evaluated['macro_supported'] if evaluated else -total_loss/max(1,len(rank_batches))
                take=(metric>best) if validation else True
                if take:
                    best=metric;best_epoch=epoch
                    u.atomic_torch(stage_dir/'best.pt',{'model':trainable_state(model),'epoch':epoch,
                        'stage':stage,'fit_ids':[d['id'] for d in fit],'score':metric})
                history.append({'epoch':epoch,'train_loss_rank0':total_loss/max(1,len(rank_batches)),
                                'development':evaluated,'best_epoch':best_epoch})
                u.atomic_json(stage_dir/'history.json',history)
                u.atomic_torch(latest,{'model':trainable_state(model),'optimizer':opt.state_dict(),
                    'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch,'best':best,'best_epoch':best_epoch})
                log('Local',stage,'epoch',epoch,'loss',round(history[-1]['train_loss_rank0'],4),
                    'dev F1',None if not evaluated else round(metric,4))
                sample_gpus('local_'+stage,epoch)
            dist.barrier()
        if rank==0:
            u.atomic_json(stage_dir/'done.json',{'stage':stage,'selected_epoch':best_epoch,
                'fit_ids':[d['id'] for d in fit],'extraction_ids':[d['id'] for d in held],
                'validation':history[best_epoch-1]['development'] if validation else None})
            latest.unlink(missing_ok=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def extraction_docs(stage,train,dev,test,conf):
    _,held=local_stage_docs(stage,train,conf)
    if stage=='full':return dev+test
    if stage.startswith('outer_'):return held+dev
    return held


def extract_local(stage):
    rank,device=u.init_dist()
    try:
        old,train,dev,test,vocab=u.sets()
        conf=u.read_json(ROOT/'config.json')
        docs=extraction_docs(stage,train,dev,test,conf)
        stage_dir=ROOT/'local'/stage
        if not (stage_dir/'done.json').is_file():raise RuntimeError('Train local stage first: '+stage)
        feature_dir=ROOT/'features'/stage
        feature_dir.mkdir(parents=True,exist_ok=True)
        tokens=torch.load(ROOT/'tokens.pt',map_location='cpu',weights_only=False)
        model,tok=local_model(old,device,vocab)
        state=torch.load(stage_dir/'best.pt',map_location='cpu',weights_only=False)
        load_trainable(model,state['model']);model.eval()
        pad=int(tok.pad_token_id)
        for doc in balanced_assignment(docs)[rank]:
            destination=feature_dir/(doc['id']+'.pt')
            if destination.is_file():continue
            embeddings=[];logits=[]
            with torch.no_grad():
                for start in range(0,len(doc['text']),MAX_SENT_BATCH):
                    rows=[(doc['id'],i) for i in range(start,min(len(doc['text']),start+MAX_SENT_BATCH))]
                    ids,mask,target,pos,domain,_=collate_sentences(rows,pairs([doc]),tokens,pad,vocab,device)
                    with torch.autocast('cuda',dtype=torch.float16):
                        emb,score=model(ids,mask,target,pos,domain)
                    embeddings.append(emb.cpu().half());logits.append(score.cpu().half())
            data={'embedding':torch.cat(embeddings),'logits':torch.cat(logits),
                  'doc_id':doc['id'],'n':len(doc['text']),'stage':stage}
            u.atomic_torch(destination,data)
            log('Extracted',stage,doc['id'],len(doc['text']))
        dist.barrier()
        if rank==0:
            for doc in docs:
                p=feature_dir/(doc['id']+'.pt')
                if not p.is_file():raise RuntimeError('Missing features: '+str(p))
                data=torch.load(p,map_location='cpu',weights_only=False)
                if data['n']!=len(doc['text']) or data['embedding'].shape!=(len(doc['text']),768):
                    raise RuntimeError('Bad feature shape: '+doc['id'])
            u.atomic_json(feature_dir/'done.json',{'stage':stage,'docs':[d['id'] for d in docs]})
        dist.barrier()
    finally:dist.destroy_process_group()


def feature_path(stage, did):
    return ROOT/'features'/stage/(did+'.pt')


_STAGE_MAP={}


def head_feature_stage(mode, doc, conf):
    mode=str(mode)
    if mode not in _STAGE_MAP:
        all_train=u.sets()[1]
        if mode=='final':
            sides=inner_split(all_train,SEED+900)
            _STAGE_MAP[mode]={did:f'final_inner_{side}' for did,side in sides.items()}
            _STAGE_MAP[mode]['__other__']='full'
        else:
            fold=int(mode)
            fit=[d for d in all_train if conf['folds'][d['id']]!=fold]
            sides=inner_split(fit,SEED+fold)
            mapping={did:f'inner_{fold}_{side}' for did,side in sides.items()}
            mapping.update({d['id']:f'outer_{fold}' for d in all_train if conf['folds'][d['id']]==fold})
            mapping['__other__']=f'outer_{fold}'
            _STAGE_MAP[mode]=mapping
    return _STAGE_MAP[mode].get(doc['id'],_STAGE_MAP[mode]['__other__'])


def load_feature(stage,doc,device,flags):
    obj=torch.load(feature_path(stage,doc['id']),map_location='cpu',weights_only=False)
    if obj['n']!=len(doc['text']):raise RuntimeError('Feature count mismatch '+doc['id'])
    emb=obj['embedding'].float()
    logits=obj['logits'].float()
    lexical=flags[doc['id']]
    if lexical.shape!=(len(doc['text']),len(FLAG_NAMES)):
        raise RuntimeError('Lexical feature count mismatch '+doc['id'])
    values=torch.cat([emb,logits.softmax(-1),lexical],dim=-1)
    return values.pin_memory().to(device,non_blocking=True)


def gold_segments(labels,vocab):
    ids=[vocab.index(y) for y in labels]
    segments=[]
    start=0
    while start<len(ids):
        end=start+1
        while end<len(ids) and ids[end]==ids[start] and end-start<MAX_SPAN:end+=1
        segments.append((start,end,ids[start]))
        start=end
    return segments


class SectionTracker(nn.Module):
    def __init__(self,classes=13):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(768+13+len(FLAG_NAMES),256),
                                 nn.LayerNorm(256),nn.GELU(),nn.Dropout(.30))
        self.conv=nn.ModuleList([nn.Conv1d(256,256,5,padding=2*d,dilation=d,groups=1)
                                 for d in (1,2,4)])
        self.conv_norm=nn.LayerNorm(256)
        self.gru=nn.GRU(256,128,num_layers=2,batch_first=True,bidirectional=True,dropout=.25)
        layer=nn.TransformerEncoderLayer(256,8,512,dropout=.20,activation='gelu',
                                          batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,2)
        self.final_gru=nn.GRU(256,128,batch_first=True,bidirectional=True)
        self.norm=nn.LayerNorm(256)
        self.role=nn.Linear(256,classes)
        self.boundary=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,1))
        self.argument=nn.Linear(256,2)
        self.precedent=nn.Linear(256,2)
        self.span=nn.Sequential(nn.Linear(256*3+1,256),nn.LayerNorm(256),
                                nn.GELU(),nn.Dropout(.25),nn.Linear(256,classes))

    def span_logits(self,h,starts,ends):
        prefix=F.pad(h.cumsum(0),(0,0,1,0))
        mean=(prefix[ends]-prefix[starts])/(ends-starts).unsqueeze(-1)
        lengths=((ends-starts).float().log1p()/math.log1p(MAX_SPAN)).unsqueeze(-1)
        return self.span(torch.cat([h[starts],h[ends-1],mean,lengths],-1))

    def forward(self,features,span_starts=None,span_ends=None):
        x=self.input(features)
        for conv in self.conv:
            x=self.conv_norm(x+F.gelu(conv(x.T.unsqueeze(0)).squeeze(0).T))
        recurrent,_=self.gru(x.unsqueeze(0))
        h=x+recurrent[0]
        summaries=[]
        for start in range(0,len(h),64):summaries.append(h[start:start+64].mean(0))
        global_state=self.blocks(torch.stack(summaries)[None])[0]
        global_state=torch.repeat_interleave(global_state,64,dim=0)[:len(h)]
        h=self.norm(h+global_state)
        contextual,_=self.final_gru(h[None])
        h=self.norm(h+contextual[0])
        boundary=self.boundary(torch.cat([h[:-1],h[1:]],-1)).squeeze(-1)
        spans=self.span_logits(h,span_starts,span_ends) if span_starts is not None else None
        return h,self.role(h),boundary,self.argument(h),self.precedent(h),spans


def global_weights(docs,vocab,device):
    counts=Counter(y for d in docs for y in d['labels'])
    typical=sum(counts.values())/len(vocab)
    values=[min(3.5,max(.65,math.sqrt(typical/max(1,counts[y])))) for y in vocab]
    result=torch.tensor(values,device=device,dtype=torch.float32)
    return result/result.mean()


def global_loss(output,doc,vocab,weights,device):
    h,role,boundary,argument,precedent,span=output
    y=torch.tensor([vocab.index(z) for z in doc['labels']],device=device)
    role_loss=F.cross_entropy(role.float(),y,weight=weights,label_smoothing=.04)
    edge=(y[:-1]!=y[1:]).float()
    edge_loss=F.binary_cross_entropy_with_logits(boundary.float(),edge,
                pos_weight=torch.tensor(3.,device=device)) if len(edge) else role_loss*0
    starts,ends,roles=zip(*gold_segments(doc['labels'],vocab))
    span_y=torch.tensor(roles,device=device)
    span_loss=F.cross_entropy(span.float(),span_y,weight=weights,label_smoothing=.02)
    # Every document must touch both auxiliary heads. Some documents contain
    # neither pair of labels; a zero-valued term keeps DDP's gradient set stable.
    pair_loss=argument.float().sum()*0+precedent.float().sum()*0
    for names,logits in ((('ArgumentPetitioner','ArgumentRespondent'),argument),
                          (('PrecedentReliedUpon','PrecedentNotReliedUpon'),precedent)):
        take=(y==vocab.index(names[0]))|(y==vocab.index(names[1]))
        if bool(take.any()):
            pair_target=(y[take]==vocab.index(names[1])).long()
            pair_loss=pair_loss+F.cross_entropy(logits[take].float(),pair_target)
    return role_loss+.40*edge_loss+.40*span_loss+.12*pair_loss


def transition_prior(docs,vocab):
    counts=np.ones((len(vocab),len(vocab)),dtype=np.float64)*.5
    for d in docs:
        seg=gold_segments(d['labels'],vocab)
        for (_,_,a),(_,_,b) in zip(seg,seg[1:]):counts[a,b]+=1
    counts=counts/counts.sum(1,keepdims=True)
    return np.log(counts).astype('float32')


@torch.no_grad()
def decode_sections(model,h,role,boundary,prior,role_override=None):
    """Duration-limited semi-Markov decoder; same-label adjacent spans are valid."""
    scores=role_override if role_override is not None else role
    n,classes=scores.shape
    token_log=F.log_softmax(scores.float(),-1)
    prefix=F.pad(token_log.cumsum(0),(0,0,1,0))
    edge_yes=F.logsigmoid(boundary.float())
    edge_no=F.logsigmoid(-boundary.float())
    edge_prefix=F.pad(edge_no.cumsum(0),(1,0))
    candidates=np.full((n+1,MAX_SPAN,classes),-1e9,dtype='float32')
    for length in range(1,min(MAX_SPAN,n)+1):
        starts=torch.arange(n-length+1,device=h.device)
        ends=starts+length
        role_part=.70*(prefix[ends]-prefix[starts])
        interior=edge_prefix[ends-1]-edge_prefix[starts]
        begins=torch.zeros_like(interior)
        active=starts>0
        begins[active]=edge_yes[starts[active]-1]
        boundary_part=.4*(interior+begins)
        span_parts=[]
        for a in range(0,len(starts),8192):
            span_parts.append(F.log_softmax(model.span_logits(h,starts[a:a+8192],ends[a:a+8192]).float(),-1))
        span_part=torch.cat(span_parts)
        joint=role_part+.85*span_part+boundary_part[:,None]
        candidates[ends.cpu().numpy(),length-1,:]=joint.cpu().numpy()
    transitions=.16*prior.copy()
    np.fill_diagonal(transitions,0.)
    dp=np.full((n+1,classes),-1e20,dtype='float32')
    back_start=np.zeros((n+1,classes),dtype='int32')
    back_label=np.zeros((n+1,classes),dtype='int16')
    for end in range(1,n+1):
        lengths=np.arange(1,min(MAX_SPAN,end)+1)
        starts=end-lengths
        previous=dp[starts,:,None]+transitions[None,:,:]
        prev_label=previous.argmax(axis=1)
        previous_best=previous.max(axis=1)
        previous_best[starts==0]=0.
        prev_label[starts==0]=0
        totals=previous_best+candidates[end,lengths-1,:]
        best_len=totals.argmax(axis=0)
        dp[end]=totals[best_len,np.arange(classes)]
        back_start[end]=starts[best_len]
        back_label[end]=prev_label[best_len,np.arange(classes)]
    labels=np.zeros(n,dtype='int16')
    end=n;role_id=int(dp[end].argmax())
    while end>0:
        start=int(back_start[end,role_id])
        labels[start:end]=role_id
        prev=int(back_label[end,role_id]);end=start;role_id=prev
    return labels


@torch.no_grad()
def evaluate_global(model,docs,mode,conf,flags,vocab,rank,device,structured=False,prior=None):
    model.eval();own={}
    for d in balanced_assignment(docs)[rank]:
        stage=head_feature_stage(mode,d,conf)
        x=load_feature(stage,d,device,flags)
        h,role,boundary,_,_,_=model(x)
        local=torch.load(feature_path(stage,d['id']),map_location='cpu',weights_only=False)['logits'].float()
        entry={'local':local.softmax(-1),'global':role.float().softmax(-1).cpu(),
               'blend':(.35*local.softmax(-1)+.65*role.float().cpu().softmax(-1))}
        if structured:
            entry['section']=torch.from_numpy(decode_sections(model,h,role,boundary,prior))
            blend_log=torch.log(entry['blend'].to(device).clamp_min(1e-7))
            entry['blend_section']=torch.from_numpy(decode_sections(model,h,role,boundary,prior,blend_log))
        own[d['id']]=entry
    return u.gather_dict(own)


def result_metrics(results,docs,vocab):
    names=results[docs[0]['id']].keys()
    output={}
    for name in names:
        if name.endswith('section'):
            probs={did:F.one_hot(rows[name].long(),len(vocab)).float() for did,rows in results.items()}
        else:probs={did:rows[name] for did,rows in results.items()}
        output[name]=u.metrics(probs,docs,vocab)
    return output


def save_global_results(path,results,docs,vocab):
    u.atomic_torch(path,results)
    metrics=result_metrics(results,docs,vocab)
    u.atomic_json(path.with_suffix('.json'),metrics)
    return metrics


def global_documents(mode,train,dev,conf):
    if mode=='final':return train,dev,[]
    fold=int(mode)
    return ([d for d in train if conf['folds'][d['id']]!=fold],dev,
            [d for d in train if conf['folds'][d['id']]==fold])


def train_global(mode):
    mode=str(mode)
    rank,device=u.init_dist()
    try:
        random_seed(SEED+421+rank+(100 if mode=='final' else int(mode)))
        _,train,dev,_,vocab=u.sets()
        conf=u.read_json(ROOT/'config.json')
        fitting,validation,outer=global_documents(mode,train,dev,conf)
        for doc in fitting+validation+outer:
            stage=head_feature_stage(mode,doc,conf)
            if not feature_path(stage,doc['id']).is_file():
                raise RuntimeError('Missing out-of-sample feature '+stage+'/'+doc['id'])
        flags=torch.load(ROOT/'lexical_flags.pt',map_location='cpu',weights_only=False)
        stage_dir=ROOT/'global'/mode
        stage_dir.mkdir(parents=True,exist_ok=True)
        if (stage_dir/'done.json').exists():
            if rank==0:
                (stage_dir/'latest.pt').unlink(missing_ok=True)
                log('Global stage already complete',mode)
            dist.barrier();return
        model=SectionTracker(len(vocab)).to(device)
        ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,
                find_unused_parameters=False)
        try:optimizer=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=.08,fused=True)
        except (TypeError,RuntimeError):optimizer=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=.08)
        steps=math.ceil(len(fitting)/2)*GLOBAL_EPOCHS
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=5e-5)
        scaler=torch.amp.GradScaler('cuda')
        weights=global_weights(fitting,vocab,device)
        latest=stage_dir/'latest.pt'
        start=1;best=-1.;best_epoch=0
        history=u.read_json(stage_dir/'history.json') if (stage_dir/'history.json').exists() else []
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            start=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch']
        prior=transition_prior(fitting,vocab)
        if rank==0:u.atomic_torch(stage_dir/'transition.pt',prior)
        dist.barrier()
        for epoch in range(start,GLOBAL_EPOCHS+1):
            ddp.train()
            order=sorted(fitting,key=lambda d:len(d['text']))
            paired=[order[i:i+2] for i in range(0,len(order),2)]
            random.Random(SEED+epoch*41).shuffle(paired)
            order=[doc for pair in paired for doc in pair]
            if len(order)%2:order.append(order[-1])
            total=0.
            for doc in order[rank::2]:
                features=load_feature(head_feature_stage(mode,doc,conf),doc,device,flags)
                segments=gold_segments(doc['labels'],vocab)
                starts=torch.tensor([s for s,_,_ in segments],device=device,dtype=torch.long)
                ends=torch.tensor([e for _,e,_ in segments],device=device,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    output=ddp(features,starts,ends)
                    loss=global_loss(output,doc,vocab,weights,device)
                u.finite_or_fail(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();scheduler.step()
                total+=float(loss.detach())
            dist.barrier()
            results=evaluate_global(model,validation,mode,conf,flags,vocab,rank,device)
            if rank==0:
                report=result_metrics(results,validation,vocab)
                metric=report['global']['macro_supported']
                if metric>best:
                    best=metric;best_epoch=epoch
                    u.atomic_torch(stage_dir/'best.pt',{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'epoch':epoch,'dev':report,'mode':mode})
                history.append({'epoch':epoch,'train_loss_rank0':total/max(1,len(order)//2),
                                'development':report,'best_epoch':best_epoch})
                u.atomic_json(stage_dir/'history.json',history)
                u.atomic_torch(latest,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch})
                log('Global',mode,'epoch',epoch,'loss',round(history[-1]['train_loss_rank0'],4),
                    'dev F1',round(metric,4),'best',round(best,4))
                sample_gpus('global_'+mode,epoch)
                stop=epoch>=10 and epoch-best_epoch>=7
            else:stop=False
            signal=torch.tensor([int(stop)],device=device)
            dist.broadcast(signal,0)
            dist.barrier()
            if int(signal.item()):break
        state=torch.load(stage_dir/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);model.eval()
        dev_results=evaluate_global(model,validation,mode,conf,flags,vocab,rank,device,True,prior)
        fold_results=evaluate_global(model,outer,mode,conf,flags,vocab,rank,device,True,prior) if outer else None
        if rank==0:
            dev_metrics=save_global_results(stage_dir/'development.pt',dev_results,validation,vocab)
            fold_metrics=save_global_results(stage_dir/'heldout.pt',fold_results,outer,vocab) if outer else None
            u.atomic_json(stage_dir/'done.json',{'mode':mode,'best_epoch':best_epoch,
                'best_dev_raw_supported_f1':best,'development':dev_metrics,'heldout':fold_metrics,
                'fitting_ids':[d['id'] for d in fitting],
                'heldout_ids':[d['id'] for d in outer]})
            latest.unlink(missing_ok=True)
        dist.barrier()
    finally:dist.destroy_process_group()


def infer_test():
    rank,device=u.init_dist()
    try:
        _,train,dev,test,vocab=u.sets()
        conf=u.read_json(ROOT/'config.json')
        stage_dir=ROOT/'global'/'final'
        if not (stage_dir/'done.json').exists():raise RuntimeError('Train final global tracker first')
        model=SectionTracker(len(vocab)).to(device)
        state=torch.load(stage_dir/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);model.eval()
        prior=torch.load(stage_dir/'transition.pt',map_location='cpu',weights_only=False)
        flags=torch.load(ROOT/'lexical_flags.pt',map_location='cpu',weights_only=False)
        result=evaluate_global(model,test,'final',conf,flags,vocab,rank,device,True,prior)
        if rank==0:
            u.atomic_torch(ROOT/'test_results.pt',result)
            log('Inferred',len(result),'test documents; all results come from the new trained model')
        dist.barrier()
    finally:dist.destroy_process_group()


def select():
    _,train,dev,_,vocab=u.sets()
    names=('local','global','blend','section','blend_section')
    final_dir=ROOT/'global'/'final'
    if not (final_dir/'development.pt').exists():raise RuntimeError('Final model missing')
    final_dev=torch.load(final_dir/'development.pt',map_location='cpu',weights_only=False)
    final_metrics=result_metrics(final_dev,dev,vocab)
    done_folds=[f for f in range(FOLDS) if (ROOT/'global'/str(f)/'done.json').exists()]
    oof={};heldout=[]
    for f in done_folds:
        docs=[d for d in train if u.read_json(ROOT/'config.json')['folds'][d['id']]==f]
        result=torch.load(ROOT/'global'/str(f)/'heldout.pt',map_location='cpu',weights_only=False)
        if set(result)!={d['id'] for d in docs}:raise RuntimeError('Incomplete fold result '+str(f))
        overlap=set(oof)&set(result)
        if overlap:raise RuntimeError('Duplicate OOF documents '+str(overlap))
        oof.update(result);heldout.extend(docs)
    oof_metrics=result_metrics(oof,heldout,vocab) if heldout else None
    if len(done_folds)>=3:
        choice=max(names,key=lambda name:oof_metrics[name]['macro_supported'])
        basis='document-held-out pooled OOF macro F1'
    else:
        choice=max(names,key=lambda name:final_metrics[name]['macro_supported'])
        basis='official development macro F1; fewer than three outer folds completed'
    report={'selected':choice,'basis':basis,'completed_outer_folds':done_folds,
            'oof_documents':len(heldout),'oof':oof_metrics,'development':final_metrics,
            'private_score':None,'target_private_macro_f1':.8,
            'caution':'0.8 is an objective; a Kaggle evaluation is required to measure it.',
            'prior_submission_used_as_output':False}
    u.atomic_json(ROOT/'selection.json',report)
    log('Selected',choice,'based on',basis,'development F1',round(final_metrics[choice]['macro_supported'],4),
        'OOF F1',None if not oof_metrics else round(oof_metrics[choice]['macro_supported'],4))


def prior_six_predictions():
    source=u.locate('CS781_Phase3_Submission_06.zip')
    if source is None:return None
    with zipfile.ZipFile(source) as z:
        if 'prediction.json' not in z.namelist():return None
        return json.loads(z.read('prediction.json'))


def package():
    _,_,_,test,vocab=u.sets()
    if not (ROOT/'test_results.pt').is_file():raise RuntimeError('Inference missing')
    selection=u.read_json(ROOT/'selection.json')
    chosen=selection['selected']
    results=torch.load(ROOT/'test_results.pt',map_location='cpu',weights_only=False)
    if set(results)!={d['id'] for d in test}:raise RuntimeError('Test ID mismatch')
    predictions={}
    for doc in test:
        value=results[doc['id']][chosen]
        labels=value.tolist() if chosen.endswith('section') else value.argmax(-1).tolist()
        predictions[doc['id']]=[vocab[int(i)] for i in labels]
        if len(labels)!=len(doc['text']):raise RuntimeError('Test sentence count mismatch')
    if any(y not in vocab for rows in predictions.values() for y in rows):
        raise RuntimeError('Unknown prediction label')
    old=prior_six_predictions()
    changed=None
    if old is not None and set(old)==set(predictions):
        changed=sum(a!=b for did in predictions for a,b in zip(predictions[did],old[did]))
    if changed==0:
        log('WARNING: new model predictions equal the old slot 06; inspect before upload')
    u.atomic_json(ROOT/'prediction.json',predictions)
    temp=SUBMISSION.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT/'prediction.json','prediction.json')
    os.replace(temp,SUBMISSION)
    with zipfile.ZipFile(SUBMISSION) as z:
        if z.namelist()!=['prediction.json'] or json.loads(z.read('prediction.json'))!=predictions:
            raise RuntimeError('Submission verification failed')
    shutil.copy2(SUBMISSION,WORK/'submission.zip')
    shutil.copy2(SUBMISSION,ROOT/'submission.zip')
    pilot_path=ROOT/'pilot_gate.json'
    pilot=u.read_json(pilot_path) if pilot_path.exists() else None
    status=('WEAK_HELDOUT_PILOT_REVIEW_BEFORE_UPLOAD' if pilot and not pilot['continue_outer_folds']
            else 'READY_FOR_MANUAL_UPLOAD')
    u.atomic_json(ROOT/'report.json',{'status':status,
        'submission':str(SUBMISSION),'submission_sha256':u.digest(SUBMISSION),
        'artifact':str(ARTIFACT),'selected':chosen,'selection_basis':selection['basis'],
        'completed_outer_folds':selection['completed_outer_folds'],
        'development_supported_macro_f1':selection['development'][chosen]['macro_supported'],
        'oof_supported_macro_f1':None if selection['oof'] is None else selection['oof'][chosen]['macro_supported'],
        'changed_labels_vs_slot06':changed,'test_documents':len(test),
        'test_sentences':sum(len(d['text']) for d in test),'private_score':None,
        'pilot_gate':pilot,
        'note':'A 0.8 private score is a target, not a measured or guaranteed result.'})
    u.snapshot()
    with zipfile.ZipFile(ARTIFACT) as z:
        integrity=json.loads(z.read('integrity.json'))['files']
        for name in ('hier/report.json','hier/prediction.json','hier/submission.zip',
                     'hier/test_results.pt','hier/global/final/best.pt','prior/bases/03/checkpoints/selected_model.pt'):
            if name not in integrity:raise RuntimeError('Recovery missing '+name)
    log('Submission',SUBMISSION,'artifact',ARTIFACT,'changed labels',changed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','local','extract','global','infer-test',
                                          'select','package','snapshot'))
    parser.add_argument('--stage',default=None)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='local':train_local(args.stage)
    elif args.action=='extract':extract_local(args.stage)
    elif args.action=='global':train_global(args.stage)
    elif args.action=='infer-test':infer_test()
    elif args.action=='select':select()
    elif args.action=='package':package()
    elif args.action=='snapshot':u.snapshot()


if __name__=='__main__':main()

