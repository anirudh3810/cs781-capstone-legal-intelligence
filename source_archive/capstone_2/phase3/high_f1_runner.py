"""Phase 3 high-F1 experiment. Embedded by build_high_f1_notebook.py.

Only the user's Phase 1 checkpoints and official Phase 3 data are used.
All distributed jobs exit before a recovery archive is rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

import cs781_phase3_source_utils as u

WORK = Path('/kaggle/working')
ROOT = WORK / 'CS781_Capstone/phase3_high_f1'
ARTIFACT = WORK / 'CS781_Phase3_HighF1_All_Artifacts.zip'
SUBMISSION = WORK / 'CS781_Phase3_HighF1_Submission.zip'
u.ROOT = ROOT
u.ARTIFACT = ARTIFACT

SEED = 781550
FOLDS = 5
WINDOW = 12
STRIDE = 8
MAX_TOKENS = 448
TARGET_LIMIT = 256
ACCUM = 2
EPOCHS = {'03': 11, '01': 9}
MIN_EPOCHS = {'03': 5, '01': 4}
PATIENCE = 3
PAIR_NAMES = (
    ('ArgumentPetitioner', 'ArgumentRespondent'),
    ('PrecedentReliedUpon', 'PrecedentNotReliedUpon'),
    ('RulingByLowerCourt', 'RulingByPresentCourt'),
)


def log(*items):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *items, flush=True)


def seed_all(value):
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def balance_folds(docs):
    mapping = {}
    for domain in ('CL', 'IT'):
        rows = [d for d in docs if d['domain'] == domain]
        # Keep the two Dissent documents in different folds. This does not
        # create examples of the single-document Overruled class.
        rows.sort(key=lambda d: (-len(d['text']), hashlib.sha256((str(SEED)+d['id']).encode()).hexdigest()))
        groups = [[] for _ in range(FOLDS)]
        loads = [0] * FOLDS
        for doc in rows:
            eligible = [i for i in range(FOLDS) if len(groups[i]) < 8]
            if 'Dissent' in (doc['labels'] or []):
                eligible.sort(key=lambda i: (sum('Dissent' in (x['labels'] or []) for x in groups[i]), loads[i]))
                slot = eligible[0]
            else:
                slot = min(eligible, key=lambda i: (loads[i], len(groups[i]), i))
            groups[slot].append(doc)
            loads[slot] += len(doc['text'])
        if any(len(g) != 8 for g in groups):
            raise RuntimeError('Fold allocation did not preserve eight documents per domain')
        for fold, group in enumerate(groups):
            for doc in group:
                mapping[doc['id']] = fold
    return mapping


def encode_tokens(docs, tokenizer, branch):
    enc = lambda s: tokenizer.backend.encode(s, add_special_tokens=False).ids
    marker = enc(' [TARGET SENTENCE] ')
    sep = enc(' [SENTENCE] ')
    left_marker = enc(' [EARLIER CONTEXT] ')
    right_marker = enc(' [LATER CONTEXT] ')
    if not marker:
        raise RuntimeError('Target marker did not tokenize')
    result = {}
    for doc in docs:
        raw = [enc(s) for s in doc['text']]
        rows = []
        for index, original in enumerate(raw):
            target = original
            if len(target) > TARGET_LIMIT:
                front = TARGET_LIMIT * 3 // 4
                target = target[:front] + target[-(TARGET_LIMIT-front):]
            target = target or [int(tokenizer.pad_token_id)]
            earlier = left_marker + [v for seg in raw[max(0,index-2):index] for v in sep+seg]
            later = right_marker + [v for seg in raw[index+1:index+3] for v in sep+seg]
            room = MAX_TOKENS - len(marker) - len(target)
            left_n = min(len(earlier), room // 2)
            right_n = min(len(later), room - left_n)
            left_n = min(len(earlier), room - right_n)
            left = earlier[-left_n:] if left_n else []
            right = later[:right_n]
            if branch == '03':
                ids = left + marker + target + right
                start = len(left) + len(marker)
            else:
                # In the causal model the target comes last, after both sides
                # of its context, so its hidden states can attend to both.
                ids = left + right + marker + target
                start = len(left) + len(right) + len(marker)
            if not 0 <= start < start+len(target) <= len(ids) <= MAX_TOKENS:
                raise RuntimeError('Invalid target/context tokenization')
            rows.append((ids, start, start+len(target)))
        result[doc['id']] = rows
    return result


def stage_base_robust(base_id):
    """Accept either a recovery ZIP or Kaggle's extracted dataset directory."""
    dest = ROOT/'bases'/base_id
    needed = [f'model/{u.SOURCES[base_id]}', 'model/course_bpe.py',
              'checkpoints/selected_model.pt', 'tokenizer/tokenizer.json',
              'tokenizer/tokenizer_metadata.json']
    if all((dest/name).is_file() for name in needed):
        return dest
    archive = u.find_one(u.BASE_FILES[base_id], required=False)
    if archive is not None:
        return u.stage_base(base_id)
    dataset_name = f'phase1_submission{base_id}_recovery'
    candidates = []
    for source in u.INPUT.rglob(u.SOURCES[base_id]):
        if source.name != u.SOURCES[base_id] or source.parent.name != 'model':
            continue
        base = source.parent.parent
        normalize = lambda value: ''.join(c for c in value.lower() if c.isalnum())
        if not any(normalize(part) == normalize(dataset_name) for part in base.parts):
            continue
        if all((base/name).is_file() for name in needed):
            candidates.append(base)
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one complete {dataset_name} Kaggle input; found {candidates}. '
                           'Attach the Phase 1 recovery dataset or ZIP.')
    source = candidates[0]
    wanted = needed + [str(path.relative_to(source)) for path in (source/'tokenizer').rglob('*') if path.is_file()]
    for name in sorted(set(wanted)):
        target = dest/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temp = target.with_suffix(target.suffix+'.tmp')
            shutil.copyfile(source/name,temp)
            os.replace(temp,target)
    u.write_json(dest/'provenance.json', {'mounted_dataset':str(source),
        'source_sha256':u.sha(dest/needed[0]),
        'checkpoint_sha256':u.sha(dest/'checkpoints/selected_model.pt')})
    return dest


def prepare():
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Select the Kaggle accelerator with exactly two GPUs before Run All')
    ROOT.mkdir(parents=True, exist_ok=True)
    u.restore_artifact()
    stage_base_robust('03')
    u.stage_data()
    train, dev, test = u.all_data()
    vocab = u.labels()
    if (len(train), len(dev), len(test), len(vocab)) != (80, 10, 10, 13):
        raise RuntimeError('Unexpected official Phase 3 schema/counts')
    config = {
        'version': 1, 'source_sha256': u.sha(__file__), 'seed': SEED,
        'folds': balance_folds(train), 'window': WINDOW, 'stride': STRIDE,
        'max_tokens': MAX_TOKENS, 'target_limit': TARGET_LIMIT,
        'train_ids': [d['id'] for d in train], 'dev_ids': [d['id'] for d in dev],
        'test_ids': [d['id'] for d in test], 'vocab': vocab,
        'official_hashes': {name: u.sha(ROOT/'official'/name) for name in u.DATA_FILES},
    }
    path = ROOT / 'run_config.json'
    if path.exists() and u.read_json(path) != config:
        raise RuntimeError('This recovery artifact belongs to different code or official data')
    u.write_json(path, config)
    shutil.copy2(__file__, ROOT/'high_f1_runner.py')
    shutil.copy2(Path(u.__file__), ROOT/'source_utils.py')
    cache = ROOT/'tokens_03.pt'
    if not cache.exists():
        _, tokenizer = u.load_base('03')
        u.save_torch(cache, encode_tokens(train+dev+test, tokenizer, '03'))
    counts = Counter(y for d in train for y in d['labels'])
    doc_support = Counter(y for d in train for y in set(d['labels']))
    u.write_json(ROOT/'data_audit.json', {
        'train_documents': len(train), 'train_sentences': sum(len(d['text']) for d in train),
        'dev_documents': len(dev), 'dev_sentences': sum(len(d['text']) for d in dev),
        'test_documents': len(test), 'test_sentences': sum(len(d['text']) for d in test),
        'label_sentences': dict(counts), 'label_documents': dict(doc_support),
        'phase1_sources': ['03', '01 (conditional second branch)'],
        'private_score': None,
    })
    log('Prepared official data, Phase 1 base 03, and target-aware token cache')


def prepare_branch01():
    stage_base_robust('01')
    if not (ROOT/'tokens_01.pt').exists():
        train,dev,test = u.all_data()
        _, tokenizer = u.load_base('01')
        u.save_torch(ROOT/'tokens_01.pt', encode_tokens(train+dev+test,tokenizer,'01'))
    log('Prepared wider Phase 1 base 01 branch')


def stage_docs(branch, stage):
    train, dev, test = u.all_data()
    foldmap = u.read_json(ROOT/'run_config.json')['folds']
    if stage == 'full':
        return train, dev, [], test
    if not stage.startswith('fold_'):
        raise ValueError(stage)
    fold = int(stage.split('_')[1])
    if fold not in range(FOLDS):
        raise ValueError(stage)
    fit = [d for d in train if foldmap[d['id']] != fold]
    held = [d for d in train if foldmap[d['id']] == fold]
    if len(fit) != 64 or len(held) != 16:
        raise RuntimeError('Wrong fold document counts')
    return fit, dev, held, test


def windows(doc):
    n = len(doc['text'])
    for start in range(0, n, STRIDE):
        yield doc['id'], start, min(n, start+WINDOW)
        if start+WINDOW >= n:
            break


def epoch_windows(docs, epoch):
    rng = random.Random(SEED + 1009*epoch)
    focus = {'ArgumentRespondent','PrecedentNotReliedUpon','RulingByLowerCourt',
             'Issue','RulingByPresentCourt'}
    sampled = []
    for doc in docs:
        allw = list(windows(doc))
        quota = min(len(allw), max(8, round(3.2*math.sqrt(len(allw)))))
        if len(allw) <= quota:
            sampled.extend(allw)
            continue
        hard = []
        gold = doc['labels']
        for item in allw:
            _,a,b = item
            if any(y in focus for y in gold[a:b]) or any(gold[j]!=gold[j-1] for j in range(a+1,b)):
                hard.append(item)
        rng.shuffle(hard)
        chosen = hard[:quota//3]
        rest = [w for w in allw if w not in chosen]
        rng.shuffle(rest)
        sampled.extend(chosen + rest[:quota-len(chosen)])
    rng.shuffle(sampled)
    while len(sampled) % (2*ACCUM):
        sampled.append(sampled[len(sampled) % len(sampled)])
    return sampled


def collate(item, docs, tokens, pad, vocab, device):
    did,a,b = item
    doc = docs[did]
    entries = tokens[did][a:b]
    width = max(len(x[0]) for x in entries)
    count = len(entries)
    ids = torch.full((count,width),pad,dtype=torch.long)
    visible = torch.zeros((count,width),dtype=torch.bool)
    target = torch.zeros((count,width),dtype=torch.bool)
    for i,(sequence,start,end) in enumerate(entries):
        ids[i,:len(sequence)] = torch.tensor(sequence,dtype=torch.long)
        visible[i,:len(sequence)] = True
        target[i,start:end] = True
    pos = torch.tensor([min(31,int(32*j/max(1,len(doc['text'])))) for j in range(a,b)],dtype=torch.long)
    domain = torch.full((count,),0 if doc['domain']=='CL' else 1,dtype=torch.long)
    y = None if doc['labels'] is None else torch.tensor([vocab.index(label) for label in doc['labels'][a:b]],dtype=torch.long)
    transfer = lambda t: t.pin_memory().to(device,non_blocking=True) if device.type=='cuda' else t
    return tuple(transfer(t) if t is not None else None for t in (ids,visible,target,pos,domain,y))


class RoleTagger(nn.Module):
    def __init__(self, base, branch, labels=13):
        super().__init__()
        self.base = base
        self.branch = branch
        for param in base.parameters():
            param.requires_grad_(False)
        if branch == '03':
            self.first_trainable = len(base.blocks)-16
            full_start = len(base.blocks)-12
            adapter_start = self.first_trainable
            adapter_names = ('query','value')
            width = base.hidden_size
        else:
            self.first_trainable = 0
            full_start = len(base.blocks)-4
            adapter_start = 0
            adapter_names = ('qkv',)
            width = base.config.hidden_size
        for index,block in enumerate(base.blocks):
            if index >= adapter_start:
                for name,module in list(block.named_modules()):
                    if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in adapter_names:
                        parent = block.get_submodule(name.rsplit('.',1)[0])
                        setattr(parent,name.rsplit('.',1)[-1],u.LoRA(module,16,32))
            if index >= full_start:
                for param in block.parameters():param.requires_grad_(True)
        for param in base.final_norm.parameters():param.requires_grad_(True)
        self.att = nn.Linear(width,1)
        self.project = nn.Sequential(nn.Linear(width*3,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(.24))
        self.position = nn.Embedding(32,512)
        self.domain = nn.Embedding(2,512)
        self.sequence = nn.GRU(512,256,num_layers=2,batch_first=True,bidirectional=True,dropout=.2)
        self.norm = nn.LayerNorm(512)
        self.fine = nn.Sequential(nn.Dropout(.22),nn.Linear(512,labels))
        self.pairs = nn.ModuleList(nn.Sequential(nn.Dropout(.2),nn.Linear(512,2)) for _ in PAIR_NAMES)
        self.transitions = nn.Parameter(torch.zeros(labels,labels))

    def hidden(self,ids,visible):
        base = self.base
        if self.branch == '03':
            x = base.embedding_dropout(base.token_embedding(ids)*math.sqrt(base.hidden_size))
            attention = base._prefix_hybrid_visibility(visible,visible.sum(1))
            cosine,sine = base.rotary(ids.size(1),x.dtype)
            for i,block in enumerate(base.blocks):
                if i < self.first_trainable:
                    with torch.no_grad(): x = block(x,attention,cosine,sine)
                else:
                    x = checkpoint(block,x,attention,cosine,sine,use_reentrant=False) if self.training else block(x,attention,cosine,sine)
        else:
            positions = torch.arange(ids.size(1),device=ids.device)
            x = base.embedding_dropout(base.token_embedding(ids)+base.position_embedding(positions)[None])
            for block in base.blocks:
                x = checkpoint(block,x,use_reentrant=False) if self.training else block(x)
        return base.final_norm(x)

    def forward(self,ids,visible,target,pos,domain):
        x = self.hidden(ids,visible)
        attention = self.att(x).squeeze(-1).masked_fill(~target,-1e4).softmax(-1)
        pooled = (x*attention.unsqueeze(-1)).sum(1)
        mean = (x*target.unsqueeze(-1)).sum(1)/target.sum(1,keepdim=True).clamp_min(1)
        maximum = x.masked_fill(~target.unsqueeze(-1),-1e4).max(1).values
        h = self.project(torch.cat([pooled,mean,maximum],-1)) + self.position(pos) + self.domain(domain)
        context,_ = self.sequence(h[None])
        h = self.norm(h+context[0])
        return self.fine(h),[head(h) for head in self.pairs],self.transitions


def trainable_state(model):
    names = {name for name,p in model.named_parameters() if p.requires_grad}
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in names}


def load_state(model,state):
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise RuntimeError(f'Trainable-state mismatch: missing={expected-set(state)}, extra={set(state)-expected}')
    model.load_state_dict(state,strict=False)


def crf_nll(emission,y,transition):
    score = emission[0,y[0]]
    alpha = emission[0]
    for index in range(1,len(y)):
        score = score + transition[y[index-1],y[index]] + emission[index,y[index]]
        alpha = torch.logsumexp(alpha[:,None] + transition,dim=0) + emission[index]
    return (torch.logsumexp(alpha,dim=0)-score)/len(y)


def class_weights(docs,vocab,device):
    counts = Counter(y for d in docs for y in d['labels'])
    typical = sum(counts.values())/len(vocab)
    w = torch.tensor([min(3.,max(.65,math.sqrt(typical/max(1,counts[y])))) for y in vocab],device=device)
    return w/w.mean()


def loss_for(output,y,vocab,weights):
    fine,pair_logits,trans = output
    loss = F.cross_entropy(fine.float(),y,weight=weights,label_smoothing=.03)
    loss = loss + .10*crf_nll(fine.float(),y,trans.float())
    for (first,second),logits in zip(PAIR_NAMES,pair_logits):
        a,b=vocab.index(first),vocab.index(second)
        mask=(y==a)|(y==b)
        pair=logits.float().sum()*0
        if bool(mask.any()):
            pair=F.cross_entropy(logits[mask].float(),(y[mask]==b).long())
        loss=loss+.13*pair
    return loss


def scores_with_pairs(fine,pairs,vocab):
    result=fine.clone()
    for (a,b),logits in zip(PAIR_NAMES,pairs):
        indexes=[vocab.index(a),vocab.index(b)]
        result[:,indexes] += .35*F.log_softmax(logits.float(),-1)
    return result


def balanced_assignment(docs):
    groups=[[],[]];load=[0,0]
    for doc in sorted(docs,key=lambda d:len(d['text']),reverse=True):
        rank=0 if load[0]<=load[1] else 1
        groups[rank].append(doc)
        load[rank]+=len(doc['text'])
    return groups


@torch.no_grad()
def predict_documents(model,docs,all_docs,tokens,pad,vocab,rank,device):
    model.eval()
    output={}
    by_id={d['id']:d for d in all_docs}
    for doc in balanced_assignment(docs)[rank]:
        n=len(doc['text'])
        summed=torch.zeros((n,len(vocab)),dtype=torch.float32)
        paired=torch.zeros_like(summed)
        counts=torch.zeros(n,dtype=torch.float32)
        for item in windows(doc):
            _,a,b=item
            ids,visible,target,pos,domain,_=collate(item,by_id,tokens,pad,vocab,device)
            with (torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()):
                fine,pairs,_=model(ids,visible,target,pos,domain)
            fine=fine.float().cpu()
            fused=scores_with_pairs(fine,[p.float().cpu() for p in pairs],vocab)
            summed[a:b]+=fine
            paired[a:b]+=fused
            counts[a:b]+=1
        if not bool((counts>0).all()):raise RuntimeError('Uncovered sentence '+doc['id'])
        output[doc['id']]={'raw':summed/counts[:,None], 'pair':paired/counts[:,None],
                           'transition':model.transitions.detach().float().cpu().clone()}
    gathered=[None,None]
    dist.all_gather_object(gathered,output)
    return {did:value for part in gathered for did,value in part.items()}


def viterbi(emission,transition,weight):
    if weight==0:return emission.argmax(-1).tolist()
    e=emission.float().numpy()
    t=transition.float().numpy()*weight
    n,classes=e.shape
    dp=e[0].copy();back=np.zeros((n,classes),dtype=np.int16)
    for i in range(1,n):
        scores=dp[:,None]+t
        back[i]=scores.argmax(0)
        dp=e[i]+scores.max(0)
    pred=np.zeros(n,dtype=np.int16);pred[-1]=dp.argmax()
    for i in range(n-1,0,-1):pred[i-1]=back[i,pred[i]]
    return pred.tolist()


def candidate(result,mode):
    source='pair' if mode.startswith('pair') else 'raw'
    weight={'raw':0.,'pair':0.,'pair_crf025':.25,'pair_crf05':.5}.get(mode)
    if weight is None:raise ValueError(mode)
    return {did:viterbi(value[source],value['transition'],weight) for did,value in result.items()}


def metric(pred,docs,vocab):
    by_id={d['id']:d for d in docs}
    if set(pred)!=set(by_id):raise RuntimeError('Metric document set mismatch')
    counters={name:{'tp':0,'pred':0,'gold':0} for name in vocab}
    domain_counts={d:{name:{'tp':0,'pred':0,'gold':0} for name in vocab} for d in ('CL','IT')}
    for did,labels in pred.items():
        doc=by_id[did]
        if len(labels)!=len(doc['labels']):raise RuntimeError('Metric sentence mismatch')
        for guess,gold in zip(labels,doc['labels']):
            predicted=vocab[int(guess)]
            for table in (counters,domain_counts[doc['domain']]):
                table[predicted]['pred']+=1
                table[gold]['gold']+=1
                if predicted==gold:table[gold]['tp']+=1
    def summarize(table):
        f1=[];supported=[];per={}
        for name,counts in table.items():
            p=counts['tp']/counts['pred'] if counts['pred'] else 0.
            r=counts['tp']/counts['gold'] if counts['gold'] else 0.
            value=2*p*r/(p+r) if p+r else 0.
            per[name]={**counts,'precision':p,'recall':r,'f1':value}
            f1.append(value)
            if counts['gold']:supported.append(value)
        return {'macro_all13':sum(f1)/len(f1),'macro_supported':sum(supported)/len(supported),
                'per_label':per}
    result=summarize(counters)
    result['domain_supported']={key:summarize(value)['macro_supported'] for key,value in domain_counts.items()}
    return result


def init_dist():
    if int(os.environ.get('WORLD_SIZE','1'))!=2 or torch.cuda.device_count()!=2:
        raise RuntimeError('Distributed stage requires torchrun with two GPUs')
    local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda',local)
    try:dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    except TypeError:dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    return int(os.environ['RANK']),device


def make_model(branch,device):
    base,tokenizer=u.load_base(branch)
    model=RoleTagger(base,branch,len(u.labels())).to(device)
    return model,tokenizer


def optimizer_for(model,total_updates):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('base.blocks.'):
            layer=int(name.split('.')[2]);depth=len(model.base.blocks)
            lr=(4e-6 + 1.1e-5*layer/max(1,depth-1)) if '.base.' in name or '.a.' not in name and '.b.' not in name else 2e-5
        elif name.startswith('base.'):
            lr=1.5e-5
        else:lr=1.2e-4
        decay=0. if name.endswith('bias') or 'norm' in name.lower() else .02
        groups.setdefault((lr,decay),[]).append(p)
    params=[{'params':ps,'lr':lr,'weight_decay':wd} for (lr,wd),ps in groups.items()]
    try:opt=torch.optim.AdamW(params,fused=True)
    except (TypeError,RuntimeError):opt=torch.optim.AdamW(params)
    warm=max(10,int(.06*total_updates))
    def factor(step):
        if step<warm:return max(.1,(step+1)/warm)
        frac=min(1.,(step-warm)/max(1,total_updates-warm))
        return .1+.9*.5*(1+math.cos(math.pi*frac))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,factor)


def gpu_sample(stage,epoch):
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
        rows=[list(map(str.strip,line.split(','))) for line in p.stdout.strip().splitlines()]
    except Exception as exc:rows=[{'error':repr(exc)}]
    with (ROOT/'gpu_samples.jsonl').open('a',encoding='utf8') as out:
        out.write(json.dumps({'stage':stage,'epoch':epoch,'time':time.time(),'gpus':rows})+'\n')


def train_stage(branch,stage,chunk_epochs):
    rank,device=init_dist()
    try:
        seed_all(SEED+sum(map(ord,branch+stage))+rank)
        fit,dev,held,test=stage_docs(branch,stage)
        vocab=u.labels()
        path=ROOT/'runs'/branch/stage
        path.mkdir(parents=True,exist_ok=True)
        status=path/'status.json'
        if status.exists() and u.read_json(status).get('complete'):
            if rank==0:log('REUSE completed',branch,stage)
            dist.barrier();return
        cache=ROOT/f'tokens_{branch}.pt'
        if not cache.exists():raise FileNotFoundError(cache)
        tokens=torch.load(cache,map_location='cpu',weights_only=False)
        all_docs={d['id']:d for d in fit+dev+held+test}
        model,tok=make_model(branch,device)
        pad=int(tok.pad_token_id)
        updates_per_epoch=len(epoch_windows(fit,1))//(2*ACCUM)
        optimizer,scheduler=optimizer_for(model,max(1,updates_per_epoch*EPOCHS[branch]))
        scaler=torch.amp.GradScaler('cuda')
        latest=path/'latest.pt'
        history=u.read_json(path/'history.json') if (path/'history.json').exists() else []
        start=1;best=-1.;best_epoch=0
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            load_state(model,state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            start=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch']
        ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=False,
                gradient_as_bucket_view=True)
        weights=class_weights(fit,vocab,device)
        limit=min(EPOCHS[branch],start+chunk_epochs-1)
        finished=False
        for epoch in range(start,limit+1):
            ddp.train();optimizer.zero_grad(set_to_none=True)
            windows_for_epoch=epoch_windows(fit,epoch)
            assigned=windows_for_epoch[rank::2]
            total=0.;steps=0
            for j,item in enumerate(assigned):
                ids,visible,target,pos,domain,y=collate(item,all_docs,tokens,pad,vocab,device)
                sync=(j+1)%ACCUM==0
                with (nullcontext() if sync else ddp.no_sync()):
                    with torch.autocast('cuda',dtype=torch.float16):
                        result=ddp(ids,visible,target,pos,domain)
                        loss=loss_for(result,y,vocab,weights)/ACCUM
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite loss')
                    scaler.scale(loss).backward()
                total+=float(loss.detach())*ACCUM
                if sync:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer);scaler.update();scheduler.step()
                    optimizer.zero_grad(set_to_none=True);steps+=1
            dist.barrier()
            validation=predict_documents(model,dev,list(all_docs.values()),tokens,pad,vocab,rank,device)
            score=metric(candidate(validation,'raw'),dev,vocab)
            if rank==0:
                value=score['macro_supported']
                if value>best+1e-5:
                    best=value;best_epoch=epoch
                    u.save_torch(path/'best.pt',{'model':trainable_state(model),'epoch':epoch,
                                                  'dev_supported_macro_f1':best,'branch':branch,'stage':stage})
                history.append({'epoch':epoch,'train_loss_rank0':total/max(1,len(assigned)),
                                'dev':score,'best_epoch':best_epoch,'optimizer_steps_rank0':steps})
                u.write_json(path/'history.json',history)
                u.save_torch(latest,{'model':trainable_state(model),'optimizer':optimizer.state_dict(),
                                     'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                                     'epoch':epoch,'best':best,'best_epoch':best_epoch})
                log('TRAIN',branch,stage,'epoch',epoch,'loss',round(history[-1]['train_loss_rank0'],4),
                    'dev F1',round(value,4),'best',round(best,4))
                gpu_sample(branch+'_'+stage,epoch)
                finished=(epoch>=MIN_EPOCHS[branch] and epoch-best_epoch>=PATIENCE) or epoch>=EPOCHS[branch]
                u.write_json(status,{'complete':False,'epoch':epoch,'best_epoch':best_epoch,'best_dev_f1':best})
            signal=torch.tensor([int(finished)],device=device)
            dist.broadcast(signal,0);finished=bool(signal.item())
            dist.barrier()
            if finished:break
        if finished:
            selected=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
            load_state(model,selected['model'])
            model.eval()
            dev_result=predict_documents(model,dev,list(all_docs.values()),tokens,pad,vocab,rank,device)
            held_result=predict_documents(model,held,list(all_docs.values()),tokens,pad,vocab,rank,device) if held else None
            if rank==0:
                u.save_torch(path/'dev_scores.pt',dev_result)
                if held_result is not None:u.save_torch(path/'heldout_scores.pt',held_result)
                u.write_json(status,{'complete':True,'epoch':history[-1]['epoch'],
                                     'best_epoch':best_epoch,'best_dev_f1':best,
                                     'heldout_documents':len(held)})
                latest.unlink(missing_ok=True)
                log('COMPLETE',branch,stage,'best epoch',best_epoch)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def infer_full(branch):
    rank,device=init_dist()
    try:
        path=ROOT/'runs'/branch/'full'
        if not (path/'status.json').exists() or not u.read_json(path/'status.json')['complete']:
            raise RuntimeError('Full model is incomplete')
        if (path/'test_scores.pt').exists() and (path/'dev_scores.pt').exists():
            if rank==0:log('REUSE full inference',branch)
            dist.barrier();return
        fit,dev,_,test=stage_docs(branch,'full')
        vocab=u.labels();tokens=torch.load(ROOT/f'tokens_{branch}.pt',map_location='cpu',weights_only=False)
        model,tok=make_model(branch,device)
        state=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
        load_state(model,state['model']);model.eval()
        all_docs=fit+dev+test
        result=predict_documents(model,test,all_docs,tokens,int(tok.pad_token_id),vocab,rank,device)
        if rank==0:
            u.save_torch(path/'test_scores.pt',result)
            log('INFERRED',branch,len(result),'official test documents')
        dist.barrier()
    finally:dist.destroy_process_group()


def completed_oof(branch):
    train,_,_=u.all_data()
    results={};docs=[];folds=[]
    mapping=u.read_json(ROOT/'run_config.json')['folds']
    for fold in range(FOLDS):
        path=ROOT/'runs'/branch/f'fold_{fold}'
        if not (path/'status.json').exists() or not u.read_json(path/'status.json').get('complete'):
            continue
        value=torch.load(path/'heldout_scores.pt',map_location='cpu',weights_only=False)
        expected={d['id'] for d in train if mapping[d['id']]==fold}
        if set(value)!=expected or set(results)&set(value):raise RuntimeError('Invalid OOF fold '+str(fold))
        results.update(value)
        docs.extend(d for d in train if d['id'] in expected)
        folds.append(fold)
    return results,docs,folds


def probe_wide():
    vocab=u.labels()
    primary,primary_docs,_=completed_oof('03')
    wide,_,wide_folds=completed_oof('01')
    if len(wide_folds)<2 or not set(wide).issubset(primary):
        result={'ready':False,'wide_folds':wide_folds,'continue':False}
    else:
        chosen=u.read_json(ROOT/'selection.json')['selected_mode']
        subset={did:primary[did] for did in wide}
        docs=[d for d in primary_docs if d['id'] in wide]
        blend={}
        for did in wide:
            blend[did]={key:.7*primary[did][key]+.3*wide[did][key] for key in ('raw','pair')}
            blend[did]['transition']=primary[did]['transition']
        base=metric(candidate(subset,chosen),docs,vocab)
        mixed=metric(candidate(blend,chosen),docs,vocab)
        gain=mixed['macro_supported']-base['macro_supported']
        result={'ready':True,'wide_folds':wide_folds,'mode':chosen,
                'primary_matched_oof':base,'blend_matched_oof':mixed,'gain':gain,
                'continue':gain>=.005 and all(mixed['domain_supported'][d]>=base['domain_supported'][d]-.02 for d in ('CL','IT'))}
    u.write_json(ROOT/'wide_probe.json',result)
    log('WIDE PROBE',result.get('gain'),result['continue'])


def select():
    train,dev,test=u.all_data()
    vocab=u.labels()
    primary,primary_docs,folds=completed_oof('03')
    modes=('raw','pair','pair_crf025','pair_crf05')
    oof={mode:metric(candidate(primary,mode),primary_docs,vocab) for mode in modes} if primary else {}
    full=torch.load(ROOT/'runs'/'03'/'full'/'dev_scores.pt',map_location='cpu',weights_only=False)
    dev_scores={mode:metric(candidate(full,mode),dev,vocab) for mode in modes}
    chosen=max(modes,key=lambda mode:oof[mode]['macro_supported'] if len(folds)>=2 else dev_scores[mode]['macro_supported'])
    wide,wide_docs,wide_folds=completed_oof('01')
    wide_gain=None;use_wide=False
    if len(wide_folds)>=2 and (ROOT/'runs'/'01'/'full'/'dev_scores.pt').exists():
        subset={did:primary[did] for did in wide}
        same_docs=[d for d in primary_docs if d['id'] in wide]
        raw_base=metric(candidate(subset,chosen),same_docs,vocab)
        mixed={}
        for did in wide:
            mixed[did]={name:.7*subset[did][name]+.3*wide[did][name] for name in ('raw','pair')}
            mixed[did]['transition']=subset[did]['transition']
        raw_blend=metric(candidate(mixed,chosen),same_docs,vocab)
        wide_gain=raw_blend['macro_supported']-raw_base['macro_supported']
        use_wide=(wide_gain>=.005 and all(raw_blend['domain_supported'][d]>=raw_base['domain_supported'][d]-.02 for d in ('CL','IT')))
    selection={'selected_mode':chosen,'use_wide':use_wide,'wide_gain_matched_oof':wide_gain,
               'primary_oof_folds':folds,'primary_oof_documents':len(primary_docs),
               'primary_oof':oof,'official_dev':dev_scores,'wide_oof_folds':wide_folds,
               'private_score':None,'target_f1':.55,
               'note':'0.55 is a target, never a guaranteed private score.'}
    u.write_json(ROOT/'selection.json',selection)
    log('SELECTED',chosen,'primary OOF',None if not oof else round(oof[chosen]['macro_supported'],4),
        'official dev',round(dev_scores[chosen]['macro_supported'],4),'wide branch',use_wide)


def package():
    train,dev,test=u.all_data()
    vocab=u.labels()
    selection=u.read_json(ROOT/'selection.json')
    mode=selection['selected_mode']
    primary=torch.load(ROOT/'runs'/'03'/'full'/'test_scores.pt',map_location='cpu',weights_only=False)
    scores=primary
    if selection['use_wide']:
        other=torch.load(ROOT/'runs'/'01'/'full'/'test_scores.pt',map_location='cpu',weights_only=False)
        if set(other)!=set(primary):raise RuntimeError('Ensemble document IDs differ')
        scores={}
        for did in primary:
            scores[did]={key:.7*primary[did][key]+.3*other[did][key] for key in ('raw','pair')}
            scores[did]['transition']=primary[did]['transition']
    index=candidate(scores,mode)
    prediction={did:[vocab[int(y)] for y in labels] for did,labels in index.items()}
    expected={d['id']:len(d['text']) for d in test}
    if set(prediction)!=set(expected) or any(len(prediction[k])!=n for k,n in expected.items()):
        raise RuntimeError('Submission IDs/sentence counts invalid')
    u.write_json(ROOT/'prediction.json',prediction)
    temp=SUBMISSION.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT/'prediction.json','prediction.json')
    os.replace(temp,SUBMISSION)
    with zipfile.ZipFile(SUBMISSION) as z:
        if z.namelist()!=['prediction.json'] or json.loads(z.read('prediction.json'))!=prediction:
            raise RuntimeError('Submission zip failed round trip')
    shutil.copy2(SUBMISSION,WORK/'submission.zip')
    shutil.copy2(SUBMISSION,ROOT/'submission.zip')
    status='READY_FOR_REVIEW' if selection['primary_oof_documents']>=32 else 'PROVISIONAL_REVIEW_OOF_INCOMPLETE'
    u.write_json(ROOT/'report.json',{'status':status,'submission':str(SUBMISSION),
        'submission_sha256':u.sha(SUBMISSION),'artifact':str(ARTIFACT),
        'selected_mode':mode,'use_wide':selection['use_wide'],
        'primary_oof_documents':selection['primary_oof_documents'],
        'primary_oof_macro_f1':None if not selection['primary_oof'] else selection['primary_oof'][mode]['macro_supported'],
        'official_dev_macro_f1':selection['official_dev'][mode]['macro_supported'],
        'test_documents':len(test),'test_sentences':sum(expected.values()),'private_score':None,
        'target_f1':.55,'caution':'Score target cannot be guaranteed without private evaluation.'})
    log('PACKAGED',SUBMISSION,'official development F1',round(selection['official_dev'][mode]['macro_supported'],4))


def snapshot():
    u.snapshot()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','prepare-01','train','infer','select','probe-wide','package','snapshot'))
    parser.add_argument('--branch',choices=('03','01'),default='03')
    parser.add_argument('--stage',default='full')
    parser.add_argument('--chunk-epochs',type=int,default=2)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='prepare-01':prepare_branch01()
    elif args.action=='train':train_stage(args.branch,args.stage,args.chunk_epochs)
    elif args.action=='infer':infer_full(args.branch)
    elif args.action=='select':select()
    elif args.action=='probe-wide':probe_wide()
    elif args.action=='package':package()
    elif args.action=='snapshot':snapshot()


if __name__=='__main__':main()
