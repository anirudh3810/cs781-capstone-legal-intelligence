"""CS781 Phase 5 LSI: six Phase-1-based, resumable multi-label experiments."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

import cs781_phase5_utils as u

WORK = Path('/kaggle/working')
ROOT = WORK/'CS781_Capstone/phase5_lsi'
ARTIFACT = WORK/'CS781_Phase5_All_Artifacts.zip'
u.ROOT, u.ARTIFACT = ROOT, ARTIFACT
REPO = 'Exploration-Lab/CS-781-Capstone'
SPLITS = {'train':'Train/train.json','dev':'Train/dev.json','test':'Test/test.json'}
SEED = 781505
SLOTS = {
    '01': {'base':'03','method':'partial','epochs':7,'lr':1.4e-5},
    '02': {'base':'01','method':'full','epochs':6,'lr':8e-6},
    '03': {'base':'04','method':'lorma','epochs':7,'lr':5e-5},
    '04': {'base':'02','method':'adapters','epochs':7,'lr':5e-5},
    '05': {'base':'03','method':'adalora','epochs':7,'lr':5e-5},
    '06': {'base':'03','method':'distill','epochs':7,'lr':1.2e-5},
}
MAX_TOKENS = 512
TRAIN_CHUNKS = 4
INFER_CHUNKS = 12
OVERLAP = 64
CASE_BATCH = 2
ACCUM = 2
PATIENCE = 2
THRESHOLDS = np.arange(.10,.701,.05).round(3).tolist()
CITATION = re.compile(r'\b(?:section|sections|sec\.?|article|articles|act|acts|rule|rules|code|constitution|statute|statutory)\b',re.I)


def log(*items):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),*items,flush=True)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32-1));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def stage_official():
    dest=ROOT/'official'
    if all((dest/f'{name}.json').is_file() for name in SPLITS) and (dest/'provenance.json').is_file():
        record=u.read_json(dest/'provenance.json')
        if all(record['sha256'][name]==u.sha(dest/f'{name}.json') for name in SPLITS):return
        raise RuntimeError('Phase 5 data hash changed after preparation')
    # Offline Kaggle Inputs are accepted when all three official basenames are unique.
    local={name:u.find_one(f'{name}.json',required=False) for name in SPLITS}
    if all(local.values()) and len(set(local.values()))==3:
        dest.mkdir(parents=True,exist_ok=True)
        for name,path in local.items():shutil.copy2(path,dest/f'{name}.json')
        provenance={'source':'attached_official_files','paths':{k:str(v) for k,v in local.items()}}
    else:
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
            if not token:raise RuntimeError('HF_TOKEN is empty')
        except Exception as exc:
            raise RuntimeError('Attach all three official Phase 5 JSON files, or enable Internet and Kaggle Secret HF_TOKEN') from exc
        try:
            from huggingface_hub import HfApi,hf_hub_download
        except ImportError:
            subprocess.check_call([sys.executable,'-m','pip','install','-q','huggingface_hub>=0.24,<2'])
            from huggingface_hub import HfApi,hf_hub_download
        api=HfApi()
        try:
            revision=api.repo_info(REPO,repo_type='dataset',token=token).sha
            files=api.list_repo_files(REPO,repo_type='dataset',revision=revision,token=token)
        except Exception as exc:
            raise RuntimeError('Cannot access gated official Phase 5 data; check release, access, token, Internet') from exc
        chosen={}
        for name,suffix in SPLITS.items():
            hits=[p for p in files if p.lower().endswith(('phase_5_lsi/'+suffix).lower())]
            if len(hits)!=1:raise RuntimeError(f'Expected one Phase 5 {suffix}; found {hits}')
            chosen[name]=hits[0]
        dest.mkdir(parents=True,exist_ok=True)
        for name,path in chosen.items():
            cached=Path(hf_hub_download(repo_id=REPO,repo_type='dataset',filename=path,
                         revision=revision,token=token,cache_dir=str(ROOT/'hf_cache')))
            with tempfile.NamedTemporaryFile(dir=dest,delete=False) as tmp:
                temporary=Path(tmp.name)
                with cached.open('rb') as src:shutil.copyfileobj(src,tmp,4<<20)
            os.replace(temporary,dest/f'{name}.json')
        provenance={'source':REPO,'revision':revision,'paths':chosen}
        del token
    provenance['sha256']={name:u.sha(dest/f'{name}.json') for name in SPLITS}
    u.write_json(dest/'provenance.json',provenance)


def parse_split(name):
    value=u.read_json(ROOT/'official'/f'{name}.json')
    if isinstance(value,dict):
        for key in ('data','examples','cases'):
            if isinstance(value.get(key),list):value=value[key];break
        else:value=[{'id':k,**(v if isinstance(v,dict) else {'text':v})} for k,v in value.items()]
    if not isinstance(value,list) or not value:raise ValueError(f'{name}: expected a nonempty case list')
    rows=[]
    for i,item in enumerate(value):
        if not isinstance(item,dict):raise ValueError(f'{name} row {i} is not an object')
        case_id=item.get('id',item.get('case_id'))
        content=item.get('text',item.get('case_text'))
        if isinstance(content,list) and all(isinstance(s,str) for s in content):content='\n'.join(content)
        if type(case_id) not in (str,int) or str(case_id)=='' or not isinstance(content,str) or not content.strip():
            raise ValueError(f'{name} row {i}: invalid ID or text')
        label_keys=[key for key in ('labels','label','statutes','statute_ids') if key in item]
        if name=='test':
            if any(item[key] is not None for key in label_keys):raise ValueError('Test labels are present; stop')
            labels=None
        else:
            if len(label_keys)!=1:raise ValueError(f'{name} row {i}: expected exactly one label-list field')
            labels=item[label_keys[0]]
            if not isinstance(labels,list) or any(type(x) is not int or x<0 for x in labels):
                raise ValueError(f'{name} row {i}: labels must be a nonnegative integer list')
            if len(labels)!=len(set(labels)):raise ValueError(f'{name} row {i}: repeated statute ID')
            labels=sorted(labels)
        rows.append({'id':f'{name}::row::{i:08d}','source_id':case_id,'text':content,'labels':labels})
    if name=='test':
        keys=[(type(r['source_id']).__name__,str(r['source_id'])) for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('Repeated official test IDs cannot form an exact submission')
    return rows


def data():
    return tuple(parse_split(name) for name in ('train','dev','test'))


def text_hash(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode('utf8')).hexdigest()


def case_groups(rows,tie_id=True):
    parent=list(range(len(rows))); seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        keys=[('text',text_hash(row['text']))]
        if tie_id:keys.append(('id',type(row['source_id']).__name__,str(row['source_id'])))
        for key in keys:
            if key in seen:parent[root(i)]=root(seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[root(i)].append(row)
    return list(groups.values())


def split_groups(train):
    groups=case_groups(train,True);mode='id_and_exact_text'
    if len(groups)<9 or max(map(len,groups))>max(50,len(train)//5):
        groups=case_groups(train,False);mode='exact_text_only'
    counts=Counter(label for row in train for label in row['labels'])
    rng=random.Random(SEED)
    rng.shuffle(groups)
    groups.sort(key=lambda group:(min((counts[l] for row in group for l in row['labels']),default=10**9),-len(group)))
    # Greedy iterative assignment gives rare positive groups priority while targeting 15% outer/10% inner.
    bins=[[],[],[]];sizes=[0,0,0];label_counts=[Counter(),Counter(),Counter()]
    targets=[.75,.10,.15]
    for group in groups:
        group_labels=Counter(l for row in group for l in row['labels'])
        def cost(k):
            size_after=sizes[k]+len(group)
            size_cost=max(0,size_after-targets[k]*len(train))/(targets[k]*len(train)+1)
            rare_cost=sum((label_counts[k][l]+v)/(max(1,counts[l])*targets[k]+1) for l,v in group_labels.items())/max(1,len(group_labels))
            return 2*size_cost+rare_cost+.001*k
        k=min(range(3),key=cost)
        bins[k].extend(group);sizes[k]+=len(group);label_counts[k].update(group_labels)
    if min(map(len,bins))==0:raise ValueError('Could not form grouped train/inner/outer splits')
    mapping={row['id']:k for k,bin_rows in enumerate(bins) for row in bin_rows}
    if len(mapping)!=len(train):raise RuntimeError('Grouped split lost rows')
    return mapping,mode


def prepare(require_cuda=True):
    ROOT.mkdir(parents=True,exist_ok=True)
    u.restore_artifact()
    if require_cuda and torch.cuda.device_count()!=2:raise RuntimeError('Enable the two-GPU Kaggle accelerator')
    for base in ('01','02','03','04'):u.stage_base(base)
    stage_official()
    train,dev,test=data()
    labels=sorted(set(l for row in train+dev for l in row['labels']))
    if not labels:raise ValueError('No statute IDs in labeled splits')
    mapping,group_mode=split_groups(train)
    config={'version':1,'seed':SEED,'source_sha256':u.sha(__file__),
            'official_sha256':{name:u.sha(ROOT/'official'/f'{name}.json') for name in SPLITS},
            'base_sha256':{b:u.read_json(ROOT/'bases'/b/'provenance.json')['checkpoint_sha256'] for b in ('01','02','03','04')},
            'labels':labels,'split':mapping,'group_mode':group_mode,'slots':SLOTS,
            'max_tokens':MAX_TOKENS,'train_chunks':TRAIN_CHUNKS,'infer_chunks':INFER_CHUNKS,
            'overlap':OVERLAP,'case_batch':CASE_BATCH,'accum':ACCUM}
    target=ROOT/'run_config.json'
    if target.is_file() and u.read_json(target)!=config:
        raise RuntimeError('Recovery source/data/config differs from this notebook')
    u.write_json(target,config)
    shutil.copy2(__file__,ROOT/'lsi_runner.py')
    shutil.copy2(Path(u.__file__),ROOT/'phase5_utils.py')
    counts=Counter(l for row in train for l in row['labels'])
    audit={'counts':{name:len(rows) for name,rows in zip(SPLITS,(train,dev,test))},
           'label_count':len(labels),'train_positive_counts':dict(counts),
           'dev_only_labels':sorted(set(l for row in dev for l in row['labels'])-set(counts)),
           'labels_per_document':{name:{'median':float(np.median([len(r['labels']) for r in rows])),
                'p95':float(np.percentile([len(r['labels']) for r in rows],95))}
                for name,rows in [('train',train),('dev',dev)]},
           'character_lengths':{name:{'median':int(np.median([len(r['text']) for r in rows])),
                'p95':int(np.percentile([len(r['text']) for r in rows],95))}
                for name,rows in [('train',train),('dev',dev),('test',test)]},
           'duplicate_texts':{name:len(rows)-len({text_hash(r['text']) for r in rows})
                for name,rows in [('train',train),('dev',dev),('test',test)]},
           'group_mode':group_mode,'split_counts':dict(Counter(mapping.values())),
           'private_score':None}
    u.write_json(ROOT/'data_audit.json',audit)
    log('PREPARED',audit['counts'],'labels',len(labels),'groups',group_mode)


def tokenize_base(base_id):
    target=ROOT/'hf_cache'/f'tokens_{base_id}.pt'
    if target.is_file():return
    _,tokenizer=u.load_base(base_id)
    train,dev,test=data();encoded={};signals={}
    for row in train+dev+test:
        result=tokenizer.backend.encode(row['text'],add_special_tokens=False)
        encoded[row['id']]=result.ids
        starts=[a for a,b in result.offsets]
        signals[row['id']]=[bisect.bisect_left(starts,m.start()) for m in CITATION.finditer(row['text'])]
    marker=tokenizer.backend.encode(' [STATUTE IDENTIFICATION]',add_special_tokens=False).ids
    if not marker or len(marker)>=MAX_TOKENS-64:raise RuntimeError('Invalid classification marker')
    target.parent.mkdir(parents=True,exist_ok=True)
    u.save_torch(target,{'tokens':encoded,'signals':signals,'marker':marker,'pad':int(tokenizer.pad_token_id)})
    audit={}
    for name,rows in [('train',train),('dev',dev),('test',test)]:
        lengths=[len(encoded[row['id']]) for row in rows]
        audit[name]={'median':int(np.median(lengths)),'p95':int(np.percentile(lengths,95)),
                     'max':max(lengths),'over_train_budget':sum(n>TRAIN_CHUNKS*(MAX_TOKENS-OVERLAP) for n in lengths),
                     'over_infer_budget':sum(n>INFER_CHUNKS*(MAX_TOKENS-OVERLAP) for n in lengths)}
    u.write_json(ROOT/f'token_audit_{base_id}.json',audit)
    log('TOKENIZED',base_id,len(encoded),'documents')


def chunk_starts(tokens,signals,width,budget,training,epoch,key):
    n=len(tokens)
    if n<=width:return [0]
    last=n-width
    candidates=list(range(0,last+1,max(1,width-OVERLAP)))
    if candidates[-1]!=last:candidates.append(last)
    if len(candidates)<=budget:return candidates
    selected={0,last}
    spaced=max(0,(budget-2)//3)
    for j in range(1,spaced+1):
        target=round(last*j/(spaced+1))
        selected.add(min(candidates,key=lambda x:abs(x-target)))
    rng=random.Random(SEED+epoch*104729+int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
    ranked=[]
    for start in candidates:
        citations=bisect.bisect_left(signals,start+width)-bisect.bisect_left(signals,start)
        ranked.append((citations+(rng.random()*.3 if training else 0),start))
    ranked.sort(reverse=True)
    for _,start in ranked:
        if len(selected)>=budget:break
        if all(abs(start-x)>=width//2 for x in selected):selected.add(start)
    if len(selected)<budget:
        for _,start in ranked:
            selected.add(start)
            if len(selected)>=budget:break
    return sorted(selected)


def document_batch(rows,cache,device,training=False,epoch=0):
    marker=cache['marker'];width=MAX_TOKENS-len(marker)
    flat=[];positions=[];counts=[]
    for row in rows:
        tokens=cache['tokens'][row['id']]
        starts=chunk_starts(tokens,cache['signals'][row['id']],width,
                            TRAIN_CHUNKS if training else INFER_CHUNKS,training,epoch,row['id'])
        counts.append(len(starts))
        for start in starts:
            flat.append((tokens[start:start+width] or [cache['pad']])+marker)
            positions.append(min(31,round(31*start/max(1,len(tokens)-width))))
    length=max(map(len,flat));ids=torch.full((len(flat),length),cache['pad'],dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,token_ids in enumerate(flat):
        ids[i,:len(token_ids)]=torch.tensor(token_ids,dtype=torch.long)
        mask[i,:len(token_ids)]=True
    pos=torch.tensor(positions,dtype=torch.long)
    if device.type=='cuda':
        ids=ids.pin_memory().to(device,non_blocking=True)
        mask=mask.pin_memory().to(device,non_blocking=True)
        pos=pos.pin_memory().to(device,non_blocking=True)
    return ids,mask,pos,counts


class BottleneckAdapter(nn.Module):
    def __init__(self,block,width,rank=64):
        super().__init__()
        self.block=block
        self.down=nn.Linear(width,rank)
        self.up=nn.Linear(rank,width)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,*args):
        result=self.block(*args)
        return result+self.up(F.gelu(self.down(result)))


class AdaLoRALinear(nn.Module):
    def __init__(self,base,rank=8):
        super().__init__()
        self.base=base
        self.a=nn.Linear(base.in_features,rank,bias=False)
        self.b=nn.Linear(rank,base.out_features,bias=False)
        self.s=nn.Parameter(torch.ones(rank)*.01)
        self.register_buffer('mask',torch.ones(rank))
        self.register_buffer('importance',torch.zeros(rank))
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5))
        nn.init.normal_(self.b.weight,std=.01)
        self.scale=1.
    def forward(self,x):
        return self.base(x)+self.b(self.a(x)*(self.s*self.mask))*self.scale
    @torch.no_grad()
    def record_importance(self):
        if self.s.grad is not None:
            self.importance.mul_(.9).add_((self.s*self.s.grad).abs(),alpha=.1)


@torch.no_grad()
def allocate_ada_ranks(model,epoch,total_epochs):
    modules=[m for m in model.modules() if isinstance(m,AdaLoRALinear)]
    if not modules:return
    initial=sum(m.mask.numel() for m in modules)
    target=max(2*len(modules),round(initial*(1.-.5*epoch/max(1,total_epochs))))
    ranking=[]
    for i,m in enumerate(modules):
        for j,value in enumerate(m.importance.tolist()):ranking.append((value,i,j))
    ranking.sort(reverse=True)
    kept={(i,j) for _,i,j in ranking[:target]}
    for i,m in enumerate(modules):
        m.mask.copy_(torch.tensor([float((i,j) in kept) for j in range(m.mask.numel())],
                                  device=m.mask.device,dtype=m.mask.dtype))


class StatuteClassifier(nn.Module):
    def __init__(self,base,base_id,method,label_count):
        super().__init__()
        self.base,self.base_id,self.method=base,base_id,method
        for p in base.parameters():p.requires_grad_(False)
        self.first_trainable=0
        if base_id=='01':width=int(base.config.hidden_size)
        elif base_id=='04':width=int(base.config['hidden_size'])
        else:width=int(base.hidden_size)
        if method=='full':
            for p in base.parameters():p.requires_grad_(True)
        elif method in ('partial','distill'):
            self.first_trainable=len(base.blocks)-(12 if method=='partial' else 8)
            for block in base.blocks[self.first_trainable:]:
                for p in block.parameters():p.requires_grad_(True)
            for p in base.final_norm.parameters():p.requires_grad_(True)
        elif method=='lorma':
            count=0
            for name,module in list(base.named_modules()):
                if (name.startswith('blocks.') and isinstance(module,nn.Linear) and
                    name.split('.')[-1] in ('q_proj','v_proj','o_proj','gate_proj','up_proj')):
                    parent=base.get_submodule(name.rsplit('.',1)[0])
                    setattr(parent,name.rsplit('.',1)[-1],u.LoRMA(module,8,8));count+=1
            if count<14:raise RuntimeError('LORMA failed to attach to base 04')
            base.set_gradient_checkpointing(True)
        elif method=='adapters':
            for i,block in enumerate(base.encoder_blocks):
                base.encoder_blocks[i]=BottleneckAdapter(block,width)
            base.set_gradient_checkpointing(True)
        elif method=='adalora':
            count=0
            for name,module in list(base.named_modules()):
                if (name.startswith('blocks.') and isinstance(module,nn.Linear) and
                    name.split('.')[-1] in ('query','value','output','gate','up')):
                    parent=base.get_submodule(name.rsplit('.',1)[0])
                    setattr(parent,name.rsplit('.',1)[-1],AdaLoRALinear(module));count+=1
            if count<24:raise RuntimeError('AdaLoRA failed to attach to base 03')
        else:raise ValueError(method)
        if base_id=='04':base.set_gradient_checkpointing(True)
        self.project=nn.Sequential(nn.Linear(width,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1))
        self.position=nn.Embedding(32,256)
        self.context=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256,4,512,dropout=.1,batch_first=True,norm_first=True),
            num_layers=1,enable_nested_tensor=False)
        self.gate=nn.Linear(256,1)
        self.global_head=nn.Sequential(nn.LayerNorm(768),nn.Dropout(.2),nn.Linear(768,256),
                                       nn.GELU(),nn.Dropout(.2),nn.Linear(256,label_count))
        self.evidence_head=nn.Linear(256,label_count)
        self.evidence_scale=nn.Parameter(torch.tensor(0.))

    def chunk_features(self,ids,mask):
        base=self.base
        if self.base_id=='01':
            positions=torch.arange(ids.size(1),device=ids.device)
            states=base.embedding_dropout(base.token_embedding(ids)+base.position_embedding(positions)[None])
            for block in base.blocks:
                states=checkpoint(block,states,use_reentrant=False) if self.training else block(states)
            states=base.final_norm(states)
        elif self.base_id=='02':
            states,_=base.encode(ids,mask)
        elif self.base_id=='03':
            states=base.embedding_dropout(base.token_embedding(ids)*math.sqrt(base.hidden_size))
            visibility=base._prefix_hybrid_visibility(mask,mask.sum(1))
            cosine,sine=base.rotary(ids.size(1),states.dtype)
            for i,block in enumerate(base.blocks):
                if i<self.first_trainable and self.method in ('partial','distill'):
                    with torch.no_grad():states=block(states,visibility,cosine,sine)
                else:
                    states=checkpoint(block,states,visibility,cosine,sine,use_reentrant=False) if self.training else block(states,visibility,cosine,sine)
            states=base.final_norm(states)
        else:
            states=base.embedding(ids)
            for block in base.blocks:
                states=checkpoint(block,states,mask,use_reentrant=False) if self.training else block(states,mask)
            states=base.final_norm(states)
        if self.base_id in ('01','04'):
            end=mask.sum(1).long()-1
            return states[torch.arange(ids.size(0),device=ids.device),end]
        return (states*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

    def forward(self,ids,mask,position,counts):
        chunks=self.project(self.chunk_features(ids,mask))+self.position(position)
        groups=torch.split(chunks,counts)
        padded=nn.utils.rnn.pad_sequence(groups,batch_first=True)
        valid=torch.arange(padded.size(1),device=padded.device)[None]<torch.tensor(counts,device=padded.device)[:,None]
        contextual=self.context(padded,src_key_padding_mask=~valid)
        weights=self.gate(contextual).squeeze(-1).masked_fill(~valid,-1e4).softmax(1)
        pooled=(contextual*weights.unsqueeze(-1)).sum(1)
        mean=(contextual*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True)
        maximum=contextual.masked_fill(~valid.unsqueeze(-1),-1e4).max(1).values
        global_logits=self.global_head(torch.cat([pooled,mean,maximum],-1))
        evidence=self.evidence_head(contextual).masked_fill(~valid.unsqueeze(-1),-1e4).max(1).values
        return global_logits+torch.sigmoid(self.evidence_scale)*evidence


def model_for(slot,device):
    spec=SLOTS[slot];base,tokenizer=u.load_base(spec['base'])
    labels=u.read_json(ROOT/'run_config.json')['labels']
    return StatuteClassifier(base,spec['base'],spec['method'],len(labels)).to(device),tokenizer


def trainable_state(model):
    keys={name for name,p in model.named_parameters() if p.requires_grad}
    buffers={name for name,_ in model.named_buffers() if name.endswith(('.mask','.importance'))}
    state=model.state_dict()
    return {name:state[name].detach().cpu().clone() for name in keys|buffers}


def load_state(model,state):
    current=model.state_dict();unknown=set(state)-set(current)
    if unknown:raise RuntimeError(f'Checkpoint has unknown tensors: {sorted(unknown)[:5]}')
    for name,value in state.items():
        if current[name].shape!=value.shape:raise RuntimeError(f'Wrong checkpoint shape: {name}')
        current[name].copy_(value)


def targets(rows,label_to_index,device):
    y=torch.zeros((len(rows),len(label_to_index)),dtype=torch.float32,device=device)
    for i,row in enumerate(rows):
        for label in row['labels']:y[i,label_to_index[label]]=1.
    return y


def metric(probabilities,rows,labels,thresholds):
    probs=np.asarray(probabilities,dtype=np.float32)
    if probs.shape!=(len(rows),len(labels)):raise ValueError('Wrong probability matrix shape')
    threshold=np.asarray(thresholds,dtype=np.float32)
    if threshold.shape!=(len(labels),):raise ValueError('Wrong threshold vector')
    index={label:i for i,label in enumerate(labels)}
    gold=np.zeros(probs.shape,dtype=bool)
    for i,row in enumerate(rows):
        for label in row['labels']:gold[i,index[label]]=True
    pred=probs>=threshold[None,:]
    tp=(pred&gold).sum(0);fp=(pred&~gold).sum(0);fn=(~pred&gold).sum(0)
    per_label=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(labels)),where=(2*tp+fp+fn)>0)
    supported=gold.sum(0)>0
    total_tp,total_fp,total_fn=int(tp.sum()),int(fp.sum()),int(fn.sum())
    return {'macro_f1':float(per_label.mean()),
            'reference_label_macro_f1':float(per_label[supported].mean()) if supported.any() else 0.,
            'micro_f1':float(2*total_tp/(2*total_tp+total_fp+total_fn)) if 2*total_tp+total_fp+total_fn else 0.,
            'subset_accuracy':float(np.mean(np.all(pred==gold,axis=1))),
            'mean_gold_cardinality':float(gold.sum(1).mean()),
            'mean_pred_cardinality':float(pred.sum(1).mean()),
            'supported_labels':int(supported.sum()),'all_labels':len(labels),
            'per_label':{str(label):{'gold':int(gold[:,j].sum()),'predicted':int(pred[:,j].sum()),
                       'tp':int(tp[j]),'fp':int(fp[j]),'fn':int(fn[j]),'f1':float(per_label[j])}
                       for j,label in enumerate(labels)}}


def fit_thresholds(probabilities,rows,labels,fit_support):
    probs=np.asarray(probabilities,dtype=np.float32)
    score_list=[]
    for value in THRESHOLDS:
        observed=metric(probs,rows,labels,[value]*len(labels))
        score_list.append((observed['macro_f1'],-abs(value-.35),value))
    global_threshold=max(score_list)[2]
    threshold=np.full(len(labels),global_threshold,dtype=np.float32)
    index={label:i for i,label in enumerate(labels)}
    gold=np.zeros(probs.shape,dtype=bool)
    for i,row in enumerate(rows):
        for label in row['labels']:gold[i,index[label]]=True
    for j,label in enumerate(labels):
        n=int(gold[:,j].sum())
        if fit_support.get(label,0)==0:
            threshold[j]=1.1
        elif n>=20 and len(rows)-n>=20:
            candidates=[]
            for value in THRESHOLDS:
                pred=probs[:,j]>=value
                tp=int((pred&gold[:,j]).sum());fp=int((pred&~gold[:,j]).sum());fn=int((~pred&gold[:,j]).sum())
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
                candidates.append((f1,-abs(value-global_threshold),value))
            local=max(candidates)[2]
            strength=n/(n+100.)
            threshold[j]=np.clip(global_threshold+strength*(local-global_threshold),
                                 global_threshold-.15,global_threshold+.15)
    return threshold.tolist(),{'global_threshold':global_threshold,
             'inner_selected_metric':metric(probs,rows,labels,threshold)}


def stage_rows(stage):
    train,dev,test=data()
    if stage=='final':return train,dev,test
    if stage!='holdout':raise ValueError(stage)
    mapping=u.read_json(ROOT/'run_config.json')['split']
    parts=[[row for row in train if mapping[row['id']]==k] for k in range(3)]
    if any(not part for part in parts):raise RuntimeError('Empty grouped validation part')
    return tuple(parts)


def epoch_batches(rows,cache,epoch,rank,world):
    shuffled=list(rows);random.Random(SEED+epoch*7919).shuffle(shuffled)
    ordered=[]
    for start in range(0,len(shuffled),64):
        block=shuffled[start:start+64]
        block.sort(key=lambda row:len(cache['tokens'][row['id']]))
        ordered.extend(block)
    quantum=world*CASE_BATCH
    if not ordered:raise ValueError('Cannot train on an empty fitting split')
    original=list(ordered)
    while len(ordered)%quantum:ordered.append(original[(len(ordered)-len(original))%len(original)])
    batches=[ordered[i:i+CASE_BATCH] for i in range(0,len(ordered),CASE_BATCH)]
    return batches[rank::world]


def optimizer_for(model,spec,updates):
    backbone=[];head=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        (backbone if name.startswith('base.') else head).append(param)
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':spec['lr']},
                                 {'params':head,'lr':2e-4}],weight_decay=.01,
                                fused=torch.cuda.is_available())
    warmup=max(1,round(.05*updates))
    def schedule(step):
        if step<warmup:return max(.05,(step+1)/warmup)
        fraction=(step-warmup)/max(1,updates-warmup)
        return .10+.90*.5*(1+math.cos(math.pi*min(1.,fraction)))
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group('nccl',init_method='env://')
    rank=dist.get_rank();world=dist.get_world_size()
    if world!=2 or torch.cuda.device_count()!=2:raise RuntimeError('Launch with torchrun --nproc_per_node=2 on two GPUs')
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    return rank,world,device


def predict_rows(model,rows,cache,rank,world,device,tag,training_budget=False):
    model.eval();local={};shard=rows[rank::world]
    with torch.no_grad():
        for start in range(0,len(shard),CASE_BATCH):
            batch=shard[start:start+CASE_BATCH]
            ids,mask,pos,counts=document_batch(batch,cache,device,training_budget,1)
            with torch.autocast('cuda',dtype=torch.float16):values=model(ids,mask,pos,counts)
            for row,value in zip(batch,values.float().cpu().tolist()):local[row['id']]=value
    temporary=ROOT/'hf_cache'/'pred_parts';temporary.mkdir(parents=True,exist_ok=True)
    path=temporary/f'{tag}_rank{rank}.pt'
    u.save_torch(path,local)
    dist.barrier()
    merged=None
    if rank==0:
        merged={}
        for i in range(world):
            part=torch.load(temporary/f'{tag}_rank{i}.pt',map_location='cpu',weights_only=False)
            if set(merged)&set(part):raise RuntimeError('Duplicate prediction rows across GPU ranks')
            merged.update(part)
        if set(merged)!={row['id'] for row in rows}:raise RuntimeError('Prediction IDs incomplete')
        for i in range(world):(temporary/f'{tag}_rank{i}.pt').unlink()
    dist.barrier()
    return merged


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


def save_latest(directory,model,optimizer,scheduler,scaler,epoch,cursor,best,best_epoch,thresholds,rank,world):
    states=[None]*world
    dist.all_gather_object(states,rng_state())
    if rank==0:
        u.save_torch(directory/'latest.pt',{'model':trainable_state(model),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'epoch':epoch,'cursor':cursor,
            'best':best,'best_epoch':best_epoch,'thresholds':thresholds,'rng':states})
    dist.barrier()


def teacher_stage(stage,max_minutes=20.):
    """Resumable, fold-safe teacher logits for slot 06; snapshot between segments."""
    rank,world,device=init_dist()
    try:
        directory=ROOT/'teacher'/stage;directory.mkdir(parents=True,exist_ok=True)
        status_path=directory/'status.json'
        if status_path.is_file() and u.read_json(status_path).get('complete'):
            if rank==0:log('REUSE TEACHER',stage)
            dist.barrier();return
        fit,_,_=stage_rows(stage)
        selected_path=ROOT/'runs'/'02'/stage/'best.pt'
        if not selected_path.is_file():raise RuntimeError('Train slot 02 first for fold-safe distillation')
        model,_=model_for('02',device)
        load_state(model,torch.load(selected_path,map_location='cpu',weights_only=False)['model'])
        model.eval();cache=torch.load(ROOT/'hf_cache'/'tokens_01.pt',map_location='cpu',weights_only=False)
        shard=fit[rank::world]
        part_path=directory/f'partial_rank{rank}.pt'
        local=torch.load(part_path,map_location='cpu',weights_only=False) if part_path.is_file() else {}
        cursor=u.read_json(status_path)['cursor'] if status_path.is_file() else 0
        iterations=math.ceil(max(len(fit[i::world]) for i in range(world))/CASE_BATCH)
        started=time.monotonic()
        for i in range(cursor,iterations):
            batch=shard[i*CASE_BATCH:(i+1)*CASE_BATCH]
            if batch:
                ids,mask,pos,counts=document_batch(batch,cache,device,True,1)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    values=model(ids,mask,pos,counts)
                for row,value in zip(batch,values.float().cpu().tolist()):local[row['id']]=value
            stop=torch.tensor([int(time.monotonic()-started>=max_minutes*60 and i<iterations-1)
                               if rank==0 else 0],device=device)
            dist.broadcast(stop,0)
            if stop.item():
                u.save_torch(part_path,local)
                dist.barrier()
                if rank==0:
                    u.write_json(status_path,{'complete':False,'cursor':i+1,'fit_cases':len(fit)})
                    log('PAUSED TEACHER',stage,'batch',i+1,'/',iterations)
                dist.barrier();return
        u.save_torch(part_path,local)
        dist.barrier()
        if rank==0:
            merged={}
            for i in range(world):
                part=torch.load(directory/f'partial_rank{i}.pt',map_location='cpu',weights_only=False)
                if set(merged)&set(part):raise RuntimeError('Overlapping teacher shards')
                merged.update(part)
            if set(merged)!={row['id'] for row in fit}:raise RuntimeError('Incomplete teacher logits')
            u.save_torch(directory/'logits.pt',merged)
            u.write_json(status_path,{'complete':True,'cursor':iterations,'fit_cases':len(fit),
                         'teacher_best_epoch':u.read_json(ROOT/'runs'/'02'/stage/'status.json')['best_epoch']})
            for i in range(world):(directory/f'partial_rank{i}.pt').unlink()
            log('COMPLETE TEACHER',stage,len(merged),'cases')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def train_stage(slot,stage,max_minutes=20.):
    rank,world,device=init_dist()
    try:
        directory=ROOT/'runs'/slot/stage;directory.mkdir(parents=True,exist_ok=True)
        status_path=directory/'status.json'
        if status_path.is_file() and u.read_json(status_path).get('complete'):
            if rank==0:log('REUSE',slot,stage)
            dist.barrier();return
        seed_all(SEED+int(slot)*1000+sum(map(ord,stage))+rank)
        fit,inner,outer=stage_rows(stage)
        labels=u.read_json(ROOT/'run_config.json')['labels']
        label_to_index={label:i for i,label in enumerate(labels)}
        support=Counter(label for row in fit for label in row['labels'])
        spec=SLOTS[slot]
        cache=torch.load(ROOT/'hf_cache'/f"tokens_{spec['base']}.pt",map_location='cpu',weights_only=False)
        teacher=None
        if slot=='06':
            teacher_path=ROOT/'teacher'/stage/'logits.pt'
            if not teacher_path.is_file():raise RuntimeError('Run the matched resumable teacher stage first')
            teacher=torch.load(teacher_path,map_location='cpu',weights_only=False)
            if set(teacher)!={row['id'] for row in fit}:raise RuntimeError('Teacher logits do not match fit IDs')
        model,_=model_for(slot,device)
        count_batches=len(epoch_batches(fit,cache,1,rank,world))
        updates=math.ceil(count_batches/ACCUM)*spec['epochs']
        optimizer,scheduler=optimizer_for(model,spec,updates)
        scaler=torch.amp.GradScaler('cuda')
        latest=directory/'latest.pt';history=u.read_json(directory/'history.json') if (directory/'history.json').is_file() else []
        epoch=1;cursor=0;best=-1.;best_epoch=0;thresholds=[.35]*len(labels)
        if latest.is_file():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            load_state(model,state['model'])
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            epoch=int(state['epoch']);cursor=int(state['cursor'])
            best=float(state['best']);best_epoch=int(state['best_epoch'])
            thresholds=list(state['thresholds']);restore_rng(state['rng'][rank])
        ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,
                find_unused_parameters=True,gradient_as_bucket_view=True)
        pos_weight=torch.tensor([max(1.,min(8.,math.sqrt(max(1,len(fit)-support.get(l,0))/max(1,support.get(l,0)))))
                                 for l in labels],dtype=torch.float32,device=device)
        started=time.monotonic()
        while epoch<=spec['epochs']:
            torch.cuda.reset_peak_memory_stats(device)
            batches=epoch_batches(fit,cache,epoch,rank,world)
            if cursor>=len(batches):raise RuntimeError('Saved cursor exceeds epoch batch count')
            ddp.train();optimizer.zero_grad(set_to_none=True)
            elapsed_train=0.;processed_tokens=0;steps=0;loss_sum=0.
            for i in range(cursor,len(batches)):
                batch=batches[i]
                ids,mask,pos,counts=document_batch(batch,cache,device,True,epoch)
                processed_tokens+=int(mask.sum())
                teacher_logits=(torch.tensor([teacher[row['id']] for row in batch],device=device,dtype=torch.float32)
                                if teacher is not None else None)
                sync=(i+1)%ACCUM==0 or i==len(batches)-1
                divisor=ACCUM if i+ACCUM<=len(batches) else len(batches)%ACCUM or ACCUM
                with (nullcontext() if sync else ddp.no_sync()):
                    with torch.autocast('cuda',dtype=torch.float16):
                        logits=ddp(ids,mask,pos,counts)
                        y=targets(batch,label_to_index,device)
                        hard=F.binary_cross_entropy_with_logits(logits.float(),y,pos_weight=pos_weight)
                        if teacher_logits is None:loss=hard/divisor
                        else:
                            temp=2.
                            soft=torch.sigmoid(teacher_logits/temp)
                            kd=F.binary_cross_entropy_with_logits(logits.float()/temp,soft)*temp**2
                            loss=(.80*hard+.20*kd)/divisor
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite Phase 5 loss')
                    scaler.scale(loss).backward()
                loss_sum+=float(loss.detach())*divisor
                if sync:
                    scaler.unscale_(optimizer)
                    for module in model.modules():
                        if isinstance(module,AdaLoRALinear):module.record_importance()
                    nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                    scaler.step(optimizer);scaler.update();scheduler.step()
                    optimizer.zero_grad(set_to_none=True);steps+=1
                    should_stop=torch.tensor([int(time.monotonic()-started>=max_minutes*60 and i<len(batches)-1)
                                              if rank==0 else 0],device=device)
                    dist.broadcast(should_stop,0)
                    if should_stop.item():
                        save_latest(directory,model,optimizer,scheduler,scaler,epoch,i+1,
                                    best,best_epoch,thresholds,rank,world)
                        if rank==0:
                            u.write_json(status_path,{'complete':False,'epoch':epoch,'cursor':i+1,
                                          'best_epoch':best_epoch,'checkpoint':'latest.pt'})
                            log('PAUSED',slot,stage,'epoch',epoch,'batch',i+1,'/',len(batches),
                                'minutes',round((time.monotonic()-started)/60,1))
                        dist.barrier();return
            if spec['method']=='adalora':allocate_ada_ranks(model,epoch,spec['epochs'])
            prediction=predict_rows(model,inner,cache,rank,world,device,f'{slot}_{stage}_inner')
            meta=[None]
            if rank==0:
                probs=np.asarray([torch.sigmoid(torch.tensor(prediction[row['id']])).numpy() for row in inner])
                thresholds_now,selection=fit_thresholds(probs,inner,labels,support)
                score=selection['inner_selected_metric']['macro_f1']
                if score>best+1e-6:
                    best=score;best_epoch=epoch;thresholds=thresholds_now
                    u.save_torch(directory/'best.pt',{'model':trainable_state(model),'epoch':epoch,
                                 'thresholds':thresholds,'inner_selected_metric':selection['inner_selected_metric'],
                                 'slot':slot,'stage':stage})
                record={'epoch':epoch,'inner_macro_f1_selected':score,'best_epoch':best_epoch,
                        'train_loss_rank0':loss_sum/max(1,len(batches)-cursor),
                        'rank0_optimizer_steps':steps,'rank0_tokens_per_second':round(processed_tokens/max(1,time.monotonic()-started),1),
                        'rank0_peak_gpu_gib':round(torch.cuda.max_memory_allocated(device)/2**30,2),
                        'elapsed_minutes':round((time.monotonic()-started)/60,1)}
                history.append(record);u.write_json(directory/'history.json',history)
                log('EPOCH',slot,stage,epoch,'inner macro',round(score,4),'best',round(best,4),
                    'tokens/s',record['rank0_tokens_per_second'],'peak GiB',record['rank0_peak_gpu_gib'])
                meta=[(best,best_epoch,thresholds)]
            dist.broadcast_object_list(meta,src=0)
            best,best_epoch,thresholds=meta[0]
            finish=(epoch>=3 and epoch-best_epoch>=PATIENCE) or epoch>=spec['epochs']
            if finish:
                chosen=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
                load_state(model,chosen['model'])
                final_inner=predict_rows(model,inner,cache,rank,world,device,f'{slot}_{stage}_selected_inner')
                final_outer=predict_rows(model,outer,cache,rank,world,device,f'{slot}_{stage}_outer')
                if rank==0:
                    u.save_torch(directory/'inner_logits.pt',final_inner)
                    u.save_torch(directory/('test_logits.pt' if stage=='final' else 'outer_logits.pt'),final_outer)
                    probabilities=np.asarray([[float(torch.sigmoid(torch.tensor(x))) for x in final_outer[row['id']]]
                                              for row in outer])
                    outer_metric=metric(probabilities,outer,labels,thresholds) if stage=='holdout' else None
                    u.write_json(status_path,{'complete':True,'best_epoch':best_epoch,'inner_macro_f1_selected':best,
                          'thresholds':thresholds,'outer_metric':outer_metric,'predicted_cases':len(outer),
                          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
                    latest.unlink(missing_ok=True)
                    log('COMPLETE',slot,stage,'heldout macro',None if outer_metric is None else round(outer_metric['macro_f1'],4))
                dist.barrier();return
            epoch+=1;cursor=0
            save_latest(directory,model,optimizer,scheduler,scaler,epoch,0,best,best_epoch,thresholds,rank,world)
            if rank==0:u.write_json(status_path,{'complete':False,'epoch':epoch,'cursor':0,'best_epoch':best_epoch})
            dist.barrier()
            # Return after each finished epoch so the parent notebook can rebuild the recovery ZIP.
            return
    finally:
        dist.destroy_process_group()


def complete(slot,stage):
    path=ROOT/'runs'/slot/stage/'status.json'
    return path.is_file() and bool(u.read_json(path).get('complete'))


def package_slot(slot):
    if not complete(slot,'holdout') or not complete(slot,'final'):
        raise RuntimeError(f'Slot {slot} requires both completed holdout and final stages')
    train,dev,test=data()
    labels=u.read_json(ROOT/'run_config.json')['labels']
    holdout=u.read_json(ROOT/'runs'/slot/'holdout'/'status.json')
    final=u.read_json(ROOT/'runs'/slot/'final'/'status.json')
    path=ROOT/'runs'/slot/'final'
    dev_logits=torch.load(path/'inner_logits.pt',map_location='cpu',weights_only=False)
    test_logits=torch.load(path/'test_logits.pt',map_location='cpu',weights_only=False)
    if set(dev_logits)!={row['id'] for row in dev} or set(test_logits)!={row['id'] for row in test}:
        raise RuntimeError('Saved final dev/test logits have wrong IDs')
    thresholds=final['thresholds']
    dev_probs=np.asarray([[float(torch.sigmoid(torch.tensor(x))) for x in dev_logits[row['id']]]
                          for row in dev])
    test_probs=np.asarray([[float(torch.sigmoid(torch.tensor(x))) for x in test_logits[row['id']]]
                           for row in test])
    dev_metric=metric(dev_probs,dev,labels,thresholds)
    pred=test_probs>=np.asarray(thresholds)[None,:]
    predictions=[{'id':row['source_id'],'pred':[label for label,flag in zip(labels,pred[i]) if flag]}
                 for i,row in enumerate(test)]
    if len(predictions)!=len(test) or [x['id'] for x in predictions]!=[x['source_id'] for x in test]:
        raise RuntimeError('Test ID order changed')
    allowed=set(labels)
    if any(any(type(x) is not int or x not in allowed for x in row['pred']) or
           len(row['pred'])!=len(set(row['pred'])) for row in predictions):
        raise RuntimeError('Invalid statute IDs in package')
    destination=ROOT/'submissions'/slot;destination.mkdir(parents=True,exist_ok=True)
    u.write_json(destination/'prediction.json',predictions)
    output=WORK/f'CS781_Phase5_Submission_{slot}.zip'
    temporary=output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(destination/'prediction.json','prediction.json')
    os.replace(temporary,output)
    with zipfile.ZipFile(output) as archive:
        if archive.namelist()!=['prediction.json'] or json.loads(archive.read('prediction.json'))!=predictions:
            raise RuntimeError('Submission ZIP round-trip failed')
    shutil.copy2(output,destination/'submission.zip')
    report={'slot':slot,'base':SLOTS[slot]['base'],'method':SLOTS[slot]['method'],
            'status':'READY_FOR_MANUAL_UPLOAD','holdout_metric':holdout['outer_metric'],
            'dev_metric_selected':dev_metric,'dev_score_is_selection_biased':True,
            'holdout_best_epoch':holdout['best_epoch'],'final_best_epoch':final['best_epoch'],
            'thresholds':thresholds,'trainable_parameters':final['trainable_parameters'],
            'test_cases':len(test),'test_predicted_cardinality':float(pred.sum(1).mean()),
            'test_empty_predictions':int((pred.sum(1)==0).sum()),
            'prediction_sha256':u.sha(destination/'prediction.json'),'zip_sha256':u.sha(output),
            'phase1_checkpoint_sha256':u.read_json(ROOT/'bases'/SLOTS[slot]['base']/'provenance.json')['checkpoint_sha256'],
            'data_sha256':u.read_json(ROOT/'run_config.json')['official_sha256'],
            'private_f1':None,'submission':str(output)}
    u.write_json(ROOT/'reports'/f'submission_{slot}.json',report)
    log('PACKAGED',slot,'holdout macro',round(holdout['outer_metric']['macro_f1'],4),
        'dev selected macro',round(dev_metric['macro_f1'],4),'empty test',report['test_empty_predictions'])


def final_report():
    reports=[]
    for slot in SLOTS:
        path=ROOT/'reports'/f'submission_{slot}.json'
        if not path.is_file():raise RuntimeError(f'Missing report for slot {slot}')
        reports.append(u.read_json(path))
    hashes=[r['prediction_sha256'] for r in reports]
    distinct=len(set(hashes))==len(hashes)
    ordered=sorted(reports,key=lambda row:row['holdout_metric']['macro_f1'],reverse=True)
    result={'status':'SIX_DISTINCT_PACKAGES_READY' if distinct else 'DUPLICATE_PREDICTIONS_REQUIRE_REPAIR',
            'count':len(reports),'distinct':distinct,'ranking':[{'slot':r['slot'],
              'holdout_macro_f1':r['holdout_metric']['macro_f1'],
              'dev_selected_macro_f1':r['dev_metric_selected']['macro_f1']} for r in ordered],
            'private_scores':None,'warning':'Official Phase 5 score is unknown until Codabench evaluates the ZIPs.',
            'artifact':str(ARTIFACT)}
    u.write_json(ROOT/'report.json',result)
    log('FINAL REPORT',result['status'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','tokenize','teacher','train','package','report','snapshot'))
    parser.add_argument('--slot',choices=tuple(SLOTS),default='01')
    parser.add_argument('--base',choices=('01','02','03','04'),default='01')
    parser.add_argument('--stage',choices=('holdout','final'),default='holdout')
    parser.add_argument('--max-minutes',type=float,default=20.)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='tokenize':tokenize_base(args.base)
    elif args.action=='teacher':teacher_stage(args.stage,args.max_minutes)
    elif args.action=='train':train_stage(args.slot,args.stage,args.max_minutes)
    elif args.action=='package':package_slot(args.slot)
    elif args.action=='report':final_report()
    elif args.action=='snapshot':u.snapshot()


if __name__=='__main__':main()
