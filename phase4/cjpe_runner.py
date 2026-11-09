"""CS781 Phase 4 CJPE: six resumable, two-GPU Phase-1 adaptations.

This source is embedded in CS781_Phase4_Six_Submissions_RunAll.ipynb.
Only the user's four Phase-1 checkpoints and official Phase-4 data are used.
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
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
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

import cs781_phase4_utils as u

WORK = Path('/kaggle/working')
ROOT = WORK/'CS781_Capstone/phase4_cjpe'
ARTIFACT = WORK/'CS781_Phase4_All_Artifacts.zip'
u.ROOT = ROOT
u.ARTIFACT = ARTIFACT
SEED = 781404
REPO = 'Exploration-Lab/CS-781-Capstone'
SPLITS = {'train':'Train/train.json', 'dev':'Train/dev.json', 'test':'Test/test.json'}
SLOTS = {
    '01': {'base':'01','method':'full','epochs':9,'min_epochs':4,'lr_base':8e-6},
    '02': {'base':'03','method':'partial','epochs':10,'min_epochs':4,'lr_base':1.4e-5},
    '03': {'base':'04','method':'lorma','epochs':12,'min_epochs':5,'lr_base':5e-5},
    '04': {'base':'02','method':'adalora','epochs':12,'min_epochs':5,'lr_base':5e-5},
    '05': {'base':'03','method':'adapters','epochs':12,'min_epochs':5,'lr_base':5e-5},
    '06': {'base':'03','method':'distill','epochs':10,'min_epochs':4,'lr_base':1.2e-5},
}
FOLDS = 3
MAX_TOKENS = 640
MAX_CHUNKS = 8
OVERLAP = 64
ACCUM = 4
PATIENCE = 3
THRESHOLDS = tuple(round(x/100,2) for x in range(30,71,5))


def log(*parts):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *parts, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32-1))
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def stage_base(base_id):
    """Find a Phase-1 recovery by its contents, regardless of Kaggle's mount name."""
    dest = ROOT/'bases'/base_id
    needed = [f'model/{u.SOURCES[base_id]}','model/course_bpe.py',
              'checkpoints/selected_model.pt','tokenizer/tokenizer.json',
              'tokenizer/tokenizer_metadata.json']
    if all((dest/name).is_file() for name in needed): return dest
    expected = f'phase1submission{base_id}recovery'
    normalize = lambda value: ''.join(c for c in value.lower() if c.isalnum())
    candidates=[]
    for source in u.INPUT.rglob(u.SOURCES[base_id]):
        if source.parent.name!='model': continue
        base=source.parent.parent
        if all((base/name).is_file() for name in needed):
            priority=0 if expected in normalize(str(base)) else 2
            candidates.append((priority,'directory',base,''))
    archives=sorted(p for p in u.INPUT.rglob('*') if p.is_file() and p.suffix.lower()=='.zip')
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as bundle:
                names=set(bundle.namelist())
                suffix=f'model/{u.SOURCES[base_id]}'
                for member in names:
                    if not member.endswith(suffix):continue
                    prefix=member[:-len(suffix)]
                    if all(prefix+name in names for name in needed):
                        if not all(u.safe_name(prefix+name) for name in needed):continue
                        priority=1 if expected in normalize(str(archive)) else 3
                        candidates.append((priority,'archive',archive,prefix))
        except (OSError,zipfile.BadZipFile):continue
    if not candidates:
        mounts=sorted(p.name for p in u.INPUT.iterdir()) if u.INPUT.is_dir() else []
        files=sorted(str(p.relative_to(u.INPUT)) for p in u.INPUT.rglob(u.SOURCES[base_id]))
        raise RuntimeError(
            f'Phase-1 base {base_id} is unavailable in the attached Kaggle Inputs. '
            f'Expected a complete recovery containing {needed}. '
            f'Mounted inputs: {mounts}. Model-file matches: {files}. '
            f'ZIP files: {[str(p.relative_to(u.INPUT)) for p in archives]}. '
            f'Attach the Phase-1 submission {base_id} recovery dataset and Run All again.')
    candidates.sort(key=lambda row:(row[0],str(row[2]),row[3]))
    selected=[row for row in candidates if row[0]==candidates[0][0]]
    if len(selected)!=1:
        raise RuntimeError(f'Ambiguous Phase-1 base {base_id} recoveries: '
                           f'{[(kind,str(path),prefix) for _,kind,path,prefix in selected]}')
    _,kind,source,prefix=selected[0]
    if kind=='directory':
        wanted=set(needed)|{p.relative_to(source).as_posix() for p in (source/'tokenizer').rglob('*') if p.is_file()}
        def copy_files():
            for name in sorted(wanted):
                target=dest/name
                if target.is_file():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
                    temporary=Path(handle.name)
                    with (source/name).open('rb') as reader:shutil.copyfileobj(reader,handle)
                os.replace(temporary,target)
        copy_files()
    else:
        with zipfile.ZipFile(source) as bundle:
            wanted=set(needed)|{name[len(prefix):] for name in bundle.namelist()
                                if name.startswith(prefix+'tokenizer/') and not name.endswith('/')}
            for name in sorted(wanted):
                if not u.safe_name(name) or prefix+name not in bundle.namelist():
                    raise RuntimeError(f'Unsafe or missing Phase-1 member: {name}')
                target=dest/name
                if target.is_file():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with bundle.open(prefix+name) as reader,tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
                    temporary=Path(handle.name);shutil.copyfileobj(reader,handle)
                os.replace(temporary,target)
    if not all((dest/name).is_file() for name in needed):
        raise RuntimeError(f'Phase-1 base {base_id} extraction was incomplete from {source}')
    u.write_json(dest/'provenance.json',{'mounted_source':str(source),'kind':kind,'member_prefix':prefix,
                 'model_sha256':u.sha(dest/needed[0]),
                 'checkpoint_sha256':u.sha(dest/'checkpoints/selected_model.pt')})
    log('STAGED PHASE-1 BASE',base_id,'from',source,'kind',kind)
    return dest


def stage_official():
    dest=ROOT/'official'
    if (dest/'provenance.json').is_file() and all((dest/f'{name}.json').is_file() for name in SPLITS):
        recorded=u.read_json(dest/'provenance.json')['sha256']
        if all(recorded.get(name)==u.sha(dest/f'{name}.json') for name in SPLITS):return
    try:
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret('HF_TOKEN')
        if not token: raise RuntimeError('HF_TOKEN is empty')
    except Exception as exc:
        raise RuntimeError('Add the enabled Kaggle Secret HF_TOKEN and turn Internet on') from exc
    try:
        from huggingface_hub import HfApi,hf_hub_download
    except ImportError:
        subprocess.check_call([sys.executable,'-m','pip','install','-q','huggingface_hub>=0.24,<2'])
        from huggingface_hub import HfApi,hf_hub_download
    api=HfApi()
    try:
        revision=api.repo_info(REPO,repo_type='dataset',token=token).sha
        paths=api.list_repo_files(REPO,repo_type='dataset',revision=revision,token=token)
    except Exception as exc:
        raise RuntimeError('Cannot access the official gated Phase-4 dataset. Check access, HF_TOKEN, Internet, and release date.') from exc
    chosen={}
    for name,suffix in SPLITS.items():
        hits=[path for path in paths if path.lower().endswith(('phase_4_cjpe/'+suffix).lower())]
        if len(hits)!=1:
            raise RuntimeError(f'Expected one official Phase-4 {suffix}; found {hits}. Dataset may not be released yet.')
        chosen[name]=hits[0]
    dest.mkdir(parents=True,exist_ok=True)
    for name,remote in chosen.items():
        target=dest/f'{name}.json'
        cached=Path(hf_hub_download(repo_id=REPO,repo_type='dataset',filename=remote,
                    revision=revision,token=token,cache_dir=str(ROOT/'hf_cache')))
        with tempfile.NamedTemporaryFile(dir=dest,delete=False) as handle:
            temporary=Path(handle.name)
            with cached.open('rb') as reader: shutil.copyfileobj(reader,handle)
        os.replace(temporary,target)
    u.write_json(dest/'provenance.json',{'repo':REPO,'revision':revision,'paths':chosen,
                'sha256':{name:u.sha(dest/f'{name}.json') for name in SPLITS}})
    del token


def parse_split(name):
    data=u.read_json(ROOT/'official'/f'{name}.json')
    if isinstance(data,dict):
        if isinstance(data.get('data'),list): data=data['data']
        elif isinstance(data.get('examples'),list): data=data['examples']
        elif isinstance(data.get('cases'),list): data=data['cases']
        else:
            data=[{'id':key,**(value if isinstance(value,dict) else {'text':value})}
                  for key,value in data.items()]
    if not isinstance(data,list): raise ValueError(f'{name}.json must be a list of cases')
    rows=[]
    for index,row in enumerate(data):
        if not isinstance(row,dict): raise ValueError(f'{name}: expected a case object')
        case_id=row.get('id',row.get('case_id'))
        content=row.get('text',row.get('case_text'))
        if isinstance(content,list) and all(isinstance(s,str) for s in content):content='\n'.join(content)
        if not isinstance(case_id,str) or not case_id or not isinstance(content,str) or not content.strip():
            raise ValueError(f'{name}: invalid id or text in case {case_id!r}')
        candidates=[key for key in ('label','judgment','judgement','outcome','gold','y') if key in row]
        if name=='test':
            if candidates and row[candidates[0]] is not None:
                raise ValueError('Official test file unexpectedly contains labels; stop before training')
            label=None
        else:
            if len(candidates)!=1:raise ValueError(f'{name}: expected one binary label field in {case_id}; found {candidates}')
            label=row[candidates[0]]
            if isinstance(label,bool):label=int(label)
            if label not in (0,1) or not isinstance(label,int):
                raise ValueError(f'{name}: non-binary label for {case_id}: {label!r}')
        # The official training file can contain several instances with the
        # same case ID. Keep each labeled instance, while using a stable
        # per-row key for token caches, checkpoints, and predictions.
        internal_id=case_id if name=='test' else f'{name}::row::{index:08d}'
        rows.append({'id':internal_id,'source_id':case_id,'text':content,'label':label})
    ids=[row['id'] for row in rows]
    if name=='test' and len(set(ids))!=len(ids):
        for index,row in enumerate(rows):row['id']=f'test::row::{index:08d}'
    if len({row['id'] for row in rows})!=len(rows):
        raise ValueError(f'Unable to construct unique internal row IDs in {name}.json')
    if not rows:raise ValueError(f'Empty {name}.json')
    return rows


def data():
    train,dev,test=(parse_split(name) for name in ('train','dev','test'))
    groups=[set(row['id'] for row in split) for split in (train,dev,test)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Official Phase-4 case IDs overlap across splits')
    if set(row['label'] for row in train)!={0,1} or set(row['label'] for row in dev)!={0,1}:
        raise ValueError('Both labels must occur in the train and dev splits for macro F1')
    return train,dev,test


def text_hash(text):
    return hashlib.sha256(re.sub(r'\s+',' ',text.lower()).strip().encode('utf8')).hexdigest()


def case_groups(rows,tie_source=True):
    """Keep repeated case IDs and exact duplicate texts in one validation group."""
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def merge(i,j):parent[root(i)]=root(j)
    seen={}
    for i,row in enumerate(rows):
        keys=[('text',text_hash(row['text']))]
        if tie_source:keys.append(('case',row['source_id']))
        for key in keys:
            if key in seen:merge(i,seen[key])
            else:seen[key]=i
    grouped=defaultdict(list)
    for i,row in enumerate(rows):grouped[root(i)].append(row)
    return list(grouped.values())


def validation_groups(rows,force_mode=None):
    if force_mode is not None:
        if force_mode not in ('case_id_and_exact_text','exact_text_only'):
            raise ValueError(f'Unknown validation grouping mode: {force_mode}')
        return case_groups(rows,tie_source=force_mode=='case_id_and_exact_text'),force_mode
    grouped=case_groups(rows,tie_source=True)
    # A generic reused ID can join a large fraction of unrelated cases.
    # In that situation, exact-text grouping is the defensible split unit.
    if len(grouped)<FOLDS*2 or max(map(len,grouped))>max(12,len(rows)//4):
        log('WARNING: source case IDs are too coarse for grouped folds; grouping by exact text only',
            'rows',len(rows),'source groups',len(grouped),'largest',max(map(len,grouped)))
        return case_groups(rows,tie_source=False),'exact_text_only'
    return grouped,'case_id_and_exact_text'


def folds_for(train,return_mode=False):
    groups,first_mode=validation_groups(train)
    modes=[first_mode]+(['exact_text_only'] if first_mode!='exact_text_only' else [])
    for mode in modes:
        groups,_=validation_groups(train,mode)
        buckets=[[],[]]
        for group in groups:
            label=int(sum(row['label'] for row in group)*2>=len(group))
            buckets[label].append(group)
        mapping={};counts=[[0,0] for _ in range(FOLDS)]
        for label,items in enumerate(buckets):
            random.Random(SEED+label).shuffle(items)
            items.sort(key=lambda group:-len(group))
            for group in items:
                added=[sum(row['label']==class_id for row in group) for class_id in (0,1)]
                fold=min(range(FOLDS),key=lambda i:(counts[i][label],sum(counts[i]),i))
                for row in group:mapping[row['id']]=fold
                for class_id in (0,1):counts[fold][class_id]+=added[class_id]
        if all(all(counts[i]) for i in range(FOLDS)):
            if mode!=first_mode:log('WARNING: source IDs prevent class-balanced folds; used exact-text grouping')
            return (mapping,mode) if return_mode else mapping
    raise ValueError('Too few independent cases of one class for three grouped outer folds')


def inner_split(rows,seed,group_mode=None):
    groups,_=validation_groups(rows,group_mode)
    by_label=[[],[]]
    for items in groups:
        label=int(sum(row['label'] for row in items)*2>=len(items))
        by_label[label].append(items)
    inner=[]
    for label,items in enumerate(by_label):
        random.Random(seed+label).shuffle(items)
        n=max(1,round(.15*sum(len(group) for group in items)))
        count=0
        for group in items:
            if count>=n:break
            inner.extend(group);count+=len(group)
    selected={row['id'] for row in inner}
    fit=[row for row in rows if row['id'] not in selected]
    if not fit or {r['label'] for r in fit}!={0,1} or {r['label'] for r in inner}!={0,1}:
        raise ValueError('Cannot form a stratified inner validation split')
    return fit,inner


def prepare():
    if torch.cuda.device_count()!=2:raise RuntimeError('Enable the Kaggle accelerator with exactly two GPUs')
    ROOT.mkdir(parents=True,exist_ok=True)
    u.restore_artifact()
    for base in ('01','02','03','04'):stage_base(base)
    stage_official()
    train,dev,test=data()
    for split_name,rows in (('train',train),('dev',dev),('test',test)):
        id_counts=Counter(row['source_id'] for row in rows)
        log('CASE-ID AUDIT',split_name,'rows',len(rows),'unique original IDs',len(id_counts),
            'repeated rows',len(rows)-len(id_counts),'largest ID group',max(id_counts.values()))
    foldmap,group_mode=folds_for(train,return_mode=True)
    hashes={name:u.sha(ROOT/'official'/f'{name}.json') for name in SPLITS}
    config={'version':2,'source_sha256':u.sha(__file__),'seed':SEED,'folds':foldmap,
            'validation_group_mode':group_mode,
            'official_sha256':hashes,'max_tokens':MAX_TOKENS,'max_chunks':MAX_CHUNKS,
            'slots':SLOTS}
    target=ROOT/'run_config.json'
    if target.is_file() and u.read_json(target)!=config:
        raise RuntimeError('Recovery archive belongs to different code, data, or configuration')
    u.write_json(target,config)
    shutil.copy2(__file__,ROOT/'cjpe_runner.py')
    shutil.copy2(Path(u.__file__),ROOT/'phase1_utils.py')
    counts={name:dict(Counter(row['label'] for row in rows)) for name,rows in (('train',train),('dev',dev))}
    duplicates={name:len(rows)-len({text_hash(r['text']) for r in rows}) for name,rows in (('train',train),('dev',dev),('test',test))}
    repeated_ids={name:len(rows)-len({r['source_id'] for r in rows})
                  for name,rows in (('train',train),('dev',dev),('test',test))}
    lengths={name:{'count':len(rows),'characters_median':int(np.median([len(r['text']) for r in rows])),
                   'characters_p95':int(np.percentile([len(r['text']) for r in rows],95))}
             for name,rows in (('train',train),('dev',dev),('test',test))}
    baseline=max(counts['dev'].values())/(sum(counts['dev'].values()))
    validation_group_list,validation_group_mode=validation_groups(train,group_mode)
    u.write_json(ROOT/'data_audit.json',{'label_counts':counts,'lengths':lengths,
                 'exact_duplicate_counts':duplicates,'repeated_source_id_rows':repeated_ids,
                 'case_groups_train':len(validation_group_list),'validation_group_mode':validation_group_mode,
                 'majority_accuracy_dev':baseline,
                 'train_case_count':len(train),'dev_case_count':len(dev),'test_case_count':len(test),
                 'test_id_sha256':hashlib.sha256('\n'.join(r['source_id'] for r in test).encode()).hexdigest(),
                 'private_score':None})
    log('PREPARED',len(train),'train',len(dev),'dev',len(test),'test','label counts',counts)


def tokenize_base(base_id):
    target=ROOT/'hf_cache'/f'tokens_{base_id}.pt'
    if target.is_file():return
    _,tokenizer=u.load_base(base_id)
    train,dev,test=data()
    rows=train+dev+test
    encoded={row['id']:tokenizer.backend.encode(row['text'],add_special_tokens=False).ids for row in rows}
    marker=tokenizer.backend.encode(' [JUDGMENT OUTCOME]',add_special_tokens=False).ids
    if not marker:raise RuntimeError('Outcome marker is empty')
    target.parent.mkdir(parents=True,exist_ok=True)
    u.save_torch(target,{'tokens':encoded,'marker':marker,'pad':int(tokenizer.pad_token_id)})
    lengths={name:[len(encoded[row['id']]) for row in split] for name,split in
             zip(('train','dev','test'),(train,dev,test))}
    u.write_json(ROOT/f'token_audit_{base_id}.json',{
        name:{'median':int(np.median(values)),'p95':int(np.percentile(values,95)),
              'max':max(values),'over_8_chunks':sum(v>MAX_CHUNKS*(MAX_TOKENS-OVERLAP) for v in values)}
        for name,values in lengths.items()})
    log('TOKENIZED',base_id,'cases',len(rows))


def chunk_starts(size,width,training,epoch,case_id):
    if size<=width:return [0]
    last=size-width
    count=min(MAX_CHUNKS,math.ceil(size/max(1,width-OVERLAP)))
    if count==1:return [0]
    values=[round(last*i/(count-1)) for i in range(count)]
    if training and count==MAX_CHUNKS and last>width:
        rng=random.Random(SEED+epoch*104729+int(hashlib.sha256(case_id.encode()).hexdigest()[:8],16))
        stride=last/(count-1)
        for i in range(1,count-1):values[i]=min(last,max(0,round(values[i]+rng.uniform(-.25,.25)*stride)))
    return sorted(set([0,last]+values))


def document_batch(row,cache,device,training=False,epoch=0):
    tokens=cache['tokens'][row['id']]
    marker=cache['marker']
    width=MAX_TOKENS-len(marker)
    if width<128:raise ValueError('Outcome marker unexpectedly long')
    starts=chunk_starts(len(tokens),width,training,epoch,row['id'])
    chunks=[(tokens[start:start+width] or [cache['pad']])+marker for start in starts]
    length=max(map(len,chunks))
    ids=torch.full((len(chunks),length),cache['pad'],dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,chunk in enumerate(chunks):
        ids[i,:len(chunk)]=torch.tensor(chunk,dtype=torch.long)
        mask[i,:len(chunk)]=True
    position=torch.tensor([min(31,round(31*start/max(1,len(tokens)-width))) for start in starts],dtype=torch.long)
    if device.type=='cuda':
        ids=ids.pin_memory().to(device,non_blocking=True)
        mask=mask.pin_memory().to(device,non_blocking=True)
        position=position.pin_memory().to(device,non_blocking=True)
    return ids,mask,position


def macro_f1(probs,labels,threshold=.5):
    if len(probs)!=len(labels) or not probs:raise ValueError('Empty or mismatched score set')
    pred=[int(value>=threshold) for value in probs]
    scores=[];details={}
    for label in (0,1):
        tp=sum(p==label and y==label for p,y in zip(pred,labels))
        fp=sum(p==label and y!=label for p,y in zip(pred,labels))
        fn=sum(p!=label and y==label for p,y in zip(pred,labels))
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        details[str(label)]={'precision':precision,'recall':recall,'f1':f1,'tp':tp,'fp':fp,'fn':fn,
                             'gold':sum(y==label for y in labels),'predicted':sum(p==label for p in pred)}
        scores.append(f1)
    return {'macro_f1':sum(scores)/2,'per_class':details,'threshold':threshold,
            'accuracy':sum(p==y for p,y in zip(pred,labels))/len(labels)}


def best_threshold(probs,labels):
    values=[macro_f1(probs,labels,t) for t in THRESHOLDS]
    return max(values,key=lambda result:(result['macro_f1'],-abs(result['threshold']-.5)))


class BottleneckAdapter(nn.Module):
    def __init__(self,block,width,rank=96):
        super().__init__()
        self.block=block
        self.down=nn.Linear(width,rank)
        self.up=nn.Linear(rank,width)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        self.scale=nn.Parameter(torch.tensor(1.))

    def forward(self,*args):
        result=self.block(*args)
        return result+self.scale*self.up(F.gelu(self.down(result)))


class AdaLoRALinear(nn.Module):
    """Low-rank SVD-like update with gradient-importance rank allocation."""
    def __init__(self,base,rank=16,alpha=16):
        super().__init__()
        self.base=base
        self.a=nn.Linear(base.in_features,rank,bias=False)
        self.b=nn.Linear(rank,base.out_features,bias=False)
        self.s=nn.Parameter(torch.full((rank,),.01))
        self.register_buffer('mask',torch.ones(rank))
        self.register_buffer('importance',torch.zeros(rank))
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5))
        nn.init.normal_(self.b.weight,std=.01)
        self.scale=alpha/rank

    def forward(self,x):
        return self.base(x)+self.b(self.a(x)*(self.s*self.mask))*self.scale

    @torch.no_grad()
    def record_importance(self):
        if self.s.grad is not None:
            self.importance.mul_(.9).add_((self.s*self.s.grad).abs(),alpha=.1)


def allocate_ada_ranks(model,epoch,total_epochs):
    modules=[module for module in model.modules() if isinstance(module,AdaLoRALinear)]
    if not modules:return
    initial=sum(module.mask.numel() for module in modules)
    target=max(4*len(modules),round(initial*(1.-.5*min(1.,epoch/max(1,total_epochs)))))
    candidates=[]
    for i,module in enumerate(modules):
        ranking=torch.argsort(module.importance,descending=True).tolist()
        for j in ranking[:4]:candidates.append((float('inf'),i,j))
        for j in ranking[4:]:candidates.append((float(module.importance[j]),i,j))
    candidates.sort(key=lambda row:row[0],reverse=True)
    selected={(i,j) for _,i,j in candidates[:target]}
    for i,module in enumerate(modules):
        module.mask.copy_(torch.tensor([float((i,j) in selected) for j in range(module.mask.numel())],
                                       device=module.mask.device,dtype=module.mask.dtype))


class CaseClassifier(nn.Module):
    def __init__(self,base,base_id,method):
        super().__init__()
        self.base=base
        self.base_id=base_id
        self.method=method
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
                if (isinstance(module,nn.Linear) and
                    name.split('.')[-1] in ('q_proj','v_proj','o_proj','gate_proj','up_proj') and
                    name.startswith('blocks.')):
                    parent=base.get_submodule(name.rsplit('.',1)[0])
                    setattr(parent,name.rsplit('.',1)[-1],u.LoRMA(module,16,16))
                    count+=1
            if count<14:raise RuntimeError('LORMA did not adapt base 04 blocks')
        elif method=='adalora':
            count=0
            for name,module in list(base.named_modules()):
                if (name.startswith('encoder_blocks.') and isinstance(module,nn.Linear) and
                    name.split('.')[-1] in ('query','value','input','output')):
                    parent=base.get_submodule(name.rsplit('.',1)[0])
                    setattr(parent,name.rsplit('.',1)[-1],AdaLoRALinear(module))
                    count+=1
            if count<12:raise RuntimeError('AdaLoRA did not adapt base 02 encoder')
            base.set_gradient_checkpointing(True)
        elif method=='adapters':
            self.first_trainable=len(base.blocks)-12
            for i in range(self.first_trainable,len(base.blocks)):
                base.blocks[i]=BottleneckAdapter(base.blocks[i],width)
        else:raise ValueError(method)
        if base_id=='04':base.set_gradient_checkpointing(True)
        self.project=nn.Sequential(nn.Linear(width,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.15))
        self.position=nn.Embedding(32,256)
        self.context=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256,4,512,dropout=.15,batch_first=True,norm_first=True),
            num_layers=1,enable_nested_tensor=False)
        self.gate=nn.Linear(256,1)
        self.classifier=nn.Sequential(nn.LayerNorm(768),nn.Dropout(.25),nn.Linear(768,256),
                                      nn.GELU(),nn.Dropout(.2),nn.Linear(256,1))

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
                if i<self.first_trainable:
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

    def forward(self,ids,mask,position):
        chunks=self.project(self.chunk_features(ids,mask))+self.position(position)
        chunks=self.context(chunks[None])[0]
        weight=self.gate(chunks).squeeze(-1).softmax(0)
        pooled=(chunks*weight[:,None]).sum(0)
        combined=torch.cat((pooled,chunks.mean(0),chunks.max(0).values),-1)
        return self.classifier(combined).squeeze(-1)


def model_for(slot,device):
    spec=SLOTS[slot]
    base,tokenizer=u.load_base(spec['base'])
    model=CaseClassifier(base,spec['base'],spec['method']).to(device)
    return model,tokenizer


def trainable_state(model):
    keys={name for name,p in model.named_parameters() if p.requires_grad}
    buffers={name for name,_ in model.named_buffers() if name.endswith(('.mask','.importance'))}
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()
            if name in keys or name in buffers}


def load_state(model,state):
    keys={name for name,p in model.named_parameters() if p.requires_grad}
    buffers={name for name,_ in model.named_buffers() if name.endswith(('.mask','.importance'))}
    expected=keys|buffers
    if set(state)!=expected:
        raise RuntimeError(f'Trainable-state mismatch: missing={expected-set(state)} extra={set(state)-expected}')
    model.load_state_dict(state,strict=False)


def optimizer_for(model,spec,updates):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('base.'):
            lr=spec['lr_base']
            if '.a.' in name or '.b.' in name or '.down.' in name or '.up.' in name or '.s' in name:
                lr=max(lr,5e-5)
        else:lr=1.2e-4
        decay=0. if name.endswith('bias') or 'norm' in name.lower() or name.endswith('.s') else .01
        groups.setdefault((lr,decay),[]).append(p)
    parameters=[{'params':values,'lr':lr,'weight_decay':decay} for (lr,decay),values in groups.items()]
    try:optimizer=torch.optim.AdamW(parameters,fused=True)
    except (TypeError,RuntimeError):optimizer=torch.optim.AdamW(parameters)
    warmup=max(4,round(.06*updates))
    def schedule(step):
        if step<warmup:return max(.1,(step+1)/warmup)
        portion=min(1.,(step-warmup)/max(1,updates-warmup))
        return .12+.88*.5*(1+math.cos(math.pi*portion))
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)


def init_dist():
    if int(os.environ.get('WORLD_SIZE','1'))!=2 or torch.cuda.device_count()!=2:
        raise RuntimeError('Use torchrun --standalone --nproc_per_node=2 with two visible GPUs')
    local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda',local)
    try:dist.init_process_group('nccl',timeout=timedelta(minutes=45),device_id=device)
    except TypeError:dist.init_process_group('nccl',timeout=timedelta(minutes=45))
    return int(os.environ['RANK']),device


def stage_rows(stage):
    train,dev,test=data()
    if stage=='final':return train,dev,test
    if not stage.startswith('fold_'):raise ValueError(stage)
    fold=int(stage.split('_')[1])
    if fold not in range(FOLDS):raise ValueError(stage)
    config=u.read_json(ROOT/'run_config.json')
    mapping=config['folds']
    outer=[row for row in train if mapping[row['id']]==fold]
    available=[row for row in train if mapping[row['id']]!=fold]
    fit,inner=inner_split(available,SEED+fold*103,config.get('validation_group_mode'))
    if not outer or not fit or not inner:raise RuntimeError('Empty fold partition')
    return fit,inner,outer


def epoch_order(rows,slot,stage,epoch):
    ids=[row['id'] for row in rows]
    seed=SEED+sum(map(ord,slot+stage))*997+epoch*17
    rng=random.Random(seed)
    rng.shuffle(ids)
    original=list(ids)
    while len(ids)%(2*ACCUM):ids.append(original[(len(ids)-len(original))%len(original)])
    return ids


@torch.no_grad()
def predict_rows(model,rows,cache,rank,device):
    model.eval()
    local={}
    for row in rows[rank::2]:
        ids,mask,pos=document_batch(row,cache,device,False)
        with torch.autocast('cuda',dtype=torch.float16):
            value=model(ids,mask,pos)
        local[row['id']]=float(value.float().cpu())
    gathered=[None,None]
    dist.all_gather_object(gathered,local)
    result={key:value for piece in gathered for key,value in piece.items()}
    if set(result)!={row['id'] for row in rows}:
        raise RuntimeError('Distributed inference did not cover each case exactly once')
    return result


def teacher_cache(stage,fit,rank,device):
    destination=ROOT/'runs'/'06'/stage/'teacher_logits.pt'
    if destination.is_file():
        values=torch.load(destination,map_location='cpu',weights_only=False)
        if set(values)!={row['id'] for row in fit}:raise RuntimeError('Teacher cache ID mismatch')
        return values
    teacher_path=ROOT/'runs'/'01'/stage/'best.pt'
    if not teacher_path.is_file():raise RuntimeError('Train the matching base-01 teacher stage before distillation')
    teacher,tokenizer=model_for('01',device)
    load_state(teacher,torch.load(teacher_path,map_location='cpu',weights_only=False)['model'])
    token_path=ROOT/'hf_cache/tokens_01.pt'
    if not token_path.is_file():raise FileNotFoundError(token_path)
    cache=torch.load(token_path,map_location='cpu',weights_only=False)
    result=predict_rows(teacher,fit,cache,rank,device)
    if rank==0:
        destination.parent.mkdir(parents=True,exist_ok=True)
        u.save_torch(destination,result)
        log('CACHED FOLD-SAFE TEACHER',stage,len(result))
    dist.barrier()
    del teacher,tokenizer,cache
    torch.cuda.empty_cache()
    return result


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def gpu_sample(slot,stage,epoch):
    try:
        process=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
              '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        rows=[line.strip() for line in process.stdout.splitlines()]
    except Exception as exc:rows=[repr(exc)]
    with (ROOT/'gpu_samples.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps({'slot':slot,'stage':stage,'epoch':epoch,'gpus':rows,'time':time.time()})+'\n')


def train_stage(slot,stage,chunk_epochs):
    rank,device=init_dist()
    try:
        spec=SLOTS[slot]
        directory=ROOT/'runs'/slot/stage
        directory.mkdir(parents=True,exist_ok=True)
        status=directory/'status.json'
        if status.is_file() and u.read_json(status).get('complete'):
            if rank==0:log('REUSE',slot,stage)
            dist.barrier();return
        seed_all(SEED+int(slot)*1000+sum(map(ord,stage))+rank)
        fit,inner,outer=stage_rows(stage)
        cache_path=ROOT/'hf_cache'/f"tokens_{spec['base']}.pt"
        if not cache_path.is_file():raise FileNotFoundError(cache_path)
        cache=torch.load(cache_path,map_location='cpu',weights_only=False)
        by_id={row['id']:row for row in fit}
        teacher=teacher_cache(stage,fit,rank,device) if slot=='06' else None
        model,tokenizer=model_for(slot,device)
        steps_per_epoch=len(epoch_order(fit,slot,stage,1))//(2*ACCUM)
        optimizer,scheduler=optimizer_for(model,spec,max(1,steps_per_epoch*spec['epochs']))
        scaler=torch.amp.GradScaler('cuda')
        latest=directory/'latest.pt'
        history=u.read_json(directory/'history.json') if (directory/'history.json').is_file() else []
        start=1;best=-1.;best_epoch=0;best_threshold_value=.5
        state=None
        if latest.is_file():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            load_state(model,state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            start=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch']
            best_threshold_value=state['best_threshold']
            restore_rng(state['rng'][rank])
        ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,
                find_unused_parameters=False,gradient_as_bucket_view=True)
        count=Counter(row['label'] for row in fit)
        pos_weight=torch.tensor(min(2.,max(.5,count[0]/max(1,count[1]))),device=device)
        limit=min(spec['epochs'],start+chunk_epochs-1)
        finished=start>spec['epochs']
        for epoch in range(start,limit+1):
            epoch_started=time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            ddp.train();optimizer.zero_grad(set_to_none=True)
            order=epoch_order(fit,slot,stage,epoch)[rank::2]
            loss_sum=0.;step_count=0;processed_tokens=0
            for j,case_id in enumerate(order):
                row=by_id[case_id]
                ids,mask,pos=document_batch(row,cache,device,True,epoch)
                processed_tokens+=int(mask.sum())
                sync=(j+1)%ACCUM==0
                with (nullcontext() if sync else ddp.no_sync()):
                    with torch.autocast('cuda',dtype=torch.float16):
                        value=ddp(ids,mask,pos)
                        target=torch.tensor(float(row['label']),device=device)
                        hard=F.binary_cross_entropy_with_logits(value.float(),target,pos_weight=pos_weight)
                        if teacher is None:loss=hard/ACCUM
                        else:
                            temperature=2.
                            soft=torch.sigmoid(torch.tensor(teacher[case_id]/temperature,device=device))
                            distill=F.binary_cross_entropy_with_logits(value.float()/temperature,soft)*temperature**2
                            loss=(.75*hard+.25*distill)/ACCUM
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite CJPE loss')
                    scaler.scale(loss).backward()
                loss_sum+=float(loss.detach())*ACCUM
                if sync:
                    scaler.unscale_(optimizer)
                    for module in model.modules():
                        if isinstance(module,AdaLoRALinear):module.record_importance()
                    nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer);scaler.update();scheduler.step()
                    optimizer.zero_grad(set_to_none=True);step_count+=1
            if spec['method']=='adalora':allocate_ada_ranks(model,epoch,spec['epochs'])
            dist.barrier()
            validation=predict_rows(model,inner,cache,rank,device)
            probabilities=[float(torch.sigmoid(torch.tensor(validation[row['id']]))) for row in inner]
            observed=best_threshold(probabilities,[row['label'] for row in inner])
            if rank==0:
                score=observed['macro_f1']
                if score>best+1e-6:
                    best=score;best_epoch=epoch;best_threshold_value=observed['threshold']
                    u.save_torch(directory/'best.pt',{'model':trainable_state(model),'epoch':epoch,
                                   'inner_macro_f1':best,'threshold':best_threshold_value,
                                   'slot':slot,'stage':stage})
                history.append({'epoch':epoch,'train_loss_rank0':loss_sum/max(1,len(order)),
                                'inner':observed,'best_epoch':best_epoch,
                                'optimizer_steps_rank0':step_count,
                                'epoch_seconds':round(time.monotonic()-epoch_started,2),
                                'rank0_tokens_per_second':round(processed_tokens/max(1,time.monotonic()-epoch_started),1),
                                'rank0_peak_gpu_gib':round(torch.cuda.max_memory_allocated(device)/2**30,2)})
                u.write_json(directory/'history.json',history)
                log('TRAIN',slot,stage,'epoch',epoch,'loss',round(history[-1]['train_loss_rank0'],4),
                    'inner macro F1',round(score,4),'best',round(best,4),
                    'rank0 tokens/s',history[-1]['rank0_tokens_per_second'],
                    'peak GiB',history[-1]['rank0_peak_gpu_gib'])
                gpu_sample(slot,stage,epoch)
                finished=(epoch>=spec['min_epochs'] and epoch-best_epoch>=PATIENCE) or epoch>=spec['epochs']
                u.write_json(status,{'complete':False,'epoch':epoch,'best_epoch':best_epoch,
                                     'best_inner_macro_f1':best})
            flags=torch.tensor([int(finished)],device=device)
            dist.broadcast(flags,0);finished=bool(flags.item())
            states=[None,None]
            dist.all_gather_object(states,rng_state())
            if rank==0:
                u.save_torch(latest,{'model':trainable_state(model),'optimizer':optimizer.state_dict(),
                             'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                             'epoch':epoch,'best':best,'best_epoch':best_epoch,
                             'best_threshold':best_threshold_value,'rng':states})
            dist.barrier()
            if finished:break
        if finished:
            selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
            load_state(model,selected['model'])
            dev_predictions=predict_rows(model,inner,cache,rank,device) if stage=='final' else None
            predictions=predict_rows(model,outer,cache,rank,device)
            if rank==0:
                if dev_predictions is not None:u.save_torch(directory/'dev_logits.pt',dev_predictions)
                u.save_torch(directory/('test_logits.pt' if stage=='final' else 'outer_logits.pt'),predictions)
                u.write_json(status,{'complete':True,'epoch':history[-1]['epoch'],
                    'best_epoch':best_epoch,'inner_macro_f1':best,'threshold':best_threshold_value,
                    'predicted_cases':len(predictions)})
                latest.unlink(missing_ok=True)
                log('COMPLETE',slot,stage,'cases',len(predictions),'best epoch',best_epoch)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def complete(slot,stage):
    target=ROOT/'runs'/slot/stage/'status.json'
    return target.is_file() and bool(u.read_json(target).get('complete'))


def summarize_slot(slot):
    if not complete(slot,'final'):
        raise RuntimeError(f'Slot {slot} final model is incomplete; resume training before packaging')
    train,dev,test=data()
    mapping=u.read_json(ROOT/'run_config.json')['folds']
    outer_logits={};thresholds=[];outer_predictions={};fold_metrics={}
    for fold in range(FOLDS):
        if not complete(slot,f'fold_{fold}'):continue
        path=ROOT/'runs'/slot/f'fold_{fold}'
        values=torch.load(path/'outer_logits.pt',map_location='cpu',weights_only=False)
        expected={row['id'] for row in train if mapping[row['id']]==fold}
        if set(values)!=expected or set(values)&set(outer_logits):
            raise RuntimeError(f'Slot {slot} fold {fold}: outer IDs wrong or repeated')
        threshold=float(u.read_json(path/'status.json')['threshold'])
        thresholds.append(threshold)
        outer_logits.update(values)
        outer_predictions.update({key:int(torch.sigmoid(torch.tensor(value)).item()>=threshold)
                                  for key,value in values.items()})
        heldout=[row for row in train if mapping[row['id']]==fold]
        fold_metrics[str(fold)]=macro_f1([outer_predictions[row['id']] for row in heldout],
                                          [row['label'] for row in heldout],.5)
    validated_rows=[row for row in train if row['id'] in outer_logits]
    outer=macro_f1([outer_predictions[row['id']] for row in validated_rows],
                   [row['label'] for row in validated_rows],.5) if validated_rows else None
    complete_outer=len(thresholds)==FOLDS
    if complete_outer and set(outer_logits)!={row['id'] for row in train}:
        raise RuntimeError(f'Slot {slot}: complete folds have incomplete OOF coverage')
    final_path=ROOT/'runs'/slot/'final'
    dev_logits=torch.load(final_path/'dev_logits.pt',map_location='cpu',weights_only=False)
    test_logits=torch.load(final_path/'test_logits.pt',map_location='cpu',weights_only=False)
    if set(dev_logits)!={row['id'] for row in dev} or set(test_logits)!={row['id'] for row in test}:
        raise RuntimeError(f'Slot {slot}: final dev/test IDs do not match official data')
    final_threshold=float(np.median(thresholds)) if len(thresholds)>=2 else float(u.read_json(final_path/'status.json')['threshold'])
    dev_probs=[float(torch.sigmoid(torch.tensor(dev_logits[row['id']]))) for row in dev]
    development=macro_f1(dev_probs,[row['label'] for row in dev],final_threshold)
    labels=[row['label'] for row in train]
    majority=int(Counter(labels)[1]>Counter(labels)[0])
    majority_macro=macro_f1([float(majority)]*len(labels),labels,.5)['macro_f1']
    report={'slot':slot,'base':SLOTS[slot]['base'],'method':SLOTS[slot]['method'],
            'validation_status':'COMPLETE_OUTER_3FOLD' if complete_outer else f'PROVISIONAL_{len(thresholds)}_FOLDS',
            'outer_macro_f1':None if outer is None else outer['macro_f1'],'outer_metric':outer,
            'outer_folds':fold_metrics,
            'dev_macro_f1':development['macro_f1'],'dev_metric':development,
            'inner_thresholds':thresholds,'final_threshold':final_threshold,
            'majority_outer_macro_f1':majority_macro,
            'outer_cases':len(validated_rows),'dev_cases':len(dev),'test_cases':len(test),
            'quality_warning':not complete_outer or outer['macro_f1']<majority_macro+.05 or
               min(outer['per_class']['0']['f1'],outer['per_class']['1']['f1'])<.25,
            'private_f1':None,
            'note':'Outer cases never selected their checkpoint or threshold; private score is unknown.'}
    u.write_json(ROOT/'reports'/f'submission_{slot}.json',report)
    return report,test_logits,test


def package_slot(slot):
    report,logits,test=summarize_slot(slot)
    threshold=report['final_threshold']
    predictions=[]
    for row in test:
        probability=float(torch.sigmoid(torch.tensor(logits[row['id']])))
        predictions.append({'id':row['source_id'],'pred':int(probability>=threshold)})
    if len(predictions)!=len(test) or [row['id'] for row in predictions]!=[row['source_id'] for row in test]:
        raise RuntimeError('Submission IDs or row order differ from official test data')
    if any(type(row['pred']) is not int or row['pred'] not in (0,1) for row in predictions):
        raise RuntimeError('Submission has a nonbinary prediction')
    destination=ROOT/'submissions'/slot
    destination.mkdir(parents=True,exist_ok=True)
    u.write_json(destination/'prediction.json',predictions)
    output=WORK/f'CS781_Phase4_Submission_{slot}.zip'
    temporary=output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(destination/'prediction.json','prediction.json')
    os.replace(temporary,output)
    with zipfile.ZipFile(output) as archive:
        if archive.namelist()!=['prediction.json'] or json.loads(archive.read('prediction.json'))!=predictions:
            raise RuntimeError('Submission ZIP round-trip verification failed')
    shutil.copy2(output,destination/'submission.zip')
    report['submission']=str(output)
    report['zip_sha256']=u.sha(output)
    report['prediction_sha256']=u.sha(destination/'prediction.json')
    u.write_json(ROOT/'reports'/f'submission_{slot}.json',report)
    log('PACKAGED',slot,'outer macro F1',None if report['outer_macro_f1'] is None else round(report['outer_macro_f1'],4),
        'official dev',round(report['dev_macro_f1'],4),'quality warning',report['quality_warning'])


def final_report():
    summaries=[]
    for slot in SLOTS:
        path=ROOT/'reports'/f'submission_{slot}.json'
        if not path.is_file():raise RuntimeError(f'Missing slot {slot} report')
        summaries.append(u.read_json(path))
    hashes=[row['prediction_sha256'] for row in summaries]
    distinct=len(set(hashes))==len(hashes)
    complete_validation=all(row['validation_status']=='COMPLETE_OUTER_3FOLD' for row in summaries)
    ranked=sorted(summaries,key=lambda row:(-1 if row['outer_macro_f1'] is None else row['outer_macro_f1']),reverse=True)
    result={'status':('SIX_DISTINCT_PACKAGES_READY' if distinct and complete_validation else
                      'SIX_PROVISIONAL_PACKAGES_READY' if distinct else
                      'DUPLICATE_PREDICTIONS_REQUIRE_NEW_EXPERIMENT'),
            'slots':summaries,'distinct_predictions':distinct,'private_scores':[None]*len(summaries),
            'outer_f1_ranking':[{'slot':row['slot'],'outer_macro_f1':row['outer_macro_f1'],
                                 'outer_cases':row['outer_cases'],
                                 'validation_status':row['validation_status']} for row in ranked],
            'artifact':str(ARTIFACT),
            'warning':'Local macro F1 does not guarantee the private Codabench score. '
                      'Each recovery ZIP must be downloaded or persisted outside a fatal draft session.'}
    u.write_json(ROOT/'report.json',result)
    log('FINAL REPORT',result['status'])


def snapshot():
    u.snapshot()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','tokenize','train','package','report','snapshot'))
    parser.add_argument('--slot',choices=tuple(SLOTS),default='01')
    parser.add_argument('--base',choices=('01','02','03','04'),default='01')
    parser.add_argument('--stage',default='final')
    parser.add_argument('--chunk-epochs',type=int,default=2)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='tokenize':tokenize_base(args.base)
    elif args.action=='train':train_stage(args.slot,args.stage,args.chunk_epochs)
    elif args.action=='package':package_slot(args.slot)
    elif args.action=='report':final_report()
    elif args.action=='snapshot':snapshot()


if __name__=='__main__':main()
