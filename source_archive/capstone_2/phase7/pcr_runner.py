"""Phase 7: original-Phase-1 retrieval, six methods, atomic segmented recovery."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import cs781_phase7_utils as u
import cs781_phase7_evaluator as official

WORK = Path('/kaggle/working')
ROOT = WORK / 'CS781_Capstone/phase7_pcr'
u.ROOT = ROOT
u.ARTIFACT = WORK / 'CS781_Phase7_All_Artifacts.zip'
SEED = 781707
CHUNK_TOKENS, CHUNKS, OVERLAP, DIM = 384, 6, 64, 256
PROMPT_TOKENS, TEMPERATURE = 20, .07
EVAL_QUERIES, TOP_CACHE, ACCUM = 128, 400, 4
SAVE_SECONDS, PATIENCE = 300, 2
COMPATIBLE_RUNNERS={'26701356d98ff07e57e8a832e38a8618b24c0d37d0f7a255194eeb76d0c20f8c'}
SLOTS = {
    '01': dict(base='03', method='full', lr=2e-5, head_lr=2e-4, epochs=6),
    '02': dict(base='01', method='partial', lr=3e-5, head_lr=2e-4, epochs=5),
    '03': dict(base='04', method='lorma', lr=1.5e-4, head_lr=2e-4, epochs=5),
    '04': dict(base='02', method='ia3', lr=3e-4, head_lr=2e-4, epochs=5),
    '05': dict(base='03', method='prompt', lr=5e-4, head_lr=2e-4, epochs=5),
    '06': dict(base='02', method='distill', lr=2e-5, head_lr=2e-4, epochs=5),
    'control': dict(base='02', method='gold_control', lr=2e-5, head_lr=2e-4, epochs=2),
}
WORD = re.compile(r'[A-Za-z][A-Za-z0-9_]*|\d+')
CUE = re.compile(r'\b(issue|held|reason|therefore|accordingly|section|principle|facts|court)\b', re.I)


def log(*args):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *args, flush=True)


def stable(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:12], 16)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % (2**32-1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def strict_json(path):
    return official.load_json(Path(path))


def text_value(value):
    if isinstance(value, dict):
        for key in ('text', 'document', 'sentences', 'content'):
            if key in value: return text_value(value[key])
    if isinstance(value, list):
        if not all(isinstance(x, str) for x in value):
            raise ValueError('Case sentence list contains non-string values')
        value = '\n'.join(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Case has no nonempty text')
    return value.strip()


def records(raw, labelled=False):
    if isinstance(raw, list):
        result = {}
        for row in raw:
            if not isinstance(row, dict): raise ValueError('Expected case records')
            key = row.get('id', row.get('case_id', row.get('query_id')))
            if key in result: raise ValueError(f'Duplicate case ID {key!r}')
            result[key] = row
        raw = result
    if not isinstance(raw, dict) or not raw: raise ValueError('Expected a nonempty case mapping')
    rows = {}
    for key, value in raw.items():
        if type(key) is not str or not key.strip(): raise ValueError('IDs must be nonempty strings')
        record = {'id': key, 'text': text_value(value)}
        if labelled:
            if not isinstance(value, dict) or 'relevant_candidates' not in value:
                raise ValueError(f'Missing relevant_candidates for labelled query {key}')
            labels = value['relevant_candidates']
            if not isinstance(labels, list): raise ValueError(f'Gold labels for {key} must be a list')
            labels = [x for x in labels if isinstance(x, str) and x.strip()]
            if len(labels) != len(set(labels)): raise ValueError(f'Duplicate gold IDs for {key}')
            record['gold'] = labels
        elif isinstance(value, dict) and value.get('relevant_candidates'):
            raise ValueError(f'Unexpected hidden labels in unlabelled case {key}')
        rows[key] = record
    return rows


@lru_cache(maxsize=3)
def data(split):
    folder = ROOT / 'official'
    queries = records(strict_json(folder / f'{split}.json'), split != 'test')
    # Candidate records may have auxiliary metadata, but are never supervision.
    raw = strict_json(folder / f'{split}_candidates.json')
    if isinstance(raw, dict): raw = {k: {'text': text_value(v)} for k, v in raw.items()}
    elif isinstance(raw, list): raw = [dict(id=v.get('id', v.get('case_id')), text=text_value(v)) for v in raw]
    candidates = records(raw)
    if len(candidates) < 7: raise ValueError(f'{split} candidate pool has fewer than seven cases')
    for q in queries.values():
        if set(q.get('gold', [])) - set(candidates):
            raise ValueError(f"{split} query {q['id']} has a gold ID outside its own pool")
    return queries, candidates


def stage_official():
    dest = ROOT / 'official'; dest.mkdir(parents=True, exist_ok=True)
    names = [f'{s}{tail}.json' for s in ('train', 'dev', 'test') for tail in ('', '_candidates')]
    provenance = dest / 'provenance.json'
    if provenance.is_file():
        record = strict_json(provenance)
        if all((dest/n).is_file() and u.sha(dest/n) == record['sha256'][n] for n in names): return
        raise RuntimeError('Official data changed after preparation')
    chosen = {}
    for name in names:
        hits = [p for p in u.INPUT.rglob(name) if p.is_file() and
                'phase_7_pcr' in str(p).lower().replace('-', '_')]
        if len(hits) != 1: hits = [p for p in u.INPUT.rglob(name) if p.is_file()]
        if len(hits) == 1: chosen[name] = hits[0]
    if len(chosen) == len(names):
        for name, path in chosen.items(): shutil.copy2(path, dest/name)
        record = {'source': 'attached_official_release', 'paths': {k:str(v) for k,v in chosen.items()}}
    else:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import HfApi, hf_hub_download
        token = UserSecretsClient().get_secret('HF_TOKEN')
        repo = 'Exploration-Lab/CS-781-Capstone'; api = HfApi()
        revision = api.repo_info(repo, repo_type='dataset', token=token).sha
        listing = api.list_repo_files(repo, repo_type='dataset', revision=revision, token=token)
        paths = {}
        for name in names:
            split = name.split('_')[0].split('.')[0]
            suffix = f'phase_7_pcr/{split}/{name}'
            hits = [p for p in listing if p.lower().replace('-', '_').endswith(suffix)]
            if len(hits) != 1: raise RuntimeError(f'Cannot identify official {name}: {hits}')
            paths[name] = hits[0]
            src = hf_hub_download(repo, filename=hits[0], repo_type='dataset', revision=revision,
                                  token=token, cache_dir=str(ROOT/'hf_cache'))
            shutil.copy2(src, dest/name)
        record = {'source': repo, 'revision': revision, 'paths': paths}
        del token
    record['sha256'] = {n:u.sha(dest/n) for n in names}; u.write_json(provenance, record)


def normalized_hash(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()


def partition_ids(queries, fraction):
    groups = {}
    for key, row in queries.items(): groups.setdefault(normalized_hash(row['text']), []).append(key)
    groups = list(groups.values()); random.Random(SEED).shuffle(groups)
    cut = max(1, min(len(groups)-1, round(len(groups)*fraction)))
    if len(groups) < 2: raise ValueError('Not enough distinct queries for a held-out split')
    return [x for g in groups[:cut] for x in g], [x for g in groups[cut:] for x in g]


def prepare(require_cuda=True):
    ROOT.mkdir(parents=True, exist_ok=True); u.restore_artifact()
    if require_cuda and torch.cuda.device_count() != 2:
        raise RuntimeError('Select the two-GPU Kaggle accelerator')
    for b in ('01','02','03','04'): u.stage_base(b)
    stage_official()
    train, tc = data('train'); dev, dc = data('dev'); test, xc = data('test')
    if len(dev) >= 60:
        tune, audit = partition_ids(dev, .70)
        heldout_hashes={normalized_hash(r['text']) for r in dev.values()}
        fit=[k for k,r in train.items() if normalized_hash(r['text']) not in heldout_hashes]
        split = {'fit':fit, 'tune':tune, 'audit':audit,
                 'tune_pool':'dev', 'audit_pool':'dev'}
    else:
        fit, tune = partition_ids(train, .85)
        split = {'fit':fit, 'tune':tune, 'audit':list(dev),
                 'tune_pool':'train', 'audit_pool':'dev'}
    config = {'version':1, 'seed':SEED, 'slots':SLOTS, 'chunks':CHUNKS,
              'chunk_tokens':CHUNK_TOKENS, 'dimension':DIM, 'split':split,
              'official':strict_json(ROOT/'official/provenance.json')['sha256'],
              'base_sha256':{b:strict_json(ROOT/'bases'/b/'provenance.json')['checkpoint_sha256']
                             for b in ('01','02','03','04')},
              'runner_sha256':u.sha(__file__)}
    target = ROOT/'run_config.json'
    if target.is_file() and strict_json(target) != config:
        old=strict_json(target)
        same={k:v for k,v in old.items() if k!='runner_sha256'}=={k:v for k,v in config.items() if k!='runner_sha256'}
        if not same or old.get('runner_sha256') not in COMPATIBLE_RUNNERS:
            raise RuntimeError('Recovery code, settings, or data differ; keep the same notebook version')
        u.write_json(ROOT/'memory_fix_migration.json',{'from':old['runner_sha256'],
                     'to':config['runner_sha256'],'preserves':'data, token caches, model/optimizer progress'})
    u.write_json(target, config)
    for src, name in ((Path(__file__),'pcr_runner.py'),(Path(u.__file__),'utils.py'),
                      (Path(official.__file__),'evaluate_phase7.py')):
        shutil.copy2(src, ROOT/name)
    audit_record = {}
    for s, q, c in [('train',train,tc),('dev',dev,dc),('test',test,xc)]:
        audit_record[s] = {'queries':len(q), 'candidates':len(c),
            'query_words_median':float(np.median([len(v['text'].split()) for v in q.values()])),
            'candidate_words_p95':float(np.percentile([len(v['text'].split()) for v in c.values()],95)),
            'query_exact_duplicates':len(q)-len({normalized_hash(v['text']) for v in q.values()})}
        if s != 'test':
            audit_record[s]['gold_counts'] = dict(Counter(len(v['gold']) for v in q.values()))
            total = sum(len(v['gold']) for v in q.values())
            audit_record[s]['oracle_f1_at7'] = 2*sum(min(7,len(v['gold'])) for v in q.values())/max(1,7*len(q)+total)
    audit_record['cross_split_query_text_overlap'] = {
        f'{a}_{b}':len({normalized_hash(v['text']) for v in qa.values()} &
                       {normalized_hash(v['text']) for v in qb.values()})
        for a,qa,b,qb in [('train',train,'dev',dev),('train',train,'test',test),('dev',dev,'test',test)]}
    # If dev duplicates a fitted train query, its score must not select a model.
    fitted_hashes = {normalized_hash(train[k]['text']) for k in split['fit']}
    if split['tune_pool'] == 'dev' and any(normalized_hash(dev[k]['text']) in fitted_hashes for k in split['tune']+split['audit']):
        raise RuntimeError('Train/dev duplicate queries found. Resolve the split audit before training.')
    u.write_json(ROOT/'data_audit.json', audit_record); log('PREPARED', audit_record)
    for slot in SLOTS:
        source=ROOT/'submissions'/slot/'submission.zip'
        if source.is_file():shutil.copy2(source,WORK/f'CS781_Phase7_Submission_{slot}.zip')


def rows_for(stage):
    config = strict_json(ROOT/'run_config.json')['split']
    pool = 'train' if stage == 'fit' else config[f'{stage}_pool']
    q, c = data(pool)
    return pool, [q[k] for k in config[stage]], c


def bm25_matrix(texts, vocabulary=None):
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy import sparse
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w\w+\b',
                                 vocabulary=vocabulary, dtype=np.float32)
    counts = vectorizer.fit_transform(texts).tocsr()
    length = np.asarray(counts.sum(1)).ravel(); avg = max(1.,float(length.mean()))
    df = np.asarray((counts > 0).sum(0)).ravel()
    idf = np.log(1 + (len(texts)-df+.5)/(df+.5))
    rows = np.repeat(np.arange(len(texts)),np.diff(counts.indptr))
    counts.data = (counts.data*2.2/(counts.data+1.2*(.25+.75*length[rows]/avg)) * idf[counts.indices])
    return vectorizer, counts, sparse


@lru_cache(maxsize=3)
def lexical(split):
    output = ROOT/'lexical'/f'{split}.pt'
    if output.is_file(): return torch.load(output,map_location='cpu',weights_only=False)
    queries, candidates = data(split); ids = sorted(candidates)
    dv, dm, _ = bm25_matrix([candidates[k]['text'] for k in ids])
    passages, owners = [], []
    for i, key in enumerate(ids):
        words = candidates[key]['text'].split()
        for start in range(0,max(1,len(words)),240):
            passages.append(' '.join(words[start:start+300])); owners.append(i)
    pv, pm, _ = bm25_matrix(passages); owners = np.asarray(owners)
    result = {'candidate_ids':ids,'queries':{}}
    for i,(key,row) in enumerate(queries.items()):
        def query(v):
            q = v.transform([row['text']]).astype(np.float32); q.data = np.log1p(q.data)
            return q
        doc = (query(dv) @ dm.T).toarray().ravel()
        passage = np.zeros(len(ids),dtype=np.float32)
        scores = (query(pv) @ pm.T).tocoo()
        np.maximum.at(passage,owners[scores.col],scores.data)
        order = lambda a:np.argsort(-a,kind='stable')[:min(TOP_CACHE,len(a))].tolist()
        result['queries'][key] = {'doc':order(doc),'passage':order(passage)}
        if (i+1)%100 == 0: log('BM25',split,i+1,'/',len(queries))
    output.parent.mkdir(parents=True,exist_ok=True);u.save_torch(output,result)
    if split != 'test':
        pred = {k:[ids[j] for j in v['doc'][:7]] for k,v in result['queries'].items()}
        gold = {k:r['gold'] for k,r in queries.items()}
        metric = official.score(gold,pred)
        ceilings = {}
        for budget in (50,100,200,400):
            recovered = [min(7,len(set(r['gold']) & {ids[j] for j in
                set(result['queries'][k]['doc'][:budget])|set(result['queries'][k]['passage'][:budget])}))
                         for k,r in queries.items()]
            ceilings[budget] = 2*sum(recovered)/max(1,7*len(gold)+sum(map(len,gold.values())))
        u.write_json(ROOT/'lexical'/f'{split}_baseline.json',{'metric':metric,'lexical_union_oracle_f1':ceilings})
        log('BASELINE',split,metric,ceilings)
    return result


def selected_chunks(text, tok):
    ids = tok.encode(text); width = CHUNK_TOKENS-2
    if not ids: ids = [tok.eos_token_id]
    starts = list(range(0,max(1,len(ids)-width+1),width-OVERLAP))
    starts = sorted(set(starts+[max(0,len(ids)-width)]))
    chosen = {starts[0],starts[-1]}
    for target in np.linspace(0,max(0,len(ids)-width),min(CHUNKS,4)):
        chosen.add(min(starts,key=lambda x:abs(x-target)))
    if len(starts)>CHUNKS:
        ranked = sorted(starts,key=lambda x:len(CUE.findall(tok.decode(ids[x:x+width]))),reverse=True)
        for start in ranked:
            if len(chosen)>=CHUNKS:break
            chosen.add(start)
    else:chosen=set(starts)
    return [[tok.bos_token_id]+ids[start:start+width]+[tok.eos_token_id] for start in sorted(chosen)[:CHUNKS]]


def tokenize(base):
    path = ROOT/'tokens'/f'{base}.pt'
    if path.is_file():return
    _, tok = u.load_base(base); result = {}; memo = {}; lengths=[]
    for split in ('train','dev','test'):
        q,c = data(split)
        for kind,records_ in [('q',q),('c',c)]:
            for i,(key,row) in enumerate(records_.items()):
                digest=normalized_hash(row['text'])
                if digest not in memo:memo[digest]=selected_chunks(row['text'],tok)
                result[f'{split}/{kind}/{key}']=memo[digest]
                lengths.append(len(memo[digest]))
                if (i+1)%1000==0:log('TOKENIZE',base,split,kind,i+1,'/',len(records_))
    result['_pad']=tok.pad_token_id
    path.parent.mkdir(parents=True,exist_ok=True);u.save_torch(path,result)
    u.write_json(ROOT/'tokens'/f'{base}_audit.json',{'case_views':len(lengths),
        'chunks_mean':float(np.mean(lengths)),'at_chunk_limit':sum(x==CHUNKS for x in lengths),
        'policy':'deterministic coverage + legal cues; identical train/inference'})


class LoRMA(nn.Module):
    def __init__(self,base,rank=16):
        super().__init__();self.base=base
        self.a=nn.Linear(base.out_features,rank,bias=False)
        self.b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5));nn.init.zeros_(self.b.weight)
    def forward(self,x):
        raw=F.linear(x,self.base.weight)
        return self.base(x)+self.b(self.a(raw)) # (I+BA)W x + original bias


class IA3(nn.Module):
    def __init__(self,base,input_gate=False):
        super().__init__();self.base=base;self.input_gate=input_gate
        self.scale=nn.Parameter(torch.ones(base.in_features if input_gate else base.out_features))
    def forward(self,x):
        return self.base(x*self.scale) if self.input_gate else self.base(x)*self.scale


class Encoder(nn.Module):
    def __init__(self,base,base_id,method):
        super().__init__();self.base=base;self.base_id=base_id;self.method=method
        if base_id=='02':
            for attr in ('decoder_blocks','decoder_position_embedding','decoder_final_norm','output_bias'):
                if hasattr(base,attr):delattr(base,attr)
        for p in base.parameters():p.requires_grad_(False)
        self.first_trainable=0
        width=int(base.config.hidden_size) if base_id=='01' else int(base.config['hidden_size']) if base_id=='04' else base.hidden_size
        if method in ('full','distill','gold_control'):
            for name,p in base.named_parameters():
                if 'output_bias' not in name:p.requires_grad_(True)
        elif method=='partial':
            self.first_trainable=max(0,len(base.blocks)-4)
            for block in base.blocks[self.first_trainable:]:
                for p in block.parameters():p.requires_grad_(True)
            for p in base.final_norm.parameters():p.requires_grad_(True)
        elif method in ('lorma','ia3'):
            count=0
            for name,module in list(base.named_modules()):
                if not isinstance(module,nn.Linear):continue
                leaf=name.rsplit('.',1)[-1]
                eligible=(leaf in ('q_proj','v_proj','o_proj','down_proj') if method=='lorma' else
                          ('self_attention.' in name and leaf in ('key','value')) or name.endswith('feed_forward.output'))
                if eligible:
                    parent=base.get_submodule(name.rsplit('.',1)[0]);attr=leaf
                    setattr(parent,attr,LoRMA(module) if method=='lorma' else IA3(module,name.endswith('feed_forward.output')))
                    count+=1
            if count<6:raise RuntimeError(f'No meaningful {method} adaptation')
        elif method!='prompt':raise ValueError(method)
        self.prompt=nn.Parameter(torch.randn(PROMPT_TOKENS,width)*.02*math.sqrt(width)) if method=='prompt' else None
        self.project=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,DIM))
        self.use_checkpoint=False

    def hidden(self,ids,mask):
        b=self.base
        if self.base_id=='02':return b.encode(ids,mask)[0],mask
        if self.base_id=='01':
            x=b.embedding_dropout(b.token_embedding(ids)+b.position_embedding(torch.arange(ids.size(1),device=ids.device))[None])
            for i,block in enumerate(b.blocks):
                if self.method=='partial' and i<self.first_trainable:
                    with torch.no_grad():x=block(x)
                else:x=checkpoint(block,x,use_reentrant=False) if self.use_checkpoint and self.training else block(x)
        elif self.base_id=='03':
            x=b.token_embedding(ids)*math.sqrt(b.hidden_size)
            if self.prompt is not None:
                x=torch.cat((self.prompt[None].expand(len(ids),-1,-1),x),1)
                mask=torch.cat((torch.ones(len(ids),PROMPT_TOKENS,device=mask.device,dtype=torch.bool),mask),1)
            x=b.embedding_dropout(x);vis=b._prefix_hybrid_visibility(mask,mask.sum(1))
            co,si=b.rotary(x.size(1),x.dtype)
            for block in b.blocks:
                x=checkpoint(block,x,vis,co,si,use_reentrant=False) if self.use_checkpoint and self.training else block(x,vis,co,si)
            if self.prompt is not None:x=x[:,PROMPT_TOKENS:];mask=mask[:,PROMPT_TOKENS:]
        else:
            x=b.embedding(ids)
            for block in b.blocks:
                x=checkpoint(block,x,mask,use_reentrant=False) if self.use_checkpoint and self.training else block(x,mask)
        return b.final_norm(x),mask

    def forward(self,ids,mask,counts):
        hidden,mask=self.hidden(ids,mask)
        if self.base_id in ('01','04'):
            features=hidden[torch.arange(len(ids),device=ids.device),mask.sum(1)-1]
        else:features=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None].clamp_min(1)
        vectors=F.normalize(self.project(features).float(),dim=-1)
        groups=torch.split(vectors,counts)
        chunks=torch.stack([F.pad(g,(0,0,0,CHUNKS-len(g))) for g in groups])
        cmask=torch.arange(CHUNKS,device=ids.device)[None] < torch.tensor(counts,device=ids.device)[:,None]
        docs=F.normalize(chunks.sum(1)/cmask.sum(1)[:,None],dim=-1)
        return docs,chunks,cmask


def model_for(slot,device):
    seed_all(SEED+stable('06' if slot=='control' else slot)%10000)
    spec=SLOTS[slot];base,tok=u.load_base(spec['base'])
    model=Encoder(base,spec['base'],spec['method']).to(device)
    return model,tok


def trainable_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}


def load_state(model,state):
    expected={k for k,p in model.named_parameters() if p.requires_grad}
    if set(state)!=expected:raise RuntimeError('Checkpoint trainable names differ')
    with torch.no_grad():
        for k,p in model.named_parameters():
            if k in state:p.copy_(state[k])


def pack(keys,cache,device):
    groups=[cache[k] for k in keys];counts=[len(x) for x in groups]
    seq=[x for g in groups for x in g];width=max(map(len,seq));pad=cache['_pad']
    ids=torch.full((len(seq),width),pad,dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,x in enumerate(seq):ids[i,:len(x)]=torch.tensor(x);mask[i,:len(x)]=True
    return ids.to(device),mask.to(device),counts


def score_vectors(q,c):
    qdoc,qchunk,qmask=q;cdoc,cchunk,cmask=c
    dots=torch.einsum('bid,njd->bnij',qchunk,cchunk)
    dots=dots.masked_fill(~cmask[None,:,None,:],-1e4).max(-1).values
    late=(dots*qmask[:,None,:]).sum(-1)/qmask.sum(-1)[:,None]
    return .5*(qdoc@cdoc.T)+.5*late


def gold_loss(scores,qrows,cids):
    pos=torch.tensor([[k in set(r['gold']) for k in cids] for r in qrows],device=scores.device)
    if not pos.any(1).all():raise RuntimeError('Training query has no sampled positive')
    valid=(~pos).any(1)
    if not valid.any():return scores.sum()*0
    scores=scores[valid];pos=pos[valid]
    scaled=scores/TEMPERATURE
    neg=torch.logsumexp(scaled.masked_fill(pos,-float('inf')),dim=1,keepdim=True)
    per=torch.logaddexp(scaled,neg)-scaled
    return (per.masked_fill(~pos,0).sum(1)/pos.sum(1)).mean()


def kd_loss(scores,qrows,cids,teacher):
    losses=[]
    for i,row in enumerate(qrows):
        values=teacher.get(row['id'],{})
        indices=[j for j,k in enumerate(cids) if k in values]
        if len(indices)<2:continue
        target=torch.tensor([values[cids[j]] for j in indices],device=scores.device)
        losses.append(F.kl_div(F.log_softmax(scores[i,indices]/TEMPERATURE,dim=-1),
                              F.softmax(target/.5,dim=-1),reduction='sum'))
    return torch.stack(losses).mean() if losses else scores.sum()*0


def device():return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def amp_context(dev):
    dtype=torch.bfloat16 if dev.type=='cuda' and torch.cuda.get_device_capability(dev)[0]>=8 else torch.float16
    return torch.autocast(device_type=dev.type,dtype=dtype,enabled=dev.type=='cuda')


@lru_cache(maxsize=1)
def training_queries():
    _,rows,_=rows_for('fit')
    groups={}
    for row in rows:
        key=normalized_hash(row['text'])
        if key not in groups:groups[key]=dict(row,gold=set(row['gold']))
        else:groups[key]['gold'].update(row['gold'])
    return {r['id']:dict(r,gold=sorted(r['gold'])) for r in groups.values()}


@lru_cache(maxsize=1)
def examples():
    return [(r['id'],i) for r in training_queries().values() for i in range(0,len(r['gold']),2)]


@lru_cache(maxsize=1)
def mining_bank():
    path=ROOT/'mining'/'negatives.json'
    return strict_json(path) if path.is_file() else {}


def training_batch(items,epoch,cache,teacher=None,use_mined=True):
    queries=training_queries();lex=lexical('train');ids=lex['candidate_ids']
    mined=mining_bank() if use_mined else {}
    qrows=[];cids=[]
    for key,offset in items:
        row=queries[key];qrows.append(row);gold=set(row['gold'])
        sampled=list(row['gold'][offset:offset+2]);rng=random.Random(SEED+stable(key)+epoch*101)
        bank=[ids[j] for j in lex['queries'][key]['doc'][:100] if ids[j] not in gold]
        bank=[x for x in bank if x not in gold]
        if bank:sampled+=rng.sample(bank,min(2,len(bank)))
        dense=[x for x in mined.get(key,[]) if x not in gold]
        if dense:sampled.append(rng.choice(dense))
        else:
            available=[x for x in ids if x not in gold]
            if available:sampled.append(rng.choice(available))
        cids.extend(sampled)
    cids=list(dict.fromkeys(cids))
    return qrows,cids,[f"train/q/{r['id']}" for r in qrows],[f'train/c/{k}' for k in cids]


def profile(slot,model,tok,cache,dev):
    output=ROOT/'performance'/f"{'06' if slot=='control' else slot}.json"
    if output.is_file() and strict_json(output).get('probe_version')==2:return strict_json(output)
    if dev.type!='cuda':
        value={'microbatch':1,'gradient_checkpointing':False,'probe_version':2};u.write_json(output,value);return value
    items=sorted(examples(),key=lambda a:sum(map(len,cache[f'train/q/{a[0]}'])),reverse=True)
    qbank=list(dict.fromkeys(f'train/q/{q}' for q,_ in items))
    cbank=sorted((k for k in cache if k.startswith('train/c/')),
                 key=lambda k:(len(cache[k]),sum(map(len,cache[k]))),reverse=True)
    candidates=[]
    for check in (False,True):
        model.use_checkpoint=check
        if hasattr(model.base,'set_gradient_checkpointing'):model.base.set_gradient_checkpointing(check)
        for batch in (1,2,4,8,16,32):
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            try:
                # Five distinct candidates per query, all at the longest
                # available view lengths. A random sampled bank understated
                # real training memory in the original probe.
                qkeys=(qbank*(batch//len(qbank)+1))[:batch]
                ckeys=(cbank*(5*batch//len(cbank)+1))[:5*batch]
                qargs=pack(qkeys,cache,dev);cargs=pack(ckeys,cache,dev)
                elapsed=[]
                for iteration in range(3):
                    model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
                    with amp_context(dev):loss=score_vectors(model(*qargs),model(*cargs)).square().mean()
                    loss.backward();torch.cuda.synchronize()
                    if iteration:elapsed.append(time.monotonic()-start)
                    del loss
                memory=torch.cuda.max_memory_reserved()/torch.cuda.get_device_properties(0).total_memory
                speed=batch/max(.001,float(np.median(elapsed)))
                log('PROBE',slot,'batch',batch,'checkpointing',check,'queries/sec',round(speed,2),'memory_fraction',round(memory,3))
                if memory<.80:candidates.append((speed,batch,check))
                if memory>.85:break
            except torch.cuda.OutOfMemoryError:
                model.zero_grad(set_to_none=True);log('PROBE OOM',slot,batch,check);break
        if candidates and max(x[1] for x in candidates)>=4:break
    if not candidates:raise RuntimeError(f'No safe training batch fits for slot {slot}')
    chosen=max(candidates,key=lambda x:x[0]);value={'microbatch':chosen[1],'probe_version':2,
        'gradient_checkpointing':chosen[2],'queries_per_second':chosen[0],
        'note':'single GPU per job; two independent jobs use the two GPUs concurrently'}
    model.zero_grad(set_to_none=True);u.write_json(output,value);return value


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'])


def save_training(folder,model,optimizer,scaler,progress):
    target=folder/'latest.pt'
    if target.is_file():
        temporary=folder/'previous.tmp';shutil.copy2(target,temporary);u.atomic_replace(temporary,folder/'previous.pt')
    u.save_torch(target,{'model':trainable_state(model),'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict(),'rng':rng_state(),'progress':dict(progress)})
    u.write_json(folder/'status.json',progress)


def train(slot,max_minutes=18.):
    dev=device();spec=SLOTS[slot];folder=ROOT/'runs'/slot;folder.mkdir(parents=True,exist_ok=True)
    cache=torch.load(ROOT/'tokens'/f"{spec['base']}.pt",map_location='cpu',weights_only=False)
    model,tok=model_for(slot,dev)
    measured=profile(slot,model,tok,cache,dev);batch=measured['microbatch']
    model.use_checkpoint=measured['gradient_checkpointing']
    if hasattr(model.base,'set_gradient_checkpointing'):model.base.set_gradient_checkpointing(model.use_checkpoint)
    backbone=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('project.')]
    head=list(model.project.parameters())
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':spec['lr']},
                                 {'params':head,'lr':spec['head_lr']}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda' and torch.cuda.get_device_capability(dev)[0]<8)
    progress={'epoch':0,'cursor':0,'steps':0,'pending_eval':False,'complete':False,
              'best_tp':-1,'best_epoch':0,'no_improve':0,'microbatch':batch,
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'active_parameters':sum(p.numel() for p in model.parameters())}
    latest=folder/'latest.pt'
    if latest.is_file():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        load_state(model,state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        progress=state['progress'];restore_rng(state['rng'])
    else:seed_all(SEED+stable('06' if slot=='control' else slot)%10000)
    if progress['complete'] or progress['pending_eval']:
        u.write_json(folder/'status.json',progress);return
    effective=progress.get('effective_batch',progress['microbatch']*ACCUM)
    old_batch=progress['microbatch']
    if batch>old_batch:
        batch=old_batch;measured['microbatch']=batch
        u.write_json(ROOT/'performance'/f"{'06' if slot=='control' else slot}.json",measured)
    if old_batch!=batch:
        completed_examples=progress['cursor']*old_batch
        if completed_examples%batch or effective%batch:raise RuntimeError('Incompatible batch recovery boundary')
        progress['cursor']=completed_examples//batch
        progress['microbatch']=batch
        log('RESUME BATCH CONVERSION',old_batch,'->',batch,'completed examples',completed_examples)
    progress['effective_batch']=effective
    accumulation=max(1,effective//batch)
    negative_file=ROOT/'mining/negatives.json'
    if progress['cursor']==0:
        progress['use_mined']=negative_file.is_file()
        progress['negative_source_sha256']=u.sha(negative_file) if negative_file.is_file() else None
    elif progress.get('use_mined') and u.sha(negative_file)!=progress.get('negative_source_sha256'):
        raise RuntimeError('Negative bank changed during a resumed query pass')
    teacher=strict_json(ROOT/'teacher/scores.json')['scores'] if slot=='06' else None
    items=list(examples());random.Random(SEED+progress['epoch']).shuffle(items)
    batches=[items[i:i+batch] for i in range(0,len(items),batch)]
    if not batches:raise ValueError('No fitting query has a positive label')
    schedule_epochs=SLOTS['06']['epochs'] if slot=='control' else spec['epochs']
    total_steps=schedule_epochs*math.ceil(len(batches)/accumulation);warm=max(1,int(.08*total_steps))
    model.train();optimizer.zero_grad(set_to_none=True);start=last_save=time.monotonic();loss_sum=0.;seen=0
    for cursor in range(progress['cursor'],len(batches)):
        qrows,cids,qkeys,ckeys=training_batch(batches[cursor],progress['epoch'],cache,teacher,progress.get('use_mined',False))
        group_start=(cursor//accumulation)*accumulation
        divisor=min(accumulation,len(batches)-group_start)
        with amp_context(dev):
            scores=score_vectors(model(*pack(qkeys,cache,dev)),model(*pack(ckeys,cache,dev)))
            loss=gold_loss(scores,qrows,cids)
            if teacher:loss=loss+.15*kd_loss(scores,qrows,cids,teacher)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite retrieval loss; last checkpoint preserved')
        scaler.scale(loss/divisor).backward();loss_sum+=float(loss.detach());seen+=1
        if (cursor+1)%accumulation and cursor+1<len(batches):continue
        step=progress['steps']+1
        factor=min(1.,step/warm)*(.10+.90*.5*(1+math.cos(math.pi*min(1.,step/max(1,total_steps)))))
        optimizer.param_groups[0]['lr']=spec['lr']*factor;optimizer.param_groups[1]['lr']=spec['head_lr']*factor
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        progress.update(steps=step,cursor=cursor+1)
        if step%25==0:log('TRAIN PROGRESS',slot,'epoch',progress['epoch']+1,
                          'batch',cursor+1,'/',len(batches),'loss',round(loss_sum/max(1,seen),4))
        now=time.monotonic()
        if now-last_save>SAVE_SECONDS:
            save_training(folder,model,optimizer,scaler,progress);last_save=now
        if now-start>=max_minutes*60:break
    if progress['cursor']==len(batches):
        progress.update(epoch=progress['epoch']+1,cursor=0,pending_eval=True)
    save_training(folder,model,optimizer,scaler,progress)
    log('TRAIN',slot,'epoch',progress['epoch'],'cursor',progress['cursor'],'steps',progress['steps'],
        'loss',round(loss_sum/max(1,seen),4),'minutes',round((time.monotonic()-start)/60,2))


def load_model(slot,path,dev):
    model,tok=model_for(slot,dev);state=torch.load(path,map_location='cpu',weights_only=False)
    load_state(model,state['model']);model.eval();return model,tok,state


def checkpoint_path(slot,best=True):return ROOT/'runs'/slot/('best.pt' if best else 'latest.pt')


@lru_cache(maxsize=16)
def _checkpoint_digest(path,mtime,size):
    return u.sha(path)


def checkpoint_digest(path):
    info=Path(path).stat()
    return _checkpoint_digest(str(path),info.st_mtime_ns,info.st_size)


def embedding_path(slot,path,split,kind):
    digest=checkpoint_digest(path)[:20]
    return ROOT/'vectors'/slot/digest/f'{split}_{kind}.pt'


@torch.inference_mode()
def encode_records(slot,path,split,kind,model,cache,dev,deadline):
    output=embedding_path(slot,path,split,kind)
    if output.is_file():return torch.load(output,map_location='cpu',weights_only=False)
    queries,candidates=data(split);rows=queries if kind=='q' else candidates;ids=sorted(rows)
    partial=output.with_name(output.stem+'_partial.pt');output.parent.mkdir(parents=True,exist_ok=True)
    saved=torch.load(partial,map_location='cpu',weights_only=False) if partial.is_file() else {'ids':ids,'done':0,'chunks':[],'docs':[],'masks':[]}
    if saved['ids']!=ids:raise RuntimeError('Embedding ID list changed')
    measured=ROOT/'performance'/f'{slot}.json'
    batch=min(16,max(2,strict_json(measured)['microbatch']*2)) if measured.is_file() else 2
    last_save=time.monotonic()
    while saved['done']<len(ids):
        start=saved['done'];keys=[f'{split}/{kind}/{k}' for k in ids[start:start+batch]]
        try:
            with amp_context(dev):doc,chunk,mask=model(*pack(keys,cache,dev))
            saved['docs'].append(doc.cpu().half());saved['chunks'].append(chunk.cpu().half());saved['masks'].append(mask.cpu())
            saved['done']+=len(keys)
        except torch.cuda.OutOfMemoryError:
            if batch==1:raise
            batch=max(1,batch//2);torch.cuda.empty_cache();continue
        if saved['done']%64<len(keys) or time.monotonic()-last_save>120:
            u.save_torch(partial,saved);last_save=time.monotonic()
        if time.monotonic()>deadline:
            u.save_torch(partial,saved);log('ENCODING PAUSED',slot,split,kind,saved['done'],'/',len(ids));return None
    result={'ids':ids,'docs':torch.cat(saved['docs']),'chunks':torch.cat(saved['chunks']),'masks':torch.cat(saved['masks'])}
    u.save_torch(output,result);partial.unlink(missing_ok=True);return result


def subset_vectors(v,indices,dev):
    return tuple(v[k][indices].to(dev,dtype=torch.float32 if k!='masks' else torch.bool) for k in ('docs','chunks','masks'))


def metric_for(rows,ranking):
    return official.score({r['id']:r['gold'] for r in rows},ranking)


def sample_queries(rows,limit=EVAL_QUERIES):
    if len(rows)<=limit:return rows
    ordered=sorted(rows,key=lambda r:len(r['text']))
    return [ordered[int(i)] for i in np.linspace(0,len(ordered)-1,limit)]


def rank_queries(slot,path,split,rows,tag,max_minutes=18.,teacher_mode=False):
    output=ROOT/'rankings'/f'{slot}_{tag}.json'
    digest=checkpoint_digest(path)
    if output.is_file():
        result=strict_json(output)
        if result['checkpoint_sha256']!=digest:raise RuntimeError('Ranking cache belongs to another checkpoint')
        return result
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_name(output.stem+'_partial.json')
    existing=strict_json(partial) if partial.is_file() else {}
    if existing and existing['checkpoint_sha256']!=digest:raise RuntimeError('Partial ranking checkpoint changed')
    result=existing or {'checkpoint_sha256':digest,'split':split,'weights':{},'teacher_scores':{},'dense':{}}
    dev=device();model,tok,_=load_model(slot,path,dev)
    cache=torch.load(ROOT/'tokens'/f"{SLOTS[slot]['base']}.pt",map_location='cpu',weights_only=False)
    deadline=time.monotonic()+max_minutes*60
    cv=encode_records(slot,path,split,'c',model,cache,dev,deadline)
    if cv is None:return None
    qv=encode_records(slot,path,split,'q',model,cache,dev,deadline)
    if qv is None:return None
    del model;torch.cuda.empty_cache() if dev.type=='cuda' else None
    lex=lexical(split);cids=cv['ids']
    if cids!=lex['candidate_ids']:raise RuntimeError('Lexical and dense ID order differ')
    vector_bytes=sum(cv[k].numel()*4 for k in ('docs','chunks','masks'))
    resident=None
    if dev.type=='cpu' or vector_bytes<.35*torch.cuda.get_device_properties(dev).total_memory:
        resident=subset_vectors(cv,slice(None),dev)
    qindex={k:i for i,k in enumerate(qv['ids'])}
    for weight in (.25,.5,.75,1.):result['weights'].setdefault(str(weight),{})
    started=last_save=time.monotonic()
    for row in rows:
        key=row['id']
        if key in result['dense']:continue
        q=subset_vectors(qv,[qindex[key]],dev);scores=[]
        with torch.inference_mode():
            for start in range(0,len(cids),512):
                c=tuple(x[start:start+512] for x in resident) if resident is not None else subset_vectors(cv,slice(start,start+512),dev)
                scores.append(score_vectors(q,c).squeeze(0).cpu().numpy())
        dense=np.concatenate(scores);order=np.argsort(-dense,kind='stable')
        result['dense'][key]=[cids[i] for i in order[:min(TOP_CACHE,len(cids))]]
        dr=np.empty(len(cids),dtype=np.float32);dr[order]=np.arange(1,len(cids)+1)
        lexical_rank=np.zeros(len(cids),dtype=np.float32)
        for branch in ('doc','passage'):
            inds=lex['queries'][key][branch]
            lexical_rank[inds]+=.5/(60+np.arange(1,len(inds)+1))
        teacher_weight=.5
        if teacher_mode:
            teacher_weight=strict_json(ROOT/'reports'/f'submission_{slot}.json')['neural_weight']
        teacher_score=None
        for weight in (.25,.5,.75,1.):
            combined=weight/(60+dr)+(1-weight)*lexical_rank
            best=np.argsort(-combined,kind='stable')[:7]
            result['weights'][str(weight)][key]=[cids[i] for i in best]
            if weight==teacher_weight:teacher_score=combined
        if teacher_mode:
            chosen=set(order[:64]) | set(lex['queries'][key]['doc'][:100])
            chosen|={i for i,k in enumerate(cids) if k in set(row['gold'])}
            result['teacher_scores'][key]={cids[i]:float(teacher_score[i]*100) for i in chosen}
        if len(result['dense'])%50==0 or time.monotonic()-last_save>120:
            u.write_json(partial,result);last_save=time.monotonic()
        if time.monotonic()>deadline:
            u.write_json(partial,result);log('RANKING PAUSED',slot,tag,len(result['dense']),'/',len(rows));return None
    result['metrics']={w:metric_for(rows,p) for w,p in result['weights'].items()} if split!='test' else {}
    u.write_json(output,result);partial.unlink(missing_ok=True)
    log('RANKED',slot,tag,'queries',len(rows),'score_minutes',round((time.monotonic()-started)/60,2))
    return result


def evaluate(slot,max_minutes=18.):
    folder=ROOT/'runs'/slot;path=folder/'latest.pt'
    state=torch.load(path,map_location='cpu',weights_only=False);progress=state['progress']
    if not progress['pending_eval']:
        u.write_json(folder/'status.json',progress);return
    split,rows,_=rows_for('tune');rows=sample_queries(rows)
    result=rank_queries(slot,path,split,rows,f"epoch{progress['epoch']:02d}",max_minutes)
    if result is None:return
    weight=max(result['metrics'],key=lambda w:(result['metrics'][w]['tp'],-abs(float(w)-.5)))
    metric=result['metrics'][weight];previous=progress['best_tp']
    if metric['tp']>previous:
        progress.update(best_tp=metric['tp'],best_epoch=progress['epoch'])
        u.save_torch(folder/'best.pt',{'model':state['model'],'progress':dict(progress)})
        u.write_json(folder/'selected.json',{'metric':metric,'neural_weight':float(weight),
                                            'epoch':progress['epoch'],'queries':len(rows)})
        # Selected weights exactly equal this evaluation's weights. Reuse their
        # vectors under the new best-file digest instead of re-encoding them.
        old_vectors=ROOT/'vectors'/slot/result['checkpoint_sha256'][:20]
        new_vectors=ROOT/'vectors'/slot/checkpoint_digest(folder/'best.pt')[:20]
        if old_vectors.is_dir():
            new_vectors.mkdir(parents=True,exist_ok=True)
            for member in old_vectors.glob('*.pt'):
                if '_partial' not in member.name:shutil.copy2(member,new_vectors/member.name)
    progress['no_improve']=0 if metric['tp']>previous else progress['no_improve']+1
    progress['pending_eval']=False
    progress['complete']=progress['epoch']>=SLOTS[slot]['epochs'] or (progress['epoch']>=2 and progress['no_improve']>=PATIENCE)
    history_path=folder/'history.json';history=strict_json(history_path) if history_path.is_file() else []
    history=[h for h in history if h['epoch']!=progress['epoch']]
    history.append({'epoch':progress['epoch'],'steps':progress['steps'],'metric':metric,'neural_weight':float(weight)})
    u.write_json(history_path,history);state['progress']=progress;u.save_torch(path,state);u.write_json(folder/'status.json',progress)
    # Stale vector versions are regenerable. Keep only selected/latest caches.
    keep={u.sha(folder/'best.pt')[:20],u.sha(folder/'latest.pt')[:20]}
    vectors=ROOT/'vectors'/slot
    if vectors.is_dir():
        for item in vectors.iterdir():
            if item.is_dir() and item.name not in keep:
                if ROOT.resolve() not in item.resolve().parents:raise RuntimeError('Unsafe cache cleanup path')
                shutil.rmtree(item)
    log('EVALUATED',slot,'epoch',progress['epoch'],metric,'best',progress['best_epoch'],'complete',progress['complete'])


def mine(max_minutes=18.):
    output=ROOT/'mining/negatives.json'
    if output.is_file():return
    path=checkpoint_path('01')
    if not path.is_file():raise RuntimeError('Finish the primary retriever before hard-negative mining')
    _,rows,_=rows_for('fit')
    result=rank_queries('01',path,'train',rows,'mining_train',max_minutes)
    if result is None:return
    u.write_json(output,{r['id']:[k for k in result['dense'][r['id']] if k not in set(r['gold'])][:100] for r in rows})
    mining_bank.cache_clear()


def teacher(max_minutes=18.):
    output=ROOT/'teacher/scores.json'
    if output.is_file():return
    reports=[strict_json(ROOT/'reports'/f'submission_{s}.json') for s in ('01','02','03')]
    selected=max(reports,key=lambda r:r['tune_sample_metric']['tp']);slot=selected['slot']
    _,rows,_=rows_for('fit')
    result=rank_queries(slot,checkpoint_path(slot),'train',rows,'teacher_train',max_minutes,teacher_mode=True)
    if result is None:return
    u.write_json(output,{'slot':slot,'checkpoint_sha256':u.sha(checkpoint_path(slot)),
                         'queries':len(rows),'scores':result['teacher_scores']})
    log('TEACHER READY',slot,len(rows),'train queries')


def package(slot,max_minutes=18.):
    q,c=data('test');rows=list(q.values())
    result=rank_queries(slot,checkpoint_path(slot),'test',rows,'test',max_minutes)
    if result is None:return
    selected=strict_json(ROOT/'runs'/slot/'selected.json');weight=selected['neural_weight']
    prediction=result['weights'][str(weight)]
    errors=official.validate(prediction,{k:[] for k in q},set(c))
    if errors:raise RuntimeError(str(errors[:10]))
    folder=ROOT/'submissions'/slot;u.write_json(folder/'prediction.json',prediction)
    archive_path=WORK/f'CS781_Phase7_Submission_{slot}.zip';temporary=archive_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:z.write(folder/'prediction.json','prediction.json')
    u.atomic_replace(temporary,archive_path);shutil.copy2(archive_path,folder/'submission.zip')
    with zipfile.ZipFile(archive_path) as z:
        if z.namelist()!=['prediction.json'] or json.loads(z.read('prediction.json'))!=prediction:raise RuntimeError('ZIP roundtrip failed')
    status=strict_json(ROOT/'runs'/slot/'status.json')
    record={'slot':slot,**SLOTS[slot],'selected_epoch':selected['epoch'],
            'tune_sample_metric':selected['metric'],'neural_weight':weight,
            'test_queries':len(q),'prediction_sha256':u.sha(folder/'prediction.json'),
            'top7_sets_sha256':hashlib.sha256(json.dumps({k:sorted(v) for k,v in prediction.items()},sort_keys=True).encode()).hexdigest(),
            'checkpoint_sha256':u.sha(checkpoint_path(slot)),
            'trainable_parameters':status['trainable_parameters'],'active_parameters':status['active_parameters'],
            'private_score':None,'status':'READY_FOR_MANUAL_UPLOAD'}
    u.write_json(ROOT/'reports'/f'submission_{slot}.json',record);log('PACKAGED',slot,selected['metric'])


def advance(slot,max_minutes=18.):
    output=ROOT/'reports'/f'submission_{slot}.json'
    if slot!='control' and output.is_file():
        saved=ROOT/'submissions'/slot/'submission.zip';target=WORK/f'CS781_Phase7_Submission_{slot}.zip'
        if not target.is_file():shutil.copy2(saved,target)
        return
    state_path=ROOT/'runs'/slot/'status.json'
    state=strict_json(state_path) if state_path.is_file() else {}
    if not state.get('complete'):
        if state.get('pending_eval'):evaluate(slot,max_minutes)
        elif slot=='01' and state.get('epoch',0)>=1 and not (ROOT/'mining/negatives.json').is_file():mine(max_minutes)
        else:
            try:train(slot,max_minutes)
            except torch.cuda.OutOfMemoryError:
                profile_path=ROOT/'performance'/f"{'06' if slot=='control' else slot}.json"
                settings=strict_json(profile_path)
                if not settings['gradient_checkpointing']:
                    settings['gradient_checkpointing']=True
                elif settings['microbatch']>1:
                    settings['microbatch']=max(1,settings['microbatch']//2)
                else:raise
                settings['oom_retries']=settings.get('oom_retries',0)+1
                u.write_json(profile_path,settings)
                log('MEMORY RETRY SCHEDULED',slot,settings,'last checkpoint preserved')
                return 'retry'
    elif slot!='control':package(slot,max_minutes)


def score_selected(slot,stage,max_minutes=18.):
    output=ROOT/'scores'/f'{slot}_{stage}.json'
    if output.is_file():return
    split,rows,_=rows_for(stage)
    result=rank_queries(slot,checkpoint_path(slot),split,rows,stage+'_all',max_minutes)
    if result is None:return
    selected=strict_json(ROOT/'runs'/slot/'selected.json');weight=str(selected['neural_weight'])
    metric=result['metrics'][weight]
    u.write_json(output,{'slot':slot,'stage':stage,'metric':metric,'neural_weight':float(weight),'queries':len(rows)})
    log('FULL SCORE',slot,stage,metric)


def report():
    reports=[strict_json(ROOT/'reports'/f'submission_{s}.json') for s in SLOTS if s!='control' and (ROOT/'reports'/f'submission_{s}.json').is_file()]
    distinct=len({r['top7_sets_sha256'] for r in reports})==len(reports)
    scored=[]
    for r in reports:
        path=ROOT/'scores'/f"{r['slot']}_tune.json"
        if path.is_file():scored.append((strict_json(path)['metric']['primary_metric_1'],r['slot']))
    recommended=sorted(scored,key=lambda x:(-x[0],x[1]))[0][1] if scored else None
    control_history=ROOT/'runs/control/history.json';student_history=ROOT/'runs/06/history.json'
    comparison={}
    if control_history.is_file() and student_history.is_file():
        control={h['epoch']:h for h in strict_json(control_history)}
        for h in strict_json(student_history):
            if h['epoch'] in control:comparison[h['epoch']]={'gold_only':control[h['epoch']]['metric'],
                'kd':h['metric'],'steps_match':control[h['epoch']]['steps']==h['steps']}
    value={'status':'SIX_DISTINCT_PACKAGES_READY' if len(reports)==6 and distinct else 'PARTIAL_OR_REVIEW_REQUIRED',
           'reports':reports,'recommended_slot':recommended,'gold_control_comparison':comparison,'private_score':None,
           'training_gpu_strategy':'two independent GPU jobs, each with its own original Phase 1 adaptation'}
    audit_path=ROOT/'scores'/f'{recommended}_audit.json'
    value['audit_metric']=strict_json(audit_path)['metric'] if recommended and audit_path.is_file() else None
    value['full_tune_scores']={slot:score for score,slot in scored}
    u.write_json(ROOT/'report.json',value);log('REPORT',value['status'],'recommended',recommended)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','lexical','tokenize','advance','mine','teacher','score','report','snapshot'])
    parser.add_argument('--slot',choices=list(SLOTS),default='01');parser.add_argument('--base',default='03')
    parser.add_argument('--split',default='train');parser.add_argument('--stage',choices=['tune','audit'],default='tune')
    parser.add_argument('--max-minutes',type=float,default=18.);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='lexical':lexical(args.split)
    elif args.action=='tokenize':tokenize(args.base)
    elif args.action=='advance':
        import gc
        for attempt in range(8):
            if advance(args.slot,args.max_minutes)!='retry':break
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        else:raise RuntimeError('Memory recovery retry limit reached; checkpoints preserved')
    elif args.action=='mine':mine(args.max_minutes)
    elif args.action=='teacher':teacher(args.max_minutes)
    elif args.action=='score':score_selected(args.slot,args.stage,args.max_minutes)
    elif args.action=='report':report()
    elif args.action=='snapshot':u.snapshot()


if __name__=='__main__':main()
