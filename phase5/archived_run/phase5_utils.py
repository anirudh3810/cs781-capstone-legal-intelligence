"""Shared recovery, Phase-1 loading, and LORMA helpers for Phase 5."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/kaggle/working/CS781_Capstone/phase5_lsi')
INPUT=Path('/kaggle/input')
ARTIFACT=Path('/kaggle/working/CS781_Phase5_All_Artifacts.zip')
BASE_FILES={f'{i:02d}':f'phase1_submission{i:02d}_recovery.zip' for i in range(1,5)}
SOURCES={'01':'decoder_lm.py','02':'encoder_decoder_lm.py',
         '03':'prefix_hybrid_lm.py','04':'efficient_causal_lm.py'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path,value):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=target.parent,delete=False) as handle:
        temporary=Path(handle.name)
        json.dump(value,handle,indent=2,ensure_ascii=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,target)


def save_torch(path,value):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:temporary=Path(handle.name)
    try:
        torch.save(value,temporary)
        os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)


def find_one(name,required=True):
    matches=sorted({p.resolve() for p in INPUT.rglob(name) if p.is_file()})
    if len(matches)>1 or (required and len(matches)!=1):
        raise RuntimeError(f'Expected exactly one {name} under {INPUT}; found {matches}')
    return matches[0] if matches else None


def safe_name(name):
    path=Path(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name


def restore_artifact():
    if (ROOT/'run_config.json').is_file():return
    source=find_one(ARTIFACT.name,required=False)
    if source is None:return
    with zipfile.ZipFile(source) as archive:
        names=archive.namelist()
        if 'integrity.json' not in names or len(names)!=len(set(names)):
            raise RuntimeError('Invalid Phase-5 recovery ZIP manifest')
        manifest=json.loads(archive.read('integrity.json'))['files']
        if set(manifest)!=set(names)-{'integrity.json'}:
            raise RuntimeError('Phase-5 recovery member list differs from manifest')
        for name in names:
            if name=='integrity.json':continue
            if not safe_name(name) or archive.getinfo(name).is_dir():
                raise RuntimeError(f'Unsafe recovery member {name}')
            target=ROOT/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(name) as reader, tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
                temporary=Path(handle.name);digest=hashlib.sha256();size=0
                for block in iter(lambda:reader.read(4<<20),b''):
                    handle.write(block);digest.update(block);size+=len(block)
            if digest.hexdigest()!=manifest[name]['sha256'] or size!=manifest[name]['bytes']:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f'Corrupt recovery member {name}')
            if target.exists():
                if sha(target)!=digest.hexdigest():
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f'Existing file conflicts with recovery: {name}')
                temporary.unlink()
            else:os.replace(temporary,target)
    print('Restored Phase-5 artifact:',source,flush=True)


def snapshot():
    ROOT.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and 'hf_cache' not in p.parts)
    temporary=ARTIFACT.with_suffix('.zip.tmp')
    manifest={'version':1,'files':{}}
    with zipfile.ZipFile(temporary,'w',allowZip64=True) as archive:
        for path in files:
            relative=path.relative_to(ROOT).as_posix()
            if relative.endswith('.tmp') or path.name.startswith('tmp'):continue
            manifest['files'][relative]={'sha256':sha(path),'bytes':path.stat().st_size}
            archive.write(path,relative,compress_type=zipfile.ZIP_STORED if path.suffix=='.pt' else zipfile.ZIP_DEFLATED)
        archive.writestr('integrity.json',json.dumps(manifest,sort_keys=True))
    os.replace(temporary,ARTIFACT)
    print(f'Updated {ARTIFACT} ({ARTIFACT.stat().st_size/2**30:.2f} GiB)',flush=True)


def stage_base(base_id):
    dest=ROOT/'bases'/base_id
    needed=[f'model/{SOURCES[base_id]}','model/course_bpe.py','checkpoints/selected_model.pt']
    if all((dest/name).is_file() for name in needed) and (dest/'tokenizer/tokenizer.json').is_file():return dest
    wanted_core=set(needed+['tokenizer/tokenizer.json'])
    candidates=[]
    for checkpoint_path in INPUT.rglob('selected_model.pt'):
        if checkpoint_path.parent.name!='checkpoints':continue
        root=checkpoint_path.parent.parent
        if all((root/name).is_file() for name in wanted_core):candidates.append(('directory',root))
    for source in INPUT.rglob('*.zip'):
        try:
            with zipfile.ZipFile(source) as archive:
                if wanted_core.issubset(archive.namelist()):candidates.append(('zip',source))
        except (OSError,zipfile.BadZipFile):continue
    # Prefer the expected archive name if a Kaggle mount exposes both its ZIP and extracted contents.
    named=[item for item in candidates if item[0]=='zip' and item[1].name==BASE_FILES[base_id]]
    if len(named)==1:kind,source=named[0]
    elif len(candidates)==1:kind,source=candidates[0]
    else:raise RuntimeError(f'Expected one identifiable Phase-1 base {base_id}; found {candidates}')
    if kind=='directory':
        wanted=needed+[p.relative_to(source).as_posix() for p in (source/'tokenizer').rglob('*') if p.is_file()]
        for name in wanted:
            target=dest/name
            if target.is_file():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
        archive_hash=None
    else:
        with zipfile.ZipFile(source) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or any(not safe_name(name) for name in names):
                raise RuntimeError(f'Unsafe Phase-1 archive: {source}')
            wanted=needed+[name for name in names if name.startswith('tokenizer/') and not name.endswith('/')]
            for name in wanted:
                target=dest/name
                if target.is_file():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as reader, tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
                    temporary=Path(handle.name);shutil.copyfileobj(reader,handle)
                os.replace(temporary,target)
        archive_hash=sha(source)
    write_json(dest/'provenance.json',{'source':str(source),'archive_sha256':archive_hash,
                 'model_sha256':sha(dest/needed[0]),'checkpoint_sha256':sha(dest/needed[2])})
    return dest


def source_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_base(base_id):
    directory=ROOT/'bases'/base_id
    model_path=directory/'model'/SOURCES[base_id]
    source=source_module(model_path,f'phase5_model_{base_id}_{os.getpid()}')
    tokenizer_source=source_module(directory/'model/course_bpe.py',f'phase5_bpe_{base_id}_{os.getpid()}')
    tokenizer=tokenizer_source.CourseBPETokenizer.from_pretrained(directory/'tokenizer')
    try:checkpoint=torch.load(directory/'checkpoints/selected_model.pt',map_location='cpu',weights_only=True)
    except TypeError:checkpoint=torch.load(directory/'checkpoints/selected_model.pt',map_location='cpu')
    if checkpoint.get('model_source_sha256') and checkpoint['model_source_sha256']!=sha(model_path):
        raise RuntimeError(f'Phase-1 base {base_id} source hash mismatch')
    metadata=read_json(directory/'tokenizer/tokenizer_metadata.json')
    if checkpoint.get('tokenizer_fingerprint')!=metadata.get('fingerprint'):
        raise RuntimeError(f'Phase-1 base {base_id} tokenizer fingerprint mismatch')
    if int(checkpoint['model_config']['vocab_size'])!=len(tokenizer):
        raise RuntimeError(f'Phase-1 base {base_id} vocabulary mismatch')
    base=source.build_model(checkpoint['model_config'])
    base.load_state_dict(checkpoint['model_state'],strict=True)
    return base,tokenizer


class LoRMA(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__()
        self.base=base
        self.a=nn.Linear(base.out_features,rank,bias=False)
        self.b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale=alpha/rank

    def forward(self,x):
        value=self.base(x)
        return value+self.b(self.a(F.linear(x,self.base.weight)))*self.scale

