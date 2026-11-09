"""Shared recovery, Phase-1 loading, and LORMA helpers for Phase 4."""
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

ROOT=Path('/kaggle/working/CS781_Capstone/phase4_cjpe')
INPUT=Path('/kaggle/input')
ARTIFACT=Path('/kaggle/working/CS781_Phase4_All_Artifacts.zip')
BASE_FILES={f'{i:02d}':f'phase1_submission{i:02d}_recovery.zip' for i in range(1,5)}
SOURCES={'01':'decoder_lm.py','02':'encoder_decoder_lm.py',
         '03':'prefix_hybrid_lm.py','04':'efficient_causal_lm.py'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path,value):
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)

    with tempfile.NamedTemporaryFile(
        'w',
        encoding='utf8',
        dir=target.parent,
        delete=False
    ) as handle:

        temporary=Path(handle.name)

        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False
        )

        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        temporary,
        target
    )


def save_torch(path,value):
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        delete=False
    ) as handle:
        temporary=Path(handle.name)

    try:
        torch.save(
            value,
            temporary
        )

        os.replace(
            temporary,
            target
        )

    finally:
        temporary.unlink(
            missing_ok=True
        )


def find_one(name,required=True):
    matches=sorted(
        {
            p.resolve()
            for p in INPUT.rglob(name)
            if p.is_file()
        }
    )

    if len(matches)>1 or (
        required and len(matches)!=1
    ):
        raise RuntimeError(
            f'Expected exactly one {name} under '
            f'{INPUT}; found {matches}'
        )

    return matches[0] if matches else None


def safe_name(name):
    path=Path(name)

    return (
        bool(name)
        and not path.is_absolute()
        and '..' not in path.parts
        and '\\' not in name
    )


# ============================================================
# EPOCH-PATCH RECOVERY
# Supports BOTH:
#   1) CS781_Phase4_Epoch*_Resume_Patch.zip
#   2) Kaggle-extracted dataset folders containing resume_info.json + stage/
# ============================================================

def _resume_patch_candidates():
    candidates=[]

    # ---------- ZIP patches ----------
    for patch in sorted(INPUT.rglob('CS781_Phase4_Epoch*_Resume_Patch.zip')):
        if not patch.is_file():
            continue
        with zipfile.ZipFile(patch) as archive:
            names=archive.namelist()
            if 'resume_info.json' not in names:
                continue
            info=json.loads(archive.read('resume_info.json'))
            epoch=int(info['epoch'])
            relative_stage=str(info['relative_stage_directory'])
            if not safe_name(relative_stage):
                raise RuntimeError(f'Unsafe resume stage in {patch}: {relative_stage}')
            candidates.append((epoch, 'zip', patch, info))

    # ---------- Extracted Kaggle dataset patches ----------
    for info_path in sorted(INPUT.rglob('resume_info.json')):
        if not info_path.is_file():
            continue
        patch_root=info_path.parent
        stage_root=patch_root/'stage'
        latest=stage_root/'latest.pt'
        if not latest.is_file():
            continue
        info=read_json(info_path)
        if 'epoch' not in info or 'relative_stage_directory' not in info:
            continue
        epoch=int(info['epoch'])
        relative_stage=str(info['relative_stage_directory'])
        if not safe_name(relative_stage):
            raise RuntimeError(f'Unsafe extracted resume stage in {info_path}: {relative_stage}')
        candidates.append((epoch, 'dir', patch_root, info))

    return candidates


def restore_epoch_patch():
    candidates=_resume_patch_candidates()

    if not candidates:
        print('No epoch resume patch found.', flush=True)
        return None

    candidates.sort(
        key=lambda item:(
            item[0],
            item[2].stat().st_mtime
        )
    )

    saved_epoch,kind,source,info=candidates[-1]
    relative_stage=Path(info['relative_stage_directory'])
    stage_dir=ROOT/relative_stage
    stage_dir.mkdir(parents=True,exist_ok=True)

    print('Found epoch resume patch:', source, flush=True)
    print('Patch type:', kind, flush=True)
    print('Patch target stage:', relative_stage.as_posix(), flush=True)

    if kind=='zip':
        with zipfile.ZipFile(source) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)):
                raise RuntimeError(f'Duplicate members in resume patch: {source}')
            for member in ('stage/latest.pt','stage/best.pt','stage/history.json'):
                if member not in names:
                    continue
                target=stage_dir/Path(member).name
                with archive.open(member) as reader, tempfile.NamedTemporaryFile(dir=stage_dir,delete=False) as handle:
                    temporary=Path(handle.name)
                    shutil.copyfileobj(reader,handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary,target)
                print('Restored patch file:', target, flush=True)
    else:
        src_stage=source/'stage'
        for filename in ('latest.pt','best.pt','history.json'):
            src=src_stage/filename
            if not src.is_file():
                continue
            target=stage_dir/filename
            with src.open('rb') as reader, tempfile.NamedTemporaryFile(dir=stage_dir,delete=False) as handle:
                temporary=Path(handle.name)
                shutil.copyfileobj(reader,handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary,target)
            print('Restored patch file:', target, flush=True)

    latest=stage_dir/'latest.pt'
    if not latest.is_file():
        raise RuntimeError(f'Resume patch {source} did not restore latest.pt')

    try:
        state=torch.load(latest,map_location='cpu',weights_only=False)
    except TypeError:
        state=torch.load(latest,map_location='cpu')

    actual_epoch=int(state['epoch'])
    if actual_epoch!=saved_epoch:
        raise RuntimeError(
            f'Resume patch epoch mismatch: metadata={saved_epoch}, checkpoint={actual_epoch}'
        )

    print('Restored Phase-4 epoch patch:', source, flush=True)
    print('Resume stage:', relative_stage.as_posix(), flush=True)
    print(
        f'Checkpoint completed epoch {actual_epoch}; training will start from epoch {actual_epoch+1}.',
        flush=True
    )
    return actual_epoch


# ============================================================
# MAIN PHASE-4 ARTIFACT RECOVERY
# Supports BOTH:
#   1) CS781_Phase4_All_Artifacts.zip
#   2) Kaggle-extracted dataset folder containing integrity.json + run_config.json
# ============================================================

def _find_extracted_main_artifact():
    candidates=[]
    for integrity in sorted(INPUT.rglob('integrity.json')):
        root=integrity.parent
        if not (root/'run_config.json').is_file():
            continue
        # Strong signals that this is the Phase-4 artifact root.
        if not (root/'runs').is_dir():
            continue
        candidates.append(root)

    if not candidates:
        return None
    if len(candidates)>1:
        # Prefer a folder whose name looks like the intended dataset.
        preferred=[p for p in candidates if 'phase4' in p.name.lower() and 'artifact' in p.name.lower()]
        if len(preferred)==1:
            return preferred[0]
        raise RuntimeError(f'Multiple extracted Phase-4 artifact roots found: {candidates}')
    return candidates[0]


def _restore_extracted_artifact(source):
    manifest_path=source/'integrity.json'
    manifest=read_json(manifest_path)['files']

    print('Found extracted Phase-4 artifact:', source, flush=True)

    for relative,meta in manifest.items():
        if not safe_name(relative):
            raise RuntimeError(f'Unsafe recovery member {relative}')
        src=source/relative
        if not src.is_file():
            raise RuntimeError(f'Missing extracted recovery member: {src}')
        if src.stat().st_size!=int(meta['bytes']):
            raise RuntimeError(f'Size mismatch in extracted recovery member: {src}')
        if sha(src)!=meta['sha256']:
            raise RuntimeError(f'Hash mismatch in extracted recovery member: {src}')

        target=ROOT/relative
        target.parent.mkdir(parents=True,exist_ok=True)

        if target.exists():
            if sha(target)!=meta['sha256']:
                raise RuntimeError(f'Existing file conflicts with recovery: {relative}')
            continue

        with src.open('rb') as reader, tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
            temporary=Path(handle.name)
            shutil.copyfileobj(reader,handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,target)

    print('Restored extracted Phase-4 artifact:', source, flush=True)


def restore_artifact():
    # Restore main Phase-4 state only when working state is not already present.
    if not (ROOT/'run_config.json').is_file():
        zip_source=find_one(ARTIFACT.name,required=False)

        if zip_source is not None:
            with zipfile.ZipFile(zip_source) as archive:
                names=archive.namelist()
                if 'integrity.json' not in names or len(names)!=len(set(names)):
                    raise RuntimeError('Invalid Phase-4 recovery ZIP manifest')

                manifest=json.loads(archive.read('integrity.json'))['files']
                if set(manifest)!=set(names)-{'integrity.json'}:
                    raise RuntimeError('Phase-4 recovery member list differs from manifest')

                for name in names:
                    if name=='integrity.json':
                        continue
                    if not safe_name(name) or archive.getinfo(name).is_dir():
                        raise RuntimeError(f'Unsafe recovery member {name}')

                    target=ROOT/name
                    target.parent.mkdir(parents=True,exist_ok=True)

                    with archive.open(name) as reader, tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
                        temporary=Path(handle.name)
                        digest=hashlib.sha256()
                        size=0
                        for block in iter(lambda:reader.read(4<<20),b''):
                            handle.write(block)
                            digest.update(block)
                            size+=len(block)
                        handle.flush()
                        os.fsync(handle.fileno())

                    if digest.hexdigest()!=manifest[name]['sha256'] or size!=manifest[name]['bytes']:
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError(f'Corrupt recovery member {name}')

                    if target.exists():
                        if sha(target)!=digest.hexdigest():
                            temporary.unlink(missing_ok=True)
                            raise RuntimeError(f'Existing file conflicts with recovery: {name}')
                        temporary.unlink()
                    else:
                        os.replace(temporary,target)

            print('Restored Phase-4 artifact:',zip_source,flush=True)

        else:
            extracted=_find_extracted_main_artifact()
            if extracted is None:
                print('No Phase-4 all-artifacts ZIP or extracted artifact found; starting without main Phase-4 recovery.',flush=True)
            else:
                _restore_extracted_artifact(extracted)
    else:
        print('Existing Phase-4 working state found; main artifact restore skipped.',flush=True)

    # Always overlay the newest/highest epoch patch after main restore.
    restore_epoch_patch()


# ============================================================
# SNAPSHOT
# ============================================================

def snapshot():

    ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    files=sorted(
        p
        for p in ROOT.rglob('*')
        if (
            p.is_file()
            and
            'hf_cache'
            not in p.parts
        )
    )

    temporary=ARTIFACT.with_suffix(
        '.zip.tmp'
    )

    manifest={
        'version':1,
        'files':{}
    }

    with zipfile.ZipFile(
        temporary,
        'w',
        allowZip64=True
    ) as archive:

        for path in files:

            relative=(
                path
                .relative_to(ROOT)
                .as_posix()
            )

            if relative.endswith(
                '.tmp'
            ):
                continue

            manifest[
                'files'
            ][relative]={
                'sha256':sha(path),
                'bytes':path.stat().st_size
            }

            archive.write(
                path,
                relative,
                compress_type=(
                    zipfile.ZIP_STORED
                    if path.suffix=='.pt'
                    else
                    zipfile.ZIP_DEFLATED
                )
            )

        archive.writestr(
            'integrity.json',
            json.dumps(
                manifest,
                sort_keys=True
            )
        )

    os.replace(
        temporary,
        ARTIFACT
    )

    print(
        f'Updated {ARTIFACT} '
        f'({ARTIFACT.stat().st_size/2**30:.2f} GiB)',
        flush=True
    )


# ============================================================
# PHASE-1 BASE RECOVERY
# ============================================================

def stage_base(base_id):

    dest=(
        ROOT
        /
        'bases'
        /
        base_id
    )

    needed=[
        f'model/{SOURCES[base_id]}',
        'model/course_bpe.py',
        'checkpoints/selected_model.pt'
    ]

    if (
        all(
            (dest/name).is_file()
            for name in needed
        )
        and
        (
            dest
            /
            'tokenizer/tokenizer.json'
        ).is_file()
    ):

        return dest

    source=find_one(
        BASE_FILES[
            base_id
        ]
    )

    with zipfile.ZipFile(
        source
    ) as archive:

        names=archive.namelist()

        if (
            len(names)
            !=
            len(set(names))
            or
            any(
                not safe_name(name)
                for name in names
            )
        ):

            raise RuntimeError(
                f'Unsafe Phase-1 archive: '
                f'{source}'
            )

        wanted=(
            needed
            +
            [
                name
                for name in names
                if (
                    name.startswith(
                        'tokenizer/'
                    )
                    and
                    not name.endswith('/')
                )
            ]
        )

        if (
            not all(
                name in names
                for name in needed
            )
            or
            'tokenizer/tokenizer.json'
            not in wanted
        ):

            raise RuntimeError(
                f'Incomplete Phase-1 '
                f'archive: {source}'
            )

        for name in wanted:

            target=(
                dest
                /
                name
            )

            if target.is_file():
                continue

            target.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with (
                archive.open(name)
                as reader,
                tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    delete=False
                ) as handle
            ):

                temporary=Path(
                    handle.name
                )

                shutil.copyfileobj(
                    reader,
                    handle
                )

            os.replace(
                temporary,
                target
            )

    write_json(
        dest
        /
        'provenance.json',
        {
            'archive_name':
                source.name,

            'archive_sha256':
                sha(source),

            'model_sha256':
                sha(
                    dest
                    /
                    needed[0]
                ),

            'checkpoint_sha256':
                sha(
                    dest
                    /
                    needed[2]
                )
        }
    )

    return dest


def source_module(path,name):

    spec=importlib.util.spec_from_file_location(
        name,
        path
    )

    module=importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        name
    ]=module

    spec.loader.exec_module(
        module
    )

    return module


def load_base(base_id):

    directory=(
        ROOT
        /
        'bases'
        /
        base_id
    )

    model_path=(
        directory
        /
        'model'
        /
        SOURCES[
            base_id
        ]
    )

    source=source_module(
        model_path,
        f'phase4_model_'
        f'{base_id}_'
        f'{os.getpid()}'
    )

    tokenizer_source=source_module(
        directory
        /
        'model/course_bpe.py',

        f'phase4_bpe_'
        f'{base_id}_'
        f'{os.getpid()}'
    )

    tokenizer=(
        tokenizer_source
        .CourseBPETokenizer
        .from_pretrained(
            directory
            /
            'tokenizer'
        )
    )

    try:

        checkpoint=torch.load(
            directory
            /
            'checkpoints/selected_model.pt',

            map_location='cpu',
            weights_only=True
        )

    except TypeError:

        checkpoint=torch.load(
            directory
            /
            'checkpoints/selected_model.pt',

            map_location='cpu'
        )

    if (
        checkpoint.get(
            'model_source_sha256'
        )
        and
        checkpoint[
            'model_source_sha256'
        ]
        !=
        sha(model_path)
    ):

        raise RuntimeError(
            f'Phase-1 base '
            f'{base_id} '
            f'source hash mismatch'
        )

    metadata=read_json(
        directory
        /
        'tokenizer/'
        'tokenizer_metadata.json'
    )

    if (
        checkpoint.get(
            'tokenizer_fingerprint'
        )
        !=
        metadata.get(
            'fingerprint'
        )
    ):

        raise RuntimeError(
            f'Phase-1 base '
            f'{base_id} '
            f'tokenizer fingerprint mismatch'
        )

    if (
        int(
            checkpoint[
                'model_config'
            ][
                'vocab_size'
            ]
        )
        !=
        len(tokenizer)
    ):

        raise RuntimeError(
            f'Phase-1 base '
            f'{base_id} '
            f'vocabulary mismatch'
        )

    base=source.build_model(
        checkpoint[
            'model_config'
        ]
    )

    base.load_state_dict(
        checkpoint[
            'model_state'
        ],
        strict=True
    )

    return (
        base,
        tokenizer
    )


# ============================================================
# LORMA
# ============================================================

class LoRMA(nn.Module):

    def __init__(
        self,
        base,
        rank,
        alpha
    ):

        super().__init__()

        self.base=base

        self.a=nn.Linear(
            base.out_features,
            rank,
            bias=False
        )

        self.b=nn.Linear(
            rank,
            base.out_features,
            bias=False
        )

        nn.init.kaiming_uniform_(
            self.a.weight,
            a=math.sqrt(5)
        )

        nn.init.zeros_(
            self.b.weight
        )

        self.scale=(
            alpha
            /
            rank
        )

    def forward(
        self,
        x
    ):

        value=self.base(
            x
        )

        return (
            value
            +
            self.b(
                self.a(
                    F.linear(
                        x,
                        self.base.weight
                    )
                )
            )
            *
            self.scale
        )
