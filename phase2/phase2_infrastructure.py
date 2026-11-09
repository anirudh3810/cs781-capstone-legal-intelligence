# %% Imports and safe run configuration
from __future__ import annotations

import ast
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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# Run All executes the six AI-assisted reference experiments when inputs exist.
# This is not a student-authorship attestation or a guarantee of course acceptance.
RUN_PHASE2 = True
STUDENT_AUTHORSHIP_CONFIRMED = False
GPU_SESSION_BUDGET_SECONDS = 31_800
INFERENCE_RECOVERY_INTERVAL = 250
SEED = 781

INPUT_ROOT = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd()
WORK_ROOT = Path('/kaggle/working/CS781_Capstone/phase2') if Path('/kaggle/working').exists() else Path.cwd() / 'CS781_Capstone/phase2/run'
WORK_ROOT.mkdir(parents=True, exist_ok=True)
RUN_MODE = 'auto'  # auto resumes verified work, or starts fresh; inference_only is optional

OFFICIAL_LABELS = frozenset({
    'APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT',
    'AUTH', 'COURT', 'STAT', 'PREC', 'DATE', 'CASENO',
})
DATA_NAMES = ('fold1-text.json', 'fold1-gold.json', 'fold2-text.json',
              'fold2-gold.json', 'labels.txt', 'fold3-text.json')
BASE_ARCHIVES = {f'{i:02d}': f'phase1_submission{i:02d}_recovery.zip' for i in range(1, 5)}
BASE_SOURCES = {
    '01': 'decoder_lm.py', '02': 'encoder_decoder_lm.py',
    '03': 'prefix_hybrid_lm.py', '04': 'efficient_causal_lm.py',
}
STAGED_BASES: dict[str, Path] = {}
RESTORED_RECOVERY_SLOTS: dict[int, Path] = {}

# Five distinct non-distillation methods and one trained-teacher distillation.
EXPERIMENTS = [
    {'slot': 1, 'base_id': '01', 'method': 'full_finetune', 'teacher_id': None,
     'module_filename': 'phase2_method_01.py'},
    {'slot': 2, 'base_id': '02', 'method': 'partial_finetune', 'teacher_id': None,
     'module_filename': 'phase2_method_02.py'},
    {'slot': 3, 'base_id': '03', 'method': 'lora', 'teacher_id': None,
     'module_filename': 'phase2_method_03.py'},
    {'slot': 4, 'base_id': '04', 'method': 'lorma', 'teacher_id': None,
     'module_filename': 'phase2_method_04.py'},
    {'slot': 5, 'base_id': '01', 'method': 'adapter', 'teacher_id': None,
     'module_filename': 'phase2_method_05.py'},
    {'slot': 6, 'base_id': '03', 'method': 'knowledge_distillation', 'teacher_id': '01',
     'module_filename': 'phase2_method_06.py'},
]

TRAINING_CONFIGS = {
    i: {
        'max_steps': (1000 if i in (3, 4, 6) else 800),
        'batch_size': (4 if i == 6 else 8), 'gradient_accumulation_steps': 1,
        'learning_rate': (2e-5 if i == 1 else 5e-5 if i == 2 else 2e-4),
        'weight_decay': 0.01, 'validation_interval': 400,
        'validation_document_cap': 64, 'max_tokens': 256, 'window_stride': 192,
        'gradient_clip_norm': 1.0, 'checkpoint_interval': 100,
        'recovery_interval': 400, 'head_dropout': .1,
        'outside_class_weight': .3, 'confidence_threshold': .45,
        'max_char_shift': 16, 'boundary_loss_weight': .2,
        'inference_window_batch_size': 4, 'unfrozen_layers': 2,
        'rank': 8, 'adapter_alpha': 16.0, 'adapter_bottleneck': 48,
        'teacher_max_tokens': 384, 'distill_temperature': 2.0,
        'distill_weight': .35,
        'optimizer': 'AdamW', 'seed': SEED + i,
    }
    for i in range(1, 7)
}
VALIDATION_FRACTION = 0.10
METHOD_SOURCE_OVERRIDES: dict[int, str] = {}
FILE_OVERRIDES: dict[str, str] = {}
RESUME_EXPECTED_SLOTS: set[int] | None = None  # optional stricter resume audit
AUTO_DOWNLOAD_PHASE2 = True
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
HF_DATASET_FILES = {
    'fold1-text.json': 'Phase_2_LNER/Train/fold1-text.json',
    'fold1-gold.json': 'Phase_2_LNER/Train/fold1-gold.json',
    'fold2-text.json': 'Phase_2_LNER/Train/fold2-text.json',
    'fold2-gold.json': 'Phase_2_LNER/Train/fold2-gold.json',
    'labels.txt': 'Phase_2_LNER/Train/labels.txt',
    'fold3-text.json': 'Phase_2_LNER/Test/fold3-text.json',
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     suffix='.json', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(stable_json(dict(value)) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def discover_exact(filename: str, *, required: bool = True) -> Path | None:
    if filename in FILE_OVERRIDES:
        selected = Path(FILE_OVERRIDES[filename]).expanduser().resolve()
        if not selected.is_file() or selected.name != filename:
            raise FileNotFoundError(f'Invalid exact override for {filename}: {selected}')
        return selected
    matches = sorted(set(path.resolve() for path in INPUT_ROOT.rglob(filename) if path.is_file()))
    if len(matches) == 1:
        return matches[0]
    if required or len(matches) > 1:
        raise RuntimeError(f'Expected one {filename}; found {len(matches)}: {matches}')
    return None


def stage_official_phase2_files() -> None:
    """Fill missing files and verify attached ones against one HF revision."""
    needed = ('labels.txt', 'fold3-text.json') if RUN_MODE == 'inference_only' else DATA_NAMES
    missing = [name for name in needed if discover_exact(name, required=False) is None]
    if not missing or not AUTO_DOWNLOAD_PHASE2:
        return
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        raise RuntimeError(
            'Missing Phase-2 files. Add an enabled Kaggle secret named HF_TOKEN, '
            'accept access to the gated Hugging Face course dataset, and enable Internet; '
            'or attach the six official files as Kaggle Inputs.'
        ) from None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError('Kaggle secret HF_TOKEN is empty; add a Hugging Face read token')
    token = token.strip()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                                   'huggingface_hub>=0.24,<2'])
            from huggingface_hub import HfApi, hf_hub_download
        except Exception:
            raise RuntimeError(
                'huggingface_hub is unavailable. Enable Kaggle Internet or attach the official files.'
            ) from None
    try:
        revision = HfApi().repo_info(HF_DATASET_REPO, repo_type='dataset', token=token).sha
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError('Hugging Face did not return a full dataset commit')
        manifest_path = WORK_ROOT / 'official_phase2_source.json'
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
            if existing.get('repo_id') != HF_DATASET_REPO or existing.get('revision') != revision:
                raise RuntimeError('Official dataset revision changed in this workspace; refusing mixed folds')
        destination = WORK_ROOT / 'official_phase2'
        destination.mkdir(parents=True, exist_ok=True)
        records = {}
        for name in needed:
            cached = Path(hf_hub_download(
                repo_id=HF_DATASET_REPO, repo_type='dataset',
                filename=HF_DATASET_FILES[name], revision=revision,
                token=token, cache_dir=str(WORK_ROOT / 'huggingface_cache')))
            if not cached.is_file() or cached.stat().st_size == 0:
                raise RuntimeError(f'Empty or missing Hugging Face file: {name}')
            digest = sha256_file(cached)
            if name not in missing:
                target = discover_exact(name)
                if sha256_file(target) != digest:
                    raise RuntimeError(f'Attached {name} differs from the pinned official dataset revision')
            else:
                target = destination / name
                if target.exists():
                    if sha256_file(target) != digest:
                        raise RuntimeError(f'Conflicting Phase-2 file in workspace: {name}')
                else:
                    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                        temporary = Path(handle.name)
                    try:
                        shutil.copyfile(cached, temporary)
                        if sha256_file(temporary) != digest:
                            raise RuntimeError(f'Phase-2 download copy failed integrity check: {name}')
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
                FILE_OVERRIDES[name] = str(target)
            records[name] = {'source_path': HF_DATASET_FILES[name],
                             'bytes': target.stat().st_size, 'sha256': digest}
        prior_files = existing.get('files', {}) if manifest_path.exists() else {}
        atomic_json(manifest_path, {'repo_id': HF_DATASET_REPO, 'revision': revision,
                                    'files': {**prior_files, **records}})
        print(f'Fetched {len(missing)} official Phase-2 file(s) from Hugging Face revision {revision[:12]}.')
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError(
            'Phase-2 Hugging Face download failed. Check HF_TOKEN access to the gated dataset '
            'and Kaggle Internet, or attach the six official files as Kaggle Inputs.'
        ) from None
    finally:
        del token


if RUN_PHASE2:
    stage_official_phase2_files()


FOUND = {name: discover_exact(name, required=False) for name in (*DATA_NAMES, *BASE_ARCHIVES.values())}
print('Detected official files and Phase-1 archives:')
for name, path in FOUND.items():
    print(f'  {name}: {path if path else "not attached"}')
print('CUDA devices:', [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])


# %% Phase-1 archive verification and selective extraction
def _safe_archive_member(name: str) -> bool:
    return bool(name) and not name.startswith('/') and '\\' not in name and '..' not in Path(name).parts


def _integrity_rows(archive: zipfile.ZipFile) -> dict[str, dict[str, Any]] | None:
    name = 'manifests/recovery_integrity.json'
    if name not in archive.namelist():
        return None
    manifest = json.loads(archive.read(name))
    rows = {row['path']: row for row in manifest['files']}
    if len(rows) != len(manifest['files']) or set(rows) != set(archive.namelist()) - {name}:
        raise ValueError('Archive integrity manifest does not match ZIP members')
    return rows


def extract_member_verified(archive: zipfile.ZipFile, member: str, destination: Path,
                            integrity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not _safe_archive_member(member) or member not in archive.namelist():
        raise ValueError(f'Missing/unsafe required archive member: {member}')
    info = archive.getinfo(member)
    if info.is_dir() or info.file_size > 1_500_000_000:
        raise ValueError(f'Unexpected member type or size: {member}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with archive.open(info) as source:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    digest.update(chunk)
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    actual = digest.hexdigest()
    try:
        if integrity is not None:
            row = integrity.get(member)
            if row is None or row.get('sha256') != actual or int(row.get('bytes', -1)) != info.file_size:
                raise ValueError(f'Integrity manifest mismatch: {member}')
        if destination.exists():
            if sha256_file(destination) != actual:
                raise RuntimeError(f'Conflicting staged base artifact: {destination}')
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'member': member, 'sha256': actual, 'bytes': info.file_size,
            'manifest_verified': integrity is not None}


def stage_phase1_base(base_id: str) -> Path:
    if base_id in STAGED_BASES:
        return STAGED_BASES[base_id]
    archive_name = BASE_ARCHIVES[base_id]
    archive_path = discover_exact(archive_name)
    destination = WORK_ROOT / 'phase1_bases' / base_id
    source = BASE_SOURCES[base_id]
    members = [f'model/{source}', 'model/course_bpe.py', 'checkpoints/selected_model.pt']
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not _safe_archive_member(name) for name in names):
            raise ValueError(f'Unsafe or duplicate member names in {archive_path}')
        members += [name for name in names if name.startswith('tokenizer/') and not name.endswith('/')]
        if not any(name == 'tokenizer/tokenizer.json' for name in members):
            raise ValueError(f'No tokenizer.json in {archive_path}')
        integrity = _integrity_rows(archive)
        records = [extract_member_verified(archive, name, destination / name, integrity)
                   for name in members]
    evidence = {'base_id': base_id, 'archive_name': archive_name,
                'archive_sha256': sha256_file(archive_path), 'selected_members': records,
                'assurance': 'manifest+SHA256+ZIP-CRC' if integrity else 'local-SHA256+ZIP-CRC; archive has no manifest'}
    manifest_path = destination / 'base_evidence.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != evidence:
        raise RuntimeError(f'Base archive changed since this workspace was initialized: {archive_path}')
    atomic_json(manifest_path, evidence)
    STAGED_BASES[base_id] = destination
    return destination


def import_local_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_phase1_base(base_id: str, device: str = 'cpu'):
    directory = stage_phase1_base(base_id)
    source_path = directory / 'model' / BASE_SOURCES[base_id]
    module = import_local_source(source_path, f'cs781_phase1_{base_id}_{sha256_file(source_path)[:12]}')
    tokenizer_module = import_local_source(
        directory / 'model' / 'course_bpe.py', f'cs781_bpe_{base_id}'
    )
    tokenizer = tokenizer_module.CourseBPETokenizer.from_pretrained(directory / 'tokenizer')
    checkpoint = torch.load(directory / 'checkpoints' / 'selected_model.pt',
                            map_location='cpu', weights_only=True)
    recorded_source_sha = checkpoint.get('model_source_sha256')
    actual_source_sha = sha256_file(source_path)
    if recorded_source_sha is not None and recorded_source_sha != actual_source_sha:
        raise RuntimeError(f'Phase-1 {base_id} checkpoint/model source SHA mismatch')
    if recorded_source_sha is None and base_id != '01':
        raise RuntimeError(f'Phase-1 {base_id} checkpoint has no model-source binding')
    tokenizer_meta = json.loads((directory / 'tokenizer' / 'tokenizer_metadata.json').read_text(encoding='utf-8'))
    if checkpoint.get('tokenizer_fingerprint') != tokenizer_meta.get('fingerprint'):
        raise RuntimeError(f'Phase-1 {base_id} tokenizer fingerprint mismatch')
    if int(checkpoint['model_config']['vocab_size']) != len(tokenizer):
        raise RuntimeError(f'Phase-1 {base_id} vocabulary mismatch')
    model = module.build_model(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model.to(device), tokenizer, {
        'base_id': base_id, 'checkpoint_sha256': sha256_file(directory / 'checkpoints' / 'selected_model.pt'),
        'source_sha256': actual_source_sha,
        'source_binding': 'checkpoint' if recorded_source_sha is not None else 'same_archive_only',
        'tokenizer_fingerprint': tokenizer_meta['fingerprint'],
        'parameter_count': parameter_count, 'checkpoint_step': checkpoint.get('step'),
        'model_config': checkpoint['model_config'],
    }


# %% Official Phase-2 dataset inspection and character-offset validation
def load_json(path: Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_text_map(path: Path) -> dict[str, str]:
    raw = load_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f'Expected nonempty ID-to-text object: {path}')
    if not all(isinstance(value, str) and value for value in raw.values()):
        raise ValueError(f'Text values are not nonempty strings in {path}')
    return {str(key): value for key, value in raw.items()}


def load_label_vocabulary(path: Path) -> list[str]:
    labels = [line.strip() for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
              if line.strip()]
    if len(labels) != len(set(labels)) or set(labels) != OFFICIAL_LABELS:
        raise ValueError(f'Official Phase-2 labels differ from document: {labels}')
    return labels


def normalize_span(raw: Any, *, document_id: str, text: str, allowed: set[str]) -> tuple[int, int, str]:
    if isinstance(raw, (tuple, list)) and len(raw) in (3, 4):
        start, end, label = raw[:3]
        supplied_text = raw[3] if len(raw) == 4 else None
    elif isinstance(raw, dict):
        start = raw.get('start', raw.get('start_offset'))
        end = raw.get('end', raw.get('end_offset'))
        label = raw.get('label', raw.get('entity_type'))
        supplied_text = raw.get('entity_text', raw.get('text'))
    else:
        raise TypeError(f'{document_id}: unsupported gold span {raw!r}')
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f'{document_id}: character offsets must be integers: {raw!r}')
    if not 0 <= start < end <= len(text) or label not in allowed:
        raise ValueError(f'{document_id}: invalid offset/label: {raw!r}')
    if supplied_text is not None and text[start:end] != supplied_text:
        raise ValueError(f'{document_id}: gold entity text differs from text[start:end]; inspect offset convention')
    return start, end, str(label)


def load_gold_map(path: Path, texts: Mapping[str, str], labels: Sequence[str]) -> dict[str, list[tuple[int, int, str]]]:
    raw = load_json(path)
    if not isinstance(raw, dict) or set(map(str, raw)) != set(texts):
        raise ValueError(f'Gold IDs must match text IDs exactly: {path}')
    result = {}
    for identifier, raw_spans in raw.items():
        if not isinstance(raw_spans, list):
            raise TypeError(f'{identifier}: expected a list of gold entities')
        spans = [normalize_span(item, document_id=str(identifier), text=texts[str(identifier)],
                                allowed=set(labels)) for item in raw_spans]
        if len(spans) != len(set(spans)):
            raise ValueError(f'{identifier}: duplicate gold spans')
        result[str(identifier)] = sorted(spans)
    return result


def load_official_phase2() -> dict[str, Any]:
    paths = {name: discover_exact(name) for name in DATA_NAMES}
    labels = load_label_vocabulary(paths['labels.txt'])
    train_text, train_gold = {}, {}
    for fold in (1, 2):
        texts = load_text_map(paths[f'fold{fold}-text.json'])
        gold = load_gold_map(paths[f'fold{fold}-gold.json'], texts, labels)
        overlap = set(train_text) & set(texts)
        if overlap:
            raise ValueError(f'Training folds share IDs: {sorted(overlap)[:8]}')
        train_text.update(texts)
        train_gold.update(gold)
    test_text = load_text_map(paths['fold3-text.json'])
    if set(test_text) & set(train_text):
        raise ValueError('Official Phase-2 test IDs overlap public training IDs')
    fingerprint = hashlib.sha256(stable_json({name: sha256_file(path) for name, path in paths.items()}).encode()).hexdigest()
    return {'train_text': train_text, 'train_gold': train_gold, 'test_text': test_text,
            'labels': labels, 'paths': paths, 'fingerprint': fingerprint}


def load_phase2_test_only() -> dict[str, Any]:
    label_path = discover_exact('labels.txt')
    test_path = discover_exact('fold3-text.json')
    labels = load_label_vocabulary(label_path)
    test_text = load_text_map(test_path)
    return {'train_text': None, 'train_gold': None, 'test_text': test_text,
            'labels': labels, 'paths': {'labels.txt': label_path, 'fold3-text.json': test_path},
            'fingerprint': None,
            'test_fingerprint': sha256_file(test_path)}


def deterministic_split(ids: Sequence[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0 < fraction < 0.5:
        raise ValueError('Student-selected validation fraction must be in (0, 0.5)')
    ordered = sorted(ids, key=lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest())
    count = max(1, round(len(ordered) * fraction))
    if count >= len(ordered):
        raise ValueError('Insufficient documents for held-out validation')
    return ordered[count:], ordered[:count]


def strict_macro_prf(predictions: Mapping[str, Sequence[Sequence[Any]]],
                     gold: Mapping[str, Sequence[Sequence[Any]]], labels: Sequence[str]) -> dict[str, Any]:
    if set(predictions) != set(gold):
        raise ValueError('Validation ID sets differ')
    per_label = {}
    for label in labels:
        predicted = {(identifier, int(item[0]), int(item[1]))
                     for identifier, spans in predictions.items() for item in spans
                     if item[2] == label}
        expected = {(identifier, int(item[0]), int(item[1]))
                    for identifier, spans in gold.items() for item in spans
                    if item[2] == label}
        true_positive = len(predicted & expected)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {'precision': precision, 'recall': recall, 'f1': f1,
                            'tp': true_positive, 'predicted': len(predicted), 'gold': len(expected)}
    return {'strict_macro_f1_local': sum(item['f1'] for item in per_label.values()) / len(labels),
            'per_label': per_label, 'note': 'Local strict-span approximation; compare with official evaluator'}


def validate_prediction_map(predictions: Mapping[str, Any], texts: Mapping[str, str],
                            labels: Sequence[str]) -> dict[str, Any]:
    if not isinstance(predictions, dict):
        raise TypeError('Predictions must be an ID-keyed object')
    if set(predictions) != set(texts):
        raise ValueError(f'Prediction IDs differ: missing={sorted(set(texts)-set(predictions))[:10]} '
                         f'extra={sorted(set(predictions)-set(texts))[:10]}')
    allowed = set(labels)
    for identifier, spans in predictions.items():
        if not isinstance(spans, list):
            raise TypeError(f'{identifier}: predictions must be a list')
        if any(not isinstance(item, (list, tuple)) for item in spans):
            raise TypeError(f'{identifier}: each predicted span must be an array')
        canonical = [normalize_span(item, document_id=identifier, text=texts[identifier],
                                    allowed=allowed) for item in spans]
        if len(canonical) != len(set(canonical)):
            raise ValueError(f'{identifier}: duplicate predicted spans')
    return {'passed': True, 'document_count': len(texts),
            'span_count': sum(len(spans) for spans in predictions.values())}


# %% Student-owned six-method integration contract and GPU preflight
def validate_experiment_plan() -> None:
    if len(EXPERIMENTS) != 6 or sorted(spec['slot'] for spec in EXPERIMENTS) != list(range(1, 7)):
        raise ValueError('Exactly six numbered experiment slots are required')
    if {spec['base_id'] for spec in EXPERIMENTS} != set(BASE_ARCHIVES):
        raise ValueError('The six submissions must involve all four Phase-1 architectures')
    methods = [spec['method'].lower() for spec in EXPERIMENTS]
    nondistill = [method for method in methods if method != 'knowledge_distillation']
    if len(set(nondistill)) < 5 or not any('lorma' in method for method in nondistill):
        raise ValueError('At least five distinct adaptation methods, including LORMA, are required')
    if methods.count('knowledge_distillation') < 1:
        raise ValueError('At least one knowledge-distillation experiment is required')
    for spec in EXPERIMENTS:
        if spec['base_id'] not in BASE_ARCHIVES or (spec['teacher_id'] is not None and spec['teacher_id'] not in BASE_ARCHIVES):
            raise ValueError(f'Unknown Phase-1 base/teacher: {spec}')
        if spec['method'] == 'knowledge_distillation' and spec['teacher_id'] in (None, spec['base_id']):
            raise ValueError('Distillation needs a distinct Phase-1 teacher')
        if Path(spec['module_filename']).name != spec['module_filename']:
            raise ValueError('Student method filename must be a basename')


def validate_training_config(slot: int) -> dict[str, Any]:
    config = dict(TRAINING_CONFIGS[slot])
    needed = ('max_steps', 'batch_size', 'gradient_accumulation_steps',
              'learning_rate', 'weight_decay', 'validation_interval',
              'max_tokens', 'window_stride', 'gradient_clip_norm')
    missing = [name for name in needed if config.get(name) is None]
    if missing:
        raise RuntimeError(f'Slot {slot}: student training configuration incomplete: {missing}')
    positive = ('max_steps', 'batch_size', 'gradient_accumulation_steps',
                'learning_rate', 'validation_interval', 'max_tokens', 'window_stride',
                'gradient_clip_norm',
                'checkpoint_interval', 'recovery_interval')
    if any(float(config[name]) <= 0 for name in positive) or float(config['weight_decay']) < 0:
        raise ValueError(f'Slot {slot}: invalid nonpositive scientific configuration')
    if config['optimizer'] != 'AdamW':
        raise ValueError('The supplied recovery runner supports AdamW only')
    return config


def student_module_path(spec: Mapping[str, Any]) -> Path:
    slot = int(spec['slot'])
    if slot in METHOD_SOURCE_OVERRIDES:
        path = Path(METHOD_SOURCE_OVERRIDES[slot]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    matches = sorted(set(path.resolve() for path in INPUT_ROOT.rglob(spec['module_filename']) if path.is_file()))
    recovered = experiment_directory(slot) / spec['module_filename']
    if not matches and recovered.is_file():
        return recovered
    if len(matches) != 1:
        raise RuntimeError(f'Slot {slot}: attach exactly one student-authored {spec["module_filename"]}; found {matches}')
    return matches[0]


def common_module_path() -> Path:
    filename = 'phase2_method_common.py'
    matches = sorted(set(path.resolve() for path in INPUT_ROOT.rglob(filename) if path.is_file()))
    recovered = WORK_ROOT / 'method_source' / filename
    if not matches and recovered.is_file():
        return recovered
    if len(matches) != 1:
        raise RuntimeError(f'Attach exactly one {filename}; found {matches}')
    return matches[0]


def load_student_method(spec: Mapping[str, Any], destination: Path):
    source = student_module_path(spec)
    data = source.read_bytes()
    ast.parse(data.decode('utf-8'), filename=str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shared_source = common_module_path()
    shared_data = shared_source.read_bytes()
    ast.parse(shared_data.decode('utf-8'), filename=str(shared_source))
    shared_destination = WORK_ROOT / 'method_source' / shared_source.name
    shared_destination.parent.mkdir(parents=True, exist_ok=True)
    if shared_destination.exists() and shared_destination.read_bytes() != shared_data:
        raise RuntimeError('Shared method source changed within this run workspace')
    if not shared_destination.exists():
        shared_destination.write_bytes(shared_data)
    local_common = destination.parent / shared_source.name
    if local_common.exists() and local_common.read_bytes() != shared_data:
        raise RuntimeError('Shared method source changed within experiment workspace')
    if not local_common.exists():
        local_common.write_bytes(shared_data)
    if destination.exists() and destination.read_bytes() != data:
        raise RuntimeError(f'Slot {spec["slot"]}: student method changed within an experiment workspace')
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    if str(destination.parent) not in sys.path:
        sys.path.insert(0, str(destination.parent))
    # A common module with the same name must not retain a prior workspace version.
    sys.modules.pop('phase2_method_common', None)
    module = import_local_source(destination, f'phase2_method_{spec["slot"]}_{sha256_file(destination)[:12]}')
    for name in ('build_task_model', 'collate_batch', 'predict_spans'):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f'Slot {spec["slot"]}: student module lacks {name}')
    combined = hashlib.sha256(data + b'\0' + shared_data).hexdigest()
    return module, combined


def choose_gpu_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError('Phase-2 training requires a Kaggle GPU; enable the accelerator')
    ids = list(range(torch.cuda.device_count()))
    bf16 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in ids) and torch.cuda.is_bf16_supported()
    return {'ids': ids, 'primary': 'cuda:0', 'precision': 'bf16' if bf16 else 'fp16',
            'amp_dtype': torch.bfloat16 if bf16 else torch.float16,
            'scaler_enabled': not bf16,
            'names': [torch.cuda.get_device_name(i) for i in ids]}


def move_batch(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_batch(item, device) for item in value)
    return value


def task_loss(output: Any) -> torch.Tensor:
    loss = output['loss'] if isinstance(output, Mapping) else output
    if not torch.is_tensor(loss) or not torch.isfinite(loss).all():
        raise FloatingPointError('Student task must return a finite torch loss tensor')
    return loss.mean()


def make_optimizer(model: torch.nn.Module, config: Mapping[str, Any]):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError('Student task has no trainable parameters')
    kwargs = {'lr': float(config['learning_rate']), 'weight_decay': float(config['weight_decay'])}
    try:
        return torch.optim.AdamW(params, fused=True, **kwargs), 'fused'
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, **kwargs), 'standard'


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# %% Stateful batches, atomic checkpoints, recovery archives
class DocumentRows(Dataset):
    def __init__(self, texts: Mapping[str, str], gold: Mapping[str, Any], ids: Sequence[str]):
        self.rows = [{'id': identifier, 'text': texts[identifier], 'spans': gold[identifier]}
                     for identifier in ids]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class ResumableBatchSampler(Sampler[list[int]]):
    def __init__(self, length: int, batch_size: int, seed: int):
        if length <= 0 or batch_size <= 0:
            raise ValueError('Sampler needs documents and a positive batch size')
        self.length, self.batch_size, self.seed = int(length), int(batch_size), int(seed)
        self.epoch, self.cursor = 0, 0

    def _order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.length, generator=generator).tolist()

    def __iter__(self):
        order = self._order()
        while self.cursor < self.length:
            end = min(self.length, self.cursor + self.batch_size)
            indices = order[self.cursor:end]
            self.cursor = end
            yield indices

    def __len__(self):
        return math.ceil((self.length - self.cursor) / self.batch_size)

    def next_epoch(self):
        self.epoch += 1
        self.cursor = 0

    def state_dict(self):
        return {'length': self.length, 'batch_size': self.batch_size, 'seed': self.seed,
                'epoch': self.epoch, 'cursor': self.cursor}

    def load_state_dict(self, state):
        expected = {'length': self.length, 'batch_size': self.batch_size, 'seed': self.seed}
        if {key: state[key] for key in expected} != expected:
            raise RuntimeError('Sampler data, batch size, or seed changed')
        self.epoch, self.cursor = int(state['epoch']), int(state['cursor'])
        if not 0 <= self.cursor <= self.length:
            raise ValueError('Invalid sampler cursor')


def capture_rng():
    np_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (np_state[0], np_state[1].tolist(),
            np_state[2], np_state[3], np_state[4]),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name, keys, position, gaussian_flag, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian_flag, cached))
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state['torch_cuda']:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise RuntimeError('CUDA device count differs from saved RNG state')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def bare_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def atomic_torch_save(path: Path, state: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(state), temporary)
        check = torch.load(temporary, map_location='cpu', weights_only=True)
        if check.get('format') != state.get('format') or check.get('step') != state.get('step'):
            raise RuntimeError(f'Checkpoint readback failed: {path}')
        if path.exists():
            previous = path.with_name(path.stem + '.previous_known_good.pt')
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as handle:
                prior_copy = Path(handle.name)
            prior_copy.unlink()
            try:
                try:
                    os.link(path, prior_copy)
                except OSError:
                    shutil.copyfile(path, prior_copy)
                os.replace(prior_copy, previous)
            finally:
                prior_copy.unlink(missing_ok=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def experiment_directory(slot: int) -> Path:
    return WORK_ROOT / 'experiments' / f'submission_{slot:02d}'


def experiment_identity(spec, config, base_meta, dataset_fingerprint, method_sha):
    content = {'spec': dict(spec), 'config': dict(config), 'base': base_meta,
               'dataset_fingerprint': dataset_fingerprint, 'method_sha256': method_sha}
    return hashlib.sha256(stable_json(content).encode()).hexdigest()


def checkpoint_payload(model, optimizer, scheduler, scaler, sampler, *, step, best_score,
                       identity, history_path, config, base_meta, method_sha, dataset_fingerprint):
    return {
        'format': 'cs781_phase2_training_v1', 'step': int(step),
        'best_validation_score': best_score, 'identity': identity,
        'model_state': bare_model(model).state_dict(),
        'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'sampler_state': sampler.state_dict(), 'rng_state': capture_rng(),
        'training_config': dict(config), 'base_meta': base_meta,
        'method_sha256': method_sha, 'dataset_fingerprint': dataset_fingerprint,
        'history_committed_step': int(step),
        'history_sha256': sha256_file(history_path) if history_path.exists() else None,
    }


def restore_training_checkpoint(path, model, optimizer, scheduler, scaler, sampler,
                                identity, config, method_sha, dataset_fingerprint):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'cs781_phase2_training_v1' or state.get('identity') != identity:
        raise RuntimeError('Checkpoint experiment identity mismatch')
    if state.get('training_config') != dict(config) or state.get('method_sha256') != method_sha:
        raise RuntimeError('Checkpoint training/method configuration changed')
    if state.get('dataset_fingerprint') != dataset_fingerprint:
        raise RuntimeError('Checkpoint dataset changed')
    bare_model(model).load_state_dict(state['model_state'], strict=True)
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    if scaler is not None:
        if state.get('scaler_state') is None:
            raise RuntimeError('FP16 scaler state missing from checkpoint')
        scaler.load_state_dict(state['scaler_state'])
    elif state.get('scaler_state') is not None:
        raise RuntimeError('Checkpoint precision policy changed')
    sampler.load_state_dict(state['sampler_state'])
    return state


def reconcile_history(path: Path, committed_step: int) -> None:
    if not path.exists():
        if committed_step:
            raise RuntimeError('Checkpoint exists but training history is missing')
        return
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    committed = [row for row in rows if int(row['step']) <= committed_step]
    if committed and int(committed[-1]['step']) != committed_step:
        raise RuntimeError('History does not contain the checkpoint step')
    excess = [row for row in rows if int(row['step']) > committed_step]
    if excess:
        archive = path.with_name(f'{path.stem}.uncommitted.after_{committed_step}.{int(time.time())}.jsonl')
        shutil.copy2(path, archive)
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         suffix='.jsonl', delete=False) as handle:
            temporary = Path(handle.name)
            for row in committed:
                handle.write(stable_json(row) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def recovery_name(slot: int) -> str:
    return f'phase2_submission{slot:02d}_recovery.zip'


def export_recovery(slot: int) -> Path:
    directory = experiment_directory(slot)
    output = directory / recovery_name(slot)
    items = sorted(path for path in directory.rglob('*') if path.is_file()
                   and path != output and path.name != 'recovery_receipt.json'
                   and not path.name.endswith('.previous_known_good.pt')
                   and not path.name.endswith('.tmp'))
    manifest = {'slot': slot, 'files': [
        {'path': path.relative_to(directory).as_posix(), 'bytes': path.stat().st_size,
         'sha256': sha256_file(path)} for path in items
    ]}
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.zip', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, 'w', allowZip64=True) as archive:
            for row in manifest['files']:
                name = row['path']
                compression = zipfile.ZIP_STORED if name.endswith('.pt') else zipfile.ZIP_DEFLATED
                archive.write(directory / name, name, compress_type=compression)
            archive.writestr('recovery_integrity.json', stable_json(manifest))
        verify_recovery_zip(temporary, slot)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(directory / 'recovery_receipt.json',
                {'path': str(output), 'sha256': sha256_file(output), 'bytes': output.stat().st_size})
    return output


def verify_recovery_zip(path: Path, slot: int):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or 'recovery_integrity.json' not in names:
            raise ValueError('Recovery ZIP lacks an unambiguous manifest')
        manifest = json.loads(archive.read('recovery_integrity.json'))
        rows = {row['path']: row for row in manifest['files']}
        if manifest['slot'] != slot or len(rows) != len(manifest['files']) or set(rows) != set(names) - {'recovery_integrity.json'}:
            raise ValueError('Recovery ZIP manifest/member mismatch')
        for name, row in rows.items():
            if not _safe_archive_member(name) or archive.getinfo(name).is_dir():
                raise ValueError(f'Unsafe recovery member: {name}')
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(chunk)
            if digest.hexdigest() != row['sha256'] or archive.getinfo(name).file_size != row['bytes']:
                raise ValueError(f'Recovery member integrity failed: {name}')
    return manifest


def restore_attached_recovery(slot: int) -> Path | None:
    if slot in RESTORED_RECOVERY_SLOTS:
        return RESTORED_RECOVERY_SLOTS[slot]
    name = recovery_name(slot)
    attached = discover_exact(name, required=False)
    if attached is None:
        return None
    manifest = verify_recovery_zip(attached, slot)
    directory = experiment_directory(slot).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(attached) as archive:
        for row in manifest['files']:
            relative = row['path']
            target = (directory / relative).resolve()
            if directory not in target.parents:
                raise ValueError(f'Recovery member escapes experiment directory: {relative}')
            extract_member_verified(archive, relative, target,
                                    {relative: row})
    RESTORED_RECOVERY_SLOTS[slot] = attached
    return attached


# %% Student-task smoke test, validation, and resumable training
def build_task(method, base_id, spec, config, labels, device):
    base_model, tokenizer, base_meta = load_phase1_base(base_id, 'cpu')
    teacher = None
    teacher_tokenizer = None
    teacher_meta = None
    if spec['teacher_id'] is not None:
        teacher, teacher_tokenizer, teacher_meta = load_phase1_base(spec['teacher_id'], 'cpu')
        if teacher_meta['parameter_count'] <= base_meta['parameter_count']:
            raise ValueError('The declared Phase-1 distillation teacher must be larger than the student base')
        teacher_spec = EXPERIMENTS[0]
        if teacher_spec['base_id'] != spec['teacher_id'] or teacher_spec['method'] != 'full_finetune':
            raise RuntimeError('Distillation requires the completed slot-1 NER teacher')
        teacher_directory = experiment_directory(1)
        teacher_method, teacher_method_sha = load_student_method(
            teacher_spec, teacher_directory / teacher_spec['module_filename'])
        teacher_config = validate_training_config(1)
        teacher_selected_path = teacher_directory / 'selected_model.pt'
        teacher_result_path = teacher_directory / 'training_result.json'
        if not teacher_selected_path.is_file() or not teacher_result_path.is_file():
            raise RuntimeError('Complete slot 1 before distillation; trained NER teacher missing')
        teacher_result = load_json(teacher_result_path)
        if teacher_result.get('status') != 'complete':
            raise RuntimeError('Slot-1 NER teacher training is incomplete')
        teacher_selected = torch.load(teacher_selected_path, map_location='cpu', weights_only=True)
        if (teacher_selected.get('method_sha256') != teacher_method_sha or
                teacher_selected.get('training_config') != teacher_config):
            raise RuntimeError('Slot-1 NER teacher checkpoint/source/config mismatch')
        teacher = teacher_method.build_task_model(teacher, teacher_tokenizer, labels, teacher_config)
        teacher.load_state_dict(teacher_selected['model_state'], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_meta = {**teacher_meta, 'ner_selected_sha256': sha256_file(teacher_selected_path),
                        'ner_method_sha256': teacher_method_sha,
                        'ner_validation_macro_f1': teacher_selected['strict_macro_f1_local']}
    task = method.build_task_model(base_model, tokenizer, labels, dict(config),
                                   teacher_model=teacher, teacher_tokenizer=teacher_tokenizer)
    if not isinstance(task, torch.nn.Module):
        raise TypeError('Student build_task_model must return a torch.nn.Module')
    task = task.to(device)
    if torch.cuda.device_count() > 1:
        if int(config['batch_size']) < torch.cuda.device_count() or int(config['batch_size']) % torch.cuda.device_count():
            raise ValueError('Batch size must be divisible by the active GPU count')
        task = torch.nn.DataParallel(task, device_ids=list(range(torch.cuda.device_count())))
    return task, tokenizer, teacher_tokenizer, base_meta, teacher_meta


@torch.inference_mode()
def evaluate_student_method(method, task, tokenizer, data, val_ids, config, device):
    task.eval()
    predictions = {}
    for identifier in val_ids:
        spans = method.predict_spans(task, data['train_text'][identifier],
                                     tokenizer, data['labels'], dict(config))
        if not isinstance(spans, list):
            raise TypeError(f'Validation prediction for {identifier} is not a list')
        predictions[identifier] = spans
    validate_prediction_map(predictions, {identifier: data['train_text'][identifier] for identifier in val_ids},
                            data['labels'])
    gold = {identifier: data['train_gold'][identifier] for identifier in val_ids}
    report = strict_macro_prf(predictions, gold, data['labels'])
    task.train()
    return report


def make_scheduler(method, optimizer, config):
    custom = getattr(method, 'build_scheduler', None)
    scheduler = custom(optimizer, dict(config)) if callable(custom) else torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _step: 1.0
    )
    if not hasattr(scheduler, 'state_dict') or not hasattr(scheduler, 'load_state_dict'):
        raise TypeError('Student scheduler must support state_dict and load_state_dict')
    return scheduler


def create_loader(method, tokenizer, teacher_tokenizer, labels, config, texts, gold, ids, sampler):
    dataset = DocumentRows(texts, gold, ids)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                      pin_memory=True,
                      collate_fn=lambda rows: method.collate_batch(
                          rows, tokenizer, labels, dict(config), teacher_tokenizer=teacher_tokenizer))


def run_gpu_smoke(spec, method, data, train_ids, val_ids, config, directory, method_sha, identity):
    manifest = directory / 'gpu_smoke.json'
    if manifest.exists():
        report = load_json(manifest)
        if report.get('identity') != identity or report.get('status') != 'PASS':
            raise RuntimeError('Stale or failed GPU smoke record')
        return report
    gpu = choose_gpu_runtime()
    task, tokenizer, teacher_tokenizer, base_meta, _teacher_meta = build_task(
        method, spec['base_id'], spec, config, data['labels'], gpu['primary']
    )
    sampler = ResumableBatchSampler(len(train_ids), int(config['batch_size']), int(config['seed']))
    loader = create_loader(method, tokenizer, teacher_tokenizer, data['labels'], config,
                           data['train_text'], data['train_gold'], train_ids, sampler)
    started = time.monotonic()
    batch = move_batch(next(iter(loader)), gpu['primary'])
    batch_prep_seconds = time.monotonic() - started
    optimizer, backend = make_optimizer(task, config)
    scheduler = make_scheduler(method, optimizer, config)
    scaler = make_scaler(gpu['scaler_enabled'])
    for gpu_id in gpu['ids']:
        torch.cuda.reset_peak_memory_stats(gpu_id)
        torch.cuda.synchronize(gpu_id)
    started = time.monotonic()
    task.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=gpu['amp_dtype']):
        loss = task_loss(task(**batch))
    (scaler.scale(loss) if gpu['scaler_enabled'] else loss).backward()
    if gpu['scaler_enabled']:
        scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(task.parameters(), float(config['gradient_clip_norm']))
    if not math.isfinite(float(gradient_norm)):
        raise FloatingPointError('GPU smoke produced nonfinite gradient norm')
    if gpu['scaler_enabled']:
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            raise FloatingPointError('GPU smoke FP16 optimizer step overflowed')
    else:
        optimizer.step()
    scheduler.step()
    for gpu_id in gpu['ids']:
        torch.cuda.synchronize(gpu_id)
    compute_seconds = time.monotonic() - started
    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name) / 'gpu_smoke.pt'
        payload = checkpoint_payload(task, optimizer, scheduler, scaler if gpu['scaler_enabled'] else None,
                                     sampler, step=1, best_score=None, identity=identity,
                                     history_path=directory / 'not_yet_created_history.jsonl',
                                     config=config, base_meta=base_meta, method_sha=method_sha,
                                     dataset_fingerprint=data['fingerprint'])
        atomic_torch_save(temporary, payload)
        restore_training_checkpoint(temporary, task, optimizer, scheduler,
                                    scaler if gpu['scaler_enabled'] else None, sampler,
                                    identity, config, method_sha, data['fingerprint'])
        if not optimizer.state:
            raise AssertionError('GPU smoke did not restore optimizer state')
    probe_id = val_ids[0]
    probe_text = data['train_text'][probe_id]
    encoding = tokenizer.backend.encode(probe_text, add_special_tokens=False)
    if len(encoding.ids) != len(encoding.offsets) or any(
        not 0 <= start <= end <= len(probe_text) for start, end in encoding.offsets
    ):
        raise AssertionError('BPE offset map is invalid for a held-out legal document')
    task.eval()
    with torch.inference_mode():
        probe_spans = method.predict_spans(task, probe_text,
                                           tokenizer, data['labels'], dict(config))
    validate_prediction_map({probe_id: probe_spans}, {probe_id: probe_text}, data['labels'])
    report = {'status': 'PASS', 'identity': identity, 'loss': float(loss.detach().cpu()),
              'gradient_norm': float(gradient_norm), 'batch_prep_seconds': batch_prep_seconds,
              'single_microbatch_compute_seconds': compute_seconds,
              'precision': gpu['precision'], 'gpu_names': gpu['names'],
              'peak_memory_gib': {str(i): torch.cuda.max_memory_allocated(i) / 2**30 for i in gpu['ids']},
              'optimizer_backend': backend, 'checkpoint_restore': 'PASS',
              'heldout_prediction_contract': 'PASS',
              'timing_note': 'One microbatch, not a full accumulated optimizer step'}
    atomic_json(manifest, report)
    del task, optimizer, scheduler, scaler, batch
    torch.cuda.empty_cache()
    return report


def save_selected_model(directory, task, *, step, score, identity, config, method_sha,
                        base_meta, tokenizer_fingerprint, dataset_fingerprint):
    path = directory / 'selected_model.pt'
    state = {'format': 'cs781_phase2_selected_v1', 'step': int(step),
             'strict_macro_f1_local': float(score), 'identity': identity,
             'model_state': bare_model(task).state_dict(), 'training_config': dict(config),
             'method_sha256': method_sha, 'base_meta': base_meta,
             'tokenizer_fingerprint': tokenizer_fingerprint,
             'dataset_fingerprint': dataset_fingerprint}
    digest = atomic_torch_save(path, state)
    return {'path': str(path), 'sha256': digest, 'step': step, 'score': score}


def train_one_experiment(spec, data, train_ids, val_ids, session_deadline):
    slot = int(spec['slot'])
    directory = experiment_directory(slot)
    directory.mkdir(parents=True, exist_ok=True)
    attached_recovery = restore_attached_recovery(slot)
    config = validate_training_config(slot)
    method, method_sha = load_student_method(spec, directory / spec['module_filename'])
    gpu = choose_gpu_runtime()
    if gpu['precision'] == 'fp16' and not gpu['scaler_enabled']:
        raise RuntimeError('FP16 must have gradient scaling')
    task, tokenizer, teacher_tokenizer, base_meta, teacher_meta = build_task(
        method, spec['base_id'], spec, config, data['labels'], gpu['primary']
    )
    identity = experiment_identity(spec, config,
                                   {'student': base_meta, 'teacher': teacher_meta},
                                   data['fingerprint'], method_sha)
    identity_path = directory / 'experiment_identity.json'
    identity_record = {'identity': identity, 'spec': dict(spec), 'config': config,
                       'base_meta': base_meta, 'teacher_meta': teacher_meta,
                       'dataset_fingerprint': data['fingerprint'], 'method_sha256': method_sha}
    if identity_path.exists() and load_json(identity_path) != identity_record:
        raise RuntimeError(f'Slot {slot}: experiment identity changed after initialization')
    atomic_json(identity_path, identity_record)
    if attached_recovery is not None:
        print(f'Slot {slot}: restored verified recovery from {attached_recovery}')
    latest = directory / 'checkpoint.latest.pt'
    history = directory / 'training.jsonl'
    sampler = ResumableBatchSampler(len(train_ids), int(config['batch_size']), int(config['seed']))
    loader = create_loader(method, tokenizer, teacher_tokenizer, data['labels'], config,
                           data['train_text'], data['train_gold'], train_ids, sampler)
    optimizer, backend = make_optimizer(task, config)
    scheduler = make_scheduler(method, optimizer, config)
    scaler = make_scaler(gpu['scaler_enabled'])
    best_score, step = -math.inf, 0
    if latest.exists():
        if RUN_MODE == 'fresh':
            raise RuntimeError(f'Slot {slot}: fresh mode found a compatible checkpoint; choose resume')
        state = restore_training_checkpoint(
            latest, task, optimizer, scheduler, scaler if gpu['scaler_enabled'] else None,
            sampler, identity, config, method_sha, data['fingerprint']
        )
        step = int(state['step'])
        best_score = float(state['best_validation_score']) if state['best_validation_score'] is not None else -math.inf
        reconcile_history(history, step)
        print(f'Slot {slot}: resuming optimizer step {step}')
    else:
        if (RUN_MODE == 'resume' and slot in (RESUME_EXPECTED_SLOTS or set())) or history.exists() or (directory / 'selected_model.pt').exists():
            raise RuntimeError(f'Slot {slot}: no latest checkpoint; refusing a silent step-zero restart')
        del task, optimizer, scheduler, scaler, loader
        torch.cuda.empty_cache()
        smoke = run_gpu_smoke(spec, method, data, train_ids, val_ids, config,
                              directory, method_sha, identity)
        print(f'Slot {slot} GPU smoke:', smoke)
        task, tokenizer, teacher_tokenizer, base_meta, teacher_meta = build_task(
            method, spec['base_id'], spec, config, data['labels'], gpu['primary']
        )
        sampler = ResumableBatchSampler(len(train_ids), int(config['batch_size']), int(config['seed']))
        loader = create_loader(method, tokenizer, teacher_tokenizer, data['labels'], config,
                               data['train_text'], data['train_gold'], train_ids, sampler)
        optimizer, backend = make_optimizer(task, config)
        scheduler = make_scheduler(method, optimizer, config)
        scaler = make_scaler(gpu['scaler_enabled'])
    iterator = iter(loader)
    if latest.exists() and step:
        restore_rng(state['rng_state'])
    if step >= int(config['max_steps']):
        if not (directory / 'selected_model.pt').exists():
            raise RuntimeError(f'Slot {slot}: training marked complete but selected checkpoint is absent')
        completed = {'slot': slot, 'status': 'complete', 'step': step,
                     'target_step': int(config['max_steps']),
                     'selected': str(directory / 'selected_model.pt')}
        atomic_json(directory / 'training_result.json', completed)
        export_recovery(slot)
        return completed
    if time.monotonic() >= session_deadline:
        if latest.exists():
            export_recovery(slot)
        return {'slot': slot, 'status': 'time_budget', 'step': step}
    started = time.monotonic()
    try:
        while step < int(config['max_steps']):
            if time.monotonic() >= session_deadline:
                break
            step_started = time.monotonic()
            task.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for _ in range(int(config['gradient_accumulation_steps'])):
                try:
                    batch = next(iterator)
                except StopIteration:
                    sampler.next_epoch()
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = move_batch(batch, gpu['primary'])
                with torch.autocast(device_type='cuda', dtype=gpu['amp_dtype']):
                    loss = task_loss(task(**batch))
                    backward_loss = loss / int(config['gradient_accumulation_steps'])
                (scaler.scale(backward_loss) if gpu['scaler_enabled'] else backward_loss).backward()
                loss_total += float(loss.detach().cpu())
            if gpu['scaler_enabled']:
                scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(task.parameters(),
                                                       float(config['gradient_clip_norm'])).cpu())
            if not math.isfinite(norm):
                raise FloatingPointError(f'Slot {slot}: nonfinite gradient norm')
            if gpu['scaler_enabled']:
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < before:
                    append_jsonl(directory / 'overflow.jsonl',
                                 {'attempted_step': step + 1, 'epoch': sampler.epoch})
                    continue
            else:
                optimizer.step()
            scheduler.step()
            step += 1
            validation = None
            improved = False
            if step % int(config['validation_interval']) == 0 or step == int(config['max_steps']):
                cap = int(config.get('validation_document_cap', len(val_ids)))
                validation_ids = val_ids if step == int(config['max_steps']) else val_ids[:cap]
                validation = evaluate_student_method(method, task, tokenizer, data, validation_ids,
                                                     config, gpu['primary'])
                validation['evaluated_documents'] = len(validation_ids)
                validation['full_validation'] = len(validation_ids) == len(val_ids)
                atomic_json(directory / f'validation_step_{step:07d}.json', validation)
                score = float(validation['strict_macro_f1_local'])
                improved = score > best_score
                best_score = max(best_score, score)
            for gpu_id in gpu['ids']:
                torch.cuda.synchronize(gpu_id)
            optimizer_step_seconds = time.monotonic() - step_started
            row = {'step': step, 'epoch': sampler.epoch, 'loss': loss_total / int(config['gradient_accumulation_steps']),
                   'validation_macro_f1': None if validation is None else validation['strict_macro_f1_local'],
                   'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm': norm,
                   'seconds_elapsed': time.monotonic() - started,
                   'optimizer_step_seconds': optimizer_step_seconds,
                   'nominal_documents_per_second': (
                       int(config['batch_size']) * int(config['gradient_accumulation_steps']) /
                       max(optimizer_step_seconds, 1e-9)
                   ),
                   'gpu_memory_gib': {str(i): torch.cuda.max_memory_allocated(i) / 2**30 for i in gpu['ids']},
                   'optimizer_backend': backend, 'precision': gpu['precision']}
            append_jsonl(history, row)
            if step % int(config['checkpoint_interval']) == 0 or improved or step == int(config['max_steps']):
                payload = checkpoint_payload(task, optimizer, scheduler,
                                             scaler if gpu['scaler_enabled'] else None, sampler,
                                             step=step, best_score=best_score, identity=identity,
                                             history_path=history, config=config, base_meta=base_meta,
                                             method_sha=method_sha, dataset_fingerprint=data['fingerprint'])
                atomic_torch_save(latest, payload)
                if improved:
                    save_selected_model(directory, task, step=step, score=best_score,
                                        identity=identity, config=config, method_sha=method_sha,
                                        base_meta=base_meta,
                                        tokenizer_fingerprint=base_meta['tokenizer_fingerprint'],
                                        dataset_fingerprint=data['fingerprint'])
                if step % int(config['recovery_interval']) == 0:
                    export_recovery(slot)
        if not latest.exists() or int(torch.load(latest, map_location='cpu', weights_only=True)['step']) != step:
            payload = checkpoint_payload(task, optimizer, scheduler,
                                         scaler if gpu['scaler_enabled'] else None, sampler,
                                         step=step, best_score=best_score, identity=identity,
                                         history_path=history, config=config, base_meta=base_meta,
                                         method_sha=method_sha, dataset_fingerprint=data['fingerprint'])
            atomic_torch_save(latest, payload)
        result = {'slot': slot, 'status': 'complete' if step == int(config['max_steps']) else 'time_budget',
                  'step': step, 'target_step': int(config['max_steps']),
                  'best_strict_macro_f1_local': None if best_score == -math.inf else best_score,
                  'checkpoint': str(latest), 'recovery': str(directory / recovery_name(slot)),
                  'selected': str(directory / 'selected_model.pt')}
        atomic_json(directory / 'training_result.json', result)
        export_recovery(slot)
        return result
    except Exception as exc:
        append_jsonl(directory / 'failure_events.jsonl',
                     {'last_committed_step': step, 'error': f'{type(exc).__name__}: {exc}'})
        if latest.exists():
            export_recovery(slot)
        raise


# %% Resumable Phase-2 test predictions and six validated ZIP packages
def load_selected_task(spec, data, config, method, method_sha, device):
    slot = int(spec['slot'])
    directory = experiment_directory(slot)
    selected_path = directory / 'selected_model.pt'
    if not selected_path.is_file():
        raise FileNotFoundError(f'Slot {slot}: selected trained checkpoint missing')
    selected = torch.load(selected_path, map_location='cpu', weights_only=True)
    if selected.get('format') != 'cs781_phase2_selected_v1' or selected.get('method_sha256') != method_sha:
        raise RuntimeError(f'Slot {slot}: selected checkpoint provenance mismatch')
    if ((data['fingerprint'] is not None and selected.get('dataset_fingerprint') != data['fingerprint'])
            or selected.get('training_config') != dict(config)):
        raise RuntimeError(f'Slot {slot}: selected checkpoint data/config mismatch')
    task, tokenizer, _teacher_tokenizer, base_meta, _teacher_meta = build_task(
        method, spec['base_id'], spec, config, data['labels'], device
    )
    if selected.get('base_meta') != base_meta or selected.get('tokenizer_fingerprint') != base_meta['tokenizer_fingerprint']:
        raise RuntimeError(f'Slot {slot}: selected checkpoint Phase-1 base mismatch')
    bare_model(task).load_state_dict(selected['model_state'], strict=True)
    task.eval()
    return task, tokenizer, selected_path, selected


def generate_test_predictions(spec, data, config, method, method_sha, *, retry_rounds=3,
                              session_deadline=None):
    slot = int(spec['slot'])
    directory = experiment_directory(slot)
    model, tokenizer, selected_path, selected_state = load_selected_task(
        spec, data, config, method, method_sha, 'cuda:0'
    )
    partial_path = directory / 'predictions.partial.json'
    metadata_path = directory / 'predictions.provenance.json'
    expected = {'slot': slot, 'selected_sha256': sha256_file(selected_path),
                'method_sha256': method_sha,
                'training_dataset_fingerprint': selected_state['dataset_fingerprint'],
                'official_test_file_sha256': sha256_file(data['paths']['fold3-text.json']),
                'test_ids_sha256': hashlib.sha256(stable_json(sorted(data['test_text'])).encode()).hexdigest(),
                'config': dict(config), 'retry_rounds': retry_rounds}
    if metadata_path.exists() and load_json(metadata_path) != expected:
        raise RuntimeError(f'Slot {slot}: partial prediction provenance changed')
    if partial_path.exists():
        if not metadata_path.exists():
            raise RuntimeError(f'Slot {slot}: partial predictions have no provenance record')
        predictions = load_json(partial_path)
        if not isinstance(predictions, dict) or set(predictions) - set(data['test_text']):
            raise ValueError(f'Slot {slot}: invalid partial prediction IDs')
        for identifier, spans in predictions.items():
            validate_prediction_map({identifier: spans},
                                    {identifier: data['test_text'][identifier]}, data['labels'])
    else:
        predictions = {}
        atomic_json(metadata_path, expected)
    failures = {}
    new_successes = 0
    for round_index in range(retry_rounds):
        pending = [identifier for identifier in data['test_text'] if identifier not in predictions]
        if not pending:
            break
        print(f'Slot {slot}: prediction retry round {round_index + 1}, {len(pending)} missing IDs')
        for identifier in pending:
            if session_deadline is not None and time.monotonic() >= session_deadline:
                export_recovery(slot)
                raise RuntimeError(f'Slot {slot}: inference session budget reached; attach recovery and resume')
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    spans = method.predict_spans(model, data['test_text'][identifier],
                                                 tokenizer, data['labels'], dict(config))
                validate_prediction_map({identifier: spans},
                                        {identifier: data['test_text'][identifier]}, data['labels'])
                predictions[identifier] = spans
                atomic_json(partial_path, predictions)
                new_successes += 1
                if new_successes % INFERENCE_RECOVERY_INTERVAL == 0:
                    export_recovery(slot)
                failures.pop(identifier, None)
                append_jsonl(directory / 'inference_attempts.jsonl',
                             {'id': identifier, 'round': round_index, 'status': 'success',
                              'seconds': time.monotonic() - started})
            except Exception as exc:
                failures[identifier] = f'{type(exc).__name__}: {exc}'
                append_jsonl(directory / 'inference_attempts.jsonl',
                             {'id': identifier, 'round': round_index, 'status': 'failure',
                              'error': failures[identifier], 'seconds': time.monotonic() - started})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        atomic_json(directory / 'inference_failures.json', failures)
    missing = set(data['test_text']) - set(predictions)
    if missing:
        export_recovery(slot)
        raise RuntimeError(f'Slot {slot}: {len(missing)} test IDs remain unpredicted; partial file is resumable')
    validate_prediction_map(predictions, data['test_text'], data['labels'])
    return predictions, expected


def make_submission_zip(slot: int, predictions: Mapping[str, Any], data, provenance) -> dict[str, Any]:
    directory = experiment_directory(slot)
    report = validate_prediction_map(predictions, data['test_text'], data['labels'])
    output = directory / 'submission.zip'
    prediction_path = directory / 'prediction.json'
    atomic_json(prediction_path, predictions)
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.zip', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(prediction_path, 'prediction.json')
        with zipfile.ZipFile(temporary) as archive:
            if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
                raise ValueError('Phase-2 ZIP structure/CRC failed')
            inside = json.loads(archive.read('prediction.json'))
            validate_prediction_map(inside, data['test_text'], data['labels'])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {**report, 'slot': slot, 'zip_path': str(output), 'zip_sha256': sha256_file(output),
              'prediction_sha256': sha256_file(prediction_path), 'provenance': provenance}
    atomic_json(directory / 'submission_validation.json', result)
    return result


def prepare_phase2_data():
    data = load_official_phase2()
    if VALIDATION_FRACTION is None:
        raise RuntimeError('Student must choose VALIDATION_FRACTION before training')
    train_ids, val_ids = deterministic_split(list(data['train_text']), float(VALIDATION_FRACTION), SEED)
    manifest = {'dataset_fingerprint': data['fingerprint'], 'train_ids': train_ids,
                'validation_ids': val_ids, 'seed': SEED,
                'student_validation_fraction': VALIDATION_FRACTION,
                'official_test_excluded_from_training': True}
    path = WORK_ROOT / 'phase2_split_manifest.json'
    if path.exists() and load_json(path) != manifest:
        raise RuntimeError('Existing Phase-2 validation split differs; use a new workspace')
    atomic_json(path, manifest)
    return data, train_ids, val_ids


def phase2_preflight():
    if RUN_MODE not in {'auto', 'fresh', 'resume', 'inference_only'}:
        raise RuntimeError('Select auto, fresh, resume, or inference_only')
    if not STUDENT_AUTHORSHIP_CONFIRMED:
        print('DISCLOSURE: six AI-assisted reference methods are configured; this flag is not an authorship claim.')
    if RUN_MODE in {'resume', 'inference_only'}:
        if RESUME_EXPECTED_SLOTS is None or not set(RESUME_EXPECTED_SLOTS) <= set(range(1, 7)):
            raise RuntimeError('Set RESUME_EXPECTED_SLOTS explicitly for a resumed session')
        if RUN_MODE == 'inference_only' and set(RESUME_EXPECTED_SLOTS) != set(range(1, 7)):
            raise RuntimeError('Inference-only six-submission run needs recoveries for all six slots')
        for slot in RESUME_EXPECTED_SLOTS:
            if not (experiment_directory(slot) / 'checkpoint.latest.pt').exists() and discover_exact(recovery_name(slot), required=False) is None:
                raise RuntimeError(f'Slot {slot}: expected recovery checkpoint/ZIP is absent')
            restore_attached_recovery(slot)
    elif RUN_MODE == 'auto':
        for slot in range(1, 7):
            recovery = discover_exact(recovery_name(slot), required=False)
            if recovery is not None:
                restore_attached_recovery(slot)
    validate_experiment_plan()
    choose_gpu_runtime()
    if importlib.util.find_spec('tokenizers') is None:
        raise RuntimeError('Install the allowed tokenizers package in Kaggle before restoring Phase-1 BPE')
    data_names = ('labels.txt', 'fold3-text.json') if RUN_MODE == 'inference_only' else DATA_NAMES
    for name in (*data_names, *BASE_ARCHIVES.values()):
        discover_exact(name)
    if RUN_MODE == 'inference_only':
        for spec in EXPERIMENTS:
            slot = int(spec['slot'])
            selected_path = experiment_directory(slot) / 'selected_model.pt'
            if not selected_path.exists():
                raise FileNotFoundError(f'Slot {slot}: selected trained checkpoint is absent')
            selected = torch.load(selected_path, map_location='cpu', weights_only=True)
            if selected.get('format') != 'cs781_phase2_selected_v1':
                raise RuntimeError(f'Slot {slot}: selected checkpoint format is invalid')
            TRAINING_CONFIGS[slot] = dict(selected['training_config'])
    for spec in EXPERIMENTS:
        validate_training_config(int(spec['slot']))
        student_module_path(spec)
    common_module_path()
    if RUN_MODE != 'inference_only' and VALIDATION_FRACTION is None:
        raise RuntimeError('Choose the document-level validation fraction')
    print('Phase-2 six-experiment preflight: PASS')


def run_all_six():
    phase2_preflight()
    if RUN_MODE == 'inference_only':
        data = load_phase2_test_only()
        train_ids = val_ids = None
    else:
        data, train_ids, val_ids = prepare_phase2_data()
    deadline = time.monotonic() + GPU_SESSION_BUDGET_SECONDS
    reports = []
    for spec in EXPERIMENTS:
        if time.monotonic() >= deadline:
            print('Kaggle session safety budget reached; attach recovery ZIPs and resume')
            break
        slot = int(spec['slot'])
        print(f'===== Phase-2 submission {slot:02d}: {spec["method"]}, Phase-1 {spec["base_id"]} =====')
        if RUN_MODE == 'inference_only':
            directory = experiment_directory(slot)
            training_result = directory / 'training_result.json'
            saved_result = load_json(training_result) if training_result.exists() else {}
            if (saved_result.get('status') != 'complete' or
                    int(saved_result.get('step', -1)) != int(saved_result.get('target_step', -2)) or
                    int(saved_result.get('target_step', -1)) != int(TRAINING_CONFIGS[slot]['max_steps'])):
                raise RuntimeError(f'Slot {slot}: no verified completed training result for inference-only run')
            result = {'slot': slot, 'status': 'complete',
                      'step': saved_result['step']}
        else:
            result = train_one_experiment(spec, data, train_ids, val_ids, deadline)
        reports.append(result)
        if result['status'] != 'complete':
            print(f'Slot {slot} has not reached its student-defined training target; stopping safely')
            break
        directory = experiment_directory(slot)
        method, method_sha = load_student_method(spec, directory / spec['module_filename'])
        config = validate_training_config(slot)
        predictions, provenance = generate_test_predictions(
            spec, data, config, method, method_sha, session_deadline=deadline
        )
        package = make_submission_zip(slot, predictions, data, provenance)
        reports[-1]['package'] = package
        reports[-1]['recovery_after_packaging'] = str(export_recovery(slot))
        atomic_json(WORK_ROOT / 'phase2_run_status.json', reports)
    packages = [row['package'] for row in reports if 'package' in row]
    if len(packages) == 6:
        hashes = [row['prediction_sha256'] for row in packages]
        if len(hashes) != len(set(hashes)):
            raise RuntimeError('At least two Phase-2 predictions are identical; repeated submissions may not count')
        atomic_json(WORK_ROOT / 'six_submission_validation.json',
                    {'status': 'READY_FOR_MANUAL_CODABENCH_UPLOAD', 'submissions': packages})
        print('All six structurally validated submission.zip files are ready for manual upload.')
    else:
        print(f'{len(packages)}/6 validated packages ready; this is not a complete six-submission result.')
    return reports


if RUN_PHASE2:
    PHASE2_RESULTS = run_all_six()
else:
    PHASE2_RESULTS = None
    print('Preparation mode only: no training or official test predictions were generated.')




