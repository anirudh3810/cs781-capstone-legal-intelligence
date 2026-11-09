"""Embed the exact inspected runner sources in a portable Kaggle Run All notebook."""
from pathlib import Path
import json
import hashlib

HERE = Path(__file__).resolve().parent
DEST = HERE.parent / 'CS781_Phase3_Context_Joint_RunAll.ipynb'


def cell(kind, source):
    identity = hashlib.sha256((kind + source).encode('utf8')).hexdigest()[:16]
    item = {'cell_type': kind, 'id': identity, 'metadata': {},
            'source': source.splitlines(keepends=True)}
    if kind == 'code':
        item.update({'execution_count': None, 'outputs': []})
    return item


intro = """# CS781 Phase 3: context-aware joint training, one submission

**Run All** in Kaggle with **two GPUs** enabled. Keep the prior six-submission session alive, or attach `CS781_Phase3_All_Artifacts.zip`, `CS781_Phase3_Hierarchical_All_Artifacts.zip`, or this notebook's own recovery ZIP as a Kaggle Input. The runner verifies the recovery manifest and restores the Phase 1 model 03, official Phase 3 files, and prior slot 06 model. It downloads no external model and needs no Hugging Face token.

The first two fits compare a slot-06-like context baseline and a new end-to-end joint encoder/document model on the **same 16 untouched training documents**. The new model reads two neighboring sentences on each side, marks the target sentence, pools only target token states, and backpropagates document and role losses through all 24 Phase 1 encoder layers. The document head covers 16 sentences with overlap. A pilot gate requires at least +0.03 supported-label macro F1 with no material CL/IT regression and improvement in at least two focus labels. If it passes, Run All fits two more folds and a final model on all 80 training documents. Official dev is checked after every epoch; it is not used to train the models. Private test labels are never used.

Training uses two-GPU DDP, FP16, activation checkpointing, layerwise learning rates, and optimizer checkpoints every 100 updates. Both GPUs' utilization and memory are sampled during training. The notebook refreshes a verified recovery archive while running and after each stage. A single 9.45-hour budget includes packaging time. Stop and rerun the notebook with its recovery archive attached if Kaggle interrupts the session.

**Outputs:** `/kaggle/working/CS781_Phase3_Context_Joint_Submission.zip` contains the root-level `prediction.json`; `/kaggle/working/CS781_Phase3_Context_Joint_All_Artifacts.zip` contains recovery code, source weights, official data, checkpoints, metrics, predictions, and logs. `/kaggle/working/submission.zip` is a convenience copy. Inspect `phase3_joint/report.json`, `pilot_gate.json`, and `selection.json` before uploading. The 0.8 private score is a target, not a promised or locally measurable outcome. If the measured gate fails, the package records and uses the known slot 06 fallback rather than silently presenting it as an improvement.
"""

orchestrate = r'''import json, os, subprocess, sys, time
from pathlib import Path

RUNNER = '/kaggle/working/cs781_phase3_context_joint_runner.py'
ROOT = Path('/kaggle/working/CS781_Capstone/phase3_joint')
DEADLINE = time.time() + 9.45 * 3600
ENV = os.environ.copy()
ENV['TOKENIZERS_PARALLELISM'] = 'false'
ENV['PYTHONUNBUFFERED'] = '1'

def call(action, stage=None, split=None, distributed=False):
    command = ([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=2', RUNNER] if distributed else [sys.executable, RUNNER])
    command.append(action)
    if stage is not None: command += ['--stage', stage]
    if split is not None: command += ['--split', split]
    command += ['--deadline', str(DEADLINE)]
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, env=ENV, check=True)

subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], check=True)
import torch
assert torch.cuda.device_count() == 2, 'Two Kaggle GPUs are required.'
call('prepare')

for stage in ('pilot_baseline', 'pilot_joint'):
    call('train', stage=stage, distributed=True)
call('gate')
gate = json.loads((ROOT / 'pilot_gate.json').read_text())

if gate['pass']:
    for stage in ('confirm_1', 'confirm_2', 'final'):
        if time.time() >= DEADLINE - 35 * 60:
            print('Time guard: preserving time for final verification and ZIP files.', flush=True)
            break
        call('train', stage=stage, distributed=True)
    if (ROOT / 'final' / 'selected.pt').is_file():
        call('infer', stage='final', split='test', distributed=True)

call('choose')
call('package')
report = json.loads((ROOT / 'report.json').read_text())
selection = json.loads((ROOT / 'selection.json').read_text())
print('SELECTED:', report['selected'])
print('REASON:', report['selection_reason'])
print('LABELS DIFFERENT FROM PRIOR:', report['labels_changed_vs_slot06'])
print('SUBMISSION:', report['submission'])
print('RECOVERY:', '/kaggle/working/CS781_Phase3_Context_Joint_All_Artifacts.zip')
print('PRIVATE SCORE: unknown until Codabench upload')
'''

verify = r'''from pathlib import Path
import json, zipfile, hashlib
root = Path('/kaggle/working/CS781_Capstone/phase3_joint')
submission = Path('/kaggle/working/CS781_Phase3_Context_Joint_Submission.zip')
recovery = Path('/kaggle/working/CS781_Phase3_Context_Joint_All_Artifacts.zip')
with zipfile.ZipFile(submission) as z:
    assert z.namelist() == ['prediction.json']
    predictions = json.loads(z.read('prediction.json'))
assert len(predictions) == 10 and sum(map(len, predictions.values())) == 2197
with zipfile.ZipFile(recovery) as z:
    manifest = json.loads(z.read('integrity.json'))['files']
    assert 'hier/report.json' in manifest and 'hier/context_joint_runner.py' in manifest
    assert set(manifest) == set(z.namelist()) - {'integrity.json'}
    for name, expected in manifest.items():
        digest = hashlib.sha256()
        size = 0
        with z.open(name) as member:
            for block in iter(lambda: member.read(4 << 20), b''):
                size += len(block)
                digest.update(block)
        assert size == expected['bytes'] and digest.hexdigest() == expected['sha256'], name
print('ZIP checks passed:', submission.stat().st_size, 'submission bytes;',
      round(recovery.stat().st_size / 2**30, 2), 'GiB recovery')
print((root / 'report.json').read_text())
'''

sources = [
    cell('markdown', intro),
    cell('markdown', '## Embedded recovery utilities and prior-run loader\n'),
    cell('code', '%%writefile /kaggle/working/cs781_phase3_hierarchical_runner.py\n' +
         (HERE / 'hierarchical_runner.py').read_text(encoding='utf8')),
    cell('markdown', '## Embedded context-aware joint trainer\n'),
    cell('code', '%%writefile /kaggle/working/cs781_phase3_context_joint_runner.py\n' +
         (HERE / 'context_joint_runner.py').read_text(encoding='utf8')),
    cell('markdown', '## Run, resume, select, and package\n'),
    cell('code', orchestrate),
    cell('markdown', '## Verify the two ZIP outputs\n'),
    cell('code', verify),
]

notebook = {'cells': sources,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                        'name': 'python3'},
                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}
DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf8')
print(DEST)
