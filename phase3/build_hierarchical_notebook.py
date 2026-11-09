"""Write a self-contained Kaggle Run All notebook with the runner embedded."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / 'hierarchical_runner.py').read_text(encoding='utf8')
OUTPUT = HERE.parent / 'CS781_Phase3_Hierarchical_RunAll.ipynb'

intro = r'''# CS781 Phase 3 — target-aware hierarchical submission

Run this notebook with **two GPUs** in the same Kaggle session as the six-run
notebook. It reads the staged six-run files. In a fresh session, attach
`CS781_Phase3_All_Artifacts.zip` or this notebook's own recovery ZIP as an
Input. No Hugging Face token or external model download is needed.

**Run All** trains five document-held-out folds and, time permitting, a final
model on all 80 training documents. Each fold uses 60 fitting documents, four
separate documents for checkpoint selection, and 16 untouched documents for
out-of-fold scoring. Each starts from your own Phase 1 model 03.
Each target sentence is encoded separately, so neighboring text cannot swamp
the target representation. A three-layer 128-sentence attention head with a
document-level GRU then learns the neighboring and long-range context. The
two GPUs share both the encoder and document training through PyTorch DDP.

The notebook reports supported-label and all-13 macro F1, including CL/IT
breakdowns, after each epoch. It saves best and resumable optimizer/scaler
checkpoints, periodically refreshes a verified recovery archive, and can be
rerun after interruption. It uses document-held-out predictions from the five
folds to gate the new model before comparing it with the known 0.5058 private
slot 06. One fixed blend is available. Development labels are used only for
evaluation and selection; test labels are never used.

At the end, download:

- `/kaggle/working/CS781_Phase3_Hierarchical_Submission.zip` — upload this
  **one** ZIP to Codabench. It has root-level `prediction.json`.
- `/kaggle/working/CS781_Phase3_Hierarchical_All_Artifacts.zip` — portable
  recovery containing staged Phase 1/Phase 3 inputs, code, completed fold
  models, optimizer states for unfinished stages, cached features, metrics,
  selection and submission.

The target is a higher private macro F1. **0.8 cannot be guaranteed** from
local validation; only Codabench can measure the private result. The notebook
does not upload the submission automatically. This code is AI assisted;
review and disclose assistance as required by the course rules.
'''

orchestrate = r'''import json, os, subprocess, sys, time
from pathlib import Path

RUNNER = '/kaggle/working/cs781_phase3_hierarchical_runner.py'
ROOT = Path('/kaggle/working/CS781_Capstone/phase3_hier')
WORK = Path('/kaggle/working')
deadline = time.time() + 9.75*3600
environment = os.environ.copy()
environment['CS781_HIER_DEADLINE'] = str(deadline)
environment['TOKENIZERS_PARALLELISM'] = 'false'

def call(action, fold=None, split=None, distributed=False):
    cmd = ([sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=2', RUNNER] if distributed else [sys.executable, RUNNER])
    cmd.append(action)
    if fold is not None: cmd += ['--fold', str(fold)]
    if split is not None: cmd += ['--split', split]
    print('RUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=environment)

def status(fold, kind):
    path = ROOT / ('final' if fold == 'final' else f'fold_{int(fold):02d}') / f'{kind}_status.json'
    return json.loads(path.read_text()) if path.exists() else {}

def location(fold):
    return ROOT / ('final' if fold == 'final' else f'fold_{int(fold):02d}')

call('prepare')
try:
    subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader'], check=False)
except FileNotFoundError:
    pass

completed = []
for fold in range(5):
    if time.time() > deadline - 70*60:
        print('Time guard: preserving time for selection and packaging.', flush=True)
        break
    if not status(fold, 'sentence').get('complete'):
        call('sentence', fold=fold, distributed=True)
    if not all((location(fold) / f'features_{split}.pt').exists()
               for split in ('train', 'dev', 'test')):
        call('extract', fold=fold, distributed=True)
    if not status(fold, 'head').get('complete'):
        call('head', fold=fold, distributed=True)
    for split in ('train', 'dev'):
        if not (location(fold) / f'head_prob_{split}.pt').exists():
            call('predict-head', fold=fold, split=split, distributed=True)
    completed.append(fold)
    call('snapshot')
    print('Completed fold', fold, 'head best supported macro F1 =',
          status(fold, 'head').get('best_supported_f1'), flush=True)
    try:
        subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                        '--format=csv,noheader'], check=False)
    except FileNotFoundError:
        pass

if len(completed) >= 3 and time.time() < deadline - 85*60:
    fold = 'final'
    if not status(fold, 'sentence').get('complete'):
        call('sentence', fold=fold, distributed=True)
    if not all((location(fold) / f'features_{split}.pt').exists()
               for split in ('train', 'dev', 'test')):
        call('extract', fold=fold, distributed=True)
    if not status(fold, 'head').get('complete'):
        call('head', fold=fold, distributed=True)
    if not (location(fold) / 'head_prob_dev.pt').exists():
        call('predict-head', fold=fold, split='dev', distributed=True)
    call('snapshot')

call('select')
choice = json.loads((ROOT / 'selection.json').read_text())
print('Chosen source:', choice['selected'])
print('Fold OOF supported macro F1:', None if choice['oof'] is None else
      choice['oof']['macro_supported'])
for name, values in choice['development'].items():
    print(name, 'dev supported macro F1', values['macro_supported'],
          'dev all-13 macro F1', values['macro_all13'])
if choice['selected'] in ('new', 'blend'):
    for model in choice['models']:
        if not (location(model) / 'head_prob_test.pt').exists():
            call('predict-head', fold=model, split='test', distributed=True)
call('finalize')
assert (WORK / 'CS781_Phase3_Hierarchical_Submission.zip').is_file()
assert (WORK / 'CS781_Phase3_Hierarchical_All_Artifacts.zip').is_file()
print((ROOT / 'report.json').read_text())
print('Upload CS781_Phase3_Hierarchical_Submission.zip to Codabench.')
'''

notebook = nbf.v4.new_notebook()
notebook.cells = [
    nbf.v4.new_markdown_cell(intro),
    nbf.v4.new_markdown_cell('## Embedded training, recovery and submission runner'),
    nbf.v4.new_code_cell('%%writefile /kaggle/working/cs781_phase3_hierarchical_runner.py\n' + SOURCE),
    nbf.v4.new_markdown_cell('## Run or resume all stages and package one submission'),
    nbf.v4.new_code_cell(orchestrate),
]
notebook.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}
nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
