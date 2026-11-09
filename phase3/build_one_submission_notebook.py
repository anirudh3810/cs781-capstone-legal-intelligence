"""Build the self-contained Phase 3 one-submission Kaggle notebook."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
RUNNER = (HERE / 'one_submission_runner.py').read_text(encoding='utf8')
OUTPUT = HERE.parent / 'CS781_Phase3_One_Strong_Submission_RunAll.ipynb'

intro = """# CS781 Phase 3 — one stronger submission

Run this notebook in **the same Kaggle session** as
`CS781_Phase3_Six_Submissions_RunAll`, with the two GPUs enabled. It uses the
six-run folder and `CS781_Phase3_All_Artifacts.zip` already in
`/kaggle/working`. If this is a fresh session, attach that complete artifact ZIP
as a Kaggle Input. No Hugging Face download or `HF_TOKEN` is needed when the
previous artifact is complete.

The notebook trains two larger models from your own Phase 1 checkpoints: a
fully trainable Phase 1 model 03 with a 3-layer bidirectional document head,
and a fully trainable model 01 with the same head. Each uses both GPUs through
distributed training. It checks CL, IT, and per-label macro F1 after **every**
epoch, saves `best.pt` and a resumable `latest.pt` every epoch, and stops when
the development score stops improving or the session time budget is reached.
It tests combinations with the strongest earlier models and a transition
decoder on the official development set, then creates **one** test prediction.

**Run All**. Download these final files from `/kaggle/working`:

- `CS781_Phase3_One_Submission.zip` — upload this to Codabench. It contains
  root-level `prediction.json`.
- `CS781_Phase3_One_All_Artifacts.zip` — portable recovery of the previous
  six-run archive plus this run's code, model states, optimizer state while
  training, metrics, predictions, and submission ZIP.

The run aims to improve the current 0.5058 private score. A private score of
0.8 cannot be guaranteed from the 80 training documents and the available
Phase 1 models; only Codabench can measure it. Development F1 is reported
separately from the private score. This notebook does not submit automatically.

### Evidence used to choose this run

| Earlier slot | Method | Best development macro F1 (all 13) | Codabench |
|---|---|---:|---:|
| 01 | LoRA | 0.4150 | 0.4600 |
| 02 | rsLoRA | 0.4226 | 0.4437 |
| 03 | IA3 | 0.3508 | 0.3835 |
| 04 | LoRMA | 0.4207 | 0.4239 |
| 05 | prompt | 0.4111 | 0.4538 |
| 06 | distillation | 0.4113 | **0.5058** |

The training set has 16,845 labeled sentences in 80 documents. Development
has 2,142 sentences in only 10 documents, so small changes in its macro F1 can
be noisy. Most earlier runs peaked before the last epoch. This notebook
anchors ensemble selection on slot 06's measured private result, requires a
meaningful development improvement to change it, and retains the best epoch.
The two new learners train the *full* Phase 1 backbones and a larger document
head instead of spending the budget on more epochs of the earlier adapters.

**Course use:** The implementation is AI assisted. Review it and disclose
assistance as required by the course rules.
"""

orchestrate = '''import json, os, subprocess, sys, time
from pathlib import Path

RUNNER = '/kaggle/working/cs781_phase3_one_runner.py'
ROOT = Path('/kaggle/working/CS781_Capstone/phase3_one')
WORK = Path('/kaggle/working')
deadline = time.time() + 9.5 * 3600
environment = os.environ.copy()
environment['CS781_ONE_DEADLINE'] = str(deadline)

def call(*args, distributed=False):
    cmd = ([sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=2', RUNNER] if distributed else [sys.executable, RUNNER])
    cmd += list(args)
    print('RUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=environment)

call('prepare')
for name in ('full03', 'full01'):
    status_path = ROOT / name / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('complete'):
        print(name, 'already complete; best supported dev macro-F1 =',
              status.get('best_present_f1'), flush=True)
        continue
    if time.time() >= deadline - 3600:
        print('Time guard: using saved best checkpoints for final packaging.', flush=True)
        break
    call('train', '--name', name, distributed=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    print(name, 'best supported dev macro-F1 =', status.get('best_present_f1'),
          'complete =', status.get('complete'), flush=True)

sources = [name for name in ('full03', 'full01') if (ROOT / name / 'best.pt').exists()]
sources += ['prior6', 'prior2', 'prior1', 'prior4', 'prior5', 'prior3']
if not any(name.startswith('full') for name in sources):
    raise RuntimeError('No new checkpoint was saved; inspect training output and rerun.')
for source in sources:
    cache = ROOT / 'probabilities' / 'dev' / (source + '.pt')
    if not cache.exists():
        call('infer', '--source', source, '--split', 'dev', distributed=True)

# The first finalization picks the development-selected ensemble and lists
# the test probability files it needs. These are inferred only once.
call('finalize')
choice = json.loads((ROOT / 'selection.json').read_text())
for source in choice['weights']:
    cache = ROOT / 'probabilities' / 'test' / (source + '.pt')
    if not cache.exists():
        call('infer', '--source', source, '--split', 'test', distributed=True)
call('finalize')

report = ROOT / 'one_report.json'
if not report.exists():
    raise RuntimeError('Final submission report is missing')
print(report.read_text())
assert (WORK / 'CS781_Phase3_One_Submission.zip').is_file()
assert (WORK / 'CS781_Phase3_One_All_Artifacts.zip').is_file()
print('Upload /kaggle/working/CS781_Phase3_One_Submission.zip to Codabench')
'''

notebook = nbf.v4.new_notebook()
notebook.cells = [
    nbf.v4.new_markdown_cell(intro),
    nbf.v4.new_markdown_cell('## Embedded runner'),
    nbf.v4.new_code_cell('%%writefile /kaggle/working/cs781_phase3_one_runner.py\n' + RUNNER),
    nbf.v4.new_markdown_cell('## Run or resume and package one submission'),
    nbf.v4.new_code_cell(orchestrate),
]
notebook.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}
nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
