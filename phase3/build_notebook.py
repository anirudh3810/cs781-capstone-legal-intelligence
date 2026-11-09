"""Build the self-contained Kaggle notebook from runner.py."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / 'runner.py').read_text(encoding='utf8')
OUTPUT = HERE.parent / 'CS781_Phase3_Six_Submissions_RunAll.ipynb'

intro = """# CS781 Phase 3 — six rhetorical-role submissions

This self-contained notebook uses **your four Phase 1 recovery ZIPs** and the
official Phase 3 data. Select Kaggle's **two-GPU accelerator**, enable Internet,
and add a Hugging Face read token as the enabled Kaggle secret `HF_TOKEN`.
Attach these exact files as Kaggle Inputs:

- `phase1_submission01_recovery.zip`
- `phase1_submission02_recovery.zip`
- `phase1_submission03_recovery.zip`
- `phase1_submission04_recovery.zip`

Then **Run All**. The notebook downloads the seven official Phase 3 JSON files
from one pinned Hugging Face dataset revision, stages the Phase 1 models,
trains six methods with two-process distributed GPU training, evaluates **all**
CL and IT development documents after every epoch, creates one submission ZIP
per completed method, and writes a self-contained recovery ZIP. It can train
each method for up to 32 epochs or 10 cumulative training hours, with early
stopping on development macro-F1. An 8.5-hour session guard pauses cleanly at
an epoch boundary. The total six-run campaign may require several Kaggle
sessions.

Download `CS781_Phase3_All_Artifacts.zip` after each session. For a new session,
attach **that one ZIP** as a Kaggle Input and Run All again; it contains the
staged model weights, source, official data, optimizer checkpoint for an active
run, completed models, reports, predictions, and packages. The four Phase 1
archives and `HF_TOKEN` are unnecessary for a complete restored artifact.
The notebook rejects ambiguous duplicate recovery ZIPs.

Each final `CS781_Phase3_Submission_XX.zip` contains root-level
`prediction.json`. Inspect the per-epoch macro-F1 and per-label F1 before
manual Codabench upload. No model can guarantee a private test score.

**Course use:** This notebook and its model design were AI-assisted. Review
the implementation and disclose this assistance truthfully under your course
rules. It does not submit automatically to Codabench.
"""

runner_cell = '%%writefile /kaggle/working/cs781_phase3_runner.py\n' + SOURCE

orchestrate = '''import json, os, subprocess, sys, time
from pathlib import Path

RUNNER = '/kaggle/working/cs781_phase3_runner.py'
ROOT = Path('/kaggle/working/CS781_Capstone/phase3')
started = time.time()
deadline = started + 8.5 * 3600
environment = os.environ.copy()
environment['PHASE3_SESSION_DEADLINE'] = str(deadline)

def call(*args, distributed=False):
    if distributed:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=2', RUNNER, *args]
    else:
        command = [sys.executable, RUNNER, *args]
    print('RUN:', ' '.join(command), flush=True)
    subprocess.run(command, env=environment, check=True)

call('prepare')
for slot in range(1, 7):
    if time.time() > deadline - 1200:
        print('Session guard reached; attach the all-artifacts ZIP in the next session.')
        break
    folder = ROOT / f'slot_{slot:02d}'
    status_path = folder / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if not status.get('complete'):
        call('train', '--slot', str(slot), distributed=True)
        status = json.loads(status_path.read_text())
    if status.get('complete') and not (folder / 'package.json').exists():
        call('predict', '--slot', str(slot), distributed=True)
    print(f'Slot {slot}: {status.get("best_macro_f1", "pending")} best dev macro-F1; '
          f'complete={status.get("complete", False)}', flush=True)
    if not status.get('complete'):
        print('Training paused. Save the all-artifacts ZIP and Run All in a new session.')
        break
call('finish')
report = ROOT / 'six_submission_report.json'
if report.exists():
    print(report.read_text())
print('Recovery: /kaggle/working/CS781_Phase3_All_Artifacts.zip')
'''

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(intro),
    nbf.v4.new_markdown_cell('## Embedded runner\n\nThis cell writes all training and recovery code into the Kaggle session.'),
    nbf.v4.new_code_cell(runner_cell),
    nbf.v4.new_markdown_cell('## Run or resume all six experiments'),
    nbf.v4.new_code_cell(orchestrate),
]
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.11'}}
nbf.validate(nb)
nbf.write(nb, OUTPUT)
print(OUTPUT)
