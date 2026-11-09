"""Build one portable Kaggle notebook for the Phase 3 high-F1 experiment."""
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / 'CS781_Phase3_HighF1_RunAll.ipynb'


def embedded(source, target):
    body = (HERE / source).read_text(encoding='utf8')
    return nbf.v4.new_code_cell(f'%%writefile /kaggle/working/{target}\n{body}')


def main():
    notebook = nbf.v4.new_notebook()
    notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook.cells = [
        nbf.v4.new_markdown_cell('''# CS781 Phase 3: one stronger submission

Attach the four Phase 1 recovery datasets shown in the screenshot, enable the **two-GPU** Kaggle accelerator and Internet, and add the private Hugging Face dataset token to Kaggle Secrets under **HF_TOKEN**. Then choose **Run All**. The notebook downloads only the official Phase 3 data from the course Hugging Face dataset and starts only from your Phase 1 weights. It does not fetch another pretrained model.

This run trains the Phase 1 base 03 model with target-sentence context, more unfrozen upper layers, a recurrent document-window head, and auxiliary heads for the three label pairs that were most often confused in the failed runs. Two-GPU distributed training, mixed precision, gradient checkpointing, early stopping on official development F1, and document-level out-of-fold checks are used. The larger Phase 1 base 01 is trained as a second branch only if time permits; it enters the final submission only if matched out-of-fold evidence supports it.

**Outputs:** `/kaggle/working/submission.zip` and `/kaggle/working/CS781_Phase3_HighF1_Submission.zip` are the same submission. `/kaggle/working/CS781_Phase3_HighF1_All_Artifacts.zip` holds data, source, checkpoints, intermediate predictions, reports, and the submission for recovery. Completed training chunks are archived only after both distributed workers exit. To resume in a new session, attach the latest recovery ZIP as a Kaggle Input alongside the four Phase 1 inputs and rerun. A Kaggle session failure can still make its local outputs inaccessible until Kaggle persists them; download or save the recovery ZIP after each completed session.

**0.55 macro F1 is the target, not a guaranteed private score.** The notebook records development and document-held-out F1, and it will show when the evidence falls short. The private score is known only after submission. Review course AI-assistance disclosure rules before uploading.'''),
        nbf.v4.new_markdown_cell('## Embedded Phase 1 recovery and official-data code'),
        embedded('runner.py', 'cs781_phase3_source_utils.py'),
        nbf.v4.new_markdown_cell('## Embedded high-F1 training, validation, and packaging code'),
        embedded('high_f1_runner.py', 'cs781_phase3_high_f1_runner.py'),
        nbf.v4.new_markdown_cell('## Run the complete experiment'),
        nbf.v4.new_code_cell('''from pathlib import Path
import json, subprocess, sys, time

RUNNER = Path('/kaggle/working/cs781_phase3_high_f1_runner.py')
ROOT = Path('/kaggle/working/CS781_Capstone/phase3_high_f1')
SUBMISSION = Path('/kaggle/working/CS781_Phase3_HighF1_Submission.zip')
ARTIFACT = Path('/kaggle/working/CS781_Phase3_HighF1_All_Artifacts.zip')
STARTED = time.monotonic()
MAX_HOURS = 9.25

def elapsed_hours():
    return (time.monotonic() - STARTED) / 3600

def run(action, *, branch=None, stage=None, chunk_epochs=None, distributed=False):
    command = [sys.executable]
    if distributed:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    command += [str(RUNNER), action]
    if branch is not None:
        command += ['--branch', branch]
    if stage is not None:
        command += ['--stage', stage]
    if chunk_epochs is not None:
        command += ['--chunk-epochs', str(chunk_epochs)]
    print('RUN', ' '.join(command), 'elapsed hours', round(elapsed_hours(), 2), flush=True)
    subprocess.run(command, check=True)

def archive():
    # The ZIP is replaced atomically only after it is completely written.
    # Keep this outside torchrun; a worker must never wait for ZIP creation.
    command = [sys.executable, str(RUNNER), 'snapshot']
    print('ARCHIVE after workers exited', flush=True)
    subprocess.run(command, check=True, timeout=1200)

def complete(branch, stage):
    status = ROOT / 'runs' / branch / stage / 'status.json'
    return status.is_file() and json.loads(status.read_text())['complete']

def train(branch, stage, reserve_hours=None):
    while not complete(branch, stage):
        run('train', branch=branch, stage=stage, chunk_epochs=2, distributed=True)
        archive()
        if reserve_hours is not None and not complete(branch, stage) and elapsed_hours() >= MAX_HOURS - reserve_hours:
            print('Time reserve reached; checkpointed incomplete stage', branch, stage, flush=True)
            return False
    return True

run('prepare')
archive()

# Create a complete, uploadable result before spending time on folds.
train('03', 'full')
run('infer', branch='03', distributed=True)
run('select')
run('package')
archive()
print('PRIMARY SUBMISSION READY:', SUBMISSION, flush=True)

# Compare modes on held-out documents. These folds never train on the document
# being scored. Package and archive after each fold for an interruptible run.
for fold in range(5):
    if elapsed_hours() >= MAX_HOURS - 1.5:
        print('Time reserve reached before primary fold', fold, flush=True)
        break
    if not train('03', f'fold_{fold}', reserve_hours=1.5):
        break
    run('select')
    run('package')
    archive()

# Try the independent width-768 Phase 1 base 01 only if there is enough time
# for two validation folds and a final full model. The probe gates its use.
if elapsed_hours() < MAX_HOURS - 3.0 and sum(complete('03', f'fold_{i}') for i in range(5)) >= 2:
    run('prepare-01')
    archive()
    for fold in range(2):
        if elapsed_hours() >= MAX_HOURS - 1.75:
            print('Time reserve reached during wider-model validation.', flush=True)
            break
        if not train('01', f'fold_{fold}', reserve_hours=1.75):
            break
        archive()
    if all(complete('01', f'fold_{i}') for i in range(2)):
        run('probe-wide')
        probe = json.loads((ROOT / 'wide_probe.json').read_text())
        if probe['continue'] and elapsed_hours() < MAX_HOURS - 2.0:
            if train('01', 'full', reserve_hours=.5):
                run('infer', branch='01', distributed=True)
                run('select')
                run('package')
                archive()
        else:
            print('Wider model was not promoted; retaining the validated primary submission.', flush=True)

# Ensure the newest reports and submission are present in the final archive.
run('select')
run('package')
archive()
report = json.loads((ROOT / 'report.json').read_text())
print(json.dumps(report, indent=2), flush=True)
print('Submission:', SUBMISSION, 'Recovery:', ARTIFACT, flush=True)
'''),
        nbf.v4.new_markdown_cell('## Verify the submission and complete recovery ZIP'),
        nbf.v4.new_code_cell('''from pathlib import Path
import hashlib, json, zipfile

submission = Path('/kaggle/working/CS781_Phase3_HighF1_Submission.zip')
artifact = Path('/kaggle/working/CS781_Phase3_HighF1_All_Artifacts.zip')
with zipfile.ZipFile(submission) as z:
    assert z.namelist() == ['prediction.json']
    prediction = json.loads(z.read('prediction.json'))
assert len(prediction) == 10
with zipfile.ZipFile(artifact) as z:
    manifest = json.loads(z.read('integrity.json'))['files']
    assert set(manifest) == set(z.namelist()) - {'integrity.json'}
    for name, expected in manifest.items():
        digest = hashlib.sha256()
        size = 0
        with z.open(name) as member:
            for block in iter(lambda: member.read(4 << 20), b''):
                size += len(block)
                digest.update(block)
        assert size == expected['bytes'] and digest.hexdigest() == expected['sha256'], name
print('Verified:', submission, artifact)
print(json.dumps(json.loads(Path('/kaggle/working/CS781_Capstone/phase3_high_f1/report.json').read_text()), indent=2))
'''),
    ]
    nbf.validate(notebook)
    OUTPUT.write_text(nbf.writes(notebook), encoding='utf8')
    print(OUTPUT)


if __name__ == '__main__':
    main()
