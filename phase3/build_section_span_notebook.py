"""Embed all Phase 3 section-run sources into one portable Kaggle notebook."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / 'CS781_Phase3_Sections_RunAll.ipynb'


def write_cell(source_name, target_name):
    source = (HERE / source_name).read_text(encoding='utf8')
    return nbf.v4.new_code_cell(f'%%writefile /kaggle/working/{target_name}\n' + source)


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        nbf.v4.new_markdown_cell('''# CS781 Phase 3: full-document section tracker

Run All in the **same Kaggle session with both GPUs enabled**. This revision repairs the two-GPU auxiliary-loss crash and the later recovery-ZIP barrier timeout. It detects either failed run's saved configuration and reuses completed checkpoints and feature files. If the Kaggle session has restarted, attach the latest `CS781_Phase3_Sections_All_Artifacts.zip` from the failed run as a Kaggle Input. The live six-run files or `CS781_Phase3_All_Artifacts.zip` can also supply the original inputs. It needs no Hugging Face token or external model. It restores the user's Phase 1 base 03 and official Phase 3 data. The prior submission is read only to audit changed labels; it is never used as the prediction.

**Repair build (2026-10-13):** upload this notebook as a new Kaggle notebook. A saved copy of the earlier execution still has `u.snapshot()` inside the distributed workers and will repeat the `outer_1` NCCL timeout. This build archives only after both workers exit.

This experiment trains a target-aware local encoder with document-balanced sampling, generates out-of-sample features by document cross-fitting, and trains a full-document model with role, boundary, speaker, precedent-stance, and span objectives. A duration-limited section decoder can represent sections longer than 64 sentences by adjoining same-role spans. Five outer document folds evaluate the candidates; the final model is trained on all 80 training documents. Official development labels select checkpoints. Test labels are never read.

The final path runs first, so a **new** submission and recovery archive are created before the outer-fold audit. Recovery ZIP writing now runs only after distributed workers have exited; completed optimizer checkpoints are pruned to keep archives manageable. An interrupted Run All can resume from its own `CS781_Phase3_Sections_All_Artifacts.zip`. The local encoder and feature extraction use two-GPU distributed execution; the full-document head also trains on both GPUs. Model, feature, prediction, and metric files, plus optimizer checkpoints for unfinished stages, are kept for recovery. After fold 0, the notebook stops further folds when every new candidate is below 0.50 supported macro F1 on that held-out fold. This avoids spending hours on a weak pilot. Otherwise expect a run of up to about 9.5 hours; actual time depends on Kaggle hardware.

**Outputs:** `/kaggle/working/CS781_Phase3_Sections_Submission.zip` contains root-level `prediction.json`. `/kaggle/working/CS781_Phase3_Sections_All_Artifacts.zip` contains recovery files. `/kaggle/working/submission.zip` is a convenience copy. Read `phase3_sections/report.json` and `selection.json` before uploading. A private macro F1 of 0.8 is a target, not a guaranteed result; only Kaggle can measure it.'''),
        nbf.v4.new_markdown_cell('## Embedded recovery and tokenizer code'),
        write_cell('hierarchical_runner.py', 'cs781_phase3_hierarchical_runner.py'),
        write_cell('context_joint_runner.py', 'cs781_phase3_context_joint_runner.py'),
        nbf.v4.new_markdown_cell('## Full-document section training and submission code'),
        write_cell('section_span_runner.py', 'cs781_phase3_section_span_runner.py'),
        nbf.v4.new_markdown_cell('## Run the final model and the document-fold audit'),
        nbf.v4.new_code_cell('''from pathlib import Path
import json, subprocess, sys, time

runner = Path('/kaggle/working/cs781_phase3_section_span_runner.py')
root = Path('/kaggle/working/CS781_Capstone/phase3_sections')
session_started = time.time()
TIME_BUDGET_HOURS = 9.5

def run(action, stage=None, distributed=False):
    command = [sys.executable]
    if distributed:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    command += [str(runner), action]
    if stage is not None:
        command += ['--stage', str(stage)]
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, check=True)

def archive_after_stage():
    # Archive in its own process, after both GPU workers have exited. A timed
    # out archive leaves the previous verified ZIP intact for recovery.
    command = [sys.executable, str(runner), 'snapshot']
    print('ARCHIVE', ' '.join(command), flush=True)
    try:
        subprocess.run(command, check=True, timeout=900)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        print('Recovery refresh failed; previous ZIP remains intact:', repr(exc), flush=True)

run('prepare')

def ensure_local_features(stage):
    if (root / 'local' / stage / 'done.json').exists() and (root / 'features' / stage / 'done.json').exists():
        print('REUSE trained encoder and extracted features:', stage, flush=True)
        return
    run('local', stage, True)
    run('extract', stage, True)
    archive_after_stage()

def ensure_global(stage):
    if (root / 'global' / str(stage) / 'done.json').exists():
        print('REUSE trained section tracker:', stage, flush=True)
        return
    run('global', stage, True)
    archive_after_stage()

# A new model is fitted and packaged first. Completed stages are reused.
for stage in ('final_inner_0', 'final_inner_1', 'full'):
    ensure_local_features(stage)
ensure_global('final')
if not (root / 'test_results.pt').exists():
    run('infer-test', distributed=True)
else:
    print('REUSE saved test inference', flush=True)
run('select')
run('package')

# Add untouched document-fold evidence while the session budget permits.
for fold in range(5):
    elapsed = (time.time() - session_started) / 3600
    if elapsed > TIME_BUDGET_HOURS - 1.0:
        print('Stopping before fold', fold, 'to preserve packaging time.', flush=True)
        break
    for stage in (f'inner_{fold}_0', f'inner_{fold}_1', f'outer_{fold}'):
        ensure_local_features(stage)
    ensure_global(str(fold))
    run('select')
    if fold == 0:
        selection = json.loads(Path('/kaggle/working/CS781_Capstone/phase3_sections/selection.json').read_text())
        candidate_f1 = {name: row['macro_supported'] for name, row in selection['oof'].items()}
        best_pilot = max(candidate_f1.values())
        pilot_pass = best_pilot >= 0.50
        gate = {'best_fold0_supported_macro_f1': best_pilot, 'candidate_f1': candidate_f1,
                'continue_at_or_above': 0.50, 'continue_outer_folds': pilot_pass}
        Path('/kaggle/working/CS781_Capstone/phase3_sections/pilot_gate.json').write_text(
            json.dumps(gate, indent=2) + '\\n')
        print('FOLD 0 PILOT', json.dumps(gate), flush=True)
    run('package')
    if fold == 0 and not pilot_pass:
        print('Stopping remaining folds: the section pilot did not show a meaningful held-out gain.', flush=True)
        break

report = json.loads(Path('/kaggle/working/CS781_Capstone/phase3_sections/report.json').read_text())
print(json.dumps(report, indent=2), flush=True)
'''),
        nbf.v4.new_markdown_cell('## Verify the submission and portable recovery archive'),
        nbf.v4.new_code_cell('''from pathlib import Path
import hashlib, json, zipfile

submission = Path('/kaggle/working/CS781_Phase3_Sections_Submission.zip')
recovery = Path('/kaggle/working/CS781_Phase3_Sections_All_Artifacts.zip')
with zipfile.ZipFile(submission) as z:
    assert z.namelist() == ['prediction.json']
    predictions = json.loads(z.read('prediction.json'))
assert len(predictions) == 10 and sum(map(len, predictions.values())) == 2197
with zipfile.ZipFile(recovery) as z:
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
print('Submission and full recovery ZIP verified.')
print(json.dumps(json.loads(Path('/kaggle/working/CS781_Capstone/phase3_sections/report.json').read_text()), indent=2))
'''),
    ]
    nbf.validate(nb)
    OUTPUT.write_text(nbf.writes(nb), encoding='utf8')
    print(OUTPUT)


if __name__ == '__main__':
    main()
