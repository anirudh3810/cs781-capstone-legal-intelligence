"""Embed all Phase-4 source and create a portable Kaggle Run All notebook."""
from pathlib import Path

import nbformat as nbf


HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parent/'CS781_Phase4_Six_Submissions_RunAll.ipynb'


def embedded(source,target):
    content=(HERE/source).read_text(encoding='utf8')
    return nbf.v4.new_code_cell(f'%%writefile /kaggle/working/{target}\n{content}')


def main():
    notebook=nbf.v4.new_notebook()
    notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    notebook.cells=[
        nbf.v4.new_markdown_cell('''# CS781 Phase 4: six court-judgment submissions

Attach the four **Phase 1 recovery** Kaggle Inputs (`phase1_submission01_recovery` through `phase1_submission04_recovery`), enable a **two-GPU** accelerator and Internet, and enable a Kaggle Secret named exactly **HF_TOKEN** with access to the official course Hugging Face dataset. Choose **Run All**. The notebook fetches `Phase_4_CJPE/Train/train.json`, `Train/dev.json`, and `Test/test.json` from one pinned dataset revision. It refuses an incomplete or changed schema. Phase 4 is scheduled to start on 17 October 2026 in the local specification; if the gated files are not released when you run, it stops with a clear error before training.

**Input-loader repair:** this build identifies each Phase 1 recovery by its actual model, tokenizer, and checkpoint files. It accepts an extracted Kaggle dataset or a ZIP even when Kaggle has changed its mount or archive name. If an input is truly absent, the error lists mounted inputs and matching files. Upload this revised notebook as a new Kaggle notebook so an old saved execution does not keep running the earlier loader.

**Repeated-case-ID repair:** the released `train.json` has repeated case IDs. Training and development rows now receive stable internal row keys, while the original case IDs remain available for grouping. Validation keeps related rows together where those IDs identify a case. Test submission IDs remain exactly as supplied in `test.json`.

The six methods are: (01) full fine-tuning of Phase 1 base 01, (02) partial layer-wise fine-tuning of base 03, (03) LORMA on base 04, (04) adaptive-rank AdaLoRA on base 02, (05) bottleneck adapters on base 03, and (06) distillation from the larger trained base-01 teacher to a base-03 student. All use the original Phase 1 weights. The classifier reads multiple chunks across each long case and predicts one binary outcome per case. No other pretrained model, corpus, or external LLM is loaded.

Each training stage uses both GPUs with one distributed process per GPU, mixed precision, gradient checkpointing, gradient accumulation, atomic checkpoints, and early stopping. The notebook first trains and packages one provisional submission per method. It then runs three **case-level outer folds** for each method. Checkpoints and thresholds are chosen on inner validation cases, while the outer cases remain untouched until scoring. It republishes each package with a threshold from the fold evidence and reports macro F1 for both classes. A weak local F1 is flagged rather than represented as a good private score.

Outputs: six `/kaggle/working/CS781_Phase4_Submission_XX.zip` files, each containing root-level `prediction.json`; and `/kaggle/working/CS781_Phase4_All_Artifacts.zip`, containing official data, Phase 1 sources/weights, model states, optimizer checkpoints for active stages, predictions, histories, reports, and code. To resume in a new session, attach the newest all-artifacts ZIP as an additional Kaggle Input with the four original Phase 1 inputs and Run All again. The artifact is rebuilt **after** distributed workers exit. A fatal Kaggle draft session can still make local files inaccessible; download or persist the recovery ZIP after a completed session.

The job may exceed one Kaggle session depending on the released dataset size and accelerator quota. After about 9.5 hours this notebook stops at a safe stage boundary with a recovery ZIP; the same notebook continues remaining stages on the next Run All. A strong private F1 cannot be guaranteed before Codabench scoring. Review the course's AI-assistance declaration requirements before using the results.'''),
        nbf.v4.new_markdown_cell('## Embedded recovery and Phase 1 model loader'),
        embedded('utils.py','cs781_phase4_utils.py'),
        nbf.v4.new_markdown_cell('## Embedded Phase 4 data, models, training, and submission code'),
        embedded('cjpe_runner.py','cs781_phase4_cjpe_runner.py'),
        nbf.v4.new_markdown_cell('## Run all six methods and case-level validation'),
        nbf.v4.new_code_cell('''from pathlib import Path
import json, subprocess, sys, time

RUNNER=Path('/kaggle/working/cs781_phase4_cjpe_runner.py')
ROOT=Path('/kaggle/working/CS781_Capstone/phase4_cjpe')
ARTIFACT=Path('/kaggle/working/CS781_Phase4_All_Artifacts.zip')
SLOTS=('01','02','03','04','05','06')
BASES={'01':'01','02':'03','03':'04','04':'02','05':'03','06':'03'}
STARTED=time.monotonic()
SESSION_HOURS=9.5
last_archive=0.0

def elapsed_hours():
    return (time.monotonic()-STARTED)/3600

def run(action, *, slot=None, stage=None, base=None, chunk_epochs=None, distributed=False):
    command=[sys.executable]
    if distributed:
        command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    command+=[str(RUNNER),action]
    if slot is not None:command+=['--slot',slot]
    if stage is not None:command+=['--stage',stage]
    if base is not None:command+=['--base',base]
    if chunk_epochs is not None:command+=['--chunk-epochs',str(chunk_epochs)]
    print('RUN',action,'slot',slot,'stage',stage,'elapsed hours',round(elapsed_hours(),2),flush=True)
    subprocess.run(command,check=True)

def archive(force=False):
    global last_archive
    if force or time.monotonic()-last_archive>=1200 or not ARTIFACT.is_file():
        # No DDP process is alive here. The old ZIP remains intact if this fails.
        print('ARCHIVE after distributed workers exit',flush=True)
        subprocess.run([sys.executable,str(RUNNER),'snapshot'],check=True,timeout=1200)
        last_archive=time.monotonic()

def complete(slot,stage):
    path=ROOT/'runs'/slot/stage/'status.json'
    return path.is_file() and json.loads(path.read_text())['complete']

def ensure_stage(slot,stage):
    while not complete(slot,stage):
        if elapsed_hours()>=SESSION_HOURS:
            print('Session time reserve reached. Recovery ZIP is ready; attach it and Run All in a new session.',flush=True)
            archive(force=True)
            return False
        run('train',slot=slot,stage=stage,chunk_epochs=2,distributed=True)
        archive()
    return True

run('prepare')
archive(force=True)
stopped=False

# Fit all six final models first. This creates individually usable packages
# before the more expensive outer-fold audit. Their reports say PROVISIONAL.
for slot in SLOTS:
    run('tokenize',base=BASES[slot])
    if slot=='06':run('tokenize',base='01')
    if not ensure_stage(slot,'final'):
        stopped=True
        break
    run('package',slot=slot)
    archive(force=True)
    report=json.loads((ROOT/'reports'/f'submission_{slot}.json').read_text())
    print('SLOT',slot,'DEV MACRO F1',round(report['dev_macro_f1'],4),
          'STATUS',report['validation_status'],'WARNING',report['quality_warning'],flush=True)

# Outer folds use their own inner validation for epoch and threshold choices.
# Slot 01 folds precede slot 06 so every distillation fold has a fold-safe teacher.
if not stopped:
    for slot in SLOTS:
        for fold in range(3):
            if not ensure_stage(slot,f'fold_{fold}'):
                stopped=True
                break
            run('package',slot=slot)
            archive(force=True)
            report=json.loads((ROOT/'reports'/f'submission_{slot}.json').read_text())
            print('SLOT',slot,'FOLD',fold,'OUTER MACRO F1',
                  None if report['outer_macro_f1'] is None else round(report['outer_macro_f1'],4),
                  'DEV MACRO F1',round(report['dev_macro_f1'],4),flush=True)
        if stopped:break

if all((ROOT/'reports'/f'submission_{slot}.json').is_file() for slot in SLOTS):
    run('report')
archive(force=True)
print('RUN STATUS:', 'SAFE RESUME REQUIRED' if stopped else 'ALL REQUESTED STAGES COMPLETED',flush=True)
print('Recovery archive:',ARTIFACT,flush=True)
'''),
        nbf.v4.new_markdown_cell('## Verify the ZIPs and show download links'),
        nbf.v4.new_code_cell('''from pathlib import Path
import hashlib,json,zipfile
from IPython.display import display,FileLink

root=Path('/kaggle/working/CS781_Capstone/phase4_cjpe')
artifact=Path('/kaggle/working/CS781_Phase4_All_Artifacts.zip')
test=json.loads((root/'official/test.json').read_text())
if isinstance(test,dict):
    test=test.get('data',test.get('examples',test.get('cases',test)))
    if isinstance(test,dict):expected=list(test)
    else:expected=[row.get('id',row.get('case_id')) for row in test]
else:expected=[row.get('id',row.get('case_id')) for row in test]
packages=sorted(Path('/kaggle/working').glob('CS781_Phase4_Submission_*.zip'))
for path in packages:
    with zipfile.ZipFile(path) as z:
        assert z.namelist()==['prediction.json']
        rows=json.loads(z.read('prediction.json'))
    assert isinstance(rows,list) and [row['id'] for row in rows]==expected
    assert len(rows)==len(expected) and all(type(row['pred']) is int and row['pred'] in (0,1) for row in rows)
    print('VERIFIED',path.name,len(rows),'cases')
    display(FileLink(str(path)))
with zipfile.ZipFile(artifact) as z:
    manifest=json.loads(z.read('integrity.json'))['files']
    assert set(manifest)==set(z.namelist())-{'integrity.json'}
    for name,metadata in manifest.items():
        digest=hashlib.sha256();size=0
        with z.open(name) as member:
            for block in iter(lambda:member.read(4<<20),b''):
                digest.update(block);size+=len(block)
        assert digest.hexdigest()==metadata['sha256'] and size==metadata['bytes'],name
print('VERIFIED RECOVERY',artifact.name,'packages',len(packages))
display(FileLink(str(artifact)))
if (root/'report.json').is_file():
    summary=json.loads((root/'report.json').read_text())
    print('FINAL STATUS',summary['status'])
    for row in summary['slots']:
        print(row['slot'],row['method'],'outer F1',row['outer_macro_f1'],
              'dev F1',row['dev_macro_f1'],'warning',row['quality_warning'])
'''),
    ]
    nbf.validate(notebook)
    OUTPUT.write_text(nbf.writes(notebook),encoding='utf8')
    print(OUTPUT)


if __name__=='__main__':main()
