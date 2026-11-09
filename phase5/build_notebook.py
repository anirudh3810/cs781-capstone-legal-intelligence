"""Build one self-contained Kaggle Run All notebook from reviewed Phase 5 sources."""
from pathlib import Path
import nbformat as nbf

HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parent/'CS781_Phase5_Six_Submissions_RunAll.ipynb'


def embedded(source,target):
    content=(HERE/source).read_text(encoding='utf8')
    return nbf.v4.new_code_cell(f'%%writefile /kaggle/working/{target}\n{content}')


def main():
    nb=nbf.v4.new_notebook()
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[
        nbf.v4.new_markdown_cell('''# CS781 Phase 5: six statute-identification submissions

Attach the four **original Phase 1 recovery** archives as Kaggle Inputs, enable the **two-GPU** accelerator and Internet, and enable Kaggle Secret `HF_TOKEN` for the official gated course dataset. If all three official Phase 5 JSON files are attached as inputs, the notebook can run offline. Choose **Run All**.

The six genuinely different methods are: (01) base-03 partial fine-tuning, (02) base-01 full fine-tuning, (03) base-04 LORMA, (04) base-02 bottleneck adapters, (05) base-03 AdaLoRA, and (06) base-03 distillation from the matched slot-02 teacher. Every model starts from the student's Phase 1 checkpoint. The multi-label head reads selected beginning, ending, spaced, and citation-dense chunks and predicts one statute-ID set per case.

**Recovery:** each two-GPU training command stops after a complete epoch or about 20 minutes at a synchronized optimizer step. Distillation teacher-logit extraction also resumes in roughly 20-minute segments. The next notebook action atomically rebuilds `CS781_Phase5_All_Artifacts.zip`, containing official data, Phase 1 bases, selected/latest checkpoints, teacher logits, metrics, code, and finished submission files. The ZIP is rebuilt after every training command and after every package. Near Kaggle's session limit the notebook stops at a safe boundary. Save/download the latest archive or commit it as a Kaggle output; attach it as a new Kaggle Input in a later session and Run All again. Files left solely in an expired/fatal `/kaggle/working` session cannot be guaranteed to survive.

Each method first trains on a grouped fitting split and scores an untouched grouped holdout, then trains on all official training rows with the official dev set for checkpoint/threshold selection. Six final ZIPs are produced with root-level `prediction.json`. The notebook validates IDs and integer statute lists. Local dev results after threshold tuning are selection-biased; private Codabench F1 is unknown until manual upload. Runtime and GPU utilization depend on the released Phase 5 dataset; the notebook logs throughput and memory rather than claiming a guaranteed score.'''),
        nbf.v4.new_markdown_cell('## Embedded Phase 1 loader and atomic artifact handling'),
        embedded('utils.py','cs781_phase5_utils.py'),
        nbf.v4.new_markdown_cell('## Embedded Phase 5 model, validation, and training runner'),
        embedded('lsi_runner.py','cs781_phase5_lsi_runner.py'),
        nbf.v4.new_markdown_cell('## Run all six experiments and rebuild recovery after each training segment'),
        nbf.v4.new_code_cell('''from pathlib import Path
import json, subprocess, sys, threading, time

RUNNER=Path('/kaggle/working/cs781_phase5_lsi_runner.py')
ROOT=Path('/kaggle/working/CS781_Capstone/phase5_lsi')
ARTIFACT=Path('/kaggle/working/CS781_Phase5_All_Artifacts.zip')
SLOTS=('01','02','03','04','05','06')
BASES={'01':'03','02':'01','03':'04','04':'02','05':'03','06':'03'}
STARTED=time.monotonic()
SESSION_HOURS=9.2

def elapsed_hours():return (time.monotonic()-STARTED)/3600

def run(action, *, slot=None, stage=None, base=None, distributed=False):
    command=[sys.executable]
    if distributed:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    command+=[str(RUNNER),action]
    if slot is not None:command+=['--slot',slot]
    if stage is not None:command+=['--stage',stage]
    if base is not None:command+=['--base',base]
    if distributed:command+=['--max-minutes','20']
    print('RUN',action,'slot',slot,'stage',stage,'elapsed hours',round(elapsed_hours(),2),flush=True)
    stop=threading.Event()
    monitor=None
    if distributed:
        def sample_gpus():
            target=ROOT/'gpu_samples.jsonl'
            while not stop.is_set():
                try:
                    probe=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('a',encoding='utf8') as handle:
                        handle.write(json.dumps({'time':time.time(),'slot':slot,'stage':stage,
                            'gpus':probe.stdout.strip().splitlines()})+'\\n')
                except Exception as exc:print('GPU sample unavailable',str(exc)[:120],flush=True)
                stop.wait(30)
        monitor=threading.Thread(target=sample_gpus,daemon=True)
        monitor.start()
    try:subprocess.run(command,check=True)
    finally:
        stop.set()
        if monitor is not None:monitor.join(timeout=15)

def archive():
    # All torchrun workers have exited before archive construction.
    run('snapshot')
    assert ARTIFACT.is_file() and ARTIFACT.stat().st_size>0
    print('DOWNLOADABLE RECOVERY',ARTIFACT,round(ARTIFACT.stat().st_size/2**30,2),'GiB',flush=True)

def complete(slot,stage):
    path=ROOT/'runs'/slot/stage/'status.json'
    return path.is_file() and json.loads(path.read_text())['complete']

def ensure_stage(slot,stage):
    while not complete(slot,stage):
        if elapsed_hours()>=SESSION_HOURS:
            archive()
            print('Session reserve reached. Save the all-artifacts ZIP, attach it next session, and Run All again.',flush=True)
            return False
        try:run('train',slot=slot,stage=stage,distributed=True)
        finally:archive()  # preserve the last committed checkpoint even if torchrun fails
    return True

def ensure_teacher(stage):
    path=ROOT/'teacher'/stage/'status.json'
    while not (path.is_file() and json.loads(path.read_text())['complete']):
        if elapsed_hours()>=SESSION_HOURS:
            archive()
            print('Session reserve reached during teacher extraction. Save the archive and resume.',flush=True)
            return False
        try:run('teacher',stage=stage,distributed=True)
        finally:archive()
    return True

run('prepare')
archive()
stopped=False
for slot in SLOTS:
    run('tokenize',base=BASES[slot])
    if slot=='06':run('tokenize',base='01')
    for stage in ('holdout','final'):
        if slot=='06' and not ensure_teacher(stage):
            stopped=True
            break
        if not ensure_stage(slot,stage):
            stopped=True
            break
    if stopped:break
    run('package',slot=slot)
    archive()
    report=json.loads((ROOT/'reports'/f'submission_{slot}.json').read_text())
    print('SLOT',slot,'HOLDOUT MACRO F1',round(report['holdout_metric']['macro_f1'],4),
          'DEV SELECTED MACRO F1',round(report['dev_metric_selected']['macro_f1'],4),flush=True)

if all((ROOT/'reports'/f'submission_{slot}.json').is_file() for slot in SLOTS):
    run('report')
archive()
print('STATUS','SAFE RESUME REQUIRED' if stopped else 'SIX METHODS FINISHED',flush=True)
'''),
        nbf.v4.new_markdown_cell('## Verify completed packages and download links'),
        nbf.v4.new_code_cell('''from pathlib import Path
import hashlib,json,sys,zipfile
from IPython.display import display,FileLink
sys.path.insert(0,'/kaggle/working')
import cs781_phase5_lsi_runner as runner

root=Path('/kaggle/working/CS781_Capstone/phase5_lsi')
artifact=Path('/kaggle/working/CS781_Phase5_All_Artifacts.zip')
test=runner.data()[2]
labels=set(json.loads((root/'run_config.json').read_text())['labels'])
packages=sorted(Path('/kaggle/working').glob('CS781_Phase5_Submission_*.zip'))
for path in packages:
    with zipfile.ZipFile(path) as z:
        assert z.namelist()==['prediction.json']
        rows=json.loads(z.read('prediction.json'))
    assert [row['id'] for row in rows]==[row['source_id'] for row in test]
    assert all(isinstance(row['pred'],list) and len(row['pred'])==len(set(row['pred'])) and
               all(type(value) is int and value in labels for value in row['pred']) for row in rows)
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
if (root/'report.json').is_file():print('FINAL STATUS',json.loads((root/'report.json').read_text())['status'])
'''),
    ]
    nbf.validate(nb)
    OUTPUT.write_text(nbf.writes(nb),encoding='utf8')
    print(OUTPUT)


if __name__=='__main__':main()
