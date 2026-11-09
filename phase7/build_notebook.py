"""Build one self-contained Kaggle notebook for the six retrieval methods."""
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parent/'CS781_Phase7_Six_Submissions_RunAll.ipynb'

def embedded(source,target):
    return nbf.v4.new_code_cell(f'%%writefile /kaggle/working/{target}\n'+(HERE/source).read_text(encoding='utf8'))

ORCHESTRATE=r'''from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib,json,os,subprocess,sys,threading,time
sys.path.insert(0,'/kaggle/working')
import cs781_phase7_utils as utils
import cs781_phase7_pcr_runner as runner
importlib.reload(utils);importlib.reload(runner)
ROOT=Path('/kaggle/working/CS781_Capstone/phase7_pcr')
RUNNER=Path('/kaggle/working/cs781_phase7_pcr_runner.py')
ARTIFACT=Path('/kaggle/working/CS781_Phase7_All_Artifacts.zip')
SEGMENT_MINUTES=18
SESSION_HOURS=8.5
active_processes={}
process_lock=threading.Lock()
try:
    identity=Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+Path('/proc/1/stat').read_text().split()[21]
except Exception: identity=str(os.getpid())
session_path=Path('/kaggle/working/phase7_backend_session.json')
session=json.loads(session_path.read_text()) if session_path.is_file() else {}
if session.get('backend')!=identity:
    session={'backend':identity,'started':time.time()}
    session_path.write_text(json.dumps(session))
def expired(): return time.time()-session['started']>=SESSION_HOURS*3600

def run(action,*,gpu=None,slot=None,base=None,split=None,stage=None):
    cmd=[sys.executable,'-u',str(RUNNER),action,'--max-minutes',str(SEGMENT_MINUTES)]
    for flag,value in [('slot',slot),('base',base),('split',split),('stage',stage)]:
        if value is not None:cmd+=['--'+flag,str(value)]
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    tag=slot or action
    print('START',action,'slot',slot,'GPU',gpu,flush=True)
    logdir=ROOT/'logs';logdir.mkdir(parents=True,exist_ok=True)
    logfile=logdir/(str(time.time_ns())+'_'+str(tag)+'_'+action+'.log')
    with logfile.open('w',encoding='utf8') as out:
        process=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        with process_lock:active_processes[process.pid]=process
        try:
            for line in process.stdout:
                out.write(line);out.flush();print('['+str(tag)+']',line.rstrip(),flush=True)
            code=process.wait()
            if code:raise RuntimeError(f'{action} slot {slot} failed ({code}); inspect {logfile}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        finally:
            with process_lock:active_processes.pop(process.pid,None)
def archive():
    run('report');run('snapshot')
    assert ARTIFACT.is_file() and ARTIFACT.stat().st_size>0
    print('RECOVERY READY',ARTIFACT,round(ARTIFACT.stat().st_size/2**30,2),'GiB',flush=True)
def done(job):
    action,slot,stage=job
    if action=='teacher':return (ROOT/'teacher/scores.json').is_file()
    if action=='advance' and slot=='control':
        p=ROOT/'runs/control/status.json'
        return p.is_file() and json.loads(p.read_text()).get('complete',False)
    if action=='advance':return (ROOT/'reports'/f'submission_{slot}.json').is_file()
    if action=='score':return (ROOT/'scores'/f'{slot}_{stage}.json').is_file()
    raise ValueError(job)
def drive(jobs):
    while True:
        pending=[j for j in jobs if not done(j)]
        if not pending:return True
        if expired():archive();return False
        stop=threading.Event()
        def monitor():
            while not stop.is_set():
                try:
                    result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
                    with (ROOT/'gpu_samples.jsonl').open('a',encoding='utf8') as f:
                        f.write(json.dumps({'time':time.time(),'jobs':pending[:2],'gpus':result.stdout.splitlines()})+'\n')
                except Exception as exc:print('GPU monitor:',str(exc)[:100],flush=True)
                stop.wait(30)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        errors=[]
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures=[executor.submit(run,j[0],gpu=gpu,slot=j[1],stage=j[2]) for gpu,j in enumerate(pending[:2])]
                try:
                    for future in futures:
                        try:future.result()
                        except Exception as exc:errors.append(str(exc))
                except BaseException:
                    with process_lock:children=list(active_processes.values())
                    for child in children:
                        if child.poll() is None:child.terminate()
                    for child in children:
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
        finally:
            stop.set();thread.join(timeout=15)
            # Both workers have exited before taking a consistent archive.
            archive()
        if errors:raise RuntimeError('\n'.join(errors))

run('prepare');archive()
stopped=False
for split in ('train','dev','test'):
    if expired():stopped=True;break
    run('lexical',split=split)
archive()
if not stopped:
    for base in ('01','02','03','04'):
        if expired():stopped=True;break
        run('tokenize',base=base);archive()
groups=[
    [('advance','01',None),('advance','02',None)],
    [('advance','03',None),('advance','04',None)],
    [('teacher',None,None),('advance','control',None)],
    [('advance','05',None),('advance','06',None)]]
if not stopped:
    for jobs in groups:
        if not drive(jobs):stopped=True;break
if not stopped:
    ranked=sorted(('01','02','03','04','05','06'),key=lambda s:json.loads(
        (ROOT/'reports'/f'submission_{s}.json').read_text())['tune_sample_metric']['tp'],reverse=True)
    if not drive([('score',s,'tune') for s in ranked[:2]]):stopped=True
    if not stopped:
        winner=max(ranked[:2],key=lambda s:(json.loads((ROOT/'scores'/f'{s}_tune.json').read_text())['metric']['tp'],-int(s)))
        if not drive([('score',winner,'audit')]):stopped=True
archive()
print('RUN STATUS:','SAVE ARCHIVE AND RESUME IN ANOTHER SESSION' if stopped else
      'SIX METHODS FINISHED; REVIEW FINAL REPORT',flush=True)
'''

VERIFY=r'''from pathlib import Path
import hashlib,json,zipfile,sys
from IPython.display import display,FileLink
sys.path.insert(0,'/kaggle/working')
import cs781_phase7_pcr_runner as runner
import cs781_phase7_evaluator as official
root=Path('/kaggle/working/CS781_Capstone/phase7_pcr')
queries,candidates=runner.data('test')
packages=sorted(Path('/kaggle/working').glob('CS781_Phase7_Submission_*.zip'))
sets=[]
for path in packages:
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist()==['prediction.json']
        pred=json.loads(archive.read('prediction.json'),object_pairs_hook=official.pairs_no_duplicates,
                        parse_constant=official.reject_constant)
    errors=official.validate(pred,{k:[] for k in queries},set(candidates))
    assert not errors,errors[:10]
    sets.append(hashlib.sha256(json.dumps({k:sorted(v) for k,v in pred.items()},sort_keys=True).encode()).hexdigest())
    print('VERIFIED',path.name,len(pred),'queries; seven unique candidates each')
    display(FileLink(str(path)))
artifact=Path('/kaggle/working/CS781_Phase7_All_Artifacts.zip')
with zipfile.ZipFile(artifact) as archive:
    manifest=json.loads(archive.read('integrity.json'))['files']
    assert set(archive.namelist())==set(manifest)|{'integrity.json'}
    for name,meta in manifest.items():
        digest=hashlib.sha256();size=0
        with archive.open(name) as member:
            for block in iter(lambda:member.read(4<<20),b''):
                digest.update(block);size+=len(block)
        assert digest.hexdigest()==meta['sha256'] and size==meta['bytes'],name
print('VERIFIED RECOVERY',artifact.name)
display(FileLink(str(artifact)))
if len(sets)!=len(set(sets)):
    print('REVIEW: identical top-seven sets found; rank-only changes are not distinct results.')
if (root/'report.json').is_file():print(json.dumps(json.loads((root/'report.json').read_text()),indent=2))
'''

def main():
    nb=nbf.v4.new_notebook()
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[
        nbf.v4.new_markdown_cell('''# CS781 Phase 7: six resumable retrieval submissions

Attach all four ORIGINAL Phase 1 recovery datasets, as extracted folders or ZIPs. Select the two-GPU accelerator. Enable Internet and allow this notebook to access the Kaggle Secret **HF_TOKEN** for the official gated course dataset; alternatively attach all six official Phase 7 query/candidate JSON files. Run All.

Methods: base-03 full tuning, base-01 partial tuning, base-04 LORMA, base-02 encoder IA³, base-03 prompt tuning, and base-02 encoder retrieval distillation. A two-pass gold-only student control is also run. Every method starts from original Phase 1 weights. Search combines full-document/passage BM25 and trained document/chunk embeddings; evaluation uses the exact official Micro-F1@7.

Two independent jobs use GPU 0 and GPU 1 concurrently. Jobs probe their batch size and use mixed precision. Model/optimizer/scaler/RNG/query-cursor checkpoints save about every five minutes and at every segment boundary. The parent rebuilds **CS781_Phase7_All_Artifacts.zip** after both workers exit. Embedding and ranking stages also resume.

Save/download the archive before ending Kaggle. A soft backend budget stops with partial outputs if needed; attach the archive in another session and Run All the SAME notebook to finish. Rerunning a cell does not extend the backend lifetime. Files left only in a failed draft session may be lost.

Each completed method produces **CS781_Phase7_Submission_XX.zip**, containing only root-level prediction.json. Upload manually to Codabench. Actual private F1 is unknown until evaluation. Disclose AI assistance accurately in the course report.'''),
        nbf.v4.new_code_cell('''import importlib.util,subprocess,sys
missing=[]
for module,package in [('numpy','numpy'),('scipy','scipy'),('sklearn','scikit-learn'),
                       ('tokenizers','tokenizers'),('huggingface_hub','huggingface_hub>=0.24,<2')]:
    if importlib.util.find_spec(module) is None:missing.append(package)
if missing:subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)
import torch
assert torch.cuda.device_count()==2,'Choose Kaggle two-GPU accelerator'
print('GPUs:',[torch.cuda.get_device_name(i) for i in range(2)])
'''),
        nbf.v4.new_markdown_cell('## Embedded recovery utilities and official evaluator'),
        embedded('utils.py','cs781_phase7_utils.py'),
        embedded('reference/evaluate_phase7.py','cs781_phase7_evaluator.py'),
        nbf.v4.new_markdown_cell('## Embedded training and retrieval implementation'),
        embedded('pcr_runner.py','cs781_phase7_pcr_runner.py'),
        nbf.v4.new_markdown_cell('## Train, evaluate, package and snapshot'),
        nbf.v4.new_code_cell(ORCHESTRATE),
        nbf.v4.new_markdown_cell('## Verify and download completed files'),
        nbf.v4.new_code_cell(VERIFY)]
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code' and not cell.source.startswith('%%'):
            compile(cell.source,'notebook cell','exec')
    OUTPUT.write_text(nbf.writes(nb),encoding='utf8');print(OUTPUT)

if __name__=='__main__':main()
