"""Build one self-contained Kaggle Run All notebook for Phase 6."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "CS781_Phase6_Six_Submissions_RunAll.ipynb"


def embedded(source, target):
    return nbf.v4.new_code_cell(
        f"%%writefile /kaggle/working/{target}\n" +
        (HERE / source).read_text(encoding="utf8"))


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                  "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell("""# CS781 Phase 6: six summarization submissions

Attach the four original Phase 1 recovery ZIPs as Kaggle Inputs. Select the two-GPU accelerator. Enable Internet and the Kaggle Secret HF_TOKEN for the gated official dataset, or attach the two official Phase 6 train.json/test.json files. Then choose Run All.

The notebook embeds every source file. Its six distinct methods are base-02 LoRA, base-03 AdaLoRA, base-04 LORMA, base-01 Houlsby adapters, base-02 Pfeiffer adapters, and base-03 distillation from the trained base-02 teacher. All start from the original Phase 1 weights. It uses the same reference-free evidence selector at training and inference, group-separated validation, and the course BERTScore F1 metric.

Every two-GPU training process stops after about 20 minutes or an epoch and saves an atomic checkpoint at most about 10 minutes apart. After each process exits, the parent rebuilds CS781_Phase6_All_Artifacts.zip. Teacher extraction and test generation are also segmented. Save/download that recovery ZIP or commit the Kaggle output, attach it in another session, and Run All again. Kaggle may not fit all six methods in a single session. Files left only in a failed draft session may be lost.

Each finished method produces CS781_Phase6_Submission_XX.zip with root-level prediction.json. These files require manual Codabench upload. Private BERTScore F1 is unknown until scoring; no result is guaranteed. The final course report should disclose AI assistance truthfully."""),
        nbf.v4.new_code_cell("""import importlib.util, subprocess, sys
missing = []
for module, package in [('tokenizers','tokenizers'),
                        ('huggingface_hub','huggingface_hub>=0.24,<2'),
                        ('rouge_score','rouge-score==0.1.2'),
                        ('bert_score','bert-score==0.3.13')]:
    if importlib.util.find_spec(module) is None:
        missing.append(package)
if missing:
    subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)
print('Dependencies ready')
"""),
        nbf.v4.new_markdown_cell("## Embedded checkpoint and archive utilities"),
        embedded("utils.py", "cs781_phase6_utils.py"),
        nbf.v4.new_markdown_cell("## Embedded Phase 6 runner"),
        embedded("summ_runner.py", "cs781_phase6_summ_runner.py"),
        nbf.v4.new_markdown_cell("## Train, evaluate, package, and snapshot all six methods"),
        nbf.v4.new_code_cell("""from pathlib import Path
import json, subprocess, sys, threading, time
sys.path.insert(0, '/kaggle/working')
RUNNER = Path('/kaggle/working/cs781_phase6_summ_runner.py')
ROOT = Path('/kaggle/working/CS781_Capstone/phase6_summ')
ARTIFACT = Path('/kaggle/working/CS781_Phase6_All_Artifacts.zip')
SLOTS = ('01','02','03','04','05','06')
BASES = {'01':'02','02':'03','03':'04','04':'01','05':'02','06':'03'}
STARTED = time.monotonic()
SESSION_HOURS = 8.7
SEGMENT_MINUTES = 20

def elapsed_hours():
    return (time.monotonic()-STARTED)/3600

def run(action, *, slot=None, base=None, stage=None, split=None, size=None,
        max_minutes=None, distributed=False):
    command = [sys.executable]
    if distributed:
        command += ['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    command += [str(RUNNER),action]
    for flag,value in [('slot',slot),('base',base),('stage',stage),('split',split),
                       ('size',size),('max-minutes',max_minutes)]:
        if value is not None:
            command += ['--'+flag,str(value)]
    print('RUN',action,'slot',slot,'stage',stage,'elapsed hours',
          round(elapsed_hours(),2),flush=True)
    stop = threading.Event()
    monitor = None
    if distributed:
        def sample():
            path = ROOT/'gpu_samples.jsonl'
            while not stop.is_set():
                try:
                    result = subprocess.run(['nvidia-smi',
                        '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,
                        text=True,check=True,timeout=10)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('a',encoding='utf8') as out:
                        out.write(json.dumps({'time':time.time(),'slot':slot,
                            'stage':stage,'gpus':result.stdout.strip().splitlines()})+'\\n')
                except Exception as exc:
                    print('GPU monitor:',str(exc)[:100],flush=True)
                stop.wait(30)
        monitor = threading.Thread(target=sample,daemon=True)
        monitor.start()
    try:
        subprocess.run(command,check=True)
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=15)

def archive():
    # No torchrun worker is alive here.
    run('snapshot')
    assert ARTIFACT.is_file() and ARTIFACT.stat().st_size > 0
    print('RECOVERY READY',ARTIFACT,
          round(ARTIFACT.stat().st_size/2**30,2),'GiB',flush=True)

def status(slot,stage='fit'):
    path = ROOT/'runs'/slot/stage/'status.json'
    return json.loads(path.read_text()) if path.is_file() else {}

def stopping():
    return elapsed_hours() >= SESSION_HOURS

def ensure_fit(slot):
    while not status(slot).get('complete'):
        if stopping():
            archive()
            return False
        if status(slot).get('pending_eval'):
            run('eval',slot=slot)
            archive()
        else:
            try:
                run('train',slot=slot,stage='fit',
                    max_minutes=SEGMENT_MINUTES,distributed=True)
            finally:
                archive()
    return True

def ensure_teacher():
    import cs781_phase6_summ_runner as runner
    target = min(runner.TEACHER_SIZE,len(runner.stage_rows('fit')))
    path = ROOT/'teacher'/'summaries.json'
    while not (path.is_file() and len(json.loads(path.read_text())) == target):
        if stopping():
            archive()
            return False
        try:
            run('teacher',max_minutes=SEGMENT_MINUTES)
        finally:
            archive()
    return True

def ensure_test(slot):
    import cs781_phase6_summ_runner as runner
    expected = len(runner.data()[1])
    stage = 'final' if status(slot,'final').get('complete') else 'fit'
    path = ROOT/'submissions'/slot/f'partial_{stage}.json'
    while not (path.is_file() and len(json.loads(path.read_text())) == expected):
        if stopping():
            archive()
            return False
        try:
            run('generate',slot=slot,max_minutes=SEGMENT_MINUTES)
        finally:
            archive()
    run('package',slot=slot)
    archive()
    return True

def ensure_score(slot,split):
    path = ROOT/'scores'/f'slot_{slot}_fit_{split}_all.json'
    while not path.is_file():
        if stopping():
            archive()
            return False
        try:
            run('score',slot=slot,split=split,max_minutes=SEGMENT_MINUTES)
        finally:
            archive()
    return True

run('prepare')
archive()
stopped = False
for slot in SLOTS:
    cache_ready = (ROOT/'token_cache'/f'base_{BASES[slot]}.pt').is_file()
    profile_ready = (ROOT/'performance'/f'slot_{slot}.json').is_file()
    run('tokenize',base=BASES[slot])
    run('autotune',slot=slot)
    if not (cache_ready and profile_ready):
        archive()
    if slot == '06' and not ensure_teacher():
        stopped = True
        break
    if not ensure_fit(slot):
        stopped = True
        break
    if not ensure_test(slot):
        stopped = True
        break
    report = json.loads((ROOT/'reports'/f'submission_{slot}.json').read_text())
    print('READY',slot,report['method'],'sample BERTScore F1',
          round(report['fit_best_bertscore_sample'],4),flush=True)
if all((ROOT/'reports'/f'submission_{slot}.json').is_file() for slot in SLOTS):
    ranked = sorted(SLOTS,key=lambda s:json.loads(
        (ROOT/'reports'/f'submission_{s}.json').read_text())[
            'fit_best_bertscore_sample'],reverse=True)
    for candidate in ranked[:2]:
        if not ensure_score(candidate,'tune'):
            stopped = True
            break
    if not stopped:
        winner = max(ranked[:2],key=lambda s:json.loads(
            (ROOT/'scores'/f'slot_{s}_fit_tune_all.json').read_text())[
                'metric']['bertscore_f1'])
        if not ensure_score(winner,'audit'):
            stopped = True
        else:
            print('BEST VALIDATED CANDIDATE',winner,flush=True)
    run('report')
    archive()
print('RUN STATUS','SAVE ARCHIVE AND RESUME' if stopped else
      'SIX PACKAGES READY',flush=True)
"""),
        nbf.v4.new_markdown_cell("## Verify and download completed files"),
        nbf.v4.new_code_cell("""from pathlib import Path
import hashlib, json, sys, zipfile
sys.path.insert(0, '/kaggle/working')
from IPython.display import display, FileLink
import cs781_phase6_summ_runner as runner
root = Path('/kaggle/working/CS781_Capstone/phase6_summ')
artifact = Path('/kaggle/working/CS781_Phase6_All_Artifacts.zip')
expected = [row['source_id'] for row in runner.data()[1]]
packages = sorted(Path('/kaggle/working').glob('CS781_Phase6_Submission_*.zip'))
hashes = []
for path in packages:
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ['prediction.json']
        rows = json.loads(archive.read('prediction.json'))
    assert [row['id'] for row in rows] == expected
    assert all(type(row['summary']) is str and row['summary'].strip()
               for row in rows)
    hashes.append(hashlib.sha256(json.dumps(rows,ensure_ascii=False,
        sort_keys=True).encode()).hexdigest())
    print('VERIFIED',path.name,len(rows),'summaries')
    display(FileLink(str(path)))
assert len(hashes) == len(set(hashes)), 'Two counted submissions are identical'
with zipfile.ZipFile(artifact) as archive:
    manifest = json.loads(archive.read('integrity.json'))['files']
    assert set(manifest) == set(archive.namelist()) - {'integrity.json'}
    for name, meta in manifest.items():
        digest = hashlib.sha256()
        size = 0
        with archive.open(name) as member:
            for block in iter(lambda: member.read(4<<20),b''):
                digest.update(block); size += len(block)
        assert digest.hexdigest() == meta['sha256'] and size == meta['bytes'], name
print('VERIFIED RECOVERY',artifact.name,'packages',len(packages))
display(FileLink(str(artifact)))
if (root/'report.json').is_file():
    print('FINAL STATUS',json.loads((root/'report.json').read_text())['status'])
"""),
    ]
    nbf.validate(nb)
    OUTPUT.write_text(nbf.writes(nb), encoding="utf8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
