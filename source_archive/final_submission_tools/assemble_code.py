"""Assemble a small, source-only release without altering the working experiments."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'final_submission'
CODE = BASE / 'package/code'
SECRET = re.compile(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def clean_notebook(source, target):
    nb = json.loads(source.read_text(encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['execution_count'] = None
            c['outputs'] = []
        c['metadata'] = {}
        c.pop('attachments', None)
    nb['metadata'] = {k: v for k, v in nb.get('metadata', {}).items() if k in ('kernelspec', 'language_info')}
    text = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
    if SECRET.search(text):
        raise RuntimeError(f'Credential-shaped string in source notebook: {source.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding='utf-8-sig')
    if SECRET.search(text):
        raise RuntimeError(f'Credential-shaped string in {source}')
    target.write_text(text, encoding='utf-8')


def main():
    CODE.mkdir(parents=True, exist_ok=True)
    for phase in range(2, 8):
        directory = ROOT / f'phase{phase}'
        for source in directory.rglob('*.py'):
            if '__pycache__' not in source.parts:
                copy(source, CODE / f'phase{phase}' / source.relative_to(directory))
    for name in ('SOURCE.md', 'evaluator_smoke.json'):
        copy(ROOT / 'phase7/reference' / name, CODE / 'phase7/reference' / name)
    phase1 = CODE / 'phase1'
    for number in range(1, 5):
        with zipfile.ZipFile(ROOT / f'phase1_submission{number:02d}_recovery.zip') as z:
            for name in z.namelist():
                if name.startswith('model/') and name.endswith('.py'):
                    path = phase1 / f'base{number:02d}' / Path(name).name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(z.read(name))
            # Configuration and numeric training summaries do not contain the legal corpus.
            for name in ('run_config.json', 'training_config.json', 'training_result.json', 'model_validation.json', 'source_sampling_config.json'):
                member = 'manifests/' + name
                if member in z.namelist():
                    path = phase1 / f'base{number:02d}' / 'saved_config' / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(z.read(member))
    names = sorted(ROOT.glob('CS781_Phase*.ipynb'))
    for source in names:
        phase = int(re.search(r'Phase(\d+)', source.name).group(1))
        clean_notebook(source, CODE / f'phase{phase}' / 'notebooks' / source.name)
    for name in ('capstone-project-phase2-notebook.ipynb', 'capstone-project-phase2-notebook (1).ipynb'):
        clean_notebook(ROOT / name, CODE / 'phase2/notebooks' / name)
    # Preserve the precise runtime that produced each selected archive, including older revisions.
    snapshots = [
        (3, 'CS781_Phase3_HighF1_All_Artifacts.zip', ('high_f1_runner.py', 'source_utils.py')),
        (4, 'CS781_Phase4_All_Artifacts (1).zip', ('cjpe_runner.py', 'phase1_utils.py', 'slot06_fast_runner.py')),
        (5, 'CS781_Phase5_All_Artifacts.zip', ('lsi_runner.py', 'phase5_utils.py')),
        (6, 'CS781_Phase6_All_Artifacts.zip', ('summ_runner.py', 'utils.py')),
        (7, 'CS781_Phase7_All_Artifacts.zip', ('pcr_runner.py', 'utils.py', 'evaluate_phase7.py')),
    ]
    provenance = []
    for phase, archive, members in snapshots:
        with zipfile.ZipFile(ROOT / archive) as z:
            for name in members:
                if name not in z.namelist():
                    continue
                content = z.read(name)
                target = CODE / f'phase{phase}/archived_run' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                provenance.append({'path': target.relative_to(CODE).as_posix(), 'source_archive': archive,
                                   'source_member': name, 'sha256': hashlib.sha256(content).hexdigest()})
    # The snapshot notebooks embed the saved parsing fixes as well as the saved model code.
    for phase, archive, bindings in [
        (4, 'CS781_Phase4_All_Artifacts (1).zip', {'cs781_phase4_utils.py': 'phase1_utils.py', 'cs781_phase4_cjpe_runner.py': 'cjpe_runner.py'}),
        (5, 'CS781_Phase5_All_Artifacts.zip', {'cs781_phase5_utils.py': 'phase5_utils.py', 'cs781_phase5_lsi_runner.py': 'lsi_runner.py'}),
    ]:
        name = f'CS781_Phase{phase}_Six_Submissions_RunAll.ipynb'
        nb = json.loads((CODE / f'phase{phase}/notebooks' / name).read_text(encoding='utf-8'))
        replaced = set()
        with zipfile.ZipFile(ROOT / archive) as z:
            for cell in nb['cells']:
                source = ''.join(cell['source'])
                if not source.startswith('%%writefile '):
                    continue
                first = source.splitlines()[0]
                basename = Path(first.split()[-1]).name
                if basename in bindings:
                    cell['source'] = first + '\n' + z.read(bindings[basename]).decode('utf-8')
                    replaced.add(basename)
        if replaced != set(bindings):
            raise RuntimeError('Snapshot notebook did not replace every runtime')
        target = CODE / f'phase{phase}/archived_run' / name
        target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (CODE / 'evidence').mkdir(exist_ok=True)
    (CODE / 'evidence/source_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    (CODE / '.gitignore').write_text('''__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
.env
.env.*
*.pt
*.pth
*.pkl
*.bin
*.safetensors
*.zip
data/
official/
checkpoints/
tokenizer/
*.log
''', encoding='utf-8')
    (CODE / 'requirements.txt').write_text('''# Python 3.10+; choose the PyTorch build appropriate for the Kaggle CUDA runtime.
torch>=2.4,<3
numpy>=1.26,<3
scipy>=1.11,<2
scikit-learn>=1.4,<2
tokenizers>=0.20,<1
huggingface-hub>=0.26,<2
bert-score>=0.3.13,<0.4
rouge-score>=0.1.2,<0.2
nbformat>=5.10,<6
psutil>=5.9,<8
''', encoding='utf-8')
    checked = []
    for path in sorted(CODE.rglob('*')):
        if not path.is_file() or '.git' in path.parts:
            continue
        content = path.read_bytes()
        if SECRET.search(content.decode('utf-8', errors='replace')):
            raise RuntimeError(f'Credential scan failed: {path}')
        if path.suffix == '.py':
            ast.parse(content.decode('utf-8-sig'), filename=str(path))
        checked.append({'path': path.relative_to(CODE).as_posix(), 'bytes': len(content),
                        'sha256': hashlib.sha256(content).hexdigest()})
    (BASE / 'review/source_checks.json').write_text(json.dumps({'files': checked, 'python_ast': 'PASS',
        'credential_scan': 'PASS', 'notebook_outputs': 'CLEARED'}, indent=2) + '\n', encoding='utf-8')
    print('Source files:', len(checked), 'bytes:', sum(x['bytes'] for x in checked))
    print('Python syntax and credential scan passed; all notebook outputs cleared.')


if __name__ == '__main__':
    main()
