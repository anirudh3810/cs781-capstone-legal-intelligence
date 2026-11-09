"""Read-only structural checks; does not run training or import CUDA modules."""
from pathlib import Path
import ast
import json
import re

ROOT = Path(__file__).resolve().parent.parent
patterns = re.compile(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
count = {'python': 0, 'notebooks': 0, 'files': 0}
for p in ROOT.rglob('*'):
    if not p.is_file() or any(q in p.parts for q in ('.git', '__pycache__', '.venv')):
        continue
    if p.suffix not in ('.py', '.ipynb', '.json', '.md', '.txt'):
        continue
    text = p.read_text(encoding='utf-8-sig')
    assert not patterns.search(text), f'Potential credential: {p.relative_to(ROOT)}'
    count['files'] += 1
    if p.suffix == '.py':
        ast.parse(text, filename=str(p))
        count['python'] += 1
    if p.suffix == '.ipynb':
        nb = json.loads(text)
        assert nb['nbformat'] == 4 and nb['cells']
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                assert not c.get('outputs') and c.get('execution_count') is None
                src = ''.join(c['source'])
                if src.startswith('%%writefile '):
                    ast.parse(src.partition('\n')[2])
        count['notebooks'] += 1
selection = json.loads((ROOT / 'evidence/selected_models.json').read_text())
assert {r['phase'] for r in selection['models']} == set(range(1, 8))
for row in selection['models']:
    assert (ROOT / row['entry']).is_file(), row['entry']
for phase in range(1, 8):
    assert (ROOT / f'phase{phase}/README.md').is_file()
print('PASS', json.dumps(count), 'seven selected implementations present')
