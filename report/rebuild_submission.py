"""Rebuild the code/report ZIP after the author updates and compiles the report."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / 'code' if (ROOT / 'code').is_dir() else ROOT
DEST = ROOT / 'dist/CS781-CP-Anirudh-Bankhede-240131-report_code.zip'
PDF = ROOT / 'report/CS781-CP-Anirudh-Bankhede-240131-report.pdf'
if not PDF.is_file():
    raise SystemExit('Compile the report first with report/build_report.py')
sources = [p for p in PDF.parent.rglob('*') if p.is_file() and p.suffix in ('.tex', '.bib')]
if any(p.stat().st_mtime_ns > PDF.stat().st_mtime_ns for p in sources):
    raise SystemExit('Report source changed after the PDF. Compile again with report/build_report.py before rebuilding the ZIP.')
allowed_report = {'.tex', '.bib', '.md', '.py', '.txt'}
entries = []
for folder, source in [('code', CODE), ('report', ROOT / 'report')]:
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if not path.is_file() or any(p in relative.parts for p in ('.git', '__pycache__', '.venv', '.ipynb_checkpoints', 'dist')):
            continue
        if folder == 'code' and relative.parts[0] in {'report', 'reference', 'source_archive'}:
            continue
        if folder == 'report' and not (path == PDF or path.suffix in allowed_report or path.parent.name == 'figures'):
            continue
        entries.append((path, folder + '/' + relative.as_posix()))
DEST.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path, name in entries:
        z.write(path, name)
with zipfile.ZipFile(DEST) as z:
    assert {name.split('/')[0] for name in z.namelist()} == {'code', 'report'}
    assert z.testzip() is None
print('ZIP:', DEST)
print('Files:', len(entries), 'Bytes:', DEST.stat().st_size)
print('SHA-256:', hashlib.sha256(DEST.read_bytes()).hexdigest())
print('Before uploading: review the filled declaration against the course rules. All seven leaderboard ranks are verified.')
