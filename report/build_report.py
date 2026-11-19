"""Compile the editable report source and create the required PDF filename."""
from pathlib import Path
import argparse
import shutil
import subprocess
import re

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--tectonic', help='Optional path to the Tectonic executable')
args = parser.parse_args()
tectonic = args.tectonic or shutil.which('tectonic')
local = ROOT.parent.parent / 'tools/tectonic/tectonic.exe'
if not tectonic and local.is_file():
    tectonic = str(local)
if tectonic:
    subprocess.run([tectonic, '--keep-logs', '--keep-intermediates', 'Report.tex'], cwd=ROOT, check=True)
elif shutil.which('latexmk'):
    subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'Report.tex'], cwd=ROOT, check=True)
else:
    raise SystemExit('Install Tectonic or TeX Live/MiKTeX with latexmk; alternatively import these sources into Overleaf.')
source = ROOT / 'Report.pdf'
target = ROOT / 'CS781-CP-Anirudh-Bankhede-240131-report.pdf'
aux = (ROOT / 'Report.aux').read_text(encoding='utf-8')
match = re.search(r'LastMainPage\}\{\{[^}]*\}\{(\d+)\}', aux)
if not match or int(match[1]) > 10:
    raise SystemExit('Main-report page limit failed; inspect Report.aux and shorten the main text.')
shutil.copyfile(source, target)
print('Compiled:', target)
print('Main-report page count:', match[1], '(maximum 10; declaration and references are additional)')
