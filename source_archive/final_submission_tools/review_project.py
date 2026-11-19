"""Bounded inspection of saved project evidence; never extracts model weights or data."""
from pathlib import Path
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'final_submission/review'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    inventory = {}
    evidence = {}
    for path in sorted(ROOT.glob('*.zip')):
        try:
            with zipfile.ZipFile(path) as z:
                members = z.infolist()
                inventory[path.name] = [{'name': m.filename, 'size': m.file_size} for m in members]
                selected = {}
                for m in members:
                    n = m.filename
                    if m.file_size > 500_000 or not n.lower().endswith(('.json', '.jsonl', '.csv')):
                        continue
                    if not re.search(r'(report|metric|history|summary|audit|run_config|training_result|model_validation|selection|selected|status|comparison|leader|scor|hyperparam|manifest)', n, re.I):
                        continue
                    if any(s in n for s in ['/tokens/', 'token_cache/', 'official/', 'lexical/', 'rankings/']):
                        continue
                    try:
                        raw = z.read(n).decode('utf-8-sig')
                        selected[n] = json.loads(raw) if n.endswith('.json') else raw
                    except Exception as e:
                        selected[n] = {'read_error': str(e)}
                evidence[path.name] = selected
        except zipfile.BadZipFile:
            inventory[path.name] = {'error': 'not a readable ZIP'}
    (OUT / 'archive_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    (OUT / 'archive_evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, data in evidence.items():
        print(name, 'evidence records', len(data), 'members', len(inventory[name]))
        report = {k: v for k, v in data.items() if k == 'report.json' or re.fullmatch(r'reports/submission_\d+\.json', k)}
        if report:
            (OUT / (Path(name).stem + '.reports.json')).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    notebooks = {}
    for path in sorted(ROOT.glob('*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        outputs = []
        for i, cell in enumerate(nb.get('cells', [])):
            for output in cell.get('outputs', []):
                txt = ''.join(output.get('text', []))
                if not txt:
                    txt = ''.join(output.get('data', {}).get('text/plain', []))
                if txt:
                    outputs.append({'cell': i, 'text': txt[-50_000:]})
        notebooks[path.name] = {'cell_count': len(nb.get('cells', [])), 'outputs': outputs}
    (OUT / 'notebook_outputs.json').write_text(json.dumps(notebooks, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Evidence files written to', OUT)


if __name__ == '__main__':
    main()
