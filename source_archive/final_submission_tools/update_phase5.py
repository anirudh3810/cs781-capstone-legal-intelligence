"""Refresh Phase 5 only; preserve other experiment sources and author edits."""
from pathlib import Path
import hashlib
import importlib.util
import json
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CODE = BASE / 'package/code'
ARCHIVE = 'CS781_Phase5_All_Artifacts.zip'


def main():
    evidence_path = BASE / 'review/archive_evidence.json'
    inventory_path = BASE / 'review/archive_inventory.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    old = evidence[ARCHIVE]
    updated = {}
    with zipfile.ZipFile(ROOT / ARCHIVE) as z:
        inventory[ARCHIVE] = [{'name': i.filename, 'size': i.file_size} for i in z.infolist()]
        # Reuse the previously reviewed member set and discover new completed runs.
        for name in sorted(set(old) | {n for n in z.namelist() if n.endswith(('history.json', 'status.json')) or n.startswith('reports/')}):
            if name in z.namelist() and name.endswith('.json') and z.getinfo(name).file_size < 500_000:
                updated[name] = json.loads(z.read(name))
        updated['report.json'] = json.loads(z.read('report.json'))
        assert updated['report.json']['status'] == 'SIX_DISTINCT_PACKAGES_READY'
        assert all(f'reports/submission_{i:02d}.json' in updated for i in range(1, 7))
        evidence[ARCHIVE] = updated
        provenance_path = CODE / 'evidence/source_provenance.json'
        provenance = json.loads(provenance_path.read_text())
        for r in provenance:
            if r['source_archive'] == ARCHIVE:
                content = z.read(r['source_member'])
                (CODE / r['path']).write_bytes(content)
                r['sha256'] = hashlib.sha256(content).hexdigest()
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        spec = importlib.util.spec_from_file_location('assemble_code', BASE / 'assemble_code.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = 'CS781_Phase5_Six_Submissions_RunAll.ipynb'
        current = CODE / 'phase5/notebooks' / name
        module.clean_notebook(ROOT / name, current)
        nb = json.loads(current.read_text(encoding='utf-8'))
        bindings = {'cs781_phase5_utils.py': 'phase5_utils.py', 'cs781_phase5_lsi_runner.py': 'lsi_runner.py'}
        replaced = set()
        for c in nb['cells']:
            source = ''.join(c['source'])
            if not source.startswith('%%writefile '):
                continue
            first = source.splitlines()[0]
            target = Path(first.split()[-1]).name
            if target in bindings:
                c['source'] = first + '\n' + z.read(bindings[target]).decode('utf-8')
                replaced.add(target)
        assert replaced == set(bindings)
        (CODE / 'phase5/archived_run' / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    inventory_path.write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
    reports = {k: v for k, v in updated.items() if k == 'report.json' or k.startswith('reports/')}
    (BASE / 'review/CS781_Phase5_All_Artifacts.reports.json').write_text(json.dumps(reports, indent=2) + '\n', encoding='utf-8')
    print('Updated Phase 5: six completed reports, notebook, archived runtimes and provenance.')


if __name__ == '__main__':
    main()
