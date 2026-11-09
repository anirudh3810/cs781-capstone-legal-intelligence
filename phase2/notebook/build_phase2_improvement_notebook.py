"""Build a self-contained Kaggle Run-All improvement notebook from the proven runner.

The old notebook is read-only.  Its setup, archive/data validation, checkpoint
implementation and stable T4 training loop are embedded verbatim.  The new
orchestration lives in improvement_runall.py and is embedded in the last cell.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / 'capstone-project-phase2-notebook.ipynb'
TARGET = PROJECT / 'CS781_Phase2_Improved_Three_RunAll.ipynb'
RUNNER = HERE / 'improvement_runall.py'
COMMON = PROJECT / 'phase2' / 'phase2_methods' / 'phase2_method_common.py'


def code_cell(source: str) -> dict:
    return {'cell_type': 'code', 'id': hashlib.sha256(source.encode()).hexdigest()[:12],
            'execution_count': None, 'metadata': {},
            'outputs': [], 'source': source.splitlines(keepends=True)}


def markdown_cell(source: str) -> dict:
    return {'cell_type': 'markdown', 'id': hashlib.sha256(source.encode()).hexdigest()[:12],
            'metadata': {}, 'source': source.splitlines(keepends=True)}


def main() -> None:
    old = json.loads(SOURCE.read_text(encoding='utf-8'))
    get = lambda index: ''.join(old['cells'][index]['source'])

    setup = get(2)
    previous_root = "Path('/kaggle/working/CS781_Capstone/phase2')"
    improved_root = "Path('/kaggle/working/CS781_Capstone/phase2_improved')"
    assert setup.count(previous_root) == 1
    setup = setup.replace(previous_root, improved_root)
    marker = 'if RUN_PHASE2:\n    stage_official_phase2_files()'
    assert setup.count(marker) == 1
    working_fallback = '''# Prefer the verified files already present in the previous Kaggle working tree.
OLD_PHASE2_ROOT = Path('/kaggle/working/CS781_Capstone/phase2')
for _name in DATA_NAMES:
    _existing = OLD_PHASE2_ROOT / 'official_phase2' / _name
    if _existing.is_file():
        FILE_OVERRIDES[_name] = str(_existing)

'''
    setup = setup.replace(marker, working_fallback + marker)

    # The earlier notebook has append-only fixes; their final definitions are
    # required.  The old six-slot orchestration is deliberately not executed.
    inference_definitions = get(16).split('def phase2_preflight():', 1)[0]
    assert 'def make_submission_zip' in inference_definitions
    assert 'def load_selected_task' in inference_definitions
    common_source = COMMON.read_text(encoding='utf-8')
    method_sources = {
        slot: (COMMON.parent / f'phase2_method_{slot:02d}.py').read_text(encoding='utf-8')
        for slot in (1, 3, 5)
    }
    runner_source = RUNNER.read_text(encoding='utf-8')
    embedded = ('IMPROVED_COMMON_SOURCE = ' + repr(common_source) + '\n'
                + 'IMPROVED_METHOD_SOURCES = ' + repr(method_sources) + '\n\n'
                + runner_source)

    notebook = {
        'cells': [
            markdown_cell('''# CS781 Phase 2 — improve submissions 01, 03 and 05

Run All in the same Kaggle environment as the previous Phase-2 run.  This
notebook keeps the old Phase-2 output directory unchanged and writes new work
under `/kaggle/working/CS781_Capstone/phase2_improved`.

It runs three document-level validation folds per method, then trains final
models from the matching Phase-1 checkpoints, makes resumable official-test
predictions and exports three **new** validated ZIPs.  Every training stage has
its own checkpoint, history and uniquely named recovery ZIP.  If a session
time guard stops work, attach the new recovery ZIPs as Kaggle Inputs and Run All
again.  No notebook can guarantee an uninterrupted Kaggle session, exact GPU
utilization or a higher official score.

The methods and hyperparameters are AI-assisted.  Follow your course's
authorship and disclosure rules.
'''),
            code_cell(setup),
            code_cell(get(4)),
            code_cell(get(6)),
            code_cell(get(8)),
            code_cell(get(10)),
            code_cell(get(12)),
            code_cell(get(13)),
            code_cell(get(14)),
            code_cell(inference_definitions),
            code_cell(embedded),
        ],
        'metadata': old.get('metadata', {}),
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(TARGET)


if __name__ == '__main__':
    main()
