"""Package the standalone Phase-2 infrastructure into one Kaggle notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parents[1] / 'phase2_infrastructure.py'
OUTPUT = PROJECT_ROOT / 'CS781_Phase2_Six_Submission_Infrastructure.ipynb'
PROJECT_COPY = Path(__file__).resolve().parent / OUTPUT.name


def markdown(value: str):
    return nbformat.v4.new_markdown_cell(value.strip() + '\n')


def code(value: str):
    ast.parse(value)
    return nbformat.v4.new_code_cell(value.strip() + '\n')


source = SOURCE.read_text(encoding='utf-8')
pieces = source.split('# %% ')
if pieces[0].strip() or len(pieces) < 5:
    raise RuntimeError('Unexpected Phase-2 source-cell markers')

notebook = nbformat.v4.new_notebook()
notebook.metadata['cs781'] = {
    'phase': 2,
    'purpose': 'six-experiment legal NER AI-assisted reference workflow',
    'ai_assisted_methods': True,
    'executed': False,
}
notebook.cells.append(markdown('''
# CS781 Phase 2 — six legal-NER reference experiments

This Kaggle notebook restores your four Phase-1 checkpoints and trains six
AI-assisted reference NER methods: full fine-tuning, partial encoder tuning,
additive LoRA, multiplicative LoRMA+, adapters, and distillation from the
trained slot-1 NER teacher. BIOUL and character-boundary heads handle exact
spans, including boundaries within a BPE token. It includes a deterministic document split, GPU
smoke tests, checkpoints, integrity-checked recovery ZIPs, retryable inference,
strict character-span checks, and one `submission.zip` per completed experiment.

**Academic-use warning:** the capstone documents restrict LLM authorship of
assessed model architectures, tuning, and conclusions. These methods and
settings were AI-assisted. `STUDENT_AUTHORSHIP_CONFIRMED` remains false; do
not present them as independently student-authored or submit contrary to the
course rules. Review with the instructor if their use is allowed.

The six official Phase-2 files are downloaded automatically from the gated
`Exploration-Lab/CS-781-Capstone` Hugging Face dataset. First accept access
to that dataset, add a Hugging Face **read** token as an enabled Kaggle Secret
named `HF_TOKEN`, and enable Kaggle Internet. Alternatively attach all six
official files as Kaggle Inputs; then no token or download is needed. Attach
the four Phase-1 recovery ZIPs and `phase2_method_common.py` plus the six
`phase2_method_XX.py` files. With a Kaggle GPU enabled, Run All starts training automatically. It uses both GPUs
when Kaggle supplies two, but speed and completion within one session cannot
be guaranteed without a real GPU timing run. Download recovery ZIPs after
each session. Reattach them in another session and Run All again; `auto` mode
resumes verified checkpoints. Incomplete training never yields a ready ZIP.

For prediction-only generation, use `RUN_MODE='inference_only'` and set
`RESUME_EXPECTED_SLOTS={1,2,3,4,5,6}` after attaching all six completed
recovery ZIPs and the four Phase-1 archives. The notebook downloads
`fold3-text.json` and `labels.txt` if they are not attached.
'''))

for piece in pieces[1:]:
    heading, separator, body = piece.partition('\n')
    if not separator or not body.strip():
        raise RuntimeError(f'Empty notebook source section: {heading}')
    notebook.cells.append(markdown(f'## {heading.strip()}'))
    notebook.cells.append(code(body))

notebook.cells.append(markdown('''
## Before uploading to Codabench

Each slot writes `experiments/submission_XX/submission.zip` with a root-level
`prediction.json`. Upload **six separate ZIPs** during the official Phase-2
window; the notebook does not upload to Codabench. Only the final
`six_submission_validation.json` indicates that all six local packages exist,
have all official IDs, and differ in prediction content. It does not certify
scientific validity or Codabench acceptance.

Download every per-experiment recovery ZIP outside Kaggle. Record checkpoint
hashes, training settings, held-out validation results, submission IDs, and
actual Codabench reports. Do not fabricate measurements or conclusions, or
claim independent authorship of these AI-assisted method implementations.
'''))

nbformat.validate(notebook)
for destination in (OUTPUT, PROJECT_COPY):
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
print(f'Wrote {len(notebook.cells)} cells to {OUTPUT}')
