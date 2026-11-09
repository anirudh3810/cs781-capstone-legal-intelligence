"""Build a standalone Kaggle notebook from proven v1 foundations and v2 stages."""

from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / "CS781_Phase2_Span_V2_RunAll.ipynb"


def sections(path: Path) -> list[str]:
    pieces = []
    for fragment in path.read_text(encoding="utf-8").split("# %% ")[1:]:
        _, separator, content = fragment.partition("\n")
        if not separator:
            raise RuntimeError(f"Malformed section in {path}")
        pieces.append(content)
    return pieces


old = sections(HERE / "span_runall.py")[:5]
runtime = old[0]
for a, b in [
    ("ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v1'",
     "ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v2'\nV1_ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v1'"),
    ("REPORT_ZIP = WORKING / 'CS781_Phase2_Span_Extractor_Report.zip'",
     "REPORT_ZIP = WORKING / 'CS781_Phase2_Span_V2_Report.zip'"),
    ("VERSION = 'span-start-end-pair-v1'",
     "VERSION = 'span-top4-boundary-reranker-v2'"),
    ("SESSION_GUARD_SECONDS = 31_200", "SESSION_GUARD_SECONDS = 17_100"),
    ("MAX_EPOCHS = 24", "MAX_EPOCHS = 16"),
    ("MIN_EPOCHS = 8", "MIN_EPOCHS = 12"),
    ("PATIENCE = 5", "PATIENCE = 6"),
    ("VALIDATE_EVERY = 2", "VALIDATE_EVERY = 1"),
    ("MIN_FINAL_CV_F1 = 0.34", "MIN_FINAL_CV_F1 = 0.44"),
]:
    if a not in runtime:
        raise RuntimeError(f"Missing expected original source: {a}")
    runtime = runtime.replace(a, b)
old[0] = runtime
old[0] += "\ntry:\n    from sklearn.ensemble import HistGradientBoostingClassifier\nexcept ImportError as exc:\n    raise RuntimeError('This Kaggle runtime needs its standard scikit-learn package') from exc\n"

config = old[1]
config = config.replace("'batch_size': 16", "'batch_size': 16")
config = config.replace("'lora_lr': 1.5e-4, 'head_lr': 4e-4", "'lora_lr': 4e-5, 'head_lr': 1e-4")
config = config.replace("'pair_hidden': 128", "'pair_hidden': 128, 'refine_hidden': 384, 'unfreeze_top_blocks': 4")
config = config.replace("'min_final_cv_f1': MIN_FINAL_CV_F1,", "'min_final_cv_f1': MIN_FINAL_CV_F1,\n    'warm_start': 'verified phase2_span_v1 fold and final adapters',")
old[1] = config

fresh = sections(HERE / "span_v2_stages.py")
names = [
    "Runtime and file checks", "Official data and verified base",
    "Token alignment and windows", "Original LoRA span model and loss",
    "Original inference and exact scorer", "Verified warm start and larger model",
    "Resumable fold continuation", "OOF comparison and document-aware reranker",
    "Final model and test inference", "Report and submission",
]
if len(fresh) != 5:
    raise RuntimeError(f"Expected five v2 sections, found {len(fresh)}")

intro = """# CS781 Phase 2 — larger span model and document-aware reranker

**Use in the exact same Kaggle session** as `CS781_Phase2_Span_Extractor_RunAll.ipynb`. Select the two-T4 accelerator and choose **Run All**. This notebook reads the saved Phase-1 base and the completed `phase2_span_v1` fold/final adapters and candidate files. It writes only under `phase2_span_v2` and leaves the 0.4584 submission untouched. No previous Python variables are required.

The first stage continues each of the three proven fold models with a new boundary refinement network and the top four encoder blocks unfrozen. It keeps the original candidate checkpoint as the fold's baseline, selects new checkpoints only when strict held-out macro-F1 improves, and saves resumable optimizer states. Both GPUs train every fold via DataParallel; one-window inference batches use GPU 0 safely. The 16-epoch cap, early stopping, and 4-hour-45-minute guard fit the requested approximately five-hour run. **Run All again in the same Kaggle session to resume** if the guard pauses work.

The second stage cross-fits a document-aware candidate reranker on the 70 labeled documents. It uses span scores from both versions, exact-text frequency, position, and local legal-role cues. Each document is scored by a reranker trained on the other folds. The notebook compares this result with the saved v1 baseline before choosing any final method. Its threshold search on the same folds is selection-biased, so the private score remains unknown.

If validation beats the gate, a final enhanced model continues from the saved v1 full-data adapter, predicts the 35 official test documents, and packages a verified `prediction.json` ZIP. The report includes the comparison, fold histories, per-label scores, chosen method, and SHA checks. Download `CS781_Phase2_Span_V2_Report.zip` and share it before uploading the optional `CS781_Phase2_Span_V2_Submission.zip`. Also download `CS781_Phase2_Span_V2_Recovery.zip` if produced; it preserves the new trainable weights for work after this Kaggle session.

This uses only the user's own Phase-1 weights and labeled Phase-2 training documents. No outside pretrained model or private labels are used. The code is AI assisted; follow your course disclosure rules.
"""


def markdown(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s: str) -> dict:
    ast.parse(s)
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}


cells = [markdown(intro)]
for name, source in zip(names, old + fresh):
    cells.extend([markdown("## " + name + "\n"), code(source)])
for i, cell in enumerate(cells):
    cell["id"] = f"cs781-span-v2-{i:03d}"
book = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "cs781": {"ai_assisted": True, "phase": 2,
                    "experiment": "base03-top4-refine-crossfit-reranker"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
