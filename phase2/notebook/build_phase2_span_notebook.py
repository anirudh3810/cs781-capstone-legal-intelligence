"""Embed the verified span-run source as a standalone Kaggle notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("span_runall.py")
OUTPUT = ROOT / "CS781_Phase2_Span_Extractor_RunAll.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    ast.parse(source)
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


intro = """# CS781 Phase 2 — span extractor, Run All

This is a **new training experiment** for the **same Kaggle working session** used by your improved Phase-2 and diagnostic notebooks. Enable **both T4 GPUs** and run **All**. The first cell checks every required file in `/kaggle/working/CS781_Capstone/phase2_improved/` and stops with an exact missing-file list if this is a different runtime. No previous Python variables are needed.

**Why this model:** the BIOUL tagger generated too few exact APP, RESP, counsel and WIT spans. In the saved public gold, all 8,393 spans map to the Phase-1 base-03 tokenizer, spans are at most 50 tokens, and whitespace-trim token boundaries reproduce about 98.2% of exact character offsets. This notebook trains label-specific start and end heads plus a learned span-pair classifier, on top of your **own Phase-1 base-03 checkpoint with LoRA**. It uses all document windows, rare-label oversampling and difficult near-boundary negative spans. It does not use outside pretrained weights or private test labels.

Training uses both GPUs in FP32, one validation fold at a time, up to 24 epochs per fold with early stopping. The selected LoRA/head state, raw held-out candidates and an optimizer checkpoint are saved separately with hashes. Run All again to resume after a time guard. The base checkpoint is reused in place; it is not copied into each fold, which protects the remaining disk space.

After three public folds, the notebook reports strict macro-F1, per-label exact candidate recall and calibrated thresholds. If its default-threshold OOF F1 clears the stated gate, it trains once on all 70 labeled documents, predicts the official test texts and writes **one** verified `prediction.json` ZIP for manual Codabench upload. This is an *additional* experiment; your previous six submissions are untouched. The private score cannot be guaranteed by a notebook.

**Download after Run All:** `/kaggle/working/CS781_Phase2_Span_Extractor_Report.zip`. If the report status is `READY_FOR_MANUAL_CODABENCH_UPLOAD`, the separate submission is `/kaggle/working/CS781_Phase2_Span_Extractor_Submission.zip`. Share the report first so we can compare it with the existing best score.

The model and code are AI assisted. Follow your course's disclosure rules.
"""

section_names = [
    "Runtime and saved-file checks",
    "Official data and verified Phase-1 base",
    "Exact token alignment and document windows",
    "LoRA span extractor and balanced training targets",
    "Two-GPU inference and strict-span scoring",
    "Resumable training of each held-out fold",
    "Three-fold evaluation and threshold selection",
    "Full-data training and verified test package",
    "Downloadable report",
]

body = SOURCE.read_text(encoding="utf-8")
parts = []
for fragment in body.split("# %% ")[1:]:
    _heading, separator, content = fragment.partition("\n")
    if not separator:
        raise RuntimeError("Malformed source section marker")
    parts.append(content)
if len(parts) != len(section_names):
    raise RuntimeError(f"Expected {len(section_names)} source sections, got {len(parts)}")

cells = [markdown(intro)]
for name, source in zip(section_names, parts):
    cells.append(markdown("## " + name + "\n"))
    cells.append(code(source))
for index, cell in enumerate(cells):
    cell["id"] = f"cs781-span-{index:03d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "cs781": {"ai_assisted": True, "phase": 2,
                   "experiment": "phase1-base03-lora-span-start-end-pair"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
