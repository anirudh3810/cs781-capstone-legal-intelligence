"""Build the self-contained, read-only Phase-2 diagnosis notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "CS781_Phase2_Diagnostics_Steps_1_2.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown("""# CS781 Phase 2: diagnosis, steps 1 and 2

Run **All** in the **same Kaggle session** that completed `capstone-project-phase2-notebook (1).ipynb`.

This notebook reads the saved three-fold, out-of-fold candidate files for improved slots 01, 03 and 05. It reproduces their local strict macro-F1, then measures exact candidate recall, boundary errors, label confusion, and threshold/overlap losses. It does **not** load checkpoints, train, make test predictions, or submit anything. The GPUs intentionally remain free for the next training notebook; using them here would slow down a small JSON analysis.

Expected source: `/kaggle/working/CS781_Capstone/phase2_improved/`. If those loose files are unavailable, it can read `CS781_Phase2_ALL_ARTIFACTS_PART_02.zip` directly without extracting the multi-GB archive. It validates every required source and reproduces the scores saved by the earlier run before reporting diagnostics.

**Output:** `/kaggle/working/CS781_Phase2_Diagnostics_Steps_1_2.zip` plus the same small files in `/kaggle/working/CS781_Phase2_Diagnostics/`. Download the ZIP and share it for the next experiment plan.

The final cell may remove only the four exact `CS781_Phase2_ALL_ARTIFACTS_PART_XX.zip` **copies** in `/kaggle/working`, after checking that the loose working artifacts, final selected models, and submissions remain available. You said those archive parts have already been downloaded to your computer. If any check fails, it skips cleanup. It never deletes model checkpoints, official data, candidate files, submissions, or directories.
"""),
    code(r'''from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORKING = Path(os.environ.get("PHASE2_DIAG_WORKING", "/kaggle/working")).resolve()
IMPROVED = WORKING / "CS781_Capstone" / "phase2_improved"
OUT = Path(os.environ.get("PHASE2_DIAG_OUTPUT", str(WORKING / "CS781_Phase2_Diagnostics"))).resolve()
OUT_ZIP = WORKING / "CS781_Phase2_Diagnostics_Steps_1_2.zip"
PART02 = Path(os.environ.get("PHASE2_DIAG_PART02", str(WORKING / "CS781_Phase2_ALL_ARTIFACTS_PART_02.zip")))
SLOTS = (1, 3, 5)
FOLDS = (0, 1, 2)
DEFAULT_THRESHOLD = 0.45
SCORE_TOLERANCE = 1e-8
AUTO_REMOVE_BACKUP_ZIPS = os.environ.get(
    "PHASE2_DIAG_REMOVE_BACKUP_ZIPS",
    "1" if WORKING.as_posix() == "/kaggle/working" else "0",
) == "1"

def disk_gib():
    usage = shutil.disk_usage(WORKING)
    return {"total": round(usage.total / 2**30, 2),
            "used": round(usage.used / 2**30, 2),
            "free": round(usage.free / 2**30, 2)}

print("Working directory:", WORKING)
print("Improved run:", IMPROVED)
print("Disk GiB:", disk_gib())
print("GPU training: not needed for cached-candidate diagnosis")
'''),
    code(r'''REQUIRED = ["cv_folds.json", "improvement_status.json", "official_phase2/labels.txt"]
for fold in (1, 2):
    REQUIRED += [f"official_phase2/fold{fold}-text.json",
                 f"official_phase2/fold{fold}-gold.json"]
for fold in FOLDS:
    for slot in SLOTS:
        REQUIRED.append(f"cv_candidates/cv{fold}_slot{slot:02d}.json")

class SavedSources:
    def __init__(self):
        self.loose_complete = all((IMPROVED / rel).is_file() for rel in REQUIRED)
        self.archive = None
        if not self.loose_complete:
            if not PART02.is_file() or not zipfile.is_zipfile(PART02):
                missing = [rel for rel in REQUIRED if not (IMPROVED / rel).is_file()]
                raise FileNotFoundError(
                    "Missing saved Phase-2 diagnostic inputs. First missing paths: "
                    + repr(missing[:8]) + ". Attach or restore ALL_ARTIFACTS_PART_02.zip."
                )
            self.archive = zipfile.ZipFile(PART02)
            members = set(self.archive.namelist())
            missing = [rel for rel in REQUIRED
                       if not (IMPROVED / rel).is_file()
                       and f"phase2_improved/{rel}" not in members]
            if missing:
                raise FileNotFoundError("Required files absent from loose workspace and Part 02: "
                                        + repr(missing))

    def read(self, rel):
        path = IMPROVED / rel
        if path.is_file():
            return path.read_bytes()
        return self.archive.read(f"phase2_improved/{rel}")

    def json(self, rel):
        return json.loads(self.read(rel).decode("utf-8"))

source = SavedSources()
fold_manifest = source.json("cv_folds.json")
status = source.json("improvement_status.json")
labels = tuple(line.strip() for line in source.read("official_phase2/labels.txt").decode("utf-8").splitlines() if line.strip())
if len(labels) != 12 or len(set(labels)) != 12:
    raise ValueError(f"Expected twelve unique official labels; found {labels}")
if set(status.get("completed", {})) < {str(slot) for slot in SLOTS}:
    raise ValueError("Improved run status does not mark all three diagnostic slots complete")

texts, gold = {}, {}
for fold in (1, 2):
    fold_text = source.json(f"official_phase2/fold{fold}-text.json")
    fold_gold = source.json(f"official_phase2/fold{fold}-gold.json")
    if set(fold_text) != set(fold_gold) or set(texts) & set(fold_text):
        raise ValueError(f"Text/gold IDs are inconsistent for official fold {fold}")
    texts.update(fold_text)
    gold.update(fold_gold)

if len(texts) != 70 or len(fold_manifest.get("folds", [])) != 3:
    raise ValueError("Expected 70 labeled documents and three CV folds")
fold_ids = [set(map(str, group)) for group in fold_manifest["folds"]]
if set.union(*fold_ids) != set(texts) or any(fold_ids[i] & fold_ids[j] for i in FOLDS for j in FOLDS if i < j):
    raise ValueError("CV folds do not partition all 70 official training IDs")

for doc_id, rows in gold.items():
    text = texts[doc_id]
    if not isinstance(text, str) or not isinstance(rows, list):
        raise TypeError(f"Invalid official text/gold schema at {doc_id}")
    for item in rows:
        a, b, label = int(item[0]), int(item[1]), item[2]
        if label not in labels or not 0 <= a < b <= len(text):
            raise ValueError(f"Invalid official gold span in {doc_id}: {item}")

candidates = {}
source_hashes = {}
for slot in SLOTS:
    merged = {}
    for fold in FOLDS:
        rel = f"cv_candidates/cv{fold}_slot{slot:02d}.json"
        payload = source.json(rel)
        source_hashes[rel] = hashlib.sha256(source.read(rel)).hexdigest()
        group = payload["candidates"]
        if set(group) != fold_ids[fold] or set(merged) & set(group):
            raise ValueError(f"Slot {slot}, fold {fold}: candidate IDs do not match the CV manifest")
        for doc_id, rows in group.items():
            for row in rows:
                if len(row) != 4:
                    raise ValueError(f"Malformed candidate: {doc_id} {row}")
                a, b, label, confidence = row
                if (label not in labels or not 0 <= int(a) < int(b) <= len(texts[doc_id])
                        or not math.isfinite(float(confidence))
                        or not 0 <= float(confidence) <= 1):
                    raise ValueError(f"Invalid candidate: {doc_id} {row}")
        merged.update(group)
    if set(merged) != set(texts):
        raise ValueError(f"Slot {slot}: incomplete out-of-fold candidates")
    candidates[slot] = merged

print("Source mode:", "loose workspace" if source.loose_complete else "loose files plus Part 02 ZIP")
print("Official labels:", labels)
print("Official documents:", len(texts), "| three complete OOF candidate sets loaded")
'''),
    code(r'''def decode(candidate_map, thresholds):
    result = {}
    for doc_id, rows in candidate_map.items():
        accepted = []
        for a, b, label, confidence in sorted(rows, key=lambda row: -row[3]):
            a, b = int(a), int(b)
            if float(confidence) < float(thresholds.get(label, DEFAULT_THRESHOLD)):
                continue
            if any(a < other_b and other_a < b for other_a, other_b, _ in accepted):
                continue
            accepted.append((a, b, label))
        result[doc_id] = sorted(accepted)
    return result

def strict_macro(predictions, reference):
    if set(predictions) != set(reference):
        raise ValueError("Prediction and reference document IDs differ")
    per_label = {}
    for label in labels:
        predicted = {(doc_id, int(row[0]), int(row[1]))
                     for doc_id, rows in predictions.items() for row in rows if row[2] == label}
        expected = {(doc_id, int(row[0]), int(row[1]))
                    for doc_id, rows in reference.items() for row in rows if row[2] == label}
        tp = len(predicted & expected)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"tp": tp, "predicted": len(predicted), "gold": len(expected),
                            "precision": precision, "recall": recall, "f1": f1}
    return {"macro_f1": sum(row["f1"] for row in per_label.values()) / len(labels),
            "per_label": per_label}

baseline_results, tuned_results, decoded = {}, {}, {}
for slot in SLOTS:
    saved = status["completed"][str(slot)]["cv"]
    baseline = strict_macro(decode(candidates[slot], {}), gold)
    thresholds = saved["confidence_threshold_by_label"]
    tuned_pred = decode(candidates[slot], thresholds)
    tuned = strict_macro(tuned_pred, gold)
    for kind, measured, recorded in (
        ("baseline", baseline["macro_f1"], saved["baseline_oof_macro_f1"]),
        ("calibrated", tuned["macro_f1"], saved["calibrated_oof_macro_f1"]),
    ):
        if abs(measured - recorded) > SCORE_TOLERANCE:
            raise AssertionError(
                f"Slot {slot} {kind} metric mismatch: reproduced {measured:.12f}, saved {recorded:.12f}"
            )
    for fold in FOLDS:
        subset = {doc_id: candidates[slot][doc_id] for doc_id in fold_ids[fold]}
        expected = {doc_id: gold[doc_id] for doc_id in fold_ids[fold]}
        for kind, choices, recorded in (
            ("baseline", {}, saved["fold_scores"][fold]["baseline"]),
            ("tuned", thresholds, saved["fold_scores"][fold]["tuned"]),
        ):
            measured = strict_macro(decode(subset, choices), expected)["macro_f1"]
            if abs(measured - recorded) > SCORE_TOLERANCE:
                raise AssertionError(f"Slot {slot} fold {fold} {kind} metric does not reproduce")
    baseline_results[slot], tuned_results[slot], decoded[slot] = baseline, tuned, tuned_pred
    print(f"Slot {slot:02d}: baseline {baseline['macro_f1']:.6f} | calibrated {tuned['macro_f1']:.6f} "
          "| all three recorded fold scores reproduced")

print("These are LOCAL out-of-fold scores; calibrated scores were selected on the same OOF data.")
'''),
    code(r'''def intersects(x, y):
    return x[0] < y[1] and y[0] < x[1]

def excerpt(text, a, b, radius=65):
    return text[max(0, a - radius):min(len(text), b + radius)].replace("\n", "\\n")

def diagnose_slot(slot):
    rows_by_label = {label: Counter() for label in labels}
    examples = []
    raw = candidates[slot]
    predictions = decoded[slot]
    for doc_id in sorted(texts):
        text = texts[doc_id]
        gold_rows = {(int(row[0]), int(row[1]), row[2]) for row in gold[doc_id]}
        predicted = set(predictions[doc_id])
        candidate_scores = {}
        for a, b, label, confidence in raw[doc_id]:
            key = (int(a), int(b), label)
            candidate_scores[key] = max(candidate_scores.get(key, 0.0), float(confidence))
        candidate_keys = set(candidate_scores)
        for target in gold_rows:
            a, b, label = target
            row = rows_by_label[label]
            row["gold"] += 1
            same_label_candidates = [item for item in candidate_keys if item[2] == label and intersects(item, target)]
            if same_label_candidates:
                row["same_label_overlap_candidate"] += 1
            if any(intersects(item, target) for item in candidate_keys):
                row["any_label_overlap_candidate"] += 1
            if target in candidate_keys:
                row["exact_candidate"] += 1
            if target in predicted:
                row["tp"] += 1
                continue
            if target not in candidate_keys:
                reason = "not_generated"
            elif candidate_scores[target] < float(status["completed"][str(slot)]["cv"]["confidence_threshold_by_label"].get(label, DEFAULT_THRESHOLD)):
                reason = "below_threshold"
            else:
                reason = "suppressed_by_overlap"
            row["missed_" + reason] += 1
            nearby = [item for item in predicted if intersects(item, target)]
            nearest = max(nearby, key=lambda item: min(b, item[1]) - max(a, item[0])) if nearby else None
            examples.append({"slot": slot, "kind": "missed_gold", "label": label,
                             "reason": reason, "doc_id": doc_id, "start": a, "end": b,
                             "text": text[a:b], "context": excerpt(text, a, b),
                             "other_label": nearest[2] if nearest else "",
                             "other_text": text[nearest[0]:nearest[1]] if nearest else "",
                             "confidence": candidate_scores.get(target, "")})
        for target in predicted:
            a, b, label = target
            row = rows_by_label[label]
            row["predicted"] += 1
            if target in gold_rows:
                continue
            same = [item for item in gold_rows if item[2] == label and intersects(item, target)]
            if same:
                reason, other = "same_label_wrong_boundary", max(same, key=lambda item: min(b, item[1]) - max(a, item[0]))
            else:
                exact_other = [item for item in gold_rows if item[0] == a and item[1] == b]
                if exact_other:
                    reason, other = "wrong_label_exact_boundary", exact_other[0]
                else:
                    overlap = [item for item in gold_rows if intersects(item, target)]
                    reason = "wrong_label_overlap" if overlap else "spurious"
                    other = max(overlap, key=lambda item: min(b, item[1]) - max(a, item[0])) if overlap else None
            row["fp_" + reason] += 1
            examples.append({"slot": slot, "kind": "false_positive", "label": label,
                             "reason": reason, "doc_id": doc_id, "start": a, "end": b,
                             "text": text[a:b], "context": excerpt(text, a, b),
                             "other_label": other[2] if other else "",
                             "other_text": text[other[0]:other[1]] if other else "",
                             "confidence": candidate_scores.get(target, "")})

    per_label = {}
    for label in labels:
        item = rows_by_label[label]
        official = tuned_results[slot]["per_label"][label]
        if any(item[key] != official[key] for key in ("gold", "predicted", "tp")):
            raise AssertionError(f"Slot {slot} label {label}: diagnostic counts do not match strict metric")
        if sum(item["missed_" + x] for x in ("not_generated", "below_threshold", "suppressed_by_overlap")) != item["gold"] - item["tp"]:
            raise AssertionError(f"Slot {slot} label {label}: missed-gold categories do not add up")
        if sum(item["fp_" + x] for x in ("same_label_wrong_boundary", "wrong_label_exact_boundary", "wrong_label_overlap", "spurious")) != item["predicted"] - item["tp"]:
            raise AssertionError(f"Slot {slot} label {label}: false-positive categories do not add up")
        per_label[label] = dict(official, **item,
                                exact_candidate_recall=item["exact_candidate"] / item["gold"] if item["gold"] else 0.0,
                                same_label_overlap_candidate_recall=item["same_label_overlap_candidate"] / item["gold"] if item["gold"] else 0.0)
    return per_label, examples

diagnostics, all_examples = {}, []
for slot in SLOTS:
    diagnostics[slot], examples = diagnose_slot(slot)
    all_examples.extend(examples)
    weak = sorted(labels, key=lambda name: diagnostics[slot][name]["f1"])[:5]
    print(f"Slot {slot:02d} five lowest-F1 labels:", ", ".join(f"{label}={diagnostics[slot][label]['f1']:.3f}" for label in weak))
    print(f"Slot {slot:02d} exact candidate recall:",
          f"{sum(diagnostics[slot][label]['exact_candidate'] for label in labels) / sum(diagnostics[slot][label]['gold'] for label in labels):.3f}")
'''),
    code(r'''OUT.mkdir(parents=True, exist_ok=True)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

summary = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Phase-2 steps 1 and 2: reproduce saved local scores and diagnose errors",
    "source_mode": "loose workspace" if source.loose_complete else "loose files plus Part 02 ZIP",
    "source_root": str(IMPROVED),
    "candidate_sha256": source_hashes,
    "documents": len(texts),
    "labels": list(labels),
    "fold_manifest": fold_manifest,
    "slots": {
        str(slot): {
            "baseline_oof_macro_f1": baseline_results[slot]["macro_f1"],
            "calibrated_oof_macro_f1": tuned_results[slot]["macro_f1"],
            "saved_thresholds": status["completed"][str(slot)]["cv"]["confidence_threshold_by_label"],
            "per_label": diagnostics[slot],
        } for slot in SLOTS
    },
    "caveats": [
        "Local strict-span metric is an approximation of the official evaluator.",
        "Thresholds were optimized on the same OOF predictions; calibrated OOF F1 is selection-biased.",
        "Candidate recall is an upper bound for threshold/decoder-only fixes, not a predicted score.",
        "No test gold or test-set score is used by this notebook."
    ],
}
write_json(OUT / "summary.json", summary)

counts = ["gold", "predicted", "tp", "exact_candidate", "same_label_overlap_candidate",
          "any_label_overlap_candidate", "missed_not_generated", "missed_below_threshold",
          "missed_suppressed_by_overlap", "fp_same_label_wrong_boundary",
          "fp_wrong_label_exact_boundary", "fp_wrong_label_overlap", "fp_spurious"]
columns = ["slot", "label", "precision", "recall", "f1", "exact_candidate_recall",
           "same_label_overlap_candidate_recall"] + counts
per_label_rows = [{"slot": slot, "label": label, **diagnostics[slot][label]}
                  for slot in SLOTS for label in labels]
write_csv(OUT / "per_label_diagnostics.csv", per_label_rows, columns)

fold_rows = []
for slot in SLOTS:
    saved = status["completed"][str(slot)]["cv"]
    for fold in FOLDS:
        fold_rows.append({"slot": slot, "fold": fold,
                          "documents": len(fold_ids[fold]),
                          **saved["fold_scores"][fold]})
write_csv(OUT / "fold_scores.csv", fold_rows, ["slot", "fold", "documents", "baseline", "tuned"])

# Keep examples compact while covering every slot, label, and error category.
example_columns = ["slot", "kind", "label", "reason", "doc_id", "start", "end",
                   "text", "context", "other_label", "other_text", "confidence"]
groups = defaultdict(list)
for row in all_examples:
    groups[(row["slot"], row["kind"], row["label"], row["reason"])].append(row)
selected_examples = []
for key in sorted(groups):
    group = groups[key]
    group.sort(key=lambda row: (-(float(row["confidence"]) if row["confidence"] != "" else -1),
                                row["doc_id"], row["start"]))
    selected_examples.extend(group[:3])
write_csv(OUT / "error_examples.csv", selected_examples, example_columns)

readme = """# CS781 Phase-2 diagnostic report

`summary.json`: reproduced three-fold OOF scores and per-label diagnostics for improved slots 01, 03, 05.
`per_label_diagnostics.csv`: exact matches, candidate upper bounds, and disjoint missed-gold/false-positive causes.
`fold_scores.csv`: scores reproduced on each held-out document fold.
`error_examples.csv`: up to three examples per slot, label, error kind and reason, copied from official labeled training documents.

Interpretation: `exact_candidate_recall` estimates how many gold spans are already available before confidence filtering and overlap suppression. `same_label_overlap_candidate_recall` is a looser boundary opportunity measure. Neither is a promised achievable F1. The local metric approximates the official scorer; no private test labels are used.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

report_files = [OUT / name for name in ("summary.json", "per_label_diagnostics.csv",
                                            "fold_scores.csv", "error_examples.csv", "README.md")]
with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in report_files:
        archive.write(path, arcname=path.name)
with zipfile.ZipFile(OUT_ZIP) as archive:
    if archive.testzip() is not None or set(archive.namelist()) != {path.name for path in report_files}:
        raise RuntimeError("Diagnostic output ZIP verification failed")
print("Diagnostic files:", *(f"{p.name} ({p.stat().st_size / 1024:.1f} KiB)" for p in report_files), sep="\n  ")
print("Download:", OUT_ZIP, f"({OUT_ZIP.stat().st_size / 1024:.1f} KiB)")
'''),
    code(r'''# Remove only exact, top-level backup ZIP copies after the small diagnosis ZIP is verified.
# All underlying artifacts remain untouched. Cleanup is skipped when using Part 02 as a source.
backup_names = [f"CS781_Phase2_ALL_ARTIFACTS_PART_{part:02d}.zip" for part in (1, 2, 3, 4)]
selected_models = [IMPROVED / "stages" / f"cv{fold}_slot{slot:02d}" / "experiments" /
                   f"submission_{slot:02d}" / "selected_model.pt"
                   for fold in FOLDS for slot in SLOTS]
selected_models += [IMPROVED / "stages" / f"final_slot{slot:02d}" / "experiments" /
                    f"submission_{slot:02d}" / "selected_model.pt" for slot in SLOTS]
submissions = [IMPROVED / "final_submissions" / f"submission_{slot:02d}" / "submission.zip"
               for slot in SLOTS]
protected_ok = (source.loose_complete and all(p.is_file() for p in selected_models + submissions)
                and OUT_ZIP.is_file() and zipfile.is_zipfile(OUT_ZIP))
cleanup = {"enabled": AUTO_REMOVE_BACKUP_ZIPS, "protected_artifacts_present": protected_ok,
           "deleted": [], "skipped": [], "disk_before_gib": disk_gib()}
for name in backup_names:
    target = WORKING / name
    if not target.exists():
        cleanup["skipped"].append({"file": name, "reason": "not present"})
        continue
    if not AUTO_REMOVE_BACKUP_ZIPS or not protected_ok:
        cleanup["skipped"].append({"file": name, "reason": "source or protection check failed"})
        continue
    if target.is_symlink() or not target.is_file() or target.resolve().parent != WORKING:
        cleanup["skipped"].append({"file": name, "reason": "unexpected target path/type"})
        continue
    if not zipfile.is_zipfile(target):
        cleanup["skipped"].append({"file": name, "reason": "not a readable ZIP"})
        continue
    size = target.stat().st_size
    target.unlink()
    cleanup["deleted"].append({"file": name, "bytes_freed": size})
cleanup["disk_after_gib"] = disk_gib()
print("Cleanup:", json.dumps(cleanup, indent=2))
print("Final diagnostic ZIP:", OUT_ZIP)
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
