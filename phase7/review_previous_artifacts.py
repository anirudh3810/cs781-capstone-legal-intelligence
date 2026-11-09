"""Read saved phase artifacts and produce bounded, reproducible strategy evidence."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "phase7/PRIOR_PHASE_EVIDENCE.json"


def read(archive, name):
    return json.loads(archive.read(name))


def review():
    evidence = {"review_date_ist": "2026-11-05", "phase7_data_present": False,
                "private_scores_are_not_inferred_from_local_metrics": True}
    bases = {}
    for number in range(1, 5):
        file = ROOT / f"phase1_submission{number:02d}_recovery.zip"
        with zipfile.ZipFile(file) as archive:
            validation = read(archive, "manifests/model_validation.json")
            training = read(archive, "manifests/training_result.json")
            config = read(archive, "manifests/run_config.json")
            bases[f"{number:02d}"] = {
                "archive": file.name,
                "architecture": config.get("architecture_record"),
                "model_config": validation.get("model_config"),
                "completed_pretraining_steps": training.get("completed_step")}
    evidence["phase1"] = bases
    with zipfile.ZipFile(ROOT / "CS781_Phase2_Span_V3_Report.zip") as archive:
        report = read(archive, "report.json")
        evidence["phase2"] = {
            "source": "CS781_Phase2_Span_V3_Report.zip:report.json",
            "v2_oof": report["v2_baseline_oof"],
            "v3_hybrid_selected_oof": report["selection"]["chosen_oof"],
            "selected_fold_epochs": {k: v["selected_epoch"] for k, v in report["fold_results"].items()},
            "note": "Selection used these folds; selected OOF is not an untouched estimate."}
    with zipfile.ZipFile(ROOT / "CS781_Phase3_HighF1_All_Artifacts.zip") as archive:
        report = read(archive, "report.json")
        evidence["phase3"] = {k: report.get(k) for k in (
            "primary_oof_macro_f1", "official_dev_macro_f1", "private_score")}
        evidence["phase3"]["source"] = "CS781_Phase3_HighF1_All_Artifacts.zip:report.json"
    for number, name in [(4, "CS781_Phase4_All_Artifacts (1).zip"),
                         (5, "CS781_Phase5_All_Artifacts.zip"),
                         (6, "CS781_Phase6_All_Artifacts.zip")]:
        records = []
        histories = {}
        with zipfile.ZipFile(ROOT / name) as archive:
            for member in archive.namelist():
                if member.startswith("reports/submission_") and member.endswith(".json"):
                    r = read(archive, member)
                    records.append({k: r.get(k) for k in (
                        "slot", "base", "method", "validation_status", "dev_macro_f1",
                        "fit_best_epoch", "fit_best_bertscore_sample", "private_score")})
                    records[-1]["holdout_macro_f1"] = r.get("holdout_metric", {}).get("macro_f1")
                elif member.endswith("history.json"):
                    h = read(archive, member)
                    if isinstance(h, list) and h:
                        histories[member] = {"evaluations": len(h),
                            "epochs": [x.get("epoch") for x in h],
                            "reported_best_epochs": [x.get("best_epoch") for x in h]}
            value = {"source": name, "completed_submission_reports": records,
                     "histories": histories}
            if "data_audit.json" in archive.namelist():
                value["data_audit"] = read(archive, "data_audit.json")
            if number == 6:
                r = read(archive, "report.json")
                value["final_status"] = r["status"]
                value["recommended_slot"] = r["recommended_slot"]
                value["full_tune_bertscore_f1"] = r["full_tune_bertscore_f1"]
                value["untouched_audit_metric"] = r["untouched_audit_metric"]
                value["target_truncation"] = {str(b): read(archive, f"token_audit_{b:02d}.json")
                                               for b in range(1, 5)}
        evidence[f"phase{number}"] = value
    evaluator = ROOT / "phase7/reference/evaluate_phase7.py"
    evidence["phase7_evaluator"] = {
        "source": "https://drive.google.com/file/d/19Raf18n0ysaIsvaMOwtt56az7qpF-lOC/view",
        "sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
        "bytes": evaluator.stat().st_size, "k": 7,
        "formula": "2 * TP / (7 * number_of_queries + total_gold_labels)",
        "prediction_format": "JSON object mapping query ID strings to seven unique candidate ID strings"}
    OUTPUT.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(OUTPUT)


if __name__ == "__main__":
    review()
