"""Export only aggregate, traceable measurements for the report and public README."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CODE = BASE / 'package/code'


def scalars(d):
    return {k: v for k, v in d.items() if not isinstance(v, (dict, list))}


def main():
    raw = json.loads((BASE / 'review/archive_evidence.json').read_text(encoding='utf-8'))
    data = {'freeze_date_ist': '2026-11-09', 'student': 'Anirudh Bankhede', 'roll': '240131',
            'codabench_username': 'a_240131', 'official_best_scores_verified': False,
            'official_ranks_verified': False, 'official_valid_submission_counts_verified': False,
            'phase1': {}, 'phase2': {}, 'phase3': {}, 'phase4': {}, 'phase5': {}, 'phase6': {}, 'phase7': {}}
    curves = {'phase1_validation': {}, 'phase6_tune': {}, 'phase7_tune': {}}
    for i in range(1, 5):
        archive = f'phase1_submission{i:02d}_recovery.zip'
        r = raw[archive]
        data['phase1'][f'{i:02d}'] = {'source_archive': archive,
            'training': scalars(r['manifests/training_result.json']),
            'parameters': r['manifests/model_validation.json']['total'],
            'corpus': r['manifests/dataset_audit.json'],
            'config': r.get('manifests/training_config.json')}
        for key in ('latest_checkpoint', 'selected_checkpoint'):
            data['phase1'][f'{i:02d}']['training'].pop(key, None)
        with zipfile.ZipFile(ROOT / archive) as z:
            records = [json.loads(s) for s in z.read('histories/training.jsonl').decode().splitlines() if s]
        curves['phase1_validation'][f'{i:02d}'] = [
            {k: s.get(k) for k in ('step', 'tokens_processed', 'validation_loss')}
            for s in records if s.get('validation_loss') is not None]
    p2 = raw['CS781_Phase2_Span_V3_Report.zip']['report.json']
    data['phase2'] = {'source_archive': 'CS781_Phase2_Span_V3_Report.zip',
        'public_labeled_documents': 70, 'official_train_documents': 35, 'official_dev_documents': 35, 'test_documents': 35,
        'v2_oof': p2['v2_baseline_oof'], 'selection': p2['selection'],
        'original_six': [scalars(raw['CS781_Phase2_COMPLETE_ARTIFACTS.zip'][f'experiments/submission_{i:02d}/training_result.json']) for i in range(1, 7)],
        'gold_alignment': raw['CS781_Phase2_Span_Extractor_Report.zip']['report.json']['gold_alignment'],
        'historical_user_reported_v2_official': p2['v2_official_score_user_reported'],
        'historical_official_reference_is_not_verified_best': True}
    # Avoid duplicating workspace paths in the public summary.
    for r in data['phase2']['original_six']:
        for k in ('checkpoint', 'selected', 'recovery'):
            r.pop(k, None)
    p3 = raw['CS781_Phase3_HighF1_All_Artifacts.zip']
    data['phase3'] = {'source_archive': 'CS781_Phase3_HighF1_All_Artifacts.zip',
        'audit': p3['data_audit.json'], 'report': scalars(p3['report.json']),
        'decoder_comparisons': {k: {x: scalars(y) for x, y in v.items()} for k, v in p3['selection.json'].items() if k in ('primary_oof', 'official_dev')},
        'original_six': raw['CS781_Phase3_All_Artifacts.zip']['six_submission_report.json']['slots'],
        'one_submission_dev_supported': raw['CS781_Phase3_One_All_Artifacts.zip']['one/one_report.json']['development_supported_macro_f1'],
        'one_submission_dev_all13': raw['CS781_Phase3_One_All_Artifacts.zip']['one/one_report.json']['development_all_13_macro_f1']}
    for p, archive in [(4, 'CS781_Phase4_All_Artifacts (1).zip'), (5, 'CS781_Phase5_All_Artifacts.zip'), (6, 'CS781_Phase6_All_Artifacts.zip'), (7, 'CS781_Phase7_All_Artifacts.zip')]:
        r = raw[archive]
        reports = []
        for name, value in r.items():
            if not name.startswith('reports/submission_'):
                continue
            row = scalars(value)
            for key in ('holdout_metric', 'dev_metric_selected', 'tune_sample_metric'):
                if isinstance(value.get(key), dict):
                    row[key] = scalars(value[key])
            reports.append(row)
        data[f'phase{p}'] = {'source_archive': archive, 'audit': r['data_audit.json'], 'reports': sorted(reports, key=lambda row: row['slot'])}
        if p == 5:
            data['phase5']['final_status'] = r['report.json']['status']
            data['phase5']['distinct_packages'] = r['report.json']['distinct']
        if p == 6:
            final = r['report.json']
            data['phase6'].update({k: final[k] for k in ('recommended_slot', 'full_tune_bertscore_f1', 'untouched_audit_metric')})
            data['phase6']['targets_truncated_base03'] = r['token_audit_03.json']['targets_truncated']
            data['phase6']['label_audit'] = json.loads(zipfile.ZipFile(ROOT / archive).read('official/label_audit.json'))
            for slot in ('01', '02', '03', '04', '05', '06'):
                curves['phase6_tune'][slot] = [{'epoch': x['epoch'], 'f1': x['metric']['bertscore_f1']} for x in r[f'runs/{slot}/fit/history.json']]
        if p == 7:
            data['phase7']['gold_control_comparison'] = r['report.json']['gold_control_comparison']
            data['phase7']['archive_status'] = r['report.json']['status']
            data['phase7']['slot05_last_archived_status'] = r['runs/05/status.json']
            data['phase7']['slot05_standalone_submission_present'] = (ROOT / 'CS781_Phase7_Submission_05.zip').is_file()
            data['phase7']['slot05_last_archived_selection'] = r['runs/05/selected.json']
            for slot in ('01', '02', '03', '04', '05', '06', 'control'):
                curves['phase7_tune'][slot] = [{'epoch': x['epoch'], 'f1': x['metric']['primary_metric_1']} for x in r[f'runs/{slot}/history.json']]
    output = CODE / 'evidence'
    output.mkdir(exist_ok=True)
    (output / 'verified_local_results.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'learning_curves.json').write_text(json.dumps(curves, indent=2) + '\n', encoding='utf-8')
    selected = [
        {'phase': 1, 'variant': '04', 'basis': 'lowest logged validation loss; separate pretraining protocols, official best unverified', 'entry': 'phase1/notebooks/CS781_Phase1_Submission04_Infrastructure_FIXED.ipynb'},
        {'phase': 2, 'variant': 'span_v3_hybrid', 'basis': 'highest saved selected OOF span macro F1; selection biased', 'entry': 'phase2/notebooks/CS781_Phase2_Span_V3_RunAll.ipynb'},
        {'phase': 3, 'variant': 'one_submission', 'basis': 'highest saved development supported-label F1; HighF1 cross-validation alternative also included', 'entry': 'phase3/notebooks/CS781_Phase3_One_Strong_Submission_RunAll.ipynb'},
        {'phase': 4, 'variant': '02', 'basis': 'highest saved official-dev macro F1; no completed outer folds', 'entry': 'phase4/archived_run/CS781_Phase4_Six_Submissions_RunAll.ipynb'},
        {'phase': 5, 'variant': '01', 'basis': 'highest saved holdout and dev macro F1 among all six completed archived reports', 'entry': 'phase5/archived_run/CS781_Phase5_Six_Submissions_RunAll.ipynb'},
        {'phase': 6, 'variant': '02', 'basis': 'archive recommendation, full tuning score and untouched audit evaluation', 'entry': 'phase6/notebooks/CS781_Phase6_Six_Submissions_RunAll.ipynb'},
        {'phase': 7, 'variant': '01', 'basis': 'highest saved tuning-sample F1; final audit absent in supplied archive', 'entry': 'phase7/notebooks/CS781_Phase7_Six_Submissions_RunAll.ipynb'},
    ]
    (output / 'selected_models.json').write_text(json.dumps({'selection_kind': 'provisional_local_evidence',
        'official_best_confirmation_pending': True, 'models': selected}, indent=2) + '\n', encoding='utf-8')
    print('Wrote aggregate results, curves and seven selected-code entry points.')


if __name__ == '__main__':
    main()
