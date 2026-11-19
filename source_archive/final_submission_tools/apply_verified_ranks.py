"""Use only the user's rows read from the actual phase leaderboards."""
from pathlib import Path
import json
import re

BASE = Path(__file__).resolve().parent
CODE = BASE / 'package/code'
REPORT = BASE / 'package/report'


def main():
    proof = json.loads((BASE / 'review/verified_codabench_ranks.json').read_text())
    rows = proof['phases']
    assert set(rows) == set(map(str, range(1, 8)))
    for value in rows.values():
        assert value['rank'] > 0 and value['participant'] == 'a_240131'
        assert value['phase_heading_verified']
    for path in (REPORT / 'sections/page01.tex', BASE / 'write_report.py'):
        text = path.read_text(encoding='utf-8')
        for phase, value in rows.items():
            pattern = rf'(?m)^({phase} & [^&\n]+ & )\d+\.\d{{4}}( & )(?:--|\\NR|\d+)( & \d+\\\\)'
            replacement = rf'\g<1>{value["score"]:.4f}\g<2>{value["rank"]}\g<3>'
            text, count = re.subn(pattern, replacement, text)
            assert count == 1, (str(path), phase, count)
        text = text.replace('Official Codabench best scores and valid submissions recorded on 9 November 2026.',
            'Official Codabench best scores, leaderboard ranks and recorded valid submissions on 9 November 2026.')
        text = text.replace('all seven phases met and exceeded this threshold', 'all seven phases met this minimum')
        path.write_text(text, encoding='utf-8')
    for path in (REPORT / 'sections/page09.tex', BASE / 'write_report.py'):
        text = path.read_text(encoding='utf-8')
        text = text.replace('leaderboard ranks remain unverified at the freeze date.',
            'leaderboard ranks were checked on the public phase results at the freeze date.')
        path.write_text(text, encoding='utf-8')
    evidence_path = CODE / 'evidence/verified_local_results.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    evidence['official_ranks_verified'] = True
    evidence['official_best_scores_verified'] = True
    evidence['official_rank_source'] = proof['source_url']
    evidence['official_rank_observed_at_utc'] = proof['observed_at_utc']
    for phase, value in rows.items():
        evidence['official_codabench_results']['phases'][phase].update({
            'rank': value['rank'], 'best_score': value['score'],
            'leaderboard_submission_id': value['submission_id']})
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (CODE / 'evidence/verified_codabench_ranks.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    table = '\n'.join(['| Phase | Official score | Rank | Submission ID |', '|---|---:|---:|---:|'] + [
        f'| {i} | {rows[str(i)]["score"]:.4f} | {rows[str(i)]["rank"]} | {rows[str(i)]["submission_id"]} |'
        for i in range(1, 8)])
    for path in (CODE / 'README.md', BASE / 'write_readmes.py'):
        text = path.read_text(encoding='utf-8')
        anchor = '## Repository structure'
        section = ('## Verified phase ranks\n\nThe rows below were read from the actual phase leaderboards for `a_240131` on 9 November 2026. '
            'Ranks may change as new submissions are scored.\n\n' + table + '\n\n'
            'Source: [Codabench phase results](' + proof['source_url'] + ').\n\n')
        assert anchor in text
        if '## Verified phase ranks' not in text:
            text = text.replace(anchor, section + anchor)
        path.write_text(text, encoding='utf-8')
    print('Updated seven verified ranks in the report and repository evidence.')


if __name__ == '__main__':
    main()
