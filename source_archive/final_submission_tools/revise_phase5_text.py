"""Targeted manuscript and README edits, preserving the author's declaration."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
replacements = {
    'Slot01 leads four completed reports; slots05/06 lack final metric records despite standalone packages.':
        'All six reports are complete. Slot01 remains strongest on holdout and selected dev F1, with slot06 a close second on holdout.',
    '05 & .26880 & .41110 & .74465 & \\NR & .82240 & .32595$^{*}$':
        '05 & .26880 & .41110 & .74465 & .34339 & .82240 & .32595$^{*}$',
    '06 & .13555 & .41130 & .68770 & \\NR & .82395 & .33127':
        '06 & .13555 & .41130 & .68770 & .37650 & .82395 & .33127',
    'NR entries lack completed metric records.': 'All six Phase~5 holdout results are now available.',
    'Phase~5 lacks final reports for two standalone packages; ': '',
}
for path in [BASE / 'write_report.py', *sorted((BASE / 'package/report/sections').glob('*.tex'))]:
    old = path.read_text(encoding='utf-8')
    new = old
    for a, b in replacements.items():
        new = new.replace(a, b)
    # Add the comparison to the results paragraph without regenerating the document.
    anchor = "The role blend's dev advantage does not establish unseen-document superiority."
    if anchor in new and 'Phase~5 distillation reaches' not in new:
        extra = ' Phase~5 distillation reaches holdout macro F1 0.37650 with 15.08M trainable parameters, close to partial tuning at 0.37934 with 22.18M; the different adaptation schemes prevent a single-factor attribution.'
        new = new.replace(anchor, anchor + extra)
    if new != old:
        path.write_text(new, encoding='utf-8')

before = 'Among the four completed archived reports, partial base03 (slot01) is strongest. Standalone packages 05/06 are present locally but their final reports are absent from the supplied recovery; their final training metrics are not fabricated.'
after = '''The updated recovery contains six completed reports and six distinct prediction packages. Partial base03 (slot01) remains strongest on holdout and selected dev macro F1. Distilled base03 (slot06) is close on holdout with fewer trainable parameters.

| Slot | Method / base | Holdout macro F1 | Selected dev macro F1 | Trainable parameters | Selected final epoch |
|---|---|---:|---:|---:|---:|
| 01 | Partial / B03 | 0.379343 | 0.385573 | 22,175,306 | 7 |
| 02 | Full / B01 | 0.374302 | 0.382241 | 82,994,378 | 6 |
| 03 | LoRMA / B04 | 0.343317 | 0.348887 | 1,863,370 | 7 |
| 04 | Adapters / B02 | 0.327873 | 0.329778 | 1,313,866 | 6 |
| 05 | Adaptive rank / B03 | 0.343392 | 0.343983 | 1,868,426 | 7 |
| 06 | Distillation / B03 | 0.376503 | 0.380021 | 15,078,474 | 7 |

These local evaluations remain separate from official Codabench scores. Each method's test package covers 13,019 cases.'''
for path in (BASE / 'write_readmes.py', BASE / 'package/code/README.md'):
    old = path.read_text(encoding='utf-8')
    if before in old:
        path.write_text(old.replace(before, after), encoding='utf-8')

print('Updated Phase 5 discussion, six-score table, limitations and repository README.')
