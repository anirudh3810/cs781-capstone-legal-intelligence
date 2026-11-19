from pathlib import Path

BASE = Path(__file__).resolve().parent
changes = {
    'Official Codabench best scores, leaderboard ranks and recorded valid submissions on 9 November 2026.':
        'Official Codabench scores, ranks and valid submission counts on 9 November 2026.',
    "The report is bounded by saved evidence. Official Codabench scores are reported in Table~\\ref{tab:leaderboard}; leaderboard ranks were checked on the public phase results at the freeze date. Phase~4 lacks completed outer validation; Phase~7's supplied archive is older than its standalone slot05 output and contains no final audit. All four foundation training notebooks (bases 01--04) are included. The repository preserves original runtime sources separately where local files differ, and retains every available alternative instead of presenting a provisional local choice as an official winner.":
        "Leaderboard scores and ranks are verified, but local evidence remains uneven. Phase~4 lacks completed outer validation, and Phase~7's recovery predates its standalone slot05 package and contains no final audit. All four foundation notebooks and the available method alternatives are included. Archived runtime sources preserve changes absent from local versions.",
    'Next experiments should hold other factors fixed: change NER boundary generation and measure participant candidate recall; reserve whole documents across both role domains; complete judgment outer folds; compare statute cardinality control on frozen probabilities; and evaluate summary target budgets at matched fitting steps. Retrieval should complete its audit once, then compare sparse-only, dense-only and fused ranks before changing context budgets. These tests address observed bottlenecks directly.':
        'Next experiments should isolate boundary generation for NER, whole-document generalization for roles and judgments, cardinality control for statutes, and source/target budgets for summaries. Retrieval should finish its audit, then compare sparse, dense and fused ranks at fixed encoder states.'
}
for p in [BASE/'write_report.py', BASE/'package/report/sections/page01.tex', BASE/'package/report/sections/page09.tex']:
    text = p.read_text(encoding='utf-8')
    for a, b in changes.items():
        text = text.replace(a, b)
    p.write_text(text, encoding='utf-8')
print('Shortened repeated discussion while preserving findings and the template format.')
