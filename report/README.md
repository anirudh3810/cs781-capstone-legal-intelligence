# Editable final report

The report follows the provided title format, eight required top-level sections, 11-point type and one-inch margins on letter paper. The abstract has 127 words. Figures are vector PDFs derived from the saved project measurements.

The declaration in `declaration.tex` is filled using the model name **GPT Sol 6**, with project-specific explanations of understanding, architecture planning and result analysis, alongside a concise disclosure of the implementation assistance used. Review the personal-contribution statements for accuracy before submitting. All seven official scores and ranks for `a_240131` were checked on the public Codabench phase results on 9 November 2026. The repository contains a record of the displayed participant rows in [`../evidence/verified_codabench_ranks.json`](../evidence/verified_codabench_ranks.json).

1. Review `declaration.tex`, the results and personal-contribution statements. The course template restricts some of the disclosed uses; an accurate declaration does not by itself establish that those uses are permitted.
2. Compile with `python build_report.py`, or import `Report.tex`, `sections/`, `figures/`, `references.bib` and `declaration.tex` into Overleaf. With a standard TeX installation, run `latexmk -pdf Report.tex`. The required PDF filename is `CS781-CP-Anirudh-Bankhede-240131-report.pdf`.
3. Confirm the final counted main report is no more than ten pages. Declaration and references are on subsequent pages.
4. Run `python rebuild_submission.py` to create `../dist/CS781-CP-Anirudh-Bankhede-240131-report_code.zip` with the edited PDF and current curated code. It puts `code/` and `report/` directly at the ZIP root and excludes Git metadata, compiler intermediates, reference slides and the additional source archive.
5. The original submission deadline recorded in the package was 9 November 2026, 11:59 PM IST. This repository is an archive; publishing here does not submit anything to Hello IITK.

Code repository: https://github.com/anirudh3810/cs781-capstone-legal-intelligence

The training recoveries and restricted datasets stay outside the public source repository and the small code/report package. They remain in the original project folder. The code README documents which recoveries are needed to reproduce each phase.

To regenerate the vector figures from aggregate evidence, install plot_requirements.txt and run python make_figures.py. The packaged figures can be used directly when only editing prose.
