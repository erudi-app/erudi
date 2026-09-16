# Corpus

Documents ingested by the `kb_ingest` phase (`standard/`) and, when the opt-in `kb_stress` phase runs, by `stress/`.

`standard/` holds documents generated for this harness: a French LaTeX PDF with ligatures and a table, the same document exported from LibreOffice, a Chinese/Japanese PDF printed from Chrome, a `.docx` with headings and a table, an `.xlsx` with formulas and dates, and a scanned image-only PDF that must land as `pending_vision`.

Two PDFs used in the first measured runs are **not** in the repository, because they are third-party papers (`corpus/**/arxiv_*.pdf` is gitignored): `arxiv_resnet_2col.pdf` (arXiv:1512.03385, two columns) in `standard/`, and `arxiv_100p_en.pdf` (arXiv:2303.08774, 100 pages, any long English paper does the same job) in `stress/`. Download them from arXiv into those folders to reproduce those runs exactly; without them the phases still run on what is present, and `kb_stress` skips when `stress/` is missing. Every report states which files were ingested, so a run with a different corpus is never mistaken for a comparable one.
