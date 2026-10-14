# Table-caption audit (reviewer 3.8: why train on titles and abstracts only, not e.g. table captions; letter only, one Discussion sentence)
```bash
A4=scripts/figures/review/core/a4_caption_audit   # from the repository root
.venv/bin/python $A4/run_analysis.py --skip-download --n-jobs 12   # parse the cached PMC-OA download (CPU, 20 s)
.venv/bin/python $A4/caption_usability.py                         # atlas lookup, caption flags, decodability (CPU, 1.5 min)
.venv/bin/python $A4/more_bad_examples.py                         # +20 uninformative examples (CPU, 10 s)
.venv/bin/python $A4/make_figure.py                               # numbers + figure (CPU, 2 s)
```
**Inputs:** the pinned corpus (`neurovlm/neuro_image_papers`, `_hf_pins.py`), Harvard-Oxford atlases (nilearn) and the pubget 0.0.9 download of the corpus' PMC
open-access papers cached in `outputs/pubget_data` (3.1 GB; PMCID map and OA status from NCBI, 2026-10-07). Without `--skip-download` it resumes, or re-fetches if
the cache is gone (network, ~12 min; a fresh fetch can differ slightly as PMC changes).
**Outputs** (`outputs/`): `caption_usability.{png,svg}`, `caption_numbers.{md,csv}` (response numbers), `caption_examples.{md,csv}` (60 seeded examples, the response
table shows 8 uninformative ones), `caption_informativeness.*`, `caption_decodability.*`, `caption_flags.parquet`, `caption_usability.json`; `run_analysis.py`:
`tables_catalog.parquet`, `coverage_summary.csv`, `caption_summary.*`, `pmid_coverage.csv`. **Result:** 39.6 % of corpus papers are PMC-OA; captions are short (median 13
words, abstract 224) and predict a table's dominant region at chance (macro-F1 0.060 vs 0.049), whereas the title + abstract does (0.165 [0.125, 0.198]).
