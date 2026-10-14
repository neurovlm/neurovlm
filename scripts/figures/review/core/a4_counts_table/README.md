# Dataset counts and the publication-level aggregation rule (comments 1.2, 1.5, 2.2)
**Comments:** 1.2 final numbers of publications, experiments, contrasts and coordinate tables, and how coordinates are pooled; 1.5 3,000 vs 4,553 NeuroVault maps,
100 vs 300 network images; 2.2 sample sizes per model. **Goes to** the reviewer letter (counts) and the supplement (per-stage sizes).
```bash
.venv/bin/python scripts/figures/review/core/a4_counts_table/run_analysis.py --model-dir scripts/outputs/runs/final --force   # CPU, 6 s
.venv/bin/python scripts/figures/review/core/a4_counts_table/make_figure.py                                                  # tables, < 1 s
```
**Inputs:** pinned HF data (`_hf_pins.py`; data-dir copies written by `scripts/data/fetch_hf.py` are sha256-checked), the final run's `pmids_split.pt` and stage
summaries `08`-`14_*.summary.json`; the curated stage 2 texts are rebuilt with `build_text_table` of `scripts/models/14_qformer_anat.py`.
**Outputs** (`outputs/`): `counts_table.{md,csv,tex}` (one row per question of 1.2 / 1.5), `stage_splits.{md,csv,tex}` (rows per training stage and split),
`counts.json`, `counts_long.csv`, `coords_*.csv` (raw counts).
**Result:** 30,826 papers, 857,298 peaks (median 17 per paper). All peaks of a paper (experiments, contrasts, tables, signs) are pooled into one map; the released
coordinates have no contrast, table or sign field. Split by paper 23,894 / 2,987 / 2,987, 958 papers in no split. NeuroVault: 3,183 released maps from 312 papers
('4,553' not reproducible); networks: 185 maps (145 labelled + 40 ICA). Per stage (train / val): anatomically centred head 6,309 / 64 synthetic pairs; stage 1
projector 22,547 / 2,837 papers; stage 2 projector 972 / 134 texts (1,106 of 1,338 kept by the self-consistency filter).
