# Warning signal for out-of-distribution inputs (comment 1.4)
**Comment 1.4:** a small quantitative analysis of failure cases, out-of-distribution (OOD) inputs and an abstention or confidence signal. **Goes to** the supplement (one figure).
```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/figures/review/core/a7_ood_confidence/run_analysis.py --model-dir scripts/outputs/runs/final   # CPU, 4 min
.venv/bin/python scripts/figures/review/core/a7_ood_confidence/make_figure.py                                                                 # 2 s
```
**Inputs:** the final run (`autoencoder.pt`, InfoNCE heads, `latent_neuro.pt`, `pmids_split.pt`), frozen SPECTER2, pinned maps, texts and WikiText-2 (`_hf_pins.py`),
leakage exclusions; the surface spin uses the fsaverage-10k sphere from neuromaps.
**Method:** reference = training split (23,894 maps and abstracts); held-out control = 2,971 test maps and abstracts. OOD maps from 500 test maps (seed 7): rotated 90°
or randomly, surface spin (each keeps >= 90 % of the source's non-zero voxels), voxel shuffle, white noise, map x 100, map x 0.01; OOD text: 500 WikiText-2 paragraphs,
20 recipes. Scores are cosines: map vs its autoencoder reconstruction; text vs the nearest training abstract (SPECTER2). Flagged = below the held-out 5th percentile.
**Outputs** (`outputs/`): `a7_ood.{png,svg}` and `a7_ood_table.{csv,tex}` (from `flagged_cosine.csv`); every other score in `detection.csv` and `a7_ood_table_full.*`;
sparsity controls in `a7_ood_table_sparsity.*` and `sparsity_strata.csv`; per input `scores.parquet`; `metrics.json`.
**Result:** flagged: rotated 74 / 82 %, spun 89 %, shuffled 100 %, white noise 0 % (caught by the latent norm instead), map x 100 52 %, map x 0.01 100 %, WikiText and
recipes 100 %. In the shared contrastive space the same scores miss most of them (2-4 % of rotated or spun maps; 39 % of WikiText, 20 % of recipes).
