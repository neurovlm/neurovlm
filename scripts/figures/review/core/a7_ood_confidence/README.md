# Warning signal for out-of-distribution inputs (comment 1.4)
**Comment 1.4:** a small quantitative analysis of failure cases, out-of-distribution (OOD) inputs and an abstention or confidence signal. **Goes to** the supplement (one figure).
```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/figures/review/core/a7_ood_confidence/run_analysis.py --model-dir scripts/outputs/runs/final   # CPU, 3 min
.venv/bin/python scripts/figures/review/core/a7_ood_confidence/make_figure.py            # 2 s, letter width (6.5 in)
.venv/bin/python scripts/figures/review/core/a7_ood_confidence/make_figure.py --paper    # paper width (5.15 in), writes paper/figures/supp_ood.png
```
**Inputs:** the final run (`autoencoder.pt`, InfoNCE heads, `proj_head_text_mse.pt`, `map_calibration.json`, `latent_neuro.pt`, `pmids_split.pt`), frozen SPECTER2,
pinned maps, texts and WikiText-2 (`_hf_pins.py`), leakage exclusions; the surface spin uses the fsaverage-10k sphere from neuromaps.
**Method:** reference = training split (23,894 maps and abstracts); thresholds = validation split (2,961 maps and abstracts); held-out control = 2,971 test maps
and abstracts. OOD maps from 500 test maps (seed 7): rotated 90° or randomly, surface spin (each keeps >= 90 % of the source's non-zero voxels), voxel shuffle,
white noise, map x 100, map x 0.01; OOD text: 500 WikiText-2 paragraphs, 20 recipes.
**Detector:** a map is flagged when its Pearson correlation with its autoencoder reconstruction is below the validation 2.5th percentile (0.46) or the
Mahalanobis distance of its autoencoder latent from the training latents is above the validation 97.5th percentile (27.0); together 5 % of validation maps.
A text is flagged when the cosine of its SPECTER2 embedding to the nearest training abstract is below the validation 5th percentile (0.79).
**Why white noise passed before:** the previous map score was the plain cosine of the map and its reconstruction, thresholded at the test 5th percentile (0.55).
White noise is dense and non-negative, so it and its flat reconstruction share a large constant component (cosine with a constant map 0.80 and 0.78) and their
cosine stays at 0.63 although the pattern is unrelated (Pearson r 0.00). Centering the score (Pearson r) removes the constant; the latent distance adds
the intensity shifts that a correlation cannot see (map x 100). `outputs/diagnosis.csv` has the per-set distributions of the old and new scores.
**Outputs** (`outputs/`): `a7_ood.{png,svg}` and `a7_ood_table.{csv,tex}` (from `flagged.csv`, the detector per set, validation and test included);
every single score (flag rate at the validation 95th percentile, AUROC test vs OOD) in `detection.csv` and `a7_ood_table_full.*`; the map similarity with
sparsity controls in `a7_ood_table_sparsity.*` and `sparsity_strata.csv`; `diagnosis.csv`; per input `scores.parquet`; `metrics.json`;
`outputs/paper/supp_ood.png` = the `--paper` drawing (`--paper-dir outputs/paper`).
**Result:** flagged: validation maps 5 %, test maps 4 %, rotated 61 / 70 %, spun 80 %, shuffled, white noise, x 100 and x 0.01 100 %; test abstracts 5 %,
WikiText and recipes 100 %. AUROC (test vs OOD) 0.89 to 1.00. Previous score (test-split threshold): rotated 74 / 82 %, spun 89 %, shuffled 100 %, white noise
0 %, x 100 52 %, x 0.01 100 %. Single scores (`detection.csv`): the correlation alone flags x 100 44 %, the latent distance alone flags only noise and x 100;
the contrastive-space scores (nearest training text or map) flag at most 17 % of any map set, and the calibrated peak of the generated map flags 15 % of
WikiText and 20 % of recipes: within distribution it tracks map quality, it is not an OOD score.
