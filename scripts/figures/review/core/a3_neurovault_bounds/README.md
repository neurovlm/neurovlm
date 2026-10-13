# NeuroVault: upper bound, lower bound, multi-target, map types (comments 1.3, 2.3, 2.8)
**Comment 1.3:** keep the model-selected image as an upper bound, add a target-independent evaluation, treat map types separately. **Goes to** the supplement.
```bash
.venv/bin/python scripts/figures/review/core/a3_neurovault_bounds/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force --n-jobs 12 --qc-count-all   # 1 min
.venv/bin/python scripts/figures/review/core/a3_neurovault_bounds/make_figure.py                                                                                        # 2 s
```
**Inputs:** the final autoencoder and InfoNCE heads (`_final_model.py`), pinned NeuroVault texts, metadata and image tensor (`_hf_pins.py`), the raw NIfTIs that
`scripts/data/fetch_hf.py` unpacks into the data dir (`neurovault_images/`), leakage exclusions. NIfTI latents are cached in `outputs/cache/` (`--force` rebuilds them).
**Method:** 305 papers, 3,154 maps. Maps are re-read from the NIfTIs and branched by type (signed: positive tail; p-maps inverted to 1 - p; parcellations,
anatomical, variance maps and ROI masks excluded), then normalised as in the paper. Upper bound = the map the model scores highest for the paper's text; lower bound
= the lowest; multi-target = every map is a target. Chance reruns the same rule with texts shuffled across papers (20 draws), so it is not 0.50.
**Outputs** (`outputs/`): `neurovault_bounds.{png,svg}`, `bounds_table.{csv,tex}`, `map_type_table.{md,csv,tex}`; `bounds_metrics.csv` and `recall_curves.csv`
hold every variant (also |value| and sign split); `counts_table.csv` traces the image counts from the scrape to the evaluated set.
**Result (AUC, chance):** text -> brain upper 0.845 (0.736), lower 0.397 (0.302), multi-target 0.865 (0.759); brain -> text 0.785 (0.697), 0.401 (0.314), 0.622
(0.503). Fig. 5b reads variant `released`: the published map tensor with its 3 p-maps (2 papers) rebuilt from the NIfTIs as 1 - p
under the release threshold (99th percentile, clusters >= 50 voxels; `outputs/p_maps_inverted.csv`). Map 11576 is labelled a p-map but is not one (5 %
negative values, 79 % below 0.05) and keeps no cluster after the threshold, so it keeps its unthresholded 1 - p as in `pos` (an empty map would drop out of
every other paper's competitors and inflate the AUCs). Fig. 5b: upper 0.845 / 0.785, multi 0.865 / 0.622, lower 0.398 / 0.401; the tensor as published
(`released_raw`, p-maps not inverted) gave 0.844 / 0.785, 0.864 / 0.622, 0.396 / 0.399.
