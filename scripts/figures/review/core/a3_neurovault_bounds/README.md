# NeuroVault: upper bound, lower bound, multi-target, map types (comments 1.3, 2.3, 2.8)
**Comment 1.3:** keep the model-selected image as an upper bound, add a target-independent evaluation, treat map types separately. **Goes to** the supplement.
```bash
.venv/bin/python scripts/figures/review/core/a3_neurovault_bounds/run_analysis.py --model-dir scripts/outputs/runs/final --device cpu --force --n-jobs 12 --qc-count-all    # 1 min
.venv/bin/python scripts/figures/review/core/a3_neurovault_bounds/make_figure.py                                                                                        # 2 s
```
**Inputs:** the final autoencoder and InfoNCE heads (`_final_model.py`), pinned NeuroVault texts, metadata and image tensor (`_hf_pins.py`), the raw NIfTIs that
`scripts/data/fetch_hf.py` unpacks into the data dir (`neurovault_images/`), leakage exclusions. NIfTI latents are cached in `outputs/cache/` (`--force` rebuilds them).
**Method:** 298 papers, 3,074 maps. Maps are re-read from the NIfTIs and branched by type (signed: positive tail; p-maps inverted to 1 - p; parcellations,
anatomical, variance maps and ROI masks excluded), then normalised as in the paper. The 480 maps (54 papers) whose uploaders chose the NeuroVault type
'other' are typed by `_other_maps.py` from the image and file names, descriptions, sibling maps and values (`outputs/other_maps.csv`; the name rules agree
with the declared T/Z/F type of 98.7 % of the declared maps they fire on, `outputs/other_maps_rule_check.csv`): 204 t, z or F statistics, 103 effect
estimates (con/beta images, Cohen's d, correlations, searchlight accuracy, overlap), 93 components or gradients, all kept with the positive tail; 70 mean or
variance images (SPM ResMS/RPV, mean PET tracer and BOLD images) and 10 maps that could not be typed are excluded like variance and anatomical maps
(`--other-maps infer`, default; 80 maps, 7 papers). `outputs/other_maps_options.csv` has the bounds with every 'other' map kept (the submission, 305 / 3,154)
or all excluded (267 / 2,674): inferring moves every AUC by <= 0.004, excluding all by up to 0.009. Upper bound = the map the model scores highest for the paper's text; lower bound
= the lowest; multi-target = every map is a target. Chance reruns the same rule with texts shuffled across papers (20 draws), so it is not 0.50.
**Outputs** (`outputs/`): `neurovault_bounds.{png,svg}`, `bounds_table.{csv,tex}`, `map_type_table.{md,csv,tex}`; `bounds_metrics.csv` and `recall_curves.csv`
hold every variant (also |value| and sign split); `counts_table.csv` traces the image counts from the scrape to the evaluated set (stage J: the 'other' maps);
`other_maps.csv`, `other_maps_rule_check.csv`, `other_maps_options.csv` document the typing of the 'other' maps.
**Result (AUC, chance):** text -> brain upper 0.847 (0.738), lower 0.397 (0.306), multi-target 0.866 (0.759); brain -> text 0.787 (0.698), 0.400 (0.318), 0.626
(0.509; 0.501 with 200 derangements). Per map 0.619 (0.502) and 0.626 (0.509). Fig. 5b reads variant `released`: the published map tensor with its 3 p-maps
(2 papers) rebuilt from the NIfTIs as 1 - p under the release threshold (99th percentile, clusters >= 50 voxels; `outputs/p_maps_inverted.csv`). Map 11576 is
labelled a p-map but is not one (5 % negative values, 79 % below 0.05) and keeps no cluster after the threshold, so it keeps its unthresholded 1 - p as in
`pos` (an empty map would drop out of every other paper's competitors and inflate the AUCs). Fig. 5b: upper 0.847 / 0.787, multi 0.866 / 0.626, lower
0.398 / 0.400; the tensor as published (`released_raw`, p-maps not inverted) gives 0.846 / 0.787, 0.865 / 0.626, 0.396 / 0.398. With every 'other' map
kept (`--other-maps keep`, the previous default) Fig. 5b read 0.845 / 0.785, 0.865 / 0.622, 0.398 / 0.401.
