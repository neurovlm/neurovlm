# 3.6: text-to-brain from network-only, anatomy-only or cognition-only text

Comment 3.6: "What if the text-to-brain outputs were generated using only text about the network, anatomy, or cognition?" Reviewer letter: one table.

```bash
.venv/bin/python scripts/figures/review/core/a6_modality_restricted_t2b/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force  # ~2 min (CPU also works)
.venv/bin/python scripts/figures/review/core/a6_modality_restricted_t2b/make_figure.py  # table, < 1 s
```

Inputs: the final run's text-to-brain head v2 and autoencoder (maps divided by their maximum); the pinned HF network atlases and Cognitive Atlas; Harvard-Oxford, AAL and Yeo-7 (nilearn); Neurosynth v7 (`--neurosynth-dir`; z-map cache copied from `a1_text_to_brain_baselines`, which `run_all.sh` runs first). Network names, region names and Cognitive Atlas terms (≥ 50 Neurosynth studies) are scored against the matching network map, ROI or Neurosynth term map; null = the same map against the non-matching references.

Outputs: `outputs/modality_table.{md,csv}` (the letter's table, `make_figure.py`); `outputs/per_item.csv`, `outputs/summary_by_{level,source}.csv`, `outputs/laterality.csv`, `outputs/results.json`, `outputs/predictions.npz`, `outputs/nifti/` (`run_analysis.py`).

Result: Dice 0.29 / 0.07 / 0.29 vs null 0.05 / 0.01 / 0.13 (network / anatomy / cognition), Pearson 0.25 / 0.15 / 0.24 vs ≤ 0.02. Anatomy Dice is low partly because a small ROI cannot fill the map's top 10 % of voxels (size-matched Dice 0.10 vs 0.01); hemisphere words are nearly ignored (named side favoured for 58 % of 122 lateralised names).
