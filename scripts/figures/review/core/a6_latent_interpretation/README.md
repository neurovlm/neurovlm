# What the latent dimensions encode (reviewer 3.5, first part)
```bash
D=scripts/figures/review/core/a6_latent_interpretation
.venv/bin/python $D/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force  # associations, GPU, ~1 min
.venv/bin/python $D/select_dimensions.py  # the six dimensions, CPU, ~1.5 min without outputs/cache
.venv/bin/python $D/make_figure.py        # figure + tables, CPU, ~15 s
```
Inputs: the final model (autoencoder, latents, text-to-brain head v2), pinned HF data and Cognitive Atlas (`_hf_pins.py`),
Neurosynth v7 in `$NEUROSYNTH_DIR` (default `~/nimare_data/neurosynth`; NiMARE `fetch_neurosynth(version="7")`), DiFuMo-512 (nilearn).
Method: each dimension is signed by its top text-side Cognitive Atlas term (r of the projected text latent with term
presence, counting a term only on its 5 % most associated dimensions, `--term-top-frac`; without this rule frequent terms
such as working memory or retrieval top dozens of unrelated dimensions); its decoded map (+2 sd minus -2 sd, halved, positive
pole) is correlated with that term's Neurosynth z-map. Shown: the 6 highest r with r >= 0.3, peak in the top 10 % of the Neurosynth
map, one per term, map |r| < 0.4 with those chosen (106 of 384 reach r >= 0.3). Panel b: r of each dimension's term profile, text
vs image latent, against dimensions shuffled. Outputs: `outputs/latent_interpretation.{png,svg}`, `outputs/table_*.csv`,
`outputs/{neurosynth_agreement,selected_dimensions_neurosynth}.csv`, `outputs/{associations.npz,results.json}`, `outputs/cache/`.
Result: 146 listening (r 0.65, superior temporal), 158 cognitive control (lateral prefrontal), 153 anticipation / reward
(striatum), 1 reading (left inferior frontal), 364 reward processing (ventral striatum, ventromedial prefrontal), 304 movement
(SMA, premotor; 0.41); b: 0.92 vs 0.00. `--term-top-frac 0` reproduces the earlier selection (4 anticipation; working memory; decision).
