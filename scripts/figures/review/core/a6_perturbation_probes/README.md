# 3.5: text-to-brain under controlled edits (substitution, composition, negation)

Comment 3.5: "Perturbing the inputs (such as adding negations) could show how NeuroVLM goes beyond linear additivity." Reviewer letter: one figure.

```bash
.venv/bin/python scripts/figures/review/core/a6_perturbation_probes/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force  # GPU, ~20 s
.venv/bin/python scripts/figures/review/core/a6_perturbation_probes/make_figure.py  # figure + table, CPU
```

Inputs: the final run's text-to-brain head v2, autoencoder, latents and split (every generated map multiplied by the head's global calibration factor and clipped at 1); Neurosynth v7 term maps (`--neurosynth-dir`; z-map cache copied from `a1_text_to_brain_baselines`, which `run_all.sh` runs first); Harvard-Oxford ROIs (nilearn); word2vec (`--w2v-path`) and SPECTER2 for the ridge controls of the context probe.

Outputs: `outputs/perturbation_probes.{png,svg}` (a: one word swapped, as maps, each map's top 5 % shown; b: 5 word pairs × 8 sentences; c: 'A and B' vs the better single concept, 4 pairs × 4 frames; paired boxplots, gray control / blue matched, paired Wilcoxon), `outputs/table_negation_context.{md,csv}` (`make_figure.py`); per-probe csvs, `outputs/results.json`, `outputs/maps.npz`, `outputs/nifti/` (`run_analysis.py`).

Result: the difference map matches the Neurosynth contrast in 40 / 40 sentences (r 0.50 vs −0.10 for other pairs); 'A and B' covers both concepts better than either alone in 16 / 16 (Dice 0.77 vs 0.69). Negation is ignored: r = 0.94–1.00 between a sentence and its negation (0.85 for the word-order swap 'auditory but not visual'); 0.6 % of training abstracts negate an activation, 46 % affirm one.
