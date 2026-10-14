# Loss ablations of the projection heads (reviewer 3.5, last part)

The 09 heads are retrained on the final model's latents, 3 seeds each (seed 123 reproduces the final heads). Contrastive:
one-direction cross-entropy, tau 0.03 / 0.20 / learnable, no cross-entropy (cosine of matched pairs), random text-head
start. Text-to-brain: head v2 (final), submitted voxel-wise BCE, latent MSE, latent cosine. PubMed test set (n = 2,971).
```bash
.venv/bin/python scripts/figures/review/core/a5_loss_ablation/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force  # GPU, 8 min
.venv/bin/python scripts/figures/review/core/a5_loss_ablation/make_figure.py  # tables, CPU, < 1 s
```
Inputs: `scripts/outputs/runs/final`, pinned HF data (`_hf_pins.py`), `scripts/eval_exclusions/`.
Outputs: `outputs/a5_loss_ablation_table.{md,csv}` (letter table), `outputs/a5_loss_ablation_{contrastive,text_to_brain}.{csv,tex}`,
`outputs/metrics.json` (per seed; `reference` = final heads), heads in `outputs/checkpoints/final/`.
Result: retrieval AUC 0.833 with the final loss, 0.618 without cross-entropy, other variants within 0.011; text-to-brain
Dice 0.530 (head v2), 0.512 (latent MSE), 0.491 (submitted BCE), 0.385 (latent cosine). Canonical projection: `a2_granularity_tradeoff`.
