# a5_mlp_projector: MLP projector vs Q-Former (comment 2.1)

Reviewer 2.1: motivate the Q-Former and compare it with simpler alternatives (MLP projector, cross-attention).

From the repository root, with `PY=.venv/bin/python` and `A5=scripts/figures/review/core/a5_mlp_projector`:
- `$PY $A5/make_figure.py` (CPU, seconds): submitted Q-Former vs the final model's MLP projector at stage 1 and stage 2, from
  `scripts/outputs/validation/{released_excl,final}/metrics.json` (`scripts/pipeline/validate_models.py`) -> `outputs/a5_mlp_projector.{png,svg}`.
- `bash $A5/run_variants.sh` (GPU, ~7.5 h): controlled study on the released model's inputs; residual MLPs of 6 to 81 M parameters trained
  like `scripts/models/13_qformer_pubmed.py` and scored like the validator's `qformer_pubmed`, with the released Q-Former as reference ->
  `outputs/checkpoints/`, `eval/` (per variant), `metrics.json`, `table.csv`, `adoption.json` (`--aggregate-only` rebuilds these three from
  `eval/` on the CPU); it ends with `make_figure.py --table` -> `outputs/a5_mlp_projector_table.{csv,tex}`, `results_summary.md`.

The Q-Former cross-attends over two tokens (map latent and semantic vector), so an MLP over their concatenation can represent the same
mapping. PubMed test set without the leakage-audit papers (n = 2,971): study retrieval AUC 0.819 (Q-Former) vs 0.905 (parameter-matched
MLP-M, the final model's projector; MLP-XL ran out of GPU memory); figure stage 1 0.819 -> 0.896, stage 2 0.940 -> 0.954.
