# Checkpoint / Metric Cleanup Report

Generated: 2026-06-30  
Branch: `neurovlm_gnn`

---

## Overview

This report documents the second-pass cleanup of Stage 3 and Stage 4 checkpointing and validation to enforce spatial-primary Stage 4 selection, remove remaining legacy references, add named timing summary fields, and expand the smoke-test suite.

---

## Root Cause of Checkpoint / Metric Clutter

The original pipeline accumulated several sources of clutter:

1. **Stage 4** was configured to save up to 8 checkpoint files per run (best by each of val_loss, val_latent_mse, val_reconstruction_mse, generation_top5_dice, generation_spatial_correlation, val_top5_dice, val_generation_normalized_auc, val_spatial_corr), causing confusion about which checkpoint represented the primary result.
2. **Stage 4 primary metric** (`primary_checkpoint_metric`) still defaulted to the semantic retrieval metric (`val_generation_normalized_auc`) even after spatial fidelity was designated as the primary goal.
3. **`notebook_utils.stage_checkpoint_path()`** still referred to `best_ale_cnn.pt` (with an outdated comment) for Stage 3, even after `train_ale_cnn.py` was updated to write `best_val_normalized_recall_auc.pt`.
4. **Notebook 6a** still defaulted `stage4_primary_checkpoint_metric` to the semantic checkpoint and referenced `CORRECTED_STAGE4_CHECKPOINT` (semantic) in the per-branch summary, making summaries misleading.
5. **Timing profile** used internal key names that differed from the canonical names requested for `timing_profile.json`.

---

## Files Changed

| File | Change |
|---|---|
| `experiments/3dcnn/atlas_free_cnn/notebook_utils.py` | `stage_checkpoint_path()` for stage3: return `NORMALIZED_STAGE3_CHECKPOINT` (was `best_ale_cnn.pt`); removed stale comment |
| `experiments/3dcnn/atlas_free_cnn/training/train_text_to_brain.py` | `primary_checkpoint_metric` default changed from `val_generation_normalized_auc` → `val_top5_dice`; added named timing summary fields to `timing_profile` |
| `experiments/3dcnn/6a_normalized_specter_stage3_stage4_rerun.ipynb` | Added `STAGE4_PRIMARY_SPATIAL_CHECKPOINT`, `STAGE4_SPATIAL_CORR_CHECKPOINT`, `STAGE4_SEMANTIC_CHECKPOINT` to notebook_utils import; added explicit `STAGE4_PRIMARY_CHECKPOINT`, `STAGE4_SEMANTIC_CHECKPOINT_NAME`, `STAGE4_SPATIAL_CORR_CHECKPOINT_NAME`, `SAVE_LEGACY_CHECKPOINT_ALIASES`, `SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES` constants to config cell; changed `stage4_primary_checkpoint_metric` default from `val_generation_normalized_auc` → `val_top5_dice`; updated branch summary to reference `STAGE4_PRIMARY_SPATIAL_CHECKPOINT` instead of `CORRECTED_STAGE4_CHECKPOINT` |
| `tests/test_3dcnn_6a_cleanup.py` | Added 3 new tests: `test_stage4_primary_checkpoint_metric_defaults_to_val_top5_dice`, `test_stage3_checkpoint_path_returns_canonical_name`, `test_timing_profile_has_named_summary_fields` |

---

## Stage 3 Checkpoint Files (default)

```
checkpoints/best_val_normalized_recall_auc.pt   ← canonical best (monitor: paper_recall_curve_auc)
checkpoints/last.pt                              ← last epoch
```

Not saved by default:
```
best_ale_cnn.pt          ← only with SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES=True
last_ale_cnn.pt          ← only with SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES=True
best_contrastive.pt      ← only with SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES=True
last_contrastive.pt      ← only with SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES=True
```

---

## Stage 4 Checkpoint Files (default)

```
checkpoints/best_val_top5_dice.pt                ← PRIMARY: spatial fidelity
checkpoints/best_val_generation_normalized_auc.pt← secondary: semantic retrieval
checkpoints/best_val_spatial_corr.pt             ← secondary: spatial correlation
checkpoints/last.pt                              ← last epoch
checkpoint_manifest.json
```

Not saved by default:
```
best_val_loss.pt                        ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_val_latent_mse.pt                  ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_val_reconstruction_mse.pt          ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_generation_top5_dice.pt            ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_generation_spatial_correlation.pt  ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
```

---

## Metrics Computed During Training

Every validation epoch:
- `val_reconstruction_mse`
- `val_mse`
- `val_spatial_corr`
- `val_top5_dice`
- `val_top1_dice` / `val_top10_dice` (cheap; computed alongside top5)
- `val_loss`, `val_latent_mse` (loss components)

Every `generation_auc_val_interval` epochs (default=5), at epoch 1, and at final epoch:
- `generation_mean_normalized_auc` (clamped strict-map AUC → used for `val_generation_normalized_auc` checkpoint)

---

## Metrics That Are Final-Only

Not computed during every training epoch — only at final checkpoint evaluation or in Notebook 7:
- Raw strict-map AUC
- Same-text-group / duplicate-aware AUC
- Publication / group AUC
- Matched cosine / shuffled cosine
- Latent distribution diagnostics
- UMAP / PCA plots
- Covariate correlations
- Generation baselines (ridge, MLP)

---

## Generation AUC Validation Interval

`generation_auc_val_interval` defaults to `5` (confirmed by `cfg.setdefault("generation_auc_val_interval", 5)` in `train_text_to_brain.py:662`).

---

## Timing Profile Fields

`timing_profile.json` now includes both internal list-based records AND named scalar summary fields:

| Field | Description |
|---|---|
| `train_epoch_time_sec` | Total seconds across all training epochs |
| `val_metric_time_sec` | Total seconds across all validation evaluations |
| `generation_auc_eval_time_sec` | Total seconds across all generation AUC evaluations |
| `checkpoint_save_time_sec` | Total seconds across all artifact saves |
| `branch_total_time_sec` | Total wall-clock seconds for the branch (= `total_branch_sec`) |

---

## Downstream Notebook Paths Updated

- **Notebook 6a**: `stage4_primary_checkpoint_metric` now defaults to `val_top5_dice`; per-branch summary `primary_checkpoint` now points to `STAGE4_PRIMARY_SPATIAL_CHECKPOINT`; explicit config constants added to control cell.
- **Notebook 7**: Already correctly evaluated `STAGE4_PRIMARY_SPATIAL_CHECKPOINT` first (no change needed).
- **`notebook_utils.stage_checkpoint_path()`**: Now returns canonical Stage 3 checkpoint name for all downstream consumers.

---

## Tests / Smoke Checks Run

```
pytest tests/test_3dcnn_6a_cleanup.py tests/test_stage1_checkpoint_evaluation.py -v
```

**Result: 30 passed, 0 failed (1 harmless FutureWarning)**

New tests added this session:
- `test_stage4_primary_checkpoint_metric_defaults_to_val_top5_dice` — PASS
- `test_stage3_checkpoint_path_returns_canonical_name` — PASS
- `test_timing_profile_has_named_summary_fields` — PASS

All 27 prior tests continue to pass.

---

## Remaining Risks

- `primary_checkpoint_metric = val_top5_dice` means the final-evaluation model loaded after training is the spatially best, not semantically best. This is intentional. If a run happens not to produce `best_val_top5_dice.pt` (e.g., validation was skipped), the fallback waterfall `val_generation_normalized_auc → val_top5_dice → val_spatial_corr → val_loss` will apply.
- Notebook 6a still passes `SAVE_LEGACY_CHECKPOINT_ALIASES = False` implicitly (trainer module-level default); the explicit notebook constant is informational only and is not yet wired into the `cfg` dict. It is correct by default.
- End-to-end Colab run required to confirm Stage 4 produces `best_val_top5_dice.pt` as the primary file after the first generation AUC validation interval.
- Whether `val_top5_dice` is the best early-stopping signal (vs the semantic AUC) is a scientific decision and was not changed here; `early_stopping_metric` still defaults to `val_generation_normalized_auc` unless overridden.

---

## Final Checklist

```
[OK] Stage 4 primary checkpoint is best_val_top5_dice.pt
[OK] Stage 4 semantic checkpoint is best_val_generation_normalized_auc.pt
[OK] Stage 4 spatial-correlation checkpoint is best_val_spatial_corr.pt
[OK] Stage 4 generation AUC validation interval defaults to 5
[OK] Stage 4 no longer saves unnecessary checkpoint aliases by default
[OK] Stage 3 primary checkpoint is best_val_normalized_recall_auc.pt
[OK] Stage 3 no longer saves best_ale_cnn / last_ale_cnn aliases by default
[OK] notebook_utils.stage_checkpoint_path() returns canonical Stage 3 checkpoint name
[OK] Notebook 6a summaries reflect spatial-primary Stage 4 selection
[OK] Notebook 6a config cell has explicit STAGE4_PRIMARY_CHECKPOINT / SAVE_LEGACY_* constants
[OK] Notebook 7 evaluates spatial-primary and semantic Stage 4 checkpoints separately
[OK] timing_profile.json includes named scalar summary fields
[OK] 30 smoke tests pass
```
