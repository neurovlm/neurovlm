# Checkpoint / Naming / Metric Verification Report

Generated: 2026-06-30  
Branch: `neurovlm_gnn`

---

## Overview

This report covers the pre-training-run checkpoint and naming verification for the
atlas-free 3D CNN pipeline (Stages 3 and 4).

---

## Findings and Fixes

### Stage 3 — Checkpoint Output

**Problem (pre-fix):**  
`train_ale_cnn.py::ALETrainer.save_best()` wrote `best_ale_cnn.pt`; `save_last()` wrote
`last_ale_cnn.pt`. Neither matched the canonical names defined in `notebook_utils.py`.  
The comment in `pipeline_outputs.py:detect_stage_status()` incorrectly stated
`best_val_normalized_recall_auc.pt` was _never_ produced by the trainer.

**Fixes applied:**
- `save_best()` → now writes `checkpoints/best_val_normalized_recall_auc.pt`
- `save_last()` → now writes `checkpoints/last.pt`
- Added module-level `SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES = False`. When `True`, also
  copies to `best_ale_cnn.pt`; with default `False` no aliases are created.
- `append_comparison_row()` checkpoint path updated to reference the canonical file.
- `main()` copy block updated; aliases are only written when the legacy flag is enabled.
- Fixed misleading comment in `pipeline_outputs.py:detect_stage_status()`.
- Detection priority reordered: canonical `best_val_normalized_recall_auc.pt` → legacy
  `best_ale_cnn.pt` → `best_contrastive.pt`.

**Current Stage 3 checkpoint files produced by default:**
```
checkpoints/best_val_normalized_recall_auc.pt   ← canonical best
checkpoints/last.pt                              ← canonical last
```

### Stage 4 — Checkpoint Output

**Problem (pre-fix):**  
`train_text_to_brain.py` called `ckpt.maybe_save_best()` for 8 metrics, producing
unnecessary files: `best_val_loss.pt`, `best_val_latent_mse.pt`,
`best_val_reconstruction_mse.pt`, `best_generation_top5_dice.pt`,
`best_generation_spatial_correlation.pt`.

**Fixes applied:**
- Added module-level `SAVE_LEGACY_CHECKPOINT_ALIASES = False`.
- `maybe_save_best` for `val_loss`, `val_latent_mse`, `val_reconstruction_mse`,
  `generation_top5_dice`, `generation_spatial_correlation` is now gated behind that flag.
- `CheckpointManager.maximize` dict cleaned up to match.

**Current Stage 4 checkpoint files produced by default:**
```
checkpoints/best_val_top5_dice.pt                ← primary spatial
checkpoints/best_val_generation_normalized_auc.pt← semantic/secondary
checkpoints/best_val_spatial_corr.pt             ← spatial correlation
checkpoints/last.pt                              ← last epoch
checkpoint_manifest.json
```

**Not produced by default:**
```
best_val_loss.pt                 ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_val_latent_mse.pt           ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_val_reconstruction_mse.pt   ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_generation_top5_dice.pt     ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
best_generation_spatial_correlation.pt ← only with SAVE_LEGACY_CHECKPOINT_ALIASES=True
```

### Stage 4 — Primary Checkpoint Policy

New constants added to `notebook_utils.py`:
```python
STAGE4_PRIMARY_SPATIAL_CHECKPOINT = "best_val_top5_dice.pt"
STAGE4_SPATIAL_CORR_CHECKPOINT    = "best_val_spatial_corr.pt"
STAGE4_SEMANTIC_CHECKPOINT        = "best_val_generation_normalized_auc.pt"
```
`CORRECTED_STAGE4_CHECKPOINT` remains `best_val_generation_normalized_auc.pt` for
backward compatibility (used in legacy detection paths).

### Notebook 7 — Multi-Checkpoint Stage 4 Evaluation

**Problem (pre-fix):**  
In `6a_normalized_corrected` mode, `stage4_checkpoint_rows_for_reg()` evaluated only
`CORRECTED_STAGE4_CHECKPOINT` (semantic). Spatial checkpoints were not evaluated.

**Fixes applied:**
- `stage4_checkpoint_rows_for_reg()` in `6a_normalized_corrected` mode now iterates over:
  1. `best_val_top5_dice.pt` (primary spatial)
  2. `best_val_generation_normalized_auc.pt` (semantic)
  3. `best_val_spatial_corr.pt` (spatial correlation)
  4. `last.pt`
  Deduplication by state checksum prevents re-evaluating identical weights.
- `STAGE4_CANDIDATE_NAMES` reordered: spatial primary first.
- Cell 1 imports updated: `LEGACY_NORMALIZED_STAGE4_DIRNAME` (broken) replaced with
  `CORRECTED_LEGACY_STAGE4_DIRNAME` (correct); `STAGE4_PRIMARY_SPATIAL_CHECKPOINT` and
  `STAGE4_SPATIAL_CORR_CHECKPOINT` added.

### Generation AUC Validation Interval

```python
cfg.setdefault("generation_auc_val_interval", 5)   # ← confirmed in train_text_to_brain.py:660
```
Verified: generation semantic AUC runs at epoch 1, every 5 epochs, and at final epoch.
Full raw/clamped/duplicate-aware diagnostics remain final-only (not every epoch).

### Convention-Aware Naming

Verified correct mapping:
| Convention | Stage 3 dir | Stage 4 dir |
|---|---|---|
| `normalized_specter2` | `stage3_normalized_specter` | `corrected_stage4_normalized_specter` |
| `legacy_specter2` | `stage3_legacy_specter` | `corrected_stage4_legacy_specter` |

Status detection works for both modes. Legacy directories do not appear when running in
normalized mode (verified by existing test).

### AE Branch Mode

| Mode | Expected branches |
|---|---|
| `mixed_only` | 3 (pubmed, nilearn, neurovault) |
| `mixed_and_specialized` | 6 |
| `specialized_only` | 3 |

Status reports respect `AE_BRANCH_MODE`; they no longer hardcode 6 branches when
`mixed_only` is selected.

---

## Files Changed

| File | Change |
|---|---|
| `experiments/3dcnn/atlas_free_cnn/training/train_ale_cnn.py` | `save_best()` → canonical name; `save_last()` → canonical name; `SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES = False`; comparison row; main() copy block |
| `experiments/3dcnn/atlas_free_cnn/training/train_text_to_brain.py` | `SAVE_LEGACY_CHECKPOINT_ALIASES = False`; removed legacy `maybe_save_best` calls; cleaned `maximize` dict |
| `experiments/3dcnn/atlas_free_cnn/pipeline_outputs.py` | Fixed Stage 3 detection comment; reordered priority |
| `experiments/3dcnn/atlas_free_cnn/notebook_utils.py` | Added `STAGE4_PRIMARY_SPATIAL_CHECKPOINT`, `STAGE4_SPATIAL_CORR_CHECKPOINT`, `STAGE4_SEMANTIC_CHECKPOINT` |
| `experiments/3dcnn/7_stage3_stage4_semantic_evaluation_diagnostics.ipynb` | Fixed imports; updated `STAGE4_CANDIDATE_NAMES`; updated `stage4_checkpoint_rows_for_reg()` |
| `tests/test_3dcnn_6a_cleanup.py` | Added 10 targeted tests (all passing) |

---

## Tests Run

```
pytest tests/test_3dcnn_6a_cleanup.py tests/test_stage1_checkpoint_evaluation.py -v
```

**Result: 27 passed, 0 failed (1 harmless FutureWarning)**

New tests:
- `test_stage3_canonical_checkpoint_name_is_best_val_normalized_recall_auc` — PASS
- `test_stage3_trainer_produces_canonical_checkpoint` — PASS
- `test_stage4_canonical_checkpoint_names` — PASS
- `test_stage4_trainer_produces_only_canonical_checkpoints` — PASS
- `test_stage3_detection_prioritises_canonical_over_legacy_aliases` — PASS
- `test_stage3_detection_accepts_legacy_alias_as_fallback` — PASS
- `test_stage4_primary_is_spatial_not_semantic` — PASS
- `test_generation_auc_val_interval_defaults_to_5` — PASS
- `test_convention_aware_stage3_checkpoint_names_do_not_mix` — PASS
- `test_mixed_only_ae_branch_mode_expects_exactly_3_branches` — PASS

---

## Remaining Risks

- `test_stage3_trainer_produces_canonical_checkpoint` uses a minimally constructed
  `ALETrainer` stub. A full end-to-end smoke run on GPU/Colab is still needed before
  treating Stage 3 output as production-ready.
- Stage 4 training on Colab should be verified to produce the 3 canonical checkpoint
  files after the first generation AUC validation interval (epoch 5 or earlier).
- Notebook 7 multi-checkpoint evaluation in `6a_normalized_corrected` mode is updated
  but has not been run end-to-end (requires completed Stage 4 checkpoints on Drive).

---

## Final Checklist

```
[OK] Stage 3 canonical checkpoint is best_val_normalized_recall_auc.pt
[OK] Stage 3 old aliases are optional fallback only (SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES=False)
[OK] Stage 4 primary checkpoint is best_val_top5_dice.pt
[OK] Stage 4 semantic checkpoint is best_val_generation_normalized_auc.pt
[OK] Stage 4 spatial checkpoint is best_val_spatial_corr.pt
[OK] Stage 4 saves no unnecessary checkpoint files by default
[OK] Notebook 7 evaluates Stage 4 checkpoint roles separately (spatial, semantic, corr, last)
[OK] Generation AUC validation interval defaults to 5
[OK] normalized_specter2 and legacy_specter2 directory names are correct
[OK] mixed_only AE branch mode expects 3 downstream branches
[OK] no Stage 1 retraining required
```
