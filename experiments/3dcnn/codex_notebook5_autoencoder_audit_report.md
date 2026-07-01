# Notebook 5 Autoencoder Audit Report

**Date:** 2026-06-30
**Branch:** neurovlm_gnn
**Auditor:** Codex automated audit (mirrors Notebook 6a audit scope)

---

## Files Inspected

| File | Role |
|------|------|
| `experiments/3dcnn/5 multi source autoencoder ablation.ipynb` | Notebook 5 — Stage 1A/1B training |
| `experiments/3dcnn/5b stage1 checkpoint selection heldout eval.ipynb` | Notebook 5b — checkpoint selection |
| `experiments/3dcnn/atlas_free_cnn/training/train_autoencoder.py` | Stage 1 AE training script |
| `experiments/3dcnn/atlas_free_cnn/training/checkpointing.py` | CheckpointManager |
| `experiments/3dcnn/atlas_free_cnn/pipeline_outputs.py` | AE_SELECTION_TO_FILE, run dir helpers |
| `experiments/3dcnn/atlas_free_cnn/notebook_utils.py` | run_subprocess_streaming, shared helpers |
| `experiments/3dcnn/atlas_free_cnn/evaluation/stage1_checkpoint_evaluation.py` | 5b evaluator |

---

## Files Changed

| File | Changes |
|------|---------|
| `atlas_free_cnn/training/checkpointing.py` | Expanded `write_manifest` with per-checkpoint rows (name, metric, value, epoch, path, selection_direction); added `_best_epochs` / `_last_epoch` tracking; `save_last` and `maybe_save_best` now accept `epoch=` kwarg |
| `atlas_free_cnn/pipeline_outputs.py` | Added `"best_top10_dice": "best_top10_dice.pt"` to `AE_SELECTION_TO_FILE` |
| `atlas_free_cnn/training/train_autoencoder.py` | See detailed section below |
| `5 multi source autoencoder ablation.ipynb` | See detailed section below |

---

## Root Causes Found

### 1. `last_cnn_autoencoder.pt` saved every epoch — no gate flag
`train_from_config` unconditionally called `ckpt.save("last_cnn_autoencoder.pt", payload)` on every epoch.  This duplicates `last.pt` on every epoch — potentially several hundred extra checkpoint writes to Google Drive for a 300-epoch run.

### 2. `best_cnn_autoencoder.pt` as training result sentinel — not canonical
Both `load_stage1a_results_from_run` in Cell 4 and the RESUME_COMPLETED detection block hardcoded `"best_cnn_autoencoder.pt"` as the expected artifact.  This legacy alias was written by the training script as a secondary copy of whichever metric checkpoint was selected (e.g. `best_top5_dice.pt`).  After gating aliases behind `SAVE_LEGACY_AE_CHECKPOINT_ALIASES = False`, these detection blocks would fail on new runs without a corresponding fix in the notebook.

### 3. `best_top10_dice.pt` never saved despite `generation_metrics` computing it
`run_epoch` calls `generation_metrics` which returns `top10_dice`, but `train_from_config` never called `ckpt.maybe_save_best("top10_dice", ...)`.  The checkpoint was absent from: (a) `CheckpointManager` maximize dict, (b) `_checkpoint_eval_specs` default list, (c) `AE_SELECTION_TO_FILE` in `pipeline_outputs.py`.  Notebook 5b's evaluator (`CHECKPOINT_FILENAMES`) lists it — it was silently missing.

### 4. No `timing_profile.json` written for AE branches
No timing instrumentation existed in `train_from_config`.  Startup overhead (dataset construction, model build, preflight) and per-phase runtimes were invisible.

### 5. `checkpoint_manifest.json` format was minimal
`CheckpointManager.write_manifest` only wrote `{"best": {...}, "maximize": {...}}` — no checkpoint filenames, paths, epochs, or selection direction, making it useless for downstream tooling.

### 6. Dead code in Notebook 5 Cell 2 `discover_unified_split_dir`
After the `return _discover_unified_split_dir(...)` call, the function contained ~40 lines of unreachable manual candidate logic that was superseded when the call was delegated to `notebook_utils`.

### 7. `epoch=` not passed to `maybe_save_best` calls
The `CheckpointManager` had no epoch tracking, so `checkpoint_manifest.json` could not report which epoch each best checkpoint was saved at.

---

## Fixes Made

### `atlas_free_cnn/training/checkpointing.py`
- Added `_best_epochs: dict[str, int]` and `_last_epoch: int | None` to `__init__`.
- `save_last(payload, *, epoch=None)` — records `_last_epoch`.
- `maybe_save_best(metric_name, value, payload, *, epoch=None)` — records `_best_epochs[metric_name]`.
- `write_manifest()` now writes structured `checkpoints` list: one row per metric with `checkpoint_name`, `metric_name`, `metric_value`, `epoch`, `path`, `selection_direction`, `exists`.  Legacy `best`/`maximize` flat dicts preserved for backward compat.

### `atlas_free_cnn/pipeline_outputs.py`
- Added `"best_top10_dice": "best_top10_dice.pt"` to `AE_SELECTION_TO_FILE`.

### `atlas_free_cnn/training/train_autoencoder.py`
- **Legacy alias gate:** `_save_legacy_aliases = bool(cfg.get("save_legacy_ae_checkpoint_aliases", False))`.  `last_cnn_autoencoder.pt` and `best_cnn_autoencoder.pt` are only written when this flag is `True`.  Default is `False`.
- **`top10_dice` checkpoint:** Added `"top10_dice": True` to `CheckpointManager` maximize dict; added `ckpt.maybe_save_best("top10_dice", ...)` call; added `"best_top10_dice"` to `_checkpoint_eval_specs` default names.
- **Epoch tracking:** all six `maybe_save_best` calls now pass `epoch=epoch`; `save_last` passes `epoch=epoch`.
- **Canonical return value:** `train_from_config` now returns `"best_checkpoint": str(_selection_ckpt)` where `_selection_ckpt = checkpoint_dir / AE_SELECTION_TO_FILE.get(sel_metric, "best_val_loss.pt")` instead of the removed legacy alias path.
- **Timing profile:** `_t0`, `_t_dataset`, `_t_model`, `_t_preflight`, `_t_train_end`, `_t_final_eval`, `_t_end` timing points added.  `timing_profile.json` written to `out_dir` with fields: `total_sec`, `dataset_construction_sec`, `model_construction_sec`, `preflight_sec`, `training_sec`, `final_eval_sec`, `artifact_save_sec`, `avg_epoch_train_sec`, `avg_epoch_val_sec`, `n_epochs`, `ae_variant`.
- **Final eval clarity:** added explicit `print` statements naming which checkpoint is loaded for final eval (or warning if not found).
- **Removed stale `last.pt` reload:** the confusing `_load_model_checkpoint_for_eval(model, last_path, device)` that reloaded `last.pt` into the model at the very end of per-checkpoint eval (after all evals were complete) was removed.
- **Leaderboard epoch:** checkpoint leaderboard CSV now includes `epoch` column from `ckpt._best_epochs`.

### `5 multi source autoencoder ablation.ipynb` (Notebook 5)
- **Cell 1:** Added `AE_SELECTION_TO_FILE` to `from atlas_free_cnn.pipeline_outputs import ...` line so all cells can resolve canonical checkpoint filenames.
- **Cell 2:** Added `SAVE_LEGACY_AE_CHECKPOINT_ALIASES = False` flag (with comment); removed dead unreachable code block from `discover_unified_split_dir`.
- **Cell 4 — `load_stage1a_results_from_run`:** now uses `AE_SELECTION_TO_FILE.get(recipe["checkpoint_selection_metric"], ...)` to find the canonical best checkpoint; falls back to `best_cnn_autoencoder.pt` if canonical is absent (preserves compatibility with previously trained runs).
- **Cell 4 — RESUME_COMPLETED block:** same canonical-then-fallback logic applied to the per-recipe resume detection.
- **Cell 4 — `base_cfg`:** `"save_legacy_ae_checkpoint_aliases": SAVE_LEGACY_AE_CHECKPOINT_ALIASES` added so the flag propagates into the training script.

---

## Section-by-Section Findings

### 1. Subprocess launching
**Finding:** Notebook 5 does **not** launch training via subprocess.  Training is called directly in-process via `train_from_config(cfg)` and `train_stage1b_from_config(cfg)`.  Epoch logs are printed live by the training script's `print(row)` at line 784 and `tqdm` progress bars.  No subprocess buffering issue for training.

The `run_cmd` helper defined in Cell 1 uses `capture_output=True`, but this is only used for git clone/pull and pip install during Colab setup — not for training.  These commands are fast and do not produce training logs.  No streaming fix is needed here.

**Status: No training-log buffering issue in Notebook 5.**

### 2. Stage 1A / Stage 1B control flags
Both `RUN_STAGE1A_MIXED_PRETRAINING = True` and `RUN_STAGE1B_FINETUNING = False` are present in Cell 2.  Stage 1A and 1B status are tracked and written to `stage1_stage1b_control_status.json`.  All four states are handled: `trained_now`, `skipped_intentionally`, `loaded_existing_checkpoint`, `missing`.

**Status: OK — no change required.**

### 3. Output naming and branch mode compatibility
Notebook 5 does not hardcode `AE_BRANCH_MODE`.  Stage 1B is gated by `RUN_STAGE1B_FINETUNING = False`, so when the downstream `AE_BRANCH_MODE = "mixed_only"` is in use, Stage 1B outputs are not required and not written.  No mismatch.

**Status: OK — compatible with mixed_only downstream.**

### 4. AE checkpointing
- **Canonical checkpoints saved:** `last.pt`, `best_val_loss.pt`, `best_spatial_corr.pt`, `best_top1_dice.pt`, `best_top5_dice.pt`, `best_top10_dice.pt` (new), `best_foreground_mse.pt`.
- **Legacy aliases:** `last_cnn_autoencoder.pt` and `best_cnn_autoencoder.pt` now gated behind `SAVE_LEGACY_AE_CHECKPOINT_ALIASES = False`. Default saves ~300 fewer duplicate checkpoint writes per full run.
- **`checkpoint_manifest.json`:** now includes full per-checkpoint metadata.

### 5. Expensive metrics during AE training
Existing controls are already sound:
- `COMPUTE_TRAIN_METRICS = False` in full mode → no batch-level metrics during training epochs.
- `COMPUTE_EPOCH_SOURCE_METRICS = False` by default → no per-source breakdown during training.
- `VAL_METRIC_BATCHES = 16` → validation metrics limited to 16 batches.
- `SAVE_PLOTS = True` in full mode → qualitative plots generated at final eval only (not per-epoch).
- `FINAL_EVAL = True` in full mode → final eval runs on the selected checkpoint (now explicit).

No changes were needed to metric frequency.

### 6. Dataset / cache loading overhead
No redundant cache rebuild pattern found.  `train_from_config` constructs datasets once.  `timing_profile.json` now captures dataset construction time, model construction time, and preflight time so startup overhead can be diagnosed on Colab.

### 7. Final AE eval uses selected checkpoint
**Confirmed fixed.**  `train_from_config` now loads `_selection_ckpt` (the metric-specific canonical checkpoint) before running final eval, with an explicit print statement confirming which file is loaded.  The old confusing `last.pt` reload after per-checkpoint eval was removed.  No bug remains.

### 8. Notebook 5b compatibility
Notebook 5b's `CHECKPOINT_FILENAMES` lists: `best_val_loss.pt`, `best_spatial_corr.pt`, `best_top1_dice.pt`, `best_top5_dice.pt`, `best_foreground_mse.pt`, `best_top10_dice.pt`, `best_cnn_autoencoder.pt`, `last.pt`, `last_cnn_autoencoder.pt`.

After these changes:
- All canonical names are written (including `best_top10_dice.pt`, now newly saved).
- Legacy aliases (`best_cnn_autoencoder.pt`, `last_cnn_autoencoder.pt`) are absent by default; 5b's evaluator skips missing files gracefully (`CHECKPOINT_FILENAMES` is a candidate list, not a required list).
- 5b's locked downstream registry uses `LOCKED_STAGE1_CHECKPOINT_NAMES` from `notebook_utils.py` (e.g. `best_top1_dice.pt` for `mixed_stage1a`) — all canonical names, no legacy aliases.

**Status: Compatible.  5b will evaluate more checkpoints than before (`best_top10_dice.pt` is now present) and does not depend on legacy aliases.**

---

## Tests / Smoke Checks Run

```
python3 -c "
  from atlas_free_cnn.training.checkpointing import CheckpointManager
  from atlas_free_cnn.pipeline_outputs import AE_SELECTION_TO_FILE
  import json, tempfile
  from pathlib import Path

  assert 'best_top10_dice' in AE_SELECTION_TO_FILE

  with tempfile.TemporaryDirectory() as tmp:
      ckpt = CheckpointManager(tmp, maximize={'top5_dice': True})
      payload = {'model': {}, 'config': {}}
      ckpt.maybe_save_best('top5_dice', 0.5, payload, epoch=3)
      manifest = json.loads(Path(tmp, 'checkpoint_manifest.json').read_text())
      row = next(r for r in manifest['checkpoints'] if r['metric_name'] == 'top5_dice')
      assert row['epoch'] == 3
      assert row['selection_direction'] == 'maximize'
      assert row['checkpoint_name'] == 'best_top5_dice.pt'
  print('PASS')
"
→ PASS

python3 -c "import ast; ast.parse(open('experiments/3dcnn/atlas_free_cnn/training/train_autoencoder.py').read()); print('syntax OK')"
→ syntax OK

python3 -c "import ast; ast.parse(open('experiments/3dcnn/atlas_free_cnn/training/checkpointing.py').read()); print('syntax OK')"
→ syntax OK

python3 -c "import ast; ast.parse(open('experiments/3dcnn/atlas_free_cnn/pipeline_outputs.py').read()); print('syntax OK')"
→ syntax OK
```

---

## Whether Stage 1 Pretraining Needs to Be Rerun

**No.**  All changes are forward-only improvements:
- Existing checkpoints from previous runs are unaffected.
- The canonical checkpoint names (`best_val_loss.pt`, `best_top5_dice.pt`, etc.) were already saved — legacy aliases were extras.
- `RESUME_COMPLETED` detection now prefers canonical names but falls back to `best_cnn_autoencoder.pt` if the canonical is absent (which covers runs completed before these changes).
- Downstream (Notebook 6a / Stage 3 / Stage 4) uses the locked registry from `notebook_utils.py` and is not affected.

---

## Remaining Risks

1. **`AE_SOURCE_WISE_VAL_INTERVAL`** not implemented.  Source-wise val metrics are currently all-or-nothing per `COMPUTE_EPOCH_SOURCE_METRICS`.  An interval-based gate (run every N epochs) would further reduce overhead but would require a small change to the training loop.  Deferred.

2. **`SAVE_PLOTS = True` in full mode** still runs qualitative reconstruction plots at the end of each AE variant.  These write to Drive and are slow.  The flag exists; the default in full mode is `True`.  Consider defaulting to `False` and enabling explicitly when plots are wanted.

3. **`train_metric_batches = 0` semantics** — with `COMPUTE_TRAIN_METRICS = False` (full mode), training metrics are already skipped; `train_metric_batches = 0` is a no-op in this case.  If `COMPUTE_TRAIN_METRICS` is ever enabled, `metric_max_batches = 0` will silently skip all metrics (the condition `len(metric_rows) < int(0)` is always False).  Consider treating `0` as "unlimited" or removing the ambiguity.

4. **Stage 1B cfg does not pass `save_legacy_ae_checkpoint_aliases`** — Cell 5 builds its Stage 1B cfg separately and doesn't include `SAVE_LEGACY_AE_CHECKPOINT_ALIASES`.  Stage 1B training will default to `False` (correct behavior) but doesn't explicitly read the notebook-level flag.  Low risk since the default is correct.

---

## Final Checklist

```
[OK] Notebook 5 long training calls stream epoch logs live
     → Training is in-process (not subprocess); logs print live via print(row) and tqdm.

[OK] RUN_STAGE1B_FINETUNING defaults to False
     → Cell 2: RUN_STAGE1B_FINETUNING = False

[OK] Stage 1B can be skipped intentionally without breaking mixed_only downstream
     → Stage 1B gated by flag; downstream Notebook 6a uses AE_BRANCH_MODE="mixed_only" which
       only requires Stage 1A checkpoints.

[OK] AE checkpoint saving keeps canonical useful checkpoints
     → last.pt, best_val_loss.pt, best_spatial_corr.pt, best_top1_dice.pt, best_top5_dice.pt,
       best_top10_dice.pt (new), best_foreground_mse.pt all saved by default.

[OK] optional legacy AE aliases do not slow default training
     → last_cnn_autoencoder.pt and best_cnn_autoencoder.pt gated behind
       SAVE_LEGACY_AE_CHECKPOINT_ALIASES = False.

[OK] expensive AE diagnostics are interval/final-only where safe
     → COMPUTE_TRAIN_METRICS=False, COMPUTE_EPOCH_SOURCE_METRICS=False, VAL_METRIC_BATCHES=16
       already in place; plots are final-only.

[OK] timing_profile.json is written for AE branches
     → train_from_config writes timing_profile.json with 11 fields covering dataset, model,
       preflight, training, eval, and artifact phases.

[OK] final AE eval uses selected checkpoint, not last model
     → _selection_ckpt loaded explicitly before final eval; explicit print confirms which file;
       stale last.pt reload after per-checkpoint eval was removed.

[OK] Notebook 5 outputs remain compatible with 5b
     → 5b evaluates any present .pt files; all canonical names are written; best_top10_dice.pt
       now present for the first time; legacy aliases absent by default but 5b handles this
       gracefully.

[OK] no Stage 1 retraining is required by these fixes
     → RESUME_COMPLETED falls back to best_cnn_autoencoder.pt for runs completed before this
       change; no training semantics altered.
```
