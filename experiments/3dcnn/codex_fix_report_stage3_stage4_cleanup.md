# Stage 3/Stage 4 Cleanup — Final Integration Report

Scope: final integration review across Notebook 5, Notebook 5b, Notebook 6a, Stage 3
scripts, and corrected Stage 4 scripts, after the prior fix passes were merged into the
working tree.

## Issue-by-issue root cause

### 1. `legacy_specter2` must never produce "normalized"-named outputs
**Root cause found.** Directory-level naming (`stage3_legacy_specter`,
`corrected_stage4_legacy_specter`) was already correct. However,
`stage_checkpoint_path()` in `notebook_utils.py` returned
`NORMALIZED_STAGE3_CHECKPOINT` ("best_val_normalized_recall_auc.pt") for **any**
non-`None` convention, including `legacy_specter2`. Worse, that filename is never
actually produced by `train_ale_cnn.py` for *either* convention — the trainer always
writes `best_ale_cnn.pt`. The same stale constant was used in
`pipeline_outputs.py::detect_stage_status` for stage3, where it was the *only* signal
behind `checkpoints_in_export_zip`, causing a false "Checkpoint file is absent from
this export" warning in Notebook 6a's status report for both legacy and normalized
stage3 runs whenever training had in fact succeeded.

### 2. `normalized_specter2` must still produce "normalized"-named outputs
Confirmed working as expected — no change needed. `stage3_normalized_specter`,
`corrected_stage4_normalized_specter` directory names and the Stage 4 checkpoint name
(`best_val_generation_normalized_auc.pt`, which matches `train_text_to_brain.py`'s
`primary_checkpoint_metric` default) are correct and unaffected by this fix.

### 3–5. Notebook 6a `AE_BRANCH_MODE`
No issues found. Defaults to `"mixed_only"`
(`6a_normalized_specter_stage3_stage4_rerun.ipynb`, setup cell). `"mixed_and_specialized"`
and `"specialized_only"` are both valid values handled by
`ae_branch_specs()`/`AE_BRANCH_MODES` in `notebook_utils.py`.

### 6. Notebook 5 Stage 1B fine-tuning default
No issues found. `RUN_STAGE1B_FINETUNING = False` by default in
`5_stage1_autoencoder_ablation_multi_source.ipynb`; downstream cells gate on
`if RUN_STAGE1B_FINETUNING and results:`.

### 7. Stage 3/Stage 4 live log streaming
No issues found. The actual Stage 3 and Stage 4 training invocations in Notebook 6a
both go through `run_subprocess_streaming()` (Popen, line-buffered, `flush=True`). The
one `subprocess.run(..., capture_output=True)` call in Notebook 6a is an unrelated
`git clone`/setup helper, not a training invocation.

### 8. Expensive Stage 3/4 metrics gated by interval/final-only
No issues found. `train_ale_cnn.py` gates validation on `epoch % args.val_interval`;
`train_text_to_brain.py` gates the (expensive) generation-AUC metric on
`generation_auc_val_interval` (default every 5 epochs) and validation in general on
`val_interval`. `final_test_eval` defaults to `True` (`train_ale_cnn.py` argparse), so
final test-set evaluation is not disabled by this gating.

### 9. Notebook 5b evaluates all Stage 1A checkpoints before selecting a winner
No issues found. `stage1_checkpoint_evaluation.py::run_evaluation()` runs
`run_checkpoint_evaluation()` over every checkpoint for every recipe/variant
(`for checkpoint_row in [r for r in manifest if r["variant"] == variant]`) and only
afterwards calls `create_stage1a_selection()` on the full `comparison_rows` result.

## Files changed

- `experiments/3dcnn/atlas_free_cnn/pipeline_outputs.py` — `detect_stage_status()`,
  stage3 branch.
- `experiments/3dcnn/atlas_free_cnn/notebook_utils.py` — `stage_checkpoint_path()`.

## Exact fixes made

1. `pipeline_outputs.py::detect_stage_status` (stage3 branch): removed the
   convention-gated `checkpoint = .../NORMALIZED_STAGE3_CHECKPOINT` lookup that was
   used as the sole basis for `checkpoints_in_export_zip`. `checkpoints_in_export_zip`
   now always uses `legacy_checkpoint_ok`, which checks the real filenames
   (`best_contrastive.pt`, `best_ale_cnn.pt`) plus the legacy `NORMALIZED_STAGE3_CHECKPOINT`
   name for backward compatibility with any older exports that happen to use it. This
   removes the false "checkpoint missing" warning for both legacy and normalized
   stage3 runs.
2. `notebook_utils.py::stage_checkpoint_path` (stage3 branch): now always returns
   `best_ale_cnn.pt`, matching what `train_ale_cnn.py` actually writes, instead of
   branching on `convention is not None` to return the never-produced
   `best_val_normalized_recall_auc.pt`. This was the one path that could leak a
   "normalized"-named checkpoint reference even for `legacy_specter2`.

No changes were made to AE architecture, AE loss, Stage 3 contrastive
architecture/loss, corrected Stage 4 generative architecture, text cache contents, data
splits, or selected Stage 1 checkpoints.

## Tests / smoke checks run

- `python -m pytest tests/test_3dcnn_6a_cleanup.py tests/test_stage1_checkpoint_evaluation.py -q`
  → 17 passed.
- `python -m pytest tests/ -q -k "3dcnn or stage1 or pipeline"` → 23 passed, 193
  deselected.
- Manual code-path verification (grep + read) for all 9 checklist items, with direct
  confirmation against the actual checkpoint-writing code in `train_ale_cnn.py` and
  `train_text_to_brain.py` rather than relying on the naming constants alone.

## Results that need to be rerun

None of the existing trained checkpoints, metrics, or exports need to be regenerated.
The fix only changes how an already-correct checkpoint file (`best_ale_cnn.pt`) is
*detected* in the status report; it does not change what gets trained, saved, or
evaluated. Any Notebook 6a status report previously showing a spurious "Checkpoint
file is absent" warning for a completed stage3 run can be safely disregarded, or
regenerated by re-running the status-report cell.

## Stage 1 pretraining

Not required. No Stage 1 code, checkpoints, or selection logic was touched.

## Remaining risks / TODOs

- `NORMALIZED_STAGE3_CHECKPOINT` ("best_val_normalized_recall_auc.pt") is kept as a
  defensive fallback in `legacy_checkpoint_ok` for compatibility with any pre-existing
  export that might contain a file by that name, but no current code path writes it.
  It would be reasonable to remove the constant entirely in a future cleanup once it's
  confirmed no archived exports rely on it — left in place here to minimize blast
  radius.
- `discover_stage_outputs()` / the stage4 branch of `stage_checkpoint_path()` were not
  changed; they were verified correct (Stage 4's checkpoint name is genuinely
  metric-derived and matches actual trainer output for both conventions).

## Final checklist

```
[OK] legacy_specter2 outputs no longer say normalized
[OK] 6a defaults to mixed_only AE branches
[OK] 6a can optionally run mixed_and_specialized
[OK] notebook 5 can skip Stage 1B finetuning by default
[OK] Stage 3/4 epoch logs stream live
[OK] unnecessary training-time metrics are disabled or interval/final-only
[OK] 5b evaluates all Stage 1A recipe checkpoints before comparing recipe winners
[OK] no Stage 1 retraining required by these fixes
```
