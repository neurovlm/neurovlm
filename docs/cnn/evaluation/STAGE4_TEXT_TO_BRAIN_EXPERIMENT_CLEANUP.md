# Stage 4 text-to-brain experiment cleanup ledger

This file records the local, uncommitted Stage 4 text-to-brain experiment
work so it can be reviewed, archived, and removed safely after the
experiments are complete.

Do not use a broad `git clean`, `git reset --hard`, or recursive deletion for
this cleanup. Results may live outside Git, and some shared tracked edits may
be worth promoting instead of discarding.

## Snapshot

- Recorded: 2026-07-28
- Branch: `neurovlm_experiments`
- Base commit: `4b7cb7278e0fe75fd6bdd88b5317ff1a52b97078`
- Remote state at capture: `HEAD == origin/neurovlm_experiments`
- Intended clean baseline: the GitHub state at the base commit above
- Validation at capture: `476 passed, 1 skipped`
- Validation after the 2026-07-29 mixed-AE/resource update:
  `490 passed, 1 skipped`
- Local experiment result directories: neither `runs/` nor
  `neurovlm_runs/` currently exists

This ledger is tracked experiment-support documentation. Keep it until the
last cleanup step and review its final diff rather than deleting it with the
experiment implementation.

### 2026-07-29 experiment-scope and resource decision

All six notebooks were restricted to the released mixed Stage 1A AE:

- `mixed_to_pubmed`
- `mixed_to_nilearn`
- `mixed_to_neurovault`

The domain-finetuned Stage 1B AE branches were removed because their
reconstruction and contrastive prerequisites did not improve over the mixed
AE. This halves the former six-branch runs without removing any evaluation
domain.

The RTX PRO 6000 Blackwell 96-GB profile keeps original training batch sizes
to preserve optimization semantics, but uses larger evaluation batches,
eight Colab data-loader workers, four-batch prefetching, pinned/persistent
workers, and high float32 matrix-multiplication precision. The standardized
ablation additionally caches validation AE latents, keeps training latents on
the active device, avoids per-component batch synchronizations, and defers
nearest-reference latent distances until selected-checkpoint evaluation.

## Experiment inventory

### 1. Standardized latent ablation

Purpose: compare raw, standardized, cosine-augmented, whitening, and PCA
latent targets while keeping the released Stage 1 AE frozen.

Files:

- `docs/cnn/evaluation/stage4_standardized_latent_ablation.ipynb`
- `src/neurovlm/experiments/stage4_latent_ablation.py`
- `tests/test_stage4_latent_ablation.py`

Shared dependencies:

- `src/neurovlm/evaluation/text_to_brain_audit.py`
- the tracked provenance, dataset, collator, CNN-freezing, and Stage 4
  training changes listed below

Expected output root:

- Colab: `/content/drive/MyDrive/neurovlm/stage4_standardized_latent_ablation`
- Local: `runs/stage4_standardized_latent_ablation`

There is no notebook-builder source for this notebook in the current
worktree.

### 2. Sparse spatial-loss ablation

Purpose: compare sparse-map spatial losses and their gradient contributions
against the dense baseline.

Files:

- `docs/cnn/evaluation/stage4_sparse_spatial_loss_ablation.ipynb`

Shared dependencies:

- `src/neurovlm/experiments/stage4_latent_ablation.py`
- `src/neurovlm/evaluation/text_to_brain_audit.py`
- the tracked shared changes listed below

Expected output root:

- Colab: `/content/drive/MyDrive/neurovlm/stage4_sparse_spatial_loss_ablation`
- Local: `runs/stage4_sparse_spatial_loss_ablation`

There is no dedicated builder, experiment module, or notebook test for this
notebook in the current worktree.

### 3. Projector architecture and optimization sweep

Purpose: compare retained, wider, deeper, normalized, residual, and gated
projectors plus optimizers/schedulers while preserving the raw AE-latent
target.

Files:

- `docs/cnn/evaluation/stage4_projector_architecture_optimization_sweep.ipynb`
- `src/neurovlm/experiments/stage4_projectors.py`
- `tests/test_stage4_projectors.py`
- `tests/test_stage4_projector_sweep_notebook.py`

Shared dependencies:

- `src/neurovlm/experiments/stage4_latent_ablation.py`
- `src/neurovlm/evaluation/text_to_brain_audit.py`
- the tracked shared changes listed below

Expected output root:

- Colab:
  `/content/drive/MyDrive/neurovlm/stage4_projector_architecture_optimization_sweep`
- Local: `runs/stage4_projector_architecture_optimization_sweep`

Important: the IDE listed
`docs/cnn/evaluation/build_stage4_projector_architecture_optimization_sweep.py`,
and an ignored compiled `.pyc` exists, but the builder source itself was not
present on disk when this ledger was captured. If the open editor buffer is
later saved, add that source file to this experiment's cleanup list.

### 4. Joint AE/projector fine-tuning

Purpose: test carefully scoped joint adaptation of the projector and selected
Stage 1 AE components while retaining an untouched released AE reference.

Files:

- `docs/cnn/evaluation/stage4_joint_ae_projector_finetuning.ipynb`
- `docs/cnn/evaluation/build_stage4_joint_ae_projector_finetuning.py`
- `src/neurovlm/experiments/stage4_joint_finetuning.py`
- `tests/test_stage4_joint_finetuning.py`
- `tests/test_stage4_joint_finetuning_notebook.py`

Shared dependencies:

- `src/neurovlm/experiments/stage4_latent_ablation.py`
- `src/neurovlm/evaluation/text_to_brain_audit.py`
- the tracked shared changes listed below

Expected output root:

- Colab:
  `/content/drive/MyDrive/neurovlm/stage4_joint_ae_projector_finetuning`
- Local: `runs/stage4_joint_ae_projector_finetuning`

### 5. Stage 3 semantic bridge

Purpose: determine whether Stage 3 semantic representations improve
text-to-AE-latent generation over the direct Stage 4 path.

Files:

- `docs/cnn/evaluation/stage4_stage3_semantic_bridge.ipynb`
- `docs/cnn/evaluation/build_stage4_stage3_semantic_bridge.py`
- `src/neurovlm/experiments/stage4_semantic_bridge.py`
- `tests/test_stage4_semantic_bridge.py`
- `tests/test_stage4_semantic_bridge_notebook.py`

Shared dependencies:

- `src/neurovlm/experiments/stage4_latent_ablation.py`
- `src/neurovlm/evaluation/text_to_brain_audit.py`
- the tracked shared changes listed below

Expected output root:

- Colab:
  `/content/drive/MyDrive/neurovlm_runs/stage4_stage3_semantic_bridge/<commit-prefix>`
- Local: `neurovlm_runs/stage4_stage3_semantic_bridge/<commit-prefix>`

### 6. Probabilistic latent generation

Purpose: test whether a conditional VAE represents multiple plausible Stage 1
AE latents and reduces deterministic conditional-mean collapse.

Files:

- `docs/cnn/evaluation/stage4_probabilistic_latent_generation.ipynb`
- `docs/cnn/evaluation/build_stage4_probabilistic_latent_generation.py`
- `src/neurovlm/experiments/stage4_probabilistic.py`
- `tests/test_stage4_probabilistic.py`
- `tests/test_stage4_probabilistic_notebook.py`

Shared dependencies:

- `src/neurovlm/experiments/stage4_latent_ablation.py`
- `src/neurovlm/evaluation/text_to_brain_audit.py`
- the tracked shared changes listed below

Expected output root:

- Colab:
  `/content/drive/MyDrive/neurovlm/stage4_probabilistic_latent_generation`
- Local: `runs/stage4_probabilistic_latent_generation`

## Shared untracked experiment infrastructure

These paths support multiple notebooks. Remove them only after all six
experiments are retired:

- `src/neurovlm/evaluation/text_to_brain_audit.py`
- `src/neurovlm/experiments/__init__.py`
- `src/neurovlm/experiments/stage4_latent_ablation.py`
- `src/neurovlm/experiments/stage4_projectors.py`
- `src/neurovlm/experiments/stage4_joint_finetuning.py`
- `src/neurovlm/experiments/stage4_semantic_bridge.py`
- `src/neurovlm/experiments/stage4_probabilistic.py`
- `tests/test_stage4_mixed_ae_resource_profiles.py`

`src/neurovlm/experiments/__init__.py` connects the experiment modules and
should be removed last. The entire `src/neurovlm/experiments/` directory is
currently untracked, but use the explicit file list rather than assuming that
will remain true.

## Shared tracked edits relative to the base commit

These files existed at the base commit and were modified locally. A future
cleanup must review their diffs before restoring them because some changes are
general correctness improvements that might deserve a separate production
commit.

| Path | Current experiment-related change |
| --- | --- |
| `docs/cnn/evaluation/STAGE4_TEXT_TO_BRAIN_EXPERIMENT_CLEANUP.md` | Records the experiment-only inventory, mixed-AE decision, resource profile, and safe eventual cleanup procedure. |
| `src/neurovlm/atlas_free_dataset.py` | Exposes immutable dataset/tensor indices and split identity to experiment loaders. |
| `src/neurovlm/atlas_free_text.py` | Exposes text-cache row indices and collates primary text, cache, dataset, tensor, and split identities. |
| `src/neurovlm/cnn.py` | Forces the Stage 1 AE to remain frozen/eval when the parent Stage 4 model enters training mode. |
| `src/neurovlm/evaluation/__init__.py` | Exports the untracked text-to-brain audit helpers. |
| `src/neurovlm/pipelines/__init__.py` | Exports `sha256_state_dict`. |
| `src/neurovlm/pipelines/provenance.py` | Adds deterministic tensor-state SHA256 hashing. |
| `src/neurovlm/pipelines/serialization.py` | Canonicalizes string/int/bool subclasses to exact JSON primitives so PyTorch 2.6+ weights-only checkpoint loading does not reject `TorchVersion` metadata. |
| `src/neurovlm/training/text_to_brain.py` | Adds strict AE/text-cache provenance, checkpoint validation, loss-component logging, projector gradient diagnostics, and stronger AE freezing. |
| `tests/test_cnn_text_to_brain_training.py` | Tests the new audit, provenance, freezing, gradient, checkpoint, and resume behavior. |
| `tests/test_pipelines.py` | Tests deterministic state-dict checksums and exact built-in-string canonicalization for PyTorch version metadata. |

Tracked deletion requiring a separate decision:

- `requirements_classification.txt`

That filename and content concern a classification pipeline, not these Stage 4
text-to-brain experiments. Its deletion is therefore **ambiguous**. Restore it
to the base commit during experiment cleanup unless there is an independent,
documented reason to delete it.

## Ignored files and IDE-only observations

Currently ignored:

- `docs/cnn/evaluation/__pycache__/`
- `src/neurovlm/experiments/__pycache__/`
- `tests/__pycache__/`
- `docs/cnn/evaluation/artifacts/`

The existing ignored `docs/cnn/evaluation/artifacts/contrastive_retrieval/`
directory is not part of this six-notebook inventory. Do not delete it as part
of Stage 4 cleanup without a separate review.

The IDE tabs referenced
`docs/cnn/evaluation/stage4_correctness_audit/README.md` and
`docs/cnn/evaluation/stage4_correctness_audit/artifacts/train_val_latent_summary.csv`,
but neither file was present on disk at capture time. The CSV pattern is
globally ignored. Re-inventory this path if those editor buffers are saved or
new audit artifacts are produced.

## Safe cleanup procedure after experiments finish

1. Archive every selected checkpoint, provenance file, effective
   configuration, metrics table, generated example, plot, and final report
   from Drive or local output roots.
2. Record the scientific conclusions and which experiment, if any, should be
   promoted into production.
3. Re-run `git status --short` and compare it with this ledger. Do not assume
   that files created after 2026-07-28 are disposable.
4. Decide whether the shared tracked correctness improvements should:
   - be promoted in a separate reviewed commit; or
   - be restored to base commit `4b7cb72`.
5. Remove notebook-specific untracked files using the explicit experiment
   lists above.
6. Remove shared experiment modules/tests only after no retained code imports
   them.
7. If no shared tracked change is being promoted, restore only the explicitly
   listed tracked files from the base commit.
8. Restore `requirements_classification.txt` unless its deletion was
   independently intended.
9. Remove this ledger last.
10. Run the full test suite and confirm `git status --short` is empty.

## Candidate future commands

These are documentation, not commands that were run. Review the worktree and
make a patch/archive before using them.

To inspect the exact tracked patch:

```bash
git diff -- \
  requirements_classification.txt \
  src/neurovlm/atlas_free_dataset.py \
  src/neurovlm/atlas_free_text.py \
  src/neurovlm/cnn.py \
  src/neurovlm/evaluation/__init__.py \
  src/neurovlm/pipelines/__init__.py \
  src/neurovlm/pipelines/provenance.py \
  src/neurovlm/pipelines/serialization.py \
  src/neurovlm/training/text_to_brain.py \
  tests/test_cnn_text_to_brain_training.py \
  tests/test_pipelines.py
```

If, after review, every tracked change above should be discarded:

```bash
git restore --source=4b7cb7278e0fe75fd6bdd88b5317ff1a52b97078 -- \
  requirements_classification.txt \
  src/neurovlm/atlas_free_dataset.py \
  src/neurovlm/atlas_free_text.py \
  src/neurovlm/cnn.py \
  src/neurovlm/evaluation/__init__.py \
  src/neurovlm/pipelines/__init__.py \
  src/neurovlm/pipelines/provenance.py \
  src/neurovlm/pipelines/serialization.py \
  src/neurovlm/training/text_to_brain.py \
  tests/test_cnn_text_to_brain_training.py \
  tests/test_pipelines.py
```

Do not run the restore command if later desired work overlaps those files.
Untracked experiment files require separate, explicit removal; `git restore`
does not remove them.

## Final-clean criteria

- Experiment results and conclusions have been archived.
- No retained source imports `neurovlm.experiments` or
  `neurovlm.evaluation.text_to_brain_audit`.
- All six notebook groups have an explicit keep/delete decision.
- Shared tracked correctness edits have an explicit promote/revert decision.
- `requirements_classification.txt` has an explicit restore/delete decision.
- The full test suite passes.
- `git status --short` is empty, or contains only deliberately retained work
  documented in a new ledger.
