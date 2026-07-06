# MLP vs CNN NeuroVLM Model Comparison Plan

This is a staged execution plan for comparing the current NeuroVLM MLP baseline from `main` against the atlas-free CNN experiment branch. Each part is designed to be run in a brand new Codex session so the session can focus on one bounded job with fresh context.

The intended high-level order is:

1. Update this branch with the latest `main` changes and resolve conflicts.
2. Add a small model/evaluation registry so MLP and CNN models can be loaded consistently.
3. Compare autoencoder reconstruction quality for MLP AE vs CNN AEs.
4. Compare contrastive text-to-brain and brain-to-text retrieval for MLP vs CNN contrastive models.
5. Compare generative text-to-brain maps for MLP vs CNN text-to-brain models.
6. Aggregate results into tables and a compact report.

Terminology used here:

- `MLP` means the packaged NeuroVLM baseline on `main`: `NeuroAutoEncoder` plus the Hugging Face projection heads loaded by `src/neurovlm/retrieval_resources.py`.
- `CNN mixed AE` means the Stage 1A mixed-source atlas-free CNN autoencoder.
- `CNN specialized AEs` means the Stage 1B PubMed-, Nilearn-, and NeuroVault-finetuned autoencoders.
- `CNN contrastive` means Stage 3 contrastive CNN models using the atlas-free CNN brain encoder and text projection head.
- `CNN text-to-brain` means Stage 4 text-to-brain generation models.
- `NeuroVault` is used for the third dataset. If a prompt says `neurovlm` as a dataset, treat it as the intended `neurovault` dataset unless the code clearly says otherwise.

## Ground Rules For All Sessions

- Keep local artifacts safe. `git status --short` currently shows an untracked cache directory at `experiments/3dcnn/atlas_free_cnn/cache/`; do not stage large cache files.
- Prefer `origin/main` after `git fetch` over stale local `main`.
- Preserve the CNN-specific additions in this branch, especially:
  - `src/neurovlm/gnn/ale_cnn.py`
  - `experiments/3dcnn/atlas_free_cnn/`
  - CNN HF loaders in `src/neurovlm/retrieval_resources.py`: `_load_mixed_ae`, `_load_pubmed_finetuned_ae`, `_load_nilearn_finetuned_ae`, `_load_neurovault_finetuned_ae`
- Adopt the `main` metric split:
  - `src/neurovlm/retrieval_metrics.py`
  - `src/neurovlm/text_to_brain_metrics.py`
  - `src/neurovlm/brain_to_text_metrics.py`
  - `src/neurovlm/metric_utils.py`
  - keep `src/neurovlm/metrics.py` as the backward-compatible import index.
- Use the same metric names and semantics as `main` where possible: normalized Recall@k AUC, Pearson/Spearman minus random, Dice percentile, spin p-values when dependencies are available, and generated-text normalized Recall@k AUC.
- Do not run full expensive evaluations unless explicitly requested. Add smoke-test CLI flags like `--limit 8`, `--device cpu`, and `--skip-spin`.
- Use `PYTHONPATH=src:experiments/3dcnn` for commands that import both `neurovlm` and `atlas_free_cnn`.

## Part 1: Update Branch With Main

Goal: merge latest `main` into this experiment branch and leave the codebase importable.

Expected conflict areas:

- `src/neurovlm/retrieval_resources.py`: keep `main`'s newer MLP HF repo constants and loaders, while preserving this branch's atlas-free CNN dataset loaders and four CNN AE loaders.
- `src/neurovlm/metrics.py`: prefer `main`'s split metric modules and import-index pattern.
- `src/neurovlm/core.py`: prefer `main` public API updates unless they directly break the CNN branch; keep any experiment-specific compatibility only if tests or imports require it.
- `tests/`: preserve branch tests for CNN while accepting `main` tests around metric split and retrieval behavior.

Prompt for a fresh Codex session:

```text
We are in /Users/borng/code/lab_work/neurovlm on branch neurovlm_experiments. Update this branch with the latest main and resolve conflicts carefully.

Steps:
1. Inspect `git status --short` and protect untracked caches/artifacts.
2. Run `git fetch origin main`.
3. Merge `origin/main` into the current branch.
4. Resolve conflicts by keeping main's current NeuroVLM MLP API, Hugging Face loaders, metric split modules, and docs/tests, while preserving all atlas-free CNN experiment code and CNN loaders.
5. In `retrieval_resources.py`, make sure both the main MLP loaders and the CNN loaders coexist:
   - main MLP loaders/constants for NeuroAutoEncoder, projection heads, QFormer/adapter if present
   - atlas-free CNN dataset loaders
   - `_load_mixed_ae`
   - `_load_pubmed_finetuned_ae`
   - `_load_nilearn_finetuned_ae`
   - `_load_neurovault_finetuned_ae`
6. Run focused tests:
   `PYTHONPATH=src:experiments/3dcnn pytest tests/test_metrics.py tests/test_core.py tests/test_ale_cnn.py tests/test_retrieval_resources_ale_cache.py`
7. If imports fail because metric modules from main are missing, restore the main versions of `retrieval_metrics.py`, `text_to_brain_metrics.py`, `brain_to_text_metrics.py`, and `metric_utils.py`.
8. Summarize conflicts resolved, files changed, tests run, and any remaining failures.

Do not stage untracked cache files. Do not delete local experiment outputs.
```

Exit criteria:

- `git status --short` shows only intentional source/docs/test changes plus existing untracked caches.
- `python -c "import neurovlm; from neurovlm.metrics import bidirectional_retrieval_metrics, pearson_correlation"` works with `PYTHONPATH=src:experiments/3dcnn`.
- Focused tests pass or failures are documented as dependency/data-cache related.

## Part 2: Add Model Registry And Shared Evaluation Adapters

Goal: create a small code layer that lets later scripts compare MLP and CNN models without duplicating loader logic.

Proposed files:

- `experiments/3dcnn/atlas_free_cnn/evaluation/model_comparison_registry.py`
- `experiments/3dcnn/atlas_free_cnn/evaluation/model_comparison_adapters.py`
- `tests/test_3dcnn_model_comparison_registry.py`

Registry responsibilities:

- Define model IDs:
  - `mlp_neurovlm`
  - `cnn_ae_mixed`
  - `cnn_ae_pubmed`
  - `cnn_ae_nilearn`
  - `cnn_ae_neurovault`
  - `cnn_contrastive_mixed`
  - `cnn_contrastive_pubmed`
  - `cnn_contrastive_nilearn`
  - `cnn_contrastive_neurovault`
  - `cnn_t2b_mixed`
  - `cnn_t2b_pubmed`
  - `cnn_t2b_nilearn`
  - `cnn_t2b_neurovault`
- Resolve checkpoint paths from either:
  - Hugging Face loaders in `retrieval_resources.py`
  - existing Stage 3/Stage 4 run manifests under `experiments/3dcnn/atlas_free_cnn/outputs/`
  - an explicit JSON manifest supplied by `--manifest`
- Write a resolved manifest to `outputs/model_comparison/model_registry_resolved.json`.

Adapter responsibilities:

- MLP AE:
  - load via `_load_autoencoder()` and `_load_masker()`
  - encode flat masker vectors with `autoencoder.encoder`
  - decode latents with `autoencoder.decoder`
- CNN AE:
  - load via `_load_mixed_ae()` and specialized loaders
  - encode/decode 3D atlas-free volumes in native tensor space
- MLP contrastive:
  - use MLP AE encoder plus `_proj_head_image_infonce()`
  - use `_proj_head_text_infonce()` for SPECTER2 text embeddings
- CNN contrastive:
  - use `atlas_free_cnn.evaluation.stage4_semantic.load_stage3_evaluator`
  - expose `encode_brain_to_shared(volume_batch)` and `encode_text_to_shared(text_embedding_batch)`
- MLP text-to-brain:
  - use `NeuroVLM.text(...).to_brain(head="mse")`
- CNN text-to-brain:
  - use Stage 4 projector loading helpers from `stage4_semantic.py`
  - generate atlas-free volumes, then evaluate in native CNN volume space and shared contrastive space.

Prompt for a fresh Codex session:

```text
Implement the model comparison registry/adapters for MLP vs atlas-free CNN evaluation.

Read:
- `src/neurovlm/retrieval_resources.py`
- `src/neurovlm/core.py`
- `experiments/3dcnn/atlas_free_cnn/evaluation/stage4_semantic.py`
- `experiments/3dcnn/atlas_free_cnn/notebook_utils.py`
- `experiments/3dcnn/atlas_free_cnn/pipeline_outputs.py`

Add:
- `experiments/3dcnn/atlas_free_cnn/evaluation/model_comparison_registry.py`
- `experiments/3dcnn/atlas_free_cnn/evaluation/model_comparison_adapters.py`
- focused tests for registry resolution and adapter shapes using tiny fake models/tensors where possible.

Do not download HF assets in unit tests. Use monkeypatching for loader functions. Keep long-running evaluation out of tests.

The registry must handle the four CNN AE variants and the four CNN contrastive/T2B variants. If a Stage 3 or Stage 4 checkpoint cannot be auto-discovered, return a clear missing-checkpoint status in the resolved manifest instead of crashing during registry construction.

Run:
`PYTHONPATH=src:experiments/3dcnn pytest tests/test_3dcnn_model_comparison_registry.py`
```

Exit criteria:

- Registry can produce a JSON-serializable model inventory.
- Unit tests cover missing checkpoint handling and fake-loader shape paths.
- No real HF/network access is required for tests.

## Part 3: AE Reconstruction Comparison

Goal: compare CNN AE variants against the NeuroVLM MLP autoencoder on held-out PubMed, Nilearn, and NeuroVault maps.

Models:

- `mlp_neurovlm`
- `cnn_ae_mixed`
- `cnn_ae_pubmed`
- `cnn_ae_nilearn`
- `cnn_ae_neurovault`

Datasets:

- PubMed test split from the atlas-free CNN packed dataset and/or main PubMed test set.
- Nilearn test split from atlas-free CNN packed dataset.
- NeuroVault test split from atlas-free CNN packed dataset.

Metrics:

- Native reconstruction MSE/MAE.
- Foreground MSE.
- Spatial correlation.
- Top-1%, top-5%, top-10% Dice/overlap.
- Voxel AUROC when meaningful.
- For MLP flatmaps, also preserve `compute_ae_performance` / bits-per-pixel metrics from `main`.
- Where both models can be projected to the same comparison space, include common-space Pearson/Spearman and Dice; otherwise report native-space metrics and mark `comparison_space`.

Proposed file:

- `experiments/3dcnn/atlas_free_cnn/evaluation/compare_ae_reconstruction.py`

Outputs:

- `outputs/model_comparison/ae_reconstruction_by_sample.csv`
- `outputs/model_comparison/ae_reconstruction_summary.csv`
- `outputs/model_comparison/ae_reconstruction_summary.json`

Prompt for a fresh Codex session:

```text
Implement AE reconstruction comparison for NeuroVLM MLP vs CNN AEs.

Use the registry/adapters from Part 2. Reuse existing metric functions:
- `atlas_free_cnn.evaluation.generation_metrics.generation_metrics`
- `neurovlm.metrics.compute_ae_performance`
- `neurovlm.metrics.pearson_correlation`
- `neurovlm.metrics.dice_percentile`

Add a CLI:
`PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_ae_reconstruction --datasets pubmed nilearn neurovault --models mlp_neurovlm cnn_ae_mixed cnn_ae_pubmed cnn_ae_nilearn cnn_ae_neurovault --limit 16 --device cpu --output-dir experiments/3dcnn/atlas_free_cnn/outputs/model_comparison`

Implementation requirements:
1. Load held-out/test examples from the atlas-free CNN packed dataset using existing loaders.
2. Keep dataset-source filtering explicit and deterministic.
3. For CNN models, evaluate native 3D reconstruction directly.
4. For MLP model, evaluate only examples that can be represented in MLP masker space. If conversion from atlas-free volume to MLP flatmap is unavailable, implement a clear `unsupported_reason` and still evaluate MLP on the main PubMed/NeuroVault flat latent/image resources where possible.
5. Produce by-sample and summary CSV/JSON outputs.
6. Include smoke tests with tiny fake tensors and monkeypatched registry/adapters.

Run focused tests and a tiny smoke run with `--limit 2` if local cached assets are available. If data are unavailable, document the exact missing cache/HF asset.
```

Exit criteria:

- Script runs with `--help`.
- Unit tests pass.
- A tiny cached-data smoke run works or produces a clear missing-data message.

## Part 4: Contrastive Text-To-Brain And Brain-To-Text Retrieval Comparison

Goal: compare MLP contrastive retrieval against CNN contrastive retrieval in both directions.

Models:

- `mlp_neurovlm`
- `cnn_contrastive_mixed`
- `cnn_contrastive_pubmed`
- `cnn_contrastive_nilearn`
- `cnn_contrastive_neurovault`

Datasets:

- PubMed test set.
- Nilearn test set.
- NeuroVault test set.

Directions:

- Text -> Brain: rank brain maps given paired text embeddings.
- Brain -> Text: rank text records given paired brain embeddings.

Metrics:

- `t2i_recall@1`, `t2i_recall@5`, `t2i_recall@10`, `t2i_recall@50`
- `i2t_recall@1`, `i2t_recall@5`, `i2t_recall@10`, `i2t_recall@50`
- `mean_normalized_k_recall_curve_auc`
- `t2i_normalized_k_recall_curve_auc`
- `i2t_normalized_k_recall_curve_auc`
- MRR and median rank.

Use `src/neurovlm/retrieval_metrics.py` from `main` as the source of truth.

Proposed file:

- `experiments/3dcnn/atlas_free_cnn/evaluation/compare_contrastive_retrieval.py`

Outputs:

- `outputs/model_comparison/contrastive_retrieval_summary.csv`
- `outputs/model_comparison/contrastive_retrieval_curves.csv`
- `outputs/model_comparison/contrastive_retrieval_examples.csv`

Prompt for a fresh Codex session:

```text
Implement contrastive retrieval comparison for MLP NeuroVLM vs CNN contrastive models.

Read:
- `src/neurovlm/retrieval_metrics.py`
- `src/neurovlm/text_to_brain_metrics.py` for projection helper patterns
- `experiments/3dcnn/atlas_free_cnn/evaluation/stage4_semantic.py`
- registry/adapters from Part 2

Add `compare_contrastive_retrieval.py` with a CLI:
`PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_contrastive_retrieval --datasets pubmed nilearn neurovault --models mlp_neurovlm cnn_contrastive_mixed cnn_contrastive_pubmed cnn_contrastive_nilearn cnn_contrastive_neurovault --limit 32 --device cpu --output-dir experiments/3dcnn/atlas_free_cnn/outputs/model_comparison`

Implementation requirements:
1. Build paired text/brain batches from each dataset's held-out/test split.
2. Use the same text embedding convention for CNN as the Stage 3/4 runs: normalized SPECTER2 unless a manifest says otherwise.
3. For MLP, use main NeuroVLM projection heads and MLP brain encoder.
4. For CNN, load Stage 3 evaluator checkpoints via adapter.
5. Use `bidirectional_retrieval_metrics` and also save full normalized recall curves.
6. Include model/dataset/checkpoint provenance columns in every output row.
7. If a requested CNN checkpoint is missing, write a skipped row with `status=missing_checkpoint` rather than failing the entire comparison.

Add tests using fake embeddings where diagonal pairs are perfect and where shuffled pairs are worse.
```

Exit criteria:

- CLI `--help` works.
- Fake-embedding tests prove metric direction and output column names.
- Missing checkpoint behavior is graceful.

## Part 5: Generative Text-To-Brain Comparison

Goal: compare text-to-brain generation quality for MLP vs CNN Stage 4 models.

Models:

- `mlp_neurovlm`
- `cnn_t2b_mixed`
- `cnn_t2b_pubmed`
- `cnn_t2b_nilearn`
- `cnn_t2b_neurovault`

Datasets:

- PubMed test set.
- Nilearn test set.
- NeuroVault test set.

Metrics:

- Pearson r.
- Pearson minus random and percentile.
- Spearman rho.
- Spearman minus random and percentile.
- Dice percentile, default `pct=90`.
- Dice sensitivity at `80, 85, 90, 95`.
- Spin p-value/significance when dependencies are available.
- Generated-image/text retrieval normalized AUC when both generated maps and text embeddings can be projected into the model's shared space.
- CNN native generation metrics from `generation_metrics`: MSE, foreground MSE, spatial correlation, top-k Dice.

Proposed file:

- `experiments/3dcnn/atlas_free_cnn/evaluation/compare_text_to_brain_generation.py`

Outputs:

- `outputs/model_comparison/text_to_brain_by_sample.csv`
- `outputs/model_comparison/text_to_brain_summary.csv`
- `outputs/model_comparison/text_to_brain_random_baseline.csv`
- `outputs/model_comparison/text_to_brain_dice_sensitivity.csv`

Prompt for a fresh Codex session:

```text
Implement generative text-to-brain comparison for MLP NeuroVLM vs CNN Stage 4 models.

Use the Part 2 adapters and reuse main metric helpers:
- `neurovlm.text_to_brain_metrics.evaluate_t2b_sample`
- `add_random_correlation_baseline`
- `sensitivity_rows_for_df`
- `generated_image_text_retrieval_curve` or equivalent shared-space generated-image retrieval
- `atlas_free_cnn.evaluation.generation_metrics.generation_metrics`
- existing Stage 4 loader/evaluator helpers from `stage4_semantic.py`

Add a CLI:
`PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_text_to_brain_generation --datasets pubmed nilearn neurovault --models mlp_neurovlm cnn_t2b_mixed cnn_t2b_pubmed cnn_t2b_nilearn cnn_t2b_neurovault --limit 8 --device cpu --skip-spin --output-dir experiments/3dcnn/atlas_free_cnn/outputs/model_comparison`

Implementation requirements:
1. The output table must identify `model_id`, `model_family`, `dataset`, `sample_id`, `checkpoint`, and `comparison_space`.
2. MLP generation should call the public NeuroVLM path where possible.
3. CNN generation should load Stage 4 projector + matching AE + Stage 3 evaluator.
4. Use random baselines per dataset and model family, not pooled across all datasets.
5. Make spin testing optional and off by default in smoke runs.
6. Missing model/data/checkpoint should produce skipped rows with reason.

Add tests for random-baseline output columns and fake generation metrics.
```

Exit criteria:

- CLI `--help` works.
- Fake tests pass.
- Tiny `--limit 2 --skip-spin` smoke run works if assets are cached, otherwise missing assets are reported clearly.

## Part 6: Brain-To-Text Capability Report

Goal: decide how much brain-to-text comparison to include beyond contrastive retrieval.

Primary comparison:

- Use Part 4 brain -> text retrieval metrics as the contrastive brain-to-text capability comparison.

Optional LLM generation comparison:

- If desired, add a thin B2T generation wrapper for CNN contrastive models:
  - retrieve top-k text evidence from CNN contrastive space
  - feed the same evidence table format into `NeuroVLM.generate_llm_response` or a shared LLM summary function
  - evaluate with `brain_to_text_metrics.py`: BERTScore, semantic similarity, NeuroVLM latent similarity, generated-text normalized Recall@k AUC

This optional part is more expensive and less direct because CNN contrastive models do not currently expose the same public `NeuroVLM.brain(...).to_text()` API. Do it only after the retrieval comparison is working.

Prompt for a fresh Codex session:

```text
Review the contrastive retrieval outputs from Part 4 and decide whether a separate LLM brain-to-text generation comparison is needed. If yes, implement it as an optional script, not as a dependency of the core comparison.

Use `src/neurovlm/brain_to_text_metrics.py` from main for metric definitions. Reuse the same LLM backend/model arguments used by the current NeuroVLM evaluation notebooks. Keep `--limit` support and write skipped rows when LLM dependencies are unavailable.

Do not block the main MLP-vs-CNN comparison on LLM generation. The required B2T capability comparison is the brain-to-text contrastive retrieval from Part 4.
```

Exit criteria:

- Either a clear decision that retrieval-only B2T comparison is sufficient, or an optional script with smoke tests.

## Part 7: Aggregate Report

Goal: produce one readable result package from Parts 3-5.

Proposed file:

- `experiments/3dcnn/atlas_free_cnn/evaluation/aggregate_model_comparison.py`

Outputs:

- `outputs/model_comparison/model_comparison_report.md`
- `outputs/model_comparison/model_comparison_summary.csv`
- optional plots under `outputs/model_comparison/plots/`

Report sections:

- Model inventory and checkpoint provenance.
- AE reconstruction: table by dataset and model.
- Contrastive retrieval: T2B and B2T retrieval table by dataset and model.
- Generative T2B: map-quality and retrieval-space table by dataset and model.
- Missing/skipped models or datasets.
- Recommended headline metrics:
  - AE: mean spatial correlation, top5 Dice, foreground MSE.
  - Contrastive: mean normalized Recall@k AUC, plus direction-specific T2B/B2T AUC.
  - Generative T2B: Pearson minus random, Spearman minus random, Dice pct90, generated-image retrieval AUC.

Prompt for a fresh Codex session:

```text
Aggregate the MLP-vs-CNN comparison outputs into one report.

Read all CSV/JSON files under:
`experiments/3dcnn/atlas_free_cnn/outputs/model_comparison`

Add `aggregate_model_comparison.py` that:
1. validates expected output files,
2. writes concise summary tables,
3. marks missing/skipped model-dataset pairs,
4. writes `model_comparison_report.md`,
5. optionally creates simple plots if matplotlib/seaborn are available.

Do not rerun expensive evaluations from the aggregator. It should only read existing outputs.
```

Exit criteria:

- Report can be regenerated from saved outputs.
- Missing sections fail gracefully with a clear "not available yet" note.

## Final Verification Checklist

Run these after implementation, adjusting for available data/cache:

```bash
PYTHONPATH=src:experiments/3dcnn pytest tests/test_metrics.py tests/test_core.py tests/test_ale_cnn.py tests/test_3dcnn_model_comparison_registry.py

PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_ae_reconstruction --help
PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_contrastive_retrieval --help
PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_text_to_brain_generation --help
PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.aggregate_model_comparison --help
```

Tiny smoke runs:

```bash
PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_ae_reconstruction --datasets pubmed --models cnn_ae_mixed --limit 2 --device cpu

PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_contrastive_retrieval --datasets pubmed --models mlp_neurovlm --limit 4 --device cpu

PYTHONPATH=src:experiments/3dcnn python -m atlas_free_cnn.evaluation.compare_text_to_brain_generation --datasets pubmed --models mlp_neurovlm --limit 2 --device cpu --skip-spin
```

Full runs should be launched only after the smoke runs are clean and checkpoint paths are resolved.
