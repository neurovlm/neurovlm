# Atlas-Free 3D CNN

This directory contains the retained implementation for the atlas-free 3D CNN
experiments. The active multi-source workflow is intentionally narrow:

1. Train one mixed-source baseline autoencoder with natural sampling and raw
   MSE.
2. Fine-tune that autoencoder separately on PubMed, Nilearn, and NeuroVault.
3. Train the mixed baseline on all three Stage 3 contrastive tasks and each
   specialized autoencoder on its matching task, producing six runs.
4. Train one text-to-brain projection head for each of those six branches and
   evaluate it on the matching held-out test split.

All Stage 3 and Stage 4 runs use the normalized SPECTER2 convention:
empty-centered embeddings followed by unit normalization. There is no retained
unnormalized SPECTER2 path.

## Notebooks

The primary workflow is:

- `../4 multi source autoencoder.ipynb`
- `../5 multi source stage3 stage4.ipynb`

Notebook 4 writes the four selected AE checkpoints, their reconstruction
metrics, and a checkpoint registry. Notebook 5 validates the corresponding
four locked AE resource checkpoints before starting the fixed six Stage 3 and
six Stage 4 runs. Stage 4 training itself writes the held-out spatial and
semantic metrics.

The retained optional notebooks are:

- `../1 best contrastive recipe on pubmed.ipynb`: PubMed-only compact
  plain-CNN recipe.
- `../3 resnet48 multi scale attention.ipynb`: ResNet48 multi-scale attention
  variant.

See `../NOTEBOOK_GUIDE.md` for the execution order and output details.

## Directory Layout

- `training/`: plain CNN and ResNet48 model construction, datasets, losses,
  checkpointing, and the Stage 1/3/4 trainers.
- `evaluation/`: canonical reconstruction, retrieval, generation, and Stage 4
  semantic metrics plus the distinct MLP-versus-CNN comparison utilities.
- `data_building/`: only the normalized SPECTER2 cache builder and package
  initializer.
- `configs/`: retained standalone configuration templates. Notebook-generated
  per-run configurations remain authoritative for the six-branch workflow.
- `cache/`: finalized local split JSONLs, shared volume tensor, source caches,
  and normalized text embeddings. The notebooks use local artifacts first and
  fall back to Hugging Face.
- `conventions.py`: fixed domains, branches, checkpoint names, normalized-text
  convention, and stage output paths.
- `notebook_utils.py`: notebook-only environment, download, path-discovery,
  and validation helpers.
- `pipeline_outputs.py`: run-status and output-table helpers.
- `stage1_selection_integration.py`: validates the four finalized AE files and
  creates the exact six-run downstream manifest.

## Retained Models and Recipes

The retained encoder surface is:

- the compact plain 3D CNN used by Notebooks 1, 4, and 5;
- the ResNet48 multi-scale attention encoder used by Notebook 3.

The multi-source Stage 1 recipe is fixed:

- model: plain 3D CNN, 384-dimensional latent space;
- sampling: natural mixed-source sampling;
- loss: raw MSE with an unconstrained decoder output;
- mixed checkpoint selection: best validation loss;
- domain-fine-tuned checkpoint selection: best validation top-5 Dice.

Notebook 4 evaluates only the selected checkpoint for each finalized branch.
Its canonical combined outputs are:

- `07_final_comparison/selected_ae_checkpoints.json`
- `07_final_comparison/final_summary_table.csv`
- per-run `metrics/reconstruction_summary_by_source.csv`

## Downstream Six-Run Matrix

For each domain, Notebook 5 compares the mixed baseline against the matching
specialized AE:

| Domain | Mixed baseline | Specialized AE |
| --- | --- | --- |
| PubMed | `mixed_stage1a` | `mixed_to_pubmed_stage1b` |
| Nilearn | `mixed_stage1a` | `mixed_to_nilearn_stage1b` |
| NeuroVault | `mixed_stage1a` | `mixed_to_neurovault_stage1b` |

Every branch uses the same normalized SPECTER2 cache and controlled Stage 3
recipe. Stage 4 freezes the AE decoder and trains a fresh 768 → 512 → 384
text-to-latent projection head for that branch. Comparisons are made within
domain on identical splits.

Stage 4 uses a spatial-first checkpoint policy:

- primary checkpoint: `best_val_top5_dice.pt`;
- secondary spatial checkpoint: `best_val_spatial_corr.pt`;
- semantic AUC during training: disabled by default;
- held-out spatial and semantic metrics: written by the Stage 4 trainer.

## Data Surface

Training consumes the finalized unified `train.jsonl`, `val.jsonl`, and
`test.jsonl` files plus `atlas_free_cnn_volumes.pt`. These are discovered under
`cache/unified_jsonl*/splits` and `cache/hf_atlas_free_cnn*`, or downloaded from
the configured Hugging Face dataset repository.

The only retained builder is
`data_building/build_normalized_specter2_cache.py`. It creates the canonical
`specter2_stage3_stage4_emptycentered_unitnorm.pt` cache and its validation
sidecars. The old ingestion, JSONL rewriting, QC, export, network-evaluation,
and alternate SPECTER2 cache builders are not part of the final pipeline.

## Evaluation Surface

Training modules produce their own canonical held-out metrics. The remaining
`evaluation/compare_*.py` modules are not alternate evaluators for the same
runs; they support the separate CNN-versus-MLP comparison notebooks under
`../model_comparison/`.

For a focused review, start with:

- `training/ale_cnn.py`
- `training/train_autoencoder.py`
- `training/train_ale_cnn.py`
- `training/train_text_to_brain.py`
- `conventions.py`
- `stage1_selection_integration.py`
