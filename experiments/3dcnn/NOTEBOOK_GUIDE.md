# Atlas-Free CNN Notebook Guide

The four notebooks in this directory are the retained atlas-free 3D CNN
workflow. Notebooks 4 and 5 form the finalized multi-source pipeline;
Notebooks 1 and 3 preserve the two still-relevant standalone model paths.

## Final Multi-Source Workflow

Run these notebooks in order:

| Order | Notebook | Purpose |
| --- | --- | --- |
| 1 | `4 multi source autoencoder.ipynb` | Train the mixed raw-MSE baseline autoencoder, fine-tune it separately on PubMed, Nilearn, and NeuroVault, and evaluate the four selected checkpoints. |
| 2 | `5 multi source stage3 stage4.ipynb` | Validate the four AE checkpoints, use normalized SPECTER2 embeddings, train the fixed six Stage 3 contrastive branches, and train/evaluate the matching six Stage 4 text-to-brain projection heads. |

Notebook 4 contains no recipe or architecture alternatives. Its fixed choices
are:

- mixed Stage 1A: natural source sampling, raw MSE, plain 3D CNN, selected by
  validation loss;
- Stage 1B: the same mixed checkpoint fine-tuned independently on PubMed,
  Nilearn, and NeuroVault, selected by validation top-5 Dice;
- full validation/test reconstruction evaluation for the four selected
  checkpoints.

Notebook 4 writes its finalized outputs below
`runs_atlas_free_cnn_stage1/finalized_stage1/` by default. Its main summary
files are:

- `07_final_comparison/selected_ae_checkpoints.json`
- `07_final_comparison/final_summary_table.csv`
- each run's `metrics/reconstruction_summary_by_source.csv`

Notebook 5 validates the four locked AE resource checkpoints corresponding to
the Notebook 4 branches before downstream training. It uses normalized,
empty-centered, unit-normalized SPECTER2 embeddings only. Its six branches are:

| Domain | Baseline branch | Specialized branch |
| --- | --- | --- |
| PubMed | mixed Stage 1A AE | PubMed Stage 1B AE |
| Nilearn | mixed Stage 1A AE | Nilearn Stage 1B AE |
| NeuroVault | mixed Stage 1A AE | NeuroVault Stage 1B AE |

Each branch receives its own Stage 3 contrastive run and its own Stage 4
text-to-AE-latent projection head. The Stage 4 trainer writes held-out spatial
and semantic generation metrics, so separate 4b/5b evaluation notebooks are
not required.

## Retained Standalone Notebooks

| Notebook | Purpose |
| --- | --- |
| `1 best contrastive recipe on pubmed.ipynb` | PubMed-only rerun of the retained compact plain-CNN contrastive recipe. |
| `3 resnet48 multi scale attention.ipynb` | The retained ResNet48 multi-scale attention variant and its matching autoencoder pretraining. |

These notebooks are useful independently; they are not prerequisites for the
multi-source Notebook 4 → Notebook 5 workflow.

## Model-Comparison Notebooks

The notebooks under `model_comparison/` compare the retained CNN with the MLP
at the autoencoder, contrastive-retrieval, and text-to-brain stages. They are
kept because these are cross-model comparisons, not duplicate training-time
evaluation passes.

## Code Review Map

- `atlas_free_cnn/training/`: retained models, losses, datasets,
  checkpointing, and Stage 1/3/4 trainers.
- `atlas_free_cnn/conventions.py`: fixed branch names, text-embedding
  convention, checkpoint names, and output paths.
- `atlas_free_cnn/stage1_selection_integration.py`: validates the four AE
  checkpoints and creates Notebook 5's six-run manifest.
- `atlas_free_cnn/notebook_utils.py`: Colab, Drive, Hugging Face, split, and
  normalized-text-cache helpers used by notebooks.
- `atlas_free_cnn/evaluation/`: canonical metric implementations and the
  distinct MLP-versus-CNN comparison utilities.
- `atlas_free_cnn/data_building/build_normalized_specter2_cache.py`: the only
  retained offline data builder.

## Removed Experiment Surface

Notebook 2, Notebooks 4b/5b, discarded architecture and loss variants,
unnormalized SPECTER2 paths, standalone duplicate evaluation commands, and the
obsolete raw-data builders have been removed. The current notebooks consume
the finalized train/validation/test JSONLs and shared volume tensor locally or
download them from Hugging Face.
