# Atlas-Free CNN Notebook Guide

This directory contains the mentor-facing notebook history for the atlas-free
3D CNN experiments. The numbered notebooks on disk are the source of truth for
the current branch.

## Recommended Reading Order

| Notebook                                                 | Status  | Purpose                                                                                                                                                                  |
| -------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `4 multi source autoencoder ablation.ipynb`              | Current | Multi-source Stage 1 baseline autoencoder training followed by required PubMed, Nilearn, and NeuroVault Stage 1B fine-tuning.                                           |
| `4b stage1 checkpoint selection heldout eval.ipynb`      | Current | Evaluation-only held-out checkpoint selection for Stage 1A/1B AE checkpoints. Writes the downstream selected-checkpoint manifest.                                        |
| `5 multi source stage3 stage4.ipynb`                     | Current | Normalized-SPECTER2 downstream notebook for the fixed six Stage 3 and six Stage 4 runs. |
| `5b stage3 stage4 semantic evaluation diagnostics.ipynb` | Current | Evaluation-only diagnostics for completed Stage 3/4 runs, including semantic AUC, checkpoint provenance, architecture audits, and Stage 4 retrieval diagnostics.         |

## Retained Optional Notebooks

| Notebook                                    | Status          | Purpose                                                                                       |
| ------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------- |
| `1 best contrastive recipe on pubmed.ipynb` | Optional | Compact PubMed-only Stage 1-3 best-recipe rerun using the retained plain CNN.                  |
| `3 resnet48 multi scale attention.ipynb`    | Optional | Retained ResNet48 multi-scale attention model with matching autoencoder pretraining.          |

## Code Review Map

Core model and training code lives under `atlas_free_cnn/training/`:
`model_wrappers.py`, `datasets.py`, `checkpointing.py`, loss modules, and the
Stage 1/3/4 trainers are the main implementation targets.

Pipeline and evaluation code that supports the current notebooks lives in
`pipeline_outputs.py`, `stage1_selection_integration.py`, and
`atlas_free_cnn/evaluation/`.

Notebook-only orchestration, Colab/Hugging Face download helpers, and display
support live in `notebook_utils.py` and `model_comparison/plotting_utils.py`.
Shared pure naming/path conventions live in `conventions.py` so core training
code does not need to import notebook helpers.

## Short Narrative

Notebooks 1 and 3 retain the two ALE-only model paths: the compact plain-CNN
best recipe and the ResNet48 multi-scale attention model. Notebook 2 and the
discarded architecture/recipe variants have been removed.

Notebooks 4, 4b, 5, and 5b are the current multi-source pipeline. Notebook 4
builds the baseline raw-MSE autoencoder and fine-tunes it once per domain,
notebook 4b selects AE checkpoints on
held-out splits, notebook 5 runs the fixed normalized Stage 3/4 branch matrix, and
notebook 5b evaluates completed Stage 3/4 outputs.

Stage 4 training is spatial-first by default: semantic AUC is disabled during
training, the primary checkpoint is `best_val_top5_dice.pt`, and the
semantic-selected checkpoint is not produced unless semantic AUC is explicitly
enabled. Notebook 5b computes semantic AUC later as a final diagnostic for the
spatial-primary checkpoint.
