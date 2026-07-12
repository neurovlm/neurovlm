# Atlas-Free CNN Notebook Guide

This directory contains the mentor-facing notebook history for the atlas-free
3D CNN experiments. The numbered notebooks on disk are the source of truth for
the current branch.

## Recommended Reading Order

| Notebook | Status | Purpose |
| --- | --- | --- |
| `4 multi source autoencoder ablation.ipynb` | Current | Multi-source Stage 1 autoencoder ablation for PubMed, Nilearn, and NeuroVault volumes, including optional Stage 1B domain fine-tuning. |
| `4b stage1 checkpoint selection heldout eval.ipynb` | Current | Evaluation-only held-out checkpoint selection for Stage 1A/1B AE checkpoints. Writes the downstream selected-checkpoint manifest. |
| `5 multi source stage3 stage4.ipynb` | Current | Convention-aware downstream Stage 3/4 notebook. It can run legacy SPECTER2 or normalized SPECTER2 branches; normalized corrected runs pass strict Stage 3 recipe checks. |
| `5b stage3 stage4 semantic evaluation diagnostics.ipynb` | Current | Evaluation-only diagnostics for completed Stage 3/4 runs, including semantic AUC, checkpoint provenance, architecture audits, and Stage 4 retrieval diagnostics. |

## Legacy And Optional Notebooks

| Notebook | Status | Purpose |
| --- | --- | --- |
| `1 best contrastive recipe on pubmed.ipynb` | Legacy optional | Compact PubMed-only Stage 1-3 best-recipe rerun. |
| `2 autoencoder pretraining contrastive ablation.ipynb` | Legacy optional | ALE-only autoencoder-pretraining contrastive ablation, including scratch, frozen-CNN, fine-tuned-CNN, frozen-text, and trainable-text variants. |
| `3 global context architecture sweep.ipynb` | Legacy optional | ResNet, multi-scale, dilation, and global-context exploration. |

## Removed Redundancy

The former 2b frozen-text companion notebook was removed because it was an
isolated rerun slice of Notebook 2. The frozen-text branch is already preserved
inside `2 autoencoder pretraining contrastive ablation.ipynb` as
`ae_pretrained_finetune_cnn_pretrained_text_frozen`.

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

Notebooks 1-3 are legacy exploratory ALE-only work: the compact best recipe,
autoencoder pretraining/text-projection choices, and architecture variants.

Notebooks 4, 4b, 5, and 5b are the current multi-source pipeline. Notebook 4
builds and fine-tunes autoencoders, notebook 4b selects AE checkpoints on
held-out splits, notebook 5 runs convention-aware Stage 3/4 branches, and
notebook 5b evaluates completed Stage 3/4 outputs.

Stage 4 training is spatial-first by default: semantic AUC is disabled during
training, the primary checkpoint is `best_val_top5_dice.pt`, and the
semantic-selected checkpoint is not produced unless semantic AUC is explicitly
enabled. Notebook 5b computes semantic AUC later as a final diagnostic for the
spatial-primary checkpoint.
