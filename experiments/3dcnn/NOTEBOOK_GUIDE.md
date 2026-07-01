# Atlas-Free CNN Notebook Guide

This directory contains the notebook history for the atlas-free CNN experiments. The current mentor-facing sequence is numbered so the story can be read from early baselines through the corrected multi-source pipeline. Filenames below are the exact files currently on disk.

## Recommended Reading Order

| Notebook | Purpose | Use this when |
| --- | --- | --- |
| `5 multi source autoencoder ablation.ipynb` | Current multi-source Stage 1 AE ablation for PubMed, Nilearn, and NeuroVault volumes, including optional Stage 1B domain fine-tuning. | You want candidate AE checkpoints for the controlled multi-source pipeline. |
| `5b stage1 checkpoint selection heldout eval.ipynb` | Current evaluation-only held-out checkpoint selection for Stage 1A/1B AE checkpoints. Writes the downstream selected-checkpoint manifest. | You want trustworthy AE checkpoint choices before Stage 3/4 training. |
| `6 multi source stage3 stage4.ipynb` | Current convention-aware downstream Stage 3/4 notebook. It can run legacy SPECTER2 or normalized SPECTER2 branches; normalized corrected 6a-style runs pass strict Stage 3 recipe checks. | You want controlled downstream Stage 3/4 training from the selected AE checkpoints. |
| `6b stage3 stage4 semantic evaluation diagnostics.ipynb` | Current evaluation-only diagnostics for completed Stage 3/4 runs, including semantic AUC, checkpoint provenance, architecture audits, and Stage 4 retrieval diagnostics. | You want final comparison tables and sanity checks without retraining. |

## Legacy And Optional Notebooks

| Notebook | Status | Use this when |
| --- | --- | --- |
| `1 best contrastive recipe on pubmed.ipynb` | Legacy optional ALE-only best-recipe rerun. | You want the compact PubMed-only Stage 1-3 recipe history. |
| `2 baseline difumo vs atlas free ale3dcnn.ipynb` | Legacy optional baseline. | You want the first DiFuMo-compatible versus atlas-free PubMed ALE comparison. |
| `3 autoencoder pretraining contrastive ablation.ipynb` | Legacy optional ablation. | You want the AE-pretraining contrastive ablation history. |
| `3b frozen text projection ablation.ipynb` | Legacy optional ablation. | You want the frozen versus trainable text-projection comparison. |
| `4 global context architecture sweep.ipynb` | Legacy optional architecture sweep. | You want ResNet, multi-scale, dilation, and global-context exploration. |

## Short Narrative

Notebooks 1-4 are legacy exploratory ALE-only work: baseline encoders, autoencoder pretraining, text-projection choices, and architecture variants.

Notebooks 5, 5b, 6, and 6b are the current multi-source pipeline. Notebook 5 builds and fine-tunes autoencoders, notebook 5b selects AE checkpoints on held-out splits, notebook 6 runs convention-aware Stage 3/4 branches, and notebook 6b evaluates completed Stage 3/4 outputs.

Stage 4 training is spatial-first by default: semantic AUC is disabled during training, the primary checkpoint is `best_val_top5_dice.pt`, and the semantic-selected checkpoint is not produced unless semantic AUC is explicitly enabled. Notebook 7 can compute semantic AUC later as a final diagnostic for the spatial-primary checkpoint.

The old decoder-only text-to-brain notebook was removed because Stage 4 generation is now handled in the controlled downstream notebooks and diagnostics.
