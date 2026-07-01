# Atlas-Free CNN Notebook Guide

This directory contains the notebook history for the atlas-free CNN experiments. The current mentor-facing sequence is numbered so the story can be read from early baselines through the corrected multi-source pipeline.

## Recommended Reading Order

| Notebook | Purpose | Use this when |
| --- | --- | --- |
| `1_baseline_difumo_vs_atlas_free_ale3dcnn.ipynb` | Original dense ALE3DCNN baseline comparing DiFuMo-compatible and atlas-free PubMed ALE inputs. | You want the first baseline comparison before autoencoder pretraining or multi-source data. |
| `2_autoencoder_pretraining_contrastive_ablation.ipynb` | Tests whether ALE3DCNN autoencoder pretraining improves contrastive retrieval, with scratch/frozen/fine-tuned CNN and frozen/trainable text-projection branches. | You want to justify why the final recipe uses AE-pretrained CNN initialization. |
| `2b_frozen_text_projection_ablation.ipynb` | Companion ablation focused on whether the pretrained text projection should stay frozen or train during contrastive learning. | You want the text-projection comparison separated from the broader AE ablation. |
| `3_global_context_architecture_sweep.ipynb` | Sweeps ResNet, multi-scale, dilation, and global-context variants while giving each architecture its own matching AE pretraining run. | You want architecture exploration beyond the plain CNN best recipe. |
| `4_best_stage1_to_stage3_contrastive_recipe.ipynb` | Streamlined reproduction of the known-best early recipe: Stage 1 AE pretraining, Stage 2 AE checkpoint selection, and Stage 3 `ae_pretrained_finetune_cnn_pretrained_text_trainable`. | You want the cleanest notebook for rerunning only the best ALE-only Stage 1-3 path. |
| `5_stage1_autoencoder_ablation_multi_source.ipynb` | Multi-source Stage 1 AE ablation for PubMed, Nilearn, and NeuroVault volumes, including optional Stage 1B domain fine-tuning. | You want candidate AE checkpoints for the controlled multi-source pipeline. |
| `5b_stage1_checkpoint_selection_heldout_eval.ipynb` | Evaluation-only held-out checkpoint selection for Stage 1A/1B AE checkpoints. Writes the downstream selected-checkpoint manifest. | You want trustworthy AE checkpoint choices before Stage 3/4 training. |
| `6_legacy_specter_stage2_stage3_stage4_pipeline.ipynb` | Downstream Stage 2/3/4 controlled pipeline using the original unnormalized SPECTER text cache. | You want the legacy/raw-text-cache baseline for all six domain/AE branches. |
| `6a_normalized_specter_stage3_stage4_rerun.ipynb` | Normalized-SPECTER rerun of Stage 3 and corrected Stage 4 using empty-string-centered, unit-normalized SPECTER2 embeddings. | You want the corrected normalized-cache comparison against notebook 6. |
| `7_stage3_stage4_semantic_evaluation_diagnostics.ipynb` | Evaluation-only diagnostics for completed Stage 3/4 runs, including semantic AUC, checkpoint provenance, architecture audits, and Stage 4 retrieval diagnostics. | You want final comparison tables and sanity checks without retraining. |

## Short Narrative

Notebooks 1-3 are exploratory ALE-only work: baseline encoders, autoencoder pretraining, text-projection choices, and architecture variants. Notebook 4 is the compact rerun path for the best ALE-only Stage 1-3 recipe.

Notebooks 5-7 are the current multi-source pipeline. Notebook 5 builds and fine-tunes autoencoders, notebook 5b selects AE checkpoints on held-out splits, notebook 6 trains the legacy SPECTER downstream branches, notebook 6a reruns the corrected normalized-SPECTER branches, and notebook 7 evaluates the completed Stage 3/4 outputs.

Stage 4 training is spatial-first by default: semantic AUC is disabled during training, the primary checkpoint is `best_val_top5_dice.pt`, and the semantic-selected checkpoint is not produced unless semantic AUC is explicitly enabled. Notebook 7 can compute semantic AUC later as a final diagnostic for the spatial-primary checkpoint.

The old decoder-only text-to-brain notebook was removed because Stage 4 generation is now handled in the controlled downstream notebooks and diagnostics.
