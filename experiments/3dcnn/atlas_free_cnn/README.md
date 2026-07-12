# Clean Atlas-Free CNN Area

This folder keeps only the cache and support code for the atlas-free 3D CNN
setup. The old multipositive notebooks/trainers were removed to avoid mixing
that architecture with the working 3D CNN recipe.

Important paths:

- `cache/`: moved mixed PubMed/NeuroVault/Nilearn cached artifacts.
- `data/ale_caches/`: old good PubMed ALE caches.
- `data_building/`: ingestion, preprocessing, packing, and audit scripts.
- `training/`: core model construction, datasets, losses, checkpointing, and
  Stage 1/3/4 trainers.
- `evaluation/`: checkpoint selection, generation/reconstruction metrics, and
  model-comparison scripts used by the notebooks.
- `conventions.py`: shared pure naming, path, checkpoint, and text-embedding
  conventions used by core code and notebooks.
- `notebook_utils.py`: notebook-only orchestration helpers for Colab, Drive,
  Hugging Face downloads, and display-time validation. Core training code
  should not import this module.

For mentor review, start with the model/training surface:
`training/ale_cnn.py`, `training/model_wrappers.py`,
`training/train_autoencoder.py`, `training/train_ale_cnn.py`,
`training/train_text_to_brain.py`, the loss modules, and
`training/checkpointing.py`. Then review the current pipeline support in
`pipeline_outputs.py`, `stage1_selection_integration.py`, and
`evaluation/stage1_checkpoint_evaluation.py`.

Notebook/report support code is intentionally separate: `notebook_utils.py`,
`evaluation/compare_*.py`, and `model_comparison/plotting_utils.py` exist to
keep notebooks readable and to reproduce figures/tables, not to define the core
model architecture or training objective.

Notebook 4b Stage 1A checkpoint comparison writes two recipe-level tables under
`01_stage1a/`: `stage1a_all_checkpoint_eval.csv` contains one row per evaluated
checkpoint per Stage 1A recipe, and
`stage1a_recipe_best_checkpoint_comparison.csv` compares only the best checkpoint
selected within each recipe. Each Stage 1A recipe was first checkpoint-selected
on the same held-out split. The table compares the best checkpoint from each
recipe.

For module-style commands from the repo root, use:

```bash
PYTHONPATH=experiments/3dcnn:src .conda/bin/python -m atlas_free_cnn.data_building.audit_preprocessing
```

To refresh the AE-training JSONL splits from the packed shared tensor:

```bash
PYTHONPATH=experiments/3dcnn:src .conda/bin/python -m atlas_free_cnn.data_building.export_hf_pack_jsonl
```

For the current checked-in/moved cache, you do not need to rerun full
preprocessing before training. The shared tensor pack and train/val/test JSONL
already exist under `cache/hf_atlas_free_cnn/` and `cache/unified_jsonl/`.
Rerun ingestion/packing only if you change the source data or rebuild the cache
from scratch.

The text-to-brain order is:

1. Train the text-to-brain projection head.
2. Use that projection plus the frozen AE decoder to generate maps.
3. Evaluate generated maps on the held-out mixed test set by source.

Stage 4 training is spatial-fidelity focused by default:

- Stage 4 training semantic AUC: disabled
- Stage 4 primary checkpoint: `best_val_top5_dice.pt`
- Stage 4 secondary spatial checkpoint: `best_val_spatial_corr.pt`
- Stage 4 semantic checkpoint: not produced during training unless semantic AUC is explicitly enabled
- Stage 4 semantic diagnostics: available in Notebook 5b final evaluation

To add the separate network-map test set used by brain-to-text semantic
evaluation:

```bash
PYTHONPATH=experiments/3dcnn:src .conda/bin/python -m atlas_free_cnn.data_building.build_network_eval_jsonl
```

After building it, make sure the SPECTER text embedding cache also includes the
network test texts before running text-to-brain generation evaluation.
