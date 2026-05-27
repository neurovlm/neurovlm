# Clean Atlas-Free CNN Area

This folder keeps only the cache and support code for the atlas-free 3D CNN
setup. The old multipositive notebooks/trainers were removed to avoid mixing
that architecture with the working 3D CNN recipe.

Important paths:

- `cache/`: moved mixed PubMed/NeuroVault/Nilearn cached artifacts.
- `data/ale_caches/`: old good PubMed ALE caches.
- `data_building/`: ingestion, preprocessing, packing, and audit scripts.
- `training/`: stable raw-MSE autoencoder and text-to-brain projection trainers.
- `evaluation/`: generation/reconstruction metrics and generation evaluation.

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

To add the separate network-map test set used by brain-to-text semantic
evaluation:

```bash
PYTHONPATH=experiments/3dcnn:src .conda/bin/python -m atlas_free_cnn.data_building.build_network_eval_jsonl
```

After building it, make sure the SPECTER text embedding cache also includes the
network test texts before running text-to-brain generation evaluation.
