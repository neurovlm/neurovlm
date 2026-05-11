# Atlas-Free Multi-Positive NeuroVLM Pipeline

This directory is an isolated first implementation of the cleaner atlas-free training dataset and multi-positive contrastive loop. It does not modify the existing NeuroVLM training code.

## What This Builds

Each JSONL row represents one brain map with multiple valid text positives:

- Nilearn atlas parcels, networks, and components with label/definition positives.
- PubMed coordinate-derived ALE maps with allowed MeSH positives, wiki-style summaries, and optional titles.
- No raw abstracts, LLM-extracted terms, molecular terms, method terms, organism terms, demographics, or broad biological process terms are added to `positive_texts`.

## Discovered Existing Project Paths

- PubMed publication metadata: `src/neurovlm/retrieval_resources.py::_load_pubmed_dataframe()`, HuggingFace file `neurovlm/neuro_image_papers/publications.parquet`.
- PubMed summaries: `src/neurovlm/retrieval_resources.py::_load_pubmed_summaries_dataframe()`, local artifact `artifacts/neuro_summaries/neuro_summaries.parquet`.
- PubMed coordinates: `src/neurovlm/retrieval_resources.py::_load_pubmed_coordinates()`, HuggingFace file `coordinates.parquet`.
- Current packed atlas-free ALE tensor cache: `data/ale_caches/atlas_free_4mm_fwhm9_crop_float16.pt`.
- ALE cache builder and dataset: `src/neurovlm/gnn/ale_dataset.py`.
- 3D CNN encoder: `src/neurovlm/gnn/ale_cnn.py::ALE3DCNNEncoder`.
- SPECTER/SPECTER2 wrapper: `src/neurovlm/models.py::Specter`.
- Existing InfoNCE: `src/neurovlm/loss.py::InfoNCELoss`.
- MeSH annotations: `experiments/data/mesh_kg/mesh_annotations.json`.
- MeSH node categories: `experiments/data/mesh_kg/mesh_kg_nodes.parquet`.
- MeSH definitions checkpoint: `experiments/data/unified_kg/kg_mesh_definitions_checkpoint.parquet`.

These are recorded in `configs/paths.yaml` and can be overridden by CLI flags or config edits.

## First Commands

```bash
python -m atlas_free_multipositive.data_building.ingest_pubmed_ale \
  --paths atlas_free_multipositive/configs/paths.yaml \
  --config atlas_free_multipositive/configs/dataset_config.yaml

python -m atlas_free_multipositive.data_building.ingest_nilearn_atlases \
  --paths atlas_free_multipositive/configs/paths.yaml \
  --config atlas_free_multipositive/configs/dataset_config.yaml

python -m atlas_free_multipositive.data_building.build_unified_dataset \
  --inputs atlas_free_multipositive/cache/unified_jsonl/pubmed_ale.jsonl \
           atlas_free_multipositive/cache/unified_jsonl/nilearn_atlases.jsonl

python -m atlas_free_multipositive.data_building.qc_dataset \
  atlas_free_multipositive/cache/unified_jsonl/unified_map_text.jsonl
```

The PubMed ingestion can start from the existing packed tensor cache. Nilearn ingestion writes individual 2mm NIfTI maps under `cache/maps/nilearn/`.

## Training Pieces

- `training/datasets.py`: reads unified JSONL and returns `[1, X, Y, Z]` volumes.
- `training/collators.py`: samples K positives per map and builds `[B, T]` masks and weights.
- `training/losses.py`: weighted multi-positive brain-to-text InfoNCE.
- `training/train_contrastive.py`: tiny smoke-test trainer that expects precomputed text embeddings.

The first intended recipe is frozen SPECTER/SPECTER2 text embeddings plus trainable 3D CNN and text projection head.

