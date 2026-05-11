# Atlas-Free Multi-Positive NeuroVLM Pipeline

This directory is an isolated first implementation of the cleaner atlas-free training dataset and multi-positive contrastive loop. It does not modify the existing NeuroVLM training code.

## What This Builds

Each JSONL row represents one brain map with multiple valid text positives:

- Nilearn atlas parcels, networks, and components with label/definition positives.
- PubMed coordinate-derived ALE maps with allowed MeSH positives, wiki-style summaries, and optional titles.
- No raw abstracts, LLM-extracted terms, molecular terms, method terms, organism terms, demographics, or broad biological process terms are added to `positive_texts`.

The configured Nilearn atlas ingestion now includes:

- `yeo_2011`: 7-network label map split into one binary network map per network.
- `schaefer_2018`: 100 cortical parcels with Yeo-7 network-aware labels.
- `harvard_oxford_cortical` and `harvard_oxford_subcortical`: anatomical region masks.
- `juelich_probabilistic`: probabilistic cytoarchitectonic maps when available through Nilearn.
- `aal`: anatomical masks, skipped gracefully if the local/downloaded Nilearn files are incomplete.
- `smith_2009`: resting-state ICA/network maps when available through Nilearn.

Priority-3 atlases such as Brainnetome, Glasser/HCP-MMP volumetric derivatives, and HCP task maps can be added as local NIfTI files under `custom_nifti_atlases` in `configs/dataset_config.yaml`.

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

## Decoder And Generation Add-On

The decoder/generation code is an add-on staged training path for the same atlas-free CNN model family. It does not replace the multi-positive retrieval model.

Retrieval path:

```text
brain volume -> ALE3DCNNEncoder -> 384-d latent
text -> frozen SPECTER/SPECTER2 -> TextProjectionHead -> 384-d latent
```

Generation path:

```text
text -> frozen SPECTER/SPECTER2 -> TextProjectionHead -> CNN latent -> ALE3DCNNDecoder -> generated brain volume
```

### Stage 1: Sparse CNN Autoencoder

Train:

```text
brain volume -> CNN encoder -> CNN latent -> CNN decoder -> reconstructed brain volume
```

Run:

```bash
python -m atlas_free_multipositive.training.train_autoencoder_sparse \
  --config atlas_free_multipositive/configs/autoencoder_sparse_config.yaml
```

Loss includes weighted reconstruction, soft Dice, top-k overlap, and spatial correlation. Checkpoints are saved by top-5% Dice, spatial correlation, and a combined generation score, not plain MSE alone.

### Stage 2: Text-To-Brain Projection

Train:

```text
SPECTER/SPECTER2 embedding -> TextProjectionHead -> CNN latent -> frozen CNN decoder -> generated brain volume
```

Run:

```bash
python -m atlas_free_multipositive.training.train_text_to_brain \
  --config atlas_free_multipositive/configs/text_to_brain_config.yaml
```

Run at least two variants by editing `text_projection_init`:

- `random`
- `pretrained_text_infonce`

The pretrained `text_infonce` projection is only an initialization variant because it was trained for the old MLP latent space, not this CNN latent manifold.

### Stage 3: Main Multi-Positive Contrastive Training

Train the retrieval model on the expanded multi-positive dataset. Compare text projection initializations:

- random
- pretrained NeuroVLM `text_infonce`
- Stage 2 CNN-decoder-trained projection checkpoint

The main question is whether the CNN-decoder-trained projection improves semantic retrieval and/or text-to-brain generation compared with random or old MLP-latent `text_infonce`.

### Stage 4: Optional Joint Training

Train:

```text
multi-positive contrastive loss
+ text-to-brain generation loss
+ latent alignment loss
```

Run:

```bash
python -m atlas_free_multipositive.training.train_joint_generation_contrastive \
  --config atlas_free_multipositive/configs/joint_generation_contrastive_config.yaml
```

Start conservatively: initialize encoder/decoder from Stage 1, initialize text projection from Stage 2 if available, freeze SPECTER, freeze decoder, and fine-tune CNN encoder plus text projection.

## Generation Evaluation

Generation metrics live in:

- `evaluation/generation_metrics.py`
- `evaluation/evaluate_generation.py`
- `evaluation/evaluate_peak_metrics.py`
- `evaluation/evaluate_generation_semantic_retrieval.py`

Baselines live in:

- `training/generation_baselines.py`

Compare against:

- global mean ALE map
- random training ALE map
- nearest-neighbor text embedding ALE map
- MeSH/category-average maps when available
- network-prior average maps when available

Voxel metrics are not enough. A generated map should also decode back into the right semantic neighborhood via generated map -> CNN encoder -> label/text ranking.

## Smoke Tests

```bash
.conda/bin/python -m pytest atlas_free_multipositive/tests -q
```

Current tests cover:

- generation loss finite scalar and backward pass
- hard top-k Dice and differentiable top-k overlap
- latent alignment loss
- baseline generation evaluation on four examples
- Stage 2 text-to-brain training loop on four toy maps

## Colab Training Notebooks

The training notebooks are now written as Colab Run all workflows, modeled after the existing experiment Colab notebooks. They mount Drive, install dependencies, locate the project folder on Drive, check data paths, then run training.

Recommended Colab notebooks:

- `06_colab_multipositive_retrieval_training.ipynb`
  Main multi-positive retrieval training plus a sampled-positive validation diagnostic.
- `09_10_colab_generation_pipeline.ipynb`
  Stage 1 sparse CNN autoencoder, then Stage 2 text-to-brain projection with both random and pretrained `text_infonce` initialization.
- `13_colab_joint_generation_contrastive.ipynb`
  Optional Stage 4 joint generation + contrastive training after `09_10` has produced checkpoints.

The training notebooks clone/pull the repo into `/content/neurovlm`, copy generated data from Drive into that clone, and write run outputs back to Drive under `atlas_free_multipositive/outputs/runs`.

Before opening in Colab, upload the generated data folder to Google Drive. The preferred path is:

```text
/content/drive/MyDrive/neurovlm/
```

The notebook itself clones the source code from GitHub, so the Drive folder mainly needs the generated data from local notebooks `1-5`:

```text
atlas_free_multipositive/cache/unified_jsonl/splits/train.jsonl
atlas_free_multipositive/cache/unified_jsonl/splits/val.jsonl
atlas_free_multipositive/cache/unified_jsonl/text_registry.jsonl
atlas_free_multipositive/cache/text_embeddings/specter_text_cache.pt
atlas_free_multipositive/cache/maps/                         # if JSONL rows reference NIfTI atlas maps
atlas_free_multipositive/data/ale_caches/atlas_free_4mm_fwhm9_crop_float16.pt
```
