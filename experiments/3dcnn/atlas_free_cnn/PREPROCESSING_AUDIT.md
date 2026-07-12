# Atlas-Free CNN Preprocessing Audit

This note documents how the atlas-free CNN image tensors, JSONL rows, text targets, and split files are produced before the training notebooks run. The important distinction is that the current notebooks consume already-preprocessed tensors through unified JSONLs; image preprocessing happens in the data-building scripts below.

## Preprocessing script map

| Step | Script or module | Main outputs | Notes |
| --- | --- | --- | --- |
| Global data paths and source choices | `experiments/3dcnn/atlas_free_cnn/configs/paths.yaml`, `experiments/3dcnn/atlas_free_cnn/configs/dataset_config.yaml` | Runtime path values and atlas/source lists | `paths.yaml` points to the old PubMed ALE cache. `dataset_config.yaml` defines Nilearn atlas choices and text-row policy. |
| Shared NIfTI helpers | `experiments/3dcnn/atlas_free_cnn/data_building/preprocessing.py` | Reusable MNI target loading, resampling, array cleaning, NIfTI metadata | Used by Nilearn/custom atlas ingestion and helper conversions. |
| PubMed ALE cache creation/loading | `atlas_free_cnn.training.ale_dataset.ALEPreprocessConfig`, `atlas_free_cnn.training.ale_dataset.build_or_load_ale_cache`; called by `experiments/3dcnn/atlas_free_cnn/training/train_ale_cnn.py` and `experiments/3dcnn/atlas_free_cnn/data_building/build_ale_fwhm_caches.py` | `data/ale_caches/atlas_free_4mm_fwhm9_crop_float16.pt` and optional FWHM sweep caches | Applies coordinate-to-ALE smoothing, 4 mm mask/crop, max normalization, clamp, and dtype conversion. |
| PubMed unified rows and text positives | `experiments/3dcnn/atlas_free_cnn/data_building/ingest_pubmed_ale.py`, `definitions.py`, `text_registry.py` | PubMed JSONL rows with `tensor_path` and `tensor_index` into the old ALE cache | Does not rebuild PubMed images in the final pack. It indexes the old good cache and attaches title/summary/MeSH positives. |
| Nilearn atlas/network map ingestion | `experiments/3dcnn/atlas_free_cnn/data_building/ingest_nilearn_atlases.py`, `preprocessing.py` | Nilearn-derived NIfTI component maps plus JSONL rows | Fetches atlases, extracts 3D labels/components, resamples to MNI152 2 mm, and writes atlas/network text positives. |
| NeuroVault collection and image preprocessing | `experiments/3dcnn/atlas_free_cnn/data_building/ingest_neurovault.py` | `neurovault_manifest.csv`, `neurovault_text_positives.jsonl`, `neurovault_cnn_volumes.pt`, `neurovault_preprocess_report.json` | Downloads accepted NeuroVault NIfTIs, quality-filters metadata, resamples to 4 mm crop, positive-clips, robust-percentile scales, and records quality flags. |
| Unified map/text rows and splits | `experiments/3dcnn/atlas_free_cnn/data_building/build_unified_dataset.py`, `text_registry.py` | `unified_map_text.jsonl`, `text_registry.jsonl`, `splits/train.jsonl`, `splits/val.jsonl`, `splits/test.jsonl` | Combines source JSONLs, attaches deterministic `text_id`s, and stratifies rows by source. |
| Final shared CNN tensor pack | `experiments/3dcnn/atlas_free_cnn/data_building/pack_atlas_free_cnn_dataset.py` | `atlas_free_cnn_volumes.pt`, `atlas_free_cnn_rows.parquet`, `atlas_free_cnn_text_pairs.parquet`, `atlas_free_cnn_manifest.json` | Loads PubMed tensors directly from `tensor_path`; preprocesses NIfTI rows into the final `(1, 36, 45, 38)` 4 mm crop; selects primary text pairs. |
| Export pack back to training JSONL | `experiments/3dcnn/atlas_free_cnn/data_building/export_hf_pack_jsonl.py` | HF-style `unified_map_text.jsonl`, split JSONLs, `source_counts_by_split.json` | Rewrites every row to point at the shared `atlas_free_cnn_volumes.pt` tensor and packed `tensor_index`. |
| Path portability | `experiments/3dcnn/atlas_free_cnn/data_building/rewrite_jsonl_paths.py`, `experiments/3dcnn/atlas_free_cnn/notebook_utils.py` | Runtime-local JSONL paths, downloaded HF split files | Lets Colab/Drive/local runs resolve the same packed data layout. |
| Raw SPECTER cache completion | `experiments/3dcnn/atlas_free_cnn/data_building/complete_specter_cache.py` | Completed `specter_text_cache.pt` | Encodes missing positive texts for the legacy text-cache pipeline. |
| Normalized SPECTER2 cache build | `experiments/3dcnn/atlas_free_cnn/data_building/build_normalized_specter2_cache.py` | Empty-string-centered, unit-normalized SPECTER2 cache plus audit files | Used by notebook 6a for normalized Stage 3/4 reruns. |
| Dataset QC and preprocessing audit | `experiments/3dcnn/atlas_free_cnn/data_building/qc_dataset.py`, `experiments/3dcnn/atlas_free_cnn/data_building/audit_preprocessing.py` | Source/count/path summaries and `cache/preprocessing_audit.json` | `audit_preprocessing.py` verifies that packed PubMed rows match the old ALE cache. |
| Training-time data loading | `experiments/3dcnn/atlas_free_cnn/training/datasets.py`, `train_autoencoder.py`, `train_ale_cnn.py`, `train_text_to_brain.py` | PyTorch datasets/loaders for Stage 1/3/4 training | These mostly consume preprocessed JSONL/tensor artifacts. `VolumeCollator` clamps/resizes defensively but is not the primary preprocessing pipeline. |

## PubMed ALE cache

The old good PubMed-only 3D CNN cache is:

`experiments/3dcnn/atlas_free_cnn/data/ale_caches/atlas_free_4mm_fwhm9_crop_float16.pt`

Its saved preprocessing config is:

- source: PubMed MNI coordinates
- kernel: Gaussian ALE-style smoothing from coordinates, `fwhm=9.0 mm`
- target resolution: `4.0 mm`
- mask/crop: NeuroVLM brain mask resampled to 4 mm, cropped with slices `[[5, 41], [5, 50], [1, 39]]`
- output shape: `(36, 45, 38)`
- normalization: max-normalize each paper map
- values: clamped to `[0, 1]`
- dtype: `float16`

Related scripts:

- `atlas_free_cnn.training.ale_dataset`: defines `ALEPreprocessConfig`, coordinate smoothing, mask/crop, normalization, and packed cache loading.
- `experiments/3dcnn/atlas_free_cnn/training/train_ale_cnn.py`: older PubMed-only path can call `build_or_load_ale_cache`.
- `experiments/3dcnn/atlas_free_cnn/data_building/build_ale_fwhm_caches.py`: builds atlas-free ALE caches for FWHM sweeps.
- `experiments/3dcnn/atlas_free_cnn/data_building/ingest_pubmed_ale.py`: creates unified PubMed rows that point into this old cache by `tensor_index`.

## HF-style atlas-free CNN pack

The packed mixed dataset is:

`experiments/3dcnn/atlas_free_cnn/cache/hf_atlas_free_cnn/atlas_free_cnn_volumes.pt`

The PubMed rows in that pack are not rebuilt or reprocessed. They are loaded
directly from the old ALE cache by `tensor_path`/`tensor_index`. A direct check
of the first 1000 PubMed rows found max absolute difference `0.0` against the
old cache, so PubMed image preprocessing is identical in the packed dataset.

The regenerated JSONL splits now point all sources at the packed shared CNN
tensor file and include all three intended sources:

- train: PubMed `24526`, NeuroVault `1779`, Nilearn `639`
- val: PubMed `3066`, NeuroVault `221`, Nilearn `79`
- test: PubMed `3066`, NeuroVault `202`, Nilearn `79`

Related scripts:

- `experiments/3dcnn/atlas_free_cnn/data_building/pack_atlas_free_cnn_dataset.py`: packs all accepted rows into `atlas_free_cnn_volumes.pt` and metadata parquet files.
- `experiments/3dcnn/atlas_free_cnn/data_building/export_hf_pack_jsonl.py`: exports the pack back to the split JSONL format consumed by the notebooks.
- `experiments/3dcnn/atlas_free_cnn/data_building/audit_preprocessing.py`: compares PubMed tensors in the pack against the old ALE cache.
- `experiments/3dcnn/atlas_free_cnn/data_building/qc_dataset.py`: summarizes source counts, map types, text-positive categories, shapes, and missing paths.

## NeuroVault images

NeuroVault maps are already images, so no ALE kernel is applied. They are:

- loaded as NIfTI
- resampled to the same 4 mm NeuroVLM mask image used by the ALE cache
- cropped with the same brain crop convention to `(1, 36, 45, 38)`
- masked to brain voxels
- optionally clipped to positive values
- robust-percentile scaled to `[0, 1]`

This is intentionally different from PubMed coordinate processing because
NeuroVault inputs are statistical images, not coordinate lists.

Related scripts:

- `experiments/3dcnn/atlas_free_cnn/data_building/ingest_neurovault.py`: downloads candidate images, reads image/collection metadata, builds text positives, scores quality, and writes staged NeuroVault outputs.
- `ingest_neurovault.py::preprocess_neurovault_nifti`: resamples each accepted NIfTI to the 4 mm mask/crop, detects binary maps for nearest-neighbor interpolation, clips positive-only values by default, robust-percentile scales to `[0, 1]`, and records preprocessing flags.
- `experiments/3dcnn/atlas_free_cnn/data_building/pack_atlas_free_cnn_dataset.py`: can load staged NeuroVault rows from `neurovault_manifest.csv` and `neurovault_text_positives.jsonl` and include them in the final shared tensor pack.

## Nilearn images

Nilearn atlas/network maps are first generated as MNI152 2 mm NIfTI images.
When packed for CNN training, they are resampled to the same 4 mm NeuroVLM mask
and crop as the PubMed ALE cache. Binary atlas regions use nearest-neighbor
resampling; continuous/probabilistic maps use continuous interpolation. No ALE
kernel is applied.

Related scripts:

- `experiments/3dcnn/atlas_free_cnn/data_building/ingest_nilearn_atlases.py`: fetches Nilearn atlases, extracts labels/components, writes processed 2 mm NIfTIs, and creates atlas/network positive text rows.
- `experiments/3dcnn/atlas_free_cnn/data_building/preprocessing.py`: provides `load_target_mni152_2mm`, `resample_to_target`, `clean_array`, `save_nifti`, and metadata helpers used during ingestion.
- `experiments/3dcnn/atlas_free_cnn/data_building/pack_atlas_free_cnn_dataset.py`: converts Nilearn NIfTI rows into the final 4 mm `(1, 36, 45, 38)` tensor shape used by CNN training.

## Text embeddings

Image tensors and text embeddings are built separately. The image pack stores text strings and metadata; Stage 3 and Stage 4 loaders join those strings to an embedding cache.

- Legacy/raw cache completion: `experiments/3dcnn/atlas_free_cnn/data_building/complete_specter_cache.py`.
- Normalized SPECTER2 cache: `experiments/3dcnn/atlas_free_cnn/data_building/build_normalized_specter2_cache.py`.
- Training cache loader: `experiments/3dcnn/atlas_free_cnn/training/train_ale_cnn.py::load_text_embedding_cache`.

## Current conclusion

The suspected PubMed preprocessing mismatch is not present in the packed CNN
data: PubMed volumes are byte-for-byte/effectively identical after tensor load.
The remaining preprocessing risk is source distribution and image-source
normalization, especially NeuroVault robust-percentile scaling producing much
denser maps than PubMed ALE maps.
