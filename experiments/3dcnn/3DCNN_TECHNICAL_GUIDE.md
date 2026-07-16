# Atlas-Free 3D CNN: Data, Models, Training, and Evaluation

This document is the technical review guide for the retained atlas-free 3D CNN
workflow in `experiments/3dcnn`. It describes the code as implemented, including
the exact data artifacts, preprocessing, tensor shapes, model parameters,
training recipes, checkpoint policies, evaluation metrics, and the behavior of
notebooks 1, 2, 3, and 4.

## 1. What the system does

The system learns three related mappings over dense, atlas-free brain volumes:

```text
Stage 1: brain volume -> 384-d latent -> reconstructed brain volume
Stage 3: brain volume -> 384-d shared contrastive space <- SPECTER2 text
Stage 4: SPECTER2 text -> 384-d AE latent -> generated brain volume
```

“Atlas-free” means that the CNN receives the spatial volume directly rather
than a vector of atlas parcel averages. All sources are converted to the same
one-channel MNI crop with shape `[1, 36, 45, 38]`.

There are two experiment tracks:

- Notebook 1 is a standalone PubMed reference experiment using a base-48 CNN.
- Notebook 2 is an independent architecture-validation experiment testing
  whether a retained ResNet48 with multi-scale features and global attention
  can perform well on the global PubMed brain-text retrieval task.
- Notebooks 3 and 4 are the finalized multi-source pipeline using a base-64 CNN
  and controlled mixed-versus-specialized comparisons.

## 2. Notebooks and execution relationships

| Notebook | Stages | Purpose | Dependency |
| --- | --- | --- | --- |
| `1 best contrastive recipe on pubmed.ipynb` | 1, checkpoint selection, 3 | Reproduce the known-best PubMed-only AE-pretrained contrastive recipe | Independent |
| `2 resnet48 multi scale attention.ipynb` | matching AE pretraining, 3 | Validate the retained ResNet48 multi-scale-attention architecture on global brain-text retrieval | Independent; not consumed by notebooks 1, 3, or 4 |
| `3 multi source autoencoder.ipynb` | 1A, 1B, reconstruction eval | Train one mixed AE and three domain-specialized AEs | Uses finalized unified data |
| `4 multi source stage3 stage4.ipynb` | resource validation, 3, 4, final comparisons | Run six controlled contrastive and generation branches | Uses locked AEs and finalized unified data |

Notebook 3 logically explains where the four AE branches come from. In the
current implementation, Notebook 4 loads four locked AEs through
`neurovlm.retrieval_resources`, writes them as local checkpoints, and validates
those checkpoints. It does not automatically find Notebook 3's latest
`selected_ae_checkpoints.json`. This is intentional in the current downstream
notebook because it makes the comparison inputs immutable and explicit.

## 3. Finalized multi-source dataset

The packaged dataset is `neurovlm/atlas_free_cnn_dataset`. The authoritative
[`atlas_free_cnn_manifest.json`](https://huggingface.co/datasets/neurovlm/atlas_free_cnn_dataset/blob/main/atlas_free_cnn_manifest.json)
reports the counts below. Image preprocessing parameters are independently
recorded in
[`preprocessing_audit.json`](https://huggingface.co/datasets/neurovlm/atlas_free_cnn_dataset/blob/main/preprocessing_audit.json).

| Quantity | Value |
| --- | ---: |
| Maps | 33,657 |
| Packed tensor shape | `[33,657, 1, 36, 45, 38]` |
| Packed dtype | float16 |
| PubMed maps | 30,658 |
| Nilearn maps | 797 |
| NeuroVault maps | 2,202 |
| Train | 26,944 |
| Validation | 3,366 |
| Test | 3,347 |
| Map-text pairs | 103,800 |
| Primary texts / normalized SPECTER2 vectors | 33,657 |

The serialized data surface is:

- `atlas_free_cnn_volumes.pt`: dense volume tensor plus map IDs and packing
  metadata;
- `train.jsonl`, `val.jsonl`, `test.jsonl`: one row per map, each pointing to a
  `tensor_index` in the shared tensor;
- `atlas_free_cnn_rows.parquet`: map-level metadata;
- `atlas_free_cnn_text_pairs.parquet`: all positive text pairs and their rank;
- `atlas_free_cnn_manifest.json`: counts, configuration, and file names;
- `preprocessing_audit.json`: image-space parameters and numerical checks;
- `specter2_stage3_stage4_emptycentered_unitnorm.pt` plus metadata, index, and
  validation sidecars.

The text-cache model revisions and numerical checks are published in its
[`metadata`](https://huggingface.co/datasets/neurovlm/atlas_free_cnn_dataset/blob/main/specter2_stage3_stage4_emptycentered_unitnorm_metadata.json)
and
[`validation`](https://huggingface.co/datasets/neurovlm/atlas_free_cnn_dataset/blob/main/specter2_stage3_stage4_emptycentered_unitnorm_validation.json)
sidecars.

### 3.1 JSONL row contract

A training row supplies:

- `map_id`;
- `source` and `source_detail`;
- `map_type` and `space`;
- `tensor_path` and `tensor_index` (or, for compatible external rows, a
  `nifti_path`);
- `positive_texts`, with text, text ID, category, source, weight, and
  reliability;
- publication/collection identifiers and negative-sampling groups where
  available;
- preprocessing and quality metadata.

`UnifiedMapTextDataset` resolves relative paths against the working directory,
JSONL parents, and `NEUROVLM_DATA_ROOT`, `NEUROVLM_DRIVE_ROOT`, and
`NEUROVLM_REPO_DIR`. It caches each shared tensor payload in memory and returns
one volume and all positive texts per row.

### 3.2 PubMed source

The PubMed part contains one coordinate-derived ALE-style map per paper:

1. MNI coordinates are grouped by PMID.
2. Unique coordinates are placed as impulses on an MNI brain mask resampled to
   4 mm isotropic resolution.
3. The impulses are smoothed with a 9 mm FWHM Gaussian. This corresponds to
   `sigma = FWHM / 2.35482 = 3.822 mm`, or `0.9555` voxels at 4 mm.
4. The map is masked, cropped from the full `[46,55,46]` grid using slices
   `[5:41, 5:50, 1:39]`, producing `[36,45,38]`.
5. NaN/Inf values are replaced by zero, negative values are clamped, and each
   nonempty map is divided by its maximum into `[0,1]`.
6. The finalized cache is stored as float16.

The affine recorded by the audit is:

```text
[[4, 0, 0, -90],
 [0, 4, 0, -126],
 [0, 0, 4, -72],
 [0, 0, 0, 1]]
```

Text positives can include a title, a wiki-style paper summary, and selected
MeSH-derived anatomical, cognitive, or disorder terms. The primary-text
priority puts summary-style text before title-only text.

### 3.3 Nilearn source

The 797 Nilearn examples are atlas regions or network/component maps:

| Atlas | Maps |
| --- | ---: |
| Schaefer 2018 | 400 |
| BASC 064 | 64 |
| DiFuMo 64 | 64 |
| Juelich max-probability | 62 |
| Juelich probabilistic | 62 |
| Harvard-Oxford cortical | 48 |
| MSDL | 39 |
| Harvard-Oxford subcortical | 21 |
| Smith 2009 | 20 |
| Yeo 2011 | 17 |

Label maps are split into one binary volume per non-background label;
probabilistic/ICA maps are split along their component axis. Binary volumes use
nearest-neighbor resampling and continuous volumes use continuous
interpolation. Human-readable label text is paired with a known network
definition when available, or with a conservative atlas-region fallback.
During final packing, every map is converted to the common 4 mm crop and
scaled to `[0,1]`.

### 3.4 NeuroVault source

The finalized package contains 2,202 strong-quality NeuroVault statistical
maps and excludes weak-tier maps. Collection representation is capped at 50
accepted maps per collection.

Preparation performs the following checks and transforms:

1. Reject surface/CIFTI-like files and non-3D shapes (except singleton 4D).
2. Resample to the 4 mm MNI brain mask, using nearest interpolation for binary
   maps and continuous interpolation otherwise.
3. Replace NaN/Inf, record whether negative values were present, clamp to the
   positive part, and apply the brain mask.
4. Crop to `[36,45,38]`.
5. Reject maps whose positive nonzero fraction is below 0.001.
6. Robustly scale positive brain values using the 1st and 99.5th percentiles,
   then clip to `[0,1]`.
7. Score metadata quality using image name/description, task or contrast,
   DOI/PMID, collection description, and Cognitive Atlas fields; penalize
   missing metadata, suspicious thresholding, failed resampling, nearly empty
   maps, and low-quality text.

Text candidates include image name/description, task or contrast labels,
collection name/description, Cognitive Atlas metadata, and publication title.
The packer ranks these candidates deterministically rather than using model
similarity, avoiding model-based target leakage.

### 3.5 Splits and sampling

The finalized split counts are 80.1% train, 10.0% validation, and 9.9% test.
The original unified builder used seed 42 and stratified random shuffling by
the full source string, then shuffled the combined split. The notebooks use the
published split files directly and fingerprint them before downstream runs.

Stage 1A uses natural sampling. There is no inverse-frequency weighting or
balanced source sampler. Consequently, PubMed dominates the mixed AE exposure
in proportion to its count. Stage 1B and every Stage 3/4 branch filter all
three splits to a single canonical domain.

## 4. Text preparation

The canonical cache embeds exactly the primary `positive_texts[0].text` string
for each map with no extra separator rewriting.

| Item | Setting |
| --- | --- |
| Encoder | `allenai/specter2_aug2023refresh` |
| Base | `allenai/specter2_aug2023refresh_base` |
| Adapter | `allenai/specter2_aug2023refresh_adhoc_query` |
| Adapter name | `adhoc_query` |
| Pooling | Last hidden state's CLS token |
| Max length | 512 tokens |
| Raw dimension | 768 |
| Output convention | Empty-string centered, L2 unit normalized |
| Output vectors | 33,657 |

For raw vector `e(text)` and an empty-string reference `e("")`, the stored
vector is:

```text
u(text) = (e(text) - e("")) / ||e(text) - e("")||2
```

The builder validates that all required text IDs are present, no NaN/Inf values
exist, the dimension is 768, and at least 99.9% of norms are within `1e-3` of
one. The published validation reports every vector within `1e-4` of unit norm.

## 5. Plain 3D CNN architecture

### 5.1 Encoder

Each plain block is:

```text
Conv3d(kernel=3, padding=1, stride=1)
 -> GroupNorm(up to 8 groups)
 -> GELU
 -> MaxPool3d(kernel=2)
```

For the base-64 multi-source model:

| Step | Tensor shape |
| --- | --- |
| Input | `[B, 1, 36, 45, 38]` |
| Block 1 | `[B, 64, 18, 22, 19]` |
| Block 2 | `[B, 128, 9, 11, 9]` |
| Block 3 | `[B, 256, 4, 5, 4]` |
| Block 4 | `[B, 512, 2, 2, 2]` |
| Adaptive average pool | `[B, 512, 1, 1, 1]` |
| Flatten + dropout 0.1 + linear | `[B, 384]` |

There is no output activation or L2 normalization in the encoder itself.
InfoNCE normalizes Stage 3 embeddings inside the loss.

### 5.2 Decoder

The decoder maps the 384-dimensional latent through a linear layer to a learned
seed of `[B,512,3,3,3]`. Four transposed-convolution blocks with kernel 2 and
stride 2 progressively reduce channels to 64 while doubling the grid. A final
3x3 Conv3D produces one channel. Because `[36,45,38]` is not a pure power-of-two
multiple of the seed, the result is trilinearly interpolated to the exact
target shape.

The decoder output is unconstrained. Stage 1 raw-MSE training operates on the
raw output; metrics clamp it to `[0,1]`. Stage 4 also trains through the raw
decoder and clamps for spatial metrics and the main semantic evaluation.

### 5.3 Parameter counts

| Model | Encoder | Decoder | Total |
| --- | ---: | ---: | ---: |
| Notebook 1 base-48 AE | 2,764,032 | 4,786,705 | 7,550,737 |
| Multi-source base-64 AE | 4,846,464 | 6,734,529 | 11,580,993 |
| Stage 4 text-to-latent projector | — | — | 590,720 |

The Stage 4 projector count comes from `768*512 + 512 + 512*384 + 384`.

## 6. Notebook 1 in depth

Notebook 1 writes to a timestamped
`runs_ale_3dcnn_best_stage1_stage3` directory and runs three operational steps.

### 6.1 Stage 1 PubMed AE

| Setting | Value |
| --- | --- |
| Data | PubMed coordinate ALE cache only |
| Resolution / smoothing | 4 mm / 9 mm FWHM |
| Architecture | Plain, base 48, 4 blocks, latent 384 |
| Epochs | 150 |
| Batch | Auto-select from 256 down to 4 |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| AMP | Enabled on CUDA; bfloat16 when supported, otherwise float16 |
| Gradient clipping | Global norm 1.0 |
| Validation | Every 5 epochs |
| Early stopping | 20 unsuccessful validation checks |
| Checkpoint | Minimum validation MSE |

The notebook passes weighted reconstruction and overlap arguments, but the
current standalone trainer calls raw `F.mse_loss(pred, x)` in its epoch loop.
Therefore those passed sparse-loss values are provenance-only in the current
implementation.

### 6.2 Checkpoint selection

Selection is file-priority based:

1. `best_cnn_autoencoder.pt`;
2. `best_val_mse.pt`;
3. `last_cnn_autoencoder.pt`.

The first existing nonempty file is recorded in
`02_stage2_selected_autoencoder_checkpoint.json`.

### 6.3 Stage 3 PubMed contrastive training

| Setting | Value |
| --- | --- |
| Domain | PubMed |
| Encoder init | Selected Stage 1 encoder, strict load |
| Text init | Pretrained NeuroVLM `text_infonce` projection |
| Trainability | CNN and text projection both trainable |
| Epochs | 200 |
| Batch | Auto-select from 2048 down to 4 |
| CNN lr | `1e-4` |
| Text projection lr | `1e-5` |
| Weight decay | `1e-4` |
| Schedule | 5-epoch warmup + cosine decay |
| Loss | Symmetric InfoNCE |
| Temperature | 0.07 |
| Validation | Every 5 epochs |
| Early stopping | Patience 25 |

The similarity matrix is `normalize(brain) @ normalize(text).T / 0.07`.
Cross-entropy is applied in both directions using the diagonal as the correct
pair and then averaged.

## 7. Notebook 3 in depth

### 7.1 Stage 1A mixed AE

| Setting | Value |
| --- | --- |
| Data mode | `mixed` |
| Source sampling | Natural |
| Architecture | Plain base-64 AE, latent 384 |
| Loss | Raw MSE, no output activation |
| Epochs | 300 maximum |
| Batch | Start 64; preflight candidates 512→16; notebook retries lower maxima on OOM |
| VRAM reserve | 12 GiB |
| Optimizer | AdamW, lr `3e-4`, weight decay `1e-4` |
| AMP / clipping | Enabled / 1.0 |
| Validation metrics | Up to 16 validation batches during epochs |
| Early stopping | Validation loss, patience 15 |
| Selection | `best_val_loss.pt` |

### 7.2 Stage 1B domain fine-tunes

Three independent jobs start from Stage 1A. Each filters train, validation, and
test to one canonical source and trains all AE parameters.

| Setting | Value |
| --- | --- |
| Domains | PubMed, Nilearn, NeuroVault |
| Epochs | 100 maximum |
| Batch | Start 64; candidates 96→16 with OOM retry |
| VRAM reserve | 28 GiB |
| Optimizer | AdamW, lr `1e-4`, weight decay `1e-4` |
| Loss | Raw MSE |
| Freeze mode | None |
| Early stopping | Patience 15 using inherited validation-loss policy |
| Selection | `best_top5_dice.pt` |

The trainer saves best checkpoints for multiple metrics, but the notebook
chooses one explicit metric for each finalized branch. It then evaluates only
that selected checkpoint on validation and test, separated by source and source
detail.

## 8. Notebook 4 in depth

### 8.1 Preflight and six-branch manifest

Notebook 4 materializes four locked resource models:

```text
mixed_stage1a
mixed_to_pubmed_stage1b
mixed_to_nilearn_stage1b
mixed_to_neurovault_stage1b
```

The selection integration verifies that each checkpoint exists and contains
both `encoder.*` and `decoder.*` tensors, computes a deterministic model-state
checksum, and writes one manifest row for each of these six runs:

```text
mixed_stage1a_on_pubmed              mixed_to_pubmed_stage1b_on_pubmed
mixed_stage1a_on_nilearn              mixed_to_nilearn_stage1b_on_nilearn
mixed_stage1a_on_neurovault           mixed_to_neurovault_stage1b_on_neurovault
```

It also records train/validation/test fingerprints so baseline and specialized
runs inside a domain can be proven to use identical examples.

### 8.2 Stage 3 controlled contrastive recipe

Every branch uses the same recipe except for domain and AE initialization:

| Setting | Value |
| --- | --- |
| Architecture | Plain base 64, 4 blocks, output 384 |
| AE initialization | Exact registered branch encoder; strict state load |
| Text projection | Pretrained `text_infonce`, 768→512→384 |
| Epochs | 150 maximum |
| Batch | 512 |
| CNN lr | `1e-4` |
| Projection lr | `1e-5` |
| Weight decay | `1e-4` |
| Warmup | 5 epochs |
| Schedule | Cosine decay after warmup |
| Objective | Symmetric InfoNCE |
| Temperature | 0.07 |
| AMP / clipping | Enabled / 1.0 |
| Validation | Every epoch by default (`NEUROVLM_STAGE3_VAL_INTERVAL`) |
| Early stopping | Patience 25 |
| Monitor | Mean bidirectional normalized recall-curve AUC |

The controlled recipe fails early if the checkpoint architecture differs from
base channels 64, four blocks, 384 output, dropout 0.1, GroupNorm, max pooling,
plain architecture, and `[36,45,38]` input.

### 8.3 Stage 4 corrected generation recipe

The Stage 4 dataflow is:

```text
normalized SPECTER2 [B,768]
 -> Linear(768,512) -> ReLU -> Linear(512,384)
 -> predicted raw AE latent [B,384]
 -> frozen exact-branch AE decoder
 -> raw prediction [B,1,36,45,38]
```

The AE encoder generates the target latent under `no_grad`. The loss is:

```text
L = 1.0 * MSE(reconstruction, target)
  + 1.0 * MSE(predicted_latent, target_AE_latent)
```

The reconstruction term is unweighted because `alpha=0`; Dice, top-k,
correlation, and latent-cosine coefficients are zero. Although the Stage 3
checkpoint is recorded and may be loaded for semantic validation, none of its
projection tensors initialize the Stage 4 projector.

| Setting | Value |
| --- | --- |
| Trainable parameters | Fresh projector only, 590,720 |
| Frozen | AE encoder and decoder; Stage 3 encoder and projection |
| Epochs | 200 maximum |
| Batch | 1024 default; preflight candidates 2048→64 and runtime OOM fallback |
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| AMP | Enabled |
| Validation | Every epoch |
| Early stopping | Validation top-5 Dice, patience 25 |
| Primary checkpoint | `best_val_top5_dice.pt` |
| Secondary | `best_val_spatial_corr.pt`, `last.pt` |
| Semantic AUC during training | Disabled by default; interval 5 if enabled |

## 9. Evaluation definitions

### 9.1 Spatial reconstruction/generation

Predictions and targets are sanitized and clamped to `[0,1]` for metrics. Dice
uses independently selected top activation fractions after per-sample positive
max normalization.

| Metric | Meaning |
| --- | --- |
| MSE / MAE | Dense error over all voxels |
| Foreground MSE | MSE where target is greater than zero |
| Spatial correlation | Pearson-style correlation over flattened valid voxels |
| Top-1/5/10 Dice | Dice overlap between predicted and target top-k% voxel masks |
| Voxel AUROC | Ranks voxels against a target activation threshold, optional |
| Nonzero fraction | Prediction/target sparsity diagnostic |

### 9.2 Stage 3 retrieval

For `N` evaluation examples, normalized brain and text embeddings form an
`N x N` cosine-similarity matrix. Metrics are computed in both text→brain and
brain→text directions:

- recall@1, 5, 10, 50;
- mean reciprocal rank;
- median best-positive rank;
- full recall@k curve AUC, averaging recall at every k from 1 to N with x-axis
  `k/N`.

Expected random AUC is approximately 0.5; a perfect ranking produces 1.0.

### 9.3 Stage 4 semantic evaluation

Generated volumes are re-encoded by the matching frozen Stage 3 brain encoder
and compared with the matching Stage 3 text projection. The evaluator reports:

- raw and clamped generated-map retrieval;
- strict map identity positives;
- same-primary-text group positives;
- publication/collection group positives;
- matched text/generated-map cosine versus a shuffled null;
- the same bidirectional recall/AUC family used for Stage 3.

The main alias points to clamped strict-map mean bidirectional AUC. Raw output
range and fractions below zero/above one are retained as diagnostics.

### 9.4 Controlled comparison policy

The supported inference is within domain:

```text
mixed Stage 1A initialization vs matching domain-specialized Stage 1B
```

Both sides must have equal split fingerprints, domain labels, text-cache
checksum, optimizer/schedule, objective, metric implementation, and random seed
where practical. Raw PubMed-versus-Nilearn-versus-NeuroVault rankings are not a
controlled specialization comparison because task distributions differ.

## 10. Output and provenance contract

Runs record enough information to reproduce and audit component matching:

- effective config and environment;
- git information;
- split paths and SHA-256 fingerprints;
- text-cache path, checksum, model/adapter identity, and normalization;
- selected AE path, stage, domain, selection reason, and state checksum;
- architecture compatibility and strict-load report;
- trainable/frozen parameter report;
- histories, timing, selected checkpoints, and checkpoint manifest;
- validation and held-out metrics;
- per-example Stage 4 metrics and generated-map manifest;
- within-domain and all-domain summary CSVs.

The test split is never used for gradients, checkpoint selection, early
stopping, or hyperparameter decisions.

## 11. Core code review path

Review in this order:

| Review question | Path | Key symbols |
| --- | --- | --- |
| What is the CNN/AE structure? | `../../src/neurovlm/ale_cnn.py` | `_ConvBlock`, `ALE3DCNNEncoder`, `ALE3DCNNDecoder`, `ALE3DCNNAutoEncoder`; the experiment path re-exports the public classes |
| How is the multi-source AE trained? | `atlas_free_cnn/training/train_autoencoder.py` | `VolumeCollator`, `run_epoch`, `train_from_config`, `train_stage1b_from_config` |
| What exactly is the Stage 1 loss? | `atlas_free_cnn/training/autoencoder_losses.py` | `AutoencoderLossConfig`, `reconstruction_loss` |
| How is Stage 3 initialized/trained? | `atlas_free_cnn/training/train_ale_cnn.py` | `_load_encoder_from_autoencoder_checkpoint`, `ALETrainer`, `bidirectional_retrieval_metrics` |
| What is symmetric InfoNCE? | `src/neurovlm/loss.py` | `InfoNCELoss` |
| How is corrected Stage 4 implemented? | `atlas_free_cnn/training/train_text_to_brain.py` | `apply_checkpoint_architecture`, `run_epoch`, `train_from_config` |
| What is the Stage 4 projector? | `atlas_free_cnn/training/model_wrappers.py` | `GenerativeTextToAELatent`, `build_generative_text_to_ae_latent` |
| What are the generative losses? | `atlas_free_cnn/training/generation_losses.py` | `latent_alignment_loss`, `weighted_reconstruction_loss`, `combined_generation_loss` |
| How are volumes and JSONL loaded? | `atlas_free_cnn/training/datasets.py` | `UnifiedMapTextDataset` |
| How is natural sampling defined? | `atlas_free_cnn/training/source_sampling.py` | `canonical_source`, `build_source_sampler` |
| How are checkpoints selected? | `atlas_free_cnn/training/checkpointing.py` | `metric_direction`, `CheckpointManager` |
| What are spatial metrics? | `atlas_free_cnn/evaluation/generation_metrics.py` | `generation_metrics`, `voxel_auroc` |
| What are retrieval metrics? | `atlas_free_cnn/evaluation/metrics.py` | `ranking_metrics`, `normalized_recall_curve_auc_from_ranks` |
| How are generated maps semantically evaluated? | `atlas_free_cnn/evaluation/stage4_semantic.py` | `semantic_positive_masks`, `semantic_retrieval_metrics`, `evaluate_generation_semantic_loader` |
| Where is the six-run matrix fixed? | `atlas_free_cnn/conventions.py` | `six_branch_specs`, `stage_output_dir` |
| How are AE inputs validated? | `atlas_free_cnn/stage1_selection_integration.py` | `_validate_checkpoint`, `integrate_completed_stage1_selection` |
| How is the text cache built? | `atlas_free_cnn/data_building/build_normalized_specter2_cache.py` | `collect_text_records`, `encode_texts_specter2`, `validate_output` |
| What notebook-specific discovery occurs? | `atlas_free_cnn/notebook_utils.py` | split discovery, cache resolution/validation, subprocess streaming |

## 12. Known review caveats

1. Notebook 1's legacy sparse-loss CLI flags do not affect the current
   standalone AE epoch loop; its effective loss is raw MSE.
2. Notebook 3 writes a fresh AE registry, but Notebook 4 currently uses locked
   resource-loader AEs and creates a new validated manifest from them.
3. The public Hugging Face viewer can fail to auto-cast the heterogeneous JSONL
   schema. The notebooks read/download raw JSONL directly, so the viewer is not
   in the training path.
4. The original image ingestion/build scripts were removed after finalization.
   The active repository retains the normalized text-cache builder; image-data
   provenance is preserved in the published manifest/audit and in git history.
5. Notebook 2 is retained as evidence for the architecture decision: it tests
   whether multi-scale spatial features plus an attention-pooled global token
   let a CNN perform well on the global PubMed retrieval task. It is not part
   of the production Notebook 1/3/4 workflow and does not feed their outputs.
