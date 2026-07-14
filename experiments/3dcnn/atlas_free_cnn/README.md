# Atlas-Free 3D CNN

This package implements the retained dense-volume 3D CNN experiments for
brain-map reconstruction, brain/text contrastive retrieval, and text-to-brain
generation.

The documented workflow uses:

1. `../1 best contrastive recipe on pubmed.ipynb` for the standalone PubMed
   reference experiment;
2. `../3 multi source autoencoder.ipynb` for mixed-source Stage 1A and three
   domain-specific Stage 1B autoencoders;
3. `../4 multi source stage3 stage4.ipynb` for the fixed six Stage 3
   contrastive branches and six matching Stage 4 text-to-brain branches.

Start with [`../3DCNN_TECHNICAL_GUIDE.md`](../3DCNN_TECHNICAL_GUIDE.md) for the
full technical design, data provenance, parameter tables, training recipes,
evaluation rules, and code-review paths. A standalone rendered version is
available at [`../3DCNN_TECHNICAL_GUIDE.html`](../3DCNN_TECHNICAL_GUIDE.html).

## Finalized Data

The multi-source workflow consumes `neurovlm/atlas_free_cnn_dataset`:

- 33,657 maps: 30,658 PubMed, 797 Nilearn, and 2,202 NeuroVault;
- one-channel `36 x 45 x 38` volumes in the `MNI152_4mm_crop` geometry;
- 26,944 train, 3,366 validation, and 3,347 test examples;
- 103,800 map-text pairs;
- one 768-dimensional normalized SPECTER2 vector for each of 33,657 primary
  texts.

The volume package and JSONL splits are discovered locally/through Drive first
and downloaded from Hugging Face when absent. The only retained offline builder
is `data_building/build_normalized_specter2_cache.py`; the finalized image-data
builders were removed after their outputs and audits were published.

## Retained Architecture

The finalized multi-source model is a plain four-block 3D CNN:

```text
input [B,1,36,45,38]
 -> channels 64 -> 128 -> 256 -> 512
 -> adaptive average pooling
 -> 384-dimensional latent
 -> learned 3x3x3 decoder seed
 -> four transposed-convolution upsampling blocks
 -> output [B,1,36,45,38]
```

Every encoder block is Conv3D → GroupNorm → GELU → 2x max pooling. The encoder
uses dropout 0.1 before its final linear projection. The decoder ends with an
unconstrained one-channel Conv3D output; training loss sees the raw output,
while reconstruction metrics clamp predictions to `[0, 1]`.

Parameter counts:

| Component | Parameters |
| --- | ---: |
| Encoder | 4,846,464 |
| Decoder | 6,734,529 |
| Autoencoder | 11,580,993 |
| Fresh Stage 4 768→512→384 projector | 590,720 |

The separately retained ResNet48 multi-scale-attention variant lives in
`ale_cnn.py` and Notebook 2, but it is not part of the requested Notebook
1→3→4 workflow.

## Stage Summary

| Stage | Trainable modules | Objective | Primary selection |
| --- | --- | --- | --- |
| 1A mixed AE | Encoder + decoder | Raw voxel MSE | Minimum validation loss |
| 1B domain AE | Encoder + decoder, initialized from 1A | Raw voxel MSE | Maximum validation top-5 Dice |
| 3 contrastive | AE-initialized encoder + pretrained text projection | Symmetric InfoNCE, temperature 0.07 | Maximum mean bidirectional normalized recall-curve AUC |
| 4 generation | Fresh 768→512→384 projector only; AE frozen | Latent alignment + reconstruction MSE | Maximum validation top-5 Dice |

Stage 1A samples the natural mixed distribution. Stage 1B creates independent
PubMed, Nilearn, and NeuroVault fine-tunes. Stage 3 and Stage 4 each compare the
mixed baseline with its matching specialization inside each domain, producing
six controlled branches.

## Package Layout

- `training/ale_cnn.py`: encoder, decoder, autoencoder, and retained ResNet.
- `training/train_autoencoder.py`: multi-source Stage 1 training and evaluation.
- `training/train_ale_cnn.py`: Stage 3 contrastive training and retrieval eval.
- `training/train_text_to_brain.py`: corrected Stage 4 training and final eval.
- `training/model_wrappers.py`: text projection and model-loading helpers.
- `training/autoencoder_losses.py`: retained raw-MSE AE objective.
- `training/generation_losses.py`: Stage 4 reconstruction/latent loss terms.
- `training/datasets.py`: unified JSONL and shared tensor loader.
- `evaluation/metrics.py`: rank, recall, MRR, and normalized AUC definitions.
- `evaluation/generation_metrics.py`: spatial reconstruction metrics.
- `evaluation/stage4_semantic.py`: semantic evaluation of generated maps.
- `conventions.py`: fixed domains, branches, cache convention, and paths.
- `stage1_selection_integration.py`: AE validation and six-run manifest.
- `notebook_utils.py`: artifact discovery, download, cache validation, and
  notebook subprocess helpers.
- `data_building/build_normalized_specter2_cache.py`: canonical SPECTER2 cache
  builder.

## Review Caveats

- Notebook 4 uses the locked AE resource loaders in
  `neurovlm.retrieval_resources`. It validates and materializes those resources
  locally; it does not automatically consume a fresh Notebook 3 checkpoint
  registry.
- Notebook 1 passes legacy sparse-loss flags to
  `training/train_ale_cnn_autoencoder.py`, but that trainer's effective loss is
  direct raw MSE.
- The Hugging Face dataset viewer currently encounters a heterogeneous JSONL
  schema cast issue. The notebooks download/read the raw JSONL files directly,
  so training does not depend on the viewer conversion.
