# Preprint Figure Reproduction

This directory contains the evaluation notebooks used to reproduce figures in
NeuroVLM preprint v3:
https://www.biorxiv.org/content/10.64898/2026.02.06.704508v3

The canonical pipeline is `02_data/` -> `03_models/` -> `figures/`. Data and
model notebooks remain in their upstream directories so that each notebook has
one canonical location.

## Core dependencies

- Data preprocessing: `../02_data/`
- Model training and model analyses: `../03_models/`
- Installation and data access: `../installation.md`

Most evaluation notebooks load released datasets, embeddings, and pretrained
models through the `neurovlm` package. Rerun upstream preprocessing or training
only when the required artifact is unavailable.

## Figure map

| Figure | Canonical notebook | Purpose and dependency notes |
|---|---|---|
| Fig. 2 | `../03_models/14_qformer_anat.ipynb` | Anatomical Query-Former model analysis; retained with the canonical model notebooks. |
| Fig. 3 | `12_neurovault_decoding.ipynb` | NeuroVault text-brain decoding examples and cached generated descriptions. |
| Fig. 4 | `20_pubmed_cv.ipynb` | PubMed cross-validation, retrieval evaluation, and training-run outputs. |
| Fig. 4 | `22_text_to_brain_metrics.ipynb` | Text-to-brain generation metrics; currently missing from the repository. |
| Fig. 4 | `24_brain_to_text_pubmed.ipynb` | Brain-to-text PubMed generation and metric summaries. |
| Fig. 5 | `11_autoencoder.ipynb` | Autoencoder reconstruction metrics for PubMed and NeuroVault data. |
| Fig. 5 | `13_network_labeling.ipynb` | Network-label confusion and one-vs-rest evaluation. |
| Fig. 6 | `16_qualatative_auto.ipynb` | Qualitative autoencoder reconstructions across network, PubMed, and NeuroVault inputs. |
| Fig. S2 | `../02_data/01_coordinate.ipynb` | Coordinate smoothing and DiFuMo projection; retained with canonical data preprocessing. |
| Fig. S3 | `23_versus_others.ipynb` | Recall comparison with external NiCLIP, NeuroConText, and language-model baselines. |
| Fig. S4 | `19_ica_networks.ipynb` | ICA network labeling analyses for HCP and UK Biobank maps. |

The table records notebook-level provenance. It does not assign individual
panels where the notebook does not identify them explicitly.

## Recommended execution order

1. Install NeuroVLM and make the packaged or downloaded data available.
2. Run `../02_data/01_coordinate.ipynb` only if the coordinate-derived artifacts
   needed for Fig. S2 are missing.
3. Load released model artifacts; run `../03_models/14_qformer_anat.ipynb` when
   reproducing the Fig. 2 model analysis.
4. Run the notebooks in this directory for the desired figures.
5. Collect generated caches, metric tables, and figures from
   `docs/figures/outputs/` when running from the repository root, or from a
   local `outputs/` directory when running inside `docs/figures/`.

`20_pubmed_cv.ipynb` performs cross-validation training and is substantially
more expensive than notebooks that only load released artifacts.
`19_ica_networks.ipynb` and `24_brain_to_text_pubmed.ipynb` currently select
CUDA explicitly. The decoding, generation, and external-baseline notebooks may
also require a GPU, model downloads, or separately installed baseline projects.

## Missing or external dependencies

> TODO: Fig. 4 references `22_text_to_brain_metrics.ipynb`, but that notebook
> was not found anywhere in the repository during this refactor. It must be
> recovered from the original analysis source before Fig. 4 is fully
> reproducible from this tree.

`23_versus_others.ipynb` expects external NiCLIP and NeuroConText resources;
their locations can be configured with the environment variables documented in
that notebook. Some generation notebooks also cache outputs from large language
models rather than retraining those models from scratch.
