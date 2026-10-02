# NeuroVLM codebase map

This is a navigation guide for collaborators and coding agents working with the
refactored repository. Paths are relative to the repository root. The map was
checked against the local `main` checkout on October 2, 2026; source files remain
the authority as the project evolves.

NeuroVLM maps between neuroscience text and neuroimaging activation maps. The
Python package lives in `src/neurovlm/`. Notebooks in `docs/` contain tutorials,
data preparation, model development, and paper analyses.

## Start here

| Goal | First place to read |
|---|---|
| Install and try the package | [README.md](README.md), [installation guide](docs/installation.md), [quickstart notebook](docs/01_tutorials/00_quickstart.ipynb) |
| Understand the public inference API | [core/client.py](src/neurovlm/core/client.py) for `NeuroVLM`; [core/runtime.py](src/neurovlm/core/runtime.py) for `load_pipeline` |
| Find or change a model | [models/registry.py](src/neurovlm/models/registry.py), then `models/` or `cnn/` |
| Find datasets, embeddings, masks, or released weights | [data/loaders.py](src/neurovlm/data/loaders.py), then [resources/loaders.py](src/neurovlm/resources/loaders.py) |
| Train a model | [training/README.md](src/neurovlm/training/README.md), then the task module in `training/` |
| Evaluate models or change metrics | [evaluation/README.md](src/neurovlm/evaluation/README.md), then `evaluation/` and `metrics/` |
| Reproduce a paper figure | [figure map](docs/figures/README.md) |
| Work on atlas-free CNNs | [CNN tutorial](docs/01_tutorials/06_atlas_free_cnn.ipynb), [technical guide](docs/experimental/cnn/technical_guide.md) |
| Check a change | [tests/README.md](tests/README.md), tests for the affected subsystem |
| Edit notebooks or build documentation | [docs/README.md](docs/README.md) |
| Read the previous refactor verification results | [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md); these are historical results, not a fresh test run |

## Repository layout

```text
neurovlm/
├── README.md                     Installation, examples, project overview
├── CODEBASE_MAP.md               This navigation guide
├── REPRODUCIBILITY_REPORT.md      Refactor verification history and limitations
├── pyproject.toml                Dependencies, extras, packaging, pytest markers
├── LICENSE                       Apache-2.0 license
├── src/neurovlm/                 Importable Python package
│   ├── core/                     High-level client and task-oriented runtime
│   ├── models/                   MLP, text encoder, adapter, QFormer, registry
│   ├── cnn/                      Atlas-free 3D CNN architectures and wrappers
│   ├── data/                     Public data loaders and dataset preparation
│   ├── resources/                Internal Hugging Face resource loaders
│   ├── retrieval/                Custom-corpus search and LLM summaries
│   ├── training/                 Typed configs and training entry points
│   ├── evaluation/               Task evaluators and model comparisons
│   ├── metrics/                  Ranking, semantic, text, and spatial metrics
│   ├── pipelines/                Run artifacts, checkpoints, provenance, resume
│   └── utils/                    Progress-display helper
├── docs/
│   ├── 01_tutorials/             User-facing examples
│   ├── 02_data/                  Canonical preprocessing notebooks
│   ├── 03_models/                Canonical model-development notebooks
│   ├── figures/                  Preprint-v3 analyses and figure reproduction
│   ├── experimental/             Retained research and supporting preparation
│   ├── api.rst                   Generated API reference configuration
│   ├── conf.py                   Sphinx configuration
│   ├── _static/                  Documentation styling
│   └── img/                      Documentation logos
├── scripts/                      Notebook export and smoke-test commands
├── tests/
│   ├── unit/                     Isolated subsystem tests
│   ├── integration/              Inference, evaluation, and training workflows
│   └── notebooks/                Notebook export/execution smoke harness
└── .github/workflows/            Tests, documentation deployment, publishing
```

## Python package: where implementations live

### Public API and inference: `core/`

| File | Responsibility |
|---|---|
| [src/neurovlm/__init__.py](src/neurovlm/__init__.py) | Lazy top-level exports: `NeuroVLM`, `load_pipeline`, runtime metadata, and atlas-free dataset/provider APIs. |
| [core/client.py](src/neurovlm/core/client.py) | `NeuroVLM`, chained `.text(...).to_brain(...)` / `.brain(...).to_text(...)` calls, dataset aliases, and result wrappers (`TextSearchResult`, `BrainSearchResult`, `BrainTopKResult`) with ranking, plotting, and NIfTI access. |
| [core/runtime.py](src/neurovlm/core/runtime.py) | `load_pipeline`, `NeuroVLMRuntime`, and `RuntimeMetadata`: explicit family/task/domain/variant selection, released or local-run loading, tensor-level inference. |
| [core/__init__.py](src/neurovlm/core/__init__.py) | Re-exports the high-level client interface. |

Use `NeuroVLM` for the chained user-facing API. Use `load_pipeline` when selecting
a task or model family explicitly. Use `neurovlm.models.load_model` when you need
the underlying PyTorch module. CNN branches default to `mixed_baseline`;
`variant="finetuned"` selects specialized weights explicitly.

### Model definitions and selection: `models/` and `cnn/`

| File | Responsibility / useful symbols |
|---|---|
| [models/base.py](src/neurovlm/models/base.py) | `NeuroAutoEncoder`, `ProjHead`, `Specter`, `ConceptClf`, `NormalizeLayer`, and public `load_model` dispatch. |
| [models/registry.py](src/neurovlm/models/registry.py) | Canonical model specifications, aliases, family/task/domain/variant enums, and `resolve_model_spec`. Read this first for model naming or selection changes. |
| [models/qformer.py](src/neurovlm/models/qformer.py) | `QFormer`, `CanonicalProjection`, and `NeuroQFormer` for brain-to-text generation. |
| [models/adapter.py](src/neurovlm/models/adapter.py) | `InterleavedDecoderAdapter`, residual blocks, and logit calibration for text-to-brain decoding. |
| [models/losses.py](src/neurovlm/models/losses.py) | InfoNCE, focal, and truncated losses. |
| [models/serialization.py](src/neurovlm/models/serialization.py) | Safetensors save/load helpers. Its `load_model` loads weights into a supplied module; the public model-selection loader lives in `models/base.py`. |
| [cnn/architectures.py](src/neurovlm/cnn/architectures.py) | 3D CNN / ResNet encoders, decoder, `ALE3DCNNAutoEncoder`, architecture validation, and summaries. |
| [cnn/models.py](src/neurovlm/cnn/models.py) | `CNNContrastiveModel`, `CNNTextToBrainModel`, checkpoint-payload builders, and conversion between atlas-free volumes and MLP masked flat maps. |

### Data and downloaded resources: `data/` and `resources/`

| File | Responsibility / useful symbols |
|---|---|
| [data/loaders.py](src/neurovlm/data/loaders.py) | Public `fetch_data`, `load_dataset`, `load_latent`, `load_masker`, `get_data_dir`, and `data_dir`; dataset/embedding key dispatch. |
| [resources/loaders.py](src/neurovlm/resources/loaders.py) | Internal `_load_*` functions, Hugging Face repository IDs and filenames, cached datasets/embeddings, masks, and released model resources. This is the place to trace the actual downloaded artifact. |
| [data/coordinates.py](src/neurovlm/data/coordinates.py) | Coordinate-to-vector conversion and DiFuMo projection utilities. |
| [data/atlas_free_dataset.py](src/neurovlm/data/atlas_free_dataset.py) | `AtlasFreeCNNDataset`, `AtlasFreeCNNDataProvider`, domain normalization, and train/validation/test splits. |
| [data/atlas_free_text.py](src/neurovlm/data/atlas_free_text.py) | Cached text-embedding lookup, primary-positive text selection, and contrastive collation. |

The loaders cover PubMed publications, coordinates and summaries; NeuroVault;
NeuroWiki; Cognitive Atlas; network maps and labels; n-grams; MeSH/KG terms;
LLM-extracted neuroscience terms; and atlas-free CNN resources. Supported keys
are defined in `data/loaders.py`; high-level search aliases are in
`core/client.py`.

### Retrieval and optional summaries: `retrieval/`

| File | Responsibility |
|---|---|
| [retrieval/user.py](src/neurovlm/retrieval/user.py) | Search a custom text corpus from neuroimages or text, resample maps to the mask, and generate an LLM response from retrieved context. |
| [retrieval/summarization.py](src/neurovlm/retrieval/summarization.py) | LLM loading, prompts, response generation, and Hugging Face / Ollama backends. |

### Training: `training/`

Typed configs and `train_*` functions are exported through
[training/__init__.py](src/neurovlm/training/__init__.py). Read the relevant module
for defaults, initialization, dataset-provider requirements, and checkpoint
reload helpers.

| File | Workflows |
|---|---|
| [training/autoencoder.py](src/neurovlm/training/autoencoder.py) | CNN: `AutoencoderTrainConfig`, `train_autoencoder`, reconstruction evaluation and checkpoint reload. |
| [training/contrastive.py](src/neurovlm/training/contrastive.py) | CNN: `ContrastiveTrainConfig`, `train_contrastive`, contrastive model construction and reload. |
| [training/text_to_brain.py](src/neurovlm/training/text_to_brain.py) | CNN: `TextToBrainTrainConfig`, `train_text_to_brain`, generation loss and reload. |
| [training/mlp.py](src/neurovlm/training/mlp.py) | MLP autoencoder, contrastive, text-to-brain, and brain-to-text retrieval configs/runners; `train_mlp_*` functions take an explicit provider. |
| [training/brain_to_text.py](src/neurovlm/training/brain_to_text.py) | QFormer generation: `BrainToTextGenerationTrainConfig`, `BrainToTextCollator`, `train_brain_to_text_generation`, and reload. |

CNN workflows use the published atlas-free provider by default. Released
initialization, deliberate local-run chaining, and resume examples are in the
[CNN technical guide](docs/experimental/cnn/technical_guide.md).

### Evaluation versus metric implementations

`evaluation/` runs models over data and returns evaluation results. `metrics/`
implements scores and analysis helpers; it also contains retained paper-specific
evaluation routines.

| File | Responsibility |
|---|---|
| [evaluation/contrastive.py](src/neurovlm/evaluation/contrastive.py) | Contrastive retrieval evaluator. |
| [evaluation/text_to_brain.py](src/neurovlm/evaluation/text_to_brain.py) | CNN text-to-brain evaluator. |
| [evaluation/brain_to_text.py](src/neurovlm/evaluation/brain_to_text.py) | Generation batch parsing, language-model forward pass, and generation evaluation. |
| [evaluation/mlp.py](src/neurovlm/evaluation/mlp.py) | MLP reconstruction, contrastive, text-to-brain, and brain-to-text retrieval evaluators. |
| [evaluation/spatial.py](src/neurovlm/evaluation/spatial.py) | Reconstruction metrics and voxel AUROC. |
| [evaluation/comparison.py](src/neurovlm/evaluation/comparison.py) | MLP/CNN comparison selection, manifests, and reconstruction/retrieval/generation comparisons. |
| [evaluation/notebook_utils.py](src/neurovlm/evaluation/notebook_utils.py) | Shared notebook setup: evaluation output paths, network labels, PubMed/NeuroVault evaluation samples, and latent projections. |
| [metrics/retrieval.py](src/neurovlm/metrics/retrieval.py) | Ranking, recall@k, bidirectional retrieval, and recall curves. |
| [metrics/semantic.py](src/neurovlm/metrics/semantic.py) | PMID retrieval, network labeling/term ranking, MeSH ranking, and semantic-neighbor evaluation. |
| [metrics/brain_to_text.py](src/neurovlm/metrics/brain_to_text.py) | Generated-text scores, network-label accuracy, gold-term ranking, and paper retrieval analyses. |
| [metrics/text_to_brain.py](src/neurovlm/metrics/text_to_brain.py) | Correlation, PSNR, Dice, reconstruction quality, surface/spin tests, and generated-image retrieval analyses. |
| [metrics/common.py](src/neurovlm/metrics/common.py) | Shared latent batching and projections. |
| [metrics/__init__.py](src/neurovlm/metrics/__init__.py) | Compatibility exports for existing metric imports; implementations live in the task modules above. |

### Shared run infrastructure: `pipelines/`

This package manages training runs and their artifacts. The inference pipeline
loader is in `core/runtime.py`.

| File | Responsibility |
|---|---|
| [pipelines/config.py](src/neurovlm/pipelines/config.py) | `RunConfig`, run identity, model selection, requested/effective configuration, primary metric direction. |
| [pipelines/artifacts.py](src/neurovlm/pipelines/artifacts.py) | `RunArtifacts` and `RunContext`: directories, manifests, provenance output, and lifecycle status. |
| [pipelines/checkpoints.py](src/neurovlm/pipelines/checkpoints.py) | `CheckpointManager`, best/last checkpoints, architecture compatibility, and resume state. |
| [pipelines/metrics.py](src/neurovlm/pipelines/metrics.py) | `MetricRecorder`, metric history, curves, and summaries. |
| [pipelines/provenance.py](src/neurovlm/pipelines/provenance.py) | File/reference hashes, environment details, and Git provenance. |
| [pipelines/serialization.py](src/neurovlm/pipelines/serialization.py) | JSON-safe conversion and atomic JSON/CSV writes. |
| [utils/progress.py](src/neurovlm/utils/progress.py) | Notebook/terminal progress-bar selection. |

Use this shared infrastructure when extending training. Its overview is in
[pipelines/README.md](src/neurovlm/pipelines/README.md).

## Notebook map

The committed `.ipynb` files are the source of truth. Python exports are generated
on demand. The canonical reproduction flow is
`docs/02_data/` → `docs/03_models/` → `docs/figures/`; released artifacts often let
you start directly with an analysis notebook.

### Tutorials: `docs/01_tutorials/`

| Notebook | What to look for |
|---|---|
| [00_quickstart.ipynb](docs/01_tutorials/00_quickstart.ipynb) | All four high-level generation/retrieval paths. |
| [01_introduction.ipynb](docs/01_tutorials/01_introduction.ipynb) | Framework overview and data access. |
| [02_contrastive.ipynb](docs/01_tutorials/02_contrastive.ipynb) | Brain-to-text and text-to-brain contrastive retrieval. Trace implementation to `core/client.py`, model heads to `models/base.py`, and losses to `models/losses.py`. |
| [03_generative_text-to-brain.ipynb](docs/01_tutorials/03_generative_text-to-brain.ipynb) | Generate activation maps from text. |
| [04_generative_brain-to-text.ipynb](docs/01_tutorials/04_generative_brain-to-text.ipynb) | QFormer-based descriptions of brain maps. |
| [05_custom_corpus.ipynb](docs/01_tutorials/05_custom_corpus.ipynb) | Search your own corpus through `retrieval/user.py`. |
| [06_atlas_free_cnn.ipynb](docs/01_tutorials/06_atlas_free_cnn.ipynb) | Atlas-free CNN inference and structured model selection. |

[cogatlas_concepts.md](docs/01_tutorials/cogatlas_concepts.md) is the supporting
Cognitive Atlas concept reference; [index.md](docs/01_tutorials/index.md) controls
the tutorial documentation listing.

### Data preparation: `docs/02_data/`

| Notebook | Purpose |
|---|---|
| [01_coordinate.ipynb](docs/02_data/01_coordinate.ipynb) | Coordinate smoothing/ALE, masked vectors, and DiFuMo projection. |
| [02_pubmed.ipynb](docs/02_data/02_pubmed.ipynb) | PubMed data and text preparation. |
| [03_neurovault.ipynb](docs/02_data/03_neurovault.ipynb) | NeuroVault data preparation. |
| [04_neurowiki.ipynb](docs/02_data/04_neurowiki.ipynb) | NeuroWiki text embeddings. |
| [05_cogatlas.ipynb](docs/02_data/05_cogatlas.ipynb) | Cognitive Atlas preparation. |
| [06_n_grams.ipynb](docs/02_data/06_n_grams.ipynb) | N-gram corpus extraction and embedding. |
| [07_transform_text.ipynb](docs/02_data/07_transform_text.ipynb) | Generate generalized summaries from publication text. |

### Model development: `docs/03_models/`

| Notebook | Purpose |
|---|---|
| [08_autoencoder.ipynb](docs/03_models/08_autoencoder.ipynb) | MLP brain-map autoencoder training. |
| [09_projection_head.ipynb](docs/03_models/09_projection_head.ipynb) | Text/brain projection-head alignment and training. |
| [10_n_grams.ipynb](docs/03_models/10_n_grams.ipynb) | Concept classifier from n-gram embeddings. |
| [11_anatomical_atlases.ipynb](docs/03_models/11_anatomical_atlases.ipynb) | Anatomical-atlas synthetic training data. |
| [12_neuroadapter.ipynb](docs/03_models/12_neuroadapter.ipynb) | NeuroAdapter training on anatomical synthetic maps. |
| [13_qformer_pubmed.ipynb](docs/03_models/13_qformer_pubmed.ipynb) | Grounded QFormer pretraining on PubMed. |
| [14_qformer_anat.ipynb](docs/03_models/14_qformer_anat.ipynb) | Anatomical/canonical-network QFormer training and analysis. |

### Paper analyses: `docs/figures/`

Read [figures/README.md](docs/figures/README.md) for figure numbers, dependencies,
and execution order. The retained notebooks are:

| Notebook | Analysis |
|---|---|
| [11_autoencoder.ipynb](docs/figures/11_autoencoder.ipynb) | Autoencoder reconstruction metrics. |
| [12_neurovault_decoding.ipynb](docs/figures/12_neurovault_decoding.ipynb) | NeuroVault decoding examples. |
| [13_network_labeling.ipynb](docs/figures/13_network_labeling.ipynb) | Network labeling and confusion analyses. |
| [16_qualatative_auto.ipynb](docs/figures/16_qualatative_auto.ipynb) | Qualitative reconstructions. |
| [19_ica_networks.ipynb](docs/figures/19_ica_networks.ipynb) | HCP / UK Biobank ICA network labeling. |
| [20_pubmed_cv.ipynb](docs/figures/20_pubmed_cv.ipynb) | PubMed cross-validation training and retrieval evaluation. |
| [23_versus_others.ipynb](docs/figures/23_versus_others.ipynb) | External-baseline comparisons. |
| [24_brain_to_text_pubmed.ipynb](docs/figures/24_brain_to_text_pubmed.ipynb) | PubMed brain-to-text generation and metrics. |

The existing `qualatative` spelling is part of the filenames. The figure guide
also references `22_text_to_brain_metrics.ipynb`, which is missing from this
checkout; use its documented recovery note when working on that analysis.

### Retained research: `docs/experimental/`

[experimental/README.md](docs/experimental/README.md) explains the scope. These
notebooks may depend on older APIs, unreleased inputs, or external models.

| Directory | Contents |
|---|---|
| [cnn/training/](docs/experimental/cnn/training/) | `architecture_background.ipynb` (architecture history), `autoencoder.ipynb`, `contrastive_and_text_to_brain.ipynb`, and `contrastive_pubmed.ipynb`. |
| [cnn/evaluation/](docs/experimental/cnn/evaluation/) | `autoencoder_comparison.ipynb`, `contrastive_comparison.ipynb`, and `text_to_brain_comparison.ipynb`. |
| [data_preparation/](docs/experimental/data_preparation/) | `create_networks_test_set_csv.ipynb`, `extract_neuroscience_terms_from_text.ipynb`, its `extract_neuroscience_terms_colab_hf.ipynb` variant, `mesh.ipynb`, and `prepare_network_gold_terms_for_llm_corpus.ipynb`. |
| [data_preparation/scripts/](docs/experimental/data_preparation/scripts/) | `prepare_neuro_summaries.py` builds summary metadata/embeddings; `prepare_llm_neuro_terms.py` filters and embeds extracted terms; `deduplicate_llm_terms.py` normalizes and merges term rows. |
| [evaluation/](docs/experimental/evaluation/) | `14_llm_concepts.ipynb`, `15_qualatative_networks.ipynb`, `17_qualatative_generative.ipynb`, `18_quant_gen_a.ipynb`, and `18_quant_gen_b.ipynb`. |

The reusable CNN implementation lives in `src/neurovlm/cnn/`, `training/`, and
`evaluation/`, even though its research notebooks are under `experimental/`.

## Where data and generated outputs go

| Location | Meaning |
|---|---|
| Hugging Face Hub cache | Downloaded released datasets and weights. `fetch_data()` uses `HUGGINGFACE_HUB_CACHE` by default and accepts `cache_dir`. Internal resource loaders use Hub downloads. |
| `~/.cache/neurovlm/` | `get_data_dir()` / `data_dir`: local intermediate artifacts used by research notebooks. This is a separate directory from the Hub cache; a downloaded artifact does not necessarily appear directly under `data_dir`. |
| `runs/<run-id>/` | Standard training output with the default `output_root="runs"`; callers can select another root. |
| `docs/figures/outputs/` or notebook-local `outputs/` | Evaluation caches, metric tables, and figures; resolution is implemented in `evaluation/notebook_utils.py`. |
| `docs/generated/notebooks/` | On-demand notebook Python exports; ignored by Git. |
| `docs/_build/html/` | Built documentation; ignored by Git. |

Standard training runs have this layout, defined in `pipelines/artifacts.py`:

```text
runs/<run-id>/
├── manifest.json
├── status.json
├── config/              requested.json, effective.json
├── provenance/          environment, Git, data, resources, initialization
├── checkpoints/         best.pt, last.pt, checkpoint_manifest.json
├── metrics/             history.csv, summary.csv, curves.csv
├── plots/
├── generated_maps/
└── logs/
```

The directories are standardized; individual tasks determine which optional
outputs they populate. Released resource IDs/filenames live in
`resources/loaders.py`, and bulk-fetch repositories live in `data/loaders.py`.
Atlas-free CNN resources use `neurovlm/atlas_free_cnn_dataset` and
`neurovlm/3d_cnn` through their dedicated loaders/provider.

Check [.gitignore](.gitignore) when tracking down local files. Model/data binaries,
notebook exports, documentation builds, and many research outputs are ignored;
they are not part of a fresh clone.

## Tests, scripts, and automation

| Location | What it checks or does |
|---|---|
| [tests/unit/](tests/unit/) | API, data loaders, metrics, model definitions/loading/registry/serialization, run infrastructure, and resource caching. Subdirectories match package responsibilities. |
| [tests/integration/](tests/integration/) | Runtime inference, model comparisons, and small training workflows, including checkpoint/resume behavior. |
| [tests/conftest.py](tests/conftest.py) | Shared random seeds, fixtures, and unit/integration markers. |
| [tests/notebooks/test_notebooks.py](tests/notebooks/test_notebooks.py) | Discovers notebooks, exports them, and executes subprocesses. `KNOWN_UNRUNNABLE` records skips and their reasons. |
| [tests/notebooks/smoke_bootstrap.py](tests/notebooks/smoke_bootstrap.py) | `NEUROVLM_SMOKE=1` patches to limit training epochs/batches. |
| [tests/notebooks/run_smoke.py](tests/notebooks/run_smoke.py) | Executes an exported notebook inside the smoke environment. |
| [scripts/export_notebook.py](scripts/export_notebook.py) | Exports a notebook with Jupytext; `-o` selects the output path. |
| [scripts/smoke_training.sh](scripts/smoke_training.sh) | Runs integration training tests. |
| [scripts/smoke_notebooks.sh](scripts/smoke_notebooks.sh) | Runs notebook smoke tests; accepts `public`, `experimental`, `figures`, or pytest arguments. |
| [.github/workflows/tests.yml](.github/workflows/tests.yml) | Offline tests and coverage on Ubuntu/macOS with Python 3.10–3.13. |
| [.github/workflows/docs.yml](.github/workflows/docs.yml) | Sphinx build and documentation deployment. |
| [.github/workflows/publish.yml](.github/workflows/publish.yml) | Package build/content checks and PyPI/TestPyPI publishing. |

Run commands from the repository root with your project environment active:

```bash
# Development installation with test and metric dependencies.
pip install -e ".[test,metrics]"

# Deterministic offline suite, using the same exclusions as CI.
python -m pytest -m "not network and not requires_data and not requires_pretrained and not requires_specter and not slow"

# Focused checks.
python -m pytest tests/unit/api tests/integration/api
scripts/smoke_training.sh

# Execute one notebook at smoke scale, or export it without running it.
scripts/smoke_notebooks.sh -k 02_contrastive
python scripts/export_notebook.py docs/01_tutorials/02_contrastive.ipynb
```

Notebook smoke tests need the corresponding dependencies, downloads/cache, and
sometimes external services. They run from the original notebook's directory
and check execution at reduced scale, not full numerical reproducibility. An
unfiltered `pytest` run can include these tests; CI excludes them through the
markers above. See [tests/README.md](tests/README.md) for details.

Documentation entry points are [docs/index.md](docs/index.md),
[docs/api.rst](docs/api.rst), and [docs/conf.py](docs/conf.py). Build commands and
export conventions are in [docs/README.md](docs/README.md); documentation
dependencies are in [docs/requirements.txt](docs/requirements.txt).

## Prompt to give a coding agent

```text
Read CODEBASE_MAP.md at the repository root before exploring NeuroVLM.
Use its task lookup to find the relevant source modules, then read their
implementations and corresponding tests before making changes.

My task is: <describe the task here>.

The package is under src/neurovlm/. The high-level API is in core/client.py;
structured inference is in core/runtime.py. Model selection is in
models/registry.py. Public data loading is in data/loaders.py, and actual
downloaded artifacts are resolved in resources/loaders.py. Training,
evaluation, metrics, and run infrastructure have separate packages.

Use the existing public APIs and shared run infrastructure where applicable.
Notebooks in docs/ are canonical; generate Python exports when needed.
Consult tests/notebooks/test_notebooks.py for documented notebook skips.
Verify current source definitions rather than assuming this map is exhaustive
or that local data/model artifacts exist in a fresh clone.
Run checks appropriate to the change and report the files changed, what was
verified, and any remaining limitations.
```

When adding, moving, or renaming a subsystem or canonical notebook, update this
map and the relevant directory README so collaborators can keep finding it.
