# Integrated CNN and multimodal pipeline

NeuroVLM now exposes the atlas-free 3D CNN through the same package-level model selection, training, inference, evaluation, and run-artifact conventions as the retained MLP models. The historical experiment notebooks are organized in this section; they no longer import experiment-local Python modules.

## Selection rules

The mixed-source CNN autoencoder is always the CNN default. For contrastive and text-to-brain tasks, `domain` selects the PubMed, Nilearn, or NeuroVault head while the underlying default remains the mixed baseline. Fine-tuned autoencoders and heads are loaded only when `variant="finetuned"` is stated explicitly.

```python
from neurovlm.runtime import load_pipeline

ae = load_pipeline(family="cnn", task="autoencoder")

pubmed_retrieval = load_pipeline(
    family="cnn", task="contrastive", domain="pubmed"
)
nilearn_retrieval = load_pipeline(
    family="cnn", task="contrastive", domain="nilearn"
)
neurovault_retrieval = load_pipeline(
    family="cnn", task="contrastive", domain="neurovault"
)

pubmed_generator = load_pipeline(
    family="cnn", task="text_to_brain", domain="pubmed"
)

# Explicit ablation, never an implicit default:
finetuned = load_pipeline(
    family="cnn", task="contrastive", domain="pubmed", variant="finetuned"
)
```

`load_model(...)` accepts the same structured selectors when the raw PyTorch module is wanted. `load_pipeline(...)` adds consistent `reconstruct`, `encode_brain`, `encode_text`, `similarity`, and `generate` task methods plus resolved metadata. Existing string aliases and the legacy high-level `NeuroVLM` API remain compatible.

## Supported tasks

| task | MLP | CNN | purpose |
|---|:---:|:---:|---|
| `autoencoder` | yes | yes | brain-map reconstruction and latent encoding |
| `contrastive` | yes | yes | symmetric text/brain representation learning |
| `text_to_brain` | yes | yes | generate a brain map from a text embedding |
| `brain_to_text_retrieval` | yes | — | rank text candidates for a brain map |
| `brain_to_text_generation` | yes | — | generate text through the adapter/Q-Former language-model path |

Brain-to-text retrieval and generation are separate tasks because they optimize, checkpoint, evaluate, and infer differently. CNN brain-to-text is not presented as supported when no released or integrated training path exists.

## Published atlas-free data

```python
from neurovlm import AtlasFreeCNNDataProvider

provider = AtlasFreeCNNDataProvider(domain="pubmed")
train, validation, test = provider.train, provider.val, provider.test
```

With no local overrides, the provider retrieves `splits/train.jsonl`, `splits/val.jsonl`, `splits/test.jsonl`, and the shared `atlas_free_cnn_volumes.pt` resource from Hugging Face. It indexes the shared tensor by `tensor_index`; legacy `tensor_path` and NIfTI path fields in JSONL rows are deliberately ignored. `split_dir=` and `volume_path=` are explicit reproducibility overrides, not defaults.

## Training

The public typed configs and runners are:

| family/task | config | runner |
|---|---|---|
| CNN autoencoder | `AutoencoderTrainConfig` | `train_autoencoder` |
| CNN contrastive | `ContrastiveTrainConfig` | `train_contrastive` |
| CNN text-to-brain | `TextToBrainTrainConfig` | `train_text_to_brain` |
| MLP autoencoder | `MLPAutoencoderTrainConfig` | `train_mlp_autoencoder` |
| MLP contrastive | `MLPContrastiveTrainConfig` | `train_mlp_contrastive` |
| MLP text-to-brain | `MLPTextToBrainTrainConfig` | `train_mlp_text_to_brain` |
| MLP brain-to-text retrieval | `MLPBrainToTextRetrievalTrainConfig` | `train_mlp_brain_to_text_retrieval` |
| MLP brain-to-text generation | `BrainToTextGenerationTrainConfig` | `train_brain_to_text_generation` |

CNN domain branches make the switch explicit:

```python
from neurovlm.training import ContrastiveTrainConfig, train_contrastive

result = train_contrastive(ContrastiveTrainConfig(
    domain="nilearn",             # pubmed | nilearn | neurovault
    variant="mixed_baseline",    # explicit alternative: finetuned
    output_root="runs",
))
```

Released Hugging Face initialization is the default. Use `from_run=` (or the text-to-brain config's `autoencoder_from_run=`) only to chain a local autoencoder run. Use `resume=` with the original `run_id` to restore the model, optimizer, next epoch, best metric, early-stopping state, and accumulated metric history.

MLP runners accept an explicit provider so the same training engine can be used with retained datasets or application-specific PyTorch datasets without duplicating loops.

## Reproducible run artifacts

Every runner creates the same layout automatically:

```text
runs/<run-id>/
├── manifest.json
├── status.json
├── config/
│   ├── requested.json
│   └── effective.json
├── provenance/
│   ├── environment.json
│   ├── git.json
│   ├── data.json
│   ├── resources.json
│   └── initialization.json
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── checkpoint_manifest.json
├── metrics/
│   ├── history.csv
│   ├── summary.csv
│   └── curves.csv
├── plots/
├── generated_maps/
└── logs/
```

Metrics are task-aware: reconstruction loss and spatial overlap for autoencoders/text-to-brain, bidirectional recall/MRR/normalized recall-curve AUC for contrastive retrieval, direction-specific retrieval selection for brain-to-text retrieval, and language-model loss plus optional generation evaluators for brain-to-text generation. Checkpoints record architecture metadata and hashes and reject incompatible resumes.

Inference from a local run is explicit and portable:

```python
runtime = load_pipeline(
    family="cnn",
    task="contrastive",
    domain="pubmed",
    from_run="runs/<run-id>",
)
```

## Comparison evidence and its scope

The preserved executed contrastive ablation evaluated 32 test examples per domain. Its mean normalized recall-curve AUC values were:

| domain | mixed baseline | fine-tuned |
|---|---:|---:|
| PubMed | 0.807617 | 0.804688 |
| Nilearn | 0.907227 | 0.907227 |
| NeuroVault | 0.850586 | 0.845703 |

These limited results motivate the mixed baseline as the safe default: it is slightly higher on PubMed and NeuroVault and tied on Nilearn. They do not establish universal superiority. Fine-tuning therefore remains available as an explicit opt-in rather than the default.

Use `default_comparison_matrix(...)` and the three `evaluate_*_comparison(...)` functions for current MLP/CNN comparisons. Each runtime is resolved independently, missing resources are recorded in the manifest instead of aborting the matrix, and metrics declare whether they are in native CNN volume space or the MLP masker flat-map space.

## Notebook map

- `training/autoencoder.ipynb`: mixed pretraining, explicit fine-tuning, resume, and artifacts.
- `training/contrastive_pubmed.ipynb`: concise retained contrastive recipe.
- `training/contrastive_and_text_to_brain.ipynb`: domain switches for both downstream tasks.
- `training/architecture_background.ipynb`: non-prescriptive architecture history.
- `evaluation/*_comparison.ipynb`: package-level reconstruction, retrieval, and generation comparisons.
- `../tutorials/06_atlas_free_cnn.ipynb`: short inference quickstart.
