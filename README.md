# NeuroVLM

NeuroVLM maps between neuroimaging activation maps and neuroscience text.

![model](https://github.com/neurovlm/neurovlm_data/blob/13dd7769f9603c036a9338b7da4adc2f3a03ec94/docs/model.png)


## System Requirements

- python >=3.10, <3.14
- ubuntu-latest, macos-latest
- GPU (NVIDIA or Apple MPS) or CPU


## Installation

Minimal, inference-only installation:

```bash
pip install neurovlm
```

With optional dependencies needed to train and reproduce analyses:

```bash
pip install "neurovlm[full]"
```

Installation take a couple minutes. After installation, calling `neurovlm.data.fetch_data()` will fetch datasets and models from huggingface, which will be slower.


## Demo

See [here](https://github.com/neurovlm/neurovlm/blob/main/docs/01_tutorials/00_quickstart.ipynb) for the introductory notebook that walks through using all NeuroVLM models. In short:

Fetch NeuroVLM's datasets and models:

```python
from neurovlm.data import fetch_data
fetch_data()
```

Use the four text/brain inference paths:

```python
from neurovlm import NeuroVLM
from neurovlm.data import load_latent

nvlm = NeuroVLM(device="cuda") # use device="cpu" if GPU not available

# Text-to-brain generation (MSE)
brain_map = nvlm.text("auditory processing").to_brain(head="mse")
brain_map.plot(0, threshold=0.1)

# Brain-to-text generation (QFormer)
auditory = load_latent("networks_neuro")["Du"]["AUD"]
description = nvlm.brain(auditory).to_text(head="qformer")
print(description)

# Text-to-brain retrieval (InfoNCE)
brain_matches = nvlm.text("auditory processing").to_brain(head="infonce")
df_text_to_brain = brain_matches.top_k(3)

# Brain-to-text retrieval (InfoNCE)
text_matches = nvlm.brain(auditory).to_text(head="infonce")
df_brain_to_text = text_matches.top_k(3)
```

Select either model family with the structured inference API. CNN autoencoders
always default to the mixed-source baseline; domain-specific contrastive and
text-to-brain heads use the mixed baseline unless `variant="finetuned"` is
requested explicitly:

```python
from neurovlm.runtime import load_pipeline

autoencoder = load_pipeline(family="cnn", task="autoencoder")
contrastive = load_pipeline(
    family="cnn", task="contrastive", domain="pubmed"
)
text_to_brain = load_pipeline(
    family="cnn", task="text_to_brain", domain="nilearn"
)

# The same task-level surface loads a standardized local training run.
local = load_pipeline(
    family="cnn", task="contrastive", domain="pubmed",
    from_run="runs/<run-id>",
)
```

Training uses typed configs and automatically writes reproducible config,
provenance, best/last checkpoints, metric CSVs, plots, and logs:

```python
from neurovlm.training import ContrastiveTrainConfig, train_contrastive

result = train_contrastive(ContrastiveTrainConfig(domain="neurovault"))
print(result.run_dir / "metrics/history.csv")
```

See the atlas-free CNN tutorial and technical guide for all PubMed, Nilearn,
and NeuroVault switches; MLP/CNN reconstruction, retrieval, and generation;
resume; and explicit local-run chaining.

## Documentation

See the [docs](https://neurovlm.github.io/neurovlm/) for the [API](https://neurovlm.github.io/neurovlm/api.html) and [tutorials](https://neurovlm.github.io/neurovlm/tutorials/index.html).


## Reproducibility

Analyses are organized as Jupyter notebooks:

1. `docs/01_tutorials`: User-facing examples
2. `docs/02_data`: Data loading and preprocessing
3. `docs/03_models`: Model training and development
4. `docs/04_evaluation`: Evaluation and publication figures
5. `docs/05_cnn`: Atlas-free CNN training and evaluation
6. `docs/06_data_preparation`: Dataset artifact generation

## License

Apache-2.0 (`LICENSE`).
