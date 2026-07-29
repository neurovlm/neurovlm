# NeuroVLM

NeuroVLM maps between neuroimaging activation maps and neuroscience text.

![model](https://github.com/neurovlm/neurovlm_data/blob/13dd7769f9603c036a9338b7da4adc2f3a03ec94/docs/model.png)


## System Requirements

- python >=3.10, <3.14
- ubuntu-latest, macos-latest
- NVIDIA GPU (recommneded) or CPU


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

See [here](https://github.com/neurovlm/neurovlm/blob/main/docs/tutorials/00_quickstart.ipynb) for the introductory notebook that walks through using all NeuroVLM models. In short:

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

## Documentation

See the [docs](https://neurovlm.github.io/neurovlm/) for the [API](https://neurovlm.github.io/neurovlm/api.html) and [tutorials](https://neurovlm.github.io/neurovlm/tutorials/index.html).


## Reproducibility

All analyses are in Juptyer notebooks. Their are three directories:

1. `docs/01_data`: Fetch raw data and preprocess
2. `docs/02_models`: Trains all models
3. `docs/03_evaluation`: Evaluates models and reproduces publication figures.

## License

Apache-2.0 (`LICENSE`).
