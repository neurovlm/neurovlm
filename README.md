# NeuroVLM

NeuroVLM maps between neuroimaging activation maps and neuroscience text.

![model](https://github.com/neurovlm/neurovlm_data/blob/13dd7769f9603c036a9338b7da4adc2f3a03ec94/docs/model.png)

## Install

```bash
pip install -e .
```

## Quickstart

Fetch NeuroVLM's datasets:

```python
from neurovlm.data import fetch_data
fetch_data()
```

Use the `NeuroVLM` object for text-to-brain, brain-to-text, text-to-text, and brain-to-brain:

```python
from neurovlm import NeuroVLM
from neurovlm.data import load_latent, load_dataset

# Load networks examples images
networks = load_latent("networks_neuro")

# Text-to-brain: generative
nvlm = NeuroVLM()
res = nvlm.to_brain("default mode network", model="mse")
imgs = res.to_nifti() # returns nib.Nifti1Image

# Brain-to-text: contrastive
res = nvlm.to_text(networks["Du"]["DN-A"])
res.top_k(1) # returns pd.DataFrame

# Text-to-brain: contrastive
res = nvlm.to_brain("default mode network", model="infonce")
imgs = res.top_k(3).to_nifti()

# Text-to-text
res = nvlm.to_text("default mode network", project=False)
res.top_k(1) # returns pd.DataFrame

# Brain-to-brain
res = nvlm.to_brain(networks["Du"]["DN-A"], model="infonce")
res.top_k(5) # returns pd.DataFrame
```

## Data and API

- Data/model fetch and loaders: `neurovlm.data`
- Full API reference: `docs/api.rst`

## License

Apache-2.0 (`LICENSE`).
