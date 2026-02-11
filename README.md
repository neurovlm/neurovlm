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

nvlm = NeuroVLM()

# Text-to-brain generation
result = nvlm.text(["vision", "default mode network"]).to_brain(head="mse")
result.to_nifti() # returns list of nib.Nifti1Image
result.plot(0, threshold=0.25) # plot image for vision
result.plot(1, threshold=0.15) # plot image for DMN

# Text-to-brain ranking and retrieval
nvlm = NeuroVLM()
result = nvlm.text("motor").to_brain(head='infonce')
top = result.top_k(2) # each row pairs to a neuorimage that is most similar to the text query
top.plot_row(1, threshold=0.1) # WashU atlas
top.plot_row(2, threshold=2.5) # NeuroVault
top.plot_row(4, threshold=0.1) # PubMed

# Brain-to-text ranking and retrieval
nvlm = NeuroVLM()
result = nvlm.brain(load_latent("networks_neuro")["Du"]["AUD"]).to_text()
result.top_k(5).query("cosine_similarity > 0.4") # return up to 5 examples per dataset
```

## Data and API

- Data/model fetch and loaders: `neurovlm.data`
- Full API reference: `docs/api.rst`

## License

Apache-2.0 (`LICENSE`).
