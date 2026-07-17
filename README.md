# NeuroVLM

NeuroVLM maps between neuroimaging activation maps and neuroscience text.

![model](https://github.com/neurovlm/neurovlm_data/blob/13dd7769f9603c036a9338b7da4adc2f3a03ec94/docs/model.png)

## Install

Minimal, inference-only installation:

```bash
pip install neurovlm
```

With optional dependencies needed to train and reproduce analyses:

```bash
pip install "neurovlm[full]"
```

## Quickstart

Fetch NeuroVLM's datasets:

```python
from neurovlm.data import fetch_data
fetch_data()
```

Use the `NeuroVLM` object for text-to-brain and brain-to-text:

```python
from neurovlm import NeuroVLM
from neurovlm.data import load_latent

# Text-to-brain: generation
nvlm = NeuroVLM()
result = nvlm.text(["vision", "default mode network"]).to_brain(head="mse")
result.to_nifti() # returns list of nib.Nifti1Image
result.plot(0, threshold=0.25) # plot image for vision
result.plot(1, threshold=0.15) # plot image for DMN

# Text-to-brain: ranking and retrieval
nvlm = NeuroVLM()
result = nvlm.text("motor").to_brain(head='infonce')
top = result.top_k(2) # each row pairs to a neuorimage that is similar to the text query
top.plot_row(1, threshold=0.1) # WashU atlas
top.plot_row(2, threshold=2.5) # NeuroVault
top.plot_row(4, threshold=0.1) # PubMed

# Brain-to-text: ranking and retrieval
nvlm = NeuroVLM()
result = nvlm.brain(load_latent("networks_neuro")["Du"]["AUD"]).to_text()
result.top_k(5).query("cosine_similarity > 0.4") # return up to 5 examples per dataset
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

## License

Apache-2.0 (`LICENSE`).
