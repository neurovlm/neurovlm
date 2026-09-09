# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .env
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch

import nibabel as nib
from nilearn import maskers
from nilearn.plotting import plot_stat_map
from neurovlm.data import data_dir, load_dataset, load_masker
from neurovlm.training import MLPAutoencoderTrainConfig, train_mlp_autoencoder


def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = select_device()

# %% [markdown]
# # Autoencoder
#
# The first step is to reduce the dimensionality of the 28k MNI space neuro vector. Several work uses difumo. Instead, we use an autoencoder to map from $d=28,000$ to a $k=384$ latent space. 

# %% [markdown]
# ## Load Neurovectors
#
# Results from the coordinate smoothing notebook.

# %%
# Load vectors from 01_coordinate.ipynb
neuro_vectors, pmids = torch.load(
    data_dir / "neuro_vectors.pt", weights_only=False,
).values()

# Filter for studies with less than 100 coordinates
df_coords = load_dataset("pubmed_coordinates")
counts = df_coords[df_coords['pmid'].isin(pmids)]["pmid"].value_counts()
mask = pd.Series(pmids).isin(counts.index[counts <= 100].values)
pmids = np.array(pmids[mask])
neuro_vectors = neuro_vectors[mask]

# %% [markdown]
# ## Training
#
# Train an autoencoder on the neuro-vectors.
#
# 1. Encoder: Neuro-vector to low-dimensional (384) latent vector / embedding space
# 2. Decoder: Latent vector to produces MNI space predictions.
#
# Training is complete in two stages:
#
# 1. Initial training anywhere neurovectors > 0
# 2. Additional training to shrink the size of activations closer to targets

# %%
# Train/test/validation split
inds = torch.arange(len(pmids))
train_inds, test_inds = train_test_split(
    inds, train_size=0.8, random_state=0
)
test_inds, val_inds = train_test_split(
    test_inds, train_size=0.5, random_state=1
)
torch.save({
    "train": pmids[train_inds],
    "test": pmids[test_inds],
    "val": pmids[val_inds]
}, data_dir / "pmids_split.pt")

# %%
training_data = {
    "train": neuro_vectors[train_inds],
    "val": neuro_vectors[val_inds],
    "test": neuro_vectors[test_inds],
}
train_config = MLPAutoencoderTrainConfig(
    output_root=Path("docs/figures/outputs/training_runs"),
    seed=0,
    device=device,
    epochs=101,
    batch_size=256,
    learning_rate=5e-5,
    weight_decay=0.01,
)
train_result = train_mlp_autoencoder(train_config, provider=training_data)

# Keep the model artifact and variable names consumed by later cells.
autoencoder = train_result.model
torch.save(autoencoder.cpu(), data_dir / "autoencoder.pt")
autoencoder = autoencoder.to(device)

# %% [markdown]
# ## Save Latent Vectors

# %%
torch.load(data_dir / "autoencoder.pt", weights_only=False)

# %%
# Encode neuro vectors
with torch.no_grad():
    latent_neuro = autoencoder.encoder(neuro_vectors.to(device)).detach()

torch.save(dict(latent=latent_neuro, pmid=pmids), data_dir / "latent_neuro.pt")

# %% [markdown]
# ## Results
#
# Plot an example.

# %%
import matplotlib.pyplot as plt

masker = load_masker()

# Prediction
idx = 0
img_pred = masker.inverse_transform(
    torch.sigmoid(autoencoder.cpu()(neuro_vectors.cpu()[val_inds][idx])).detach()
)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))

plot_stat_map(img_pred, cut_coords=np.array([8, 4, 60]),
              cmap="cold_hot", vmin=-1, vmax=1, threshold=0.0, draw_cross=False, axes=axes[0]);

# ALEKernel target
plot_stat_map(masker.inverse_transform(neuro_vectors.cpu().numpy()[val_inds][idx]), cut_coords=np.array([8, 4, 60]),
              cmap="cold_hot", vmin=-1, vmax=1, threshold=0.01, draw_cross=False, axes=axes[1]);
