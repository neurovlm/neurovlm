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
import gzip, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from nilearn.image import resample_to_img
from neurovlm.data import data_dir, load_masker, load_dataset, load_latent
from neurovlm.models import load_model
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

from neurovlm.evaluation.notebook_utils import resolve_evaluation_output_dir

evaluation_output_dir = resolve_evaluation_output_dir()
evaluation_output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load models
device = select_device()
autoencoder = load_model("autoencoder").to(device).eval()
masker = load_masker()

# Load pubmed test set
neuro_vectors, pmids = torch.load(
    data_dir / "neuro_vectors.pt", weights_only=False,
).values()

train, test, val = torch.load(data_dir / "pmids_split.pt", weights_only=False).values()
X_pubmed_masked = neuro_vectors[pd.Series(pmids).isin(test)].to("cpu")

np.random.seed(0)
inds = np.random.choice(np.arange(len(X_pubmed_masked)), 8)

X_pubmed_masked = X_pubmed_masked[inds]
X_pubmed = [masker.inverse_transform(i) for i in X_pubmed_masked.cpu().numpy()]

# %%
# Load network atlases
networks = load_dataset("networks")

network_imgs = []
for k in networks.keys():
    for a in networks[k].keys():
        network_imgs.append((k, a, nib.Nifti1Image(networks[k][a]["array"], affine=networks[k][a]["affine"])))

networks = [i for i in network_imgs if i[0] not in ["UKBICA", "HCPICA"]]

df = pd.read_csv(data_dir / "network_examples.csv")
inds = df["Unnamed: 0"].values

X_networks_masked = torch.load(data_dir / "networks_emb.pt")
X_networks_masked = X_networks_masked[inds]

X_networks = [
    resample_to_img(net[2], masker.mask_img, force_resample=True, copy_header=True)
    for i, net in enumerate([networks[i] for i in inds])
]

# %%
# Load neurovault
neurovault_data = torch.load(
    data_dir / "neurovault.pt", weights_only=False
)
df_neuro, df_pubs, _, neuro_clust, _, _, _ = neurovault_data.values()

np.random.seed(1)
inds = np.random.choice(np.arange(len(neuro_clust)), 8)

X_neurovault_masked = torch.tensor(neuro_clust[inds])
X_neurovault = [masker.inverse_transform(i) for i in neuro_clust[inds]]

# %%
X_pubmed_masked = torch.stack([torch.from_numpy(masker.transform(i)).float() for i in X_pubmed])
X_networks_masked = torch.stack([torch.from_numpy(masker.transform(i)).float() for i in X_networks])
X_neurovault_masked = torch.stack([(i > 0).float() for i in X_neurovault_masked])

# %%
with torch.no_grad():
    X_networks_re = torch.sigmoid(autoencoder(X_networks_masked.to(device))).detach().cpu().numpy()
    X_pubmed_re = torch.sigmoid(autoencoder(X_pubmed_masked.to(device))).detach().cpu().numpy()
    X_neurovault_re = torch.sigmoid(autoencoder(X_neurovault_masked.to(device))).detach().cpu().numpy()

X_networks_re = [masker.inverse_transform(i) for i in X_networks_re]
X_pubmed_re = [masker.inverse_transform(i) for i in X_pubmed_re]
X_neurovault_re = [masker.inverse_transform(i) for i in X_neurovault_re]

# %%
from nilearn.plotting import plot_stat_map
from nilearn.datasets import load_mni152_template
from nilearn.image import threshold_img
from joblib import Parallel, delayed
from tqdm.notebook import tqdm


# %%
def cluster(i, img):
    thr_img = threshold_img(
        img, "99%",
        cluster_threshold=50, two_sided=False, copy_header=True
    )
    return i, thr_img

def compute(X):
    X = Parallel(n_jobs=16, backend="loky")(
        delayed(cluster)(i, arr) for i, arr in
        enumerate(tqdm(X, total=len(X)))
    )
    X.sort(key=lambda i: i[0])
    X = [i[1] for i in X]
    return X

X_pubmed = compute(X_pubmed)
X_pubmed_re = compute(X_pubmed_re)
X_neurovault_re = compute(X_neurovault_re)
X_networks_re = compute(X_networks_re)

# %%
import os
out = evaluation_output_dir / "ae_qual"
if not out.exists():
    os.mkdir(out)

# %%
temp = load_mni152_template(resolution=1)

for i in range(8):

    # Networks
    pred = plot_stat_map(X_networks_re[i], bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                         annotate=False, cmap="Reds", vmin=0, vmax=1)
    plt.savefig(out / f"ae_{str(i).zfill(2)}_networks_pred.png", dpi=300)
    plt.close()

    plot_stat_map(X_networks[i], bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                  cut_coords=pred.cut_coords, annotate=False, cmap="Reds")
    plt.savefig(out / f"ae_{str(i).zfill(2)}_networks_orig.png", dpi=300)
    plt.close()

    # PubMed
    pred = plot_stat_map(X_pubmed_re[i], bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                         annotate=False, cmap="Reds", vmin=0, vmax=1)
    plt.savefig(out / f"ae_{str(i).zfill(2)}_pm_pred.png", dpi=300)
    plt.close()

    plot_stat_map(X_pubmed[i], bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                  cut_coords=pred.cut_coords, annotate=False, cmap="Reds")
    plt.savefig(out / f"ae_{str(i).zfill(2)}_pm_orig.png", dpi=300)
    plt.close()

    # NeuroVault
    pred = plot_stat_map(X_neurovault_re[i], bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                         annotate=False, cmap="Reds", vmin=0, vmax=1)
    plt.savefig(out / f"ae_{str(i).zfill(2)}_nv_pred.png", dpi=300)
    plt.close()

    plot_stat_map(X_neurovault[i], bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                  cut_coords=pred.cut_coords, annotate=False, cmap="Reds")
    plt.savefig(out / f"ae_{str(i).zfill(2)}_nv_orig.png", dpi=300)
    plt.close()

    print(i)

# %%
from nilearn import datasets
bg = datasets.load_mni152_template(resolution=4)

def plot_set(X, X_re, row, axes):

    for i in range(len(X)):
        ijk = np.unravel_index(np.argmax(X_re[i].get_fdata()),
                        X_re[i].shape)

        xyz = nib.affines.apply_affine(X_re[i].affine, ijk)

        plot_stat_map(X[i], vmin=0, cut_coords=xyz, cmap="Reds", annotate=False, bg_img=bg,
                      colorbar=False, draw_cross=False, threshold=1e-2, axes=axes[row, i])

        plot_stat_map(X_re[i], vmin=0, cut_coords=xyz, cmap="Reds", annotate=False, bg_img=bg,
                      colorbar=False, draw_cross=False, threshold=1e-2, axes=axes[row+1, i])

# fig, axes = plt.subplots(nrows=2, ncols=24, figsize=(18, 2))
# plot_set(X_networks, X_networks_re, 0, axes)

fig, axes = plt.subplots(nrows=int(2*3), ncols=8, figsize=(14, 6))
plot_set(X_networks, X_networks_re, 0, axes)
plot_set(X_neurovault, X_neurovault_re, 2, axes)
plot_set(X_pubmed, X_pubmed_re, 4, axes)
plt.savefig(evaluation_output_dir / "autoencoder_examples_highres.svg")

# %%
from nilearn import datasets
bg = datasets.load_mni152_template(resolution=4)

def plot_set(X, X_re, row, axes):

    for i in range(len(X)):
        ijk = np.unravel_index(np.argmax(X_re[i].get_fdata()),
                        X_re[i].shape)

        xyz = nib.affines.apply_affine(X_re[i].affine, ijk)

        plot_stat_map(X[i], vmin=0, cut_coords=xyz, cmap="Reds", annotate=False, bg_img=bg, black_bg=False,
                      colorbar=False, draw_cross=False, threshold=1e-2, axes=axes[row, i])

        plot_stat_map(X_re[i], vmin=0, cut_coords=xyz, cmap="Reds", annotate=False, bg_img=bg, black_bg=False,
                      colorbar=False, draw_cross=False, threshold=1e-2, axes=axes[row+1, i])

# fig, axes = plt.subplots(nrows=2, ncols=24, figsize=(18, 2))
# plot_set(X_networks, X_networks_re, 0, axes)

fig, axes = plt.subplots(nrows=int(2*3), ncols=8, figsize=(14, 6))
plot_set(X_networks, X_networks_re, 0, axes)
plot_set(X_neurovault, X_neurovault_re, 2, axes)
plot_set(X_pubmed, X_pubmed_re, 4, axes)
plt.savefig(evaluation_output_dir / "autoencoder_examples_highres_.svg")

# %%
