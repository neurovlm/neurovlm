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
#     display_name: .conda
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import torch
from nilearn import image, datasets
from neurovlm.data import data_dir, load_dataset, load_masker
from neurovlm.data.coordinates import coords_to_vectors
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = select_device()

# %% [markdown]
# # Coordinate Smoothing
#
# Use nimare to smooth coordinates with activation likelihood estimation (ALE). The result of this notebook is a 28542 vector per publication that represents coordinates reported in MNI space.
#
# ## Load

# %%
# Load
df_pubs = load_dataset("pubmed_text")
df_coords = load_dataset("pubmed_coordinates")

# Sort
df_pubs = df_pubs.sort_values("pmid")
df_coords = df_coords.sort_values("pmid")

df_coords.shape, df_pubs.shape

# %% [markdown]
# ## Smoothing
#
# Use a 9mm ALE kernel.

# %%
neuro_vectors_ale = coords_to_vectors(df_coords, fwhm=9)

torch.save(
    {"neuro_vectors": neuro_vectors_ale, "pmid": df_pubs["pmid"].to_numpy()},
    data_dir / "neuro_vectors_ale.pt"
)

# %% [markdown]
# ## DiFuMo
#
# Move ALE images to DiFuMo.

# %%
# Tensor implementation
from nilearn.maskers import NiftiMapsMasker

def build_difumo_projection_matrix(difumo_masker, masker, device="cuda", dtype=torch.float32):
    """
    Builds P (n_vox, k) in mask space, and s = sum(P, dim=0).
    Requires difumo_masker and masker to already be aligned (same mask/space),
    and neuro_vectors to be in masker.transform() voxel order.
    """
    # Ensure difumo masker has resampled maps ready
    difumo_masker.fit()

    # Get maps image that difumo_masker is using after fit/resampling
    maps_img = difumo_masker.maps_img_  # nibabel-like img
    maps_4d = maps_img.get_fdata(dtype=np.float32)  # (X,Y,Z,k)

    # Mask in the same space/order as neuro_vectors (masker.transform output)
    mask_img = masker.mask_img_ if hasattr(masker, "mask_img_") else masker.mask_img
    mask = mask_img.get_fdata().astype(bool)

    # Flatten maps into (n_vox, k) in mask voxel order
    # NOTE: nilearn's masker uses C-order flattening; this matches get_fdata()[mask]
    P = maps_4d[mask, :]  # (n_vox, k)

    P = torch.as_tensor(P, device=device, dtype=dtype).contiguous()
    s = P.sum(dim=0).clamp_min(1e-12)  # avoid divide-by-zero

    return P, s

@torch.no_grad()
def difumo_project_voxels(X, P, s):
    """
    X: (..., n_vox) tensor (single vector or batch)
    P: (n_vox, k)
    s: (k,)
    Returns: (..., n_vox) reconstructed/projection result
    """
    # coefficients: (..., k)
    C = (X @ P) / s
    # reconstruction: (..., n_vox)
    X_hat = C @ P.T
    return X_hat


# %%
masker = load_masker()
difumo = datasets.fetch_atlas_difumo(dimension=512)
atlas_img = image.load_img(difumo.maps)
atlas_img = image.resample_to_img(
    atlas_img, masker.mask_img,
    force_resample=True,
    interpolation="nearest",
    copy_header=True
)
difumo_masker = NiftiMapsMasker(atlas_img)
difumo_masker.fit()

# %%
neuro_vectors, pmids = torch.load(
    data_dir / "neuro_vectors_ale.pt", weights_only=False
).values()

# Move onto gpu
X = neuro_vectors.to(device=device, dtype=torch.float32)

# DiFuMo projection
P, s = build_difumo_projection_matrix(difumo_masker, masker, device=device)
X_hat = difumo_project_voxels(X, P, s)

# Filter for non-zero
mask = X_hat.sum(dim=1) != 0
neuro_vectors = neuro_vectors[mask.cpu()]
pmids = pmids[mask.cpu()]
X_hat = X_hat[mask]

# Quantile ceiling
thresholds = torch.quantile(X_hat, .9999, dim=1)
X_hat_norm = X_hat / thresholds.unsqueeze(1)

# Clamp to (0, 1)
X_hat_norm[X_hat_norm < 0] = 0
X_hat_norm[X_hat_norm > 1] = 1

# Save
torch.save(
    {"neuro_vectors": X_hat_norm, "pmid": pmids},
    data_dir / "neuro_vectors.pt"
)

# %%
# Read
neuro_vectors, pmids = torch.load(data_dir / "neuro_vectors.pt", weights_only=False).values()
neuro_vectors_ale, pmids_ale = torch.load(data_dir / "neuro_vectors_ale.pt", weights_only=False).values()

mask = pd.Series(pmids_ale).isin(pmids)
neuro_vectors_ale = neuro_vectors_ale[mask]
pmids_ale = pmids_ale[mask]
assert (pmids == pmids_ale).all()

# %%
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from neurovlm.data import load_masker
from nilearn.datasets import load_mni152_template
masker = load_masker()

# select random examples to plot
torch.manual_seed(0)
inds = torch.randperm(len(neuro_vectors))
inds = inds[:20]

temp = load_mni152_template(resolution=1)
out = data_dir / "difumo_examples"
out.mkdir(exist_ok=True)

# %%
for i in tqdm(range(len(inds)), total=len(inds)):

    if (out / f"{str(i).zfill(2)}_difumo.png").exists():
        continue

    img_a = masker.inverse_transform(neuro_vectors[inds[i]].cpu().numpy())
    img_b = masker.inverse_transform(neuro_vectors_ale[inds[i]].cpu().numpy())

    difumo = plot_stat_map(img_a, bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                         annotate=False, cmap="hot", vmin=0.1, vmax=1)

    plt.savefig(out / f"{str(i).zfill(2)}_difumo.png", dpi=300)
    plt.close()

    plot_stat_map(img_b, bg_img=temp, black_bg=False, draw_cross=False, colorbar=False,
                 cut_coords=difumo.cut_coords, annotate=False, cmap="hot", vmin=.1, vmax=1)
    plt.savefig(out / f"{str(i).zfill(2)}_ale.png", dpi=300)
    plt.close()
