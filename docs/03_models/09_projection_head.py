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
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from neurovlm.data import data_dir, load_masker
from neurovlm.models import ProjHead
from neurovlm.models.losses import InfoNCELoss


def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = select_device()

# %% [markdown]
# # Projection Head
#
# Projection head refers to a small network to align the latent spaces between text and neuroimages. The training regime starts with MSELoss, then gradually removed the influences of outliers through truncation, i.e. masking out the top-k% of loss instances from gradient computation.

# %%
# Load autoencoder
autoencoder = torch.load(data_dir / "autoencoder.pt", weights_only=False)

# Load vectors from 01_coordinate.ipynb
neuro_vectors, pmids = torch.load(
    data_dir / f"neuro_vectors.pt", weights_only=False,
).values()

# Load splits
ids_train, ids_test, ids_val = torch.load(
    data_dir / "pmids_split.pt", weights_only=False, map_location="cpu"
).values()

# Load encoded neuroimagesb from 08_autoencoder.ipynb
latent_neuro, pmids_latent = torch.load(
    data_dir / f"latent_neuro.pt", weights_only=False, map_location="cpu"
).values()

# Load encoded text from 02_pubmned.ipynb
latent_text_specter, pmids_specter = torch.load(
    data_dir / f"latent_specter2_adhoc.pt", weights_only=False, map_location="cpu"
).values()

# Reduce to instances with <= 100 coordinates
mask = pd.Series(pmids).isin(pmids_latent)
neuro_vectors = neuro_vectors[mask]
pmids = pmids[mask]
assert (pmids == pmids_latent).all()

mask = pd.Series(pmids_specter).isin(pmids_latent)
latent_text_specter = latent_text_specter[mask]
pmids_specter = pmids_specter[mask]
assert (pmids == pmids_specter).all()

# Split
pmids = pd.Series(pmids)
train_mask = pmids.isin(ids_train)
val_mask = pmids.isin(ids_val)
test_mask = pmids.isin(ids_test)

X_train  = latent_text_specter[train_mask]
X_test  = latent_text_specter[test_mask]
X_val  = latent_text_specter[val_mask]


# %% [markdown]
# ## MSE

# %%
class ProjHeadWithDecoder(nn.Module):
    def __init__(self, base, decoder):
        super().__init__()
        self.base = base
        self.decoder = decoder
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    def forward(self, X):
        return self.decoder(self.base(X))


y_train = neuro_vectors[train_mask]
y_test = neuro_vectors[test_mask]
y_val = neuro_vectors[val_mask]

proj_head_with_decoder = ProjHeadWithDecoder(
    ProjHead(seed=123, latent_in_dim=768, hidden_dim=512, latent_out_dim=384),
    autoencoder.decoder,
).to(device)
X_train_device = X_train.to(device)
y_train_device = y_train.to(device)
X_val_device = X_val.to(device)
y_val_device = y_val.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(proj_head_with_decoder.parameters(), lr=5e-5)
best_val_loss = float("inf")
best_state = None

for epoch in tqdm(range(201), total=201):
    proj_head_with_decoder.train()
    torch.manual_seed(epoch)
    random_indices = torch.randperm(len(X_train_device))
    for start in range(0, len(X_train_device), 1024):
        indices = random_indices[start : start + 1024]
        prediction = proj_head_with_decoder(X_train_device[indices])
        loss = loss_fn(prediction, y_train_device[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0 or epoch == 200:
        proj_head_with_decoder.eval()
        with torch.no_grad():
            val_loss = loss_fn(proj_head_with_decoder(X_val_device), y_val_device)
        print(f"Epoch: {epoch}, val loss: {float(val_loss):.5g}")
        if float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in proj_head_with_decoder.state_dict().items()
            }

if best_state is not None:
    proj_head_with_decoder.load_state_dict(best_state)
proj_head = proj_head_with_decoder.base.cpu()
torch.save(proj_head, data_dir / "proj_head_text_mse.pt")

# %%
proj_head = torch.load(data_dir / "proj_head_text_mse.pt", weights_only=False)

# %%
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.plotting import plot_glass_brain, view_img
from neurovlm.models import Specter

# Load models
decoder = autoencoder.decoder.to("cpu")
specter = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query")

# Load mask
masker = load_masker()
mask = masker.mask_img_.get_fdata().astype(bool)
affine = masker.mask_img_.affine

# %% [markdown]
# ## Contrastive Loss

# %%
# Split data
X_train_image = latent_neuro[train_mask].to(device)
X_train_text  = latent_text_specter[train_mask].to(device)
X_val_image = latent_neuro[val_mask].to(device)
X_val_text  = latent_text_specter[val_mask].to(device)

# Unit norm
X_train_text  = F.normalize(X_train_text,  dim=1, eps=1e-8)
X_val_text    = F.normalize(X_val_text,    dim=1, eps=1e-8)

# Models
proj_head = torch.load(data_dir / f"proj_head_text_mse.pt", weights_only=False)

proj_head_text  = proj_head.to(device) # initialize with the decoder model
proj_head_image = ProjHead(
    seed=123, latent_in_dim=384, hidden_dim=384, latent_out_dim=384
).to(device)

# Settings
loss_fn = InfoNCELoss(temperature=0.07)
n_epochs = 301
batch_size = 2048
lr = 1e-5
optimizer = AdamW([*proj_head_text.parameters(), *proj_head_image.parameters()], lr=lr)
interval = 20

# Train
iterable = tqdm(range(n_epochs), total=n_epochs)

for iepoch in iterable:

    proj_head_text.train()
    proj_head_image.train()

    # Randomly shuffle and batch
    torch.manual_seed(iepoch)
    rand_inds = torch.randperm(len(X_train_image), device=device)

    for i in range(0, len(X_train_image), batch_size):
        idx = rand_inds[i:i+batch_size]

        # Forward
        y_text  = proj_head_text(X_train_text[idx])
        y_image = proj_head_image(X_train_image[idx])

        # Loss
        loss = loss_fn(y_text, y_image)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Step
        optimizer.step()

    # Report validation
    if iepoch % interval == 0 or iepoch == (n_epochs - 1):
        proj_head_text.eval()
        proj_head_image.eval()
        with torch.no_grad():
            y_text  = proj_head_text(X_val_text)
            y_image = proj_head_image(X_val_image)
            val_loss = loss_fn(y_text, y_image)
            print(f"Epoch: {iepoch}, val loss: {float(val_loss):.5g}")

torch.save(proj_head_text, data_dir / "proj_head_text_infonce.pt")
torch.save(proj_head_image, data_dir / "proj_head_image_infonce.pt")

# %% [markdown]
# ## Example Predictions

# %%
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.plotting import plot_glass_brain
from neurovlm.models import Specter

# Load models
decoder = autoencoder.decoder.to("cpu")
specter = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query")
proj_head = torch.load(data_dir / f"proj_head_text_mse.pt", weights_only=False)

# %%
queries = [
    # Regions
    "visual cortex",
    "motor cortex",
    "temporal lobe",
    "cerebellum",
    "precuneus",
    "hippocampus",
    # Neurotransmitters
    "dopamine",
    "serotonin",
    "gaba",
    "norepinephrine",
    # Networks
    "default mode network",
    "cingulo opercular network",
    "executive control network",
    "sensorimotor network",
    # Cognitiion
    "memory",
    "emotion",
    "attention",
    "language",
    "reward",
    "planning"
]

# %%
# Load mask
masker = load_masker()
mask = masker.mask_img_.get_fdata().astype(bool)
affine = masker.mask_img_.affine

# Query
decoder = autoencoder.decoder.to("cpu")

fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(8, 20))
axes = axes.flatten()

for i, query in tqdm(enumerate(queries), total=len(queries)):

    with torch.no_grad():
        # Encode text
        encoded_text_specter = specter(query)
        encoded_text_specter = encoded_text_specter / encoded_text_specter.norm()

        # Projection head
        aligned_text_specter = proj_head.to("cpu")(encoded_text_specter)

        # Decode brain
        neuro_pred = torch.sigmoid(decoder(aligned_text_specter)).detach().numpy()[0]

    # Plot
    pred = np.zeros(mask.shape)
    pred[mask] = neuro_pred
    img = nib.Nifti1Image(pred, affine)
    plot_glass_brain(img, threshold=0, axes=axes[i], colorbar=False, resampling_interpolation="nearest")
    axes[i].set_title(query)
