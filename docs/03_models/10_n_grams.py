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
import numpy as np
import pandas as pd
import torch
from torch import nn
from neurovlm.resources.loaders import (
    _load_pubmed_dataframe, _load_latent_text
)
from neurovlm.data import data_dir
from neurovlm.models import ConceptClf


def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# %% [markdown]
# # Concept Classifier
#
# The concept classifier predicts which concepts are present given a latent neuro embeddings. The top-10 related concepts are passed to an LLM to summarize the brain map. Here, Llama-3.1-8B-Instruct is used to generated interpretations. Any language model may be used. Larger models or models trained one neuroscience literature may provided better brain map interpretations.

# %%
# N-gram embeddings, from 06_n_grams.ipynb
ngram_emb = torch.load(data_dir / "ngram_emb.pt")

# load text
df = _load_pubmed_dataframe()
df.sort_values(by="pmid", inplace=True)
text = df["name"] + " [SEP] " + df["description"]

# load pre-computed ngrams from 06_n_grams.ipynb
X = np.load(data_dir / "ngram_matrix.npy")
features = np.load(data_dir / "ngram_labels.npy")

# load latent text
latent, pmids = _load_latent_text()

# %%
# cosine similarity as target
y = latent @ (ngram_emb / ngram_emb.norm(dim=1)[:, None]).T
m = (y < 0.) | (torch.from_numpy(X) == 0.)
y[m==1] = 0.

# transform cosine similarity ~= probabilities
t = 0.03
tau = 0.08
y = torch.sigmoid((y - t)/ tau)

y[m] = 0.
y = y.numpy()

# %%
import torch.nn.functional as F
proj_head = torch.load(data_dir / f"proj_head_image_infonce.pt", weights_only=False, map_location="cpu")

# %%
# ensure latent neuro vectors align with df
latent_neuro, pmid = torch.load(
    data_dir / "latent_neuro.pt", weights_only=False, map_location="cpu"
).values()
with torch.no_grad():
    latent_neuro = F.normalize(proj_head(F.normalize(latent_neuro, dim=1)), dim=1)

assert (df["pmid"] == df["pmid"].sort_values()).all()

mask = df['pmid'].isin(pmid)
df, y = df[mask], y[mask]
df.reset_index(inplace=True, drop=True)

# %%
# load data splits
train_ids, test_ids, val_ids = torch.load(data_dir / "pmids_split.pt", weights_only=False).values()
train_ids.sort()
val_ids.sort()
test_ids.sort()

def split(df, latent, y, pmids, device):
    mask = df['pmid'].isin(pmids).to_numpy()
    X = latent[torch.from_numpy(mask)].clone().to(device)
    y = torch.from_numpy(y[mask].copy()).float().to(device)
    pmids = pmids[pd.Series(pmids).isin(df["pmid"])]
    return X, y, pmids

device = select_device()
X_train, y_train, train_ids = split(df, latent_neuro, y, train_ids, device)
X_val, y_val, val_ids = split(df, latent_neuro, y,  val_ids, device)
X_test, y_test, test_ids = split(df, latent_neuro, y, test_ids, device)

# ensure sorted
assert (df['pmid'] == df['pmid'].sort_values()).all()
assert (train_ids == np.sort(train_ids)).all()
assert (val_ids == np.sort(val_ids)).all()
assert (test_ids == np.sort(test_ids)).all()

# %%
clf = ConceptClf(X.shape[1]).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(clf.parameters(), lr=3e-5)
best_val_loss = float("inf")
best_state = None

for epoch in range(201):
    clf.train()
    torch.manual_seed(epoch)
    random_indices = torch.randperm(len(X_train))
    for start in range(0, len(X_train), 1028):
        indices = random_indices[start : start + 1028]
        prediction = clf(X_train[indices])
        loss = loss_fn(prediction, y_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0 or epoch == 200:
        clf.eval()
        with torch.no_grad():
            val_loss = loss_fn(clf(X_val), y_val)
        print(f"Epoch: {epoch}, val loss: {float(val_loss):.5g}")
        if float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in clf.state_dict().items()
            }

if best_state is not None:
    clf.load_state_dict(best_state)
clf.eval()
torch.save(clf.cpu(), data_dir / "concept_clf.pt")
clf = clf.to(device)

# %%
from neurovlm.resources.loaders import _load_masker, _load_autoencoder
import gzip, pickle
import nibabel as nib
from nilearn.image import resample_to_img

masker = _load_masker()
autoencoder = _load_autoencoder()

# Load network atlases
with gzip.open(data_dir / "networks_arrays.pkl.gz", "rb") as f:
    networks = pickle.load(f)
    
networks = [(k, n, nib.Nifti1Image(networks[k][n]["array"], affine=networks[k][n]["affine"]))
            for k in networks.keys() for n in networks[k].keys()]

# %%
i = 0

x = masker.transform(
    resample_to_img(networks[i][2], masker.mask_img, interpolation="nearest", force_resample=True, copy_header=True)
)
x = autoencoder.encoder(torch.from_numpy(x))

scores = torch.sigmoid(clf(x.to(device)).cpu().detach())
network_name = f"{networks[i][0]}_{networks[i][1]}"
print(f"top predicted terms for {network_name}:")
features[scores.argsort(descending=True)][:50]
