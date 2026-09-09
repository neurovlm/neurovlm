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
import os
from tqdm.notebook import tqdm
from hashlib import sha256
import pandas as pd
import torch
from neurovlm.data import get_data_dir, load_dataset
from neurovlm.models import Specter, load_model
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

data_dir = get_data_dir()

# %% [markdown]
# ## Load neurowiki
#
# This was scraped outside this notebook.

# %%
# Load
df = load_dataset("wiki")
if os.environ.get("NEUROVLM_SMOKE"):
    df = df.head(32).reset_index(drop=True)
text = list(df["title"] + " [SEP] " + df["summary"])
df["id"] = [sha256(i.encode("utf-8")).hexdigest()[:8] for i in text]

# %% [markdown]
# ## Encode with Specter

# %%
# Encode text in batches
os.makedirs(data_dir / "specter_wiki", exist_ok=True)

specter_adhoc = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query", device=select_device())

batch_size = 32

for i in tqdm(range(0, len(text), batch_size), total=len(text)//batch_size):

    with torch.no_grad():
        latent_specter = specter_adhoc(text[i:i+batch_size])
        ids = df["id"].iloc[i:i+batch_size].tolist()

    torch.save(
        {"embeddings": latent_specter, "id": ids},
        data_dir / "specter_wiki" / f"encoded_text_specter2_adhoc_{str(i).zfill(4)}.pt",
        pickle_protocol=5
    )

# %%
# Stack and save
latent_specter = torch.zeros((len(text), 768))
ids_specter = []
for i in tqdm(range(0, len(text), batch_size), total=len(text)//batch_size):

    enc = torch.load(data_dir / "specter_wiki" / f"encoded_text_specter2_adhoc_{str(i).zfill(4)}.pt", weights_only=False, map_location="cpu")

    with torch.no_grad():
        latent_specter[i:i+batch_size] = enc['embeddings']

    ids_specter.extend(enc["id"])

torch.save(
    {"latent": latent_specter, "id": ids_specter},
    data_dir / "latent_specter2_wiki.pt"
)

# %%
proj_head_mse = load_model("proj_head_text_mse").to("cpu").eval()
proj_head_infonce = load_model("proj_head_text_infonce").to("cpu").eval()

# %%
latent_specter_aligned_mse = torch.zeros((len(text), 384))
latent_specter_aligned_infonce = torch.zeros((len(text), 384))
ids_specter = []
for i in tqdm(range(0, len(text), batch_size), total=len(text)//batch_size):

    enc = torch.load(data_dir / "specter_wiki" / f"encoded_text_specter2_adhoc_{str(i).zfill(4)}.pt", weights_only=False, map_location="cpu")

    with torch.no_grad():
        latent_specter_aligned_mse[i:i+batch_size] = proj_head_mse(enc['embeddings'])
        latent_specter_aligned_mse[i:i+batch_size] = latent_specter_aligned_mse[i:i+batch_size] / latent_specter_aligned_mse[i:i+batch_size].norm(dim=1)[:, None]

        latent_specter_aligned_infonce[i:i+batch_size] = proj_head_infonce(enc['embeddings'])
        latent_specter_aligned_infonce[i:i+batch_size] = latent_specter_aligned_infonce[i:i+batch_size] / latent_specter_aligned_infonce[i:i+batch_size].norm(dim=1)[:, None]
    ids_specter.extend(enc["id"])

# %%
torch.save(
    {"latent": latent_specter_aligned_mse, "id": ids_specter},
    data_dir / "latent_specter_wiki_aligned_adhoc_mse.pt"
)

torch.save(
    {"latent": latent_specter_aligned_infonce, "id": ids_specter},
    data_dir / "latent_specter_wiki_aligned_adhoc_infonce.pt"
)
