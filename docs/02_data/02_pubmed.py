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
import datetime
import hashlib
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import torch
from neurovlm.data import data_dir, load_dataset
from neurovlm.models import Specter
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = select_device()

# %% [markdown]
# # Text Encoding
#
# This notebook encodes (title, abstract) pairs using Specter. Specter was trained on scientific (title, abstract) pairs, suggesting it is likely to perform well with medium length queries. MiniLM-L6 is expected to better handle short form queries. 
#
#
# Use specter is used to to encode (title, abstract) pairs to a 768 dimensional space.
#
# > A. Singh, M. D'Arcy, A. Cohan, D. Downey, and S. Feldman, “SciRepEval: A Multi-Format Benchmark for Scientific Document Representations,” in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:254018137
#

# %%
# Load publications dataframe
df_pubs = load_dataset("pubmed_text")

# Check existing
batch_size = 16
overwrite = False
suffix = ""

if not overwrite and (data_dir / "latent_specter2_adhoc.pt").exists():
    # Append to existing results
    #   save compute when adding more papers
    latent_text_adhoc_exist, pmids_text_adhoc_exist = torch.load(data_dir / "latent_specter2_adhoc.pt", weights_only=False).values()

    mask = ~df_pubs["pmid"].isin(pmids_text_adhoc_exist)
    df_pubs = df_pubs[mask]
    suffix = "_" + hashlib.sha256(
        datetime.datetime.now().isoformat().encode("utf-8")
    ).hexdigest()[:8] # unique identifier

# Load specter
specter_adhoc = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query", device=device)

# %%
# Encode text in batches
os.makedirs(data_dir / "specter", exist_ok=True)

papers = [title + "[SEP]" + abstract
          for title, abstract in zip(df_pubs['name'], df_pubs['description'])]

for i in tqdm(range(0, len(papers), batch_size), total=len(papers)//batch_size):

    with torch.no_grad():
        latent_specter_adhoc = specter_adhoc(papers[i:i+batch_size])

    torch.save(
        {"embeddings": latent_specter_adhoc, "pmid": df_pubs["pmid"].values[i:i+batch_size]},
        data_dir / "specter" / f"encoded_text_specter2_adhoc_{str(i).zfill(5)}{suffix}.pt",
        pickle_protocol=5
    )

# %%
# Stack batched vectors
latent_text_adhoc = torch.zeros((len(df_pubs), 768), dtype=torch.float32)
pmids_text_adhoc = np.zeros(len(df_pubs), dtype=int)

for idx in range(0, len(df_pubs), batch_size):
    latent_text_adhoc[idx:idx+batch_size] , pmids_text_adhoc[idx:idx+batch_size] = torch.load(
        data_dir / "specter" /  f"encoded_text_specter2_adhoc_{str(idx).zfill(5)}{suffix}.pt", weights_only=False
    ).values()

latent_text_adhoc = latent_text_adhoc / torch.norm(latent_text_adhoc, dim=1)[:, None]

if suffix != "":
    # Stack with existing
    latent_text_adhoc = torch.vstack((latent_text_adhoc_exist, latent_text_adhoc))
    pmids_text_adhoc = np.concatenate((pmids_text_adhoc_exist, pmids_text_adhoc))

# %%
# Sort and save
inds = np.argsort(pmids_text_adhoc)
latent_text_adhoc = latent_text_adhoc[inds]
pmids_text_adhoc = pmids_text_adhoc[inds]

torch.save({"latent": latent_text_adhoc, "pmid": pmids_text_adhoc}, data_dir / "latent_specter2_adhoc.pt")
