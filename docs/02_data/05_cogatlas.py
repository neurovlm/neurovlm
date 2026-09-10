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
import torch
import networkx as nx
from neurovlm.data import data_dir, load_dataset
from neurovlm.models import Specter
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# %% [markdown]
# ## Cognitive Atlas
# Embed with specter2.

# %%
# Load
df = load_dataset("cogatlas")
df_graph = load_dataset("cogatlas_graph")

# Replace special characters
df["definition"] = df["definition"].str.replace("\n", "").replace("\r", "")

# Manual filter, these descriptions were bad
drop = [
    "active cognitive inhibition",
    "transfer data"
]

df['term'] = df['term'].str.lower()

df = df[~df['term'].isin(drop)]

df_graph = df_graph[
    (~df_graph["parent"].isin(drop)) &
    (~df_graph["child"].isin(drop))
]

# Build graph
G = nx.DiGraph()
for _, row in df_graph.iterrows():
    parent = str(row["parent"]).strip()
    child  = str(row["child"]).strip()
    rel    = str(row["relationship"]).strip().upper()
    G.add_edge(child, parent, relationship=rel)

# Node centrality
centrality = nx.degree_centrality(G)

# Top-k most central nodes
k = 300
top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:k]

# Encode top-k titles + descriptions
df = df[df["term"].isin(top_nodes)]
text = (df["term"] + "[SEP]" + df["definition"]).tolist()

# %%
# Embed
specter_adhoc = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query", device=select_device())
latent_text = torch.zeros((len(text), 768))
batch_size = 32
for i in range(0, len(text), batch_size):
    with torch.no_grad():
        latent_text[i:i+batch_size] = specter_adhoc(text[i:i+batch_size])

# Save
torch.save(latent_text, data_dir / "latent_cogatlas.pt")
