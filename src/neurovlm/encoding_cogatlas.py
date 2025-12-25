import pandas as pd
import torch
import networkx as nx
from neurovlm.data import data_dir
from neurovlm.retrieval_resources import _load_specter

# Load
df = pd.read_parquet(data_dir / "cogatlas.parquet")
df_graph = pd.read_parquet(data_dir / "cogatlas_graph.parquet")

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
specter = _load_specter()
df_filtered = df[df["term"].isin(top_nodes)].reset_index(drop=True)
text = (df_filtered["term"] + "[SEP]" + df_filtered["definition"]).tolist()
terms = df_filtered["term"].tolist()

latent_text = torch.zeros((len(text), 768))
batch_size = 16
for i in range(0, len(text), batch_size):
    with torch.no_grad():
        latent_text[i:i+batch_size] = specter(text[i:i+batch_size])

# Save embeddings with term IDs (similar to latent_specter_wiki and latent_specter2_adhoc)
torch.save({"latent": latent_text, "term": terms}, data_dir / "latent_cogatlas.pt")

# Also save the filtered dataframe with the same terms for easy lookup
df_filtered.to_parquet(data_dir / "cogatlas_filtered.parquet")