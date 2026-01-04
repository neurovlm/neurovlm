import pandas as pd
import torch
from neurovlm.data import data_dir
from neurovlm.retrieval_resources import _load_specter

# Load
df = pd.read_parquet(data_dir / "cogatlas_disorder.parquet")

# Replace special characters
df["definition"] = df["definition"].str.replace("\n", "").replace("\r", "")

# Manual filter if needed
drop = []

df['term'] = df['term'].str.lower()

df = df[~df['term'].isin(drop)]

# Since disorders don't have a graph structure, we'll encode all disorders
df_filtered = df.reset_index(drop=True)

# Encode titles + descriptions
specter = _load_specter()
text = (df_filtered["term"] + "[SEP]" + df_filtered["definition"]).tolist()
terms = df_filtered["term"].tolist()

latent_text = torch.zeros((len(text), 768))
batch_size = 16
for i in range(0, len(text), batch_size):
    with torch.no_grad():
        latent_text[i:i+batch_size] = specter(text[i:i+batch_size])

# Save embeddings with term IDs (similar to latent_specter_wiki and latent_specter2_adhoc)
torch.save({"latent": latent_text, "term": terms}, data_dir / "latent_cogatlas_disorder.pt")

# Also save the filtered dataframe with the same terms for easy lookup
df_filtered.to_parquet(data_dir / "cogatlas_disorder_filtered.parquet")
