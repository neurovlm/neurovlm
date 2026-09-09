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
#     display_name: neurovlm
#     language: python
#     name: neurovlm
# ---

# %% [markdown]
# # KG-MeSH: Unified Knowledge Graph Term Definitions
#
# Generates LLM definitions for MeSH-primary nodes in the unified knowledge graph,
# then embeds them with SPECTER2 for use as a retrieval dataset (`pubmed_mesh`).
#
# **Output files** (upload to `neurovlm/embedded_text` on HuggingFace):
# - `kg_mesh.parquet` — term + definition table
# - `latent_kg_mesh.pt` — SPECTER2 embeddings `{"latent": Tensor[N, 768], "term": List[str]}`
#
# **Runtime**: ~2-4 hours for ~33K terms using 8 concurrent Ollama workers.

# %%
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm.notebook import tqdm

from neurovlm.data import data_dir
from neurovlm.models import Specter
def select_device() -> str:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# %%
# ── Config ────────────────────────────────────────────────────────────────────
KG_DIR          = Path("../../../experiments/data/unified_kg")
CHECKPOINT_PATH = KG_DIR / "kg_mesh_definitions_checkpoint.parquet"
OLLAMA_MODEL    = "qwen2.5:7b-instruct"   # change to 14b for higher quality
OLLAMA_URL      = "http://localhost:11434/api/generate"
WORKERS         = 8    # concurrent Ollama requests
CHECKPOINT_EVERY = 200 # save checkpoint every N completed terms
MAX_EDGES       = 5    # top-k relations to include in LLM prompt

print(f"data_dir : {data_dir}")
print(f"KG_DIR   : {KG_DIR}")

# %% [markdown]
# ## 1. Load Knowledge Graph

# %%
nodes = pd.read_parquet(KG_DIR / "unified_kg_nodes.parquet")
edges = pd.read_parquet(KG_DIR / "unified_kg_edges.parquet")

print(f"Total nodes : {len(nodes):,}")
print(f"Total edges : {len(edges):,}")
print("\nNode types:")
print(nodes["node_type"].value_counts())
print("\nPrimary sources:")
print(nodes["primary_source"].value_counts())

# %% [markdown]
# ## 2. Filter Out CogAtlas-Primary Nodes
#
# The unified KG includes nodes whose `primary_source` is `cogatlas`.  
# Those terms are already embedded in the `cogatlas` retrieval dataset, so we
# exclude them here to avoid duplicates.

# %%
mesh_nodes = nodes[nodes["primary_source"] != "cogatlas"].copy().reset_index(drop=True)

removed = len(nodes) - len(mesh_nodes)
print(f"Removed {removed:,} cogatlas-primary nodes")
print(f"Remaining : {len(mesh_nodes):,} nodes")
print("\nRemaining node types:")
print(mesh_nodes["node_type"].value_counts())

# %% [markdown]
# ## 3. Build Edge Context Helper

# %%
# Fast id → name lookup
name_lookup: dict = nodes.set_index("canonical_id")["name"].to_dict()

# Pre-index edges by subject and object for fast lookup
print("Indexing edges by subject_id …")
edges_by_subject = edges.groupby("subject_id", sort=False)
print("Indexing edges by object_id …")
edges_by_object  = edges.groupby("object_id",  sort=False)
print("Done.")


# %%
def get_edge_context(node_id: str, max_edges: int = MAX_EDGES) -> str:
    """Return a short human-readable relation summary for a node."""
    parts: list[str] = []

    # Edges where this node is the subject (outgoing)
    if node_id in edges_by_subject.groups:
        sub = edges_by_subject.get_group(node_id).nlargest(max_edges, "weight")[
            ["relation_type", "object_id"]
        ]
        for _, row in sub.iterrows():
            other = name_lookup.get(row["object_id"], row["object_id"])
            rel   = row["relation_type"].replace("_", " ")
            parts.append(f"{rel} {other}")

    # Edges where this node is the object (incoming)
    if node_id in edges_by_object.groups:
        obj = edges_by_object.get_group(node_id).nlargest(max_edges, "weight")[
            ["subject_id", "relation_type"]
        ]
        for _, row in obj.iterrows():
            other = name_lookup.get(row["subject_id"], row["subject_id"])
            rel   = row["relation_type"].replace("_", " ")
            parts.append(f"{other} {rel} [this term]")

    # Deduplicate and cap
    seen, unique = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
        if len(unique) >= max_edges * 2:
            break

    return "; ".join(unique)


# Smoke test
sample = mesh_nodes.iloc[5]
ctx    = get_edge_context(sample["canonical_id"])
print(f"Node  : {sample['name']} ({sample['node_type']})")
print(f"Context: {ctx}")


# %% [markdown]
# ## 4. Ollama Definition Generator

# %%
def generate_definition(
    term: str,
    node_type: str,
    edge_context: str,
    model: str = OLLAMA_MODEL,
) -> str:
    """Generate a biomedical definition via Ollama, incorporating graph relations."""
    context_line = f"\nKnowledge graph relations: {edge_context}" if edge_context else ""

    prompt = (
        f'Define the biomedical term "{term}" (category: {node_type}).'
        f"{context_line}\n"
        "Write a concise 2–3 sentence definition that describes what it is, "
        "its biomedical significance, and incorporates the relationships listed above. "
        "Reply with only the definition text, no preamble or bullet points."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 180},
            },
            timeout=90,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        pass
    return ""


# Smoke test
test_node = mesh_nodes.iloc[10]
test_ctx  = get_edge_context(test_node["canonical_id"])
test_def  = generate_definition(test_node["name"], test_node["node_type"], test_ctx)
print(f"Term      : {test_node['name']}")
print(f"Type      : {test_node['node_type']}")
print(f"Relations : {test_ctx}")
print(f"Definition: {test_def}")

# %% [markdown]
# ## 5. Generate Definitions (with Checkpointing)
#
# Uses `ThreadPoolExecutor` for concurrent Ollama requests.  
# Progress is saved to `CHECKPOINT_PATH` every `CHECKPOINT_EVERY` terms so the
# notebook can be safely interrupted and resumed.

# %%
# Load checkpoint if available
if CHECKPOINT_PATH.exists():
    done_df  = pd.read_parquet(CHECKPOINT_PATH)
    done_ids = set(done_df["canonical_id"].tolist())
    print(f"Resuming from checkpoint: {len(done_ids):,} terms already done")
else:
    done_df  = pd.DataFrame(columns=["canonical_id", "term", "definition"])
    done_ids = set()
    print("Starting fresh")

remaining = mesh_nodes[~mesh_nodes["canonical_id"].isin(done_ids)].reset_index(drop=True)
print(f"Remaining : {len(remaining):,} terms to process")


# %%
def process_node(row: pd.Series) -> dict:
    """Process a single node row; returns a record dict."""
    ctx = get_edge_context(row["canonical_id"])
    definition = generate_definition(row["name"], row["node_type"], ctx)
    return {
        "canonical_id": row["canonical_id"],
        "term":         row["name"],
        "definition":   definition,
    }


rows = [remaining.iloc[i] for i in range(len(remaining))]
buffer: list[dict] = []
t0 = time.time()

with tqdm(total=len(rows), desc="Generating definitions") as pbar:
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_node, row): row for row in rows}

        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                row = futures[future]
                record = {"canonical_id": row["canonical_id"], "term": row["name"], "definition": ""}

            buffer.append(record)
            pbar.update(1)

            # Checkpoint
            if len(buffer) >= CHECKPOINT_EVERY:
                done_df = pd.concat([done_df, pd.DataFrame(buffer)], ignore_index=True)
                done_df.to_parquet(CHECKPOINT_PATH, index=False)
                buffer.clear()
                elapsed = time.time() - t0
                rate    = pbar.n / max(elapsed, 1)
                eta     = (len(rows) - pbar.n) / max(rate, 1e-6)
                pbar.set_postfix({"rate": f"{rate:.1f}/s", "ETA": f"{eta/60:.0f}min"})

# Flush remainder
if buffer:
    done_df = pd.concat([done_df, pd.DataFrame(buffer)], ignore_index=True)
    done_df.to_parquet(CHECKPOINT_PATH, index=False)

elapsed = time.time() - t0
print(f"\nFinished {len(done_df):,} terms in {elapsed/60:.1f} min")

# %% [markdown]
# ## 6. Review & Finalize Dataset

# %%
df = pd.read_parquet(CHECKPOINT_PATH).copy()

# Drop duplicates (can happen if notebook was rerun)
df = df.drop_duplicates("canonical_id")

empty_mask = df["definition"].str.strip().eq("")
print(f"Total       : {len(df):,}")
print(f"Empty defs  : {empty_mask.sum():,}")

df = df[~empty_mask][["term", "definition"]].reset_index(drop=True)
df["term"]       = df["term"].str.strip()
df["definition"] = df["definition"].str.replace(r"\s+", " ", regex=True).str.strip()

print(f"Final rows  : {len(df):,}")
df.head()

# %%
# Definition length distribution
df["def_len"] = df["definition"].str.len()
print(df["def_len"].describe())
df = df.drop(columns=["def_len"])

# %% [markdown]
# ## 7. Save Parquet

# %%
out_parquet = data_dir / "kg_mesh.parquet"
df.to_parquet(out_parquet, index=False)
print(f"Saved: {out_parquet}")
print(df.dtypes)

# %% [markdown]
# ## 8. Embed with SPECTER2

# %%
device  = select_device()
specter = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query", device=device)

texts   = (df["term"] + "[SEP]" + df["definition"]).tolist()
latent  = torch.zeros((len(texts), 768))
batch_size = 32

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch = texts[i : i + batch_size]
    with torch.no_grad():
        latent[i : i + batch_size] = specter(batch)

print(f"Latent shape: {latent.shape}")

# %% [markdown]
# ## 9. Save Latent

# %%
out_latent = data_dir / "latent_kg_mesh.pt"
torch.save({"latent": latent, "term": df["term"].tolist()}, out_latent)
print(f"Saved: {out_latent}")

# %% [markdown]
# ## 10. Upload to HuggingFace
#
# Upload both files to the `neurovlm/embedded_text` dataset repository:
#
# ```bash
# huggingface-cli upload neurovlm/embedded_text \
#     ~/.cache/neurovlm/kg_mesh.parquet     kg_mesh.parquet     --repo-type dataset
#
# huggingface-cli upload neurovlm/embedded_text \
#     ~/.cache/neurovlm/latent_kg_mesh.pt   latent_kg_mesh.pt   --repo-type dataset
# ```
#
# Once uploaded, this artifact is available as the `pubmed_mesh` retrieval dataset in NeuroVLM.

# %% [markdown]
#
