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

# %% [markdown]
# # Prepare Network Gold Terms for LLM Corpus
#
# This notebook fixes network gold-term fairness for B2T term-ranking evaluation. It finds network gold terms from `network_test_set_labels.csv` that are missing from `llm_neuro_terms + pubmed_mesh + cogatlas`, appends them to `llm_neuro_terms`, and regenerates `latent_llm_neuro_terms.pt` for upload to Hugging Face.

# %%
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from IPython.display import display

from neurovlm.data import load_dataset
from neurovlm.models import Specter

# %%
OUTPUT_DIR = Path("artifacts/network_gold_terms_llm_update")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedding device:", DEVICE)

UPDATED_PARQUET = OUTPUT_DIR / "llm_neuro_terms.parquet"
UPDATED_CSV = OUTPUT_DIR / "llm_neuro_terms.csv"
ADDED_TERMS_CSV = OUTPUT_DIR / "network_gold_terms_added_to_llm_neuro_terms.csv"
MISSING_TERMS_CSV = OUTPUT_DIR / "network_gold_terms_missing_from_retrieval_corpora.csv"
UPDATED_LATENT = OUTPUT_DIR / "latent_llm_neuro_terms.pt"


# %%
def normalize_term_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.split("/")[0]
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def split_gold_terms(value) -> list[str]:
    if pd.isna(value):
        return []
    terms = []
    for chunk in re.split(r";|\n|\|", str(value)):
        term = chunk.strip()
        if term:
            terms.append(term)
    return terms


def load_network_labels():
    return load_dataset("network_test_set_labels"), "huggingface:neurovlm/embedded_text/network_test_set_labels.csv"


def candidate_terms_for_dataset(dataset_name: str) -> set[str]:
    df = load_dataset(dataset_name)
    for col in ["term", "title", "name", "label"]:
        if col in df.columns:
            return {normalize_term_text(x) for x in df[col].dropna().astype(str)}
    return set()


# %%
network_labels_df, network_source = load_network_labels()
network_labels_df = network_labels_df[network_labels_df["network_key"] != "unknown"].copy()
print(f"Loaded {len(network_labels_df):,} network label rows from {network_source}")

gold_rows = []
for _, row in network_labels_df.iterrows():
    for col in ["mapped_terms", "region_terms", "cognitive_terms"]:
        if col not in network_labels_df.columns:
            continue
        for term in split_gold_terms(row.get(col)):
            gold_rows.append({
                "term": term,
                "normalized_term": normalize_term_text(term),
                "source_column": col,
                "network_key": row.get("network_key", ""),
                "network_name": row.get("network_name", ""),
                "raw_network_label": row.get("raw_network_label", ""),
                "short_definition": row.get("short_definition", ""),
                "long_definition": row.get("long_definition", ""),
            })

gold_terms_df = pd.DataFrame(gold_rows).drop_duplicates()
gold_terms_df = gold_terms_df[gold_terms_df["normalized_term"] != ""].reset_index(drop=True)
print(f"Unique network gold term rows: {len(gold_terms_df):,}")
display(gold_terms_df.head())

# %%
candidate_terms = {}
for dataset_name in ["llm_neuro_terms", "pubmed_mesh", "cogatlas"]:
    candidate_terms[dataset_name] = candidate_terms_for_dataset(dataset_name)
    print(dataset_name, len(candidate_terms[dataset_name]))

all_candidates = set().union(*candidate_terms.values())
missing_terms_df = gold_terms_df[~gold_terms_df["normalized_term"].isin(all_candidates)].copy()
missing_terms_df = missing_terms_df.drop_duplicates(["normalized_term", "term"]).reset_index(drop=True)
missing_terms_df.to_csv(MISSING_TERMS_CSV, index=False)
print(f"Missing network gold terms: {len(missing_terms_df):,}")
display(missing_terms_df.head(50))


# %% [markdown]
# ## Build Definitions
#
# This cell creates deterministic definitions from the network rows. Review the output before embedding. If you want to hand-edit definitions, edit `added_terms_df` before running the embedding cell.

# %%
def build_definition(group: pd.DataFrame) -> str:
    term = group["term"].iloc[0]
    network_names = ", ".join(sorted(set(group["network_name"].dropna().astype(str))))
    source_cols = ", ".join(sorted(set(group["source_column"].dropna().astype(str))))
    long_defs = []
    for value in group["long_definition"].dropna().astype(str):
        clean = " ".join(value.split())
        if clean and clean not in long_defs:
            long_defs.append(clean)
    context = " ".join(long_defs[:3])
    return (
        f"{term} is a neuroscience term used as network-label ground truth in NeuroVLM evaluation. "
        f"It is associated with the {network_names} network label(s) and appears in the {source_cols} field(s). "
        f"Network context: {context}"
    ).strip()

added_rows = []
for norm, group in missing_terms_df.groupby("normalized_term", sort=True):
    display_term = group.sort_values("term")["term"].iloc[0]
    added_rows.append({
        "term": display_term,
        "definition": build_definition(group),
        "source": "network_test_set_labels",
        "normalized_term": norm,
        "network_keys": "; ".join(sorted(set(group["network_key"].astype(str)))),
    })

added_terms_df = pd.DataFrame(
    added_rows,
    columns=["term", "definition", "source", "normalized_term", "network_keys"],
)
added_terms_df.to_csv(ADDED_TERMS_CSV, index=False)
print(f"Terms to add to llm_neuro_terms: {len(added_terms_df):,}")
display(added_terms_df.head(50))

# %%
existing_llm_df = load_dataset("llm_neuro_terms").copy()
if "definition" not in existing_llm_df.columns:
    if "description" in existing_llm_df.columns:
        existing_llm_df = existing_llm_df.rename(columns={"description": "definition"})
    else:
        existing_llm_df["definition"] = ""
if "term" not in existing_llm_df.columns:
    raise ValueError("llm_neuro_terms must contain a term column")

existing_llm_df["normalized_term"] = existing_llm_df["term"].map(normalize_term_text)
new_only_df = added_terms_df[~added_terms_df["normalized_term"].isin(set(existing_llm_df["normalized_term"]))].copy()
updated_llm_df = pd.concat([existing_llm_df, new_only_df], ignore_index=True)
updated_llm_df = updated_llm_df.drop_duplicates("normalized_term", keep="first").reset_index(drop=True)

updated_llm_df.to_csv(UPDATED_CSV, index=False)
updated_llm_df.to_parquet(UPDATED_PARQUET, index=False)
print(f"Existing LLM terms: {len(existing_llm_df):,}")
print(f"New terms actually appended: {len(new_only_df):,}")
print(f"Updated LLM terms: {len(updated_llm_df):,}")
print("Wrote:", UPDATED_PARQUET)
print("Wrote:", UPDATED_CSV)


# %% [markdown]
# ## Regenerate Embeddings
#
# Run this cell after reviewing `added_terms_df`. It writes `latent_llm_neuro_terms.pt`, aligned to the updated `llm_neuro_terms.parquet`.

# %%
def batched_indices(n_rows: int, batch_size: int):
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)

specter = Specter(
    "allenai/specter2_aug2023refresh",
    adapter="adhoc_query",
    device=DEVICE,
)
encode_df = updated_llm_df[["term", "definition"]].rename(columns={"term": "title", "definition": "summary"}).copy()
embeddings = []
for start, end in batched_indices(len(encode_df), BATCH_SIZE):
    with torch.no_grad():
        emb = specter(encode_df.iloc[start:end]).detach().cpu()
        emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    embeddings.append(emb)
    print(f"Embedded {end:,}/{len(encode_df):,}", end="\r", flush=True)
print()
latent = torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 768))
torch.save({"latent": latent, "term": updated_llm_df["term"].astype(str).to_numpy()}, UPDATED_LATENT)
print("Wrote:", UPDATED_LATENT)
print("Latent shape:", tuple(latent.shape))

# %% [markdown]
# ## Upload These Files
#
# Upload the following files to the appropriate Hugging Face repo when ready:
#
# - `artifacts/network_gold_terms_llm_update/llm_neuro_terms.parquet`
# - `artifacts/network_gold_terms_llm_update/latent_llm_neuro_terms.pt`
# - `artifacts/network_gold_terms_llm_update/network_gold_terms_added_to_llm_neuro_terms.csv` for audit/versioning
# - `artifacts/network_gold_terms_llm_update/network_gold_terms_missing_from_retrieval_corpora.csv` for audit/versioning
