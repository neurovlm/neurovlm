"""Text-driven retrieval helpers for NeuroVLM.

These utilities accept natural language queries and return the most
relevant publications or NeuroWiki entries along with formatted context
blocks suitable for prompting an LLM.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd
import torch

from neurovlm.retrieval_resources import (
    _load_dataframe,
    _load_latent_text,
    _load_latent_wiki,
    _load_neuro_wiki,
    _load_specter,
    _proj_head_mse_adhoc,
)

__all__ = ["search_papers_from_text", "search_wiki_from_text", "generate_llm_response_from_text"]


def search_papers_from_text(
    query: str,
    top_k: int = 5,
    show_titles: bool = False,
) -> Tuple[str, List[str]]:
    """Return a context block of top papers and their titles for a text query."""
    if not isinstance(query, str):
        raise TypeError("query must be a string for text-based retrieval")

    df = _load_dataframe()
    latent_text, latent_pmids = _load_latent_text()

    specter = _load_specter()
    proj_head = _proj_head_mse_adhoc()

    encoded_query = specter(query)[0].detach().to("cpu")
    proj_query = proj_head(encoded_query)
    proj_query = proj_query / proj_query.norm()

    proj_text = proj_head(latent_text)
    proj_text = proj_text / proj_text.norm(dim=1)[:, None]
    cos_sim = proj_text @ proj_query

    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()
    pmids_top = [latent_pmids[i] for i in inds_top]

    if "pmid" in df.columns:
        pmid_lookup = df.drop_duplicates("pmid").set_index("pmid", drop=False)
        selected_rows = []
        for pmid in pmids_top:
            try:
                row = pmid_lookup.loc[pmid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                selected_rows.append(row)
            except KeyError:
                continue
        rows = pd.DataFrame(selected_rows) if selected_rows else df.iloc[inds_top]
    else:
        rows = df.iloc[inds_top]

    pieces = []
    title_col = "name" if "name" in rows.columns else ("title" if "title" in rows.columns else None)
    titles: List[str]
    if title_col is not None:
        titles = rows[title_col].astype(str).tolist()
    else:
        titles = ["Untitled"] * len(rows)

    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        title = (
            str(row["name"]) if "name" in rows.columns else str(row.get("title", "Untitled"))
        )
        desc = str(row.get("description", "")).replace("\n", " ")
        desc = re.sub(r"\s+", " ", desc).strip()
        pieces.append(f"[{idx}] {title}\n{desc}\n")

    if show_titles:
        print("Top matches:")
        for idx, title in enumerate(titles, start=1):
            print(f"{idx}. {title}")

    papers_context = "\n".join(pieces)
    return papers_context, titles


def search_wiki_from_text(
    query: str,
    top_k: int = 2,
    show_titles: bool = False,
) -> Tuple[str, List[str]]:
    """Return a context block of top NeuroWiki entries for a text query."""
    if not isinstance(query, str):
        raise TypeError("query must be a string for text-based retrieval")

    df = _load_neuro_wiki()
    latent_wiki, latent_ids = _load_latent_wiki()
    if not (df['id'] == latent_ids).all():
        raise ValueError("Mismatch between DataFrame 'id' column and latent_ids: ensure they are aligned.")
    specter = _load_specter()
    proj_head = _proj_head_mse_adhoc()

    encoded_query = specter(query)[0].detach().to("cpu")
    encoded_query = encoded_query / encoded_query.norm()
    proj_query = proj_head(encoded_query)
    proj_query = proj_query / proj_query.norm()

    wiki_embed = latent_wiki / latent_wiki.norm(dim=1)[:, None]
    proj_wiki = proj_head(wiki_embed)
    proj_wiki = proj_wiki / proj_wiki.norm(dim=1)[:, None]
    cos_sim = proj_wiki @ proj_query


    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()
    ids_top = [latent_ids[i] for i in inds_top]
    missing_columns = [col for col in ("id", "title", "summary") if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"NeuroWiki DataFrame is missing required columns: {', '.join(missing_columns)}"
        )

    id_lookup = df.drop_duplicates("id").set_index("id", drop=False)
    selected_rows = []
    for entry_id in ids_top:
        try:
            row = id_lookup.loc[entry_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            selected_rows.append(row)
        except KeyError:
            continue

    rows = pd.DataFrame(selected_rows) if selected_rows else df.iloc[inds_top]

    pieces = []
    titles: List[str] = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        title = str(row["title"]) if pd.notna(row["title"]) else "Untitled"
        titles.append(title)
        summary = str(row["summary"]) if pd.notna(row["summary"]) else ""
        summary = re.sub(r"\s+", " ", summary).strip()
        pieces.append(f"[{idx}] {title}\n{summary}\n")

    if show_titles:
        print("Top matches:")
        for idx, title in enumerate(titles, start=1):
            print(f"{idx}. {title}")

    wiki_context = "\n".join(pieces)
    return wiki_context, titles



def generate_llm_response_from_text(query: str):
    """
    Generate an LLM response for a given natural language query.

    Parameters
    ----------
    query : str
        A natural language query to be encoded.

    Returns
    -------
    str
        The generated LLM response.
    """
    from neurovlm.llm_summary import generate_response

    return generate_response(query, top_k_similar_papers=5)

