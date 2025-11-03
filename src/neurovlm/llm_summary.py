"""Search publications and summarize with a local LLM.

This module provides utilities to:
- Embed a natural language query with `Specter`
- Retrieve the most similar publications
- Prompt a local LLM (via `ollama`) to synthesize a focused summary

It can be imported in notebooks, or executed as a small CLI.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from typing import List, Tuple

import ollama
import pandas as pd
import numpy as np
import torch

from neurovlm.data import get_data_dir
from neurovlm.models import Specter


@lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    """Load the publications DataFrame with a robust parquet engine fallback.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including at least `name` and `description`.
    """
    data_dir = get_data_dir()
    parquet_path = data_dir / "publications.parquet"
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as e:  # pragma: no cover - depends on local engines
        print(f"pyarrow failed: {e}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")
    
@lru_cache(maxsize=1)
def _load_neuro_wiki() -> pd.DataFrame:
    """Load the Neurowiki DataFrame with a robust parquet engine fallback.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including at least `title` and `summary`.
    """
    data_dir = get_data_dir()
    parquet_path = data_dir / "neurowiki_with_ids.parquet"
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as e:  # pragma: no cover - depends on local engines
        print(f"pyarrow failed: {e}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_specter() -> Specter:
    """Construct and cache a Specter encoder."""
    return Specter()


@lru_cache(maxsize=1)
def _load_latent_text() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent embeddings and their PubMed IDs."""
    data_dir = get_data_dir()
    latent_payload = torch.load(
        data_dir / "latent_text_aligned.pt",
        weights_only=False,
    )

    latent = latent_payload["latent"]
    if getattr(latent, "is_sparse", False):
        latent = latent.to_dense()
    latent = latent.to(dtype=torch.float32, device="cpu")

    latent_norm = latent.norm(dim=1, keepdim=True).clamp_min(1e-12)
    latent_unit = latent / latent_norm

    latent_pmid = np.asarray(latent_payload["pmid"])
    return latent_unit, latent_pmid

def _load_latent_wiki() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent embeddings and their IDs."""
    data_dir = get_data_dir()
    latent_payload = torch.load(
        data_dir / "latent_specter_wiki.pt",
        weights_only=False,
    )

    latent = latent_payload["latent"]
    if getattr(latent, "is_sparse", False):
        latent = latent.to_dense()
    latent = latent.to(dtype=torch.float32, device="cpu")

    latent_norm = latent.norm(dim=1, keepdim=True).clamp_min(1e-12)
    latent_unit = latent / latent_norm

    latent_id = np.asarray(latent_payload["id"])
    return latent_unit, latent_id

@lru_cache(maxsize=1)
def _load_autoencoder() -> torch.nn.Module:
    """Load and return the text encoder model."""
    data_dir = get_data_dir()
    encoder = torch.load(data_dir / "autoencoder_sparse.pt", weights_only=False).to("cpu")
    return encoder

def _proj_head_image_infonce() -> torch.nn.Module:
    """Load and return the text projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_image_infonce.pt", weights_only=False).to("cpu")
    return proj_head

def _proj_head_mse_adhoc() -> torch.nn.Module:
    """Load and return the text projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_mse_sparse_adhoc.pt", weights_only=False).to("cpu")
    return proj_head

def _proj_head_text_infonce() -> torch.nn.Module:
    """Load and return the text projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_text_infonce.pt", weights_only=False).to("cpu")
    return proj_head

def search_papers(
    query: str | torch.Tensor,
    top_k: int = 5,
    show_titles: bool = False,
) -> Tuple[str, List[str]]:
    """Return a context block of top papers and their titles.

    Parameters
    ----------
    query : str | torch.Tensor
        Natural language query or pre-computed brain embedding.
    top_k : int, optional
        Number of publications to retrieve.
    show_titles : bool, optional
        When True, print the ranked titles to stdout.

    Returns
    -------
    tuple[str, list[str]]
        The formatted context block and the ordered list of titles.
    """
    df = _load_dataframe()
    latent_text, latent_pmids = _load_latent_text()

    if isinstance(query, str):
        specter = _load_specter()
        proj_head = _proj_head_mse_adhoc()
        # Encode and normalize query
        encoded_query = specter(query)[0].detach().to("cpu")
        encoded_query_norm = encoded_query / encoded_query.norm()
        proj_query = proj_head(encoded_query_norm.unsqueeze(0)).squeeze(0)
        proj_query = proj_query / proj_query.norm()

        # should i project latent text aligned too?

        # Cosine similarity and ranking
        cos_sim = latent_text @ proj_query
    else:
        # Assume query is already a brain-derived embedding
        proj_head_img = _proj_head_image_infonce()
        proj_head_text = _proj_head_text_infonce()

        encoded_norm = query / query.norm()
        img_embed = proj_head_img(encoded_norm.unsqueeze(0)).squeeze(0)
        img_embed = img_embed / img_embed.norm()

        # text_embed = proj_head_text(latent_text.unsqueeze(0)).squeeze(0)
        # text_embed = text_embed / text_embed.norm()

        # print(f"text_embed shape: {text_embed.shape}, img_embed shape: {img_embed.shape}")
        cos_sim = latent_text @ img_embed
        

    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()

    pmids_top = [latent_pmids[i] for i in inds_top]

    rows: pd.DataFrame
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
        if selected_rows:
            rows = pd.DataFrame(selected_rows)
        else:
            rows = df.iloc[inds_top]
    else:
        rows = df.iloc[inds_top]
    pieces = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        title = (
            str(row["name"]) if "name" in rows.columns else str(row.get("title", "Untitled"))
        )
        desc = str(row.get("description", "")).replace("\n", " ")
        desc = re.sub(r"\s+", " ", desc).strip()
        pieces.append(f"[{idx}] {title}\n{desc}\n")
    papers_context = "\n".join(pieces)

    title_col = "name" if "name" in rows.columns else ("title" if "title" in rows.columns else None)
    titles: List[str]
    if title_col is not None:
        titles = rows[title_col].astype(str).tolist()
    else:
        titles = ["Untitled"] * len(rows)

    if show_titles:
        print("Top matches:")
        for idx, title in enumerate(titles, start=1):
            print(f"{idx}. {title}")

    return papers_context, titles


def search_wiki(
    query: str | torch.Tensor,
    top_k: int = 2,
    show_titles: bool = False,
) -> Tuple[str, List[str]]:
    """Return a context block of top NeuroWiki entries and their titles.

    Parameters
    ----------
    query : str | torch.Tensor
        Natural language query or pre-computed brain embedding.
    top_k : int, optional
        Number of entries to retrieve.
    show_titles : bool, optional
        When True, print the ranked titles to stdout.

    Returns
    -------
    tuple[str, list[str]]
        The formatted context block and the ordered list of titles.
    """
    df = _load_neuro_wiki()
    latent_wiki, latent_ids = _load_latent_wiki()

    if isinstance(query, str):
        specter = _load_specter()
        proj_head = _proj_head_mse_adhoc()

        encoded_query = specter(query)[0].detach().to("cpu")
        encoded_query_norm = encoded_query / encoded_query.norm()
        proj_query = proj_head(encoded_query_norm)
        proj_query = proj_query / proj_query.norm()

        proj_wiki = proj_head(latent_wiki)
        proj_wiki = proj_wiki / proj_wiki.norm()

        cos_sim = proj_wiki @ proj_query
    else:
        proj_head_img = _proj_head_image_infonce()
        proj_head_text = _proj_head_text_infonce()

        encoded_norm = query / query.norm()
        img_embed = proj_head_img(encoded_norm)
        img_embed = img_embed / img_embed.norm()
        
        wiki_embed = proj_head_text(latent_wiki)
        wiki_embed = wiki_embed / wiki_embed.norm()
        cos_sim = wiki_embed @ img_embed

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

    wiki_context = "\n".join(pieces)

    if show_titles:
        print("Top matches:")
        for idx, title in enumerate(titles, start=1):
            print(f"{idx}. {title}")

    return wiki_context, titles


def system_prompt(for_brain_input: bool = False) -> str:
    """Return instructions for the LLM.

    The prompt adapts to whether we have a textual query or a brain-derived input.
    """
    if for_brain_input:
        return """
        You are a helpful neuroscience research assistant.
        You will receive a set of publications associated with an input brain representation and a set of neuroscience concepts with its meaning. Your task is to summarize key findings and insights from these publications, focusing on what they reveal about the brain and if the neuroscience concept relates to the paper talk about it.

        Your response must:
        - Start with a brief overview (2-4 sentences) summarizing the main themes or takeaways across the publications.
        - Ground every statement in the provided publications and explain how the evidence informs interpretation of the brain input. Do not add outside knowledge or speculation.
        - Identify how each publication contributes to understanding the brain-derived signal. If evidence is indirect, clarify the most relevant methods, populations, or findings that help characterize the input.
        - Synthesize across studies, noting key areas of agreement or convergence, and conflicting or divergent findings with balanced context (e.g., methods, populations, analyses).
        - Use paragraphs or bullet points depending on the structure that best communicates the interpretation.
        - Maintain an objective, precise, scholarly tone suitable for neuroscience research contexts.
        """

    return """
    You are a helpful neuroscience research assistant.
    You will receive a set of publications, set of neuroscience concepts with its meaning and a user query. Your task is to summarize key findings and insights from these publications, focusing on how they relate to the query, use the neuroscience concepts as added context and information.

    Your response must:
    - Start with a brief overview (2-4 sentences) summarizing the main themes or takeaways across the publications.
    - Be entirely based on the information in the publications and how it directly ties to the user's query. Do not add outside knowledge or speculation.
    - Identify how each publication relates to the query. If the publications directly answer the query, state the answer clearly. If they do not answer it fully, highlight relevant points, evidence, or gaps that inform the query.
    - Synthesize across studies, noting key areas of agreement or convergence, and conflicting or divergent findings with balanced context (e.g., methods, populations, analyses).
    - Use paragraphs or bullet points depending on the query: bullet points for lists and comparisons; paragraphs for integrative summaries.
    - Maintain an objective, precise, scholarly tone suitable for neuroscience research contexts.
    """


def build_user_prompt(
    query: str | torch.Tensor,
    papers_context: str,
    wiki_context: str | None = None,
) -> str:
    """Compose the user-facing prompt given a query, publications, and NeuroWiki context."""
    if isinstance(query, str):
        prompt = (
            f"Here are publications related to the query \"{query}\":\n{papers_context}\n"
        )
    else:
        prompt = f"Here are the publications related to the input brain:\n{papers_context}\n"

    if wiki_context:
        prompt += f"\nHere are neuroscience concepts from the NeuroWiki:\n{wiki_context}\n"

    return prompt


def generate_response(
    query: str | torch.Tensor,
    top_k: int = 5,
    model: str = "qwen2.5:3b-instruct",
    verbose: bool = True,
) -> str:
    """Summarize publications relevant to a query or brain-derived input.

    Parameters
    ----------
    query : str, optional
        Natural language question or topic. When empty, the summary assumes
        the publications are tied to a brain-derived representation.
    top_k : int, optional
        Number of publications to retrieve for context.
    model : str, optional
        Ollama model name, e.g., 'qwen2.5:3b-instruct' or 'llama3.2:3b'.
    verbose : bool, optional
        If True, prints selected titles and progress messages.

    Returns
    -------
    str
        The LLM-generated summary text.
    """
    papers_context, titles = search_papers(query, top_k=top_k)
    wiki_context = ""
    wiki_titles: List[str] = []
    try:
        wiki_context, wiki_titles = search_wiki(query, top_k=top_k)
    except Exception as exc:  # pragma: no cover - defensive fallback
        if verbose:
            print(f"Warning: failed to retrieve NeuroWiki concepts ({exc})")

    # Decide prompt flavor early so it is always defined
    for_brain_input = not isinstance(query, str)

    if verbose:
        header = (
            f"Top {top_k} publications for query: '{query}'"
            if isinstance(query, str)
            else f"Top {top_k} publications for brain-derived input"
        )
        print(header)
        for i, title in enumerate(titles, start=1):
            print(f"[{i}] {title}")
        if wiki_titles:
            print("Top NeuroWiki concepts:")
            for i, title in enumerate(wiki_titles, start=1):
                print(f"[{i}] {title}")
        print("LLM writing summary...")

    response = ollama.chat(
        messages=[
            {"role": "system", "content": system_prompt(for_brain_input=for_brain_input)},
            {
                "role": "user",
                "content": build_user_prompt(query, papers_context, wiki_context or None),
            },
        ],
        model=model,
    )

    if verbose:
        print("LLM finished.")

    output_text = response["message"]["content"]
    if verbose:
        print(output_text)
    return output_text

