"""Search publications and summarize with a local LLM.

This module provides utilities to:
- Embed a natural language query with `Specter`
- Retrieve the most similar publications
- Prompt a local LLM (via `ollama`) to synthesize a focused summary

It can be imported in notebooks, or executed as a small CLI.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Tuple

import ollama
import pandas as pd
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
def _load_specter() -> Specter:
    """Construct and cache a Specter encoder."""
    return Specter()


@lru_cache(maxsize=1)
def _load_latent_text() -> torch.Tensor:
    """Load and return unit-normalized latent text embeddings.

    Returns
    -------
    torch.Tensor
        Tensor of shape [num_papers, dim] with unit-norm rows.
    """
    data_dir = get_data_dir()
    latent = torch.load(data_dir / "latent_text.pt", weights_only=True).to("cpu")
    return latent / latent.norm(dim=1, keepdim=True)


def search_papers(query: str, top_k: int = 5) -> Tuple[str, List[str]]:
    """Retrieve top-k related publications and a formatted context string.

    Parameters
    ----------
    query : str
        Natural language question or topic.
    top_k : int, optional
        Number of publications to retrieve, by cosine similarity.

    Returns
    -------
    papers_context : str
        Concise, numbered list with titles and one-line descriptions.
    titles : list[str]
        The corresponding top-k publication titles.
    """
    df = _load_dataframe()
    specter = _load_specter()
    latent_text = _load_latent_text()

    # Encode and normalize query
    encoded_text = specter(query)[0].detach()
    encoded_text_norm = encoded_text / encoded_text.norm()

    # Cosine similarity and ranking
    cos_sim = latent_text @ encoded_text_norm
    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()

    # Aggregate publications to pass to LLM
    rows = df.iloc[inds_top]
    pieces = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        title = str(row["name"]) if "name" in row else str(row.get("title", "Untitled"))
        desc = str(row.get("description", "")).replace("\n", " ")
        desc = re.sub(r"\s+", " ", desc).strip()
        pieces.append(f"[{idx}] {title}\n{desc}\n")
    papers_context = "\n".join(pieces)

    titles = rows["name"].astype(str).tolist() if "name" in rows else rows.get("title", []).astype(str).tolist()
    return papers_context, titles


def system_prompt() -> str:
    """Return instructions for the LLM.

    The prompt enforces grounded, scholarly synthesis focused on the user query.
    """
    return """
    You are a helpful neuroscience research assistant.
    You will receive a set of publications and a user query. Your task is to summarize key findings and insights from these publications, focusing on how they relate to the query.

    Your response must:
    - Start with a brief overview (2-4 sentences) summarizing the main themes or takeaways across the publications.
    - Be entirely based on the information in the publications and how it directly ties to the user's query. Do not add outside knowledge or speculation.
    - Identify how each publication relates to the query. If the publications directly answer the query, state the answer clearly. If they do not answer it fully, highlight relevant points, evidence, or gaps that inform the query.
    - Synthesize across studies, noting key areas of agreement or convergence, and conflicting or divergent findings with balanced context (e.g., methods, populations, analyses).
    - Use paragraphs or bullet points depending on the query: bullet points for lists and comparisons; paragraphs for integrative summaries.
    - Maintain an objective, precise, scholarly tone suitable for neuroscience research contexts.
    """


def build_user_prompt(query: str, papers_context: str) -> str:
    """Compose the user-facing prompt given a query and context block."""
    return f"""
    Here are publications related to the query "{query}":
    {papers_context}
    """


def summarize_papers(query: str, top_k: int = 5, model: str = "qwen2.5:3b-instruct", verbose: bool = True) -> str:
    """Summarize publications relevant to a query using a local LLM via Ollama.

    Parameters
    ----------
    query : str
        Natural language question or topic.
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

    if verbose:
        print(f"Top {top_k} publications for query: '{query}'")
        for i, title in enumerate(titles, start=1):
            print(f"[{i}] {title}")
        print("LLM writing summary...")

    response = ollama.chat(
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": build_user_prompt(query, papers_context)},
        ],
        model=model,
    )

    if verbose:
        print("LLM finished.")

    output_text = response["message"]["content"]
    if verbose:
        print(output_text)
    return output_text


# Command to run CLI in the termina: python docs/llm_summary.py "your query" --top-k 5 --model qwen2.5:3b-instruct

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize publications for a query using a local LLM.")
    parser.add_argument("query", type=str, help="Question or topic to search and summarize")
    parser.add_argument("--top-k", type=int, default=5, help="Number of publications to include")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:3b-instruct",
        help="Ollama model to use (e.g., qwen2.5:3b-instruct, llama3.2:3b)",
    )
    args = parser.parse_args()

    summarize_papers(args.query, top_k=args.top_k, model=args.model, verbose=True)
