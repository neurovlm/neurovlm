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

def _load_neurowiki() -> pd.DataFrame:
    """Load the Neurowiki DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including at least `title` and `summary`.
    """
    data_dir = get_data_dir()
    neurowiki_path = data_dir / "neurowiki.parquet"
    return pd.read_parquet(neurowiki_path, engine="fastparquet")


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
    # Unit-normalize rows for cosine similarity via dot product
    return latent / latent.norm(dim=1, keepdim=True)

@lru_cache(maxsize=1)
def _load_autoencoder() -> torch.nn.Module:
    """Load and return the text encoder model."""
    data_dir = get_data_dir()
    encoder = torch.load(data_dir / "autoencoder.pt", weights_only=True).to("cpu")
    return encoder


def search_papers(query: str | torch.Tensor, top_k: int = 5) -> Tuple[str, List[str]]:
    """Return a context block of top papers and their titles.

    Accepts either a natural language query or a brain-derived vector.
    """
    df = _load_dataframe()
    latent_text = _load_latent_text()  # [num_papers, dim]

    if isinstance(query, str):
        specter = _load_specter()
        # Encode and normalize query
        encoded_text = specter(query)[0].detach().to("cpu")
        encoded_text_norm = encoded_text / encoded_text.norm()

        # Cosine similarity and ranking
        cos_sim = latent_text @ encoded_text_norm  # [num_papers]
    else:
        # Encode brain vector to latent space and normalize
        encoder = _load_autoencoder()
        brain_vec = query.to("cpu").unsqueeze(0)
        with torch.no_grad():
            latent = encoder.encoder(brain_vec).squeeze(0)
        encoded_text_norm = latent / latent.norm()
        cos_sim = latent_text @ encoded_text_norm  # [num_papers]

    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()

    # Aggregate publications to pass to LLM
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

    return papers_context, titles


def system_prompt(for_brain_input: bool = False) -> str:
    """Return instructions for the LLM.

    The prompt adapts to whether we have a textual query or a brain-derived input.
    """
    if for_brain_input:
        return """
        You are a helpful neuroscience research assistant.
        You will receive a set of publications associated with an input brain representation rather than a textual query. Your task is to summarize key findings and insights from these publications, focusing on what they reveal about the brain.

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
    You will receive a set of publications and a user query. Your task is to summarize key findings and insights from these publications, focusing on how they relate to the query.

    Your response must:
    - Start with a brief overview (2-4 sentences) summarizing the main themes or takeaways across the publications.
    - Be entirely based on the information in the publications and how it directly ties to the user's query. Do not add outside knowledge or speculation.
    - Identify how each publication relates to the query. If the publications directly answer the query, state the answer clearly. If they do not answer it fully, highlight relevant points, evidence, or gaps that inform the query.
    - Synthesize across studies, noting key areas of agreement or convergence, and conflicting or divergent findings with balanced context (e.g., methods, populations, analyses).
    - Use paragraphs or bullet points depending on the query: bullet points for lists and comparisons; paragraphs for integrative summaries.
    - Maintain an objective, precise, scholarly tone suitable for neuroscience research contexts.
    """


def build_user_prompt(query: str | torch.Tensor, papers_context: str) -> str:
    """Compose the user-facing prompt given a query and context block."""
    if isinstance(query, str):
        return (
            f"Here are publications related to the query \"{query}\":\n{papers_context}\n"
        )
    return f"Here are the publications related to the input brain:\n{papers_context}\n"


def summarize_papers(
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
        print("LLM writing summary...")

    response = ollama.chat(
        messages=[
            {"role": "system", "content": system_prompt(for_brain_input=for_brain_input)},
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


def _main() -> None:
    """Simple CLI for text or brain-vector queries.

    Examples
    --------
    Text query:
        python -m neurovlm.llm_summary "neural correlates of working memory" --top-k 5

    Brain vector from a .pt file containing a 1D tensor:
        python -m neurovlm.llm_summary --brain-vector path/to/vector.pt --top-k 10
    """
    parser = argparse.ArgumentParser(description="Search and summarize publications with a local LLM")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("query", nargs="?", help="Natural language query to search")
    grp.add_argument("--brain-vector", dest="brain_vector", help="Path to a .pt file with a 1D torch tensor")
    parser.add_argument("--top-k", type=int, default=5, help="Number of publications to include in context")
    parser.add_argument(
        "--model",
        default="qwen2.5:3b-instruct",
        help="Ollama model name (e.g., qwen2.5:3b-instruct, llama3.2:3b)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress printing")

    args = parser.parse_args()

    if args.query is not None:
        query: str | torch.Tensor = args.query
    else:
        vec = torch.load(args.brain_vector, map_location="cpu")
        if isinstance(vec, dict) and "tensor" in vec:
            vec = vec["tensor"]
        if vec.dim() == 2 and vec.size(0) == 1:
            vec = vec.squeeze(0)
        if vec.dim() != 1:
            raise ValueError("Brain vector must be a 1D tensor or [1, D] tensor")
        query = vec

    try:
        summarize_papers(query, top_k=args.top_k, model=args.model, verbose=not args.quiet)
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure Ollama is running and the model is pulled: `ollama run qwen2.5:3b-instruct`.")


if __name__ == "__main__":
    _main()
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
