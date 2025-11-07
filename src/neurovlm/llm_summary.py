"""Summarize retrieved neuroscience content with a local LLM.

This module focuses on constructing prompts (system + user) and calling
the local Ollama runtime to synthesise summaries. Retrieval of relevant
publications or NeuroWiki concepts is delegated to the specialised
`brain2text` and `text_to_brain` modules.
"""

from __future__ import annotations

from typing import List

import ollama
import torch

from neurovlm.brain_input import search_papers_from_brain, search_wiki_from_brain
from neurovlm.text_input import search_papers_from_text, search_wiki_from_text


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
    top_k_similar_papers: int = 5,
    top_n_wiki_articles: int = 2,
    model: str = "qwen2.5:3b-instruct",
    verbose: bool = True,
) -> str:
    """Summarize publications relevant to a query or brain-derived input.

    Parameters
    ----------
    query : str | torch.Tensor
        Natural language query or pre-computed brain embedding.
    top_k_similar_papers : int, optional
        Number of publications (titles + summaries) to include in the context.
    top_n_wiki_articles : int, optional
        Number of NeuroWiki articles to include alongside the publications.
    model : str, optional
        Ollama model name, e.g., 'qwen2.5:3b-instruct' or 'llama3.2:3b'.
    verbose : bool, optional
        If True, prints selected titles and progress messages.

    Returns
    -------
    str
        The LLM-generated summary text.
    """
    for_brain_input = not isinstance(query, str)

    if for_brain_input:
        papers_context, titles = search_papers_from_brain(
            query, top_k=top_k_similar_papers, show_titles=False
        )
        wiki_context = ""
        wiki_titles: List[str] = []
        try:
            wiki_context, wiki_titles = search_wiki_from_brain(
                query, top_k=top_n_wiki_articles, show_titles=False
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if verbose:
                print(f"Warning: failed to retrieve NeuroWiki concepts ({exc})")
    else:
        papers_context, titles = search_papers_from_text(
            query, top_k=top_k_similar_papers, show_titles=False
        )
        wiki_context = ""
        wiki_titles = []
        try:
            wiki_context, wiki_titles = search_wiki_from_text(
                query, top_k=top_n_wiki_articles, show_titles=False
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if verbose:
                print(f"Warning: failed to retrieve NeuroWiki concepts ({exc})")

    if verbose:
        header = (
            f"Top {top_k_similar_papers} publications for query: '{query}'"
            if not for_brain_input
            else f"Top {top_k_similar_papers} publications for brain-derived input"
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
