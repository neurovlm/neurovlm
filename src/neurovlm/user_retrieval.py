"""User-data retrieval utilities for NeuroVLM.

Functions for ranking a user-supplied text corpus against a neuroimage or a
text query using the NeuroVLM contrastive model, plus the NIfTI resampling
helpers previously housed in ``brain_input.py``.

Typical workflow
----------------
1. Prepare your corpus as a ``pandas.DataFrame`` with ``name`` and
   ``description`` columns.
2. Pre-embed every row with SPECTER (768-d) and stack the vectors into a
   ``torch.Tensor`` of shape ``(N, 768)``.
3. Call :func:`search_text_corpus_given_neuroimage` or
   :func:`search_text_corpus_given_text` to get a ranked result table.
4. Optionally call :func:`generate_llm_response` to get a natural-language
   summary of the top hits.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.image import resample_img
from nilearn.maskers import NiftiMasker
from nilearn import maskers
from tqdm import tqdm

from neurovlm.retrieval_resources import (
    _load_autoencoder,
    _load_masker,
    _load_specter,
    _proj_head_image_infonce,
    _proj_head_text_infonce,
)

__all__ = [
    "search_text_corpus_given_neuroimage",
    "search_text_corpus_given_text",
    "_load_mask_bundle",
    "_resample_to_mask",
    "resample_nifti",
    "resample_array_nifti",
    "resample_networks_to_mask",
    "generate_llm_response",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_corpus(
    corpus_df: pd.DataFrame,
    corpus_embeddings: torch.Tensor,
) -> None:
    """Assert corpus format requirements."""
    assert isinstance(corpus_df, pd.DataFrame), (
        "corpus_df must be a pandas DataFrame."
    )
    missing = [c for c in ("name", "description") if c not in corpus_df.columns]
    assert not missing, (
        f"corpus_df is missing required column(s): {', '.join(missing)}. "
        "Ensure your DataFrame has both a 'name' and a 'description' column."
    )
    assert isinstance(corpus_embeddings, torch.Tensor), (
        "corpus_embeddings must be a torch.Tensor of shape (N, 768)."
    )
    assert corpus_embeddings.ndim == 2 and corpus_embeddings.shape[1] == 768, (
        f"corpus_embeddings must have shape (N, 768), got {tuple(corpus_embeddings.shape)}. "
        "Each row should be a 768-d SPECTER embedding for the corresponding corpus row."
    )
    assert len(corpus_df) == corpus_embeddings.shape[0], (
        f"corpus_df has {len(corpus_df)} rows but corpus_embeddings has "
        f"{corpus_embeddings.shape[0]} rows — they must match."
    )


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# NIfTI / mask utilities  (moved from brain_input.py)
# ---------------------------------------------------------------------------

def _load_mask_bundle(
    data_dir: Optional[Path | str] = None,
) -> tuple[dict[str, Any], nib.Nifti1Image, NiftiMasker]:
    """Return mask arrays, mask image, and fitted NiftiMasker.

    The ``data_dir`` parameter is kept for backward compatibility; the masker
    is always loaded from HuggingFace via :func:`retrieval_resources._load_masker`.
    """
    masker = _load_masker()
    mask_img = masker.mask_img_
    mask_arrays = {
        "mask": mask_img.get_fdata().astype(bool),
        "affine": mask_img.affine,
    }
    return mask_arrays, mask_img, masker


def _resample_to_mask(
    img: nib.Nifti1Image,
    mask_arrays: Mapping[str, Any],
) -> nib.Nifti1Image:
    """Resample *img* to the project mask affine.

    Binary images are resampled with nearest-neighbour interpolation.
    Continuous images are resampled, negative values zeroed, and then
    thresholded + binarised at the 95th percentile.
    """
    img_arr = img.get_fdata()
    unique_values = np.unique(img_arr)
    is_binary = len(unique_values) == 2 and set(np.round(unique_values).tolist()) <= {0, 1}

    if is_binary:
        return resample_img(
            img,
            target_affine=mask_arrays["affine"],
            interpolation="nearest",
        )

    img_resampled = resample_img(img, target_affine=mask_arrays["affine"])
    arr = img_resampled.get_fdata()
    arr[arr < 0] = 0.0
    thresh = np.percentile(arr.flatten(), 95)
    arr[arr < thresh] = 0.0
    arr[arr >= thresh] = 1.0
    return nib.Nifti1Image(arr, affine=mask_arrays["affine"])


def resample_nifti(
    nifti_img: nib.Nifti1Image,
    data_dir: Optional[Path | str] = None,
) -> torch.Tensor:
    """Resample and flatten a NIfTI image using the project mask definition.

    Parameters
    ----------
    nifti_img:
        Input NIfTI image.
    data_dir:
        Unused; kept for backward compatibility.

    Returns
    -------
    torch.Tensor
        Binarised flat brain vector of shape ``(28542,)``.
    """
    mask_arrays, _, masker = _load_mask_bundle(data_dir=data_dir)
    img_resampled = _resample_to_mask(nifti_img, mask_arrays)
    flattened = masker.transform(img_resampled)
    flattened[flattened > 0] = 1
    return torch.tensor(flattened).squeeze(0)


def resample_array_nifti(
    networks: Mapping[str, Mapping[str, Any]],
    data_dir: Optional[Path | str] = None,
) -> dict[str, nib.Nifti1Image]:
    """Resample network arrays into NIfTI images aligned with the project mask.

    Parameters
    ----------
    networks:
        Mapping of network identifiers to dicts containing ``array`` and
        ``affine``.
    data_dir:
        Unused; kept for backward compatibility.

    Returns
    -------
    dict[str, nib.Nifti1Image]
    """
    mask_arrays, _, _ = _load_mask_bundle(data_dir=data_dir)
    networks_resampled: dict[str, nib.Nifti1Image] = {}
    for key, payload in networks.items():
        array = np.asarray(payload["array"])
        affine = np.asarray(payload["affine"])
        img = nib.Nifti1Image(array, affine=affine)
        networks_resampled[key] = _resample_to_mask(img, mask_arrays)
    return networks_resampled


def resample_networks_to_mask(networks: dict) -> dict[str, nib.Nifti1Image]:
    """Resample network images to match the mask affine and resolution.

    For binary networks, uses nearest-neighbour interpolation.
    For continuous networks, applies thresholding at the 95th percentile and
    binarises.

    Parameters
    ----------
    networks:
        Dict of networks.  Each value may be either a flat dict with ``array``
        and ``affine`` keys, or a nested dict whose values contain ``array``
        and ``affine`` keys.

    Returns
    -------
    dict[str, nib.Nifti1Image]
    """
    if any(isinstance(v, dict) and "array" not in v for v in networks.values()):
        networks = {k: v for _k in networks for k, v in networks[_k].items()}

    masker_obj = _load_masker()
    mask_img = masker_obj.mask_img_
    mask_arrays = {
        "mask": mask_img.get_fdata().astype(bool),
        "affine": mask_img.affine,
    }

    networks_resampled: dict[str, nib.Nifti1Image] = {}
    for k in tqdm(networks, total=len(networks), desc="Resampling networks"):
        img = nib.Nifti1Image(networks[k]["array"], affine=networks[k]["affine"])
        if len(np.unique(networks[k]["array"])) == 2:
            img_resampled = resample_img(img, mask_arrays["affine"], interpolation="nearest")
        else:
            img_resampled = resample_img(img, mask_arrays["affine"])
            arr = img_resampled.get_fdata()
            arr[arr < 0] = 0.0
            thresh = np.percentile(arr.flatten(), 95)
            arr[arr < thresh] = 0.0
            arr[arr >= thresh] = 1.0
            img_resampled = nib.Nifti1Image(arr, affine=mask_arrays["affine"])
        networks_resampled[k] = img_resampled
    return networks_resampled


# ---------------------------------------------------------------------------
# Main retrieval functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def search_text_corpus_given_neuroimage(
    neuroimage: nib.Nifti1Image,
    corpus_df: pd.DataFrame,
    corpus_embeddings: torch.Tensor,
    top_k: int = 10,
    show_names: bool = False,
) -> pd.DataFrame:
    """Rank a user corpus against a neuroimage using the contrastive model.

    Parameters
    ----------
    neuroimage:
        Input NIfTI brain map.  Will be resampled to the project mask
        internally.
    corpus_df:
        DataFrame describing your corpus.  Must contain ``name`` and
        ``description`` columns.
    corpus_embeddings:
        Pre-computed SPECTER embeddings for every row in *corpus_df*.
        Shape must be ``(N, 768)`` where ``N == len(corpus_df)``.  Embed each
        row's text (e.g. name + description) with SPECTER before calling this
        function.
    top_k:
        Number of top results to return.
    show_names:
        If True, print the ranked names to stdout.

    Returns
    -------
    pandas.DataFrame
        Columns: ``name``, ``description``, ``cosine_similarity``, sorted
        descending by similarity.
    """
    assert isinstance(neuroimage, nib.Nifti1Image), (
        "neuroimage must be a nibabel.Nifti1Image."
    )
    _validate_corpus(corpus_df, corpus_embeddings)

    # --- encode the neuroimage ---
    mask_arrays, _, masker = _load_mask_bundle()
    img_resampled = _resample_to_mask(neuroimage, mask_arrays)
    flat_np = masker.transform(img_resampled)
    flat_np[flat_np > 0] = 1.0
    flat_tensor = torch.tensor(flat_np, dtype=torch.float32)  # (1, 28542)

    autoencoder = _load_autoencoder()
    autoencoder.eval()
    latent = autoencoder.encoder(flat_tensor)  # (1, 384)

    proj_img = _proj_head_image_infonce()
    proj_img.eval()
    query_emb = proj_img(latent)             # (1, proj_dim)
    query_emb = _l2_normalize(query_emb)     # (1, proj_dim)

    # --- project corpus embeddings ---
    proj_txt = _proj_head_text_infonce()
    proj_txt.eval()
    corpus_proj = proj_txt(corpus_embeddings.float())  # (N, proj_dim)
    corpus_proj = _l2_normalize(corpus_proj)

    # --- cosine similarity and ranking ---
    cos_sim = (corpus_proj @ query_emb.T).squeeze(1)  # (N,)
    top_k = min(top_k, len(corpus_df))
    top_indices = torch.topk(cos_sim, k=top_k, largest=True, sorted=True).indices.tolist()

    rows = corpus_df.iloc[top_indices].copy().reset_index(drop=True)
    rows["cosine_similarity"] = cos_sim[top_indices].cpu().numpy()

    if show_names:
        print("Top matches:")
        for i, name in enumerate(rows["name"].tolist(), start=1):
            print(f"  {i}. {name}")

    return rows[["name", "description", "cosine_similarity"]]


@torch.no_grad()
def search_text_corpus_given_text(
    query: str,
    corpus_df: pd.DataFrame,
    corpus_embeddings: torch.Tensor,
    top_k: int = 10,
    show_names: bool = False,
) -> pd.DataFrame:
    """Rank a user corpus against a natural-language query using the contrastive model.

    Parameters
    ----------
    query:
        A natural-language string.
    corpus_df:
        DataFrame describing your corpus.  Must contain ``name`` and
        ``description`` columns.
    corpus_embeddings:
        Pre-computed SPECTER embeddings for every row in *corpus_df*.
        Shape must be ``(N, 768)`` where ``N == len(corpus_df)``.
    top_k:
        Number of top results to return.
    show_names:
        If True, print the ranked names to stdout.

    Returns
    -------
    pandas.DataFrame
        Columns: ``name``, ``description``, ``cosine_similarity``, sorted
        descending by similarity.
    """
    assert isinstance(query, str) and query.strip(), (
        "query must be a non-empty string."
    )
    _validate_corpus(corpus_df, corpus_embeddings)

    # --- encode query text ---
    specter = _load_specter()
    query_raw = specter(query)[0].detach()  # (768,)
    query_raw = query_raw.unsqueeze(0)      # (1, 768)

    proj_txt = _proj_head_text_infonce()
    proj_txt.eval()
    query_emb = proj_txt(query_raw)          # (1, proj_dim)
    query_emb = _l2_normalize(query_emb)

    # --- project corpus embeddings ---
    corpus_proj = proj_txt(corpus_embeddings.float())  # (N, proj_dim)
    corpus_proj = _l2_normalize(corpus_proj)

    # --- cosine similarity and ranking ---
    cos_sim = (corpus_proj @ query_emb.T).squeeze(1)  # (N,)
    top_k = min(top_k, len(corpus_df))
    top_indices = torch.topk(cos_sim, k=top_k, largest=True, sorted=True).indices.tolist()

    rows = corpus_df.iloc[top_indices].copy().reset_index(drop=True)
    rows["cosine_similarity"] = cos_sim[top_indices].cpu().numpy()

    if show_names:
        print("Top matches:")
        for i, name in enumerate(rows["name"].tolist(), start=1):
            print(f"  {i}. {name}")

    return rows[["name", "description", "cosine_similarity"]]


# ---------------------------------------------------------------------------
# LLM summarisation
# ---------------------------------------------------------------------------

def _system_prompt_user_corpus(query_type: Literal["neuroimage", "text"]) -> str:
    if query_type == "neuroimage":
        return (
            "You are a helpful neuroscience research assistant.\n"
            "You will receive a ranked list of entries from a user-provided corpus that are "
            "most similar to an input brain activation map as determined by a contrastive "
            "neuro-language model. Your task is to interpret what the brain activation pattern "
            "represents based on these corpus entries.\n\n"
            "Your response must:\n"
            "- Begin with a 2-4 sentence overview of the main themes the brain pattern suggests.\n"
            "- Ground every statement in the provided corpus entries. Do not speculate beyond them.\n"
            "- Synthesize across entries, noting convergent themes.\n"
            "- Address the user's query directly if one is provided.\n"
            "- Maintain an objective, scholarly tone."
        )
    return (
        "You are a helpful neuroscience research assistant.\n"
        "You will receive a ranked list of entries from a user-provided corpus that are "
        "most similar to a text query as determined by a contrastive neuro-language model. "
        "Your task is to summarize how these entries relate to the query.\n\n"
        "Your response must:\n"
        "- Begin with a 2-4 sentence overview of the main themes in the top entries.\n"
        "- Be entirely grounded in the provided entries. Do not add outside knowledge.\n"
        "- Synthesize across entries, noting convergent themes or divergences.\n"
        "- Address the user's query directly.\n"
        "- Maintain an objective, scholarly tone."
    )


def _format_context(context_df: pd.DataFrame) -> str:
    """Format a ranked results DataFrame into a numbered context string."""
    pieces = []
    for idx, (_, row) in enumerate(context_df.iterrows(), start=1):
        name = str(row.get("name", "Untitled")).strip()
        desc = re.sub(r"\s+", " ", str(row.get("description", ""))).strip()
        pieces.append(f"[{idx}] {name}\n{desc}")
    return "\n\n".join(pieces)


def generate_llm_response(
    context_df: pd.DataFrame,
    query_type: Literal["neuroimage", "text"],
    backend: Literal["ollama", "huggingface"],
    model_name: str,
    user_prompt: str = "",
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> str:
    """Generate an LLM summary from ranked user-corpus results.

    Parameters
    ----------
    context_df:
        DataFrame returned by :func:`search_text_corpus_given_neuroimage` or
        :func:`search_text_corpus_given_text`.  Must contain ``name`` and
        ``description`` columns.
    query_type:
        ``"neuroimage"`` if the context was obtained by querying with a brain
        map, or ``"text"`` if it was obtained with a text string.  This
        controls the LLM system prompt framing.
    backend:
        LLM backend to use.

        - ``"ollama"``: requires Ollama installed and running locally (fast).
        - ``"huggingface"``: loads the model directly from HuggingFace
          (slower, works offline).
    model_name:
        Model identifier.

        - Ollama examples: ``"llama3.2:3b"``, ``"qwen2.5:3b-instruct"``
        - HuggingFace examples: ``"Qwen/Qwen2.5-1.5B-Instruct"``,
          ``"HuggingFaceTB/SmolLM2-360M-Instruct"``
    user_prompt:
        Optional extra question or instruction appended to the LLM prompt.
    max_new_tokens:
        Maximum tokens to generate (HuggingFace backend only).
    verbose:
        Print progress messages.

    Returns
    -------
    str
        LLM-generated response text.

    Examples
    --------
    Brain-to-corpus::

        results = search_text_corpus_given_neuroimage(nifti_img, my_df, my_embs)
        response = generate_llm_response(
            results,
            query_type="neuroimage",
            backend="ollama",
            model_name="qwen2.5:3b-instruct",
        )

    Text-to-corpus::

        results = search_text_corpus_given_text("working memory", my_df, my_embs)
        response = generate_llm_response(
            results,
            query_type="text",
            backend="huggingface",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
        )
    """
    assert query_type in ("neuroimage", "text"), (
        "query_type must be 'neuroimage' or 'text'."
    )
    assert isinstance(context_df, pd.DataFrame) and not context_df.empty, (
        "context_df must be a non-empty DataFrame."
    )
    missing = [c for c in ("name", "description") if c not in context_df.columns]
    assert not missing, (
        f"context_df is missing column(s): {', '.join(missing)}."
    )

    context_str = _format_context(context_df)
    source_label = "brain activation map" if query_type == "neuroimage" else "text query"
    user_message = (
        f"Here are the top-ranked entries from the user corpus most similar to the {source_label}:\n\n"
        f"{context_str}"
    )
    if user_prompt:
        user_message = f"User query: \"{user_prompt}\"\n\n{user_message}"

    messages = [
        {"role": "system", "content": _system_prompt_user_corpus(query_type)},
        {"role": "user", "content": user_message},
    ]

    if verbose:
        print(f"Generating LLM response (query_type='{query_type}', backend='{backend}')...")

    from neurovlm.llm_summary import _generate_with_ollama, _generate_with_huggingface

    if backend == "ollama":
        output = _generate_with_ollama(messages, model_name=model_name, verbose=verbose)
    elif backend == "huggingface":
        output = _generate_with_huggingface(
            messages, model_name=model_name, max_new_tokens=max_new_tokens, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'ollama' or 'huggingface'.")

    if verbose:
        print("LLM finished.")
        print(output)

    return output
