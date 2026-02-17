"""
Utilities for mapping brain activation maps to relevant literature.

This module consolidates the logic experimented with in the
``brain_to_text`` notebook into reusable functions. The workflow:

1. Load NeuroVLM models and metadata tables.
2. Optionally deduplicate latent text batches saved on disk.
3. Flatten neuroimaging volumes into masked vectors.
4. Encode a query map, search for the closest text embeddings, and
   summarise the corresponding publications.

The top-level :func:`run_brain_to_text_pipeline` function shows how the
pieces fit together.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, List, Mapping, Tuple, LiteralString, Literal

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.image import resample_img
from nilearn.maskers import NiftiMasker

from nilearn import maskers
from tqdm import tqdm

from neurovlm.data import fetch_data, get_data_dir
from neurovlm.train import which_device
from neurovlm.retrieval_resources import (
    _load_pubmed_dataframe,
    _load_latent_text,
    _load_latent_wiki,
    _load_latent_cogatlas,
    _load_neuro_wiki,
    _load_cogatlas_dataset,
    _proj_head_image_infonce,
    _proj_head_text_infonce,
    _load_latent_cogatlas_disorder,
    _load_latent_cogatlas_task,
    _load_cogatlas_disorder_dataset,
    _load_cogatlas_task_dataset,
    _load_masker,
    _load_networks
)
data_dir = get_data_dir()

warnings.filterwarnings("ignore")


@torch.no_grad()
def search_papers_from_brain(
    query: torch.Tensor,
    top_k: int = 5,
    show_titles: bool = False,
) -> tuple[LiteralString, list[str], list]:
    """Return context for the most similar papers to a brain-derived embedding."""
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor for brain-based retrieval")

    df = _load_pubmed_dataframe()
    latent_text, latent_pmids = _load_latent_text()
    proj_head_img = _proj_head_image_infonce()
    proj_head_text = _proj_head_text_infonce()

    proj_img = proj_head_img(query).squeeze(0)
    proj_img = proj_img / proj_img.norm()

    text_embeddings = latent_text / latent_text.norm(dim=1)[:, None]
    proj_text = proj_head_text(text_embeddings)
    proj_text = proj_text / proj_text.norm(dim=1)[:, None]

    cos_sim = proj_text @ proj_img

    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()
    cos_sim_top = cos_sim[inds_top].detach().cpu().numpy()
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
                print(f"PMID {pmid} not found in DataFrame.")
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
    return papers_context, titles, cos_sim_top


@torch.no_grad()
def search_wiki_from_brain(
    query: torch.Tensor,
    top_k: int = 2,
    show_titles: bool = False,
) -> Tuple[str, List[str], np.ndarray]:
    """Return context for the most similar NeuroWiki entries to a brain-derived embedding."""
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor for brain-based retrieval")

    df = _load_neuro_wiki()
    latent_wiki, latent_ids = _load_latent_wiki()

    proj_head_img = _proj_head_image_infonce()
    proj_head_text = _proj_head_text_infonce()

    proj_img = proj_head_img(query).squeeze(0)
    proj_img = proj_img / proj_img.norm()

    wiki_embed = latent_wiki / latent_wiki.norm(dim=1)[:, None]
    proj_wiki = proj_head_text(wiki_embed)
    proj_wiki = proj_wiki / proj_wiki.norm(dim=1)[:, None]

    cos_sim = proj_wiki @ proj_img

    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()
    cos_sim_top = cos_sim[inds_top].detach().cpu().numpy()
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
    return wiki_context, titles, cos_sim_top


@torch.no_grad()
def search_cogatlas_from_brain(
    query: torch.Tensor,
    top_k: int = 5,
    show_titles: bool = False,
    category: Literal["cogatlas", "cogatlas_task", "cogatlas_disorder"] = "cogatlas",
) -> Tuple[str, List[str], np.ndarray]:
    """Return context for the most similar Cognitive Atlas terms to a brain-derived embedding."""
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor for brain-based retrieval")

    if category == "cogatlas":
        df = _load_cogatlas_dataset()
        latent_cogatlas, latent_terms = _load_latent_cogatlas()
    elif category == "cogatlas_task":
        df = _load_cogatlas_task_dataset(filtered=True)
        latent_cogatlas, latent_terms = _load_latent_cogatlas_task()
    elif category == "cogatlas_disorder":
        df = _load_cogatlas_disorder_dataset()
        latent_cogatlas, latent_terms = _load_latent_cogatlas_disorder()
    else:
        raise ValueError()

    proj_head_img = _proj_head_image_infonce()
    proj_head_text = _proj_head_text_infonce()

    proj_img = proj_head_img(query).squeeze(0)
    proj_img = proj_img / proj_img.norm()

    cogatlas_embed = latent_cogatlas / latent_cogatlas.norm(dim=1)[:, None]
    proj_cogatlas = proj_head_text(cogatlas_embed)
    proj_cogatlas = proj_cogatlas / proj_cogatlas.norm(dim=1)[:, None]

    cos_sim = proj_cogatlas @ proj_img

    inds = torch.argsort(cos_sim, descending=True)
    inds_top = inds[:top_k].tolist()
    cos_sim_top = cos_sim[inds_top].detach().cpu().numpy()
    terms_top = [latent_terms[i] for i in inds_top]

    missing_columns = [col for col in ("term", "definition") if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"CogAtlas DataFrame is missing required columns: {', '.join(missing_columns)}"
        )

    # Ensure term column is lowercase for matching
    df["term"] = df["term"].str.lower()
    term_lookup = df.drop_duplicates("term").set_index("term", drop=False)
    selected_rows = []
    for term in terms_top:
        try:
            row = term_lookup.loc[term]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            selected_rows.append(row)
        except KeyError:
            print(f"Term '{term}' not found in CogAtlas DataFrame.")
            continue

    rows = pd.DataFrame(selected_rows) if selected_rows else df.iloc[inds_top]

    pieces = []
    terms: List[str] = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        term = str(row["term"]) if pd.notna(row["term"]) else "Untitled"
        terms.append(term)
        definition = str(row["definition"]) if pd.notna(row["definition"]) else ""
        definition = re.sub(r"\s+", " ", definition).strip()
        pieces.append(f"[{idx}] {term}\n{definition}\n")

    if show_titles:
        print("Top matches:")
        for idx, term in enumerate(terms, start=1):
            print(f"{idx}. {term}")

    cogatlas_context = "\n".join(pieces)
    return cogatlas_context, terms, cos_sim_top


def load_metadata(data_dir: Path | str | None = data_dir) -> dict[str, pd.DataFrame]:
    """
    Load publication metadata used to map latent vectors to textual data.

    Parameters
    ----------
    data_dir:
        Optional path returned by :func:`fetch_data`. When omitted the data
        directory is fetched (or validated) automatically.
    """
    # Load publications from HuggingFace
    df_pubs = _load_pubmed_dataframe()

    # Coordinates are still loaded locally as they're not yet on HuggingFace
    root = Path(fetch_data() if data_dir is None else data_dir)
    df_coords = pd.read_parquet(root / "coordinates.parquet")

    return {"publications": df_pubs, "coordinates": df_coords}


def _load_mask_bundle(
    data_dir: Path | str | None = None,
) -> tuple[dict[str, Any], nib.Nifti1Image, NiftiMasker]:
    """Return mask arrays, image, and fitted masker.

    Note: The masker is loaded from HuggingFace via retrieval_resources.
    The data_dir parameter is kept for backward compatibility but is now optional.
    """
    # Load masker from HuggingFace (already returns a fitted masker)
    masker = _load_masker()
    mask_img = masker.mask_img_

    # Extract mask arrays from the mask image
    mask_arrays = {
        "mask": mask_img.get_fdata().astype(bool),
        "affine": mask_img.affine
    }

    return mask_arrays, mask_img, masker


def _resample_to_mask(
    img: nib.Nifti1Image,
    mask_arrays: Mapping[str, Any],
) -> nib.Nifti1Image:
    """Resample an image to match the project mask affine."""
    img_arr = img.get_fdata()
    unique_values = np.unique(img_arr)
    is_binary = len(unique_values) == 2 and set(np.round(unique_values).tolist()) <= {0, 1}

    if is_binary:
        return resample_img(
            img,
            target_affine=mask_arrays["affine"],
            interpolation="nearest",
        )

    img_resampled = resample_img(
        img,
        target_affine=mask_arrays["affine"],
    )
    img_resampled_arr = img_resampled.get_fdata()
    img_resampled_arr[img_resampled_arr < 0] = 0.0
    thresh = np.percentile(img_resampled_arr.flatten(), 95)
    img_resampled_arr[img_resampled_arr < thresh] = 0.0
    img_resampled_arr[img_resampled_arr >= thresh] = 1.0
    return nib.Nifti1Image(
        img_resampled_arr,
        affine=mask_arrays["affine"],
    )


def resample_nifti(
    nifti_img: nib.Nifti1Image,
    data_dir: Path | str | None = None,
) -> torch.Tensor:
    """
    Resample and flatten a NIfTI image using the project mask definition.
    """
    mask_arrays, _, masker = _load_mask_bundle(data_dir=data_dir)
    img_resampled = _resample_to_mask(nifti_img, mask_arrays)

    flattened = masker.transform(img_resampled)
    flattened[flattened > 0] = 1
    out_tensor = torch.tensor(flattened).squeeze(0)
    return out_tensor


def resample_array_nifti(
    networks: Mapping[str, Mapping[str, Any]],
    data_dir: Path | str | None = None,
) -> dict[str, nib.Nifti1Image]:
    """
    Resample stored network arrays into NIfTI images aligned with the project mask.

    Parameters
    ----------
    networks:
        Mapping of network identifiers to dictionaries containing ``array`` and ``affine``.
    data_dir:
        Optional directory pointing to the cached NeuroVLM data (defaults to :func:`fetch_data`).

    Returns
    -------
    dict[str, nib.Nifti1Image]
        Resampled NIfTI images keyed by the original network identifier.
    """
    mask_arrays, _, _ = _load_mask_bundle(data_dir=data_dir)
    networks_resampled: dict[str, nib.Nifti1Image] = {}

    for key, payload in networks.items():
        array = np.asarray(payload["array"])
        affine = np.asarray(payload["affine"])
        img = nib.Nifti1Image(array, affine=affine)
        networks_resampled[key] = _resample_to_mask(img, mask_arrays)

    return networks_resampled


def resample_networks_to_mask(networks):
    """
    Resample network images to match the mask affine and resolution.

    For binary networks, uses nearest neighbor interpolation.
    For continuous networks, applies thresholding at 95th percentile and binarizes.

    Args:
        networks: Dict of networks, where each network has nested dicts with 'array' and 'affine' keys,
                  OR already flattened dict with 'array' and 'affine' keys

    Returns:
        networks_resampled: Dict mapping network names to resampled NIfTI images
    """

    # Flatten networks if needed (handle nested dict structure)
    if any(isinstance(v, dict) and 'array' not in v for v in networks.values()):
        networks = {k: v for _k in networks.keys() for k, v in networks[_k].items()}

    # Load mask from HuggingFace
    masker = _load_masker()
    mask_img = masker.mask_img_
    mask_arrays = {
        "mask": mask_img.get_fdata().astype(bool),
        "affine": mask_img.affine
    }

    networks_resampled = {}

    for k in tqdm(networks.keys(), total=len(networks), desc="Resampling networks"):
        img = nib.Nifti1Image(networks[k]["array"], affine=networks[k]["affine"])

        if len(np.unique(networks[k]["array"])) == 2:
            # Binary data: use nearest neighbor interpolation
            img_resampled = resample_img(img, mask_arrays["affine"], interpolation="nearest")
        else:
            # Continuous data: resample, then threshold and binarize
            img_resampled = resample_img(img, mask_arrays["affine"])
            img_resampled_arr = img_resampled.get_fdata()

            # Remove negative values
            img_resampled_arr[img_resampled_arr < 0] = 0.

            # Threshold at 95th percentile
            thresh = np.percentile(img_resampled_arr.flatten(), 95)
            img_resampled_arr[img_resampled_arr < thresh] = 0.
            img_resampled_arr[img_resampled_arr >= thresh] = 1.

            img_resampled = nib.Nifti1Image(img_resampled_arr, affine=mask_arrays["affine"])

        networks_resampled[k] = img_resampled

    return networks_resampled


@torch.no_grad()
def generate_llm_response_from_brain(
    query_vector: torch.Tensor,
    top_k_wiki: int = 5,
    top_k_cogatlas_concepts: int = 5,
    top_k_cogatlas_disorders: int = 5,
    top_k_cogatlas_tasks: int = 5,
    user_prompt: str = "",
    backend: Literal["ollama", "huggingface"] = "ollama",
    model_name: str | None = None,
):
    """
    Generate an LLM response based on a brain-derived vector.

    For brain input, focuses on NeuroWiki terms and CogAtlas terms (concepts, disorders, tasks).
    Papers are NOT included for brain-based queries.

    Parameters
    ----------
    query_vector : torch.Tensor
        An already encoded brain-derived vector.
    top_k_wiki : int
        Number of top NeuroWiki entries to retrieve.
    top_k_cogatlas_concepts : int
        Number of top CogAtlas concept terms to retrieve.
    top_k_cogatlas_disorders : int
        Number of top CogAtlas disorder terms to retrieve.
    top_k_cogatlas_tasks : int
        Number of top CogAtlas task terms to retrieve.
    user_prompt : str
        Optional user query/prompt to provide context.
    backend : {"ollama", "huggingface"}, optional
        Which LLM backend to use. Default: "ollama" (faster, requires Ollama installed).
    model_name : str, optional
        Model name. If None, uses backend defaults.

    Returns
    -------
    str
        The generated response from the LLM.

    Examples
    --------
    >>> # Use Ollama (default, fast)
    >>> output = generate_llm_response_from_brain(brain_vector)

    >>> # Use HuggingFace
    >>> output = generate_llm_response_from_brain(
    ...     brain_vector,
    ...     backend="huggingface",
    ...     model_name="Qwen/Qwen2.5-0.5B-Instruct"
    ... )
    """
    from neurovlm.llm_summary import generate_response

    # Retrieve top k for each category
    wiki_context, wiki_titles, _ = search_wiki_from_brain(
        query_vector, top_k=top_k_wiki, show_titles=False
    )

    cogatlas_concepts_context, cogatlas_concepts_terms, _ = search_cogatlas_from_brain(
        query_vector, top_k=top_k_cogatlas_concepts, show_titles=False, category="cogatlas"
    )

    cogatlas_disorders_context, cogatlas_disorders_terms, _ = search_cogatlas_from_brain(
        query_vector, top_k=top_k_cogatlas_disorders, show_titles=False, category="cogatlas_disorder"
    )

    cogatlas_tasks_context, cogatlas_tasks_terms, _ = search_cogatlas_from_brain(
        query_vector, top_k=top_k_cogatlas_tasks, show_titles=False, category="cogatlas_task"
    )

    # Combine all cogatlas contexts
    cogatlas_combined_context = "\n".join([
        "Concepts:\n" + cogatlas_concepts_context,
        "Disorders:\n" + cogatlas_disorders_context,
        "Tasks:\n" + cogatlas_tasks_context,
    ])

    return generate_response(
        query=query_vector,
        papers_context=None,  # No papers for brain input
        wiki_context=wiki_context,
        cogatlas_context=cogatlas_combined_context,
        user_prompt=user_prompt,
        backend=backend,
        model_name=model_name,
    )


__all__ = [
    "load_metadata",
    "resample_nifti",
    "resample_array_nifti",
    "search_papers_from_brain",
    "search_wiki_from_brain",
    "search_cogatlas_from_brain",
    "resample_networks_to_mask",
    "generate_llm_response_from_brain",
]
