"""Shared loaders for NeuroVLM retrieval tasks.

These helpers centralize cached access to publication metadata,
latent embeddings, and projection heads so they can be shared across
brain- and text-driven retrieval workflows.

Files are now loaded from HuggingFace repositories under the neurovlm organization.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple
import gzip, json, pickle

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors

import nibabel as nib
from nilearn import maskers
from huggingface_hub import hf_hub_download

from neurovlm.models import Specter, ProjHead, NeuroAutoEncoder
from neurovlm.io import load_model

__all__ = [
    "_load_pubmed_dataframe",
    "_load_pubmed_summaries_dataframe",
    "_load_neuro_wiki",
    "_load_pubmed_coordinates",
    "_load_latent_neuro",
    "_load_cogatlas_dataset",
    "_load_cogatlas_task_dataset",
    "_load_cogatlas_disorder_dataset",
    "_load_cogatlas_term_data",
    "_load_cogatlas_term_threshold_data",
    "_load_ngram_data",
    "_load_threshold_analysis_cache",
    "_load_threshold_analysis_text_cache",
    "_load_specter",
    "_load_latent_text",
    "_load_latent_neuro_summaries",
    "_load_latent_wiki",
    "_load_latent_cogatlas",
    "_load_latent_cogatlas_disorder",
    "_load_latent_cogatlas_task",
    "_load_autoencoder",
    "_load_masker",
    "_load_networks",
    "_load_network_test_set_labels",
    "_load_pubmed_mesh_annotations",
    "_proj_head_image_infonce",
    "_proj_head_text_mse",
    "_proj_head_text_infonce",
    "_load_images_neurovault_dataframe",
    "_load_publications_neurovault_dataframe",
    "_load_latent_neurovault_images",
    "_load_latent_neurovault_text",
    "_load_neurovault_images",
    "_load_pubmed_images",
    "_load_latent_ngram",
    "_load_ngram",
    "_load_kg_mesh_dataset",
    "_load_mesh_kg_nodes",
    "_load_mesh_kg_descriptors",
    "_load_kg_mesh_brain_rankable_dataset",
    "_load_latent_kg_mesh",
    "_load_latent_kg_mesh_brain_rankable",
    "_load_llm_neuro_terms_dataset",
    "_load_latent_llm_neuro_terms",
]


def _download_from_hf(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Download a file from HuggingFace and return the local path.

    Uses local cache first, only downloads if not cached.

    Args:
        repo_id: Repository ID (e.g., 'neurovlm/embedded_text')
        filename: Name of the file to download
        repo_type: Type of repository - "dataset" or "model" (default: "dataset")
    """
    import os
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_files_only=True,
        )
    except Exception:
        pass

    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        # Scan all snapshots for the file as a fallback
        from huggingface_hub.constants import HF_HUB_CACHE
        repo_folder_name = f"{'models' if repo_type == 'model' else 'datasets'}--{repo_id.replace('/', '--')}"
        snapshots_dir = os.path.join(HF_HUB_CACHE, repo_folder_name, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snapshot in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, snapshot, filename)
                if os.path.exists(candidate):
                    return candidate
        raise FileNotFoundError(
            f"Cannot find {filename} in local HF cache for {repo_id} "
            f"and offline mode is enabled."
        )

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )


@lru_cache(maxsize=1)
def _load_pubmed_dataframe() -> pd.DataFrame:
    """Load the publications DataFrame from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/neuro_image_papers",
        "publications.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_pubmed_summaries_dataframe() -> pd.DataFrame:
    """Load NeuroVLM PubMed summary text from HuggingFace.

    The table is expected to contain at least ``pmid`` and ``summary`` columns.
    Boolean ``train``, ``val``, and ``test`` columns may be present after
    running the preparation script that aligns summaries with
    ``publications.parquet``.
    """
    parquet_path = _download_from_hf(
        "neurovlm/neuro_image_papers",
        "neuro_summaries.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_pubmed_coordinates() -> pd.DataFrame:
    """Load the x, y, z coordinates DataFrame from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/neuro_image_papers",
        "coordinates.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_neuro_wiki() -> pd.DataFrame:
    """Load the NeuroWiki DataFrame from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/neuro_wiki",
        "neurowiki_with_ids.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")

@lru_cache(maxsize=1)
def _load_neuro_wiki_graph() -> pd.DataFrame:
    """Load the NeuroWiki graph DataFrame from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/neuro_wiki",
        "neurowiki_graph.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_latent_neuro() -> Tuple[torch.Tensor, np.ndarray]:
    """Load the Neuro brain map embedding from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_neuro.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location=torch.device("cpu"),
    )

    latent = latent_payload["latent"]
    latent_pmid = np.asarray(latent_payload["pmid"])
    return latent, latent_pmid


@lru_cache(maxsize=1)
def _load_cogatlas_dataset() -> pd.DataFrame:
    """Load the CogAtlas DataFrame from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "cogatlas.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


def _load_cogatlas_task_dataset(filtered=False) -> pd.DataFrame:
    """Load the CogAtlas task DataFrame from HuggingFace."""
    filename = "cogatlas_task_filtered.parquet" if filtered else "cogatlas_task.parquet"
    parquet_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        filename
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


def _load_cogatlas_disorder_dataset(filtered=False) -> pd.DataFrame:
    """Load the CogAtlas disorder DataFrame from HuggingFace."""
    filename = "cogatlas_disorder_filtered.parquet" if filtered else "cogatlas_disorder.parquet"
    parquet_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        filename
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


def _load_cogatlas_graph_dataset(filtered=False) -> pd.DataFrame:
    """Load the CogAtlas graph DataFrame from HuggingFace."""
    filename = "cogatlas_graph.parquet"
    parquet_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        filename
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


def _load_cogatlas_term_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load CogAtlas term classification training data from HuggingFace.

    Returns:
        matrix: Document-term co-occurrence matrix (n_documents x n_terms)
        labels: Term labels/names
        pmids: Publication IDs
        category_info: Dictionary with term category information
    """
    matrix_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "cogatlas_term_matrix.npy"
    )
    labels_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "cogatlas_term_labels.npy"
    )
    pmids_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "cogatlas_term_pmids.npy"
    )
    category_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "cogatlas_term_category_info.json"
    )

    import json
    matrix = np.load(matrix_path)
    labels = np.load(labels_path)
    pmids = np.load(pmids_path)
    with open(category_path, 'r') as f:
        category_info = json.load(f)

    return matrix, labels, pmids, category_info


def _load_cogatlas_term_threshold_data(threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load CogAtlas term classification data filtered by threshold from HuggingFace.

    Args:
        threshold: Similarity threshold used to filter terms (e.g., 0.6, 0.65)

    Returns:
        matrix: Document-term co-occurrence matrix (n_documents x n_terms)
        labels: Term labels/names
        pmids: Publication IDs
        category_info: Dictionary with term category information
    """
    # Format threshold as string with underscore (e.g., 0.6 -> "0_6000", 0.65 -> "0_6500")
    threshold_str = f"{threshold:.4f}".replace(".", "_")

    matrix_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        f"cogatlas_term_threshold_{threshold_str}_matrix.npy"
    )
    labels_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        f"cogatlas_term_threshold_{threshold_str}_labels.npy"
    )
    pmids_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        f"cogatlas_term_threshold_{threshold_str}_pmids.npy"
    )
    category_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        f"cogatlas_term_threshold_{threshold_str}_category_info.json"
    )

    import json
    matrix = np.load(matrix_path)
    labels = np.load(labels_path)
    pmids = np.load(pmids_path)
    with open(category_path, 'r') as f:
        category_info = json.load(f)

    return matrix, labels, pmids, category_info


def _load_ngram_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load n-gram classification data from HuggingFace.

    Returns:
        matrix: Document-ngram co-occurrence matrix
        labels: N-gram labels
    """
    matrix_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "ngram_matrix.npy"
    )
    labels_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "ngram_labels.npy"
    )

    matrix = np.load(matrix_path)
    labels = np.load(labels_path)

    return matrix, labels


def _load_threshold_analysis_cache():
    """Load threshold analysis similarities cache from HuggingFace.

    Returns:
        Cache dictionary containing precomputed similarities for threshold analysis
    """
    cache_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "threshold_analysis_similarities_cache.pkl"
    )

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    return cache


def _load_threshold_analysis_text_cache():
    """Load threshold analysis text similarities cache from HuggingFace.

    Returns:
        Cache dictionary containing precomputed text similarities for threshold analysis
    """
    cache_path = _download_from_hf(
        "neurovlm/cognitive_atlas",
        "threshold_analysis_text_similarities_cache.pkl"
    )

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    return cache


@lru_cache(maxsize=1)
def _load_specter() -> Specter:
    """Construct and cache a Specter encoder."""
    return Specter()


@lru_cache(maxsize=1)
def _load_latent_text() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent text embeddings from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_specter2_adhoc.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    latent = latent_payload["latent"]
    latent_pmid = np.asarray(latent_payload["pmid"])
    return latent, latent_pmid


@lru_cache(maxsize=1)
def _load_latent_neuro_summaries() -> Tuple[torch.Tensor, np.ndarray]:
    """Load SPECTER embeddings for PubMed neuro summaries from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_neuro_summaries.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    latent = latent_payload["latent"]
    latent_pmid = np.asarray(latent_payload["pmid"])
    return latent, latent_pmid


@lru_cache(maxsize=1)
def _load_latent_wiki() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent wiki embeddings from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_specter_wiki.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    latent = latent_payload["latent"]
    latent_id = np.asarray(latent_payload["id"])
    return latent, latent_id


@lru_cache(maxsize=1)
def _load_latent_cogatlas() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent cognitive atlas embeddings from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_cogatlas.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    latent = latent_payload["latent"]
    latent_terms = np.asarray(latent_payload["term"])
    return latent, latent_terms


@lru_cache(maxsize=1)
def _load_latent_cogatlas_disorder() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent cognitive atlas disorder embeddings from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_cogatlas_disorder.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    latent = latent_payload["latent"]
    latent_terms = np.asarray(latent_payload["term"])
    return latent, latent_terms


@lru_cache(maxsize=1)
def _load_latent_cogatlas_task() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent cognitive atlas task embeddings from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_cogatlas_task.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    latent = latent_payload["latent"]
    latent_terms = np.asarray(latent_payload["term"])
    return latent, latent_terms


@lru_cache(maxsize=1)
def _load_images_neurovault_dataframe() -> pd.DataFrame:
    """Load NeuroVault image metadata from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/embedded_text",
        "images_neurovault.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_publications_neurovault_dataframe() -> pd.DataFrame:
    """Load NeuroVault publication metadata from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/embedded_text",
        "publications_neurovault.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_latent_neurovault_images() -> torch.Tensor:
    """Load NeuroVault image latent tensor from HuggingFace on CPU."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_neurovault_images.pt"
    )
    latent = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )
    if not isinstance(latent, torch.Tensor):
        raise TypeError("Expected `latent_neurovault_images.pt` to contain a torch.Tensor.")
    return latent.cpu()


@lru_cache(maxsize=1)
def _load_latent_neurovault_text() -> torch.Tensor:
    """Load NeuroVault text latent tensor from HuggingFace on CPU."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_neurovault_text.pt"
    )
    latent = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )
    if not isinstance(latent, torch.Tensor):
        raise TypeError("Expected `latent_neurovault_text.pt` to contain a torch.Tensor.")
    return latent.cpu()


@lru_cache(maxsize=1)
def _load_neurovault_images() -> torch.Tensor:
    """Load NeuroVault image tensor from HuggingFace on CPU."""
    image_path = _download_from_hf(
        "neurovlm/embedded_text",
        "neurovault_images.pt"
    )
    images = torch.load(
        image_path,
        weights_only=False,
        map_location="cpu"
    )
    if not isinstance(images, torch.Tensor):
        raise TypeError("Expected `neurovault_images.pt` to contain a torch.Tensor.")
    return images.cpu()


@lru_cache(maxsize=1)
def _load_pubmed_images() -> torch.Tensor:
    """Load PubMed image tensor from HuggingFace on CPU."""
    image_path = _download_from_hf(
        "neurovlm/embedded_text",
        "pubmed_images.pt"
    )
    images, pmids = torch.load(
        image_path,
        weights_only=False,
        map_location="cpu"
    ).values()
    return images, pmids

@lru_cache(maxsize=1)
def _load_latent_networks_canonical_text() -> dict:
    """Load latent network atlases."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_networks_text.pt"
    )
    latents = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    return latents

@lru_cache(maxsize=1)
def _load_latent_networks_neuro() -> dict:
    """Load latent network atlases."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_networks_image.pt"
    )
    latents = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )

    return latents


@lru_cache(maxsize=1)
def _load_autoencoder() -> torch.nn.Module:
    """Load and return the text encoder model from HuggingFace."""
    model_path = _download_from_hf(
        "neurovlm/encoder_and_proj_head",
        "autoencoder.safetensors",
        repo_type="model"
    )
    autoencoder = load_model(NeuroAutoEncoder(seed=0, out="logit"), model_path)
    return autoencoder


@lru_cache(maxsize=1)
def _load_masker() -> nib.Nifti1Image:
    """Load mask from HuggingFace."""
    mask_path = _download_from_hf(
        "neurovlm/encoder_and_proj_head",
        "mask.npz",
        repo_type="model"
    )
    mask_arrays = np.load(mask_path, allow_pickle=True)
    mask_img = nib.Nifti1Image(mask_arrays["mask"].astype(float), mask_arrays["affine"])
    masker = maskers.NiftiMasker(mask_img=mask_img, dtype=np.float32).fit()
    return masker


@lru_cache(maxsize=1)
def _load_networks() -> dict:
    """Load network atlases from HuggingFace."""
    networks_path = _download_from_hf(
        "neurovlm/embedded_text",
        "networks_arrays.pkl.gz"
    )
    with gzip.open(networks_path, "rb") as f:
        networks = pickle.load(f)

    return networks


@lru_cache(maxsize=1)
def _load_networks_labels() -> dict:
    """Load network atlases from HuggingFace."""
    networks_path = _download_from_hf(
        "neurovlm/embedded_text",
        "networks_labels.parquet"
    )
    return pd.read_parquet(networks_path)


@lru_cache(maxsize=1)
def _load_networks_canonical() -> pd.DataFrame:
    """Load names and descriptions of common networks."""
    networks_path = _download_from_hf(
        "neurovlm/embedded_text",
        "network_text.parquet"
    )
    return pd.read_parquet(networks_path)


@lru_cache(maxsize=1)
def _load_network_test_set_labels() -> pd.DataFrame:
    """Load labeled network evaluation rows from HuggingFace."""
    labels_path = _download_from_hf(
        "neurovlm/embedded_text",
        "network_test_set_labels.csv"
    )
    return pd.read_csv(labels_path)


@lru_cache(maxsize=1)
def _load_pubmed_mesh_annotations() -> dict:
    """Load PubMed PMID-to-MeSH gold annotations from HuggingFace."""
    annotations_path = _download_from_hf(
        "neurovlm/mesh_kg",
        "mesh_annotations.json"
    )
    with open(annotations_path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _proj_head_image_infonce() -> torch.nn.Module:
    """Load and return the image projection head from HuggingFace."""
    model_path = _download_from_hf(
        "neurovlm/encoder_and_proj_head",
        "proj_head_image_infonce.safetensors",
        repo_type="model"
    )
    proj_head = load_model(ProjHead(384, 384, 384), model_path)
    return proj_head


@lru_cache(maxsize=1)
def _proj_head_text_mse() -> torch.nn.Module:
    """Load and return the MSE projection head from HuggingFace."""
    model_path = _download_from_hf(
        "neurovlm/encoder_and_proj_head",
        "proj_head_text_mse.safetensors",
        repo_type="model"
    )
    proj_head = load_model(ProjHead(768, 512, 384), model_path)
    return proj_head


@lru_cache(maxsize=1)
def _proj_head_text_infonce() -> torch.nn.Module:
    """Load and return the text projection head from HuggingFace."""
    model_path = _download_from_hf(
        "neurovlm/encoder_and_proj_head",
        "proj_head_text_infonce.safetensors",
        repo_type="model"
    )
    proj_head = load_model(ProjHead(768, 512, 384), model_path)
    return proj_head


@lru_cache(maxsize=1)
def _load_latent_ngram() -> Tuple[torch.Tensor, np.ndarray]:
    """Load the Neuro brain map embedding from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "ngram_embeddings.pt"
    )
    latent = torch.load(
        latent_path,
        weights_only=False,
        map_location="cpu"
    )
    return latent

@lru_cache(maxsize=1)
def _load_ngram() -> Tuple[torch.Tensor, np.ndarray]:
    """Load the Neuro brain map embedding from HuggingFace."""
    labels_path = _download_from_hf(
        "neurovlm/embedded_text",
        "ngram_labels.npy"
    )
    labels = np.load(
        labels_path
    )
    return labels


@lru_cache(maxsize=1)
def _load_kg_mesh_dataset() -> pd.DataFrame:
    """Load the KG-MeSH term/definition DataFrame from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/embedded_text",
        "kg_mesh.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_mesh_kg_nodes() -> pd.DataFrame:
    """Load MeSH KG node metadata, including node_type, from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/mesh_kg",
        "mesh_kg_nodes.parquet"
    )
    return pd.read_parquet(parquet_path)


@lru_cache(maxsize=1)
def _load_mesh_kg_descriptors() -> pd.DataFrame:
    """Load MeSH descriptor metadata from HuggingFace."""
    parquet_path = _download_from_hf(
        "neurovlm/mesh_kg",
        "mesh_descriptors.parquet"
    )
    return pd.read_parquet(parquet_path)


def _normalize_mesh_term(text: str) -> str:
    text = str(text or "").lower().split("/")[0]
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def _brain_rankable_mesh_terms(include_molecular: bool = False) -> set[str]:
    allowed = {
        "disorder",
        "anatomical_region",
        "biological_process",
        "cognitive_construct",
    }
    if include_molecular:
        allowed.add("molecular")
    nodes = _load_mesh_kg_nodes()
    node_type_col = "node_type" if "node_type" in nodes.columns else None
    name_col = "name" if "name" in nodes.columns else "term"
    if node_type_col is None or name_col not in nodes.columns:
        raise ValueError("mesh_kg_nodes.parquet must include name and node_type columns.")
    keep = nodes[nodes[node_type_col].isin(allowed)]
    return {_normalize_mesh_term(x) for x in keep[name_col].dropna().astype(str)}


@lru_cache(maxsize=2)
def _load_kg_mesh_brain_rankable_dataset(include_molecular: bool = False) -> pd.DataFrame:
    """Load KG-MeSH terms filtered to brain-rankable node types."""
    df = _load_kg_mesh_dataset().copy()
    allowed_terms = _brain_rankable_mesh_terms(include_molecular=include_molecular)
    keep = df["term"].map(_normalize_mesh_term).isin(allowed_terms)
    return df.loc[keep].reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_latent_kg_mesh() -> Tuple[torch.Tensor, np.ndarray]:
    """Load KG-MeSH SPECTER2 embeddings from HuggingFace."""
    latent_path = _download_from_hf(
        "neurovlm/embedded_text",
        "latent_kg_mesh.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    latent = latent_payload["latent"]
    terms  = np.asarray(latent_payload["term"])
    return latent, terms


@lru_cache(maxsize=2)
def _load_latent_kg_mesh_brain_rankable(include_molecular: bool = False) -> Tuple[torch.Tensor, np.ndarray]:
    """Load KG-MeSH embeddings filtered to brain-rankable node types."""
    latent, terms = _load_latent_kg_mesh()
    allowed_terms = _brain_rankable_mesh_terms(include_molecular=include_molecular)
    mask = np.asarray([_normalize_mesh_term(term) in allowed_terms for term in terms], dtype=bool)
    return latent[mask], terms[mask]


@lru_cache(maxsize=1)
def _load_llm_neuro_terms_dataset() -> pd.DataFrame:
    """Load novel LLM-extracted neuroscience terms."""
    local_path = Path("artifacts/llm_extracted_neuro_terms/llm_neuro_terms.parquet")
    parquet_path = str(local_path) if local_path.exists() else _download_from_hf(
        "neurovlm/neuro_image_papers",
        "llm_neuro_terms.parquet"
    )
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_latent_llm_neuro_terms() -> Tuple[torch.Tensor, np.ndarray]:
    """Load SPECTER2 embeddings for novel LLM-extracted neuroscience terms."""
    local_path = Path("artifacts/llm_extracted_neuro_terms/latent_llm_neuro_terms.pt")
    latent_path = str(local_path) if local_path.exists() else _download_from_hf(
        "neurovlm/embedded_text",
        "latent_llm_neuro_terms.pt"
    )
    latent_payload = torch.load(
        latent_path,
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    latent = latent_payload["latent"]
    terms = np.asarray(latent_payload["term"])
    return latent, terms
