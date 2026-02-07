"""Shared loaders for NeuroVLM retrieval tasks.

These helpers centralize cached access to publication metadata,
latent embeddings, and projection heads so they can be shared across
brain- and text-driven retrieval workflows.

Files are now loaded from HuggingFace repositories under the neurovlm organization.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple
import gzip, pickle

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
    "_load_neuro_wiki",
    "_load_pubmed_coordinates",
    "_load_latent_neuro",
    "_load_cogatlas_dataset",
    "_load_cogatlas_task_dataset",
    "_load_cogatlas_disorder_dataset",
    "_load_specter",
    "_load_latent_text",
    "_load_latent_wiki",
    "_load_latent_cogatlas",
    "_load_latent_cogatlas_disorder",
    "_load_latent_cogatlas_task",
    "_load_autoencoder",
    "_load_masker",
    "_load_networks",
    "_proj_head_image_infonce",
    "_proj_head_text_mse",
    "_proj_head_text_infonce",
]


def _download_from_hf(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Download a file from HuggingFace and return the local path.

    Args:
        repo_id: Repository ID (e.g., 'neurovlm/embedded_text')
        filename: Name of the file to download
        repo_type: Type of repository - "dataset" or "model" (default: "dataset")
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type
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
    )

    latent = latent_payload["latent"]
    latent_terms = np.asarray(latent_payload["term"])
    return latent, latent_terms


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
