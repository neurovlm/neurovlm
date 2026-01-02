"""Shared loaders for NeuroVLM retrieval tasks.

These helpers centralize cached access to publication metadata,
latent embeddings, and projection heads so they can be shared across
brain- and text-driven retrieval workflows.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple
import gzip, pickle

import numpy as np
import pandas as pd
import torch

import nibabel as nib
from nilearn import maskers

from neurovlm.data import get_data_dir
from neurovlm.models import Specter

__all__ = [
    "_load_dataframe",
    "_load_neuro_wiki",
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
    "_proj_head_mse_sparse_adhoc",
    "_proj_head_text_infonce",
]


@lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    """Load the publications DataFrame with a parquet engine fallback."""
    data_dir = get_data_dir()
    parquet_path = data_dir / "publications_more.parquet"
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover - depends on local engines
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_neuro_wiki() -> pd.DataFrame:
    """Load the NeuroWiki DataFrame with a parquet engine fallback."""
    data_dir = get_data_dir()
    parquet_path = data_dir / "neurowiki_with_ids.parquet"
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover - depends on local engines
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")

@lru_cache(maxsize=1)
def _load_cogatlas_dataset() -> pd.DataFrame:
    """Load the CogAtlas DataFrame with a parquet engine fallback."""
    data_dir = get_data_dir()
    parquet_path = data_dir / "cogatlas.parquet"
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover - depends on local engines
        print(f"pyarrow failed: {exc}, trying fastparquet...")
        return pd.read_parquet(parquet_path, engine="fastparquet")

def _load_cogatlas_task_dataset(filtered = False) -> pd.DataFrame:
    """Load the CogAtlas DataFrame with a parquet engine fallback."""
    data_dir = get_data_dir()
    if filtered:
        parquet_path = data_dir / "cogatlas_task_filtered.parquet"
        try:
            return pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:  # pragma: no cover - depends on local engines
            print(f"pyarrow failed: {exc}, trying fastparquet...")
            return pd.read_parquet(parquet_path, engine="fastparquet")
    else:
        parquet_path = data_dir / "cogatlas_task.parquet"
        try:
            return pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:  # pragma: no cover - depends on local engines
            print(f"pyarrow failed: {exc}, trying fastparquet...")
            return pd.read_parquet(parquet_path, engine="fastparquet")

def _load_cogatlas_disorder_dataset(filtered = False) -> pd.DataFrame:
    """Load the CogAtlas DataFrame with a parquet engine fallback."""
    data_dir = get_data_dir()
    if filtered:
        parquet_path = data_dir / "cogatlas_disorder_filtered.parquet"
        try:
            return pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:  # pragma: no cover - depends on local engines
            print(f"pyarrow failed: {exc}, trying fastparquet...")
            return pd.read_parquet(parquet_path, engine="fastparquet")
    else:
        parquet_path = data_dir / "cogatlas_disorder.parquet"
        try:
            return pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:  # pragma: no cover - depends on local engines
            print(f"pyarrow failed: {exc}, trying fastparquet...")
            return pd.read_parquet(parquet_path, engine="fastparquet")


@lru_cache(maxsize=1)
def _load_specter() -> Specter:
    """Construct and cache a Specter encoder."""
    return Specter()


@lru_cache(maxsize=1)
def _load_latent_text() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent text embeddings and associated PubMed IDs."""
    data_dir = get_data_dir()
    latent_payload = torch.load(
        data_dir / "latent_specter2_adhoc.pt",
        weights_only=False,
    )

    latent = latent_payload["latent"]
    latent_pmid = np.asarray(latent_payload["pmid"])
    return latent, latent_pmid


@lru_cache(maxsize=1)
def _load_latent_wiki() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent wiki embeddings and their IDs."""
    data_dir = get_data_dir()
    latent_payload = torch.load(
        data_dir / "latent_specter_wiki.pt",
        weights_only=False,
    )

    latent = latent_payload["latent"]
    latent_id = np.asarray(latent_payload["id"])
    return latent, latent_id


@lru_cache(maxsize=1)
def _load_latent_cogatlas() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent cognitive atlas embeddings and their term IDs."""
    data_dir = get_data_dir()
    latent_payload = torch.load(
        data_dir / "latent_cogatlas.pt",
        weights_only=False,
    )

    latent = latent_payload["latent"]
    latent_terms = np.asarray(latent_payload["term"])
    return latent, latent_terms

@lru_cache(maxsize=1)
def _load_latent_cogatlas_disorder() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent cognitive atlas embeddings and their term IDs."""
    data_dir = get_data_dir()
    latent_payload = torch.load(
        data_dir / "latent_cogatlas_disorder.pt",
        weights_only=False,
    )

    latent = latent_payload["latent"]
    latent_terms = np.asarray(latent_payload["term"])
    return latent, latent_terms

@lru_cache(maxsize=1)
def _load_latent_cogatlas_task() -> Tuple[torch.Tensor, np.ndarray]:
    """Load unit-normalized latent cognitive atlas embeddings and their term IDs."""
    data_dir = get_data_dir()
    latent_payload = torch.load(
        data_dir / "latent_cogatlas_task.pt",
        weights_only=False,
    )

    latent = latent_payload["latent"]
    latent_terms = np.asarray(latent_payload["term"])
    return latent, latent_terms


@lru_cache(maxsize=1)
def _load_autoencoder() -> torch.nn.Module:
    """Load and return the text encoder model."""
    data_dir = get_data_dir()
    encoder = torch.load(data_dir / "autoencoder.pt", weights_only=False, map_location="cpu")
    return encoder


@lru_cache(maxsize=1)
def _load_masker() -> nib.Nifti1Image:
    """Load mask."""
    data_dir = get_data_dir()
    mask_arrays = np.load(data_dir / "mask.npz", allow_pickle=True)
    mask_img = nib.Nifti1Image(mask_arrays["mask"].astype(float),  mask_arrays["affine"])
    masker = maskers.NiftiMasker(mask_img=mask_img, dtype=np.float32).fit()
    return masker


@lru_cache(maxsize=1)
def _load_networks() -> torch.nn.Module:
    """Load network atlases."""
    with gzip.open(get_data_dir() / "networks_arrays.pkl.gz", "rb") as f:
        networks = pickle.load(f)

    network_imgs = []
    for k in networks.keys():
        for a in networks[k].keys():
            network_imgs.append((k, a, nib.Nifti1Image(networks[k][a]["array"], affine=networks[k][a]["affine"])))

    return networks


@lru_cache(maxsize=1)
def _proj_head_image_infonce() -> torch.nn.Module:
    """Load and return the image projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_image_infonce.pt", weights_only=False, map_location="cpu")
    return proj_head


@lru_cache(maxsize=1)
def _proj_head_mse_sparse_adhoc() -> torch.nn.Module:
    """Load and return the MSE projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_mse_sparse_adhoc", weights_only=False, map_location="cpu")
    return proj_head


@lru_cache(maxsize=1)
def _proj_head_text_infonce() -> torch.nn.Module:
    """Load and return the text projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_text_infonce.pt", weights_only=False, map_location="cpu")
    return proj_head
