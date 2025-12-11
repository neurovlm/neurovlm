"""Shared loaders for NeuroVLM retrieval tasks.

These helpers centralize cached access to publication metadata,
latent embeddings, and projection heads so they can be shared across
brain- and text-driven retrieval workflows.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

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
    "_load_specter",
    "_load_latent_text",
    "_load_latent_wiki",
    "_load_autoencoder",
    "_load_masker",
    "_proj_head_image_infonce",
    "_proj_head_mse_adhoc",
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
def _load_autoencoder() -> torch.nn.Module:
    """Load and return the text encoder model."""
    data_dir = get_data_dir()
    encoder = torch.load(data_dir / "autoencoder_sparse.pt", weights_only=False, map_location="cpu")
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
def _proj_head_image_infonce() -> torch.nn.Module:
    """Load and return the image projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_image_infonce.pt", weights_only=False, map_location="cpu")
    return proj_head


@lru_cache(maxsize=1)
def _proj_head_mse_adhoc() -> torch.nn.Module:
    """Load and return the MSE projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_mse_sparse_adhoc.pt", weights_only=False, map_location="cpu")
    return proj_head


@lru_cache(maxsize=1)
def _proj_head_text_infonce() -> torch.nn.Module:
    """Load and return the text projection head."""
    data_dir = get_data_dir()
    proj_head = torch.load(data_dir / "proj_head_text_infonce.pt", weights_only=False, map_location="cpu")
    return proj_head
