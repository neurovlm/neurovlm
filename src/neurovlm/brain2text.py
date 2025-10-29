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

import warnings
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.image import resample_img
from nilearn.maskers import NiftiMasker

from neurovlm.data import get_data_dir
from neurovlm.data import fetch_data
from neurovlm.models import NeuroAutoEncoder
from neurovlm.train import which_device
from neurovlm.llm_summary import search_papers, generate_response
data_dir = get_data_dir()

warnings.filterwarnings("ignore")


def load_metadata(data_dir: Path | str | None = data_dir) -> dict[str, pd.DataFrame]:
    """
    Load publication metadata used to map latent vectors to textual data.

    Parameters
    ----------
    data_dir:
        Optional path returned by :func:`fetch_data`. When omitted the data
        directory is fetched (or validated) automatically.
    """
    root = Path(fetch_data() if data_dir is None else data_dir)
    df_pubs = pd.read_parquet(root / "publications.parquet")
    df_coords = pd.read_parquet(root / "coordinates.parquet")
    return {"publications": df_pubs, "coordinates": df_coords}


def load_models(
    autoencoder_path: Path | str = f"{data_dir} / autoencoder_sparse.pt",
    latent_paper_path: Path | str = f"{data_dir} / latent_text_aligned.pt",
    latent_wiki_path: Path | str = f"{data_dir} / latent_specter_wiki_aligned.pt",
    proj_head_path: Path | str = f"{data_dir} / proj_head_mse_sparse_adhoc.pt",
    device: torch.device | None = None,
) -> dict[str, torch.nn.Module | torch.Tensor]:
    """
    Load the trained NeuroVLM components required for the brain-to-text task.

    Returns a dictionary containing encoder/decoder models and cached latent
    representations. All models are moved to the requested ``device``.
    """
    target_device = device if device is not None else which_device()

    autoencoder = torch.load(
        autoencoder_path, weights_only=False
    ).to(target_device)
   
    proj_head = torch.load(
        proj_head_path, weights_only=False
    ).to(target_device)

    latent_paper = torch.load(
        latent_paper_path, weights_only=False
    ).to(target_device)
    latent_paper = latent_paper['latent']
    latent_pmid = latent_paper['pmid']

    latent_wiki = torch.load(
        latent_wiki_path, weights_only=False
    ).to(target_device)
    latent_wiki = latent_wiki['latent']
    latent_wikiid = latent_wiki['id']

    return {
        "autoencoder": autoencoder,
        "proj_head": proj_head,
        "latent_paper": latent_paper,
        "latent_paper_ids": latent_pmid,
        "latent_wiki": latent_wiki,
        "latent_wiki_ids": latent_wikiid,
    }


def _load_mask_bundle(
    data_dir: Path | str | None = None,
) -> tuple[dict[str, Any], nib.Nifti1Image, NiftiMasker]:
    """Return mask arrays, image, and fitted masker."""
    root = Path(fetch_data() if data_dir is None else data_dir)
    mask_arrays = np.load(root / "mask.npz", allow_pickle=True)
    mask_img = nib.Nifti1Image(
        mask_arrays["mask"].astype(float),
        mask_arrays["affine"],
    )
    masker = NiftiMasker(mask_img=mask_img, dtype=np.float32).fit()
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


def resmaple_nifti(
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


def resmaple_array_nifti(
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



def run_brain2text(query_vector: torch.Tensor):
    """
    query_vector: torch.Tensor
        An already encoded brain-derived vector."""
    output = generate_response(query_vector, top_k=5)
    return output


__all__ = [
    "load_metadata",
    "load_models",
    "resmaple_nifti",
    "resmaple_array_nifti",
    "run_brain2text",
]
