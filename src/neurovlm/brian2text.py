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

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.image import load_img, resample_to_img
from nilearn.maskers import NiftiMasker

from neuroquery import datasets as neuroquery_datasets
from neurovlm.data import fetch_data
from neurovlm.models import NeuroAutoEncoder, TextAligner
from neurovlm.train import which_device
from neurovlm.llm_summary import search_papers, summarize_papers

warnings.filterwarnings("ignore")


def load_metadata(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
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
    autoencoder_path: Path | str = Path("docs/autoencoder.pt"),
    decoder_path: Path | str = Path("docs/decoder_half.pt"),
    aligner_path: Path | str = Path("specter/aligner.pt"),
    aligner_half_path: Path | str = Path("specter/aligner_half.pt"),
    latent_text_path: Path | str = Path("specter/latent_text.pt"),
    aligned_text_path: Path | str = Path("specter/aligned_text.pt"),
    device: torch.device | None = None,
) -> dict[str, torch.nn.Module | torch.Tensor]:
    """
    Load the trained NeuroVLM components required for the brain-to-text task.

    Returns a dictionary containing encoder/decoder models and cached latent
    representations. All models are moved to the requested ``device``.
    """
    target_device = device if device is not None else which_device()

    neuro_encoder: NeuroAutoEncoder = torch.load(
        autoencoder_path, weights_only=False
    ).to(target_device)
    neuro_encoder.eval()

    neuro_decoder: NeuroAutoEncoder = torch.load(
        decoder_path, weights_only=False
    ).to(target_device)
    neuro_decoder.eval()

    text_aligner: TextAligner = torch.load(
        aligner_path, weights_only=False
    ).to(target_device)
    text_aligner.eval()

    text_aligner_half: TextAligner = torch.load(
        aligner_half_path, weights_only=False
    ).to(target_device)
    text_aligner_half.eval()

    latent_text = torch.load(latent_text_path, weights_only=False).to("cpu")
    aligned_text = torch.load(aligned_text_path, weights_only=False).to("cpu")

    return {
        "neuro_encoder": neuro_encoder,
        "neuro_decoder": neuro_decoder,
        "text_aligner": text_aligner,
        "text_aligner_half": text_aligner_half,
        "latent_text": latent_text,
        "aligned_text": aligned_text,
    }


def transform_nifti_to_2d(nifti_img: nib.Nifti1Image, mask_img_path: Path | str | None = None) -> torch.Tensor:
    """
    Flatten a NIfTI image to a masked 2D vector.
    """
    region_img = (nifti_img.get_fdata()).astype(float)
    region_nii = nib.Nifti1Image(region_img.astype(np.int32), nifti_img.affine, dtype=np.int32)

    if mask_img_path is None:
        mask_img = load_img(
            f"{neuroquery_datasets.fetch_neuroquery_model()}/mask_img.nii.gz",
            dtype=np.float32,
        )
    else:
        mask_img = load_img(mask_img_path, dtype=np.float32)

    masker = NiftiMasker(mask_img=mask_img, dtype=np.float32).fit()

    region_nii_resampled = resample_to_img(
        region_nii,
        mask_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=False,
    )

    flattened = masker.transform(region_nii_resampled)
    flattened[flattened > 0] = 1
    out_tensor = torch.tensor(flattened).squeeze(0)
    return out_tensor



def run_brain_to_text_pipeline(query_vector: torch.Tensor):
    output = summarize_papers(query_vector,top_k=10)
    return output


__all__ = [
    "load_metadata",
    "load_models",
    "transform_nifti_to_2d",
    "run_brain_to_text_pipeline",
]
