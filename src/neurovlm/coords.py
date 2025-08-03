"""Process coordinate tables."""

from typing import Optional
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from nilearn import maskers
from nimare.meta.kernel import ALEKernel
from nimare.meta.utils import get_ale_kernel

from .progress import select_tqdm
from .data import fetch_data


def coords_to_vectors(df_coords: pd.DataFrame, fwhm: int) -> pd.DataFrame:
    """Convert a dataframe of coordinates to neuro-tensors using nimare.

    Parameters
    ----------
    df_coords : pandas.DataFrame
        Coordinates fetched with pubget, e.g. a df loaded from:
            subset_articlesWithCoords_extractedData/coordinates.csv

    Returns
    -------
    neuro_vectors : 2d float32 torch.tensor
        Each row is a publication's neuro tensor, e.g. downsampled
        MNI, masked, and flattened. Sorted on ascending "pmid".
    """

    # Rename columns for nimare
    df = df_coords.copy()[['pmid', 'x', 'y', 'z']]
    df.rename(columns={"pmid": "id"}, inplace=True)

    # Load neuroquery mask
    data_dir = fetch_data(files=[])
    mask_arrays = np.load(f"{data_dir}/mask.npz", allow_pickle=True)
    mask_img = nib.Nifti1Image(mask_arrays["mask"].astype(float),  mask_arrays["affine"])
    masker = maskers.NiftiMasker(mask_img=mask_img, dtype=np.float32).fit()

    # Max value of convolutional kernel
    _, kernel = get_ale_kernel(masker.mask_img, fwhm=fwhm)
    kmax = kernel.max()

    # Kernel smoothing with nimare
    kernel = ALEKernel(fwhm=fwhm)
    neuro_vectors = torch.from_numpy(
        kernel.transform(df, masker=masker, return_type="array")
    ).float() / kmax
    neuro_vectors = torch.clamp_max(neuro_vectors, 1) # clamp to (0, 1)

    return neuro_vectors
