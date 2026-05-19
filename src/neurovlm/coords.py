"""Process coordinate tables."""

import pandas as pd
import numpy as np
import torch
from nimare.meta.kernel import ALEKernel
from nimare.meta.utils import get_ale_kernel

from .progress import select_tqdm
from .retrieval_resources import _load_masker


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

    # Load mask from HuggingFace
    masker = _load_masker()
    mask_img = masker.mask_img_

    # Max value of convolutional kernel
    _, kernel = get_ale_kernel(masker.mask_img, fwhm=fwhm)
    kmax = kernel.max()

    # Kernel smoothing with nimare
    kernel = ALEKernel(fwhm=fwhm)
    neuro_vectors = torch.from_numpy(
        kernel.transform(df, masker=masker, return_type="array")
    ).float() / kmax
    neuro_vectors = torch.clamp_max(neuro_vectors, 1) # clamp to (0, 1)
    # neuro_vectors = neuro_vectors / neuro_vectors.max(dim=1, keepdim=True).values

    return neuro_vectors


def build_difumo_projection_matrix(difumo_masker, masker, device="cuda", dtype=torch.float32):
    """
    Builds P (n_vox, k) in mask space, and s = sum(P, dim=0).
    Requires difumo_masker and masker to already be aligned (same mask/space),
    and neuro_vectors to be in masker.transform() voxel order.
    """
    # Ensure difumo masker has resampled maps ready
    difumo_masker.fit()

    # Get maps image that difumo_masker is using after fit/resampling
    maps_img = difumo_masker.maps_img_  # nibabel-like img
    maps_4d = maps_img.get_fdata(dtype=np.float32)  # (X,Y,Z,k)

    # Mask in the same space/order as neuro_vectors (masker.transform output)
    mask_img = masker.mask_img_ if hasattr(masker, "mask_img_") else masker.mask_img
    mask = mask_img.get_fdata().astype(bool)

    # Flatten maps into (n_vox, k) in mask voxel order
    # NOTE: nilearn's masker uses C-order flattening; this matches get_fdata()[mask]
    P = maps_4d[mask, :]  # (n_vox, k)

    P = torch.as_tensor(P, device=device, dtype=dtype).contiguous()
    s = P.sum(dim=0).clamp_min(1e-12)  # avoid divide-by-zero

    return P, s

@torch.no_grad()
def difumo_project_voxels(X, P, s):
    """
    X: (..., n_vox) tensor (single vector or batch)
    P: (n_vox, k)
    s: (k,)
    Returns: (..., n_vox) reconstructed/projection result
    """
    # coefficients: (..., k)
    C = (X @ P) / s
    # reconstruction: (..., n_vox)
    X_hat = C @ P.T
    return X_hat
