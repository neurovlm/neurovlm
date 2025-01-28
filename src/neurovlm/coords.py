"""Process coordinate tables."""

from typing import Optional
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from nilearn import image

from .data import fetch_data

def coords_to_vectors(df_coords: pd.DataFrame, verbose: Optional[bool]=True):
    """Convert a dataframe of coordinates to neuro-tensors.

    Parameters
    ----------
    df_coords : pandas.DataFrame
        Coordinates fetched with pubget, e.g. a df loaded from:
            subset_articlesWithCoords_extractedData/coordinates.csv
    verbose : bool, optional, default: True
        Prints progress.
    """
    # Load mask
    save_dir = fetch_data()
    mask_arrays = np.load(f"{save_dir}/mask.npz", allow_pickle=True)
    mask = mask_arrays["mask"]
    affine = mask_arrays["affine"]
    affine_inv = np.linalg.pinv(affine)
    mask_img = nib.Nifti1Image(mask.astype(float), affine)

    # MNI coordinate bounds
    bounds = [
        (-90.0, -126.0, -72.0),
        (698.0, 806.0, 684.0)
    ]

    # Compute the neuro vectors
    neuro_vectors = []
    selected_pmcids = []
    pmcids = df_coords['pmcid'].unique()

    # Import tqdm
    tqdm = select_tqdm()

    iterable = range(len(pmcids))
    if verbose:
        iterable =  tqdm(iterable, total=len(pmcids))

    # Iterate over dataframe rows
    for i in iterable:

        coords = np.array(df_coords[df_coords['pmcid'] == pmcids[i]][['x', 'y', 'z']])

        # Exclusion criteria
        if len(coords) == 0:
            # Skip papers with no reported coordinates
            continue

        x, y, z = coords.T
        in_bounds = np.all(x > bounds[0][0]) & np.all(x < bounds[1][0])
        in_bounds = in_bounds & np.all(y > bounds[0][1]) & np.all(x < bounds[1][1])
        in_bounds = in_bounds & np.all(z > bounds[0][2]) & np.all(z < bounds[1][2])

        if not in_bounds or len(coords) > 100:
            # Drop coords if either:
            #   a) Coordinates lie out-of-bounds in MNI space
            #   b) Study reported over 100 coordinates.
            #      We want relatively localized results only.
            continue

        # Coords to image, adapted from neuroquery
        coords = np.atleast_2d(coords)
        coords = np.hstack([coords, np.ones((len(coords), 1))])

        voxels = affine_inv.dot(coords.T)[:-1].T
        voxels = voxels[(voxels >= 0).all(axis=1)]
        voxels = voxels[(voxels < mask_img.shape[:3]).all(axis=1)]
        voxels = np.floor(voxels).astype(int)

        peaks = np.zeros(mask_img.shape)
        np.add.at(peaks, tuple(voxels.T), 1.0)
        peaks_img = image.new_img_like(mask_img, peaks)

        # Smooth
        img = image.smooth_img(peaks_img, fwhm=9.0)

        # Mask
        neuro_vec = img.get_fdata()[mask]
        if neuro_vec.sum() == 0:
            # Skip papers with out-of-bounds coordinates
            #   out-of-bounds contracted with neuroquery's masker
            continue

        neuro_vectors.append(torch.from_numpy(neuro_vec))
        selected_pmcids.append(pmcids[i])

    neuro_vectors = torch.squeeze(torch.stack(neuro_vectors))

    return neuro_vectors, selected_pmcids


def select_tqdm():
    # Check if running in IPython
    from IPython import get_ipython
    ipython = get_ipython()

    if ipython is None:
        # Not in IPython, assume a regular script
        from tqdm import tqdm
    elif "IPKernelApp" in ipython.config:
        # In Jupyter Notebook or JupyterLab
        from tqdm.notebook import tqdm
    else:
        # In IPython shell
        from tqdm import tqdm
    return tqdm