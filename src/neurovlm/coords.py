"""Process coordinate tables."""

from typing import Optional
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from nilearn import image

from .progress import select_tqdm
from .data import fetch_data

def coords_to_vectors(
    df_coords: pd.DataFrame,
    df_pubs: pd.DataFrame,
    max_n_coords: Optional[int]=100,
    fwhm: Optional[float]=9.0,
    verbose: Optional[bool]=True,
) -> pd.DataFrame:
    """Convert a dataframe of coordinates to neuro-tensors.

    Parameters
    ----------
    df_coords : pandas.DataFrame
        Coordinates fetched with pubget, e.g. a df loaded from:
            subset_articlesWithCoords_extractedData/coordinates.csv
    df_pubs : pandas.DataFrame
        Metadata per publication.
    max_n_coords : int, optional, default: 100
        Drop publications with more than max_n_coords reported.
    fwhm : float, optional, default: 9.0
        Size of the Gaussian smoothing kernel.
    verbose : bool, optional, default: True
        Prints progress.

    Returns
    -------
    neuro_vectors : 2d float torch.tensor
        Each row is a publication's neuro tensor, e.g. downsampled
        MNI, masked, and flattened.
    df_pubs : pandas.DataFrame
        Publication info. Each row corresponds the same neuro_vector index,
        e.g. df_pubs.iloc[i] corresponds to neuro_vectors[i].
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

    # Merge PMCIDs and PMIDs
    pmids = df_pubs['pmid']

    # Import tqdm
    tqdm = select_tqdm()
    iterable = pmids
    if verbose:
        iterable =  tqdm(iterable, total=len(pmids))

    # Iterate over dataframe rows
    neuro_vectors = []
    row_inds = []
    for i in iterable:

        coords = np.array(df_coords[df_coords["pmid"] == i][['x', 'y', 'z']])

        # Exclusion criteria
        if len(coords) == 0:
            # Skip papers with no reported coordinates
            continue

        x, y, z = coords.T
        in_bounds = np.all(x > bounds[0][0]) & np.all(x < bounds[1][0])
        in_bounds = in_bounds & np.all(y > bounds[0][1]) & np.all(y < bounds[1][1])
        in_bounds = in_bounds & np.all(z > bounds[0][2]) & np.all(z < bounds[1][2])

        if not in_bounds or len(coords) > max_n_coords:
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
        img = image.smooth_img(peaks_img, fwhm=fwhm)

        # Mask
        neuro_vec = img.get_fdata()[mask]
        if neuro_vec.sum() == 0:
            # Skip papers with out-of-bounds coordinates
            continue

        neuro_vectors.append(torch.from_numpy(neuro_vec))
        row_inds.append(int(df_pubs[df_pubs['pmid'] == i].index[0]))

    # Stack vectors
    neuro_vectors = torch.squeeze(torch.stack(neuro_vectors)).float()

    df_pubs = df_pubs.iloc[row_inds].reset_index(drop=True)

    return neuro_vectors, df_pubs
