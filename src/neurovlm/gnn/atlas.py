"""DiFuMo atlas loading and brain-image-to-coefficient projection.

Phase 1, steps 1–2: build the P matrix (512 components × V masked voxels)
and project raw brain flatmaps onto it to obtain per-paper DiFuMo coefficient
vectors — the node features for the GAT.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Optional

# Number of DiFuMo components used throughout the GNN experiment
DIFUMO_DIM = 512


@lru_cache(maxsize=1)
def load_difumo_components(
    dimension: int = DIFUMO_DIM,
    resolution_mm: int = 2,
) -> np.ndarray:
    """Return the DiFuMo component matrix masked to the NeuroVLM brain mask.

    Downloads the DiFuMo atlas via nilearn (cached to ``~/.nilearn``) and
    applies the same NiftiMasker used by NeuroVLM so the voxel ordering
    matches the 28,542-dim flatmaps produced by NeuroVLM's preprocessing
    pipeline.

    Parameters
    ----------
    dimension:
        Number of DiFuMo components. Must be one of 64, 128, 256, 512, 1024.
    resolution_mm:
        Atlas resolution. 2 mm is recommended; 3 mm is faster.

    Returns
    -------
    P : np.ndarray, shape (dimension, n_masked_voxels)
        Each row is a soft component map evaluated at the masked voxels.
        ``n_masked_voxels`` matches ``BRAIN_FLAT_DIM`` (28,542) from
        ``neurovlm.core``.
    """
    from nilearn import datasets as nilearn_datasets
    from neurovlm.data import load_masker

    atlas = nilearn_datasets.fetch_atlas_difumo(
        dimension=dimension, resolution_mm=resolution_mm
    )

    masker = load_masker()

    # atlas.maps is a 4D NIfTI: (x, y, z, dimension).
    # NiftiMasker.transform treats each 3-D volume as a sample → returns
    # (dimension, n_masked_voxels).
    P = masker.transform(atlas.maps)  # (dimension, n_voxels)
    return P.astype(np.float32)


def compute_difumo_coefficients(
    brain_flat: np.ndarray,
    P: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Project raw brain flatmaps onto DiFuMo components.

    Implements the inner-product projection
    ``s_i = P @ y_i``
    giving a 512-dim coefficient vector for each paper.

    Parameters
    ----------
    brain_flat:
        Raw brain flatmaps, shape ``(n_papers, n_masked_voxels)``.
        Values should be float32 activations (e.g. z-scores or thresholded
        stat maps) as produced by NeuroVLM's NiftiMasker step.
    P:
        Component matrix of shape ``(n_components, n_masked_voxels)``.
        Loaded automatically via :func:`load_difumo_components` when *None*.
    normalize:
        If *True* (default), z-score the coefficient matrix column-wise
        (per component, across papers) to zero mean and unit variance.
        The GAT input benefits strongly from this normalization — without it
        recall@k can stay near random for the first 50 epochs.

    Returns
    -------
    coeffs : np.ndarray, shape (n_papers, n_components)
        DiFuMo coefficient vectors; one row per paper, one column per
        component.
    """
    if P is None:
        P = load_difumo_components()

    # (N, V) @ (V, K) = (N, K)
    coeffs = brain_flat @ P.T

    if normalize:
        mu = coeffs.mean(axis=0, keepdims=True)
        sigma = coeffs.std(axis=0, keepdims=True) + 1e-8
        coeffs = (coeffs - mu) / sigma

    return coeffs.astype(np.float32)


def get_component_centroids(dimension: int = DIFUMO_DIM, resolution_mm: int = 2) -> np.ndarray:
    """Return MNI centroids for each DiFuMo component.

    Useful as additional node features (x, y, z in mm) and for computing
    spatial-distance edge features.

    Returns
    -------
    centroids : np.ndarray, shape (dimension, 3)
        MNI-space (x, y, z) centroid of each component's probability map.
    """
    from nilearn import datasets as nilearn_datasets, image
    import nibabel as nib

    atlas = nilearn_datasets.fetch_atlas_difumo(
        dimension=dimension, resolution_mm=resolution_mm
    )
    img = nib.load(atlas.maps)
    data = img.get_fdata()           # (x, y, z, K)
    affine = img.affine

    K = data.shape[-1]
    centroids = np.zeros((K, 3), dtype=np.float32)
    for k in range(K):
        vol = data[..., k]
        vol = np.clip(vol, 0, None)
        total = vol.sum()
        if total < 1e-12:
            continue
        # Voxel-space weighted centroid
        xs, ys, zs = np.mgrid[
            : data.shape[0], : data.shape[1], : data.shape[2]
        ]
        cx = (vol * xs).sum() / total
        cy = (vol * ys).sum() / total
        cz = (vol * zs).sum() / total
        # Convert to MNI mm
        vox = np.array([cx, cy, cz, 1.0])
        mni = affine @ vox
        centroids[k] = mni[:3]

    return centroids
