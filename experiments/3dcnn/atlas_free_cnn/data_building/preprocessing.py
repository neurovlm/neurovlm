"""NIfTI preprocessing helpers for common MNI-space map rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def load_target_mni152_2mm():
    """Return a Nilearn MNI152 2mm target image."""

    from nilearn.datasets import load_mni152_template

    return load_mni152_template(resolution=2)


def clean_array(data: np.ndarray, *, clamp_nonnegative: bool = False) -> np.ndarray:
    arr = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_nonnegative:
        arr = np.clip(arr, 0.0, None)
    return arr.astype(np.float32)


def resample_to_target(img, target_img=None, *, binary: bool = False, clamp_nonnegative: bool = False):
    """Resample an image to the target grid and clean numeric edge cases."""

    from nilearn.image import resample_to_img

    if target_img is None:
        target_img = load_target_mni152_2mm()
    interpolation = "nearest" if binary else "continuous"
    out = resample_to_img(img, target_img, interpolation=interpolation, force_resample=True, copy_header=True)
    data = clean_array(out.get_fdata(), clamp_nonnegative=clamp_nonnegative)
    if binary:
        data = (data > 0.5).astype(np.float32)
    return nib.Nifti1Image(data, out.affine, out.header)


def save_nifti(img, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def nifti_metadata(path_or_img) -> dict[str, Any]:
    img = nib.load(str(path_or_img)) if isinstance(path_or_img, (str, Path)) else path_or_img
    affine = np.asarray(img.affine, dtype=float)
    zooms = list(map(float, img.header.get_zooms()[:3]))
    return {
        "affine": affine.tolist(),
        "resolution": zooms,
        "shape": list(map(int, img.shape[:3])),
    }


def volume_tensor_to_nifti(volume, affine, path: str | Path) -> Path:
    import torch

    arr = volume.detach().cpu() if torch.is_tensor(volume) else volume
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    img = nib.Nifti1Image(clean_array(arr, clamp_nonnegative=True), np.asarray(affine, dtype=float))
    return save_nifti(img, path)


def tensor_from_nifti(path: str | Path, *, dtype="float32"):
    import torch

    arr = clean_array(nib.load(str(path)).get_fdata())
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor.to(getattr(torch, dtype))

