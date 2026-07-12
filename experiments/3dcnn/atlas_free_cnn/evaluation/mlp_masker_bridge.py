"""Bridge between atlas-free CNN volumes and the MLP's masker-flat space.

The atlas-free CNN's packed ``(36, 45, 38)`` volumes are built by cropping
the exact same MNI152 4mm grid the MLP ``NeuroAutoEncoder`` masker uses to
its brain bounding box (see
``atlas_free_cnn.training.ale_dataset._build_difumo_compatible_volumes``, which performs
the forward direction: MLP flatmap -> scatter into the masker's ``(46, 55,
46)`` grid -> crop to the brain bounding box). Both the crop box and the
affine are derived directly from the packaged MLP mask
(``neurovlm.retrieval_resources._load_masker``), and at 4mm resolution no
resampling is needed -- confirmed by checking that the crop shape is exactly
``(36, 45, 38)`` (the atlas-free CNN's ``TARGET_SHAPE``) and that the crop
contains exactly 28542 unmasked voxels (the MLP masker's voxel count).

This means any atlas-free CNN volume -- PubMed, Nilearn, or NeuroVault --
can be converted into MLP masker-flat space (or back) with a single boolean
index, no interpolation. That conversion previously did not exist in this
codebase, so the MLP autoencoder/contrastive comparisons could not run on
Nilearn or NeuroVault at all; this module makes that comparison possible.

Nilearn and NeuroVault maps are continuous statistic/probability images,
unlike the binary PubMed activation masks the MLP encoder was trained on
(see ``docs/03_evaluation/11_autoencoder.ipynb`` and
``12_neurovault_decoding.ipynb``, which binarize with ``(x > 0).float()``
before encoding). ``binarize=True`` applies the same convention here so the
input distribution matches what the MLP encoder was trained on.
"""

from __future__ import annotations

from functools import lru_cache

import torch

NATIVE_SHAPE = (36, 45, 38)
MLP_MASKER_VOXEL_COUNT = 28542


@lru_cache(maxsize=1)
def _crop_mask() -> torch.Tensor:
    from atlas_free_cnn.training.ale_dataset import _brain_crop, _mask_data_and_affine

    mask, _ = _mask_data_and_affine(None)
    crop = _brain_crop(mask)
    crop_mask = mask[crop]
    native_shape = tuple(int(s.stop - s.start) for s in crop)
    if native_shape != NATIVE_SHAPE:
        raise ValueError(
            f"MLP mask brain-crop shape {native_shape} no longer matches the atlas-free "
            f"CNN's {NATIVE_SHAPE} packed volumes; the masker-flat bridge is out of date."
        )
    if int(crop_mask.sum()) != MLP_MASKER_VOXEL_COUNT:
        raise ValueError(
            f"MLP mask brain-crop voxel count {int(crop_mask.sum())} no longer matches "
            f"the MLP masker ({MLP_MASKER_VOXEL_COUNT})."
        )
    return torch.from_numpy(crop_mask)


def atlas_free_volume_to_mlp_flat(volume_batch: torch.Tensor, *, binarize: bool) -> torch.Tensor:
    """Convert atlas-free CNN volumes to MLP masker-flat vectors.

    Parameters
    ----------
    volume_batch : (B, 1, 36, 45, 38) or (B, 36, 45, 38) tensor
    binarize : apply ``(x > 0).float()`` before flattening, matching the
        established convention for feeding continuous maps into the
        binary-trained MLP encoder.

    Returns
    -------
    (B, 28542) tensor
    """
    crop_mask = _crop_mask()
    vol = volume_batch.detach().cpu().float()
    if vol.ndim == 5:
        vol = vol.squeeze(1)
    if tuple(vol.shape[-3:]) != NATIVE_SHAPE:
        raise ValueError(f"Expected atlas-free volumes with shape {NATIVE_SHAPE}, got {tuple(vol.shape[-3:])}")
    if binarize:
        vol = (vol > 0).float()
    return vol[:, crop_mask]


def mlp_flat_to_atlas_free_volume(flat_batch: torch.Tensor) -> torch.Tensor:
    """Convert MLP masker-flat vectors back to atlas-free CNN volume space.

    Parameters
    ----------
    flat_batch : (B, 28542) tensor

    Returns
    -------
    (B, 1, 36, 45, 38) tensor
    """
    crop_mask = _crop_mask()
    flat = flat_batch.detach().cpu().float()
    if flat.shape[-1] != MLP_MASKER_VOXEL_COUNT:
        raise ValueError(f"Expected MLP flat vectors with {MLP_MASKER_VOXEL_COUNT} features, got {flat.shape[-1]}")
    out = torch.zeros((flat.shape[0], *NATIVE_SHAPE), dtype=torch.float32)
    out[:, crop_mask] = flat
    return out.unsqueeze(1)
