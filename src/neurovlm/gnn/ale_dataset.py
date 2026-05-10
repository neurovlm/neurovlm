"""Datasets and packed caches for ALE 3D CNN NeuroVLM experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


ALEMode = Literal["difumo_compatible", "atlas_free"]


@dataclass(frozen=True)
class ALEPreprocessConfig:
    """Preprocessing parameters saved with every packed ALE cache."""

    mode: ALEMode
    kernel_fwhm_mm: float = 9.0
    resolution_mm: float = 4.0
    crop_to_brain: bool = True
    normalize: str = "max"
    clamp: bool = True
    cache_dtype: str = "float16"
    max_papers: Optional[int] = None

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _pmids_digest(pmids: Iterable[str]) -> str:
    return hashlib.sha1("\n".join(map(str, pmids)).encode("utf-8")).hexdigest()


def load_pubmed_text_embeddings() -> tuple[Tensor, np.ndarray]:
    """Return SPECTER embeddings and PMIDs as CPU tensors/strings."""
    from neurovlm.data import load_latent

    print("Loading SPECTER PubMed text embeddings ...", flush=True)
    payload = load_latent("pubmed_text")
    if isinstance(payload, tuple):
        text, pmids = payload
    elif isinstance(payload, dict):
        pmids = np.asarray(list(payload.keys())).astype(str)
        text = torch.stack([torch.as_tensor(payload[p], dtype=torch.float32) for p in pmids])
    else:
        raise TypeError(
            "Expected load_latent('pubmed_text') to return (tensor, pmids) or a dict."
        )
    text = torch.as_tensor(text, dtype=torch.float32).cpu()
    pmids = np.asarray(pmids).astype(str)
    print(f"Loaded SPECTER text embeddings: {tuple(text.shape)}", flush=True)
    return text, pmids


def load_pubmed_flatmaps() -> tuple[Tensor, np.ndarray]:
    """Return NeuroVLM processed PubMed brain flatmaps and PMIDs."""
    from neurovlm.data import load_dataset

    payload = load_dataset("pubmed_images")
    if not isinstance(payload, (tuple, list)) or len(payload) < 2:
        raise TypeError("Expected load_dataset('pubmed_images') to return (images, pmids).")
    images, pmids = payload[0], payload[1]
    return torch.as_tensor(images, dtype=torch.float32).cpu(), np.asarray(pmids).astype(str)


def align_modalities_by_pmid(
    brain_pmids: np.ndarray,
    text_embeddings: Tensor,
    text_pmids: np.ndarray,
) -> tuple[np.ndarray, Tensor, np.ndarray]:
    """Return aligned brain row indices, text embeddings, and shared PMIDs.

    Alignment follows SPECTER/text PMID order so the same random seed gives
    comparable train/val/test splits across ALE, coordinate, and flatmap runs.
    """
    brain_lookup = {str(p): i for i, p in enumerate(brain_pmids)}
    brain_rows: list[int] = []
    text_rows: list[int] = []
    shared_pmids: list[str] = []
    for j, pmid in enumerate(text_pmids.astype(str)):
        i = brain_lookup.get(str(pmid))
        if i is None:
            continue
        brain_rows.append(i)
        text_rows.append(j)
        shared_pmids.append(str(pmid))
    if not brain_rows:
        raise RuntimeError("No overlapping PMIDs between ALE brain data and SPECTER text.")
    return (
        np.asarray(brain_rows, dtype=np.int64),
        text_embeddings[text_rows].float().cpu(),
        np.asarray(shared_pmids).astype(str),
    )


def _get_mask_img_for_resolution(resolution_mm: float):
    from nilearn.image import resample_img
    from neurovlm.data import load_masker

    mask_img = load_masker().mask_img_
    if resolution_mm <= 0:
        return mask_img
    affine = np.asarray(mask_img.affine)
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    if np.allclose(voxel_sizes, resolution_mm, atol=0.05):
        return mask_img
    target_affine = np.diag([resolution_mm, resolution_mm, resolution_mm, 1.0])
    return resample_img(mask_img, target_affine=target_affine, interpolation="nearest")


def _mask_data_and_affine(resolution_mm: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    from neurovlm.data import load_masker

    if resolution_mm is None:
        mask_img = load_masker().mask_img_
    else:
        mask_img = _get_mask_img_for_resolution(resolution_mm)
    mask = np.asarray(mask_img.get_fdata() > 0)
    return mask, np.asarray(mask_img.affine, dtype=np.float32)


def _brain_crop(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Mask contains no brain voxels.")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def _normalize_volume(vol: np.ndarray, normalize: str, clamp: bool) -> np.ndarray:
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if clamp:
        vol = np.clip(vol, 0.0, None)
    if normalize == "max":
        mx = float(vol.max())
        if mx > 0:
            vol = vol / mx
    elif normalize == "mass":
        total = float(vol.sum())
        if total > 0:
            vol = vol / total
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of {'max', 'mass', 'none'}")
    if clamp:
        vol = np.clip(vol, 0.0, 1.0)
    return vol.astype(np.float32)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError("cache_dtype must be one of {'float16', 'bfloat16', 'float32'}")


def _build_difumo_compatible_volumes(config: ALEPreprocessConfig) -> tuple[Tensor, np.ndarray, dict]:
    """Reconstruct NeuroVLM processed flatmaps into 3D masked volumes."""
    flatmaps, pmids = load_pubmed_flatmaps()
    if config.max_papers is not None:
        flatmaps = flatmaps[: config.max_papers]
        pmids = pmids[: config.max_papers]

    mask, affine = _mask_data_and_affine(None)
    crop = _brain_crop(mask) if config.crop_to_brain else (slice(None), slice(None), slice(None))
    crop_mask = mask[crop]
    if int(mask.sum()) != int(flatmaps.shape[1]):
        raise ValueError(
            f"Mask has {int(mask.sum())} voxels but flatmaps have "
            f"{int(flatmaps.shape[1])} features."
        )
    if int(crop_mask.sum()) != int(flatmaps.shape[1]):
        raise ValueError("Brain crop unexpectedly excluded masked voxels.")

    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
    target_resolution = float(config.resolution_mm)
    native_shape = tuple(int(s.stop - s.start) for s in crop)
    target_shape = native_shape
    needs_resample = target_resolution > 0 and not np.allclose(
        voxel_sizes, target_resolution, atol=0.05
    )
    if needs_resample:
        target_shape = tuple(
            max(1, int(round(native_shape[i] * float(voxel_sizes[i]) / target_resolution)))
            for i in range(3)
        )

    dtype = _torch_dtype(config.cache_dtype)
    out = torch.empty((len(flatmaps), *target_shape), dtype=dtype)
    chunk_size = 64

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x  # type: ignore[assignment]

    for start in tqdm(
        range(0, len(flatmaps), chunk_size),
        desc="Reconstructing flatmaps",
        unit="chunk",
    ):
        stop = min(start + chunk_size, len(flatmaps))
        rows = flatmaps[start:stop].float().numpy()
        chunk = np.zeros((stop - start, *native_shape), dtype=np.float32)
        chunk[:, crop_mask] = rows
        for i in range(len(chunk)):
            chunk[i] = _normalize_volume(chunk[i], config.normalize, config.clamp)
        chunk_t = torch.from_numpy(chunk).unsqueeze(1)
        if needs_resample:
            import torch.nn.functional as F

            chunk_t = F.interpolate(
                chunk_t,
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
        out[start:stop] = chunk_t.squeeze(1).to(dtype)

    tensor = out.contiguous()
    meta = {
        "source": "pubmed_images flatmaps scattered into NeuroVLM mask",
        "affine": affine.tolist(),
        "original_shape": list(mask.shape),
        "native_crop_shape": list(native_shape),
        "native_voxel_sizes": voxel_sizes.tolist(),
        "target_resolution_mm": target_resolution,
        "resampled_from_native": bool(needs_resample),
        "crop_slices": [[s.start, s.stop, s.step] for s in crop],
    }
    return tensor, pmids.astype(str), meta


def _coords_to_volume(
    coords: np.ndarray,
    *,
    mask: np.ndarray,
    affine: np.ndarray,
    sigma_vox: float,
    normalize: str,
    clamp: bool,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    impulse = np.zeros(mask.shape, dtype=np.float32)
    inv_affine = np.linalg.inv(affine)
    for xyz in np.unique(coords.astype(np.float32), axis=0):
        vox = inv_affine @ np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float32)
        ijk = np.rint(vox[:3]).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.asarray(mask.shape)):
            continue
        if not mask[tuple(ijk)]:
            continue
        impulse[tuple(ijk)] += 1.0

    if impulse.sum() == 0:
        return np.zeros(mask.shape, dtype=np.float32)

    vol = gaussian_filter(impulse, sigma=sigma_vox, mode="constant")
    vol *= mask.astype(np.float32)
    return _normalize_volume(vol, normalize, clamp)


def _build_atlas_free_volumes(
    config: ALEPreprocessConfig,
) -> tuple[Tensor, np.ndarray, dict, pd.DataFrame]:
    """Build low-resolution atlas-free ALE-style volumes from MNI coordinates."""
    from neurovlm.retrieval_resources import _load_pubmed_coordinates

    coords_df = _load_pubmed_coordinates().copy()
    coords_df["pmid"] = coords_df["pmid"].astype(str)
    mask, affine = _mask_data_and_affine(config.resolution_mm)
    crop = _brain_crop(mask) if config.crop_to_brain else (slice(None), slice(None), slice(None))

    sigma_mm = config.kernel_fwhm_mm / 2.354820045
    sigma_vox = sigma_mm / float(config.resolution_mm)

    grouped = list(coords_df.groupby("pmid", sort=False))
    if config.max_papers is not None:
        grouped = grouped[: config.max_papers]

    vols: list[Tensor] = []
    pmids: list[str] = []
    cov_rows: list[dict] = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x  # type: ignore[assignment]

    for pmid, grp in tqdm(grouped, desc="Building atlas-free ALE", unit="paper"):
        coords = grp[["x", "y", "z"]].to_numpy(dtype=np.float32)
        vol = _coords_to_volume(
            coords,
            mask=mask,
            affine=affine,
            sigma_vox=sigma_vox,
            normalize=config.normalize,
            clamp=config.clamp,
        )
        if not np.any(vol):
            continue
        cropped = vol[crop]
        vols.append(torch.from_numpy(cropped))
        pmids.append(str(pmid))
        centroid = coords.mean(axis=0)
        cov_rows.append(
            {
                "pmid": str(pmid),
                "n_peaks": int(len(np.unique(coords, axis=0))),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "total_activation_mass": float(cropped.sum()),
                "sparsity": float((cropped <= 0).mean()),
            }
        )

    if not vols:
        raise RuntimeError("Atlas-free ALE builder produced no non-empty volumes.")

    tensor = torch.stack(vols).to(_torch_dtype(config.cache_dtype)).contiguous()
    meta = {
        "source": "pubmed MNI coordinates painted as low-resolution Gaussian ALE volumes",
        "affine": affine.tolist(),
        "original_shape": list(mask.shape),
        "crop_slices": [[s.start, s.stop, s.step] for s in crop],
        "sigma_vox": sigma_vox,
    }
    return tensor, np.asarray(pmids).astype(str), meta, pd.DataFrame(cov_rows)


def build_or_load_ale_cache(
    cache_file: str | Path,
    config: ALEPreprocessConfig,
    *,
    force_rebuild: bool = False,
) -> dict:
    """Load or build one packed ALE volume cache."""
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_rebuild:
        print(f"Loading packed ALE cache: {cache_path}", flush=True)
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"Cache load failed ({exc}); rebuilding.", flush=True)
            payload = None
        if payload is not None and payload.get("version") == 1 and payload.get("config", {}).get("cache_key") == config.cache_key():
            shape = tuple(payload["volumes"].shape)
            print(f"Loaded packed ALE cache: volumes={shape}", flush=True)
            return payload
        if payload is not None:
            print("ALE cache config changed; rebuilding packed cache.")

    if config.mode == "difumo_compatible":
        volumes, pmids, meta = _build_difumo_compatible_volumes(config)
        covariates = None
    elif config.mode == "atlas_free":
        volumes, pmids, meta, covariates = _build_atlas_free_volumes(config)
    else:
        raise ValueError(f"Unknown ALE mode: {config.mode}")

    payload = {
        "version": 1,
        "config": {**asdict(config), "cache_key": config.cache_key()},
        "volumes": volumes,
        "pmids": pmids,
        "metadata": {
            **meta,
            "shape": list(volumes.shape[1:]),
            "n_volumes": int(volumes.shape[0]),
            "pmids_digest": _pmids_digest(pmids),
        },
        "covariates": covariates.to_dict(orient="list") if covariates is not None else None,
    }
    torch.save(payload, cache_path)
    print(f"Saved packed ALE cache: {cache_path}", flush=True)
    return payload


class ALEVolumeDataset(Dataset):
    """PyTorch dataset returning ``volume``/``text`` pairs aligned by PMID."""

    def __init__(
        self,
        volumes: Tensor,
        pmids: np.ndarray,
        text_embeddings: Tensor,
        text_pmids: np.ndarray,
        *,
        covariates: Optional[pd.DataFrame] = None,
    ):
        rows, text, shared_pmids = align_modalities_by_pmid(pmids, text_embeddings, text_pmids)
        self.volumes = volumes[rows].cpu()
        self.text_embeddings = text.cpu()
        self.pmids = shared_pmids
        self.source_rows = rows
        self.input_shape = tuple(self.volumes.shape[1:])

        self.covariates = None
        if covariates is not None:
            cov = covariates.copy()
            cov["pmid"] = cov["pmid"].astype(str)
            self.covariates = cov.set_index("pmid").reindex(shared_pmids).reset_index()

    @classmethod
    def from_cache(cls, cache_payload: dict) -> "ALEVolumeDataset":
        text, text_pmids = load_pubmed_text_embeddings()
        print("Aligning ALE volumes with SPECTER PMIDs ...", flush=True)
        ds = cls(
            volumes=cache_payload["volumes"],
            pmids=np.asarray(cache_payload["pmids"]).astype(str),
            text_embeddings=text,
            text_pmids=text_pmids,
            covariates=pd.DataFrame(cache_payload["covariates"]) if isinstance(cache_payload.get("covariates"), dict) else cache_payload.get("covariates"),
        )
        print(
            f"Aligned ALE/text dataset: n={len(ds):,}, input_shape={ds.input_shape}",
            flush=True,
        )
        return ds

    def __len__(self) -> int:
        return int(self.volumes.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "volume": self.volumes[idx].unsqueeze(0),
            "text": self.text_embeddings[idx].float(),
            "paper_idx": torch.tensor(idx, dtype=torch.long),
        }

    def split(
        self,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> tuple["_SubsetALEDataset", "_SubsetALEDataset", "_SubsetALEDataset"]:
        n = len(self)
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng).tolist()
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        if n >= 3:
            if test_frac > 0 and n_test == 0:
                n_test = 1
            if val_frac > 0 and n_val == 0:
                n_val = 1
            if n_test + n_val >= n:
                n_val = max(0, min(n_val, n - n_test - 1))
        test_pos = perm[:n_test]
        val_pos = perm[n_test : n_test + n_val]
        train_pos = perm[n_test + n_val :]
        return (
            _SubsetALEDataset(self, train_pos),
            _SubsetALEDataset(self, val_pos),
            _SubsetALEDataset(self, test_pos),
        )

    def covariate_frame(self, positions: Optional[list[int]] = None) -> pd.DataFrame:
        if positions is None:
            positions = list(range(len(self)))
        if self.covariates is not None:
            cov = self.covariates.iloc[positions].copy().reset_index(drop=True)
        else:
            vols = self.volumes[positions].float()
            flat = vols.flatten(1)
            coords = []
            for pos, vol in zip(positions, vols):
                active = vol > 0
                nz = active.nonzero(as_tuple=False).float()
                if len(nz):
                    centroid = nz.mean(dim=0)
                else:
                    centroid = torch.zeros(3)
                coords.append(
                    {
                        "pmid": str(self.pmids[pos]),
                        "n_peaks": float("nan"),
                        "centroid_x": float(centroid[0]),
                        "centroid_y": float(centroid[1]),
                        "centroid_z": float(centroid[2]),
                    }
                )
            cov = pd.DataFrame(coords)
            cov["total_activation_mass"] = flat.sum(dim=1).numpy()
            cov["sparsity"] = (flat <= 0).float().mean(dim=1).numpy()
        cov.insert(0, "sample_pos", np.arange(len(cov)))
        return cov


class _SubsetALEDataset(Dataset):
    """Lightweight subset view with stable split positions."""

    def __init__(self, parent: ALEVolumeDataset, positions: list[int]):
        self._parent = parent
        self._positions = positions
        self.input_shape = parent.input_shape

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, idx: int) -> dict:
        return self._parent[self._positions[idx]]

    @property
    def pmids(self) -> np.ndarray:
        return self._parent.pmids[self._positions]

    def covariate_frame(self) -> pd.DataFrame:
        return self._parent.covariate_frame(self._positions)
