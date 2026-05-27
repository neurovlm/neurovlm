"""Staged NeuroVault collector for atlas-free CNN training volumes.

The collector intentionally starts from a filtered subset instead of mirroring
NeuroVault. It keeps every decision in a manifest, writes text positives as
JSONL, and packs accepted maps into the same 4 mm crop-to-brain geometry used by
the atlas-free ALE CNN cache.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.data_building.definitions import (  # noqa: E402
    POSITIVE_WEIGHTS,
    normalize_key,
    slugify,
    text_pair,
)


NEUROVAULT_API = "https://neurovault.org/api/"
QUALITY_FLAGS = (
    "missing_metadata",
    "no_task_label",
    "no_doi_or_pmid",
    "weird_shape",
    "mostly_empty",
    "negative_values_present",
    "thresholded_map_possible",
    "failed_resample",
    "low_quality_text",
)


@dataclass(frozen=True)
class NeuroVaultConfig:
    """Runtime limits and preprocessing settings for staged collection."""

    max_images: int = 500
    max_pages: int = 100
    min_quality_score: int = 3
    strong_quality_score: int = 6
    target_resolution_mm: float = 4.0
    crop_to_brain: bool = True
    positive_only: bool = True
    robust_lower_percentile: float = 1.0
    robust_upper_percentile: float = 99.5
    mostly_empty_min_fraction: float = 0.001
    request_sleep_s: float = 0.05
    timeout_s: float = 60.0
    max_accepted_per_collection: int | None = None
    target_accepted_volumes: int | None = None
    show_progress: bool = True
    progress_interval: int = 100


def _clean_text(value: Any, *, max_words: int | None = None) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    if max_words is not None:
        text = " ".join(text.split()[:max_words])
    return text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first(meta: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = meta.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _api_get(url: str, *, timeout_s: float) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "neurovlm-neurovault-collector/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_neurovault_image_summaries(
    *,
    api_base: str = NEUROVAULT_API,
    max_images: int = 500,
    max_pages: int = 100,
    collection_ids: Iterable[int | str] | None = None,
    timeout_s: float = 60.0,
    request_sleep_s: float = 0.05,
) -> Iterable[dict[str, Any]]:
    """Yield image summary records from `/api/images/` or selected collections."""

    api_base = api_base.rstrip("/") + "/"
    seen: set[str] = set()

    if collection_ids:
        for collection_id in collection_ids:
            url = urljoin(api_base, f"collections/{collection_id}/images")
            pages = 0
            while url and len(seen) < max_images and pages < max_pages:
                payload = _api_get(url, timeout_s=timeout_s)
                pages += 1
                records = payload.get("results", payload if isinstance(payload, list) else [])
                for rec in records:
                    image_id = str(rec.get("id") or rec.get("pk") or "")
                    if image_id and image_id not in seen:
                        seen.add(image_id)
                        yield rec
                        if len(seen) >= max_images:
                            return
                url = payload.get("next") if isinstance(payload, dict) else None
                if request_sleep_s:
                    time.sleep(request_sleep_s)
        return

    url = urljoin(api_base, "images/")
    pages = 0
    while url and len(seen) < max_images and pages < max_pages:
        payload = _api_get(url, timeout_s=timeout_s)
        pages += 1
        records = payload.get("results", payload if isinstance(payload, list) else [])
        for rec in records:
            image_id = str(rec.get("id") or rec.get("pk") or "")
            if image_id and image_id not in seen:
                seen.add(image_id)
                yield rec
                if len(seen) >= max_images:
                    return
        url = payload.get("next") if isinstance(payload, dict) else None
        if request_sleep_s:
            time.sleep(request_sleep_s)


def load_image_metadata(image_id: int | str, *, api_base: str = NEUROVAULT_API, timeout_s: float = 60.0) -> dict[str, Any]:
    return _api_get(urljoin(api_base.rstrip("/") + "/", f"images/{image_id}/"), timeout_s=timeout_s)


def load_collection_metadata(collection_id: int | str, *, api_base: str = NEUROVAULT_API, timeout_s: float = 60.0) -> dict[str, Any]:
    return _api_get(urljoin(api_base.rstrip("/") + "/", f"collections/{collection_id}/"), timeout_s=timeout_s)


def image_id_from_summary(summary: dict[str, Any]) -> str:
    raw = str(summary.get("id") or summary.get("pk") or summary.get("image_id") or summary.get("url") or "").strip()
    match = re.search(r"/images/(\d+)/?", raw)
    return match.group(1) if match else raw


def collection_id_from_image(image_meta: dict[str, Any]) -> str:
    collection = image_meta.get("collection")
    if isinstance(collection, dict):
        raw = str(collection.get("id") or collection.get("pk") or collection.get("url") or "").strip()
    else:
        raw = str(image_meta.get("collection_id") or collection or "").strip()
    match = re.search(r"/collections/(\d+)/?", raw)
    return match.group(1) if match else raw


def collection_key(collection_id: str, collection_meta: dict[str, Any]) -> str:
    if collection_id:
        return str(collection_id)
    name = _clean_text(_first(collection_meta, ("name", "title")))
    return name or "unknown_collection"


def usable_download_url(image_meta: dict[str, Any], *, api_base: str = NEUROVAULT_API) -> str:
    """Return a usable NIfTI URL if one is visible in NeuroVault metadata."""

    candidates = [
        image_meta.get("file"),
        image_meta.get("download_url"),
        image_meta.get("download"),
        image_meta.get("file_url"),
        image_meta.get("url"),
        image_meta.get("absolute_url"),
    ]
    for candidate in candidates:
        url = _clean_text(candidate)
        if not url:
            continue
        if url.startswith("/"):
            url = urljoin(api_base.rstrip("/") + "/", url)
        if url.startswith("http") and (".nii" in url.lower() or "download" in url.lower()):
            return url
    return ""


def is_probably_volumetric_nifti(image_meta: dict[str, Any], download_url: str) -> bool:
    lowered = " ".join(
        _clean_text(v).lower()
        for v in (
            download_url,
            image_meta.get("file"),
            image_meta.get("map_type"),
            image_meta.get("image_type"),
            image_meta.get("modality"),
        )
    )
    if any(token in lowered for token in (".gii", "surface", "fsaverage", "cifti", ".dtseries", ".dscalar")):
        return False
    return ".nii" in lowered or "nifti" in lowered or "download" in lowered


def doi_or_pmid(image_meta: dict[str, Any], collection_meta: dict[str, Any]) -> tuple[str, str]:
    doi = _clean_text(_first(image_meta, ("doi", "DOI")) or _first(collection_meta, ("doi", "DOI")))
    pmid = _clean_text(
        _first(image_meta, ("pmid", "PMID", "pubmed_id", "pubmed"))
        or _first(collection_meta, ("pmid", "PMID", "pubmed_id", "pubmed"))
    )
    return doi, pmid


def task_or_contrast_terms(image_meta: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract conservative task/contrast labels and optional definitions."""

    pairs: list[tuple[str, str]] = []
    for key in ("task", "task_name", "cognitive_paradigm_cogatlas", "contrast_definition", "contrast_name"):
        value = _clean_text(image_meta.get(key))
        if value:
            pairs.append((value, ""))
    for item in _as_list(image_meta.get("cognitive_concepts")) + _as_list(image_meta.get("cognitive_contrast_cogatlas")):
        if isinstance(item, dict):
            name = _clean_text(_first(item, ("name", "label", "term")))
            definition = _clean_text(_first(item, ("definition", "description")), max_words=120)
            if name:
                pairs.append((name, definition))
        else:
            text = _clean_text(item)
            if text:
                pairs.append((text, ""))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for term, definition in pairs:
        key = normalize_key(term)
        if key and key not in seen:
            out.append((term, definition))
            seen.add(key)
    return out


def quality_flags_for_metadata(
    image_meta: dict[str, Any],
    collection_meta: dict[str, Any],
    *,
    download_url: str,
) -> dict[str, bool]:
    name = _clean_text(_first(image_meta, ("name", "title")))
    description = _clean_text(_first(image_meta, ("description", "map_description")), max_words=30)
    collection_name = _clean_text(_first(collection_meta, ("name", "title")))
    collection_description = _clean_text(_first(collection_meta, ("description", "abstract")), max_words=30)
    doi, pmid = doi_or_pmid(image_meta, collection_meta)
    task_terms = task_or_contrast_terms(image_meta)
    threshold_text = " ".join(
        _clean_text(v).lower()
        for v in (
            image_meta.get("name"),
            image_meta.get("description"),
            image_meta.get("map_type"),
            image_meta.get("image_type"),
            image_meta.get("statistic_parameters"),
        )
    )
    low_quality_text = not any(len(x.split()) >= 3 for x in (description, collection_description)) and not task_terms
    return {
        "missing_metadata": not any([name, description, collection_name, collection_description]),
        "no_task_label": not bool(task_terms),
        "no_doi_or_pmid": not bool(doi or pmid),
        "weird_shape": False,
        "mostly_empty": False,
        "negative_values_present": False,
        "thresholded_map_possible": _truthy(image_meta.get("is_thresholded"))
        or any(token in threshold_text for token in ("threshold", "cluster-corrected", "fdr", "fwe", "tfce")),
        "failed_resample": False,
        "low_quality_text": low_quality_text,
    }


def quality_score(flags: dict[str, bool], image_meta: dict[str, Any], collection_meta: dict[str, Any]) -> int:
    score = 0
    if _clean_text(_first(image_meta, ("name", "title"))):
        score += 1
    if _clean_text(_first(image_meta, ("description", "map_description")), max_words=6):
        score += 1
    if task_or_contrast_terms(image_meta):
        score += 2
    if not flags.get("no_doi_or_pmid", True):
        score += 1
    if _clean_text(_first(collection_meta, ("description", "abstract")), max_words=8):
        score += 1
    if any(
        _clean_text(image_meta.get(key))
        for key in ("cognitive_paradigm_cogatlas", "cognitive_contrast_cogatlas", "cognitive_concepts")
    ):
        score += 1
    for penalty in (
        "missing_metadata",
        "weird_shape",
        "mostly_empty",
        "failed_resample",
        "low_quality_text",
        "thresholded_map_possible",
    ):
        if flags.get(penalty, False):
            score -= 1
    return max(0, int(score))


def quality_tier(score: int, flags: dict[str, bool], config: NeuroVaultConfig) -> str:
    if flags.get("failed_resample") or flags.get("mostly_empty") or flags.get("weird_shape"):
        return "skipped"
    if score < config.min_quality_score:
        return "skipped"
    if score >= config.strong_quality_score:
        return "strong"
    return "weak"


def build_text_positives(
    image_meta: dict[str, Any],
    collection_meta: dict[str, Any],
    *,
    map_id: str,
) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    image_name = _clean_text(_first(image_meta, ("name", "title")))
    image_desc = _clean_text(_first(image_meta, ("description", "map_description")), max_words=180)
    if image_name and image_desc:
        positives.append(
            {
                "map_id": map_id,
                "text": text_pair(image_name, image_desc),
                "term": image_name,
                "category": "image_description",
                "source": "neurovault_image",
                "weight": 0.85,
                "reliability": "medium",
            }
        )
    elif image_name:
        positives.append(
            {
                "map_id": map_id,
                "text": image_name,
                "term": image_name,
                "category": "image_name",
                "source": "neurovault_image",
                "weight": 0.55,
                "reliability": "weak",
            }
        )

    for term, definition in task_or_contrast_terms(image_meta):
        positives.append(
            {
                "map_id": map_id,
                "text": text_pair(term, definition) if definition else term,
                "term": term,
                "category": "cognitive_task_or_contrast",
                "source": "neurovault_cogatlas_metadata" if definition else "neurovault_task_label",
                "weight": POSITIVE_WEIGHTS.get("cognitive_atlas_exact_match", 0.9),
                "reliability": "strong" if definition else "medium",
            }
        )

    collection_name = _clean_text(_first(collection_meta, ("name", "title")))
    collection_desc = _clean_text(_first(collection_meta, ("description", "abstract")), max_words=220)
    if collection_name and collection_desc:
        positives.append(
            {
                "map_id": map_id,
                "text": text_pair(collection_name, collection_desc),
                "term": collection_name,
                "category": "collection_description",
                "source": "neurovault_collection",
                "weight": 0.65,
                "reliability": "medium",
            }
        )

    paper_title = _clean_text(_first(collection_meta, ("paper_title", "publication_title", "title")) or image_meta.get("paper_title"))
    doi, pmid = doi_or_pmid(image_meta, collection_meta)
    if paper_title and (doi or pmid):
        positives.append(
            {
                "map_id": map_id,
                "text": paper_title,
                "term": paper_title,
                "category": "paper_title",
                "source": "neurovault_publication_metadata",
                "weight": POSITIVE_WEIGHTS.get("paper_title", 0.5),
                "reliability": "medium",
                "doi": doi,
                "pmid": pmid,
            }
        )

    # Generated definitions are deliberately narrow: only attach a templated
    # definition when a clean Cognitive Atlas-like term is explicitly present.
    generated: list[dict[str, Any]] = []
    for pos in positives:
        if pos["category"] == "cognitive_task_or_contrast" and "[SEP]" not in pos["text"] and len(pos["term"].split()) <= 6:
            generated.append(
                {
                    **pos,
                    "text": text_pair(pos["term"], f"{pos['term']} is the task or contrast label supplied by NeuroVault for this brain map."),
                    "source": "neurovault_generated_clear_metadata",
                    "reliability": "weak",
                    "weight": min(float(pos["weight"]), 0.5),
                }
            )
    positives.extend(generated)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for pos in positives:
        key = normalize_key(pos["text"])
        if key and key not in seen:
            deduped.append(pos)
            seen.add(key)
    return deduped


def download_nifti(url: str, out_path: str | Path, *, timeout_s: float = 60.0) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "neurovlm-neurovault-collector/1.0"})
    with urlopen(req, timeout=timeout_s) as resp, out_path.open("wb") as f:
        shutil.copyfileobj(resp, f)
    return out_path


def _load_downloaded_nifti(path: Path):
    try:
        return nib.load(str(path))
    except Exception:
        if path.suffix == ".gz" and not path.name.endswith(".nii.gz"):
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with gzip.open(path, "rb") as src, tmp_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                return nib.load(str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)
        raise


def _looks_binary_volume(data: np.ndarray) -> bool:
    """Return True for 0/1 mask-like volumes that should use nearest resampling."""

    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    unique = np.unique(finite)
    if unique.size > 3:
        return False
    return bool(np.all(np.isclose(unique, 0.0, atol=1e-5) | np.isclose(unique, 1.0, atol=1e-5)))


def preprocess_neurovault_nifti(
    nifti_path: str | Path,
    *,
    config: NeuroVaultConfig,
) -> tuple[Any | None, dict[str, Any], dict[str, bool]]:
    """Resample one NeuroVault map into `(1, D, H, W)` atlas-free CNN format."""

    flags = {key: False for key in QUALITY_FLAGS}
    meta: dict[str, Any] = {"input_path": str(nifti_path)}
    try:
        from nilearn.image import resample_to_img
        from neurovlm.gnn.ale_dataset import _brain_crop, _get_mask_img_for_resolution
        import torch

        img = _load_downloaded_nifti(Path(nifti_path))
        shape = tuple(int(x) for x in img.shape)
        meta["source_shape"] = list(shape)
        if len(shape) == 4 and shape[3] == 1:
            img = nib.Nifti1Image(np.asarray(img.get_fdata())[..., 0], img.affine, img.header)
            shape = tuple(int(x) for x in img.shape)
        if len(shape) != 3 or min(shape[:3]) < 2:
            flags["weird_shape"] = True
            return None, meta, flags

        mask_img = _get_mask_img_for_resolution(config.target_resolution_mm)
        mask = np.asarray(mask_img.get_fdata() > 0)
        crop = _brain_crop(mask) if config.crop_to_brain else (slice(None), slice(None), slice(None))
        source_data = np.nan_to_num(img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        interpolation = "nearest" if _looks_binary_volume(source_data) else "continuous"
        meta["resample_interpolation"] = interpolation
        resampled = resample_to_img(img, mask_img, interpolation=interpolation, force_resample=True, copy_header=True)
        arr = np.nan_to_num(resampled.get_fdata().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        flags["negative_values_present"] = bool(np.any(arr[mask] < 0))
        if config.positive_only:
            arr = np.clip(arr, 0.0, None)
        arr *= mask.astype(np.float32)
        cropped = arr[crop].astype(np.float32)
        crop_mask = mask[crop]
        brain_values = cropped[crop_mask]
        nonzero_fraction = float(np.count_nonzero(brain_values > 0) / max(1, int(crop_mask.sum())))
        flags["mostly_empty"] = nonzero_fraction < config.mostly_empty_min_fraction

        if brain_values.size and np.any(brain_values > 0):
            values_for_scale = brain_values[brain_values > 0] if config.positive_only else brain_values
            lo = float(np.percentile(values_for_scale, config.robust_lower_percentile))
            hi = float(np.percentile(values_for_scale, config.robust_upper_percentile))
            if hi <= lo:
                lo = 0.0
                hi = float(np.max(values_for_scale))
            if hi > lo:
                cropped = (cropped - lo) / (hi - lo)
                cropped = np.clip(cropped, 0.0, 1.0)
            else:
                flags["mostly_empty"] = True
        else:
            flags["mostly_empty"] = True

        tensor = torch.from_numpy(cropped.astype(np.float32)).unsqueeze(0).contiguous()
        meta.update(
            {
                "target_resolution_mm": float(config.target_resolution_mm),
                "target_shape": list(mask.shape),
                "crop_shape": list(tensor.shape[1:]),
                "crop_slices": [[s.start, s.stop, s.step] for s in crop],
                "tensor_shape": list(tensor.shape),
                "nonzero_fraction": nonzero_fraction,
                "positive_only": bool(config.positive_only),
                "normalization": {
                    "method": "robust_percentile",
                    "lower": float(config.robust_lower_percentile),
                    "upper": float(config.robust_upper_percentile),
                    "range": [0.0, 1.0],
                },
            }
        )
        return tensor, meta, flags
    except Exception as exc:
        flags["failed_resample"] = True
        meta["error"] = repr(exc)
        return None, meta, flags


def _manifest_row(
    *,
    image_id: str,
    collection_id: str,
    map_id: str,
    image_meta: dict[str, Any],
    collection_meta: dict[str, Any],
    download_url: str,
    score: int,
    tier: str,
    flags: dict[str, bool],
    tensor_index: int | None,
    error: str = "",
) -> dict[str, Any]:
    doi, pmid = doi_or_pmid(image_meta, collection_meta)
    row = {
        "map_id": map_id,
        "image_id": image_id,
        "collection_id": collection_id,
        "image_name": _clean_text(_first(image_meta, ("name", "title"))),
        "collection_name": _clean_text(_first(collection_meta, ("name", "title"))),
        "download_url": download_url,
        "doi": doi,
        "pmid": pmid,
        "quality_score": int(score),
        "quality_tier": tier,
        "tensor_index": "" if tensor_index is None else int(tensor_index),
        "error": error,
    }
    row.update({flag: bool(flags.get(flag, False)) for flag in QUALITY_FLAGS})
    return row


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "map_id",
        "image_id",
        "collection_id",
        "image_name",
        "collection_name",
        "download_url",
        "doi",
        "pmid",
        "quality_score",
        "quality_tier",
        "tensor_index",
        *QUALITY_FLAGS,
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return path


def collect_neurovault(
    *,
    output_dir: str | Path = "experiments/3dcnn/atlas_free_cnn/cache/neurovault",
    api_base: str = NEUROVAULT_API,
    collection_ids: Iterable[int | str] | None = None,
    config: NeuroVaultConfig = NeuroVaultConfig(),
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw_niftis"
    manifest_rows: list[dict[str, Any]] = []
    positives_rows: list[dict[str, Any]] = []
    volumes: list[Any] = []
    volume_map_ids: list[str] = []
    preprocess_records: list[dict[str, Any]] = []
    collection_cache: dict[str, dict[str, Any]] = {}
    accepted_by_collection: dict[str, int] = defaultdict(int)
    candidate_count = 0
    cap_skipped = 0
    quality_skipped = 0
    preprocess_skipped = 0
    error_count = 0

    progress_bar = None
    if config.show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(total=config.max_images, desc="Collecting NeuroVault", unit="image")
        except ImportError:
            progress_bar = None

    def progress_status() -> dict[str, int]:
        return {
            "candidates": candidate_count,
            "accepted": len(volumes),
            "strong": sum(1 for row in manifest_rows if row.get("quality_tier") == "strong"),
            "weak": sum(1 for row in manifest_rows if row.get("quality_tier") == "weak"),
            "skipped": sum(1 for row in manifest_rows if row.get("quality_tier") == "skipped"),
            "cap_skipped": cap_skipped,
            "quality_skipped": quality_skipped,
            "preprocess_skipped": preprocess_skipped,
            "errors": error_count,
            "texts": len(positives_rows),
            "collections": len(accepted_by_collection),
        }

    def update_progress(*, force: bool = False) -> None:
        status = progress_status()
        if progress_bar is not None:
            progress_bar.set_postfix(
                accepted=status["accepted"],
                skipped=status["skipped"],
                cap=status["cap_skipped"],
                collections=status["collections"],
                texts=status["texts"],
            )
            return
        interval = max(1, int(config.progress_interval))
        if config.show_progress and (force or candidate_count == 1 or candidate_count % interval == 0):
            print(f"NeuroVault progress: {status}", flush=True)

    try:
        for summary in iter_neurovault_image_summaries(
            api_base=api_base,
            max_images=config.max_images,
            max_pages=config.max_pages,
            collection_ids=collection_ids,
            timeout_s=config.timeout_s,
            request_sleep_s=config.request_sleep_s,
        ):
            candidate_count += 1
            if progress_bar is not None:
                progress_bar.update(1)
            image_id = image_id_from_summary(summary)
            if not image_id:
                update_progress()
                continue
            map_id = f"neurovault_{image_id}"
            try:
                image_meta = load_image_metadata(image_id, api_base=api_base, timeout_s=config.timeout_s)
                collection_id = collection_id_from_image(image_meta)
                collection_meta = collection_cache.get(collection_id, {})
                if collection_id and collection_id not in collection_cache:
                    collection_meta = load_collection_metadata(collection_id, api_base=api_base, timeout_s=config.timeout_s)
                    collection_cache[collection_id] = collection_meta
                coll_key = collection_key(collection_id, collection_meta)
                download_url = usable_download_url(image_meta, api_base=api_base)
                flags = quality_flags_for_metadata(image_meta, collection_meta, download_url=download_url)
                if not download_url or not is_probably_volumetric_nifti(image_meta, download_url):
                    flags["weird_shape"] = True
                score = quality_score(flags, image_meta, collection_meta)
                tier = quality_tier(score, flags, config)
                if (
                    tier != "skipped"
                    and config.max_accepted_per_collection is not None
                    and config.max_accepted_per_collection > 0
                    and accepted_by_collection[coll_key] >= config.max_accepted_per_collection
                ):
                    cap_skipped += 1
                    manifest_rows.append(
                        _manifest_row(
                            image_id=image_id,
                            collection_id=collection_id,
                            map_id=map_id,
                            image_meta=image_meta,
                            collection_meta=collection_meta,
                            download_url=download_url,
                            score=score,
                            tier="skipped",
                            flags=flags,
                            tensor_index=None,
                            error=f"collection_cap_reached:{config.max_accepted_per_collection}",
                        )
                    )
                    update_progress()
                    continue
                if tier == "skipped":
                    quality_skipped += 1
                    manifest_rows.append(
                        _manifest_row(
                            image_id=image_id,
                            collection_id=collection_id,
                            map_id=map_id,
                            image_meta=image_meta,
                            collection_meta=collection_meta,
                            download_url=download_url,
                            score=score,
                            tier=tier,
                            flags=flags,
                            tensor_index=None,
                        )
                    )
                    update_progress()
                    continue

                raw_path = raw_dir / f"{map_id}_{slugify(_clean_text(image_meta.get('name')) or image_id)}.nii.gz"
                download_nifti(download_url, raw_path, timeout_s=config.timeout_s)
                tensor, pre_meta, pre_flags = preprocess_neurovault_nifti(raw_path, config=config)
                flags.update(pre_flags)
                score = quality_score(flags, image_meta, collection_meta)
                tier = quality_tier(score, flags, config)
                tensor_index = None
                if tensor is not None and tier != "skipped":
                    tensor_index = len(volumes)
                    volumes.append(tensor)
                    volume_map_ids.append(map_id)
                    accepted_by_collection[coll_key] += 1
                    for pos in build_text_positives(image_meta, collection_meta, map_id=map_id):
                        pos.update({"quality_score": score, "quality_tier": tier, "image_id": image_id, "collection_id": collection_id})
                        positives_rows.append(pos)
                else:
                    preprocess_skipped += 1
                preprocess_records.append({"map_id": map_id, "image_id": image_id, "flags": flags, "metadata": pre_meta})
                manifest_rows.append(
                    _manifest_row(
                        image_id=image_id,
                        collection_id=collection_id,
                        map_id=map_id,
                        image_meta=image_meta,
                        collection_meta=collection_meta,
                        download_url=download_url,
                        score=score,
                        tier=tier,
                        flags=flags,
                        tensor_index=tensor_index,
                        error=pre_meta.get("error", ""),
                    )
                )
                update_progress()
                if config.target_accepted_volumes is not None and len(volumes) >= config.target_accepted_volumes:
                    update_progress(force=True)
                    break
            except Exception as exc:
                error_count += 1
                flags = {key: False for key in QUALITY_FLAGS}
                flags["failed_resample"] = True
                manifest_rows.append(
                    _manifest_row(
                        image_id=image_id,
                        collection_id="",
                        map_id=map_id,
                        image_meta=summary,
                        collection_meta={},
                        download_url="",
                        score=0,
                        tier="skipped",
                        flags=flags,
                        tensor_index=None,
                        error=repr(exc),
                    )
                )
                update_progress()
    finally:
        update_progress(force=True)
        if progress_bar is not None:
            progress_bar.close()

    manifest_path = output_dir / "neurovault_manifest.csv"
    positives_path = output_dir / "neurovault_text_positives.jsonl"
    volumes_path = output_dir / "neurovault_cnn_volumes.pt"
    report_path = output_dir / "neurovault_preprocess_report.json"

    write_manifest(manifest_rows, manifest_path)
    write_jsonl(positives_rows, positives_path)
    import torch

    if volumes:
        packed = torch.stack(volumes).contiguous()
    else:
        packed = torch.empty((0, 1, 0, 0, 0), dtype=torch.float32)
    torch.save(
        {
            "version": 1,
            "source": "neurovault",
            "config": asdict(config),
            "volumes": packed,
            "map_ids": np.asarray(volume_map_ids).astype(str),
            "metadata": {
                "n_volumes": int(packed.shape[0]),
                "shape": list(packed.shape[1:]),
                "positive_only": bool(config.positive_only),
                "note": "NeuroVault maps are resampled images; ALE smoothing is not applied.",
            },
        },
        volumes_path,
    )
    report = {
        "config": asdict(config),
        "counts": {
            "candidate_images": len(manifest_rows),
            "accepted_volumes": len(volumes),
            "strong": sum(1 for row in manifest_rows if row["quality_tier"] == "strong"),
            "weak": sum(1 for row in manifest_rows if row["quality_tier"] == "weak"),
            "skipped": sum(1 for row in manifest_rows if row["quality_tier"] == "skipped"),
            "text_positives": len(positives_rows),
        },
        "accepted_by_collection": dict(sorted(accepted_by_collection.items(), key=lambda item: item[1], reverse=True)),
        "skipped_by_reason": {
            "collection_cap_reached": cap_skipped,
            "metadata_or_quality": quality_skipped,
            "preprocess_or_empty": preprocess_skipped,
            "errors": error_count,
        },
        "quality_flags": {
            flag: sum(1 for row in manifest_rows if row.get(flag) is True)
            for flag in QUALITY_FLAGS
        },
        "records": preprocess_records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    return {
        "manifest": manifest_path,
        "text_positives": positives_path,
        "volumes": volumes_path,
        "preprocess_report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/3dcnn/atlas_free_cnn/cache/neurovault")
    parser.add_argument("--api-base", default=NEUROVAULT_API)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--collection-ids", nargs="*", default=None)
    parser.add_argument("--min-quality-score", type=int, default=3)
    parser.add_argument("--strong-quality-score", type=int, default=6)
    parser.add_argument("--target-resolution-mm", type=float, default=4.0)
    parser.add_argument("--allow-negative", action="store_true", help="Disable positive-only clipping.")
    parser.add_argument("--request-sleep-s", type=float, default=0.05)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm/progress status output.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print status every N candidates when tqdm is unavailable.")
    parser.add_argument(
        "--max-accepted-per-collection",
        type=int,
        default=0,
        help="Skip additional accepted maps from a collection after this cap. Use 0 for no cap.",
    )
    parser.add_argument(
        "--target-accepted-volumes",
        type=int,
        default=0,
        help="Stop once this many accepted volumes have been saved. Use 0 to rely on --max-images.",
    )
    args = parser.parse_args()

    config = NeuroVaultConfig(
        max_images=args.max_images,
        max_pages=args.max_pages,
        min_quality_score=args.min_quality_score,
        strong_quality_score=args.strong_quality_score,
        target_resolution_mm=args.target_resolution_mm,
        positive_only=not args.allow_negative,
        request_sleep_s=args.request_sleep_s,
        max_accepted_per_collection=None if args.max_accepted_per_collection <= 0 else args.max_accepted_per_collection,
        target_accepted_volumes=None if args.target_accepted_volumes <= 0 else args.target_accepted_volumes,
        show_progress=not args.no_progress,
        progress_interval=args.progress_interval,
    )
    outputs = collect_neurovault(
        output_dir=args.output_dir,
        api_base=args.api_base,
        collection_ids=args.collection_ids,
        config=config,
    )
    print("Wrote NeuroVault staged collector outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
