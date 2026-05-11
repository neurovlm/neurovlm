"""Fetch Nilearn atlases and write one JSONL row per atlas-derived map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from atlas_free_multipositive.data_building.definitions import (
    POSITIVE_WEIGHTS,
    atlas_label_definition,
    display_atlas_label,
    slugify,
    text_pair,
)
from atlas_free_multipositive.data_building.preprocessing import (
    load_target_mni152_2mm,
    nifti_metadata,
    resample_to_target,
    save_nifti,
)
from atlas_free_multipositive.data_building.text_registry import write_jsonl


def _load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _first_existing_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return None


def _atlas_value(atlas: Any, key: str) -> Any:
    if isinstance(atlas, dict):
        return atlas.get(key)
    return getattr(atlas, key, None)


def _labels(atlas: Any) -> list[str]:
    labels = getattr(atlas, "labels", None)
    if labels is None and isinstance(atlas, dict):
        labels = atlas.get("labels")
    if labels is None:
        return []

    def clean(value: Any) -> str:
        if isinstance(value, bytes):
            value = value.decode()
        return str(value).strip()

    # DiFuMo can return a pandas DataFrame with label metadata.
    columns = getattr(labels, "columns", None)
    if columns is not None:
        preferred = (
            "difumo_names",
            "name",
            "names",
            "label",
            "labels",
            "component",
            "network",
            "yeo_networks7",
            "yeo_networks17",
        )
        for key in preferred:
            if key in columns:
                return [clean(x) for x in labels[key].tolist()]
        readable = [key for key in columns if not str(key).lower().endswith(("x", "y", "z", "id", "index"))]
        if readable:
            return [
                " ".join(clean(row[key]) for key in readable if clean(row[key]))
                for _, row in labels.iterrows()
            ]

    # DiFuMo returns a numpy recarray rather than a plain list of strings.
    # Prefer human-readable columns when present, and avoid IDs/coordinates.
    dtype_names = getattr(labels, "dtype", None)
    dtype_names = getattr(dtype_names, "names", None)
    if dtype_names:
        preferred = (
            "difumo_names",
            "name",
            "names",
            "label",
            "labels",
            "component",
            "network",
            "yeo_networks7",
            "yeo_networks17",
        )
        for key in preferred:
            if key in dtype_names:
                return [clean(row[key]) for row in labels]
        readable = [key for key in dtype_names if not key.lower().endswith(("x", "y", "z", "id", "index"))]
        if readable:
            return [" ".join(clean(row[key]) for key in readable if clean(row[key])) for row in labels]

    return [clean(x) for x in labels]


def _maps_path(atlas: Any) -> str | None:
    keys = (
        "maps",
        "map",
        "filename",
        # BASC legacy return keys.
        "scale007",
        "scale012",
        "scale020",
        "scale036",
        "scale064",
        "scale122",
        "scale197",
        "scale325",
        "scale444",
        # Craddock legacy return keys.
        "scorr_mean",
        "tcorr_mean",
        "scorr_2level",
        "tcorr_2level",
    )
    if isinstance(atlas, dict):
        for key in keys:
            path = _first_existing_path(atlas.get(key))
            if path:
                return path
    for key in keys:
        path = _first_existing_path(getattr(atlas, key, None))
        if path:
            return path
    return None


def _maps_img(atlas: Any):
    maps = _atlas_value(atlas, "maps")
    if isinstance(maps, nib.spatialimages.SpatialImage):
        return maps
    filename = _atlas_value(atlas, "filename")
    if isinstance(filename, nib.spatialimages.SpatialImage):
        return filename
    path = _maps_path(atlas)
    if path is None:
        return None
    return nib.load(path)


def fetch_atlas(name: str):
    from nilearn import datasets

    name = name.lower()
    if name == "yeo_2011":
        return datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
    if name == "yeo_2011_17":
        return datasets.fetch_atlas_yeo_2011(n_networks=17, thickness="thick")
    if name == "schaefer_2018":
        return datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    if name == "schaefer_2018_200":
        return datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    if name == "harvard_oxford_cortical":
        return datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    if name == "harvard_oxford_subcortical":
        return datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    if name in {"juelich", "juelich_maxprob"}:
        return datasets.fetch_atlas_juelich("maxprob-thr25-2mm")
    if name in {"juelich_probabilistic", "juelich_prob"}:
        return datasets.fetch_atlas_juelich("prob-2mm")
    if name == "aal":
        return datasets.fetch_atlas_aal(version="SPM12")
    if name == "smith_2009":
        return datasets.fetch_atlas_smith_2009(dimension=10, resting=True)
    if name in {"difumo", "difumo_64"}:
        try:
            return datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2, legacy_format=False)
        except TypeError as exc:
            if "legacy_format" not in str(exc):
                raise
            return datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2)
    if name == "difumo_128":
        try:
            return datasets.fetch_atlas_difumo(dimension=128, resolution_mm=2, legacy_format=False)
        except TypeError as exc:
            if "legacy_format" not in str(exc):
                raise
            return datasets.fetch_atlas_difumo(dimension=128, resolution_mm=2)
    if name == "msdl":
        return datasets.fetch_atlas_msdl()
    if name in {"basc", "basc_064"}:
        return datasets.fetch_atlas_basc_multiscale_2015(resolution=64, version="sym")
    if name == "basc_122":
        return datasets.fetch_atlas_basc_multiscale_2015(resolution=122, version="sym")
    if name in {"craddock", "craddock_spatial"}:
        return datasets.fetch_atlas_craddock_2012(homogeneity="spatial", grp_mean=True)
    if name == "craddock_temporal":
        return datasets.fetch_atlas_craddock_2012(homogeneity="temporal", grp_mean=True)
    raise ValueError(f"Unknown atlas: {name}")


def infer_map_kind(name: str, img) -> str:
    if img.ndim == 4 and img.shape[3] == 1:
        return "network_map" if "yeo" in name.lower() else "atlas_region"
    if img.ndim == 4:
        if any(key in name.lower() for key in ("smith", "difumo", "msdl")):
            return "network_map"
        if any(key in name.lower() for key in ("craddock",)):
            return "probabilistic_atlas_region"
        return "probabilistic_atlas_region"
    if any(key in name.lower() for key in ("basc", "craddock")):
        return "network_map"
    return "atlas_region"


def _positive_for_label(label_name: str, atlas_name: str, map_type: str) -> tuple[dict, bool]:
    display = display_atlas_label(label_name, atlas_name)
    definition, fallback = atlas_label_definition(label_name, atlas_name)
    atlas_key = atlas_name.lower()
    is_network = (
        map_type in {"network_map", "ica_component"}
        or "network" in label_name.lower()
        or "network" in display.lower()
        or "yeo" in atlas_key
        or "smith" in atlas_key
        or "difumo" in atlas_key
        or "msdl" in atlas_key
        or "basc" in atlas_key
        or "craddock" in atlas_key
    )
    source = "nilearn_network_label" if is_network else "nilearn_atlas_label"
    category = "network" if source == "nilearn_network_label" else "anatomical_region"
    return (
        {
            "text": text_pair(display, definition),
            "term": display,
            "category": category,
            "source": source,
            "weight": POSITIVE_WEIGHTS[source],
            "reliability": "strong" if not fallback else "medium",
        },
        fallback,
    )


def _row(
    *,
    map_id: str,
    atlas_name: str,
    map_type: str,
    nifti_path: Path,
    label_name: str,
    label_id: int,
    positive: dict,
    definition_fallback: bool,
    preprocessing_config: dict[str, Any],
) -> dict:
    meta = nifti_metadata(nifti_path)
    return {
        "map_id": map_id,
        "source": f"nilearn:{atlas_name}",
        "map_type": map_type,
        "nifti_path": str(nifti_path),
        "tensor_path": None,
        "space": "MNI152_2mm",
        "affine": meta["affine"],
        "resolution": meta["resolution"],
        "shape": meta["shape"],
        "preprocessing_config": preprocessing_config,
        "positive_texts": [positive],
        "positive_terms": [positive["term"]],
        "positive_categories": [positive["category"]],
        "negative_sampling_groups": {"source": "nilearn", "atlas": atlas_name, "map_type": map_type},
        "pmid": None,
        "mesh_terms": [],
        "quality_flags": {
            "definition_fallback": bool(definition_fallback),
            "is_coordinate_derived": False,
            "no_positive_texts": False,
            "label_id": int(label_id),
            "label_name": label_name,
        },
    }


def _iter_3d_components(data: np.ndarray, labels: list[str], atlas_name: str, map_type: str) -> Iterable[tuple[int, str, np.ndarray, bool]]:
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        n = data.shape[3]
        for i in range(n):
            label = labels[i] if i < len(labels) else f"{atlas_name} component {i + 1}"
            yield i + 1, label, data[..., i].astype(np.float32), False
    else:
        unique = [int(v) for v in np.unique(data) if int(v) != 0]
        for label_id in unique:
            label_idx = label_id
            label = labels[label_idx] if label_idx < len(labels) else f"{atlas_name} label {label_id}"
            if str(label).strip().lower() in {"background", "0"}:
                continue
            yield label_id, label, (data == label_id).astype(np.float32), True


def _custom_atlas_rows(custom_cfg: list[dict[str, Any]], paths: dict[str, Any], dataset_cfg: dict[str, Any]) -> list[dict]:
    """Ingest optional user-provided priority-3 atlases from local NIfTI files."""

    out_dir = Path(paths.get("map_cache_dir", "atlas_free_multipositive/cache/maps")) / "custom"
    target_img = load_target_mni152_2mm()
    rows: list[dict] = []
    preprocessing_config = {
        "target_space": "MNI152_2mm",
        "target_resolution_mm": dataset_cfg.get("target_resolution_mm", 2.0),
        "source": "custom_nifti_atlas",
    }
    for spec in custom_cfg or []:
        atlas_name = str(spec.get("name", "custom_atlas"))
        path = Path(spec.get("path", ""))
        if not path.exists():
            print(f"Skipping custom atlas {atlas_name}: missing path {path}")
            continue
        labels = spec.get("labels") or []
        map_type = spec.get("map_type", "probabilistic_atlas_region")
        img = nib.load(str(path))
        data = np.asarray(img.get_fdata())
        for label_id, label_name, component, is_binary in _iter_3d_components(data, labels, atlas_name, map_type):
            component_img = nib.Nifti1Image(component, img.affine, img.header)
            resampled = resample_to_target(component_img, target_img, binary=is_binary, clamp_nonnegative=not is_binary)
            label_slug = slugify(label_name)
            map_id = f"custom_{slugify(atlas_name)}_{label_id}_{label_slug}"
            out_path = out_dir / slugify(atlas_name) / f"{map_id}.nii.gz"
            save_nifti(resampled, out_path)
            positive, fallback = _positive_for_label(label_name, atlas_name, map_type)
            row = _row(
                map_id=map_id,
                atlas_name=atlas_name,
                map_type=map_type,
                nifti_path=out_path,
                label_name=label_name,
                label_id=label_id,
                positive=positive,
                definition_fallback=fallback,
                preprocessing_config=preprocessing_config,
            )
            row["source"] = f"custom_atlas:{atlas_name}"
            row["negative_sampling_groups"]["source"] = "custom_atlas"
            rows.append(row)
    return rows


def build_nilearn_rows(atlas_names: list[str], paths: dict[str, Any], dataset_cfg: dict[str, Any]) -> list[dict]:
    out_dir = Path(paths.get("map_cache_dir", "atlas_free_multipositive/cache/maps")) / "nilearn"
    target_img = load_target_mni152_2mm()
    rows: list[dict] = []
    preprocessing_config = {
        "target_space": "MNI152_2mm",
        "target_resolution_mm": dataset_cfg.get("target_resolution_mm", 2.0),
        "binary_resample_interpolation": "nearest",
        "continuous_resample_interpolation": "continuous",
    }

    for atlas_name in atlas_names:
        try:
            atlas = fetch_atlas(atlas_name)
            img = _maps_img(atlas)
            if img is None:
                print(f"Skipping {atlas_name}: no maps path")
                continue
        except Exception as exc:
            print(f"Skipping {atlas_name}: {exc}")
            continue
        data = np.asarray(img.get_fdata())
        labels = _labels(atlas)
        map_type = infer_map_kind(atlas_name, img)
        for label_id, label_name, component, is_binary in _iter_3d_components(data, labels, atlas_name, map_type):
            component_img = nib.Nifti1Image(component, img.affine, img.header)
            resampled = resample_to_target(component_img, target_img, binary=is_binary, clamp_nonnegative=not is_binary)
            label_slug = slugify(label_name)
            map_id = f"nilearn_{slugify(atlas_name)}_{label_id}_{label_slug}"
            out_path = out_dir / slugify(atlas_name) / f"{map_id}.nii.gz"
            save_nifti(resampled, out_path)
            positive, fallback = _positive_for_label(label_name, atlas_name, map_type)
            rows.append(
                _row(
                    map_id=map_id,
                    atlas_name=atlas_name,
                    map_type=map_type,
                    nifti_path=out_path,
                    label_name=label_name,
                    label_id=label_id,
                    positive=positive,
                    definition_fallback=fallback,
                    preprocessing_config=preprocessing_config,
                )
            )
    rows.extend(_custom_atlas_rows(dataset_cfg.get("custom_nifti_atlases", []), paths, dataset_cfg))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", default="atlas_free_multipositive/configs/paths.yaml")
    p.add_argument("--config", default="atlas_free_multipositive/configs/dataset_config.yaml")
    p.add_argument("--output", default="atlas_free_multipositive/cache/unified_jsonl/nilearn_atlases.jsonl")
    p.add_argument("--atlases", nargs="*", default=None)
    args = p.parse_args()
    paths = _load_yaml(args.paths)
    cfg = _load_yaml(args.config)
    atlas_names = args.atlases or cfg.get("nilearn_atlases", [])
    rows = build_nilearn_rows(atlas_names, paths, cfg)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} Nilearn rows to {args.output}")


if __name__ == "__main__":
    main()
