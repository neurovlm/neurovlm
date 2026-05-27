"""Pack atlas-free CNN volumes and text pairs for Hugging Face upload.

This script converts the mixed PubMed/Nilearn/NeuroVault JSONL artifacts into a
single compact dataset:

* ``atlas_free_cnn_volumes.pt``: dense ``[N, 1, D, H, W]`` CNN input tensor.
* ``atlas_free_cnn_rows.parquet``: one row per map with split/source metadata.
* ``atlas_free_cnn_text_pairs.parquet``: one row per positive text pair.
* ``atlas_free_cnn_manifest.json``: shape/count/config summary.

The text-pair table includes ``is_primary_text``. That primary text is chosen by
metadata specificity, not by model similarity, so held-out evaluation does not
leak model preferences into the target labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.data_building.ingest_neurovault import (  # noqa: E402
    NeuroVaultConfig,
    preprocess_neurovault_nifti,
)
from atlas_free_cnn.data_building.text_registry import read_jsonl  # noqa: E402


@dataclass(frozen=True)
class PackConfig:
    target_resolution_mm: float = 4.0
    crop_to_brain: bool = True
    positive_only: bool = True
    cache_dtype: str = "float16"
    include_weak_neurovault: bool = True
    neurovault_max_per_collection: int | None = 50
    show_progress: bool = True
    progress_interval: int = 100
    primary_text_only: bool = False
    primary_text_categories: tuple[str, ...] = (
        # PubMed: use title + summary/abstract-style text before titles alone.
        "paper_summary",
        "wiki_style_summary",
        # NeuroVault: prefer image/collection title + description when present,
        # then fall back to task/contrast labels.
        "image_description",
        "neurovault_image",
        "collection_description",
        "neurovault_collection",
        "cognitive_task_or_contrast",
        "neurovault_task_label",
        # Nilearn: keep atlas/network labels as the primary text.
        "nilearn_network_label",
        "nilearn_atlas_label",
        "network",
        "anatomical_region",
        "paper_title",
        "image_name",
    )


def _split_from_path(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    if stem in {"train", "val", "valid", "validation", "test"}:
        return "val" if stem in {"valid", "validation"} else stem
    parent = Path(path).parent.name.lower()
    if parent in {"train", "val", "valid", "validation", "test", "splits"}:
        if stem in {"train", "val", "valid", "validation", "test"}:
            return "val" if stem in {"valid", "validation"} else stem
    return "unsplit"


def _torch_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError("cache_dtype must be one of {'float16', 'bfloat16', 'float32'}")


def _load_tensor_from_row(row: dict[str, Any], tensor_cache: dict[str, Any] | None = None):
    import torch

    tensor_path = row.get("tensor_path")
    if not tensor_path:
        return None
    tensor_path = str(tensor_path)
    if tensor_cache is not None:
        if tensor_path not in tensor_cache:
            tensor_cache[tensor_path] = torch.load(tensor_path, map_location="cpu", weights_only=False)
        payload = tensor_cache[tensor_path]
    else:
        payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "volumes" in payload:
        tensor = payload["volumes"][int(row["tensor_index"])]
    else:
        tensor = payload[int(row.get("tensor_index", 0))]
    tensor = torch.as_tensor(tensor).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _load_or_preprocess_volume(row: dict[str, Any], config: PackConfig, tensor_cache: dict[str, Any] | None = None):
    import torch

    tensor = _load_tensor_from_row(row, tensor_cache=tensor_cache)
    if tensor is not None:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            return tensor.float(), {"loaded_from": "tensor_path"}, {}
        raise ValueError(f"Unexpected tensor shape for {row.get('map_id')}: {tuple(tensor.shape)}")

    nifti_path = row.get("nifti_path")
    if not nifti_path:
        raise ValueError(f"Row {row.get('map_id')} has no tensor_path or nifti_path")
    nv_cfg = NeuroVaultConfig(
        target_resolution_mm=config.target_resolution_mm,
        crop_to_brain=config.crop_to_brain,
        positive_only=config.positive_only,
    )
    tensor, meta, flags = preprocess_neurovault_nifti(nifti_path, config=nv_cfg)
    if tensor is None:
        raise RuntimeError(f"Could not preprocess {row.get('map_id')}: {meta.get('error', flags)}")
    return torch.as_tensor(tensor).float(), meta, flags


def text_pair_priority(pos: dict[str, Any], config: PackConfig) -> tuple[int, float]:
    category = str(pos.get("category", ""))
    source = str(pos.get("source", ""))
    priority_lookup = {name: i for i, name in enumerate(config.primary_text_categories)}
    if source in priority_lookup:
        rank = priority_lookup[source]
    else:
        rank = priority_lookup.get(category, len(priority_lookup))
    weight = float(pos.get("weight", 0.0) or 0.0)
    reliability_bonus = {"strong": 0.2, "medium": 0.1}.get(str(pos.get("reliability", "")), 0.0)
    return rank, -(weight + reliability_bonus)


def text_pairs_for_row(row: dict[str, Any], config: PackConfig) -> list[dict[str, Any]]:
    positives = [dict(pos) for pos in row.get("positive_texts", []) if str(pos.get("text", "")).strip()]
    positives.sort(key=lambda pos: text_pair_priority(pos, config))
    out = []
    for rank, pos in enumerate(positives):
        out.append(
            {
                "map_id": row["map_id"],
                "split": row.get("split", "unsplit"),
                "text": pos.get("text", ""),
                "term": pos.get("term", ""),
                "category": pos.get("category", ""),
                "source": pos.get("source", ""),
                "weight": float(pos.get("weight", 1.0) or 1.0),
                "reliability": pos.get("reliability", ""),
                "pair_rank": rank,
                "is_primary_text": rank == 0,
            }
        )
    return out


def load_split_jsonl_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        split = _split_from_path(path)
        for row in read_jsonl(path):
            row = dict(row)
            row.setdefault("split", split)
            if split != "unsplit":
                row["split"] = split
            rows.append(row)
    return rows


def deterministic_split(key: str, *, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    bucket = int(hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def _stable_hash_int(value: str) -> int:
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12], 16)


def load_neurovault_rows(
    neurovault_dir: str | Path,
    *,
    include_weak: bool = True,
    split: str = "train",
    max_per_collection: int | None = 50,
) -> list[dict[str, Any]]:
    """Convert staged NeuroVault outputs into unified map-text rows."""

    neurovault_dir = Path(neurovault_dir)
    manifest = pd.read_csv(neurovault_dir / "neurovault_manifest.csv")
    positives = [json.loads(line) for line in (neurovault_dir / "neurovault_text_positives.jsonl").read_text().splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos in positives:
        grouped[str(pos["map_id"])].append(pos)
    tiers = {"strong", "weak"} if include_weak else {"strong"}
    manifest = manifest.copy()
    manifest["_map_id"] = manifest["map_id"].astype(str)
    manifest["_collection_key"] = manifest.get("collection_id", "").fillna("").astype(str)
    if "collection_name" in manifest:
        missing_collection = manifest["_collection_key"].isin({"", "nan", "None"})
        manifest.loc[missing_collection, "_collection_key"] = manifest.loc[missing_collection, "collection_name"].fillna("").astype(str)
    manifest["_tier_rank"] = manifest["quality_tier"].map({"strong": 0, "weak": 1}).fillna(9).astype(int)
    manifest["_stable_rank"] = manifest["_map_id"].map(_stable_hash_int)
    manifest = manifest.sort_values(
        ["_collection_key", "_tier_rank", "quality_score", "_stable_rank"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    collection_counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for _, rec in manifest.iterrows():
        tier = str(rec.get("quality_tier", ""))
        map_id = str(rec.get("map_id", ""))
        if tier not in tiers or not grouped.get(map_id) or pd.isna(rec.get("tensor_index")):
            continue
        collection_key = str(rec.get("_collection_key", "") or "")
        if max_per_collection is not None and max_per_collection > 0:
            if collection_counts[collection_key] >= max_per_collection:
                continue
            collection_counts[collection_key] += 1
        row_split = deterministic_split(map_id) if split == "auto" else split
        rows.append(
            {
                "map_id": map_id,
                "source": "neurovault",
                "map_type": "statistical_map",
                "tensor_path": str(neurovault_dir / "neurovault_cnn_volumes.pt"),
                "tensor_index": int(rec["tensor_index"]),
                "nifti_path": None,
                "split": row_split,
                "space": "MNI152_4mm_crop",
                "positive_texts": grouped[map_id],
                "positive_terms": sorted({p.get("term", "") for p in grouped[map_id] if p.get("term")}),
                "positive_categories": sorted({p.get("category", "") for p in grouped[map_id] if p.get("category")}),
                "quality_score": int(rec.get("quality_score", 0)),
                "quality_tier": tier,
                "neurovault_collection_key": collection_key,
                "pmid": str(rec.get("pmid", "")) if not pd.isna(rec.get("pmid")) else "",
                "doi": str(rec.get("doi", "")) if not pd.isna(rec.get("doi")) else "",
                "negative_sampling_groups": {"source": "neurovault", "collection_id": str(rec.get("collection_id", ""))},
                "selection": {
                    "collection_cap": max_per_collection,
                    "collection_rank": collection_counts.get(collection_key, 0),
                },
                "quality_flags": {key: bool(rec.get(key, False)) for key in (
                    "missing_metadata",
                    "no_task_label",
                    "no_doi_or_pmid",
                    "weird_shape",
                    "mostly_empty",
                    "negative_values_present",
                    "thresholded_map_possible",
                    "failed_resample",
                    "low_quality_text",
                )},
            }
        )
    return rows


def pack_rows(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    config: PackConfig = PackConfig(),
) -> dict[str, Path]:
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volumes = []
    map_rows = []
    text_rows = []
    skipped = []
    expected_shape = None
    dtype = _torch_dtype(config.cache_dtype)
    tensor_cache: dict[str, Any] = {}
    progress_bar = None
    if config.show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(total=len(rows), desc="Packing atlas-free CNN rows", unit="row")
        except ImportError:
            progress_bar = None

    def update_progress(source_index: int, *, force: bool = False) -> None:
        status = {
            "packed": len(volumes),
            "skipped": len(skipped),
            "text_pairs": len(text_rows),
            "cached_tensors": len(tensor_cache),
        }
        if progress_bar is not None:
            progress_bar.set_postfix(**status)
            return
        interval = max(1, int(config.progress_interval))
        if config.show_progress and (force or source_index == 0 or (source_index + 1) % interval == 0):
            print(f"Packing progress: row={source_index + 1}/{len(rows)} {status}", flush=True)

    try:
        for source_index, row in enumerate(rows):
            if progress_bar is not None:
                progress_bar.update(1)
            if not row.get("positive_texts"):
                skipped.append({"map_id": row.get("map_id"), "reason": "no_positive_texts"})
                update_progress(source_index)
                continue
            try:
                tensor, pre_meta, flags = _load_or_preprocess_volume(row, config, tensor_cache=tensor_cache)
            except Exception as exc:
                skipped.append({"map_id": row.get("map_id"), "reason": repr(exc)})
                update_progress(source_index)
                continue
            if expected_shape is None:
                expected_shape = tuple(tensor.shape)
            if tuple(tensor.shape) != expected_shape:
                skipped.append({"map_id": row.get("map_id"), "reason": f"shape {tuple(tensor.shape)} != {expected_shape}"})
                update_progress(source_index)
                continue
            tensor_index = len(volumes)
            volumes.append(tensor.to(dtype))
            row_pairs = text_pairs_for_row(row, config)
            if config.primary_text_only:
                row_pairs = row_pairs[:1]
            text_rows.extend(row_pairs)
            map_rows.append(
                {
                    "map_id": row["map_id"],
                    "tensor_index": tensor_index,
                    "split": row.get("split", "unsplit"),
                    "source": row.get("source", ""),
                    "map_type": row.get("map_type", ""),
                    "pmid": row.get("pmid", ""),
                    "doi": row.get("doi", ""),
                    "quality_score": row.get("quality_score", np.nan),
                    "quality_tier": row.get("quality_tier", ""),
                    "neurovault_collection_key": row.get("neurovault_collection_key", ""),
                    "n_text_pairs": len(row_pairs),
                    "primary_text": row_pairs[0]["text"] if row_pairs else "",
                    "source_index": source_index,
                    "preprocess_metadata": json.dumps(pre_meta, sort_keys=True),
                    "preprocess_flags": json.dumps(flags, sort_keys=True),
                }
            )
            update_progress(source_index)
    finally:
        update_progress(len(rows) - 1 if rows else 0, force=True)
        if progress_bar is not None:
            progress_bar.close()

    if not volumes:
        raise RuntimeError("No rows could be packed into CNN volumes.")

    volumes_tensor = torch.stack(volumes).contiguous()
    volumes_path = output_dir / "atlas_free_cnn_volumes.pt"
    rows_path = output_dir / "atlas_free_cnn_rows.parquet"
    text_path = output_dir / "atlas_free_cnn_text_pairs.parquet"
    manifest_path = output_dir / "atlas_free_cnn_manifest.json"
    torch.save(
        {
            "version": 1,
            "volumes": volumes_tensor,
            "map_ids": np.asarray([row["map_id"] for row in map_rows]).astype(str),
            "config": asdict(config),
            "shape": list(volumes_tensor.shape),
        },
        volumes_path,
    )
    pd.DataFrame(map_rows).to_parquet(rows_path, index=False)
    pd.DataFrame(text_rows).to_parquet(text_path, index=False)
    manifest = {
        "version": 1,
        "config": asdict(config),
        "files": {
            "volumes": volumes_path.name,
            "rows": rows_path.name,
            "text_pairs": text_path.name,
        },
        "counts": {
            "maps": len(map_rows),
            "text_pairs": len(text_rows),
            "skipped": len(skipped),
            "splits": pd.Series([row["split"] for row in map_rows]).value_counts().to_dict(),
            "sources": pd.Series([row["source"] for row in map_rows]).value_counts().to_dict(),
            "neurovault_collections": pd.Series(
                [row["neurovault_collection_key"] for row in map_rows if row["source"] == "neurovault"]
            ).value_counts().to_dict(),
        },
        "volume_shape": list(volumes_tensor.shape),
        "skipped_examples": skipped[:100],
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return {
        "volumes": volumes_path,
        "rows": rows_path,
        "text_pairs": text_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", nargs="*", default=[], help="Unified JSONL files, usually split train/val/test files.")
    parser.add_argument("--neurovault-dir", default=None, help="Optional staged NeuroVault output directory.")
    parser.add_argument("--neurovault-split", default="auto", help="Split label for NeuroVault rows, or 'auto' for deterministic train/val/test.")
    parser.add_argument(
        "--neurovault-max-per-collection",
        type=int,
        default=50,
        help="Cap accepted NeuroVault maps per collection during packing. Use 0 for no cap.",
    )
    parser.add_argument("--strong-neurovault-only", action="store_true")
    parser.add_argument("--output-dir", default="experiments/3dcnn/atlas_free_cnn/cache/hf_atlas_free_cnn")
    parser.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--primary-text-only", action="store_true", help="Write only the selected primary text pair per map.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm/progress status output while packing.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print status every N rows when tqdm is unavailable.")
    args = parser.parse_args()

    max_per_collection = None if args.neurovault_max_per_collection <= 0 else args.neurovault_max_per_collection
    cfg = PackConfig(
        cache_dtype=args.cache_dtype,
        include_weak_neurovault=not args.strong_neurovault_only,
        neurovault_max_per_collection=max_per_collection,
        show_progress=not args.no_progress,
        progress_interval=args.progress_interval,
        primary_text_only=args.primary_text_only,
    )
    rows = load_split_jsonl_rows(args.jsonl)
    if args.neurovault_dir:
        rows.extend(
            load_neurovault_rows(
                args.neurovault_dir,
                include_weak=cfg.include_weak_neurovault,
                split=args.neurovault_split,
                max_per_collection=cfg.neurovault_max_per_collection,
            )
        )
    outputs = pack_rows(rows, args.output_dir, config=cfg)
    print("Packed atlas-free CNN dataset:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
