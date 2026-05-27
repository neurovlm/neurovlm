"""Compare old PubMed ALE cache preprocessing with packed atlas-free CNN data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def tensor_stats(x: torch.Tensor) -> dict[str, float | list[int] | str]:
    xf = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "nonzero_fraction": float((xf > 0).float().mean().item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-cache", default="experiments/3dcnn/atlas_free_cnn/data/ale_caches/atlas_free_4mm_fwhm9_crop_float16.pt")
    p.add_argument("--packed-volumes", default="experiments/3dcnn/atlas_free_cnn/cache/hf_atlas_free_cnn/atlas_free_cnn_volumes.pt")
    p.add_argument("--packed-rows", default="experiments/3dcnn/atlas_free_cnn/cache/hf_atlas_free_cnn/atlas_free_cnn_rows.parquet")
    p.add_argument("--n-compare", type=int, default=1000)
    p.add_argument("--output", default="experiments/3dcnn/atlas_free_cnn/cache/preprocessing_audit.json")
    args = p.parse_args()

    old = torch.load(args.old_cache, map_location="cpu", weights_only=False, mmap=True)
    packed = torch.load(args.packed_volumes, map_location="cpu", weights_only=False, mmap=True)
    rows = pd.read_parquet(args.packed_rows)

    pmid_to_old = {str(pmid): i for i, pmid in enumerate(old["pmids"])}
    pubmed = rows[rows["source"].eq("pubmed")].head(args.n_compare)
    max_diffs = []
    for _, row in pubmed.iterrows():
        old_idx = pmid_to_old.get(str(row["pmid"]))
        if old_idx is None:
            continue
        packed_vol = packed["volumes"][int(row["tensor_index"]), 0].float()
        old_vol = old["volumes"][old_idx].float()
        max_diffs.append(float((packed_vol - old_vol).abs().max().item()))

    source_stats = {}
    for source, group in rows.groupby("source"):
        indices = group["tensor_index"].astype(int).head(100).tolist()
        source_stats[source] = tensor_stats(packed["volumes"][indices])

    report = {
        "old_cache_config": old.get("config", {}),
        "old_cache_metadata": old.get("metadata", {}),
        "packed_config": packed.get("config", {}),
        "packed_shape": list(packed["volumes"].shape),
        "packed_source_counts": rows["source"].value_counts().to_dict(),
        "pubmed_compared": len(max_diffs),
        "pubmed_max_abs_diff_max": max(max_diffs) if max_diffs else None,
        "pubmed_max_abs_diff_mean": sum(max_diffs) / len(max_diffs) if max_diffs else None,
        "source_stats_first_100": source_stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:6000])


if __name__ == "__main__":
    main()
