"""Export packed atlas-free CNN parquet metadata back to training JSONL.

The exported rows point every source at ``atlas_free_cnn_volumes.pt`` so PubMed,
NeuroVault, and Nilearn all load as the shared CNN tensor shape
``(1, 36, 45, 38)``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return ""
    return value


def _positive_texts(pairs: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for pair in pairs.sort_values("pair_rank").to_dict("records"):
        out.append(
            {
                "text": _clean(pair.get("text")),
                "term": _clean(pair.get("term")),
                "category": _clean(pair.get("category")),
                "source": _clean(pair.get("source")),
                "weight": float(pair.get("weight", 1.0) or 1.0),
                "reliability": _clean(pair.get("reliability")),
                "text_id": f"{pair['map_id']}::pair_{int(pair.get('pair_rank', 0))}",
            }
        )
    return out


def export_jsonl(pack_dir: str | Path, output_dir: str | Path) -> dict[str, dict[str, int]]:
    pack_dir = Path(pack_dir)
    output_dir = Path(output_dir)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(pack_dir / "atlas_free_cnn_rows.parquet")
    pairs = pd.read_parquet(pack_dir / "atlas_free_cnn_text_pairs.parquet")
    pairs_by_map = {map_id: group for map_id, group in pairs.groupby("map_id", sort=False)}
    tensor_path = str(pack_dir / "atlas_free_cnn_volumes.pt")

    exported: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        map_id = str(row["map_id"])
        positive_texts = _positive_texts(pairs_by_map.get(map_id, pd.DataFrame()))
        if not positive_texts:
            continue
        source = str(_clean(row.get("source")))
        exported.append(
            {
                "map_id": map_id,
                "source": source,
                "source_detail": source,
                "split": str(_clean(row.get("split")) or "unsplit"),
                "map_type": _clean(row.get("map_type")),
                "pmid": str(_clean(row.get("pmid"))),
                "doi": str(_clean(row.get("doi"))),
                "tensor_path": tensor_path,
                "tensor_index": int(row["tensor_index"]),
                "nifti_path": None,
                "space": "MNI152_4mm_crop",
                "shape": [36, 45, 38],
                "positive_texts": positive_texts,
                "positive_terms": sorted({p["term"] for p in positive_texts if p.get("term")}),
                "positive_categories": sorted({p["category"] for p in positive_texts if p.get("category")}),
                "quality_score": _clean(row.get("quality_score")),
                "quality_tier": _clean(row.get("quality_tier")),
                "neurovault_collection_key": _clean(row.get("neurovault_collection_key")),
                "preprocess_metadata": _clean(row.get("preprocess_metadata")),
                "preprocess_flags": _clean(row.get("preprocess_flags")),
                "negative_sampling_groups": {"source": source.split(":", 1)[0]},
            }
        )

    def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item, sort_keys=True) + "\n")

    write_jsonl(output_dir / "unified_map_text.jsonl", exported)
    counts: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_rows = [row for row in exported if row["split"] == split]
        write_jsonl(split_dir / f"{split}.jsonl", split_rows)
        with (split_dir / f"{split}_map_ids.json").open("w") as f:
            json.dump([row["map_id"] for row in split_rows], f, indent=2)
        counts[split] = dict(Counter(row["source"].split(":", 1)[0] for row in split_rows))

    with (output_dir / "source_counts_by_split.json").open("w") as f:
        json.dump(counts, f, indent=2, sort_keys=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", default="experiments/3dcnn/atlas_free_cnn/cache/hf_atlas_free_cnn")
    parser.add_argument("--output-dir", default="experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl")
    args = parser.parse_args()
    counts = export_jsonl(args.pack_dir, args.output_dir)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
