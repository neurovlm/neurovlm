"""QC utilities for unified JSONL datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> dict:
    source_counts = Counter(str(r.get("source", "")).split(":")[0] for r in rows)
    map_type_counts = Counter(r.get("map_type", "") for r in rows)
    pos_category_counts = Counter()
    pos_source_counts = Counter()
    missing_paths = []
    shapes = Counter()
    for row in rows:
        path = row.get("nifti_path") or row.get("tensor_path")
        if path and not Path(path).exists():
            missing_paths.append(path)
        shapes[str(row.get("shape"))] += 1
        for pos in row.get("positive_texts", []):
            pos_category_counts[pos.get("category", "")] += 1
            pos_source_counts[pos.get("source", "")] += 1
    return {
        "n_rows": len(rows),
        "source_counts": dict(source_counts),
        "map_type_counts": dict(map_type_counts),
        "positive_category_counts": dict(pos_category_counts),
        "positive_source_counts": dict(pos_source_counts),
        "shape_counts": dict(shapes),
        "missing_path_count": len(missing_paths),
        "missing_path_examples": missing_paths[:10],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--samples", type=int, default=3)
    args = p.parse_args()
    rows = read_jsonl(args.jsonl)
    print(json.dumps(summarize(rows), indent=2))
    for row in rows[: args.samples]:
        print(json.dumps({k: row[k] for k in ("map_id", "source", "map_type", "positive_terms", "positive_categories")}, indent=2))


if __name__ == "__main__":
    main()

