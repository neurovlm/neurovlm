"""Combine source JSONL files into a unified map-text dataset and registry."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from atlas_free_multipositive.data_building.text_registry import attach_text_ids, read_jsonl, write_jsonl


def split_group_key(row: dict) -> str:
    """Return the stratum used to preserve source proportions across splits."""

    source = str(row.get("source") or "")
    if source:
        return source
    if row.get("pmid"):
        return "pubmed"
    return "unknown"


def split_rows(rows: list[dict], *, seed: int, val_frac: float, test_frac: float) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[split_group_key(row)].append(row)

    def split_one(group: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        return group[n_test + n_val :], group[n_test : n_test + n_val], group[:n_test]

    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        train, val, test = split_one(groups[key])
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)
    for split_rows_ in splits.values():
        rng.shuffle(split_rows_)
    return splits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", default="atlas_free_multipositive/cache/unified_jsonl/unified_map_text.jsonl")
    p.add_argument("--text-registry", default="atlas_free_multipositive/cache/unified_jsonl/text_registry.jsonl")
    p.add_argument("--split-dir", default="atlas_free_multipositive/cache/unified_jsonl/splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    args = p.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))
    rows = [r for r in rows if r.get("positive_texts")]
    rows, registry = attach_text_ids(rows)
    write_jsonl(rows, args.output)
    write_jsonl(registry.values(), args.text_registry)

    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    splits = split_rows(rows, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)
    for name, split_rows_ in splits.items():
        write_jsonl(split_rows_, split_dir / f"{name}.jsonl")
        with (split_dir / f"{name}_map_ids.json").open("w") as f:
            json.dump([r["map_id"] for r in split_rows_], f, indent=2)

    print(f"Wrote {len(rows)} unified rows and {len(registry)} unique texts")


if __name__ == "__main__":
    main()
