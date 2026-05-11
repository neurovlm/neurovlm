"""Combine source JSONL files into a unified map-text dataset and registry."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from atlas_free_multipositive.data_building.text_registry import attach_text_ids, read_jsonl, write_jsonl


def split_rows(rows: list[dict], *, seed: int, val_frac: float, test_frac: float) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    pubmed = [r for r in rows if r.get("pmid")]
    atlas = [r for r in rows if not r.get("pmid")]
    rng.shuffle(pubmed)
    rng.shuffle(atlas)

    def split_one(group: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        return group[n_test + n_val :], group[n_test : n_test + n_val], group[:n_test]

    p_train, p_val, p_test = split_one(pubmed)
    a_train, a_val, a_test = split_one(atlas)
    return {
        "train": p_train + a_train,
        "val": p_val + a_val,
        "test": p_test + a_test,
    }


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

