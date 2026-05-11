"""Rewrite relative paths inside unified JSONL files.

This is useful when Colab/Drive layouts differ from the local machine that
built the dataset. The JSONL should point to paths that exist relative to the
runtime repo root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rewrite_jsonl_paths(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    old: str,
    new: str,
    fields: tuple[str, ...] = ("tensor_path", "nifti_path"),
) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path
    rows = []
    changed = 0
    with input_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for field in fields:
                value = row.get(field)
                if isinstance(value, str) and old in value:
                    row[field] = value.replace(old, new)
                    changed += 1
            rows.append(row)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_path.replace(output_path)
    return changed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    args = p.parse_args()
    total = 0
    for path in args.jsonl:
        changed = rewrite_jsonl_paths(path, old=args.old, new=args.new)
        total += changed
        print(f"{path}: rewrote {changed} path values")
    print(f"total rewritten path values: {total}")


if __name__ == "__main__":
    main()

