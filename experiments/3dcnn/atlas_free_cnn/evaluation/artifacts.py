"""Shared artifact I/O and finalized test-split resolution for comparisons."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from atlas_free_cnn import notebook_utils


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fields = list(fieldnames or [])
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def resolve_test_jsonl(test_jsonl: str | Path | None) -> Path:
    if test_jsonl is not None:
        return Path(test_jsonl).expanduser()
    return notebook_utils.discover_default_unified_split_dir() / "test.jsonl"
