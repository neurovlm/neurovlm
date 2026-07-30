"""Deterministic, atomic serialization used by training pipelines."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def json_safe(value: Any) -> Any:
    """Convert common scientific Python values into strict JSON values.

    Non-finite floating-point values intentionally become ``null``.  This
    keeps manifests portable and prevents Python's non-standard ``NaN`` and
    ``Infinity`` tokens from leaking into run metadata.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [json_safe(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))

    # Keep NumPy and PyTorch optional at import time.
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy":
        if hasattr(value, "tolist"):
            return json_safe(value.tolist())
        if hasattr(value, "item"):
            return json_safe(value.item())
    if module == "torch" and hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return json_safe(tensor.item() if tensor.ndim == 0 else tensor.tolist())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _temporary_path(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return descriptor, Path(name)


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Atomically replace *path* with deterministic, strict JSON."""

    path = Path(path)
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                json_safe(value),
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def union_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return all CSV columns in stable first-seen order."""

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            name = str(field)
            if name not in seen:
                seen.add(name)
                fields.append(name)
    return fields


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Atomically write CSV rows, retaining the union of heterogeneous fields."""

    path = Path(path)
    materialized = [dict(row) for row in rows]
    fields = list(fieldnames) if fieldnames is not None else union_fieldnames(materialized)
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in materialized:
                converted = json_safe(row)
                writer.writerow(
                    {
                        field: (
                            json.dumps(converted.get(field), sort_keys=True)
                            if isinstance(converted.get(field), (dict, list))
                            else converted.get(field)
                        )
                        for field in fields
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries; missing files yield no rows."""

    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def append_csv_union(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Append logically while atomically rewriting with the union schema."""

    existing = read_csv_rows(path)
    return atomic_write_csv(path, [*existing, *rows])


__all__ = [
    "append_csv_union",
    "atomic_write_csv",
    "atomic_write_json",
    "json_safe",
    "read_csv_rows",
    "union_fieldnames",
]
