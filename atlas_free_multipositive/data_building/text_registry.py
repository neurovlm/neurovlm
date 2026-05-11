"""Small JSONL text registry for deduplicated positive strings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def text_id(text: str) -> str:
    return "txt_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def attach_text_ids(rows: Iterable[dict]) -> tuple[list[dict], dict[str, dict]]:
    registry: dict[str, dict] = {}
    out_rows: list[dict] = []
    for row in rows:
        row = dict(row)
        positives = []
        for pos in row.get("positive_texts", []):
            pos = dict(pos)
            tid = text_id(pos["text"])
            pos["text_id"] = tid
            registry.setdefault(
                tid,
                {
                    "text_id": tid,
                    "text": pos["text"],
                    "term": pos.get("term", ""),
                    "category": pos.get("category", ""),
                    "source": pos.get("source", ""),
                },
            )
            positives.append(pos)
        row["positive_texts"] = positives
        out_rows.append(row)
    return out_rows, registry


def write_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

