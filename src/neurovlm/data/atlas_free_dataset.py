"""Hugging Face-backed atlas-free CNN datasets.

The published JSONL files contain historical ``tensor_path`` and
``nifti_path`` fields.  Those fields are metadata only: this module always
indexes the one published volume payload by ``tensor_index``.  Explicit local
split and volume paths are supported for custom datasets and offline use.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from neurovlm.resources import loaders as retrieval_resources

ATLAS_FREE_CNN_SPLITS = ("train", "val", "test")

__all__ = [
    "ATLAS_FREE_CNN_SPLITS",
    "AtlasFreeCNNDataset",
    "AtlasFreeCNNDataProvider",
    "atlas_free_cnn_splits",
    "canonical_atlas_free_domain",
]


def _validate_split(split: str) -> str:
    split = str(split).lower()
    if split not in ATLAS_FREE_CNN_SPLITS:
        expected = ", ".join(ATLAS_FREE_CNN_SPLITS)
        raise ValueError(f"Unknown atlas-free CNN split {split!r}; expected one of: {expected}.")
    return split


def _filter_values(value: str | Iterable[str] | None, *, name: str) -> set[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    normalized = {str(item).strip().lower() for item in values if str(item).strip()}
    if not normalized:
        raise ValueError(f"{name} filter must contain at least one non-empty value.")
    return normalized


def canonical_atlas_free_domain(row: Mapping[str, Any]) -> str:
    """Return the stable training domain represented by a metadata row."""

    source = str(row.get("source") or "").strip().lower()
    if source.startswith("pubmed") or row.get("pmid"):
        return "pubmed"
    if source.startswith("nilearn"):
        return "nilearn"
    if source.startswith("neurovault"):
        return "neurovault"
    if source.startswith("network"):
        return "network"
    return source or "unknown"


def _read_local_split(path: str | Path) -> tuple[dict[str, Any], ...]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Atlas-free CNN split does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}.") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object in {path} at line {line_number}.")
            rows.append(row)
    return tuple(rows)


def _read_local_volumes(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Atlas-free CNN volume payload does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected {path} to contain a dictionary payload.")
    return payload


def _coerce_tensor_index(value: Any, *, map_id: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Row {map_id!r} has invalid tensor_index {value!r}.")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {map_id!r} has invalid tensor_index {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Row {map_id!r} has invalid tensor_index {value!r}.")
    if isinstance(value, str) and value.strip() != str(index):
        raise ValueError(f"Row {map_id!r} has invalid tensor_index {value!r}.")
    return index


def _validated_volumes(payload: Mapping[str, Any]) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if "volumes" not in payload:
        raise TypeError("Atlas-free CNN volume payload must contain a 'volumes' tensor.")
    volumes = torch.as_tensor(payload["volumes"]).cpu()
    if volumes.ndim != 5 or volumes.shape[1] != 1 or any(size <= 0 for size in volumes.shape[2:]):
        raise ValueError(
            "Atlas-free CNN volumes must have shape N x 1 x D x H x W; "
            f"got {tuple(volumes.shape)}."
        )

    declared_shape = payload.get("shape")
    if declared_shape is not None:
        normalized_shape = tuple(int(size) for size in declared_shape)
        compatible_shapes = {tuple(volumes.shape), tuple(volumes.shape[-3:])}
        if normalized_shape not in compatible_shapes:
            raise ValueError(
                f"Volume payload shape metadata {tuple(declared_shape)} does not match tensor shape "
                f"{tuple(volumes.shape)}."
            )

    raw_map_ids = payload.get("map_ids")
    map_ids = None if raw_map_ids is None else tuple(str(value) for value in raw_map_ids)
    if map_ids is not None and len(map_ids) != len(volumes):
        raise ValueError(
            f"Volume payload has {len(volumes)} volumes but {len(map_ids)} map_ids."
        )
    return volumes, map_ids


def _validate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    volumes: torch.Tensor,
    payload_map_ids: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    validated_rows: list[dict[str, Any]] = []
    indices: list[int] = []
    spatial_shape = tuple(volumes.shape[2:])
    for position, source_row in enumerate(rows):
        row = dict(source_row)
        map_id = str(row.get("map_id") or "")
        if not map_id:
            raise ValueError(f"Atlas-free CNN {split} row {position} has no map_id.")

        row_split = str(row.get("split") or split).lower()
        if row_split != split:
            raise ValueError(
                f"Row {map_id!r} declares split {row_split!r}, but it was loaded as {split!r}."
            )

        index = _coerce_tensor_index(row.get("tensor_index"), map_id=map_id)
        if index < 0 or index >= len(volumes):
            raise IndexError(
                f"Row {map_id!r} tensor_index {index} is outside volume bounds [0, {len(volumes)})."
            )

        declared_shape = row.get("shape")
        if declared_shape is not None and tuple(int(size) for size in declared_shape) != spatial_shape:
            raise ValueError(
                f"Row {map_id!r} declares volume shape {tuple(declared_shape)}, "
                f"but the payload shape is {spatial_shape}."
            )

        if payload_map_ids is not None and payload_map_ids[index] != map_id:
            raise ValueError(
                f"Row {map_id!r} does not align with payload map_ids[{index}]="
                f"{payload_map_ids[index]!r}."
            )

        positives = row.get("positive_texts", [])
        if positives is None:
            row["positive_texts"] = []
        elif not isinstance(positives, list):
            raise TypeError(f"Row {map_id!r} positive_texts must be a list.")

        validated_rows.append(row)
        indices.append(index)
    return validated_rows, indices


class AtlasFreeCNNDataset(Dataset):
    """A split view over the shared atlas-free CNN volume tensor.

    By default, both the split JSONL and volume payload are resolved from
    ``neurovlm/atlas_free_cnn_dataset`` via the normal Hugging Face cache.
    ``split_path`` and ``volume_path`` opt into explicit local resources.
    """

    def __init__(
        self,
        split: str,
        *,
        source: str | Iterable[str] | None = None,
        domain: str | Iterable[str] | None = None,
        limit: int | None = None,
        split_path: str | Path | None = None,
        volume_path: str | Path | None = None,
        _rows: Iterable[Mapping[str, Any]] | None = None,
        _volume_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.split = _validate_split(split)
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer or None.")

        source_filter = _filter_values(source, name="source")
        domain_filter = _filter_values(domain, name="domain")

        if _rows is not None and split_path is not None:
            raise ValueError("Internal rows and split_path cannot both be supplied.")
        if _volume_payload is not None and volume_path is not None:
            raise ValueError("Internal volume payload and volume_path cannot both be supplied.")

        if _rows is not None:
            raw_rows = tuple(_rows)
        elif split_path is not None:
            raw_rows = _read_local_split(split_path)
        else:
            raw_rows = retrieval_resources._load_atlas_free_cnn_split_rows(self.split)

        if _volume_payload is not None:
            volume_payload = _volume_payload
        elif volume_path is not None:
            volume_payload = _read_local_volumes(volume_path)
        else:
            volume_payload = retrieval_resources._load_atlas_free_cnn_volumes()

        volumes, payload_map_ids = _validated_volumes(volume_payload)
        rows, indices = _validate_rows(
            raw_rows,
            split=self.split,
            volumes=volumes,
            payload_map_ids=payload_map_ids,
        )

        selected = [
            position
            for position, row in enumerate(rows)
            if (source_filter is None or str(row.get("source") or "").strip().lower() in source_filter)
            and (domain_filter is None or canonical_atlas_free_domain(row) in domain_filter)
        ]
        if limit is not None:
            selected = selected[:limit]

        self.rows = [rows[position] for position in selected]
        self.metadata = self.rows
        self.positive_texts = [row["positive_texts"] for row in self.rows]
        self._tensor_indices = [indices[position] for position in selected]
        self._volumes = volumes

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tensor_index = self._tensor_indices[index]
        return {
            "volume": self._volumes[tensor_index].float(),
            "map_id": row["map_id"],
            "positive_texts": row["positive_texts"],
            "metadata": row,
        }


class AtlasFreeCNNDataProvider:
    """Lazily provide validated train, validation, and test dataset views."""

    def __init__(
        self,
        *,
        source: str | Iterable[str] | None = None,
        domain: str | Iterable[str] | None = None,
        limit: int | None = None,
        split_dir: str | Path | None = None,
        volume_path: str | Path | None = None,
    ) -> None:
        self.source = source
        self.domain = domain
        self.limit = limit
        self.split_dir = None if split_dir is None else Path(split_dir).expanduser()
        self.volume_path = volume_path
        self._datasets: dict[str, AtlasFreeCNNDataset] = {}
        self._volume_payload: Mapping[str, Any] | None = None

    def split(self, split: str) -> AtlasFreeCNNDataset:
        split = _validate_split(split)
        if split not in self._datasets:
            if self._volume_payload is None:
                self._volume_payload = (
                    _read_local_volumes(self.volume_path)
                    if self.volume_path is not None
                    else retrieval_resources._load_atlas_free_cnn_volumes()
                )
            split_path = None if self.split_dir is None else self.split_dir / f"{split}.jsonl"
            self._datasets[split] = AtlasFreeCNNDataset(
                split,
                source=self.source,
                domain=self.domain,
                limit=self.limit,
                split_path=split_path,
                _volume_payload=self._volume_payload,
            )
        return self._datasets[split]

    @property
    def train(self) -> AtlasFreeCNNDataset:
        return self.split("train")

    @property
    def val(self) -> AtlasFreeCNNDataset:
        return self.split("val")

    @property
    def test(self) -> AtlasFreeCNNDataset:
        return self.split("test")


def atlas_free_cnn_splits(**kwargs: Any) -> dict[str, AtlasFreeCNNDataset]:
    """Return all three atlas-free CNN split views with shared resources."""

    provider = AtlasFreeCNNDataProvider(**kwargs)
    return {split: provider.split(split) for split in ATLAS_FREE_CNN_SPLITS}


__all__ = [
    "ATLAS_FREE_CNN_SPLITS",
    "AtlasFreeCNNDataProvider",
    "AtlasFreeCNNDataset",
    "atlas_free_cnn_splits",
    "canonical_atlas_free_domain",
]
