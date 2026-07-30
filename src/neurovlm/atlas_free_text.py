"""Published SPECTER2 lookup and paired CNN contrastive collation.

The Stage 3/4 text cache contains the exact first positive text for every
published map after empty-string centering and L2 normalization.  This module
indexes that immutable cache by ``text_id``; it never re-encodes text or
consults experiment-local paths.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .atlas_free_dataset import canonical_atlas_free_domain


def _primary_positive(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the validated ``positive_texts[0]`` mapping."""

    positives = item.get("positive_texts") or []
    if not isinstance(positives, Sequence) or isinstance(positives, (str, bytes)):
        raise TypeError(f"Row {item.get('map_id', '')!r} positive_texts must be a sequence")
    if not positives:
        raise ValueError(f"Row {item.get('map_id', '')!r} has no positive_texts")
    primary = positives[0]
    if not isinstance(primary, Mapping):
        raise TypeError(f"Row {item.get('map_id', '')!r} first positive text must be a mapping")
    return primary


def primary_positive_text_id(item: Mapping[str, Any]) -> str:
    """Return the stable ID of ``positive_texts[0]``.

    Selecting by position is intentional: the published cache was built from
    the first positive and downstream retrieval treats each map and that text
    as one diagonal pair.
    """

    primary = _primary_positive(item)
    text_id = str(primary.get("text_id") or primary.get("id") or "").strip()
    if not text_id:
        raise ValueError(f"Row {item.get('map_id', '')!r} first positive text has no text_id")
    return text_id


def primary_positive_text(item: Mapping[str, Any]) -> str:
    """Return the raw text paired with a map for family-native re-encoding."""

    primary = _primary_positive(item)
    text = str(primary.get("text") or "").strip()
    if not text:
        raise ValueError(f"Row {item.get('map_id', '')!r} first positive text has no text")
    return text


@dataclass(frozen=True)
class AtlasFreeTextEmbeddingLookup:
    """Validated ID-to-vector lookup for normalized 768-d SPECTER2 vectors."""

    embeddings: torch.Tensor
    text_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        embeddings: torch.Tensor,
        text_ids: Sequence[Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        vectors = torch.as_tensor(embeddings, dtype=torch.float32).cpu()
        ids = tuple(str(value).strip() for value in text_ids)
        if vectors.ndim != 2 or vectors.shape[1] != 768:
            raise ValueError(f"Expected text embeddings with shape N x 768, got {tuple(vectors.shape)}")
        if len(ids) != len(vectors):
            raise ValueError("text_ids length does not match the embedding count")
        if any(not text_id for text_id in ids):
            raise ValueError("text_ids must not contain empty IDs")
        # A single linear pass matters for the published cache (tens of
        # thousands of IDs); tuple.count here would make construction O(N^2).
        duplicates = sorted(text_id for text_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            preview = ", ".join(repr(value) for value in duplicates[:5])
            raise ValueError(f"Duplicate text IDs in normalized SPECTER2 cache: {preview}")
        if not torch.isfinite(vectors).all():
            raise ValueError("Normalized SPECTER2 cache contains NaNs or infinities")
        norms = vectors.norm(dim=1)
        within_tolerance = (norms.sub(1).abs() <= 1e-3).float().mean() if len(norms) else 1.0
        if float(within_tolerance) < 0.999:
            raise ValueError("SPECTER2 vectors must use the empty-centered, unit-normalized convention")
        object.__setattr__(self, "embeddings", vectors)
        object.__setattr__(self, "text_ids", ids)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "_indices", {text_id: index for index, text_id in enumerate(ids)})

    @classmethod
    def published(cls) -> "AtlasFreeTextEmbeddingLookup":
        """Load the canonical cache through NeuroVLM's HF retrieval resources."""

        from .retrieval_resources import _load_atlas_free_cnn_normalized_specter2_embeddings

        embeddings, text_ids, metadata = _load_atlas_free_cnn_normalized_specter2_embeddings()
        return cls(embeddings, text_ids.tolist(), metadata)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AtlasFreeTextEmbeddingLookup":
        if not isinstance(payload, Mapping):
            raise TypeError("Text embedding cache payload must be a mapping")
        text_ids = payload.get("text_ids")
        return cls(
            payload.get("embeddings"),
            () if text_ids is None else text_ids,
            payload.get("metadata"),
        )

    def __len__(self) -> int:
        return len(self.text_ids)

    def __contains__(self, text_id: object) -> bool:
        return str(text_id) in self._indices

    def __getitem__(self, text_id: str) -> torch.Tensor:
        key = str(text_id)
        try:
            return self.embeddings[self._indices[key]]
        except KeyError as error:
            raise KeyError(f"Text ID {key!r} is missing from the published normalized SPECTER2 cache") from error

    def validate_dataset(self, rows: Sequence[Mapping[str, Any]]) -> None:
        missing = sorted(
            {text_id for row in rows if (text_id := primary_positive_text_id(row)) not in self}
        )
        if missing:
            preview = ", ".join(repr(value) for value in missing[:5])
            raise KeyError(f"{len(missing)} primary text IDs are absent from the SPECTER2 cache: {preview}")


class AtlasFreeContrastiveCollator:
    """Collate map/primary-text pairs without any local path dependency."""

    def __init__(
        self,
        lookup: AtlasFreeTextEmbeddingLookup,
        target_shape: tuple[int, int, int] = (36, 45, 38),
    ) -> None:
        self.lookup = lookup
        self.target_shape = tuple(int(value) for value in target_shape)

    def __call__(self, batch: list[Mapping[str, Any]]) -> dict[str, Any]:
        volumes: list[torch.Tensor] = []
        texts: list[torch.Tensor] = []
        map_ids: list[str] = []
        text_ids: list[str] = []
        sources: list[str] = []
        for item in batch:
            volume = torch.as_tensor(item["volume"], dtype=torch.float32)
            if volume.ndim != 4 or volume.shape[0] != 1:
                raise ValueError(f"Expected a 1 x D x H x W volume, got {tuple(volume.shape)}")
            if tuple(volume.shape[-3:]) != self.target_shape:
                volume = F.interpolate(
                    volume.unsqueeze(0), size=self.target_shape, mode="trilinear", align_corners=False
                ).squeeze(0)
            text_id = primary_positive_text_id(item)
            volumes.append(torch.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0))
            texts.append(self.lookup[text_id])
            map_ids.append(str(item.get("map_id") or ""))
            text_ids.append(text_id)
            sources.append(canonical_atlas_free_domain(item.get("metadata") or item))
        return {
            "volume": torch.stack(volumes),
            "text_embedding": torch.stack(texts),
            "map_id": map_ids,
            "text_id": text_ids,
            "source": sources,
        }


__all__ = [
    "AtlasFreeContrastiveCollator",
    "AtlasFreeTextEmbeddingLookup",
    "primary_positive_text",
    "primary_positive_text_id",
]
