"""Standardized evaluation for CNN text-to-brain generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neurovlm.data.atlas_free_text import (
    AtlasFreeContrastiveCollator,
    AtlasFreeTextEmbeddingLookup,
)
from neurovlm.evaluation.spatial import reconstruction_metrics


@dataclass(frozen=True)
class TextToBrainEvaluation:
    """Overall, source-stratified, per-sample, and bounded generated outputs."""

    summary: Mapping[str, float]
    by_source: tuple[Mapping[str, Any], ...]
    by_sample: tuple[Mapping[str, Any], ...]
    generated: tuple[Mapping[str, Any], ...]
    n: int


def _means(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(sum(float(row[key]) for row in rows) / len(rows))
        for key in rows[0]
    }


def _loader(
    data: Dataset | DataLoader,
    *,
    lookup: AtlasFreeTextEmbeddingLookup,
    batch_size: int,
    target_shape: tuple[int, int, int],
    num_workers: int,
    seed: int,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    rows = getattr(data, "rows", None)
    if rows is not None:
        lookup.validate_dataset(rows)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=AtlasFreeContrastiveCollator(lookup, target_shape),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate_text_to_brain(
    model: nn.Module,
    data: Dataset | DataLoader,
    *,
    lookup: AtlasFreeTextEmbeddingLookup,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    target_shape: tuple[int, int, int] = (36, 45, 38),
    num_workers: int = 0,
    seed: int = 42,
    max_batches: int | None = None,
    generated_limit: int = 0,
    reconstruction_weight: float = 1.0,
    latent_weight: float = 1.0,
    semantic_evaluator: Callable[..., Mapping[str, float]] | None = None,
) -> TextToBrainEvaluation:
    """Evaluate raw loss while clamping only inside spatial metrics.

    ``semantic_evaluator`` is a clean callback into contrastive semantic
    evaluation. It receives ``model``, ``data``, and ``evaluation`` keyword
    arguments, avoiding a duplicate Stage 4 semantic implementation.
    """

    resolved_device = torch.device(device)
    loader = _loader(
        data,
        lookup=lookup,
        batch_size=batch_size,
        target_shape=tuple(target_shape),
        num_workers=num_workers,
        seed=seed,
    )
    was_training = model.training
    model.to(resolved_device).eval()
    ae = getattr(model, "autoencoder", None)
    projector = getattr(model, "text_projection", None)
    if not isinstance(ae, nn.Module) or not isinstance(projector, nn.Module):
        raise TypeError("model must expose .autoencoder and .text_projection modules")

    metric_rows: list[dict[str, float]] = []
    by_sample: list[dict[str, Any]] = []
    source_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    generated: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["volume"].to(resolved_device)
        text = batch["text_embedding"].to(resolved_device)
        brain_z = ae.encoder(target)
        text_z = projector(text)
        prediction = ae.decoder(text_z)
        for index in range(len(target)):
            spatial = reconstruction_metrics(
                prediction[index : index + 1], target[index : index + 1]
            )
            latent_mse = float(F.mse_loss(text_z[index], brain_z[index]))
            latent_cosine = float(F.cosine_similarity(text_z[index], brain_z[index], dim=0))
            reconstruction_mse = float(F.mse_loss(prediction[index], target[index]))
            total = reconstruction_weight * reconstruction_mse + latent_weight * latent_mse
            row = {
                "loss": total,
                "total": total,
                "latent_mse": latent_mse,
                "latent_cosine": latent_cosine,
                "raw_reconstruction_mse": reconstruction_mse,
                **spatial,
            }
            metric_rows.append(row)
            source = str(batch.get("source", ["unknown"] * len(target))[index])
            source_metrics[source].append(row)
            sample = {
                "map_id": str(batch.get("map_id", [""] * len(target))[index]),
                "text_id": str(batch.get("text_id", [""] * len(target))[index]),
                "source": source,
                **row,
            }
            by_sample.append(sample)
            if len(generated) < generated_limit:
                generated.append(
                    {
                        "map_id": sample["map_id"],
                        "text_id": sample["text_id"],
                        "source": source,
                        "prediction": prediction[index].detach().cpu(),
                        "target": target[index].detach().cpu(),
                    }
                )

    summary = _means(metric_rows)
    by_source = tuple(
        {"source": source, "n": len(rows), **_means(rows)}
        for source, rows in sorted(source_metrics.items())
    )
    preliminary = TextToBrainEvaluation(
        summary=summary,
        by_source=by_source,
        by_sample=tuple(by_sample),
        generated=tuple(generated),
        n=len(metric_rows),
    )
    if semantic_evaluator is not None:
        semantic = dict(semantic_evaluator(model=model, data=data, evaluation=preliminary))
        summary = {
            **summary,
            **{f"semantic_{key}": float(value) for key, value in semantic.items()},
        }
    model.train(was_training)
    ae.eval()
    return TextToBrainEvaluation(
        summary=summary,
        by_source=by_source,
        by_sample=tuple(by_sample),
        generated=tuple(generated),
        n=len(metric_rows),
    )


__all__ = ["TextToBrainEvaluation", "evaluate_text_to_brain"]
