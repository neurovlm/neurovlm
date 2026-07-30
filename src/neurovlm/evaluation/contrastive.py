"""Full-split retrieval evaluation for atlas-free contrastive models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neurovlm.data.atlas_free_text import (
    AtlasFreeContrastiveCollator,
    AtlasFreeTextEmbeddingLookup,
)
from neurovlm.metrics.retrieval import (
    bidirectional_retrieval_metrics,
    normalized_k_values,
    recall_curve,
)
from neurovlm.models.losses import InfoNCELoss


@dataclass(frozen=True)
class ContrastiveEvaluation:
    """Retrieval metrics, full recall curves, and paired embeddings."""

    summary: Mapping[str, float]
    recall_curves: tuple[Mapping[str, Any], ...]
    by_source: tuple[Mapping[str, Any], ...]
    n: int
    brain_embeddings: torch.Tensor
    text_embeddings: torch.Tensor


def _loader(
    data: Dataset | DataLoader,
    *,
    lookup: AtlasFreeTextEmbeddingLookup | None,
    batch_size: int,
    target_shape: tuple[int, int, int],
    num_workers: int,
    seed: int,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    if lookup is None:
        lookup = AtlasFreeTextEmbeddingLookup.published()
    rows = getattr(data, "rows", None)
    if rows is not None:
        lookup.validate_dataset(rows)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=AtlasFreeContrastiveCollator(lookup, target_shape),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _metrics(text: torch.Tensor, brain: torch.Tensor, temperature: float) -> dict[str, float]:
    output = bidirectional_retrieval_metrics(text, brain, ks=(1, 5, 10, 50))
    auc = float(output["mean_normalized_k_recall_curve_auc"])
    output.update(
        {
            "loss": float(InfoNCELoss(temperature)(brain, text)),
            "normalized_k_recall_curve_auc": auc,
            "paper_recall_curve_auc": auc,
            "full_recall_curve_auc_k1_to_N": auc,
        }
    )
    return output


@torch.no_grad()
def evaluate_contrastive(
    model: nn.Module,
    data: Dataset | DataLoader,
    *,
    lookup: AtlasFreeTextEmbeddingLookup | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    target_shape: tuple[int, int, int] = (36, 45, 38),
    num_workers: int = 0,
    seed: int = 42,
    temperature: float = 0.07,
    max_batches: int | None = None,
) -> ContrastiveEvaluation:
    """Collect full-split projected embeddings and rank paired diagonals."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    resolved_device = torch.device(device)
    loader = _loader(
        data,
        lookup=lookup,
        batch_size=batch_size,
        target_shape=target_shape,
        num_workers=num_workers,
        seed=seed,
    )
    was_training = model.training
    model.to(resolved_device).eval()
    all_brain: list[torch.Tensor] = []
    all_text: list[torch.Tensor] = []
    sources: list[str] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        brain, text = model(
            batch["volume"].to(resolved_device, non_blocking=True),
            batch["text_embedding"].to(resolved_device, non_blocking=True),
        )
        all_brain.append(brain.float().cpu())
        all_text.append(text.float().cpu())
        sources.extend(str(value) for value in batch.get("source", ["unknown"] * len(brain)))
    model.train(was_training)
    if not all_brain:
        raise RuntimeError("Evaluation dataset produced no batches")
    brain_embeddings = torch.cat(all_brain)
    text_embeddings = torch.cat(all_text)
    summary = _metrics(text_embeddings, brain_embeddings, temperature)
    t2i_curve, i2t_curve = recall_curve(text_embeddings, brain_embeddings)
    normalized_k = normalized_k_values(len(t2i_curve))
    curves = tuple(
        {
            "k": index + 1,
            "normalized_k": float(normalized_k[index]),
            "t2i_recall": float(t2i_curve[index]),
            "i2t_recall": float(i2t_curve[index]),
            "mean_recall": float((t2i_curve[index] + i2t_curve[index]) / 2),
            "random_recall": float(normalized_k[index]),
        }
        for index in range(len(t2i_curve))
    )
    positions: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        positions[source].append(index)
    by_source = tuple(
        {
            "source": source,
            "n": len(indices),
            **_metrics(text_embeddings[indices], brain_embeddings[indices], temperature),
        }
        for source, indices in sorted(positions.items())
        if indices
    )
    return ContrastiveEvaluation(
        summary=summary,
        recall_curves=curves,
        by_source=by_source,
        n=len(brain_embeddings),
        brain_embeddings=brain_embeddings,
        text_embeddings=text_embeddings,
    )


__all__ = ["ContrastiveEvaluation", "evaluate_contrastive"]
