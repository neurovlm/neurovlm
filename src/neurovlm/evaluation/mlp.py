"""Evaluation utilities for the retained flat-map (MLP) models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from neurovlm.metrics.retrieval import (
    bidirectional_retrieval_metrics,
    normalized_k_values,
    recall_curve,
)
from neurovlm.models.losses import InfoNCELoss


@dataclass(frozen=True)
class MLPEvaluation:
    summary: Mapping[str, float]
    by_source: tuple[Mapping[str, Any], ...]
    by_sample: tuple[Mapping[str, Any], ...]
    n: int


@dataclass(frozen=True)
class MLPContrastiveEvaluation(MLPEvaluation):
    recall_curves: tuple[Mapping[str, Any], ...]
    brain_embeddings: torch.Tensor
    text_embeddings: torch.Tensor


def _loader(data: Dataset | DataLoader, batch_size: int, num_workers: int) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _mapping_tensor(batch: Mapping[str, Any], names: tuple[str, ...]) -> torch.Tensor:
    for name in names:
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(f"Batch must contain one of {names}")


def _metadata(batch: Any, n: int, offset: int) -> tuple[list[str], list[str]]:
    if not isinstance(batch, Mapping):
        return ["unknown"] * n, [str(offset + i) for i in range(n)]
    raw_source = batch.get("source", ["unknown"] * n)
    raw_id = batch.get("sample_id", batch.get("id", batch.get("map_id", None)))
    if isinstance(raw_source, (str, int)):
        raw_source = [raw_source] * n
    if raw_id is None:
        raw_id = [offset + i for i in range(n)]
    elif isinstance(raw_id, (str, int)):
        raw_id = [raw_id] * n
    return [str(v) for v in raw_source], [str(v) for v in raw_id]


def _autoencoder_input(batch: Any) -> torch.Tensor:
    if isinstance(batch, Mapping):
        return _mapping_tensor(batch, ("brain", "neuro", "flatmap", "image", "x", "target"))
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError("Unsupported autoencoder batch")


def _paired_input(batch: Any, *, latent_brain: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        text = _mapping_tensor(batch, ("text_embedding", "text", "specter"))
        brain_names = (
            ("brain_embedding", "brain_latent", "image_embedding", "brain", "neuro", "target")
            if latent_brain
            else ("brain", "neuro", "flatmap", "image", "target", "y")
        )
        return text, _mapping_tensor(batch, brain_names)
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Paired batches must be mappings or (text, brain) tuples")


def _safe_auroc(target: torch.Tensor, probability: torch.Tensor) -> float:
    target = (target.reshape(-1) > 0.5).float()
    probability = probability.reshape(-1).float()
    positives = int(target.sum())
    negatives = int(target.numel() - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = probability.argsort(descending=True)
    ranked = target[order]
    tpr = ranked.cumsum(0) / positives
    fpr = (1 - ranked).cumsum(0) / negatives
    return float(torch.trapezoid(torch.cat((torch.zeros(1), tpr)), torch.cat((torch.zeros(1), fpr))))


def _reconstruction_metrics(target: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    probability = torch.sigmoid(logits)
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits, target)),
        "bce_with_logits": float(F.binary_cross_entropy_with_logits(logits, target)),
        "probability_mse": float(F.mse_loss(probability, target)),
        "cosine_similarity": float(F.cosine_similarity(probability, target, dim=1).mean()),
        "voxel_auroc": _safe_auroc(target, probability),
    }


def _aggregate_by_source(
    sources: list[str], targets: torch.Tensor, predictions: torch.Tensor
) -> tuple[Mapping[str, Any], ...]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        positions[source].append(index)
    return tuple(
        {"source": source, "n": len(indices), **_reconstruction_metrics(targets[indices], predictions[indices])}
        for source, indices in sorted(positions.items())
    )


@torch.no_grad()
def evaluate_mlp_autoencoder(
    model: nn.Module,
    data: Dataset | DataLoader,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    max_batches: int | None = None,
) -> MLPEvaluation:
    resolved = torch.device(device)
    was_training = model.training
    model.to(resolved).eval()
    targets: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    sources: list[str] = []
    sample_ids: list[str] = []
    offset = 0
    for index, batch in enumerate(_loader(data, batch_size, num_workers)):
        if max_batches is not None and index >= max_batches:
            break
        target = _autoencoder_input(batch).float().to(resolved)
        output = model(target)
        targets.append(target.cpu())
        logits.append(output.float().cpu())
        batch_sources, batch_ids = _metadata(batch, len(target), offset)
        sources.extend(batch_sources)
        sample_ids.extend(batch_ids)
        offset += len(target)
    model.train(was_training)
    if not targets:
        raise RuntimeError("Evaluation dataset produced no batches")
    target = torch.cat(targets)
    output = torch.cat(logits)
    probability = torch.sigmoid(output)
    per_sample = F.binary_cross_entropy_with_logits(output, target, reduction="none").mean(1)
    by_sample = tuple(
        {
            "sample_id": sample_ids[i],
            "source": sources[i],
            "bce_with_logits": float(per_sample[i]),
            "probability_mse": float(F.mse_loss(probability[i], target[i])),
            "cosine_similarity": float(F.cosine_similarity(probability[i:i+1], target[i:i+1])),
        }
        for i in range(len(target))
    )
    return MLPEvaluation(
        summary=_reconstruction_metrics(target, output),
        by_source=_aggregate_by_source(sources, target, output),
        by_sample=by_sample,
        n=len(target),
    )


def _retrieval_summary(text: torch.Tensor, brain: torch.Tensor, temperature: float) -> dict[str, float]:
    summary = bidirectional_retrieval_metrics(text, brain, ks=(1, 5, 10, 50))
    summary["loss"] = float(InfoNCELoss(temperature)(brain, text))
    summary["normalized_k_recall_curve_auc"] = float(summary["mean_normalized_k_recall_curve_auc"])
    return summary


@torch.no_grad()
def evaluate_mlp_contrastive(
    model: nn.Module,
    data: Dataset | DataLoader,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    temperature: float = 0.07,
    max_batches: int | None = None,
) -> MLPContrastiveEvaluation:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    resolved = torch.device(device)
    was_training = model.training
    model.to(resolved).eval()
    text_all: list[torch.Tensor] = []
    brain_all: list[torch.Tensor] = []
    sources: list[str] = []
    sample_ids: list[str] = []
    offset = 0
    for index, batch in enumerate(_loader(data, batch_size, num_workers)):
        if max_batches is not None and index >= max_batches:
            break
        text, brain = _paired_input(batch, latent_brain=True)
        brain_projected, text_projected = model(brain.float().to(resolved), text.float().to(resolved))
        brain_all.append(brain_projected.float().cpu())
        text_all.append(text_projected.float().cpu())
        batch_sources, batch_ids = _metadata(batch, len(text), offset)
        sources.extend(batch_sources)
        sample_ids.extend(batch_ids)
        offset += len(text)
    model.train(was_training)
    if not text_all:
        raise RuntimeError("Evaluation dataset produced no batches")
    text = torch.cat(text_all)
    brain = torch.cat(brain_all)
    summary = _retrieval_summary(text, brain, temperature)
    t2i, i2t = recall_curve(text, brain)
    normalized = normalized_k_values(len(t2i))
    curves = tuple(
        {"k": i + 1, "normalized_k": float(normalized[i]), "t2i_recall": float(t2i[i]),
         "i2t_recall": float(i2t[i]), "mean_recall": float((t2i[i] + i2t[i]) / 2)}
        for i in range(len(t2i))
    )
    text_n = F.normalize(text, dim=1)
    brain_n = F.normalize(brain, dim=1)
    similarities = (text_n * brain_n).sum(1)
    by_sample = tuple(
        {"sample_id": sample_ids[i], "source": sources[i], "paired_cosine_similarity": float(similarities[i])}
        for i in range(len(text))
    )
    positions: dict[str, list[int]] = defaultdict(list)
    for i, source in enumerate(sources):
        positions[source].append(i)
    by_source = tuple(
        {"source": source, "n": len(indices), **_retrieval_summary(text[indices], brain[indices], temperature)}
        for source, indices in sorted(positions.items())
    )
    return MLPContrastiveEvaluation(summary, by_source, by_sample, len(text), curves, brain, text)


def evaluate_mlp_brain_to_text_retrieval(
    model: nn.Module,
    data: Dataset | DataLoader,
    **kwargs: Any,
) -> MLPContrastiveEvaluation:
    """Evaluate brain-to-text retrieval using the shared contrastive metrics.

    The returned summary retains both directions.  For this task the canonical
    selection metric is ``i2t_normalized_k_recall_curve_auc``.
    """

    return evaluate_mlp_contrastive(model, data, **kwargs)


@torch.no_grad()
def evaluate_mlp_text_to_brain(
    model: nn.Module,
    data: Dataset | DataLoader,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    max_batches: int | None = None,
) -> MLPEvaluation:
    resolved = torch.device(device)
    was_training = model.training
    model.to(resolved).eval()
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    flat_targets: list[torch.Tensor] = []
    decoded_logits: list[torch.Tensor] = []
    has_flat_targets = True
    sources: list[str] = []
    sample_ids: list[str] = []
    offset = 0
    for index, batch in enumerate(_loader(data, batch_size, num_workers)):
        if max_batches is not None and index >= max_batches:
            break
        text, target = _paired_input(batch, latent_brain=True)
        output = model(text.float().to(resolved))
        target = target.float().to(resolved)
        flat_target: torch.Tensor | None = None
        if isinstance(batch, Mapping):
            for name in ("brain", "neuro", "flatmap", "image"):
                candidate = batch.get(name)
                if isinstance(candidate, torch.Tensor) and candidate.shape[-1] != output.shape[-1]:
                    flat_target = candidate.float().to(resolved)
                    break
        if target.shape[-1] != output.shape[-1]:
            autoencoder = getattr(model, "autoencoder", None)
            if autoencoder is None:
                raise ValueError("Raw brain targets require model.autoencoder.encoder")
            flat_target = target
            target = autoencoder.encoder(target)
        targets.append(target.float().cpu())
        predictions.append(output.float().cpu())
        if flat_target is not None:
            decoder = getattr(getattr(model, "autoencoder", None), "decoder", None)
            if decoder is None:
                raise ValueError("Map-level metrics require model.autoencoder.decoder")
            flat_targets.append(flat_target.float().cpu())
            decoded_logits.append(decoder(output).float().cpu())
        else:
            has_flat_targets = False
        batch_sources, batch_ids = _metadata(batch, len(target), offset)
        sources.extend(batch_sources)
        sample_ids.extend(batch_ids)
        offset += len(target)
    model.train(was_training)
    if not targets:
        raise RuntimeError("Evaluation dataset produced no batches")
    target = torch.cat(targets)
    output = torch.cat(predictions)
    sample_loss = F.mse_loss(output, target, reduction="none").mean(1)
    sample_cosine = F.cosine_similarity(output, target, dim=1)
    by_sample = tuple(
        {"sample_id": sample_ids[i], "source": sources[i], "latent_mse": float(sample_loss[i]),
         "latent_cosine_similarity": float(sample_cosine[i])}
        for i in range(len(target))
    )
    summary = {
        "loss": float(F.mse_loss(output, target)),
        "latent_mse": float(F.mse_loss(output, target)),
        "latent_cosine_similarity": float(sample_cosine.mean()),
    }
    if has_flat_targets and flat_targets:
        decoded = _reconstruction_metrics(torch.cat(flat_targets), torch.cat(decoded_logits))
        summary.update({f"decoded_{name}": value for name, value in decoded.items()})
    positions: dict[str, list[int]] = defaultdict(list)
    for i, source in enumerate(sources):
        positions[source].append(i)
    flat_all = torch.cat(flat_targets) if has_flat_targets and flat_targets else None
    decoded_all = torch.cat(decoded_logits) if has_flat_targets and decoded_logits else None
    by_source_rows = []
    for source, indices in sorted(positions.items()):
        row = {
            "source": source,
            "n": len(indices),
            "loss": float(F.mse_loss(output[indices], target[indices])),
            "latent_mse": float(F.mse_loss(output[indices], target[indices])),
            "latent_cosine_similarity": float(F.cosine_similarity(output[indices], target[indices], dim=1).mean()),
        }
        if flat_all is not None and decoded_all is not None:
            row.update({f"decoded_{name}": value for name, value in
                        _reconstruction_metrics(flat_all[indices], decoded_all[indices]).items()})
        by_source_rows.append(row)
    by_source = tuple(by_source_rows)
    return MLPEvaluation(
        summary=summary,
        by_source=by_source,
        by_sample=by_sample,
        n=len(target),
    )


__all__ = [
    "MLPContrastiveEvaluation",
    "MLPEvaluation",
    "evaluate_mlp_autoencoder",
    "evaluate_mlp_brain_to_text_retrieval",
    "evaluate_mlp_contrastive",
    "evaluate_mlp_text_to_brain",
]
