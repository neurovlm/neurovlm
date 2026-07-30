"""Spatial reconstruction metrics for dense atlas-free brain volumes.

Training losses operate on the raw decoder output.  This module owns the
separate evaluation convention: non-finite values are made finite and both
prediction and target are clamped to ``[0, 1]`` before spatial metrics are
computed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _as_flat_batch(value: Tensor) -> Tensor:
    value = torch.as_tensor(value).float()
    if value.ndim < 2:
        raise ValueError("Spatial metrics require a batch dimension")
    return value.reshape(value.shape[0], -1)


def _topk_masks(value: Tensor, fraction: float) -> Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Top-k fraction must be in (0, 1]")
    k = max(1, min(value.shape[1], int(math.ceil(value.shape[1] * fraction))))
    indices = value.topk(k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(value, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def _topk_diagnostics(pred: Tensor, target: Tensor, fraction: float) -> tuple[Tensor, Tensor, Tensor]:
    pred_mask = _topk_masks(pred, fraction)
    target_mask = _topk_masks(target, fraction)
    intersection = (pred_mask & target_mask).sum(dim=1).float()
    denom = pred_mask.sum(dim=1) + target_mask.sum(dim=1)
    dice = 2.0 * intersection / denom.clamp_min(1).float()
    overlap = intersection / target_mask.sum(dim=1).clamp_min(1).float()
    return dice, overlap, intersection


def voxel_auroc(pred: Tensor, target: Tensor, *, threshold: float | None = None) -> float:
    """Return rank-based voxel AUROC without an optional sklearn dependency."""

    pred = torch.as_tensor(pred).float().flatten()
    target = torch.as_tensor(target).float().flatten()
    if threshold is None:
        threshold = float(torch.quantile(target, 0.95))
    positive = target > threshold
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        return float("nan")
    order = torch.argsort(pred)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, pred.numel() + 1, dtype=torch.float64, device=pred.device)
    positive_count = positive.sum().double()
    negative_count = (~positive).sum().double()
    auc = (
        ranks[positive].sum() - positive_count * (positive_count + 1.0) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


@torch.no_grad()
def reconstruction_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    include_voxel_auroc: bool = False,
) -> dict[str, float]:
    """Compute the standardized autoencoder reconstruction metric set.

    Metrics are averaged per sample so that sparse maps and dense maps have
    equal weight.  Raw-output diagnostics are retained alongside clamped
    reconstruction metrics.
    """

    raw_pred = _as_flat_batch(prediction)
    raw_target = _as_flat_batch(target)
    if raw_pred.shape != raw_target.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    finite_pred = torch.nan_to_num(raw_pred, nan=0.0, posinf=1.0, neginf=0.0)
    finite_target = torch.nan_to_num(raw_target, nan=0.0, posinf=1.0, neginf=0.0)
    pred = finite_pred.clamp(0.0, 1.0)
    truth = finite_target.clamp(0.0, 1.0)
    error = pred - truth
    mse_by_sample = error.square().mean(dim=1)
    mae_by_sample = error.abs().mean(dim=1)

    foreground = truth > 0
    foreground_count = foreground.sum(dim=1)
    foreground_squared_error = (error.square() * foreground).sum(dim=1)
    foreground_mse = torch.where(
        foreground_count > 0,
        foreground_squared_error / foreground_count.clamp_min(1),
        mse_by_sample,
    )

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    truth_centered = truth - truth.mean(dim=1, keepdim=True)
    corr_denom = pred_centered.square().sum(dim=1).sqrt() * truth_centered.square().sum(dim=1).sqrt()
    correlation = torch.where(
        corr_denom > 0,
        (pred_centered * truth_centered).sum(dim=1) / corr_denom.clamp_min(1e-12),
        torch.zeros_like(corr_denom),
    )

    values: dict[str, Tensor] = {
        "mse": mse_by_sample,
        "reconstruction_mse": mse_by_sample,
        "mae": mae_by_sample,
        "foreground_mse": foreground_mse,
        "spatial_corr": correlation,
        "target_nonzero_fraction": foreground.float().mean(dim=1),
        "pred_nonzero_fraction": (pred > 0).float().mean(dim=1),
        "raw_pred_fraction_below_zero": (raw_pred < 0).float().mean(dim=1),
        "raw_pred_fraction_above_one": (raw_pred > 1).float().mean(dim=1),
        "raw_pred_nonfinite_fraction": (~torch.isfinite(raw_pred)).float().mean(dim=1),
        "pred_mean": pred.mean(dim=1),
        "pred_max": pred.max(dim=1).values,
    }
    for percent in (1, 5, 10):
        dice, target_recall, intersection = _topk_diagnostics(pred, truth, percent / 100.0)
        values[f"top{percent}_dice"] = dice
        # Compatibility: historical generation_metrics exposed ``overlap``
        # as an alias of Dice.  Keep that schema stable and give the actual
        # target-set overlap fraction an unambiguous name.
        values[f"top{percent}_overlap"] = dice
        values[f"top{percent}_target_recall"] = target_recall
        values[f"top{percent}_intersection_voxels"] = intersection

    output = {name: float(value.mean()) for name, value in values.items()}
    if include_voxel_auroc:
        aucs = [voxel_auroc(pred[index], truth[index]) for index in range(pred.shape[0])]
        finite = [value for value in aucs if math.isfinite(value)]
        output["voxel_auroc"] = float(sum(finite) / len(finite)) if finite else float("nan")
    return output


__all__ = ["reconstruction_metrics", "voxel_auroc"]
