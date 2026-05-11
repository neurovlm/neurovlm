"""Generation metrics for sparse ALE-style brain maps."""

from __future__ import annotations

import torch

from atlas_free_multipositive.training.generation_losses import (
    hard_topk_dice,
    normalize_positive,
    spatial_correlation_loss,
)


def _flatten_pair(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = mask.to(target.device).bool()
        while valid.ndim < target.ndim:
            valid = valid.unsqueeze(0)
        valid = valid.expand_as(target)
    return pred[valid].float(), target[valid].float()


def voxel_auroc(pred: torch.Tensor, target: torch.Tensor, *, threshold: float | None = None, mask: torch.Tensor | None = None) -> float:
    """Dependency-free AUROC over voxels using target > threshold as positives."""

    p, t = _flatten_pair(pred, target, mask)
    if threshold is None:
        threshold = float(torch.quantile(t, 0.95).item())
    y = t > threshold
    if int(y.sum()) == 0 or int((~y).sum()) == 0:
        return float("nan")
    order = torch.argsort(p)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(p) + 1, device=p.device).float()
    n_pos = y.sum().float()
    n_neg = (~y).sum().float()
    auc = (ranks[y].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


@torch.no_grad()
def generation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    include_voxel_auroc: bool = True,
) -> dict[str, float]:
    pred = normalize_positive(pred, mask)
    target = normalize_positive(target, mask)
    out = {
        "mse": float((pred - target).pow(2).mean().item()),
        "spatial_corr": float(1.0 - spatial_correlation_loss(pred, target, mask=mask).item()),
        "top1_dice": float(hard_topk_dice(pred, target, k_percent=0.01, mask=mask).mean().item()),
        "top5_dice": float(hard_topk_dice(pred, target, k_percent=0.05, mask=mask).mean().item()),
        "top10_dice": float(hard_topk_dice(pred, target, k_percent=0.10, mask=mask).mean().item()),
    }
    if include_voxel_auroc:
        try:
            out["voxel_auroc"] = voxel_auroc(pred, target, mask=mask)
        except Exception:
            out["voxel_auroc"] = float("nan")
    return out
