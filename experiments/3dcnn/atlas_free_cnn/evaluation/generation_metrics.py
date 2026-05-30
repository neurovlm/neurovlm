"""Generation metrics for sparse ALE-style brain maps."""

from __future__ import annotations

import torch

from atlas_free_cnn.training.generation_losses import (
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
    pred_raw = torch.nan_to_num(pred.float(), nan=0.0, posinf=1.0, neginf=0.0)
    target_raw = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0)
    pred_eval = pred_raw.clamp(0.0, 1.0)
    target_eval = target_raw.clamp(0.0, 1.0)
    pred = normalize_positive(pred_eval, mask)
    target = normalize_positive(target_eval, mask)
    top1 = hard_topk_dice(pred, target, k_percent=0.01, mask=mask)
    top5 = hard_topk_dice(pred, target, k_percent=0.05, mask=mask)
    top10 = hard_topk_dice(pred, target, k_percent=0.10, mask=mask)
    mse = float((pred_eval - target_eval).pow(2).mean().item())
    foreground = target_eval > 0
    if bool(foreground.any()):
        foreground_mse = float((pred_eval[foreground] - target_eval[foreground]).pow(2).mean().item())
    else:
        foreground_mse = mse
    out = {
        "mse": mse,
        "reconstruction_mse": mse,
        "mae": float((pred_eval - target_eval).abs().mean().item()),
        "foreground_mse": foreground_mse,
        "spatial_corr": float(1.0 - spatial_correlation_loss(pred, target, mask=mask).item()),
        "top1_dice": float(top1.mean().item()),
        "top5_dice": float(top5.mean().item()),
        "top10_dice": float(top10.mean().item()),
        "top1_overlap": float(top1.mean().item()),
        "top5_overlap": float(top5.mean().item()),
        "top10_overlap": float(top10.mean().item()),
        "target_nonzero_fraction": float((target_eval > 0).float().mean().item()),
        "pred_nonzero_fraction": float((pred_eval > 0).float().mean().item()),
        "pred_mean": float(pred_eval.mean().item()),
        "pred_max": float(pred_eval.max().item()),
    }
    if include_voxel_auroc:
        try:
            out["voxel_auroc"] = voxel_auroc(pred_eval, target_eval, mask=mask)
        except Exception:
            out["voxel_auroc"] = float("nan")
    return out
