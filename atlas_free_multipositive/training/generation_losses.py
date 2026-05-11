"""Sparse-aware losses for CNN autoencoding and text-to-brain generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


def _mask_like(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(x, dtype=torch.bool)
    mask = mask.to(device=x.device)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(0)
    return mask.bool().expand_as(x)


def normalize_positive(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Clamp to nonnegative and normalize each sample by its masked max."""

    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    valid = _mask_like(x, mask)
    flat = x.masked_fill(~valid, 0.0).flatten(1)
    mx = flat.max(dim=1).values.view(-1, *([1] * (x.ndim - 1)))
    return x / mx.clamp_min(eps)


def weighted_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    loss_type: str = "mse",
    alpha: float = 10.0,
    gamma: float = 1.0,
    normalize_target: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted MSE/BCE that upweights high target activation voxels."""

    valid = _mask_like(target, mask)
    target_for_weight = normalize_positive(target, mask, eps) if normalize_target else target.clamp_min(0.0)
    weights = 1.0 + float(alpha) * target_for_weight.pow(float(gamma))
    if loss_type == "mse":
        loss = (pred - target).pow(2)
    elif loss_type == "bce":
        loss = F.binary_cross_entropy(pred.clamp(eps, 1.0 - eps), target.clamp(0.0, 1.0), reduction="none")
    else:
        raise ValueError("loss_type must be 'mse' or 'bce'")
    loss = loss * weights
    return loss[valid].mean()


def latent_alignment_loss(
    text_z: torch.Tensor,
    brain_z: torch.Tensor,
    *,
    loss_type: str = "cosine",
    detach_brain_z: bool = True,
) -> torch.Tensor:
    """Align text latent to CNN brain latent."""

    if detach_brain_z:
        brain_z = brain_z.detach()
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(text_z, brain_z, dim=1, eps=1e-8)).mean()
    if loss_type == "mse":
        return F.mse_loss(F.normalize(text_z, dim=1, eps=1e-8), F.normalize(brain_z, dim=1, eps=1e-8))
    raise ValueError("loss_type must be 'cosine' or 'mse'")


def soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = normalize_positive(pred, mask, eps) if normalize else pred.clamp_min(0.0)
    target = normalize_positive(target, mask, eps) if normalize else target.clamp_min(0.0)
    valid = _mask_like(target, mask)
    pred = pred.masked_fill(~valid, 0.0).flatten(1)
    target = target.masked_fill(~valid, 0.0).flatten(1)
    dice = (2.0 * (pred * target).sum(dim=1) + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)
    return (1.0 - dice).mean()


def hard_topk_mask(x: torch.Tensor, k_percent: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return a binary mask for the top k percent voxels per sample."""

    if not (0.0 < float(k_percent) <= 1.0):
        raise ValueError("k_percent must be in (0, 1]")
    x = x.float()
    valid = _mask_like(x, mask)
    flat = x.masked_fill(~valid, -torch.inf).flatten(1)
    valid_counts = valid.flatten(1).sum(dim=1).clamp_min(1)
    k = torch.clamp((valid_counts.float() * float(k_percent)).ceil().long(), min=1)
    out = torch.zeros_like(flat, dtype=torch.bool)
    for i in range(flat.shape[0]):
        idx = torch.topk(flat[i], int(k[i].item())).indices
        out[i, idx] = True
    return out.view_as(x) & valid


def topk_overlap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    k_percents: Iterable[float] = (0.01, 0.05, 0.10),
    beta_outside: float = 0.25,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable approximation that rewards predicted mass in target top-k voxels."""

    pred_norm = normalize_positive(pred, mask, eps)
    valid = _mask_like(target, mask)
    losses = []
    for k_percent in k_percents:
        target_topk = hard_topk_mask(target, float(k_percent), mask).float()
        pred_flat = pred_norm.masked_fill(~valid, 0.0).flatten(1)
        target_flat = target_topk.flatten(1)
        total_mass = pred_flat.sum(dim=1).clamp_min(eps)
        inside = (pred_flat * target_flat).sum(dim=1)
        outside = (pred_flat * (1.0 - target_flat)).sum(dim=1)
        losses.append(1.0 - inside / total_mass + float(beta_outside) * outside / total_mass)
    return torch.stack(losses, dim=0).mean()


def spatial_correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = _mask_like(target, mask)
    pred_flat = pred.masked_fill(~valid, 0.0).flatten(1)
    target_flat = target.masked_fill(~valid, 0.0).flatten(1)
    valid_flat = valid.flatten(1).float()
    denom = valid_flat.sum(dim=1).clamp_min(1.0)
    pred_mean = (pred_flat * valid_flat).sum(dim=1, keepdim=True) / denom.view(-1, 1)
    target_mean = (target_flat * valid_flat).sum(dim=1, keepdim=True) / denom.view(-1, 1)
    pred_centered = (pred_flat - pred_mean) * valid_flat
    target_centered = (target_flat - target_mean) * valid_flat
    corr = (pred_centered * target_centered).sum(dim=1) / (
        pred_centered.pow(2).sum(dim=1).sqrt() * target_centered.pow(2).sum(dim=1).sqrt() + eps
    )
    return (1.0 - corr).mean()


def hard_topk_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    k_percent: float,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_mask = hard_topk_mask(pred, k_percent, mask)
    target_mask = hard_topk_mask(target, k_percent, mask)
    inter = (pred_mask & target_mask).flatten(1).sum(dim=1).float()
    denom = pred_mask.flatten(1).sum(dim=1).float() + target_mask.flatten(1).sum(dim=1).float()
    return (2.0 * inter + eps) / (denom + eps)


@dataclass
class GenerationLossConfig:
    lambda_recon: float = 1.0
    lambda_latent: float = 0.0
    lambda_dice: float = 0.5
    lambda_topk: float = 0.5
    lambda_corr: float = 0.25
    recon_type: str = "mse"
    recon_alpha: float = 10.0
    recon_gamma: float = 1.0
    prediction_activation: str = "sigmoid"


def apply_prediction_activation(pred: torch.Tensor, activation: str = "sigmoid") -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(pred)
    if activation == "softplus":
        return F.softplus(pred)
    if activation in {"none", None}:
        return pred
    raise ValueError("prediction_activation must be 'sigmoid', 'softplus', or 'none'")


def combined_generation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    brain_z: torch.Tensor | None = None,
    text_z: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    config: GenerationLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted reconstruction + sparse overlap + optional latent losses."""

    cfg = config or GenerationLossConfig()
    pred_active = apply_prediction_activation(pred, cfg.prediction_activation)
    parts: dict[str, torch.Tensor] = {}
    parts["weighted_recon"] = weighted_reconstruction_loss(
        pred_active,
        target,
        mask=mask,
        loss_type=cfg.recon_type,
        alpha=cfg.recon_alpha,
        gamma=cfg.recon_gamma,
    )
    parts["soft_dice"] = soft_dice_loss(pred_active, target, mask=mask)
    parts["topk_overlap"] = topk_overlap_loss(pred_active, target, mask=mask)
    parts["spatial_corr"] = spatial_correlation_loss(pred_active, target, mask=mask)
    if brain_z is not None and text_z is not None and cfg.lambda_latent:
        parts["latent_alignment"] = latent_alignment_loss(text_z, brain_z)
    else:
        parts["latent_alignment"] = pred_active.sum() * 0.0
    total = (
        cfg.lambda_recon * parts["weighted_recon"]
        + cfg.lambda_dice * parts["soft_dice"]
        + cfg.lambda_topk * parts["topk_overlap"]
        + cfg.lambda_corr * parts["spatial_corr"]
        + cfg.lambda_latent * parts["latent_alignment"]
    )
    parts["total"] = total
    return total, parts

