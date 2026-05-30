"""Configurable reconstruction losses for Stage 1 CNN autoencoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from atlas_free_cnn.training.generation_losses import (
    apply_prediction_activation,
    hard_topk_mask,
    normalize_positive,
)


@dataclass
class AutoencoderLossConfig:
    type: str = "raw_mse"
    lambda_foreground: float = 0.0
    lambda_topk: float = 0.0
    foreground_threshold: float | None = None
    topk_percent: float = 5.0
    prediction_activation: str = "none"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AutoencoderLossConfig":
        raw = dict(cfg.get("loss") or {})
        if not raw:
            raw = {"type": cfg.get("loss_type", "raw_mse")}
        return cls(
            type=str(raw.get("type", "raw_mse")),
            lambda_foreground=float(raw.get("lambda_foreground", 0.0) or 0.0),
            lambda_topk=float(raw.get("lambda_topk", raw.get("lambda_dice", 0.0)) or 0.0),
            foreground_threshold=(
                None
                if raw.get("foreground_threshold") is None
                else float(raw.get("foreground_threshold"))
            ),
            topk_percent=float(raw.get("topk_percent", 5.0) or 5.0),
            prediction_activation=str(raw.get("prediction_activation", cfg.get("prediction_activation", "none"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "lambda_foreground": self.lambda_foreground,
            "lambda_topk": self.lambda_topk,
            "foreground_threshold": self.foreground_threshold,
            "topk_percent": self.topk_percent,
            "prediction_activation": self.prediction_activation,
        }


def foreground_mask(target: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
    target_eval = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if threshold is None:
        return target_eval > 0
    return target_eval > float(threshold)


def foreground_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float | None = None,
) -> torch.Tensor:
    mask = foreground_mask(target, threshold)
    loss = (pred - target).pow(2)
    if not bool(mask.any()):
        return loss.mean()
    return loss[mask].mean()


def soft_topk_overlap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    topk_percent: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    k_percent = float(topk_percent)
    if k_percent > 1.0:
        k_percent = k_percent / 100.0
    pred_pos = normalize_positive(pred)
    target_topk = hard_topk_mask(target, k_percent=k_percent).float()
    pred_flat = pred_pos.flatten(1)
    target_flat = target_topk.flatten(1)
    inside = (pred_flat * target_flat).sum(dim=1)
    total = pred_flat.sum(dim=1).clamp_min(eps)
    return (1.0 - inside / total).mean()


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: AutoencoderLossConfig | dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg if isinstance(cfg, AutoencoderLossConfig) else AutoencoderLossConfig.from_config(cfg or {})
    pred_for_loss = apply_prediction_activation(pred, loss_cfg.prediction_activation)
    parts: dict[str, torch.Tensor] = {}
    parts["raw_mse"] = F.mse_loss(pred_for_loss, target)
    if loss_cfg.type == "raw_mse":
        parts["total"] = parts["raw_mse"]
        return parts["total"], parts
    if loss_cfg.type != "hybrid_recon":
        raise ValueError("loss.type must be raw_mse or hybrid_recon")
    zero = pred_for_loss.sum() * 0.0
    parts["foreground_mse"] = (
        foreground_weighted_mse(
            pred_for_loss,
            target,
            threshold=loss_cfg.foreground_threshold,
        )
        if loss_cfg.lambda_foreground
        else zero
    )
    parts["topk_overlap"] = (
        soft_topk_overlap_loss(
            pred_for_loss,
            target,
            topk_percent=loss_cfg.topk_percent,
        )
        if loss_cfg.lambda_topk
        else zero
    )
    parts["total"] = (
        parts["raw_mse"]
        + float(loss_cfg.lambda_foreground) * parts["foreground_mse"]
        + float(loss_cfg.lambda_topk) * parts["topk_overlap"]
    )
    return parts["total"], parts
