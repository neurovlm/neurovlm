"""Raw-MSE reconstruction loss for the retained Stage 1 CNN autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from atlas_free_cnn.training.generation_losses import apply_prediction_activation


@dataclass
class AutoencoderLossConfig:
    type: str = "raw_mse"
    prediction_activation: str = "none"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AutoencoderLossConfig":
        raw = dict(cfg.get("loss") or {})
        if not raw:
            raw = {"type": cfg.get("loss_type", "raw_mse")}
        loss_type = str(raw.get("type", "raw_mse"))
        if loss_type != "raw_mse":
            raise ValueError("The retained Stage 1 recipe requires loss.type='raw_mse'")
        return cls(
            type=loss_type,
            prediction_activation=str(raw.get("prediction_activation", cfg.get("prediction_activation", "none"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "prediction_activation": self.prediction_activation,
        }


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: AutoencoderLossConfig | dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg if isinstance(cfg, AutoencoderLossConfig) else AutoencoderLossConfig.from_config(cfg or {})
    pred_for_loss = apply_prediction_activation(pred, loss_cfg.prediction_activation)
    parts = {"raw_mse": F.mse_loss(pred_for_loss, target)}
    parts["total"] = parts["raw_mse"]
    return parts["total"], parts
