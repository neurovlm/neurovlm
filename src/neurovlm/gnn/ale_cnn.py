"""Lightweight 3D CNN encoders for ALE-smoothed NeuroVLM brain maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


NormType = Literal["group", "batch", "instance", "none"]
PoolType = Literal["max", "stride"]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of model parameters."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _norm_layer(norm: NormType, channels: int) -> nn.Module:
    if norm == "group":
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm}")


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: NormType = "group",
        pooling: PoolType = "max",
    ):
        super().__init__()
        stride = 2 if pooling == "stride" else 1
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.norm = _norm_layer(norm, out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool3d(kernel_size=2) if pooling == "max" else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.norm(self.conv(x)))
        return self.pool(x)


class ALE3DCNNEncoder(nn.Module):
    """Small dense 3D CNN for paper-level ALE activation volumes.

    The output dimension defaults to 384 so it can be trained against the same
    projected SPECTER space as the coordinate GNN and NeuroVLM contrastive
    encoders.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        num_blocks: int = 3,
        out_dim: int = 384,
        dropout: float = 0.1,
        norm: NormType = "group",
        pooling: PoolType = "max",
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if num_blocks > 4:
            raise ValueError("num_blocks > 4 is intentionally unsupported for this lightweight encoder")

        channels = [base_channels * (2**i) for i in range(num_blocks)]
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(_ConvBlock(prev, ch, norm=norm, pooling=pooling))
            prev = ch

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(channels[-1], out_dim)
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, D, H, W), got {tuple(x.shape)}")
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.proj(x)

    def count_parameters(self) -> int:
        return count_parameters(self)


class ALEFlatMLPEncoder(nn.Module):
    """Flattened atlas-free ALE baseline with lazy input sizing."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        out_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return count_parameters(self)


@dataclass(frozen=True)
class ModelSummary:
    name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parameters: int


@torch.no_grad()
def summarize_encoder(model: nn.Module, input_shape: tuple[int, ...]) -> ModelSummary:
    """Run a CPU shape check and return a compact model summary."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(input_shape, device=device)
    out = model(dummy)
    if was_training:
        model.train()
    return ModelSummary(
        name=model.__class__.__name__,
        input_shape=tuple(input_shape),
        output_shape=tuple(out.shape),
        parameters=count_parameters(model),
    )


def embedding_covariate_correlations(
    embeddings: Tensor,
    covariates,
    n_components: int = 8,
):
    """Reuse the coordinate diagnostic implementation for ALE experiments."""
    from neurovlm.gnn.coord_diagnostics import embedding_covariate_correlations as _impl

    return _impl(embeddings, covariates, n_components=n_components)

