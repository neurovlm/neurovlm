"""Lightweight 3D CNN encoders for ALE-smoothed NeuroVLM brain maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


NormType = Literal["group", "batch", "instance", "none"]
PoolType = Literal["max", "stride"]
GlobalContextType = Literal["none", "se", "attention"]


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


class _ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        norm: NormType = "group",
    ):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = _norm_layer(norm, out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = _norm_layer(norm, out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _norm_layer(norm, out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class _SEGlobalContext3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.net(x).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * weights


class _AttentionGlobalToken3D(nn.Module):
    """Small token-mixing block over a pooled 3D feature grid."""

    def __init__(
        self,
        channels: int,
        *,
        pool_shape: tuple[int, int, int] = (4, 4, 4),
        num_heads: int = 4,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(pool_shape)
        heads = min(num_heads, channels)
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.pool(x).flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        mixed, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return mixed.mean(dim=1)


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


class ALEResNet3DEncoder(nn.Module):
    """Deeper residual 3D CNN for atlas-free ALE volumes.

    The raw input remains a dense 3D volume through spatial Conv3D stages.
    Flattening happens only after global pooling / multi-scale pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        num_stages: int = 4,
        blocks_per_stage: int = 2,
        out_dim: int = 384,
        dropout: float = 0.1,
        norm: NormType = "group",
        use_dilation: bool = False,
        multi_scale: bool = False,
        global_context: GlobalContextType = "none",
        attention_pool_shape: tuple[int, int, int] = (4, 4, 4),
    ):
        super().__init__()
        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if num_stages > 5:
            raise ValueError("num_stages > 5 is not recommended for current ALE volumes")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1")

        self.out_dim = out_dim
        self.multi_scale = multi_scale
        self.global_context_type = global_context

        stem_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, stem_channels),
            nn.GELU(),
        )

        stages = []
        channels = [base_channels * (2**i) for i in range(num_stages)]
        prev = stem_channels
        for stage_idx, ch in enumerate(channels):
            stage_blocks = []
            stride = 1 if stage_idx == 0 else 2
            dilation = 1
            if use_dilation and stage_idx >= max(2, num_stages - 2):
                dilation = 2 if stage_idx == num_stages - 2 else 4
                stride = 1
            stage_blocks.append(
                _ResidualBlock3D(prev, ch, stride=stride, dilation=dilation, norm=norm)
            )
            for _ in range(1, blocks_per_stage):
                stage_blocks.append(_ResidualBlock3D(ch, ch, dilation=dilation, norm=norm))
            stages.append(nn.Sequential(*stage_blocks))
            prev = ch
        self.stages = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        final_channels = channels[-1]
        if global_context == "se":
            self.global_context = _SEGlobalContext3D(final_channels)
            context_dim = 0
        elif global_context == "attention":
            self.global_context = _AttentionGlobalToken3D(
                final_channels, pool_shape=attention_pool_shape
            )
            context_dim = final_channels
        elif global_context == "none":
            self.global_context = nn.Identity()
            context_dim = 0
        else:
            raise ValueError("global_context must be 'none', 'se', or 'attention'")

        pooled_dim = sum(channels) if multi_scale else final_channels
        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim + context_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim + context_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, D, H, W), got {tuple(x.shape)}")
        x = self.stem(x)
        pooled = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            is_last = stage_idx == len(self.stages) - 1
            if is_last and self.global_context_type not in {"none", "attention"}:
                x = self.global_context(x)
            if self.multi_scale:
                pooled.append(self.global_pool(x).flatten(1))
        if self.global_context_type == "attention":
            context = self.global_context(x)
        else:
            context = None
        if self.multi_scale:
            feats = torch.cat(pooled, dim=1)
        else:
            feats = self.global_pool(x).flatten(1)
        if context is not None:
            feats = torch.cat([feats, context], dim=1)
        return self.proj(feats)

    def count_parameters(self) -> int:
        return count_parameters(self)


class _DeconvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: NormType = "group",
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = _norm_layer(norm, out_ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.deconv(x)))


class ALE3DCNNDecoder(nn.Module):
    """Decode an ALE3DCNN latent vector back to a dense ALE volume.

    The encoder compresses the spatial feature map with adaptive average
    pooling, so the decoder learns a compact seed grid from the latent vector
    and upsamples it back to the requested output shape. A final interpolation
    step makes non power-of-two cropped brain shapes reconstruct exactly.
    """

    def __init__(
        self,
        output_shape: tuple[int, int, int],
        latent_dim: int = 384,
        out_channels: int = 1,
        base_channels: int = 16,
        num_blocks: int = 3,
        norm: NormType = "group",
    ):
        super().__init__()
        if len(output_shape) != 3:
            raise ValueError("output_shape must be a 3D (D, H, W) tuple")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.output_shape = tuple(int(v) for v in output_shape)
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks

        start_channels = base_channels * (2 ** (num_blocks - 1))
        scale = 2**num_blocks
        self.seed_shape = tuple(
            max(1, (dim + scale - 1) // scale) for dim in self.output_shape
        )
        self.fc = nn.Linear(latent_dim, start_channels * math.prod(self.seed_shape))

        channels = [base_channels * (2**i) for i in reversed(range(num_blocks))]
        blocks = []
        prev = start_channels
        for ch in channels[1:]:
            blocks.append(_DeconvBlock(prev, ch, norm=norm))
            prev = ch
        blocks.append(_DeconvBlock(prev, base_channels, norm=norm))
        self.up = nn.Sequential(*blocks)
        self.out = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected latent shape (B, D), got {tuple(z.shape)}")
        x = self.fc(z).view(
            z.shape[0],
            self.base_channels * (2 ** (self.num_blocks - 1)),
            *self.seed_shape,
        )
        x = self.up(x)
        if tuple(x.shape[-3:]) != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode="trilinear", align_corners=False)
        return self.out(x)

    def count_parameters(self) -> int:
        return count_parameters(self)


class ALE3DCNNAutoEncoder(nn.Module):
    """Atlas-free ALE volume autoencoder using the existing ALE3DCNN encoder."""

    def __init__(
        self,
        output_shape: tuple[int, int, int],
        *,
        in_channels: int = 1,
        base_channels: int = 16,
        num_blocks: int = 3,
        latent_dim: int = 384,
        dropout: float = 0.1,
        norm: NormType = "group",
        pooling: PoolType = "max",
        encoder_arch: Literal["plain", "resnet"] = "plain",
        blocks_per_stage: int = 2,
        use_dilation: bool = False,
        multi_scale: bool = False,
        global_context: GlobalContextType = "none",
    ):
        super().__init__()
        if encoder_arch == "plain":
            self.encoder = ALE3DCNNEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks=num_blocks,
                out_dim=latent_dim,
                dropout=dropout,
                norm=norm,
                pooling=pooling,
            )
        elif encoder_arch == "resnet":
            self.encoder = ALEResNet3DEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_blocks,
                blocks_per_stage=blocks_per_stage,
                out_dim=latent_dim,
                dropout=dropout,
                norm=norm,
                use_dilation=use_dilation,
                multi_scale=multi_scale,
                global_context=global_context,
            )
        else:
            raise ValueError("encoder_arch must be 'plain' or 'resnet'")
        self.decoder = ALE3DCNNDecoder(
            output_shape=output_shape,
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            norm=norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))

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
