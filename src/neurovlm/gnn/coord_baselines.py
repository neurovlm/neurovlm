"""Coordinate-only baseline encoders for graph ablations."""

from __future__ import annotations

import torch
from torch import nn, Tensor


class CoordDeepSet(nn.Module):
    """Permutation-invariant coordinate set encoder.

    This intentionally ignores ``edge_index`` and ``edge_attr`` while keeping
    the same trainer interface as :class:`CoordGNN`. If this matches CoordGNN,
    graph connectivity is probably not adding much beyond peak features.
    """

    def __init__(
        self,
        in_dim: int = 5,
        hidden: int = 256,
        out_dim: int = 384,
        dropout: float = 0.1,
        pool: str = "mean_max",
    ):
        super().__init__()
        if pool not in {"mean", "mean_max"}:
            raise ValueError("pool must be 'mean' or 'mean_max'")

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        pooled_dim = hidden * 2 if pool == "mean_max" else hidden
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.pool = pool

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None,
        edge_attr: Tensor | None,
        batch: Tensor,
        return_attention: bool = False,
    ) -> Tensor:
        if return_attention:
            raise ValueError("CoordDeepSet does not expose attention weights")

        from torch_geometric.nn import global_max_pool, global_mean_pool

        h = self.node_mlp(x)
        pooled = global_mean_pool(h, batch)
        if self.pool == "mean_max":
            pooled = torch.cat([pooled, global_max_pool(h, batch)], dim=1)
        return self.proj(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
