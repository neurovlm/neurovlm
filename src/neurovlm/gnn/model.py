"""BrainGAT — Graph Attention Network over DiFuMo brain components.

Phase 3: three-layer GAT encoder that maps a brain graph (512 nodes,
variable edges) to a fixed-size embedding via global mean pooling.

Architecture
------------
::

    Input node features  (512, in_dim)
          ↓
    GATConv(in_dim, hidden, heads=8, concat=True)  + ELU
          ↓  (512, hidden*8)
    GATConv(hidden*8, hidden, heads=8, concat=True) + ELU
          ↓  (512, hidden*8)
    GATConv(hidden*8, hidden, heads=1, concat=False) + ELU
          ↓  (512, hidden)
    global_mean_pool(batch)
          ↓  (B, hidden)
    Linear(hidden → out_dim)
          ↓  (B, out_dim)

The default output dimension is 384 to match the NeuroVLM shared latent
space; the text side is projected separately by :class:`TextProjHead`.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple


class BrainGAT(nn.Module):
    """GAT encoder for DiFuMo brain graphs.

    Parameters
    ----------
    in_dim:
        Number of input features per node.  Use 1 for the scalar DiFuMo
        activation value.  Increase to 4 if you append centroid coordinates
        (x, y, z) as additional node features.
    hidden:
        Hidden dimension per attention head.
    heads:
        Number of attention heads in layers 1 and 2.
    out_dim:
        Output embedding dimension.  Default 384 matches NeuroVLM's latent
        space.
    dropout:
        Dropout probability applied to attention coefficients.
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden: int = 64,
        heads: int = 8,
        out_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv, global_mean_pool
        except ImportError as exc:
            raise ImportError(
                "PyTorch Geometric is required for BrainGAT. "
                "Install it with: pip install torch_geometric"
            ) from exc

        self.conv1 = GATConv(
            in_dim, hidden, heads=heads, concat=True,
            dropout=dropout, add_self_loops=True,
        )
        self.conv2 = GATConv(
            hidden * heads, hidden, heads=heads, concat=True,
            dropout=dropout, add_self_loops=True,
        )
        self.conv3 = GATConv(
            hidden * heads, hidden, heads=1, concat=False,
            dropout=dropout, add_self_loops=True,
        )
        self.proj = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Node features, shape ``(total_nodes, in_dim)``.
            When using PyG's DataLoader, *total_nodes* = ``batch_size × 512``.
        edge_index:
            Edge indices, shape ``(2, total_edges)``.
        edge_attr:
            Edge weights, shape ``(total_edges, 1)`` or ``None``.
        batch:
            Node-to-graph assignment vector, shape ``(total_nodes,)``.

        Returns
        -------
        Tensor, shape ``(batch_size, out_dim)``
        """
        from torch_geometric.nn import global_mean_pool

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)   # (B, hidden)
        return self.proj(x)              # (B, out_dim)

    @torch.no_grad()
    def get_attention_weights(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return attention weights from all three layers (for interpretability).

        Returns
        -------
        attn1, attn2, attn3 : each a FloatTensor of shape ``(E, heads)``
        """
        from torch_geometric.nn import global_mean_pool  # noqa: F401

        _, (ei1, a1) = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(self.conv1(x, edge_index))

        _, (ei2, a2) = self.conv2(x1, edge_index, return_attention_weights=True)
        x2 = F.elu(self.conv2(x1, edge_index))

        _, (ei3, a3) = self.conv3(x2, edge_index, return_attention_weights=True)

        return a1, a2, a3


class TextProjHead(nn.Module):
    """Project SPECTER text embeddings (768-dim) to the GAT latent space (384-dim).

    Architecture: Linear(768, 512) → ReLU → Linear(512, 384).
    Identical in structure to NeuroVLM's ``ProjHead`` so that the two models
    are directly comparable.

    Parameters
    ----------
    in_dim:
        Input dimension.  Default 768 (SPECTER output).
    hidden_dim:
        Hidden layer size.  Default 512.
    out_dim:
        Output dimension, must match :attr:`BrainGAT.out_dim`.  Default 384.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 384,
    ):
        super().__init__()
        self.aligner = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.aligner(x)


class BrainGATWithProjHead(nn.Module):
    """Convenience wrapper: BrainGAT + TextProjHead as a single module.

    Useful for saving/loading the full model in a single checkpoint.

    Parameters
    ----------
    gat_kwargs:
        Keyword arguments forwarded to :class:`BrainGAT`.
    proj_kwargs:
        Keyword arguments forwarded to :class:`TextProjHead`.
    """

    def __init__(self, gat_kwargs: dict | None = None, proj_kwargs: dict | None = None):
        super().__init__()
        self.brain_encoder = BrainGAT(**(gat_kwargs or {}))
        self.text_proj = TextProjHead(**(proj_kwargs or {}))

    def encode_brain(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
    ) -> Tensor:
        return self.brain_encoder(x, edge_index, edge_attr, batch)

    def encode_text(self, text_emb: Tensor) -> Tensor:
        return self.text_proj(text_emb)
