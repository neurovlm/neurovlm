"""CoordGNN — Atlas-free coordinate Graph Attention Network (Track 2).

Architecture
------------
::

    Input node features  (N, 5)   [x, y, z, hemisphere, depth]
          ↓
    MLP: Linear(5→H) + LayerNorm + GELU + Linear(H→H) + LayerNorm
          ↓  (N, H)
    GATConv(H, H, heads=8, concat=True, edge_dim=4) + ELU + LayerNorm
          ↓  (N, H*8)
    GATConv(H*8, H, heads=8, concat=True, edge_dim=4) + ELU + LayerNorm
          ↓  (N, H*8)
    GATConv(H*8, H, heads=1, concat=False, edge_dim=4)
          ↓  (N, H)
    global_mean_pool(batch)
          ↓  (B, H)
    Linear(H → out_dim)
          ↓  (B, out_dim=384)

Critical differences from Track 1
-----------------------------------
- ``edge_dim=4`` in every GATConv — feeds 4-dim edge features (dist, dx, dy, dz)
  into the attention computation.  Without this, edge features are silently ignored.
- LayerNorm everywhere instead of BatchNorm — BatchNorm statistics are unreliable
  with variable-size graphs.
- GELU in the input MLP — smoother gradient for continuous coordinate inputs.
- Two-layer input projection with intermediate LayerNorm.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CoordGNN(nn.Module):
    """GAT encoder for coordinate-based brain graphs.

    Parameters
    ----------
    in_dim:
        Node feature dimension.  Default 5 (x, y, z, hemisphere, depth).
    hidden:
        Hidden dimension *per attention head*.  Default 128.
    heads:
        Number of attention heads in layers 1 and 2.  Default 8.
    out_dim:
        Output embedding dimension.  Default 384 (matches NeuroVLM latent space).
    dropout:
        Dropout applied inside GATConv attention.  Default 0.1.
    add_self_loops:
        If True, each GAT layer adds self-loop edges. Turn this off for
        ablations that test whether attention is dominated by node identity.
    """

    def __init__(
        self,
        in_dim: int = 5,
        hidden: int = 128,
        heads: int = 8,
        out_dim: int = 384,
        dropout: float = 0.1,
        add_self_loops: bool = True,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError as exc:
            raise ImportError(
                "PyTorch Geometric is required. "
                "Install with: pip install torch_geometric"
            ) from exc

        # Two-layer input MLP: maps raw (x,y,z,hemisphere,depth) → hidden space.
        # LayerNorm (not BatchNorm) so it's stable under variable batch statistics.
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # GATConv layers — edge_dim=4 is required to use the 4-dim edge features.
        # Omitting edge_dim silently discards the distance/direction information.
        self.conv1 = GATConv(
            hidden, hidden,
            heads=heads, concat=True,
            dropout=dropout, edge_dim=4, add_self_loops=add_self_loops,
        )
        self.conv2 = GATConv(
            hidden * heads, hidden,
            heads=heads, concat=True,
            dropout=dropout, edge_dim=4, add_self_loops=add_self_loops,
        )
        self.conv3 = GATConv(
            hidden * heads, hidden,
            heads=1, concat=False,
            dropout=dropout, edge_dim=4, add_self_loops=add_self_loops,
        )

        # LayerNorm after each concat layer (not BatchNorm)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)

        self.proj = nn.Linear(hidden, out_dim)
        self.dropout_p = dropout
        self.add_self_loops = add_self_loops

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, tuple]]:
        """Forward pass.

        Parameters
        ----------
        x:
            Node features, shape (total_nodes_in_batch, in_dim).
        edge_index:
            (2, total_edges).
        edge_attr:
            (total_edges, 4) — [dist_norm, dx, dy, dz].
        batch:
            Node-to-graph assignment vector, shape (total_nodes,).
        return_attention:
            If True, also return the attention weight tuple from conv3.

        Returns
        -------
        out : Tensor of shape (B, out_dim)
            or (out, attn_weights) if return_attention=True.
        """
        from torch_geometric.nn import global_mean_pool

        x = self.input_proj(x)                                         # (N, H)

        x1 = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))    # (N, H*8)
        x1 = self.norm1(x1)

        x2 = F.elu(self.conv2(x1, edge_index, edge_attr=edge_attr))   # (N, H*8)
        x2 = self.norm2(x2)

        if return_attention:
            x3, attn = self.conv3(
                x2, edge_index, edge_attr=edge_attr,
                return_attention_weights=True,
            )
        else:
            x3 = self.conv3(x2, edge_index, edge_attr=edge_attr)      # (N, H)

        out = global_mean_pool(x3, batch)   # (B, H)
        out = self.proj(out)                # (B, out_dim)

        if return_attention:
            return out, attn
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
