"""R-GCN link prediction model for the unified neuroscience knowledge graph.

Phase 3, Step 8: relational graph convolutional network (Schlichtkrull et al.,
2018) with DistMult scoring for triple-level link prediction.

Architecture
------------
::

    Entity indices  (N_entities,)
          ↓
    nn.Embedding(N_entities, emb_dim)           ← learned
          ↓  (N_entities, emb_dim)
    RGCNConv(emb_dim, emb_dim, num_rels,
             num_bases=num_bases)  + ReLU
          ↓  (N_entities, emb_dim)
    RGCNConv(emb_dim, emb_dim, num_rels,
             num_bases=num_bases)
          ↓  (N_entities, emb_dim)          ← entity embeddings used for scoring

    DistMult scoring:
        score(s, r, o) = Σ  e_s ⊙ W_r ⊙ e_o
    where W_r = nn.Embedding(num_rels, emb_dim) diagonal relation matrix.

Relation types are passed as contiguous integers 0 … num_rels-1, which is
required by ``torch_geometric.nn.RGCNConv``.

Typical usage
-------------
>>> from neurovlm.gnn.rgcn import RGCNLinkPredictor
>>> model = RGCNLinkPredictor(num_entities=33784, num_relations=6)
>>> entity_emb = model.encode(edge_index, edge_type, device="cuda")
>>> scores = model.score(entity_emb, subj_idx, rel_idx, obj_idx)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class RGCNLinkPredictor(nn.Module):
    """R-GCN encoder + DistMult decoder for KG link prediction.

    Parameters
    ----------
    num_entities:
        Total number of canonical entities.
    num_relations:
        Number of distinct relation types (must be contiguous 0-based).
    emb_dim:
        Embedding dimension.  Default 256; use 512 for larger compute budgets.
    num_bases:
        Basis decomposition size for ``RGCNConv``.  Reduces parameter count
        when ``num_relations`` is small relative to ``emb_dim``.  Default 4.
    num_layers:
        Number of R-GCN message-passing layers (2 or 3).  Default 2.
    dropout:
        Dropout applied between layers.  Default 0.1.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        emb_dim: int = 256,
        num_bases: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        try:
            from torch_geometric.nn import RGCNConv
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required. Install with: pip install torch_geometric"
            ) from exc

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.dropout = dropout

        # Learned entity embeddings (input to R-GCN)
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        # R-GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(emb_dim, emb_dim, num_relations=num_relations, num_bases=num_bases)
            )

        # DistMult relation embeddings (diagonal W_r per relation)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(
        self,
        edge_index: Tensor,
        edge_type: Tensor,
        device: Optional[torch.device | str] = None,
    ) -> Tensor:
        """Run R-GCN message passing and return contextual entity embeddings.

        Parameters
        ----------
        edge_index:
            Training-graph edge index, shape ``(2, E_train)``.
        edge_type:
            Relation-type integers per training edge, shape ``(E_train,)``.
        device:
            Device to move tensors to before forward pass.  If *None*, uses
            the device of ``self.entity_emb``.

        Returns
        -------
        Tensor of shape ``(num_entities, emb_dim)`` — one embedding per entity.
        """
        if device is not None:
            edge_index = edge_index.to(device)
            edge_type  = edge_type.to(device)

        x = self.entity_emb.weight  # (N_e, emb_dim)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # (N_e, emb_dim)

    # ------------------------------------------------------------------
    # Decoder — DistMult scoring
    # ------------------------------------------------------------------

    def score(
        self,
        entity_emb: Tensor,
        subj_idx: Tensor,
        rel_idx: Tensor,
        obj_idx: Tensor,
    ) -> Tensor:
        """DistMult triple scoring.

        ``score(s, r, o) = Σ_d  e_s[d] * W_r[d] * e_o[d]``

        Parameters
        ----------
        entity_emb:
            Contextual entity embedding matrix, shape ``(N_e, emb_dim)``.
            Obtained from :meth:`encode`.
        subj_idx, rel_idx, obj_idx:
            1-D LongTensors of equal length — triple component indices.

        Returns
        -------
        1-D FloatTensor of scores, one per triple.
        """
        s_emb = entity_emb[subj_idx]        # (B, d)
        r_emb = self.relation_emb(rel_idx)  # (B, d)
        o_emb = entity_emb[obj_idx]         # (B, d)
        return (s_emb * r_emb * o_emb).sum(dim=-1)  # (B,)

    def score_all_objects(
        self,
        entity_emb: Tensor,
        subj_idx: Tensor,
        rel_idx: Tensor,
    ) -> Tensor:
        """Score a batch of (s, r, ?) against every entity as object.

        Efficient batch matrix form:

        ``score_all[b, j] = Σ_d  e_s[b, d] * W_r[b, d] * E[j, d]``

        Parameters
        ----------
        entity_emb:
            ``(N_e, emb_dim)``
        subj_idx:
            ``(B,)``
        rel_idx:
            ``(B,)``

        Returns
        -------
        Tensor of shape ``(B, N_e)`` — score of every entity as object.
        """
        s_emb = entity_emb[subj_idx]          # (B, d)
        r_emb = self.relation_emb(rel_idx)    # (B, d)
        sr    = s_emb * r_emb                 # (B, d)  element-wise
        return sr @ entity_emb.t()            # (B, N_e)

    # ------------------------------------------------------------------
    # Forward (used during training)
    # ------------------------------------------------------------------

    def forward(
        self,
        edge_index: Tensor,
        edge_type: Tensor,
        pos_subj: Tensor,
        pos_rel: Tensor,
        pos_obj: Tensor,
        neg_subj: Tensor,
        neg_rel: Tensor,
        neg_obj: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute positive and negative triple scores in one forward pass.

        Returns
        -------
        pos_scores : Tensor, shape ``(B,)``
        neg_scores : Tensor, shape ``(B * neg_ratio,)``
        """
        entity_emb = self.encode(edge_index, edge_type)
        pos_scores = self.score(entity_emb, pos_subj, pos_rel, pos_obj)
        neg_scores = self.score(entity_emb, neg_subj, neg_rel, neg_obj)
        return pos_scores, neg_scores
