"""PyG dataset for DiFuMo brain graphs.

Phase 2: each sample is one paper represented as a graph whose 512 nodes
are DiFuMo components.  The graph topology is identical for every paper;
only the node features (DiFuMo coefficient vector) change.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional, Tuple


class BrainGraphDataset:
    """In-memory PyG dataset of brain graphs.

    Each item returned by :meth:`__getitem__` is a
    ``torch_geometric.data.Data`` object with the following attributes:

    * ``x``          — node features, shape ``(512, n_node_features)``
    * ``edge_index`` — fixed graph topology, shape ``(2, E)``
    * ``edge_attr``  — FC edge weights, shape ``(E, 1)``
    * ``y``          — SPECTER text embedding, shape ``(768,)``

    Parameters
    ----------
    difumo_coeffs:
        DiFuMo coefficient matrix, shape ``(N, 512)``.  One row per paper.
    edge_index:
        Fixed adjacency, shape ``(2, E)``.  Same for all papers.
    edge_attr:
        Fixed edge weights, shape ``(E, 1)``.  Same for all papers.
    text_embeddings:
        SPECTER text embeddings, shape ``(N, 768)``.  Aligned with
        *difumo_coeffs* row-for-row.
    extra_node_feats:
        Optional additional node features to concatenate with the scalar
        coefficient value.  Shape ``(512, d_extra)``.  Typical use: component
        centroid coordinates ``(512, 3)`` or hemisphere/lobe flags.
    """

    def __init__(
        self,
        difumo_coeffs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        text_embeddings: Tensor,
        extra_node_feats: Optional[Tensor] = None,
    ):
        assert difumo_coeffs.shape[0] == text_embeddings.shape[0], (
            f"difumo_coeffs has {difumo_coeffs.shape[0]} rows but "
            f"text_embeddings has {text_embeddings.shape[0]} rows."
        )
        self.difumo_coeffs = difumo_coeffs.float()      # (N, 512)
        self.edge_index = edge_index.long()             # (2, E)
        self.edge_attr = edge_attr.float()              # (E, 1)
        self.text_embeddings = text_embeddings.float()  # (N, 768)
        self.extra_node_feats = extra_node_feats        # (512, d_extra) | None

    # ------------------------------------------------------------------
    # Standard dataset interface (works with PyG DataLoader out of the box)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.difumo_coeffs.shape[0]

    def __getitem__(self, idx: int):
        from torch_geometric.data import Data

        # Node features: scalar activation per node → (512, 1)
        # Optionally concatenate extra features → (512, 1 + d_extra)
        x = self.difumo_coeffs[idx].unsqueeze(1)          # (512, 1)
        if self.extra_node_feats is not None:
            x = torch.cat([x, self.extra_node_feats], dim=1)

        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.text_embeddings[idx].unsqueeze(0),  # (1, 768) so Batch.cat → (B, 768)
        )

    # ------------------------------------------------------------------
    # Train / val / test splitting
    # ------------------------------------------------------------------

    def split(
        self,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple["BrainGraphDataset", "BrainGraphDataset", "BrainGraphDataset"]:
        """Randomly split into train / val / test subsets.

        Returns three :class:`BrainGraphDataset` instances sharing the same
        ``edge_index`` / ``edge_attr`` tensors (no copy).

        Parameters
        ----------
        val_frac, test_frac:
            Fraction of data for validation / test sets.  The remainder is
            used for training.
        seed:
            Random seed for reproducible splits.
        """
        N = len(self)
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=rng)

        n_test = int(N * test_frac)
        n_val = int(N * val_frac)

        test_idx = perm[:n_test]
        val_idx = perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]

        def _subset(idx: Tensor) -> "BrainGraphDataset":
            extra = (
                self.extra_node_feats if self.extra_node_feats is None
                else self.extra_node_feats   # shared; node-level, not sample-level
            )
            return BrainGraphDataset(
                difumo_coeffs=self.difumo_coeffs[idx],
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                text_embeddings=self.text_embeddings[idx],
                extra_node_feats=extra,
            )

        return _subset(train_idx), _subset(val_idx), _subset(test_idx)

    def split_by_index(
        self,
        train_idx: Tensor,
        val_idx: Tensor,
        test_idx: Tensor,
    ) -> Tuple["BrainGraphDataset", "BrainGraphDataset", "BrainGraphDataset"]:
        """Split using precomputed index arrays (e.g. DOI-based holdout).

        Use this when you want the same paper-level splits as the baseline
        NeuroVLM experiment for an apples-to-apples comparison.
        """
        def _subset(idx: Tensor) -> "BrainGraphDataset":
            return BrainGraphDataset(
                difumo_coeffs=self.difumo_coeffs[idx],
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                text_embeddings=self.text_embeddings[idx],
                extra_node_feats=self.extra_node_feats,
            )
        return _subset(train_idx), _subset(val_idx), _subset(test_idx)
