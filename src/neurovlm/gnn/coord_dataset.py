"""CoordGraphDataset — PyG-compatible dataset for Track 2 (Atlas-Free Coord GNN).

Each sample is one paper represented as a spatial KNN graph over its raw MNI
peak activation coordinates.  Unlike Track 1, each paper has a *different*
number of nodes (peaks), so PyG handles variable-size batching automatically
via the batch vector.

Graphs are pre-computed and cached to disk on first load.  Building KNN graphs
lazily inside the DataLoader would be ~50× slower.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from .coord_graph import coords_to_graph, normalize_coords


class CoordGraphDataset:
    """Dataset of coordinate-based brain graphs, one per paper.

    Parameters
    ----------
    coords_df:
        DataFrame with columns ['pmid', 'x', 'y', 'z'] (MNI mm space).
    text_embeddings:
        Tensor of shape (N, 768) with SPECTER embeddings aligned row-for-row
        with unique_pmids.
    unique_pmids:
        Array of length N giving the PMID for each row in text_embeddings.
    cache_dir:
        Directory for pre-computed graph .pt files.  Created automatically.
    k:
        KNN neighbor count.  Default 7.
    max_dist_mm:
        Max edge distance in mm.  Default 30.0.
    """

    def __init__(
        self,
        coords_df: pd.DataFrame,
        text_embeddings: Tensor,
        unique_pmids: np.ndarray,
        cache_dir: str = "data/coord_graphs",
        k: int = 7,
        max_dist_mm: float = 30.0,
    ):
        self.k = k
        self.max_dist_mm = max_dist_mm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.text_embeddings = text_embeddings.float()   # (N, 768)
        self.unique_pmids = np.asarray(unique_pmids)     # (N,)

        # Build per-paper coordinate lookup: str(pmid) → np.ndarray (M, 3)
        coords_df = coords_df.copy()
        coords_df["pmid"] = coords_df["pmid"].astype(str)
        self._pmid_to_coords: dict[str, np.ndarray] = {
            pmid: grp[["x", "y", "z"]].values.astype(np.float32)
            for pmid, grp in coords_df.groupby("pmid")
        }

        # Only keep indices that have at least 1 coordinate row
        self._valid_indices: List[int] = [
            i for i, pmid in enumerate(self.unique_pmids)
            if str(pmid) in self._pmid_to_coords
        ]

        # Build disk cache before any __getitem__ call
        self._build_cache()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, dataset_idx: int) -> Path:
        return self.cache_dir / f"paper_{dataset_idx}.pt"

    def _build_cache(self) -> None:
        """Pre-compute and save all graphs to disk (with progress bar)."""
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kw):  # type: ignore[misc]
                return it

        missing = [i for i in self._valid_indices
                   if not self._cache_path(i).exists()]
        cached = len(self._valid_indices) - len(missing)

        print(f"CoordGraphDataset: {cached} graphs cached, "
              f"{len(missing)} need building "
              f"(k={self.k}, max_dist={self.max_dist_mm}mm).")

        if not missing:
            return

        for i in tqdm(missing, desc="Building coord graphs", unit="graph"):
            pmid = str(self.unique_pmids[i])
            raw_coords = self._pmid_to_coords[pmid]

            # Deduplicate peaks within the same paper
            raw_coords = np.unique(raw_coords, axis=0)

            if len(raw_coords) == 0:
                continue

            norm_coords = normalize_coords(raw_coords)
            graph = coords_to_graph(norm_coords, k=self.k,
                                    max_dist_mm=self.max_dist_mm)
            torch.save(graph, self._cache_path(i))

        print("Graph cache build complete.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Data:
        dataset_idx = self._valid_indices[idx]
        graph: Data = torch.load(self._cache_path(dataset_idx),
                                 weights_only=False)

        # Attach text embedding and paper identifier
        graph.y = self.text_embeddings[dataset_idx].unsqueeze(0)  # (1, 768)
        graph.paper_idx = dataset_idx
        return graph

    # ------------------------------------------------------------------
    # Train / val / test splitting
    # ------------------------------------------------------------------

    def split(
        self,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple["_SubsetCoordDataset", "_SubsetCoordDataset", "_SubsetCoordDataset"]:
        """Random 80/10/10 split (reproducible)."""
        N = len(self)
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=rng).tolist()

        n_test = int(N * test_frac)
        n_val = int(N * val_frac)
        test_pos = perm[:n_test]
        val_pos = perm[n_test: n_test + n_val]
        train_pos = perm[n_test + n_val:]

        return (
            _SubsetCoordDataset(self, train_pos),
            _SubsetCoordDataset(self, val_pos),
            _SubsetCoordDataset(self, test_pos),
        )

    def split_by_index(
        self,
        train_pos: List[int],
        val_pos: List[int],
        test_pos: List[int],
    ) -> Tuple["_SubsetCoordDataset", "_SubsetCoordDataset", "_SubsetCoordDataset"]:
        """Split using precomputed position lists (e.g., DOI-based holdout)."""
        return (
            _SubsetCoordDataset(self, train_pos),
            _SubsetCoordDataset(self, val_pos),
            _SubsetCoordDataset(self, test_pos),
        )


class _SubsetCoordDataset:
    """Lightweight view over a subset of CoordGraphDataset positions."""

    def __init__(self, parent: CoordGraphDataset, positions: List[int]):
        self._parent = parent
        self._positions = positions

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, idx: int) -> Data:
        return self._parent[self._positions[idx]]
