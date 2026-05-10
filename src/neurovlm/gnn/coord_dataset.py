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
    cache_file:
        Optional single-file cache for all pre-computed graphs. This is much
        faster on Colab/Drive than managing thousands of tiny ``paper_*.pt``
        files.
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
        cache_file: Optional[str] = None,
        k: int = 7,
        max_dist_mm: float = 30.0,
        preload_to_ram: bool = False,
    ):
        self.k = k
        self.max_dist_mm = max_dist_mm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = Path(cache_file) if cache_file is not None else None
        if self.cache_file is not None:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

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

        # Optional: load all graphs into RAM for fast __getitem__ (eliminates Drive I/O).
        # Each graph is ~2KB; 30K graphs ≈ 60MB total — trivial on A100 (83GB system RAM).
        self._ram_cache: Optional[List[Data]] = None
        if self.cache_file is not None:
            self._build_or_load_packed_cache()
        else:
            # Backward-compatible legacy mode: one graph file per paper.
            self._build_cache()

        if preload_to_ram and self._ram_cache is None:
            self._load_to_ram()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, dataset_idx: int) -> Path:
        return self.cache_dir / f"paper_{dataset_idx}.pt"

    def _build_graph(self, dataset_idx: int) -> Data:
        pmid = str(self.unique_pmids[dataset_idx])
        raw_coords = self._pmid_to_coords[pmid]

        # Deduplicate peaks within the same paper
        raw_coords = np.unique(raw_coords, axis=0)

        if len(raw_coords) == 0:
            raise ValueError(f"Paper {pmid} has no coordinates after deduplication")

        norm_coords = normalize_coords(raw_coords)
        return coords_to_graph(norm_coords, k=self.k, max_dist_mm=self.max_dist_mm)

    def _pmids_digest(self) -> str:
        import hashlib

        pmids = np.asarray(self.unique_pmids, dtype=str)
        return hashlib.sha1("\n".join(pmids.tolist()).encode("utf-8")).hexdigest()

    def _build_or_load_packed_cache(self) -> None:
        """Load or build one packed graph cache file.

        A single packed cache avoids the main Colab bottleneck: repeated
        metadata checks and tiny-file reads/writes through Google Drive FUSE.
        """
        assert self.cache_file is not None
        pmids_digest = self._pmids_digest()

        if self.cache_file.exists():
            payload = torch.load(self.cache_file, weights_only=False)
            if (
                payload.get("version") == 1
                and payload.get("k") == self.k
                and float(payload.get("max_dist_mm", "nan")) == float(self.max_dist_mm)
                and payload.get("n_unique_pmids") == len(self.unique_pmids)
                and payload.get("pmids_digest") == pmids_digest
            ):
                self._valid_indices = list(payload["valid_indices"])
                self._ram_cache = list(payload["graphs"])
                print(
                    f"CoordGraphDataset: loaded {len(self._ram_cache):,} graphs "
                    f"from packed cache {self.cache_file}."
                )
                return

            print(
                "CoordGraphDataset: packed cache parameters changed; "
                "rebuilding graph cache."
            )

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kw):  # type: ignore[misc]
                return it

        print(
            f"CoordGraphDataset: building {len(self._valid_indices):,} graphs "
            f"into packed cache {self.cache_file} "
            f"(k={self.k}, max_dist={self.max_dist_mm}mm)."
        )

        valid_indices: List[int] = []
        graphs: List[Data] = []
        for i in tqdm(self._valid_indices, desc="Building coord graphs", unit="graph"):
            try:
                graph = self._build_graph(i)
            except ValueError:
                continue
            valid_indices.append(i)
            graphs.append(graph)

        payload = {
            "version": 1,
            "k": self.k,
            "max_dist_mm": self.max_dist_mm,
            "n_unique_pmids": len(self.unique_pmids),
            "pmids_digest": pmids_digest,
            "valid_indices": valid_indices,
            "graphs": graphs,
        }
        torch.save(payload, self.cache_file)

        self._valid_indices = valid_indices
        self._ram_cache = graphs
        print(f"Packed graph cache build complete ({len(graphs):,} graphs).")

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
            try:
                graph = self._build_graph(i)
            except ValueError:
                continue
            torch.save(graph, self._cache_path(i))

        print("Graph cache build complete.")

    def _load_to_ram(self) -> None:
        """Load all cached graphs into a RAM list for zero-latency __getitem__."""
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kw):  # type: ignore[misc]
                return it

        print(f"Preloading {len(self._valid_indices)} graphs to RAM …")
        ram: List[Optional[Data]] = [None] * len(self._valid_indices)
        for pos, dataset_idx in enumerate(tqdm(self._valid_indices, desc="Loading to RAM")):
            ram[pos] = torch.load(self._cache_path(dataset_idx), weights_only=False)
        self._ram_cache = ram
        # Estimate memory
        try:
            import sys
            mb = sum(sys.getsizeof(g) for g in ram if g is not None) / 1e6
            print(f"RAM cache loaded ({mb:.0f} MB estimated).")
        except Exception:
            print("RAM cache loaded.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._valid_indices)

    @property
    def pmids(self) -> np.ndarray:
        return self.unique_pmids[self._valid_indices].astype(str)

    @property
    def raw_text_embeddings(self) -> Tensor:
        return self.text_embeddings[self._valid_indices]

    def __getitem__(self, idx: int) -> Data:
        dataset_idx = self._valid_indices[idx]

        if self._ram_cache is not None:
            # Zero-latency path: graph is already in RAM
            graph = self._ram_cache[idx]
        else:
            graph = torch.load(self._cache_path(dataset_idx), weights_only=False)

        # Clone so DataLoader workers don't share tensors across batches
        graph = graph.clone()
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

    @property
    def pmids(self) -> np.ndarray:
        return self._parent.pmids[self._positions]

    @property
    def raw_text_embeddings(self) -> Tensor:
        return self._parent.raw_text_embeddings[self._positions]
