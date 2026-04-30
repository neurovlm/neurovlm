"""Coordinate-based brain graph construction for Track 2 (Atlas-Free).

Converts raw MNI peak coordinates (N, 3) into a PyG Data object with
5-dim node features and 4-dim edge features.

Node features (N, 5): [x_norm, y_norm, z_norm, hemisphere_flag, depth_proxy]
Edge features (E, 4): [dist_normalized, dx, dy, dz]

MNI normalization uses the half-widths of the MNI152 bounding box:
    x ÷ 90,  y ÷ 126,  z ÷ 108
so that all valid coordinates land in [-1, 1].
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

# MNI bounding-box half-widths (mm) — used for normalizing and denormalizing
MNI_HALF = torch.tensor([90.0, 126.0, 108.0])

_MNI_X = 90.0
_MNI_Y = 126.0
_MNI_Z = 108.0


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize raw MNI (x, y, z) coordinates to [-1, 1].

    Outliers beyond ±1.05 (malformed coordinates) are clipped to ±1.0.

    Parameters
    ----------
    coords : (N, 3) float array in MNI mm space.

    Returns
    -------
    (N, 3) float32 array with values in [-1, 1].
    """
    half = np.array([_MNI_X, _MNI_Y, _MNI_Z], dtype=np.float32)
    normalized = coords.astype(np.float32) / half
    return np.clip(normalized, -1.0, 1.0)


def denormalize_coords(coords_norm: np.ndarray) -> np.ndarray:
    """Invert normalize_coords — returns MNI mm values."""
    half = np.array([_MNI_X, _MNI_Y, _MNI_Z], dtype=np.float32)
    return coords_norm.astype(np.float32) * half


def coords_to_graph(
    coords: np.ndarray,
    k: int = 7,
    max_dist_mm: float = 30.0,
) -> Data:
    """Build a KNN spatial graph from normalized MNI peak coordinates.

    Parameters
    ----------
    coords:
        (N, 3) *normalized* MNI coordinates (values in [-1, 1]).
        Call normalize_coords() first if you have raw mm values.
    k:
        Number of nearest-neighbor edges per node.
    max_dist_mm:
        Prune edges whose Euclidean distance (in mm) exceeds this value.

    Returns
    -------
    torch_geometric.data.Data with:
        x          — node features,  shape (N, 5)
        edge_index — COO edge list,  shape (2, E)
        edge_attr  — edge features,  shape (E, 4)
    """
    from torch_geometric.nn import knn_graph

    x = torch.tensor(coords, dtype=torch.float)  # (N, 3) normalized

    if len(x) == 1:
        # Single-node paper: give it a self-loop so the model can still run
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros(1, 4)
        return Data(x=_build_node_features(x),
                    edge_index=edge_index,
                    edge_attr=edge_attr)

    # actual_k clamp is NOT optional — papers with fewer than k+1 peaks crash without it
    actual_k = min(k, len(x) - 1)
    edge_index = knn_graph(x, k=actual_k, loop=False)

    src, dst = edge_index
    diff = x[src] - x[dst]                          # (E, 3) direction in normed space
    diff_mm = diff * MNI_HALF.to(x.device)          # (E, 3) direction in mm
    dist_mm = diff_mm.norm(dim=-1, keepdim=True)     # (E, 1) Euclidean dist in mm

    # Prune long-range edges
    mask = dist_mm.squeeze(-1) <= max_dist_mm
    if not mask.any():
        # Degenerate case: keep shortest edge to avoid disconnected graph
        mask[dist_mm.squeeze(-1).argmin()] = True

    edge_index = edge_index[:, mask]
    diff = diff[mask]
    dist_mm = dist_mm[mask]

    # 4-dim edge feature: [dist_normalized, dx, dy, dz]
    edge_attr = torch.cat([dist_mm / max_dist_mm, diff], dim=-1)  # (E, 4)

    return Data(x=_build_node_features(x),
                edge_index=edge_index,
                edge_attr=edge_attr)


def _build_node_features(x: torch.Tensor) -> torch.Tensor:
    """Build 5-dim node features from 3-dim normalized coordinates.

    Features:
        x, y, z          — normalized MNI coordinates
        hemisphere_flag   — (x > 0).float(): right=1, left=0, midline≈0
        depth_proxy       — z coordinate (rough cortico-subcortical proxy)
    """
    hemisphere = (x[:, 0] > 0).float().unsqueeze(-1)  # (N, 1)
    depth = x[:, 2].unsqueeze(-1)                      # (N, 1)
    return torch.cat([x, hemisphere, depth], dim=-1)   # (N, 5)
