"""Brain graph construction from a DiFuMo functional-connectivity matrix.

Phase 1, steps 3–6: build the 512-node brain graph whose edges encode
resting-state functional connectivity between DiFuMo components.

Two FC sources are supported:

1. **Precomputed file** — a (512, 512) numpy array saved as ``.npy`` or
   ``.npz``.  Ideal source: the HCP group-average parcellated FC matrix
   projected into DiFuMo space (download from
   humanconnectome.org → Group Average Data → Dense FC).

2. **Data-derived** — correlate DiFuMo coefficient vectors across the
   training papers.  Quick to compute with no external download required;
   captures co-activation structure rather than true resting-state FC, but
   serves as a reasonable baseline.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# FC matrix loading / computation
# ---------------------------------------------------------------------------

def load_fc_matrix(path: str | Path) -> np.ndarray:
    """Load a precomputed FC matrix from disk.

    Parameters
    ----------
    path:
        Path to a ``.npy`` (shape ``(K, K)``) or ``.npz`` file.
        For ``.npz``, the array must be stored under the key ``"fc"``.

    Returns
    -------
    fc : np.ndarray, shape (K, K)
        Symmetric functional connectivity matrix with values in ``[-1, 1]``.
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        fc = data["fc"]
    else:
        fc = np.load(path)
    return fc.astype(np.float32)


def compute_fc_from_coefficients(difumo_coeffs: np.ndarray) -> np.ndarray:
    """Estimate a DiFuMo-space FC matrix from per-paper coefficient vectors.

    Computes the Pearson correlation matrix across the *paper* dimension:
    ``FC[i, j] = corr(s[:, i], s[:, j])`` where ``s`` has shape
    ``(n_papers, n_components)``.

    This is a co-activation proxy for true resting-state FC and requires
    no external data download.

    Parameters
    ----------
    difumo_coeffs:
        DiFuMo coefficient matrix, shape ``(n_papers, n_components)``.

    Returns
    -------
    fc : np.ndarray, shape (n_components, n_components)
        Symmetric correlation matrix with diagonal 1.
    """
    # Center each component across papers
    X = difumo_coeffs - difumo_coeffs.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0, keepdims=True) + 1e-8
    X_norm = X / norms
    fc = (X_norm.T @ X_norm) / X.shape[0]   # (K, K)
    np.fill_diagonal(fc, 0.0)               # remove self-loops
    return fc.astype(np.float32)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def threshold_fc_matrix(
    fc: np.ndarray,
    percentile: float = 90.0,
    positive_only: bool = True,
) -> np.ndarray:
    """Sparsify FC matrix by keeping only the top edges.

    Parameters
    ----------
    fc:
        Square FC matrix, shape ``(K, K)``.
    percentile:
        Keep edges whose absolute FC value exceeds this percentile of all
        absolute values.  Default 90 keeps the top 10 % of connections.
        Aim for average degree between 5 and 80; adjust percentile if needed.
    positive_only:
        If *True* (default), set negative entries to zero before thresholding
        so only excitatory functional connections form edges.

    Returns
    -------
    adj : np.ndarray, shape (K, K)
        Thresholded adjacency matrix with zeros below the threshold.
    """
    adj = fc.copy()
    if positive_only:
        adj = np.clip(adj, 0, None)

    threshold = np.percentile(np.abs(adj), percentile)
    adj[np.abs(adj) < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def adjacency_to_pyg(adj: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense adjacency matrix to PyG edge tensors.

    Parameters
    ----------
    adj:
        Dense adjacency matrix, shape ``(K, K)``.

    Returns
    -------
    edge_index : torch.LongTensor, shape (2, E)
        Source and destination node indices.
    edge_attr : torch.FloatTensor, shape (E, 1)
        FC weight for each edge.
    """
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(
        np.stack([rows, cols], axis=0), dtype=torch.long
    )
    weights = adj[rows, cols]
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def sanity_check_graph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    n_nodes: int = 512,
) -> None:
    """Print graph statistics and warn if degree is out of range.

    Checks: number of nodes, number of edges, average degree, connectivity.
    If average degree < 5 or > 80, prints a recommendation to adjust the
    threshold percentile.
    """
    n_edges = edge_index.shape[1]
    avg_degree = n_edges / n_nodes
    print(f"  Nodes        : {n_nodes}")
    print(f"  Edges        : {n_edges}")
    print(f"  Avg degree   : {avg_degree:.1f}")
    print(f"  Edge weight  : min={edge_attr.min():.3f}  max={edge_attr.max():.3f}")

    if avg_degree < 5:
        print(
            "  WARNING: avg degree < 5 — graph is too sparse. "
            "Lower the threshold percentile (e.g. 80)."
        )
    elif avg_degree > 80:
        print(
            "  WARNING: avg degree > 80 — graph is too dense. "
            "Raise the threshold percentile (e.g. 95)."
        )
    else:
        print("  Degree OK.")

    # Connectivity check via BFS over undirected edge set
    adj_dict: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj_dict[s].append(d)
        adj_dict[d].append(s)

    visited = set()
    queue = [0]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj_dict[node])

    n_components = n_nodes - len(visited)
    if n_components == 0:
        print("  Graph is connected.")
    else:
        print(f"  WARNING: {n_components} nodes unreachable — graph is disconnected.")


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def build_brain_graph(
    fc_path: Optional[str | Path] = None,
    difumo_coeffs: Optional[np.ndarray] = None,
    percentile: float = 90.0,
    positive_only: bool = True,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the DiFuMo brain graph as PyG tensors.

    Exactly one of *fc_path* or *difumo_coeffs* must be provided.

    Parameters
    ----------
    fc_path:
        Path to a precomputed ``.npy`` / ``.npz`` FC matrix ``(512, 512)``.
        Takes priority over *difumo_coeffs*.
    difumo_coeffs:
        DiFuMo coefficient matrix ``(n_papers, 512)`` used to compute a
        data-derived correlation FC when no external file is available.
    percentile:
        Threshold percentile (default 90 → top 10 % of edges kept).
    positive_only:
        Only retain positive FC edges (default *True*).
    verbose:
        Print graph statistics (default *True*).

    Returns
    -------
    edge_index : torch.LongTensor, shape (2, E)
    edge_attr  : torch.FloatTensor, shape (E, 1)
    """
    if fc_path is not None:
        if verbose:
            print(f"Loading FC matrix from {fc_path}")
        fc = load_fc_matrix(fc_path)
    elif difumo_coeffs is not None:
        if verbose:
            print("Computing FC matrix from DiFuMo coefficients (co-activation proxy)...")
        fc = compute_fc_from_coefficients(difumo_coeffs)
    else:
        raise ValueError("Provide either fc_path or difumo_coeffs.")

    adj = threshold_fc_matrix(fc, percentile=percentile, positive_only=positive_only)
    edge_index, edge_attr = adjacency_to_pyg(adj)

    if verbose:
        sanity_check_graph(edge_index, edge_attr)

    return edge_index, edge_attr
