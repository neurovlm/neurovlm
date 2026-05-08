"""Diagnostics for coordinate-graph retrieval experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from neurovlm.metrics import retrieval_ranks


def coord_graph_covariates(dataset) -> pd.DataFrame:
    """Compute simple per-paper graph/coordinate covariates.

    These features help test whether embeddings mostly encode peak count,
    centroid, spread, or graph density rather than semantic neurobiology.
    """
    rows = []
    for pos in range(len(dataset)):
        graph = dataset[pos]
        xyz = graph.x[:, :3].float()
        n = int(xyz.size(0))
        edge_index = graph.edge_index
        nonself = edge_index[0].ne(edge_index[1])
        n_edges = int(nonself.sum().item())
        density = n_edges / float(max(1, n * (n - 1)))
        centroid = xyz.mean(dim=0)
        spread = xyz.std(dim=0, unbiased=False) if n > 1 else torch.zeros(3)
        radius = xyz.norm(dim=1)

        rows.append(
            {
                "sample_pos": pos,
                "paper_idx": int(graph.paper_idx),
                "n_peaks": n,
                "n_edges": n_edges,
                "edge_density": density,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "spread_x": float(spread[0]),
                "spread_y": float(spread[1]),
                "spread_z": float(spread[2]),
                "radius_mean": float(radius.mean()),
                "radius_max": float(radius.max()),
            }
        )
    return pd.DataFrame(rows)


def subset_pmids(dataset) -> np.ndarray:
    """Return PMIDs in the exact order used by a coordinate dataset/subset."""
    if hasattr(dataset, "_parent") and hasattr(dataset, "_positions"):
        parent = dataset._parent
        dataset_indices = [parent._valid_indices[pos] for pos in dataset._positions]
        return np.asarray(parent.unique_pmids)[dataset_indices].astype(str)
    if hasattr(dataset, "_valid_indices") and hasattr(dataset, "unique_pmids"):
        return np.asarray(dataset.unique_pmids)[dataset._valid_indices].astype(str)
    raise TypeError("Expected CoordGraphDataset or _SubsetCoordDataset")


@torch.no_grad()
def evaluate_pretrained_neurovlm_baseline(
    dataset,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate pretrained NeuroVLM projectors on a coordinate split.

    Use this when comparing to the historical ``~0.81`` baseline. It reports
    the same retrieval metrics used for CoordGNN and restricts evaluation to
    the PMIDs in ``dataset``.
    """
    from neurovlm.metrics import bidirectional_retrieval_metrics
    from neurovlm.models import ProjHead
    from neurovlm.retrieval_resources import _load_latent_neuro, _load_latent_text

    wanted_pmids = subset_pmids(dataset)
    neuro_latent, neuro_pmids = _load_latent_neuro()
    text_latent, text_pmids = _load_latent_text()

    neuro_lookup = {str(p): i for i, p in enumerate(neuro_pmids)}
    text_lookup = {str(p): i for i, p in enumerate(text_pmids)}
    keep = [
        (neuro_lookup[p], text_lookup[p])
        for p in wanted_pmids
        if p in neuro_lookup and p in text_lookup
    ]
    if not keep:
        raise RuntimeError("No overlap between CoordGNN split and NeuroVLM baseline PMIDs")

    dev = torch.device(device if device in {"cuda", "cpu", "mps"} else "cpu")
    image_proj = ProjHead.from_pretrained("image_infonce").to(dev).eval()
    text_proj = ProjHead.from_pretrained("text_infonce").to(dev).eval()

    neuro_idx, text_idx = zip(*keep)
    brain = neuro_latent[list(neuro_idx)].float().to(dev)
    text = text_latent[list(text_idx)].float().to(dev)
    brain_emb = image_proj(brain).float()
    text_emb = text_proj(text).float()

    metrics = bidirectional_retrieval_metrics(text_emb, brain_emb)
    metrics["n_eval"] = float(len(keep))
    metrics["n_requested"] = float(len(wanted_pmids))
    return metrics


def retrieval_diagnostics(
    text_emb: torch.Tensor,
    brain_emb: torch.Tensor,
    covariates: pd.DataFrame | None = None,
    k: int = 10,
) -> pd.DataFrame:
    """Return per-sample retrieval ranks and optional covariates."""
    text_n = F.normalize(text_emb.float(), dim=1, eps=1e-8)
    brain_n = F.normalize(brain_emb.float(), dim=1, eps=1e-8)
    sim = text_n @ brain_n.T
    ranks = retrieval_ranks(sim).cpu().numpy()
    top1 = sim.argmax(dim=1).cpu().numpy()
    true_scores = sim.diag().cpu().numpy()
    top1_scores = sim.max(dim=1).values.cpu().numpy()

    df = pd.DataFrame(
        {
            "sample_pos": np.arange(len(ranks)),
            "rank": ranks,
            f"hit@{k}": ranks <= k,
            "top1_pos": top1,
            "true_score": true_scores,
            "top1_score": top1_scores,
            "score_margin": true_scores - top1_scores,
        }
    )
    if covariates is not None:
        df = df.merge(covariates, on="sample_pos", how="left")
    return df


def embedding_covariate_correlations(
    embeddings: torch.Tensor,
    covariates: pd.DataFrame,
    n_components: int = 8,
) -> pd.DataFrame:
    """Correlate embedding PCs with coordinate/graph covariates."""
    emb = F.normalize(embeddings.float(), dim=1, eps=1e-8).cpu()
    emb = emb - emb.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(emb, q=min(n_components, emb.shape[1]))
    pcs = (emb @ v[:, :n_components]).numpy()

    numeric = covariates.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=["sample_pos", "paper_idx"], errors="ignore")
    rows = []
    for pc_idx in range(pcs.shape[1]):
        pc = pcs[:, pc_idx]
        pc_std = pc.std()
        if pc_std == 0:
            continue
        for name in numeric.columns:
            values = numeric[name].to_numpy(dtype=float)
            mask = np.isfinite(values)
            if mask.sum() < 3 or values[mask].std() == 0:
                continue
            corr = np.corrcoef(pc[mask], values[mask])[0, 1]
            rows.append(
                {
                    "pc": pc_idx + 1,
                    "covariate": name,
                    "pearson_r": float(corr),
                    "abs_pearson_r": float(abs(corr)),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
    return out
