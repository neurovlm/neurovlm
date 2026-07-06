"""Modality-agnostic retrieval and ranking metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def normalized_k_values(n: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return normalized full-curve k values ``[1/n, ..., n/n]``."""

    if n < 1:
        raise ValueError("n must be >= 1")
    return torch.arange(1, n + 1, device=device).float() / float(n)


@torch.no_grad()
def normalized_recall_curve_auc(curve: torch.Tensor) -> float:
    """Area under recall(k) vs normalized k = k/n.

    Recall curves are saved at every integer k, so the normalized-k samples are
    uniformly spaced by 1/n. The right-endpoint Riemann area is therefore the
    mean of the full recall curve. This is the paper-style AUC, not recall@K.
    """

    if curve.numel() == 0:
        raise ValueError("curve must contain at least one point")
    return float(curve.float().mean().item())


def recall_at_k(cos_sim: torch.Tensor, k: int) -> float:
    """
    Parameters
    ----------
    cos_sim : 2d tensor
        Cosine similarity matrix.
    k : int
        Top k most similar items to consider.

    Returns
    -------
    recall : float
        Recall @ k.
    """
    ranks = cos_sim.argsort(dim=1, descending=True)
    correct = torch.arange(ranks.size(0), device=ranks.device)
    hit = (ranks[:, :k] == correct[:, None]).any(dim=1)
    return hit.float().mean().item()


@torch.no_grad()
def retrieval_ranks(cos_sim: torch.Tensor) -> torch.Tensor:
    """Return 1-indexed retrieval ranks for diagonal matches.

    ``cos_sim[i, j]`` is the similarity between query ``i`` and target ``j``;
    the correct target is assumed to be on the diagonal.
    """
    ranks = cos_sim.argsort(dim=1, descending=True)
    correct = torch.arange(ranks.size(0), device=ranks.device)
    pos = ranks.eq(correct[:, None]).to(torch.int32).argmax(dim=1)
    return pos + 1


@torch.no_grad()
def retrieval_metrics(
    latent_query: torch.Tensor,
    latent_target: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """Compute retrieval metrics for aligned query/target embeddings."""
    q = F.normalize(latent_query.float(), dim=1, eps=1e-8)
    t = F.normalize(latent_target.float(), dim=1, eps=1e-8)
    sim = q @ t.T
    ranks = retrieval_ranks(sim).float()
    n = float(sim.size(0))

    out: dict[str, float] = {
        "median_rank": float(ranks.median().item()),
        "mean_rank": float(ranks.mean().item()),
        "mrr": float((1.0 / ranks).mean().item()),
    }
    for k in ks:
        out[f"recall@{k}"] = float((ranks <= k).float().mean().item())
        out[f"random_recall@{k}"] = min(float(k) / n, 1.0)

    curve, _ = recall_curve(q, t)
    auc = normalized_recall_curve_auc(curve)
    out["auc"] = auc
    out["paper_recall_curve_auc"] = auc
    out["normalized_k_recall_curve_auc"] = auc
    return out


@torch.no_grad()
def bidirectional_retrieval_metrics(
    latent_text: torch.Tensor,
    latent_image: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """Compute text-to-image, image-to-text, and averaged retrieval metrics."""
    t2i = retrieval_metrics(latent_text, latent_image, ks=ks)
    i2t = retrieval_metrics(latent_image, latent_text, ks=ks)
    out = {f"t2i_{k}": v for k, v in t2i.items()}
    out.update({f"i2t_{k}": v for k, v in i2t.items()})
    for key in t2i:
        out[f"mean_{key}"] = (t2i[key] + i2t[key]) / 2.0
    return out


@torch.no_grad()
def recall_curve(latent_text: torch.Tensor,
                 latent_image: torch.Tensor,
                 step: int = 1) -> tuple[torch.Tensor, torch.Tensor]:

    t = F.normalize(latent_text, dim=1, eps=1e-8)
    v = F.normalize(latent_image, dim=1, eps=1e-8)

    n = t.shape[0]
    device = t.device
    ks = torch.arange(0, n, step, device=device) + 1

    def _curve_from_sim(sim: torch.Tensor) -> torch.Tensor:
        ranks = sim.argsort(dim=1, descending=True)
        correct = torch.arange(n, device=sim.device)
        pos = ranks.eq(correct[:, None]).to(torch.int32).argmax(dim=1)  # FIX

        counts = torch.bincount(pos, minlength=n)
        recall_all = counts.cumsum(0).float() / float(n)
        return recall_all.index_select(0, ks - 1)

    sim = t @ v.T
    t_to_i = _curve_from_sim(sim)
    i_to_t = _curve_from_sim(sim.T)
    return t_to_i, i_to_t


__all__ = [
    "bidirectional_retrieval_metrics",
    "normalized_k_values",
    "normalized_recall_curve_auc",
    "recall_at_k",
    "recall_curve",
    "retrieval_metrics",
    "retrieval_ranks",
]
