"""Ranking metrics used by atlas/MeSH retrieval evaluations."""

from __future__ import annotations

import torch


@torch.no_grad()
def normalized_recall_curve_auc_from_ranks(ranks: torch.Tensor, n_candidates: int | None = None) -> tuple[float, torch.Tensor]:
    """Full recall@k AUC over normalized rank ``k / N``.

    ``ranks`` are 1-indexed true-pair ranks. The returned curve has one point
    for every integer k from 1 through N. With random rankings the expected AUC
    is approximately 0.5; with perfect rankings it is 1.0.
    """

    ranks = ranks.float()
    valid = torch.isfinite(ranks)
    if not bool(valid.any()):
        raise ValueError("at least one finite rank is required")
    if n_candidates is None:
        n_candidates = int(torch.nan_to_num(ranks[valid], posinf=0).max().item())
    ks = torch.arange(1, int(n_candidates) + 1, device=ranks.device).float()
    curve = (ranks[valid, None] <= ks[None, :]).float().mean(dim=0)
    return float(curve.mean().item()), curve.cpu()


def ranks_from_scores(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, dim=1, descending=True)
    sorted_pos = torch.gather(positive_mask.bool(), 1, order)
    has_pos = sorted_pos.any(dim=1)
    first = torch.argmax(sorted_pos.float(), dim=1) + 1
    first = first.to(torch.float32)
    first[~has_pos] = torch.inf
    return first


def ranking_metrics(scores: torch.Tensor, positive_mask: torch.Tensor, ks=(1, 5, 10, 50)) -> dict[str, float]:
    ranks = ranks_from_scores(scores.float(), positive_mask.bool())
    valid = torch.isfinite(ranks)
    out: dict[str, float] = {"n": float(valid.sum().item())}
    if not bool(valid.any()):
        return out
    vr = ranks[valid]
    auc, _ = normalized_recall_curve_auc_from_ranks(vr, n_candidates=scores.shape[1])
    out["normalized_recall_curve_auc"] = auc
    out["paper_recall_curve_auc"] = auc
    out["mrr"] = float((1.0 / vr).mean().item())
    out["median_best_positive_rank"] = float(vr.median().item())
    for k in ks:
        out[f"recall@{k}"] = float((vr <= k).float().mean().item())
    return out


def average_precision(scores: torch.Tensor, positive_mask: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    pos = positive_mask[order].bool()
    if not bool(pos.any()):
        return 0.0
    precision = torch.cumsum(pos.float(), dim=0) / torch.arange(1, len(pos) + 1, device=pos.device)
    return float(precision[pos].mean().item())
