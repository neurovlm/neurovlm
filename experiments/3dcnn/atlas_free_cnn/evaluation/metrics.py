"""Ranking metrics used by atlas/MeSH retrieval evaluations."""

from __future__ import annotations

import torch


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

