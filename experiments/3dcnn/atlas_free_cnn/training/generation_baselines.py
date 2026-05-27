"""Generation baselines for sparse ALE text-to-brain experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def global_mean_map(train_volumes: torch.Tensor) -> torch.Tensor:
    return train_volumes.float().mean(dim=0, keepdim=False)


@torch.no_grad()
def random_training_maps(train_volumes: torch.Tensor, n: int, *, seed: int = 0) -> torch.Tensor:
    rng = random.Random(seed)
    idx = [rng.randrange(len(train_volumes)) for _ in range(n)]
    return train_volumes[idx].float()


@torch.no_grad()
def nearest_neighbor_text_maps(
    query_text_embeddings: torch.Tensor,
    train_text_embeddings: torch.Tensor,
    train_volumes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = F.normalize(query_text_embeddings.float(), dim=1, eps=1e-8)
    train = F.normalize(train_text_embeddings.float(), dim=1, eps=1e-8)
    sim = query @ train.T
    idx = sim.argmax(dim=1)
    return train_volumes[idx].float(), idx


@torch.no_grad()
def category_average_maps(rows: Iterable[dict], volumes: torch.Tensor, *, key: str = "positive_terms") -> dict[str, torch.Tensor]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for value in row.get(key, []) or []:
            buckets[str(value).lower()].append(i)
    return {term: volumes[idx].float().mean(dim=0) for term, idx in buckets.items() if idx}


@torch.no_grad()
def predict_category_average(query_terms: Iterable[str], averages: dict[str, torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    hits = [averages[t.lower()] for t in query_terms if t.lower() in averages]
    if not hits:
        return fallback.float()
    return torch.stack(hits).mean(dim=0)

