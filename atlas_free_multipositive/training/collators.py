"""Collators for multi-positive contrastive batches."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

import torch


def _sample_positives(positives: list[dict], k: int, source: str, rng: random.Random) -> list[dict]:
    if len(positives) <= k:
        return list(positives)
    mesh = [p for p in positives if p.get("source") == "mesh"]
    summaries = [p for p in positives if p.get("category") == "paper_summary"]
    titles = [p for p in positives if p.get("category") == "paper_title"]
    atlas = [p for p in positives if str(p.get("source", "")).startswith("nilearn")]
    chosen: list[dict] = []
    if "pubmed" in source:
        rng.shuffle(mesh)
        chosen.extend(mesh[: max(1, min(3, k, len(mesh)))])
        if summaries and len(chosen) < k and rng.random() < 0.5:
            chosen.append(rng.choice(summaries))
        if titles and len(chosen) < k and rng.random() < 0.25:
            chosen.append(rng.choice(titles))
    else:
        chosen.extend(atlas[:1] or positives[:1])
    remaining = [p for p in positives if id(p) not in {id(c) for c in chosen}]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, k - len(chosen))])
    return chosen[:k]


class MultiPositiveCollator:
    """Sample K positives per map and construct positive masks/weights."""

    def __init__(
        self,
        positives_per_map: int = 3,
        *,
        tokenizer: Callable[[list[str]], dict] | None = None,
        seed: int = 0,
        target_shape: tuple[int, int, int] | None = None,
    ):
        self.positives_per_map = positives_per_map
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.target_shape = target_shape

    def __call__(self, batch: list[dict]) -> dict:
        volume_list = [b["volume"].float() for b in batch]
        if self.target_shape is not None:
            import torch.nn.functional as F

            volume_list = [
                F.interpolate(
                    v.unsqueeze(0),
                    size=self.target_shape,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
                for v in volume_list
            ]
        volumes = torch.stack(volume_list)
        texts: list[str] = []
        text_entries: list[dict] = []
        owners: list[int] = []
        weights: list[float] = []
        for i, item in enumerate(batch):
            source = str(item.get("metadata", {}).get("source", ""))
            positives = _sample_positives(item["positive_texts"], self.positives_per_map, source, self.rng)
            for pos in positives:
                texts.append(pos["text"])
                text_entries.append(pos)
                owners.append(i)
                weights.append(float(pos.get("weight", 1.0)))

        bsz = len(batch)
        t = len(texts)
        pos_mask = torch.zeros((bsz, t), dtype=torch.bool)
        pos_weights = torch.zeros((bsz, t), dtype=torch.float32)
        for j, i in enumerate(owners):
            pos_mask[i, j] = True
            pos_weights[i, j] = weights[j]

        out = {
            "volume": volumes,
            "map_id": [b["map_id"] for b in batch],
            "texts": texts,
            "text_entries": text_entries,
            "pos_mask": pos_mask,
            "pos_weights": pos_weights,
            "metadata": [b["metadata"] for b in batch],
        }
        if self.tokenizer is not None:
            out["tokens"] = self.tokenizer(texts)
        return out


class BalancedSourceBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Simple sampler that tries to include atlas and PubMed examples per batch."""

    def __init__(self, dataset, batch_size: int, *, atlas_fraction: float = 0.25, pubmed_fraction: float = 0.25, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.atlas_n = max(1, int(round(batch_size * atlas_fraction)))
        self.pubmed_n = max(1, int(round(batch_size * pubmed_fraction)))
        self.seed = seed
        groups = defaultdict(list)
        for i, row in enumerate(dataset.rows):
            groups["pubmed" if row.get("pmid") else "atlas"].append(i)
        self.groups = dict(groups)

    def __iter__(self):
        rng = random.Random(self.seed)
        atlas = self.groups.get("atlas", [])[:]
        pubmed = self.groups.get("pubmed", [])[:]
        other = list(set(range(len(self.dataset))) - set(atlas) - set(pubmed))
        rng.shuffle(atlas)
        rng.shuffle(pubmed)
        rng.shuffle(other)
        while atlas or pubmed or other:
            batch = []
            for pool, n in ((atlas, self.atlas_n), (pubmed, self.pubmed_n)):
                for _ in range(min(n, len(pool))):
                    batch.append(pool.pop())
            fill = [*atlas, *pubmed, *other]
            rng.shuffle(fill)
            for idx in fill:
                if len(batch) >= self.batch_size:
                    break
                if idx not in batch:
                    batch.append(idx)
            used = set(batch)
            atlas = [i for i in atlas if i not in used]
            pubmed = [i for i in pubmed if i not in used]
            other = [i for i in other if i not in used]
            if batch:
                yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
