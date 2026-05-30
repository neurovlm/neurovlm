"""Source-aware sampling helpers for mixed atlas-free AE training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import WeightedRandomSampler


def canonical_source(row: dict[str, Any]) -> str:
    source = str(row.get("source", "")).lower()
    if source == "pubmed" or row.get("pmid"):
        return "pubmed"
    if source.startswith("neurovault"):
        return "neurovault"
    if source.startswith("nilearn"):
        return "nilearn"
    return source or "unknown"


def source_detail(row: dict[str, Any]) -> str:
    return str(row.get("source_detail") or row.get("source") or canonical_source(row))


@dataclass
class SourceSamplingConfig:
    mode: str = "natural"
    alpha: float = 0.5
    manual_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SourceSamplingConfig":
        raw_weights = (
            cfg.get("source_sampling_weights")
            or cfg.get("SOURCE_SAMPLING_WEIGHTS")
            or {}
        )
        return cls(
            mode=str(cfg.get("source_sampling", cfg.get("SOURCE_SAMPLING", "natural"))).lower(),
            alpha=float(cfg.get("source_sampling_alpha", cfg.get("SOURCE_SAMPLING_ALPHA", 0.5))),
            manual_weights={str(k).lower(): float(v) for k, v in dict(raw_weights).items()},
        )


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(canonical_source(row) for row in rows))


def source_probabilities(
    counts: dict[str, int],
    cfg: SourceSamplingConfig,
) -> dict[str, float]:
    if not counts:
        return {}
    mode = cfg.mode
    if mode == "natural":
        denom = float(sum(counts.values()))
        return {src: float(n) / denom for src, n in counts.items()}
    if mode == "balanced":
        return {src: 1.0 / float(len(counts)) for src in counts}
    if mode == "temperature":
        denom = sum(float(n) ** float(cfg.alpha) for n in counts.values())
        return {src: (float(n) ** float(cfg.alpha)) / denom for src, n in counts.items()}
    if mode == "manual":
        missing = sorted(set(counts) - set(cfg.manual_weights))
        if missing:
            raise ValueError(f"manual source_sampling_weights missing sources: {missing}")
        total = sum(max(0.0, cfg.manual_weights[src]) for src in counts)
        if total <= 0:
            raise ValueError("manual source_sampling_weights must sum to a positive value")
        return {src: max(0.0, cfg.manual_weights[src]) / total for src in counts}
    raise ValueError("source_sampling must be natural, balanced, temperature, or manual")


def build_source_sampler(rows: list[dict[str, Any]], cfg: dict[str, Any]):
    sampling = SourceSamplingConfig.from_config(cfg)
    counts = source_counts(rows)
    probs = source_probabilities(counts, sampling)
    if sampling.mode == "natural":
        return None, sampler_report(rows, sampling, counts, probs)
    weights = [probs[canonical_source(row)] / float(counts[canonical_source(row)]) for row in rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, sampler_report(rows, sampling, counts, probs)


def sampler_report(
    rows: list[dict[str, Any]],
    cfg: SourceSamplingConfig,
    counts: dict[str, int] | None = None,
    probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    counts = counts or source_counts(rows)
    probs = probs or source_probabilities(counts, cfg)
    n = len(rows)
    return {
        "source_sampling": cfg.mode,
        "source_sampling_alpha": cfg.alpha,
        "source_sampling_weights": cfg.manual_weights,
        "dataset_source_counts": counts,
        "effective_source_probabilities": probs,
        "expected_source_exposures_per_epoch": {
            src: float(prob) * float(n) for src, prob in probs.items()
        },
    }


def epoch_source_exposure(epoch: int, observed_counts: dict[str, int], report: dict[str, Any]) -> dict[str, Any]:
    total = max(1, sum(int(v) for v in observed_counts.values()))
    row = {
        "epoch": int(epoch),
        "source_sampling": report.get("source_sampling", "natural"),
        "total_examples_seen": int(total),
    }
    all_sources = sorted(set(observed_counts) | set(report.get("expected_source_exposures_per_epoch", {})))
    for src in all_sources:
        obs = int(observed_counts.get(src, 0))
        row[f"{src}_observed"] = obs
        row[f"{src}_observed_fraction"] = float(obs) / float(total)
        row[f"{src}_expected"] = float(report.get("expected_source_exposures_per_epoch", {}).get(src, 0.0))
    return row
