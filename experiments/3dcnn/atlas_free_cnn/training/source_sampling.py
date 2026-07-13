"""Natural source-sampling reports for mixed atlas-free AE training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


def canonical_source(row: dict[str, Any]) -> str:
    source = str(row.get("source", "")).lower()
    if source == "pubmed" or row.get("pmid"):
        return "pubmed"
    if source.startswith("neurovault"):
        return "neurovault"
    if source.startswith("nilearn"):
        return "nilearn"
    if source.startswith("network"):
        return "networks"
    return source or "unknown"


def source_detail(row: dict[str, Any]) -> str:
    return str(row.get("source_detail") or row.get("source") or canonical_source(row))


@dataclass
class SourceSamplingConfig:
    mode: str = "natural"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SourceSamplingConfig":
        mode = str(cfg.get("source_sampling", cfg.get("SOURCE_SAMPLING", "natural"))).lower()
        if mode != "natural":
            raise ValueError("The retained Stage 1 recipe requires source_sampling='natural'")
        return cls(mode=mode)


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(canonical_source(row) for row in rows))


def source_probabilities(
    counts: dict[str, int],
    cfg: SourceSamplingConfig,
) -> dict[str, float]:
    if not counts:
        return {}
    denom = float(sum(counts.values()))
    return {src: float(n) / denom for src, n in counts.items()}


def build_source_sampler(rows: list[dict[str, Any]], cfg: dict[str, Any]):
    sampling = SourceSamplingConfig.from_config(cfg)
    counts = source_counts(rows)
    probs = source_probabilities(counts, sampling)
    return None, sampler_report(rows, sampling, counts, probs)


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
