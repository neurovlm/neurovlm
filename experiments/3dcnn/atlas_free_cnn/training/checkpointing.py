"""Checkpoint helpers with configurable multi-metric model selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_COMBINED_SCORE_WEIGHTS = {
    "mesh_recall_at_10": 0.30,
    "semantic_recall_at_50": 0.20,
    "network_mrr": 0.15,
    "generation_top5_dice": 0.20,
    "generation_semantic_mesh_recall_at_10": 0.15,
}

MINIMIZE_METRICS = {
    "val_loss",
    "loss",
    "mse",
    "reconstruction_mse",
    "foreground_mse",
    "mae",
    "latent_mse",
    "val_latent_mse",
    "val_reconstruction_mse",
}

MAXIMIZE_METRICS = {
    "spatial_corr",
    "top1_dice",
    "top5_dice",
    "top10_dice",
    "top1_overlap",
    "top5_overlap",
    "top10_overlap",
    "generation_normalized_auc",
    "generation_mean_normalized_auc",
    "val_generation_normalized_auc",
    "val_spatial_corr",
    "val_top5_dice",
    "generation_top5_dice",
    "generation_spatial_correlation",
}


def canonical_metric_name(metric_name: str) -> str:
    metric = str(metric_name).strip()
    if metric.endswith(".pt"):
        metric = metric[:-3]
    if metric.startswith("best_"):
        metric = metric.removeprefix("best_")
    return metric


def metric_direction(metric_name: str) -> str:
    metric = canonical_metric_name(metric_name)
    if metric in MAXIMIZE_METRICS:
        return "maximize"
    if metric in MINIMIZE_METRICS:
        return "minimize"
    if metric.startswith("val_"):
        base = metric.removeprefix("val_")
        if base in MAXIMIZE_METRICS:
            return "maximize"
        if base in MINIMIZE_METRICS:
            return "minimize"
    raise ValueError(f"Unknown checkpoint/selection metric direction for {metric_name!r}")


def metric_higher_is_better(metric_name: str) -> bool:
    return metric_direction(metric_name) == "maximize"


def combined_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_COMBINED_SCORE_WEIGHTS
    return float(sum(float(metrics.get(k, 0.0)) * float(w) for k, w in weights.items()))


class CheckpointManager:
    """Save last and best checkpoints by named metric."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        maximize: dict[str, bool] | None = None,
        require_explicit_direction: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize or {}
        self.require_explicit_direction = bool(require_explicit_direction)
        self.best: dict[str, float] = {}
        self._best_epochs: dict[str, int] = {}
        self._last_epoch: int | None = None

    def save(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.out_dir / name
        torch.save(payload, path)
        return path

    def save_last(self, payload: dict[str, Any], *, epoch: int | None = None) -> Path:
        if epoch is not None:
            self._last_epoch = epoch
        path = self.save("last.pt", payload)
        self.write_manifest()
        return path

    def _maximize_for(self, metric_name: str) -> bool:
        if metric_name in self.maximize:
            return bool(self.maximize[metric_name])
        if self.require_explicit_direction:
            raise ValueError(f"Missing explicit checkpoint direction for {metric_name!r}")
        return True

    def maybe_save_best(self, metric_name: str, metric_value: float, payload: dict[str, Any], *, epoch: int | None = None) -> bool:
        maximize = self._maximize_for(metric_name)
        old = self.best.get(metric_name)
        is_better = old is None or (metric_value > old if maximize else metric_value < old)
        if is_better:
            self.best[metric_name] = float(metric_value)
            if epoch is not None:
                self._best_epochs[metric_name] = epoch
            self.save(f"best_{metric_name}.pt", payload)
            self.write_manifest()
        return is_better

    def write_manifest(self) -> None:
        rows = []
        for metric_name, value in sorted(self.best.items()):
            maximize = self._maximize_for(metric_name)
            path = self.out_dir / f"best_{metric_name}.pt"
            rows.append({
                "checkpoint_name": f"best_{metric_name}.pt",
                "metric_name": metric_name,
                "metric_value": value,
                "epoch": self._best_epochs.get(metric_name),
                "path": str(path),
                "selection_direction": "maximize" if maximize else "minimize",
                "exists": path.exists(),
            })
        last_path = self.out_dir / "last.pt"
        rows.append({
            "checkpoint_name": "last.pt",
            "metric_name": "last_epoch",
            "metric_value": None,
            "epoch": self._last_epoch,
            "path": str(last_path),
            "selection_direction": "n/a",
            "exists": last_path.exists(),
        })
        with (self.out_dir / "checkpoint_manifest.json").open("w") as f:
            json.dump({"checkpoints": rows, "best": self.best, "maximize": self.maximize}, f, indent=2)
