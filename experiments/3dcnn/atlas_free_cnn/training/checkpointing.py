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


def combined_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_COMBINED_SCORE_WEIGHTS
    return float(sum(float(metrics.get(k, 0.0)) * float(w) for k, w in weights.items()))


class CheckpointManager:
    """Save last and best checkpoints by named metric."""

    def __init__(self, out_dir: str | Path, *, maximize: dict[str, bool] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize or {}
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
        return self.save("last.pt", payload)

    def maybe_save_best(self, metric_name: str, metric_value: float, payload: dict[str, Any], *, epoch: int | None = None) -> bool:
        maximize = self.maximize.get(metric_name, True)
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
            maximize = self.maximize.get(metric_name, True)
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

