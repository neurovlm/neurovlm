"""Evaluate generated maps or baselines against true maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from atlas_free_cnn.evaluation.generation_metrics import generation_metrics


def evaluate_prediction_tensor(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return generation_metrics(pred.float(), target.float())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="Torch tensor [B,1,X,Y,Z] predictions")
    p.add_argument("--target", required=True, help="Torch tensor [B,1,X,Y,Z] targets")
    p.add_argument("--output", default="experiments/3dcnn/atlas_free_cnn/outputs/eval/generation_metrics.json")
    args = p.parse_args()
    pred = torch.load(args.pred, map_location="cpu", weights_only=False)
    target = torch.load(args.target, map_location="cpu", weights_only=False)
    metrics = evaluate_prediction_tensor(pred, target)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

