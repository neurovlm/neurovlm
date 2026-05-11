"""Evaluate MeSH term ranking for unified PubMed examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from atlas_free_multipositive.evaluation.metrics import ranking_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="Torch tensor [B, C] of brain-to-candidate scores")
    p.add_argument("--positive-mask", required=True, help="Torch bool tensor [B, C]")
    p.add_argument("--output", default="atlas_free_multipositive/outputs/eval/mesh_ranking_metrics.json")
    args = p.parse_args()
    scores = torch.load(args.scores, map_location="cpu", weights_only=False)
    pos = torch.load(args.positive_mask, map_location="cpu", weights_only=False)
    metrics = ranking_metrics(scores, pos)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

