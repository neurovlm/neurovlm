"""Semantic retrieval diagnostics for generated maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from atlas_free_multipositive.evaluation.metrics import ranking_metrics


@torch.no_grad()
def generated_map_retrieval_metrics(
    generated_brain_embeddings: torch.Tensor,
    candidate_text_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, float]:
    scores = F.normalize(generated_brain_embeddings.float(), dim=1) @ F.normalize(candidate_text_embeddings.float(), dim=1).T
    return ranking_metrics(scores, positive_mask.bool(), ks=(1, 5, 10, 50))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated-brain-embeddings", required=True)
    p.add_argument("--candidate-text-embeddings", required=True)
    p.add_argument("--positive-mask", required=True)
    p.add_argument("--output", default="atlas_free_multipositive/outputs/eval/generated_semantic_retrieval.json")
    args = p.parse_args()
    metrics = generated_map_retrieval_metrics(
        torch.load(args.generated_brain_embeddings, map_location="cpu", weights_only=False),
        torch.load(args.candidate_text_embeddings, map_location="cpu", weights_only=False),
        torch.load(args.positive_mask, map_location="cpu", weights_only=False),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(args.output, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

