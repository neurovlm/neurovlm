"""Tiny smoke-test trainer for the multi-positive contrastive pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_multipositive.training.collators import MultiPositiveCollator
from atlas_free_multipositive.training.datasets import UnifiedMapTextDataset
from atlas_free_multipositive.training.losses import multi_positive_infonce
from atlas_free_multipositive.training.model_wrappers import (
    build_brain_encoder,
    build_text_projection,
    load_text_projection_checkpoint,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="atlas_free_multipositive/cache/unified_jsonl/splits/train.jsonl")
    p.add_argument("--text-embeddings", default=None, help="Optional torch dict text->768d embedding cache")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--text-proj-init", choices=["random", "pretrained_text_infonce"], default="random")
    p.add_argument("--text-proj-checkpoint", default=None, help="Optional Stage 2 text-to-brain projection checkpoint.")
    args = p.parse_args()

    ds = UnifiedMapTextDataset(args.jsonl)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=MultiPositiveCollator(positives_per_map=2))
    brain = build_brain_encoder().to(args.device)
    text_proj = build_text_projection(args.text_proj_init, device=args.device)
    if args.text_proj_checkpoint:
        load_text_projection_checkpoint(text_proj, args.text_proj_checkpoint)
    opt = torch.optim.AdamW([*brain.parameters(), *text_proj.parameters()], lr=1e-4)

    emb_cache = torch.load(args.text_embeddings, map_location="cpu", weights_only=False) if args.text_embeddings else {}
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        volume = batch["volume"].to(args.device)
        missing = [t for t in batch["texts"] if t not in emb_cache]
        if missing:
            raise RuntimeError("Smoke trainer needs --text-embeddings for all sampled texts; run notebook 05 or use a tiny mocked cache.")
        raw_text = torch.stack([torch.as_tensor(emb_cache[t], dtype=torch.float32) for t in batch["texts"]]).to(args.device)
        loss = multi_positive_infonce(
            brain(volume),
            text_proj(raw_text),
            batch["pos_mask"].to(args.device),
            batch["pos_weights"].to(args.device),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step={step} loss={float(loss.detach().cpu()):.4f}")


if __name__ == "__main__":
    main()
