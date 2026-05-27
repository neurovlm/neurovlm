#!/usr/bin/env python
"""DiFuMo Soft Atlas GAT — end-to-end training script.

Phases 1–5 in one file.  Run from the repo root:

    python experiments/train_difumo_gat.py

Optional flags
--------------
--fc-path PATH          Path to a precomputed (512, 512) FC matrix (.npy/.npz).
                        If omitted, FC is estimated from training data co-activations.
--epochs N              Training epochs.  Default 150.
--batch-size N          Graphs per mini-batch.  Default 512.
--lr-gat FLOAT          GAT learning rate.  Default 1e-4.
--lr-proj FLOAT         TextProjHead learning rate.  Default 1e-5.
--hidden N              GAT hidden dim per head.  Default 64.
--heads N               GAT attention heads.  Default 8.
--threshold FLOAT       FC percentile threshold (e.g. 90 → top 10 %).  Default 90.
--checkpoint-dir PATH   Directory to save best model checkpoint.  Default ./checkpoints/gat.
--device {cpu,cuda,mps,auto}  Target device.  Default auto.
--seed N                Random seed.  Default 42.
--add-centroids         Append component (x, y, z) centroids as additional node features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── ensure the package is importable when running from the repo root ──────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurovlm.data import load_dataset, load_latent
from neurovlm.metrics import recall_at_k, recall_curve
from neurovlm.gnn.atlas import (
    load_difumo_components,
    compute_difumo_coefficients,
    get_component_centroids,
    DIFUMO_DIM,
)
from neurovlm.gnn.graph import build_brain_graph
from neurovlm.gnn.dataset import BrainGraphDataset
from neurovlm.gnn.model import BrainGAT, TextProjHead
from neurovlm.gnn.train import GATTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DiFuMo Soft Atlas GAT on NeuroVLM data."
    )
    p.add_argument("--fc-path", default=None, help="Path to precomputed FC matrix (.npy/.npz).")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr-gat", type=float, default=1e-4)
    p.add_argument("--lr-proj", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--threshold", type=float, default=90.0,
                   help="FC percentile threshold (default 90 → keep top 10%%).")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--checkpoint-dir", default="checkpoints/gat")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--add-centroids", action="store_true",
                   help="Append DiFuMo component MNI centroids (x,y,z) as node features.")
    p.add_argument("--val-interval", type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: brain representation (DiFuMo coefficients + graph)
# ---------------------------------------------------------------------------

def phase1_build_graph(args: argparse.Namespace):
    """Load brain images, compute DiFuMo coefficients, build the brain graph.

    Returns
    -------
    difumo_coeffs : np.ndarray, shape (N, 512)
    edge_index    : torch.LongTensor, shape (2, E)
    edge_attr     : torch.FloatTensor, shape (E, 1)
    pmids         : list  (N,) paper identifiers for alignment with text
    """
    print("\n── Phase 1: Brain Graph ─────────────────────────────────────────")

    # ── 1a. Load raw brain flatmaps ──────────────────────────────────────
    print("Loading pubmed brain flatmaps …")
    images_data = load_dataset("pubmed_images")

    # load_dataset("pubmed_images") returns (images_tensor, pmids_tensor)
    if isinstance(images_data, (tuple, list)):
        brain_flat_tensor, pmids_raw = images_data[0], images_data[1]
    else:
        brain_flat_tensor = images_data
        pmids_raw = None

    brain_flat = brain_flat_tensor.numpy().astype(np.float32)
    print(f"  Brain flatmaps shape : {brain_flat.shape}")

    # ── 1b. DiFuMo component matrix P ────────────────────────────────────
    print(f"Loading DiFuMo {DIFUMO_DIM} component atlas …")
    P = load_difumo_components(dimension=DIFUMO_DIM)
    print(f"  P shape (components × voxels) : {P.shape}")

    # ── 1c. Per-paper DiFuMo coefficient vectors ─────────────────────────
    print("Projecting brain flatmaps onto DiFuMo components …")
    difumo_coeffs = compute_difumo_coefficients(brain_flat, P, normalize=True)
    print(f"  DiFuMo coefficients shape : {difumo_coeffs.shape}")

    # ── 1d. Brain graph ───────────────────────────────────────────────────
    print(f"Building brain graph (FC threshold: {args.threshold}th percentile) …")
    edge_index, edge_attr = build_brain_graph(
        fc_path=args.fc_path,
        difumo_coeffs=difumo_coeffs,
        percentile=args.threshold,
        verbose=True,
    )

    pmids = pmids_raw.tolist() if pmids_raw is not None else list(range(len(brain_flat)))
    return difumo_coeffs, edge_index, edge_attr, pmids


# ---------------------------------------------------------------------------
# Phase 2: assemble dataset
# ---------------------------------------------------------------------------

def phase2_build_dataset(
    difumo_coeffs: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pmids: list,
    args: argparse.Namespace,
) -> tuple[BrainGraphDataset, BrainGraphDataset, BrainGraphDataset]:
    """Align brain graphs with SPECTER text embeddings and split.

    Returns
    -------
    train_ds, val_ds, test_ds : BrainGraphDataset
    """
    print("\n── Phase 2: Dataset ─────────────────────────────────────────────")

    # ── Load text data and align by PMID ─────────────────────────────────
    print("Loading SPECTER text embeddings …")
    text_latents = load_latent("pubmed_text")   # (N_text, 768) or dict

    pubmed_df = load_dataset("pubmed_text")     # DataFrame with 'pmid' column

    # Resolve text_latents: may be a tensor or dict keyed by pmid
    if isinstance(text_latents, dict):
        # Build aligned arrays from the dict
        text_embed_map = text_latents
        aligned_brain, aligned_text = [], []
        valid_pmids = set(str(p) for p in text_embed_map.keys())
        for i, pmid in enumerate(pmids):
            key = str(pmid)
            if key in valid_pmids:
                aligned_brain.append(difumo_coeffs[i])
                aligned_text.append(np.array(text_embed_map[key], dtype=np.float32))
        aligned_brain = np.stack(aligned_brain)
        aligned_text = np.stack(aligned_text)
    else:
        # text_latents is a tensor of shape (N_text, 768); align via PMID join
        text_tensor = text_latents if isinstance(text_latents, torch.Tensor) \
            else torch.tensor(text_latents)

        if "pmid" in pubmed_df.columns and pmids[0] != 0:
            # Build a pmid → row-index map for the text tensor
            pmid_to_text_idx = {
                str(row["pmid"]): idx
                for idx, row in pubmed_df.iterrows()
                if idx < len(text_tensor)
            }
            brain_rows, text_rows = [], []
            for i, pmid in enumerate(pmids):
                key = str(pmid)
                if key in pmid_to_text_idx:
                    brain_rows.append(i)
                    text_rows.append(pmid_to_text_idx[key])

            aligned_brain = difumo_coeffs[brain_rows]
            aligned_text = text_tensor[text_rows].numpy()
            print(f"  Aligned {len(brain_rows)} paper pairs via PMID.")
        else:
            # Fallback: assume the two tensors are already row-aligned
            n = min(len(difumo_coeffs), len(text_tensor))
            aligned_brain = difumo_coeffs[:n]
            aligned_text = text_tensor[:n].numpy()
            print(f"  No PMID column found — assuming row-aligned ({n} pairs).")

    print(f"  Brain : {aligned_brain.shape}  |  Text : {aligned_text.shape}")

    # ── Optional centroid node features ──────────────────────────────────
    extra_node_feats = None
    if args.add_centroids:
        print("Computing DiFuMo component MNI centroids …")
        centroids = get_component_centroids(dimension=DIFUMO_DIM)
        # Normalize centroids to [-1, 1] MNI range (~150 mm)
        centroids = centroids / 150.0
        extra_node_feats = torch.tensor(centroids, dtype=torch.float32)
        print(f"  Centroid features shape : {extra_node_feats.shape}")

    # ── Build and split dataset ───────────────────────────────────────────
    ds = BrainGraphDataset(
        difumo_coeffs=torch.tensor(aligned_brain, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        text_embeddings=torch.tensor(aligned_text, dtype=torch.float32),
        extra_node_feats=extra_node_feats,
    )

    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds = ds.split(val_frac=0.1, test_frac=0.1, seed=args.seed)
    print(f"  Split: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Phase 3 + 4: build model and train
# ---------------------------------------------------------------------------

def phase3_build_model(args: argparse.Namespace) -> tuple[BrainGAT, TextProjHead]:
    """Instantiate BrainGAT and TextProjHead."""
    in_dim = 1 + (3 if args.add_centroids else 0)
    brain_encoder = BrainGAT(
        in_dim=in_dim,
        hidden=args.hidden,
        heads=args.heads,
        out_dim=384,
    )
    text_proj = TextProjHead(in_dim=768, hidden_dim=512, out_dim=384)

    n_brain = sum(p.numel() for p in brain_encoder.parameters())
    n_text = sum(p.numel() for p in text_proj.parameters())
    print(f"\n── Phase 3: Model ───────────────────────────────────────────────")
    print(f"  BrainGAT params     : {n_brain:,}")
    print(f"  TextProjHead params : {n_text:,}")

    # Shape verification
    with torch.no_grad():
        from torch_geometric.data import Data, Batch

        dummy_x = torch.randn(512, in_dim)
        ei = torch.zeros(2, 1, dtype=torch.long)
        ea = torch.zeros(1, 1)
        b = torch.zeros(512, dtype=torch.long)
        out = brain_encoder(dummy_x, ei, ea, b)
        assert out.shape == (1, 384), f"Unexpected output shape: {out.shape}"
    print(f"  Shape check passed: GAT output (1, 384) ✓")

    return brain_encoder, text_proj


def phase4_train(
    brain_encoder: BrainGAT,
    text_proj: TextProjHead,
    train_ds: BrainGraphDataset,
    val_ds: BrainGraphDataset,
    args: argparse.Namespace,
) -> GATTrainer:
    print(f"\n── Phase 4: Training ────────────────────────────────────────────")
    trainer = GATTrainer(
        brain_encoder=brain_encoder,
        text_proj=text_proj,
        lr_gat=args.lr_gat,
        lr_proj=args.lr_proj,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        temperature=args.temperature,
        device=args.device,
        val_interval=args.val_interval,
        checkpoint_dir=args.checkpoint_dir,
        verbose=True,
    )
    trainer.fit(train_ds, val_ds)
    return trainer


# ---------------------------------------------------------------------------
# Phase 5: evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def phase5_evaluate(
    trainer: GATTrainer,
    test_ds: BrainGraphDataset,
    val_ds: BrainGraphDataset,
) -> None:
    """Recall@k evaluation on val and test sets.

    Reports AUC and recall@1/5/10 side-by-side for easy comparison with the
    NeuroVLM MLP baseline (baseline AUC ≈ 0.81 on PubMed contrastive).
    """
    print("\n── Phase 5: Evaluation ──────────────────────────────────────────")

    trainer.restore_best()
    brain_encoder = trainer.brain_encoder.eval()
    text_proj = trainer.text_proj.eval()
    device = trainer.device

    from torch_geometric.loader import DataLoader

    def _collect(ds: BrainGraphDataset):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        all_b, all_t = [], []
        for batch in loader:
            batch = batch.to(device)
            b = brain_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            t = text_proj(batch.y)
            all_b.append(b)
            all_t.append(t)
        return torch.cat(all_b), torch.cat(all_t)

    for split_name, ds in [("Val", val_ds), ("Test", test_ds)]:
        brain_emb, text_emb = _collect(ds)
        brain_n = F.normalize(brain_emb, dim=1)
        text_n = F.normalize(text_emb, dim=1)
        sim = text_n @ brain_n.T   # (N, N)

        r1 = recall_at_k(sim, 1)
        r5 = recall_at_k(sim, 5)
        r10 = recall_at_k(sim, 10)
        t2i, i2t = recall_curve(text_emb, brain_emb)
        auc = float((t2i.mean() + i2t.mean()) / 2)

        print(f"\n  {split_name} set ({len(ds)} papers):")
        print(f"    recall@1  = {r1:.4f}")
        print(f"    recall@5  = {r5:.4f}")
        print(f"    recall@10 = {r10:.4f}")
        print(f"    AUC       = {auc:.4f}  (NeuroVLM MLP baseline ≈ 0.81)")

    # Interpretability snapshot (first 64 val samples)
    print("\n── Attention snapshot (top 10 edges in layer 3, val set) ────────")
    try:
        snap = trainer.get_attention_snapshot(val_ds, n_samples=64)
        top10 = snap["attn_mean"].topk(10)
        ei = snap["edge_index"]
        print(f"  {'src':>5} → {'dst':>5}  |  attn")
        for rank, (idx, val) in enumerate(
            zip(top10.indices.tolist(), top10.values.tolist())
        ):
            s = ei[0, idx].item()
            d = ei[1, idx].item()
            print(f"  {s:5d} → {d:5d}  |  {val:.4f}")
    except Exception as exc:
        print(f"  (Attention snapshot failed: {exc})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Phase 1 — brain graph
    difumo_coeffs, edge_index, edge_attr, pmids = phase1_build_graph(args)

    # Phase 2 — dataset
    train_ds, val_ds, test_ds = phase2_build_dataset(
        difumo_coeffs, edge_index, edge_attr, pmids, args
    )

    # Phase 3 — model
    brain_encoder, text_proj = phase3_build_model(args)

    # Phase 4 — training
    trainer = phase4_train(brain_encoder, text_proj, train_ds, val_ds, args)

    # Phase 5 — evaluation
    phase5_evaluate(trainer, test_ds, val_ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
