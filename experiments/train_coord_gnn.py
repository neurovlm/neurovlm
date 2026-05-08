#!/usr/bin/env python
"""Atlas-Free Coordinate GNN (Track 2) — end-to-end training script.

Runs from the repo root:

    python experiments/train_coord_gnn.py

Optional flags
--------------
--epochs N              Training epochs. Default 200.
--batch-size N          Graphs per mini-batch. Default 64.
--k N                   KNN neighbor count. Default 7.
--max-dist-mm FLOAT     Max KNN edge distance in mm. Default 30.
--lr-gnn FLOAT          CoordGNN learning rate. Default 1e-4.
--lr-proj FLOAT         TextProjHead learning rate. Default 1e-5.
--hidden N              Hidden dim per attention head. Default 128.
--heads N               Number of attention heads. Default 8.
--warmup-epochs N       Linear LR warmup. Default 15.
--temperature FLOAT     InfoNCE temperature. Default 0.07.
--checkpoint-dir PATH   Where to save checkpoints. Default ./checkpoints/coord_gnn.
--cache-dir PATH        Where to cache KNN graphs. Default ./data/coord_graphs.
--device {auto,cpu,cuda,mps}
--seed N                Random seed. Default 42.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurovlm.data import load_dataset, load_latent
from neurovlm.metrics import bidirectional_retrieval_metrics
from neurovlm.retrieval_resources import _load_pubmed_coordinates
from neurovlm.gnn.coord_graph import normalize_coords
from neurovlm.gnn.coord_dataset import CoordGraphDataset
from neurovlm.gnn.coord_baselines import CoordDeepSet
from neurovlm.gnn.coord_diagnostics import (
    coord_graph_covariates,
    embedding_covariate_correlations,
    evaluate_pretrained_neurovlm_baseline,
    retrieval_diagnostics,
)
from neurovlm.gnn.coord_model import CoordGNN
from neurovlm.gnn.coord_train import CoordTrainer
from neurovlm.gnn.model import TextProjHead


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Atlas-Free Coordinate GNN (Track 2) on NeuroVLM data."
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--k", type=int, default=7, help="KNN neighbors per node.")
    p.add_argument("--max-dist-mm", type=float, default=30.0,
                   help="Max edge distance in mm (edges beyond this are pruned).")
    p.add_argument("--lr-gnn", type=float, default=1e-4)
    p.add_argument("--lr-proj", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--out-dim", type=int, default=384)
    p.add_argument("--model", default="coord_gnn",
                   choices=["coord_gnn", "coord_deepset"])
    p.add_argument("--no-self-loops", action="store_true",
                   help="Disable GAT-added self-loops for CoordGNN ablations.")
    p.add_argument("--text-proj-init", default="random",
                   choices=["random", "pretrained_infonce"],
                   help="Initialize text projector randomly or from NeuroVLM InfoNCE.")
    p.add_argument("--freeze-text-proj", action="store_true",
                   help="Train coordinates into the fixed pretrained text latent space.")
    p.add_argument("--warmup-epochs", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--checkpoint-dir", default="checkpoints/coord_gnn")
    p.add_argument("--cache-dir", default="data/coord_graphs")
    p.add_argument("--cache-file", default=None,
                   help="Optional packed graph cache file. Faster than tiny .pt files.")
    p.add_argument("--preload-to-ram", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=None)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--monitor-metric", default="mean_auc")
    p.add_argument("--run-dir", default=None,
                   help="Directory for config, metrics, and diagnostics.")
    p.add_argument("--diagnostics", action="store_true",
                   help="Save covariate/rank/PC correlation diagnostics.")
    p.add_argument("--eval-neurovlm-baseline", action="store_true",
                   help="Evaluate pretrained NeuroVLM image/text projectors on the same split.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 — Coordinate data
# ---------------------------------------------------------------------------

def phase1_load_coordinates(args: argparse.Namespace):
    """Load raw MNI peak coordinates and print descriptive statistics.

    Returns
    -------
    coords_df : pd.DataFrame  columns ['pmid', 'x', 'y', 'z']
    """
    print("\n── Phase 1: Coordinate Data ─────────────────────────────────────")
    print("Loading pubmed peak coordinates from HuggingFace …")
    coords_df = _load_pubmed_coordinates()

    print(f"  Coordinates shape: {coords_df.shape}")
    print(f"  Columns: {list(coords_df.columns)}")
    print(f"  First 3 papers:")
    for pmid, grp in list(coords_df.groupby("pmid"))[:3]:
        print(f"    PMID {pmid}: {len(grp)} peaks, "
              f"x∈[{grp.x.min():.1f},{grp.x.max():.1f}] "
              f"y∈[{grp.y.min():.1f},{grp.y.max():.1f}] "
              f"z∈[{grp.z.min():.1f},{grp.z.max():.1f}]")

    # Per-paper peak count statistics
    peak_counts = coords_df.groupby("pmid").size()
    print(f"\n  Peak counts across {len(peak_counts):,} papers:")
    print(f"    min={peak_counts.min()}  max={peak_counts.max()}")
    print(f"    mean={peak_counts.mean():.1f}  median={peak_counts.median():.0f}")
    print(f"    p5={peak_counts.quantile(0.05):.0f}  "
          f"p95={peak_counts.quantile(0.95):.0f}")

    n_too_few = (peak_counts < 3).sum()
    print(f"  Papers with <3 peaks (cannot form KNN-3): {n_too_few} "
          f"({100*n_too_few/len(peak_counts):.1f}%)")

    # Coordinate normalization statistics
    coords_arr = coords_df[["x", "y", "z"]].values
    norm = normalize_coords(coords_arr)
    outliers = (np.abs(norm) > 1.05).any(axis=1).sum()
    print(f"\n  After normalization: "
          f"range [{norm.min():.3f}, {norm.max():.3f}]")
    if outliers:
        print(f"  WARNING: {outliers} outlier coordinates clipped to ±1.0")

    # Duplicates per paper
    dup_papers = 0
    max_dups = 0
    for _, grp in coords_df.groupby("pmid"):
        raw = grp[["x", "y", "z"]].values
        unique = np.unique(raw, axis=0)
        n_dups = len(raw) - len(unique)
        if n_dups > 0:
            dup_papers += 1
            max_dups = max(max_dups, n_dups)
    print(f"  Papers with duplicate peaks: {dup_papers} "
          f"(max dups in one paper: {max_dups})")

    return coords_df


# ---------------------------------------------------------------------------
# Phase 2 — Dataset
# ---------------------------------------------------------------------------

def phase2_build_dataset(
    coords_df,
    args: argparse.Namespace,
):
    """Align coordinates with SPECTER embeddings and build CoordGraphDataset."""
    print("\n── Phase 2: Dataset ─────────────────────────────────────────────")

    print("Loading SPECTER text embeddings …")
    text_latents = load_latent("pubmed_text")
    pubmed_df = load_dataset("pubmed_text")

    # Resolve text tensor and PMID array
    if isinstance(text_latents, dict):
        pmid_list = list(text_latents.keys())
        text_tensor = torch.stack([
            torch.tensor(text_latents[p], dtype=torch.float32) for p in pmid_list
        ])
        unique_pmids = np.array([str(p) for p in pmid_list])
    else:
        text_tensor = text_latents if isinstance(text_latents, torch.Tensor) \
            else torch.tensor(text_latents)
        if hasattr(text_latents, "pmid"):
            unique_pmids = np.asarray(text_latents.pmid).astype(str)
        elif "pmid" in pubmed_df.columns:
            unique_pmids = pubmed_df["pmid"].astype(str).values[: len(text_tensor)]
        else:
            unique_pmids = np.arange(len(text_tensor)).astype(str)

    # Handle dict-payload format from _load_latent_text
    if isinstance(text_latents, tuple):
        text_tensor, pmids_arr = text_latents
        unique_pmids = np.asarray(pmids_arr).astype(str)

    print(f"  Text embeddings: {text_tensor.shape}")
    print(f"  Unique PMIDs in text: {len(unique_pmids):,}")

    print(f"\nBuilding CoordGraphDataset (k={args.k}, "
          f"max_dist={args.max_dist_mm}mm) …")
    print("  Graphs will be cached to disk on first run.")

    ds = CoordGraphDataset(
        coords_df=coords_df,
        text_embeddings=text_tensor,
        unique_pmids=unique_pmids,
        cache_dir=args.cache_dir,
        cache_file=args.cache_file,
        k=args.k,
        max_dist_mm=args.max_dist_mm,
        preload_to_ram=args.preload_to_ram,
    )
    print(f"  Dataset size: {len(ds):,} papers")

    # Verify PyG batching works
    from torch_geometric.loader import DataLoader
    probe_loader = DataLoader(ds, batch_size=4, shuffle=False)
    probe_batch = next(iter(probe_loader))
    print(f"\n  PyG batch check (batch_size=4):")
    print(f"    batch.x.shape      = {probe_batch.x.shape}  "
          f"(total_nodes × 5)")
    print(f"    batch.edge_index.shape = {probe_batch.edge_index.shape}")
    print(f"    batch.batch.shape  = {probe_batch.batch.shape}")
    print(f"    batch.y.shape      = {probe_batch.y.shape}")

    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds = ds.split(val_frac=0.1, test_frac=0.1,
                                          seed=args.seed)
    print(f"  Split: train={len(train_ds):,}  "
          f"val={len(val_ds):,}  test={len(test_ds):,}")

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Phase 3 — Model
# ---------------------------------------------------------------------------

def phase3_build_model(args: argparse.Namespace) -> tuple[torch.nn.Module, TextProjHead]:
    print("\n── Phase 3: Model ───────────────────────────────────────────────")
    if args.model == "coord_gnn":
        brain_encoder = CoordGNN(
            in_dim=5,
            hidden=args.hidden,
            heads=args.heads,
            out_dim=args.out_dim,
            add_self_loops=not args.no_self_loops,
        )
    elif args.model == "coord_deepset":
        brain_encoder = CoordDeepSet(
            in_dim=5,
            hidden=args.hidden,
            out_dim=args.out_dim,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.text_proj_init == "pretrained_infonce":
        if args.out_dim != 384:
            raise ValueError("pretrained_infonce text projector requires --out-dim 384")
        from neurovlm.models import ProjHead

        text_proj = ProjHead.from_pretrained("text_infonce")
    else:
        text_proj = TextProjHead(in_dim=768, hidden_dim=512, out_dim=args.out_dim)

    n_brain = brain_encoder.count_parameters()
    n_text = sum(p.numel() for p in text_proj.parameters())
    print(f"  Brain encoder       : {args.model}")
    print(f"  Text projector init : {args.text_proj_init}")
    print(f"  Text projector train: {not args.freeze_text_proj}")
    print(f"  Brain params        : {n_brain:,}")
    print(f"  TextProjHead params : {n_text:,}")

    if n_brain < 500_000:
        print("  WARNING: CoordGNN has <500K params — model may be too small.")
    elif n_brain > 5_000_000:
        print("  WARNING: CoordGNN has >5M params — may be slow on MPS.")

    # Shape verification
    with torch.no_grad():
        from torch_geometric.data import Data, Batch
        dummy = Data(
            x=torch.randn(10, 5),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            edge_attr=torch.randn(3, 4),
        )
        batch_obj = Batch.from_data_list([dummy, dummy])
        out = brain_encoder(
            batch_obj.x, batch_obj.edge_index,
            batch_obj.edge_attr, batch_obj.batch
        )
        assert out.shape == (2, args.out_dim), f"Unexpected output shape: {out.shape}"
    print(f"  Shape check passed: brain output (2, {args.out_dim}) ✓")

    return brain_encoder, text_proj


# ---------------------------------------------------------------------------
# Phase 4 — Training
# ---------------------------------------------------------------------------

def phase4_train(
    brain_encoder: torch.nn.Module,
    text_proj: TextProjHead,
    train_ds,
    val_ds,
    args: argparse.Namespace,
) -> CoordTrainer:
    print(f"\n── Phase 4: Training ────────────────────────────────────────────")
    trainer = CoordTrainer(
        brain_encoder=brain_encoder,
        text_proj=text_proj,
        lr_gnn=args.lr_gnn,
        lr_proj=args.lr_proj,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        temperature=args.temperature,
        device=args.device,
        val_interval=args.val_interval,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        use_amp=args.amp,
        monitor_metric=args.monitor_metric,
        early_stopping_patience=args.early_stopping_patience,
        freeze_text_proj=args.freeze_text_proj,
        config=vars(args),
        verbose=True,
    )
    trainer.fit(train_ds, val_ds)
    return trainer


# ---------------------------------------------------------------------------
# Phase 5 — Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def phase5_evaluate(
    trainer: CoordTrainer,
    test_ds,
    val_ds,
    args: argparse.Namespace,
) -> None:
    print("\n── Phase 5: Evaluation ──────────────────────────────────────────")
    trainer.restore_best()
    all_metrics: dict[str, dict[str, float]] = {}

    for split_name, ds in [("Val", val_ds), ("Test", test_ds)]:
        brain_emb, text_emb = trainer.collect_embeddings(ds)
        metrics = bidirectional_retrieval_metrics(text_emb, brain_emb)
        all_metrics[split_name.lower()] = metrics

        print(f"\n  {split_name} set ({len(ds)} papers):")
        print(f"    recall@1   = {metrics['mean_recall@1']:.4f}")
        print(f"    recall@5   = {metrics['mean_recall@5']:.4f}")
        print(f"    recall@10  = {metrics['mean_recall@10']:.4f}")
        print(f"    recall@50  = {metrics['mean_recall@50']:.4f}")
        print(f"    MRR        = {metrics['mean_mrr']:.4f}")
        print(f"    median rank= {metrics['mean_median_rank']:.1f}")
        print(
            f"    AUC        = {metrics['mean_auc']:.4f}  "
            "(recall-curve AUC; compare only to same metric)"
        )

    if args.eval_neurovlm_baseline:
        print("\n  Pretrained NeuroVLM baseline on the same PMID split:")
        for split_name, ds in [("Val", val_ds), ("Test", test_ds)]:
            baseline = evaluate_pretrained_neurovlm_baseline(ds, args.device)
            all_metrics[f"{split_name.lower()}_neurovlm_baseline"] = baseline
            print(
                f"    {split_name}: AUC={baseline['mean_auc']:.4f} "
                f"r@10={baseline['mean_recall@10']:.4f} "
                f"MRR={baseline['mean_mrr']:.4f} "
                f"n={int(baseline['n_eval'])}/{int(baseline['n_requested'])}"
            )

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n  ┌─────────────────────────┬──────────┬────────────┬────────┐")
    print("  │ Model                   │ AUC      │ Atlas-free │ Params │")
    print("  ├─────────────────────────┼──────────┼────────────┼────────┤")
    print("  │ NeuroVLM MLP baseline   │ report   │ No         │ —      │")
    print("  │ Track 1 DiFuMo GAT      │ (run it) │ No         │ —      │")
    auc = all_metrics["test"]["mean_auc"]
    print(f"  │ Track 2 {args.model:<14} │ {auc:.4f}   │ Yes        │ {trainer.brain_encoder.count_parameters()//1000}K  │")
    print("  └─────────────────────────┴──────────┴────────────┴────────┘")

    # ── Attention analysis ─────────────────────────────────────────────────
    print("\n── Phase 6: Attention Analysis (top-5 edges per paper) ──────────")
    if args.model == "coord_gnn":
        snapshots = trainer.get_attention_snapshot(
            val_ds, n_samples=10, exclude_self_loops=True
        )
        for snap in snapshots:
            print(f"\n  Paper idx={snap['paper_idx']} "
                  f"({snap['node_coords_mni'].shape[0]} peaks, "
                  f"self-loop mass={snap['attention_self_loop_mass']:.2f}):")
            for e in snap["top_edges"]:
                src = [f"{v:.1f}" for v in e["src_mni"]]
                dst = [f"{v:.1f}" for v in e["dst_mni"]]
                print(f"    [{', '.join(src)}] → [{', '.join(dst)}]  "
                      f"attn={e['weight']:.4f}")
    else:
        print("  Skipped: selected baseline has no attention weights.")

    # ── Failure analysis ───────────────────────────────────────────────────
    print("\n── Failure Analysis (20 worst papers) ───────────────────────────")
    brain_emb, text_emb = trainer.collect_embeddings(test_ds)
    brain_n = F.normalize(brain_emb.float(), dim=1)
    text_n = F.normalize(text_emb.float(), dim=1)
    sim = text_n @ brain_n.T
    hits = (sim.argsort(dim=1, descending=True)[:, :10] ==
            torch.arange(len(sim)).unsqueeze(1)).any(dim=1)
    worst_20 = (~hits).nonzero(as_tuple=True)[0][:20].tolist()
    print(f"  Found {(~hits).sum()} papers with recall@10=0 "
          f"out of {len(sim)} test papers.")
    print("  (Run UMAP visualization in the notebook for spatial analysis.)")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        if args.diagnostics:
            cov = coord_graph_covariates(test_ds)
            rank_df = retrieval_diagnostics(text_emb, brain_emb, covariates=cov)
            corr = embedding_covariate_correlations(brain_emb, cov)
            cov.to_csv(run_dir / "test_coord_covariates.csv", index=False)
            rank_df.to_csv(run_dir / "test_retrieval_diagnostics.csv", index=False)
            corr.to_csv(run_dir / "test_embedding_pc_covariates.csv", index=False)
            print(f"  Diagnostics saved to {run_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.run_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(Path("runs") / f"{args.model}_{stamp}")
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_file is None:
        cache_name = f"coord_graphs_k{args.k}_d{str(args.max_dist_mm).replace('.', 'p')}.pt"
        args.cache_file = str(Path(args.cache_dir).with_suffix("") / cache_name)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.run_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Phase 1 — Coordinate data
    coords_df = phase1_load_coordinates(args)

    # Phase 2 — Dataset
    train_ds, val_ds, test_ds = phase2_build_dataset(coords_df, args)

    # Phase 3 — Model
    brain_encoder, text_proj = phase3_build_model(args)

    # Phase 4 — Training
    trainer = phase4_train(brain_encoder, text_proj, train_ds, val_ds, args)

    # Phase 5 — Evaluation
    phase5_evaluate(trainer, test_ds, val_ds, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
