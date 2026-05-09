#!/usr/bin/env python
"""Train ALE 3D CNN experiments for NeuroVLM.

Examples
--------
Atlas-free default:

    python experiments/train_ale_cnn.py --mode atlas_free --epochs 50 --batch-size 16

DiFuMo-compatible reconstruction from existing NeuroVLM flatmaps:

    python experiments/train_ale_cnn.py --mode difumo_compatible --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurovlm.gnn.ale_cnn import ALE3DCNNEncoder, ALEFlatMLPEncoder, count_parameters
from neurovlm.gnn.ale_dataset import ALEPreprocessConfig, ALEVolumeDataset, build_or_load_ale_cache
from neurovlm.gnn.model import TextProjHead
from neurovlm.loss import InfoNCELoss
from neurovlm.metrics import bidirectional_retrieval_metrics, recall_curve, retrieval_ranks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ALE 3D CNN on NeuroVLM PubMed pairs.")
    p.add_argument("--mode", choices=["difumo_compatible", "atlas_free"], default="atlas_free")
    p.add_argument("--model", choices=["ale_3dcnn", "ale_flat_mlp"], default="ale_3dcnn")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batch-size-auto", action="store_true")
    p.add_argument("--lr-cnn", type=float, default=1e-4)
    p.add_argument("--lr-proj", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--monitor-metric", default="paper_recall_curve_auc")

    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--out-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm", choices=["group", "batch", "instance", "none"], default="group")
    p.add_argument("--pooling", choices=["max", "stride"], default="max")
    p.add_argument("--mlp-hidden-dim", type=int, default=1024)

    p.add_argument("--kernel-fwhm-mm", type=float, default=9.0)
    p.add_argument("--resolution-mm", type=float, default=4.0)
    p.add_argument("--crop-to-brain", dest="crop_to_brain", action="store_true", default=True)
    p.add_argument("--no-crop-to-brain", dest="crop_to_brain", action="store_false")
    p.add_argument("--normalize", choices=["max", "mass", "none"], default="max")
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--cache-file", default=None)
    p.add_argument("--force-rebuild-cache", action="store_true")
    p.add_argument("--max-papers", type=int, default=None, help="Smoke-test subset size.")

    p.add_argument("--text-proj-init", choices=["random", "pretrained_infonce"], default="random")
    p.add_argument("--freeze-text-proj", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)

    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--comparison-file", default="runs/ale_model_comparison.csv")
    p.add_argument("--save-plots", action="store_true", default=True)
    p.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    p.add_argument("--umap", action="store_true", help="Save UMAP/PCA diagnostics.")
    p.add_argument("--eval-neurovlm-baseline", action="store_true")
    return p.parse_args()


def which_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    import math

    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(ds, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )


def build_dataset(args: argparse.Namespace):
    config = ALEPreprocessConfig(
        mode=args.mode,
        kernel_fwhm_mm=args.kernel_fwhm_mm,
        resolution_mm=args.resolution_mm,
        crop_to_brain=args.crop_to_brain,
        normalize=args.normalize,
        clamp=not args.no_clamp,
        cache_dtype=args.cache_dtype,
        max_papers=args.max_papers,
    )
    if args.cache_file is None:
        name = (
            f"{args.mode}_ale_{int(args.resolution_mm)}mm_"
            f"fwhm{str(args.kernel_fwhm_mm).replace('.', 'p')}_"
            f"{'crop' if args.crop_to_brain else 'full'}_{args.cache_dtype}.pt"
        )
        args.cache_file = str(Path("data/ale_caches") / name)
    payload = build_or_load_ale_cache(
        args.cache_file, config, force_rebuild=args.force_rebuild_cache
    )
    ds = ALEVolumeDataset.from_cache(payload)
    train_ds, val_ds, test_ds = ds.split(args.val_frac, args.test_frac, seed=args.seed)
    return ds, train_ds, val_ds, test_ds, payload, config


def build_model(args: argparse.Namespace, input_shape: tuple[int, ...]):
    if args.batch_size_auto:
        voxels = int(np.prod(input_shape))
        args.batch_size = 8 if voxels > 150_000 else 16 if voxels > 75_000 else 32

    if args.model == "ale_3dcnn":
        brain_encoder = ALE3DCNNEncoder(
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            out_dim=args.out_dim,
            dropout=args.dropout,
            norm=args.norm,
            pooling=args.pooling,
        )
    else:
        brain_encoder = ALEFlatMLPEncoder(
            hidden_dim=args.mlp_hidden_dim,
            out_dim=args.out_dim,
            dropout=args.dropout,
        )

    if args.text_proj_init == "pretrained_infonce":
        if args.out_dim != 384:
            raise ValueError("pretrained_infonce text projector requires --out-dim 384")
        from neurovlm.models import ProjHead

        text_proj = ProjHead.from_pretrained("text_infonce")
    else:
        text_proj = TextProjHead(in_dim=768, hidden_dim=512, out_dim=args.out_dim)

    with torch.no_grad():
        dummy = torch.zeros(2, 1, *input_shape)
        out = brain_encoder(dummy)
        assert out.shape == (2, args.out_dim), out.shape
    return brain_encoder, text_proj


class ALETrainer:
    def __init__(
        self,
        brain_encoder: nn.Module,
        text_proj: nn.Module,
        args: argparse.Namespace,
        device: torch.device,
    ):
        self.brain_encoder = brain_encoder.to(device)
        self.text_proj = text_proj.to(device)
        self.args = args
        self.device = device
        self.loss_fn = InfoNCELoss(args.temperature)
        if args.freeze_text_proj:
            for p in self.text_proj.parameters():
                p.requires_grad_(False)
        groups = [{"params": self.brain_encoder.parameters(), "lr": args.lr_cnn}]
        if not args.freeze_text_proj:
            groups.append({"params": self.text_proj.parameters(), "lr": args.lr_proj})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.use_amp = args.amp and device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.best_state: Optional[dict] = None
        self.best_score = -float("inf")
        self.history: dict[str, list] = {
            "train_loss": [],
            "epoch_time_sec": [],
            "peak_vram_mb": [],
            "lr_brain": [],
            "lr_proj": [],
            "val_epoch": [],
        }

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        volume = batch["volume"].to(self.device, non_blocking=self.args.pin_memory)
        text = batch["text"].to(self.device, non_blocking=self.args.pin_memory)
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                brain = self.brain_encoder(volume)
                text_emb = self.text_proj(text)
        else:
            brain = self.brain_encoder(volume)
            text_emb = self.text_proj(text)
        return brain, text_emb

    def fit(self, train_ds, val_ds) -> None:
        loader = make_loader(train_ds, self.args, shuffle=True)
        total_steps = max(1, len(loader) * self.args.epochs)
        warmup_steps = max(1, len(loader) * self.args.warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_warmup(step, warmup_steps, total_steps),
        )

        bad_checks = 0
        for epoch in range(self.args.epochs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
            losses = []
            self.brain_encoder.train()
            self.text_proj.train()

            for batch in loader:
                brain, text = self._forward(batch)
                loss = self.loss_fn(brain, text)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.brain_encoder.parameters()) + list(self.text_proj.parameters()),
                    max_norm=1.0,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                losses.append(float(loss.detach().cpu()))

            epoch_time = time.perf_counter() - start
            peak_vram = (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
                if self.device.type == "cuda"
                else 0.0
            )
            self.history["train_loss"].append(float(np.mean(losses)))
            self.history["epoch_time_sec"].append(float(epoch_time))
            self.history["peak_vram_mb"].append(float(peak_vram))
            self.history["lr_brain"].append(float(self.optimizer.param_groups[0]["lr"]))
            self.history["lr_proj"].append(
                0.0 if self.args.freeze_text_proj else float(self.optimizer.param_groups[1]["lr"])
            )

            if epoch % self.args.val_interval == 0:
                metrics, _, _ = self.evaluate(val_ds)
                self.history["val_epoch"].append(epoch)
                for k, v in metrics.items():
                    self.history.setdefault(f"val_{k}", []).append(float(v))
                score = float(metrics[self.args.monitor_metric])
                print(
                    f"Epoch {epoch:03d}/{self.args.epochs} "
                    f"loss={self.history['train_loss'][-1]:.4f} "
                    f"val_paper_recall_curve_auc={metrics['paper_recall_curve_auc']:.4f} "
                    f"val_r@10={metrics['mean_recall@10']:.4f} "
                    f"time={epoch_time:.1f}s vram={peak_vram:.0f}MB"
                )
                if score > self.best_score:
                    self.best_score = score
                    self.best_state = {
                        "brain_encoder": deepcopy(self.brain_encoder.state_dict()),
                        "text_proj": deepcopy(self.text_proj.state_dict()),
                        "epoch": epoch,
                        "metrics": metrics,
                        "config": vars(self.args),
                    }
                    bad_checks = 0
                else:
                    bad_checks += 1
                    if (
                        self.args.early_stopping_patience is not None
                        and bad_checks >= self.args.early_stopping_patience
                    ):
                        print(f"Early stopping after {bad_checks} validation checks.")
                        break
            else:
                print(
                    f"Epoch {epoch:03d}/{self.args.epochs} "
                    f"loss={self.history['train_loss'][-1]:.4f} time={epoch_time:.1f}s"
                )

        self.save_best()
        self.save_last()

    @torch.no_grad()
    def collect_embeddings(self, ds) -> tuple[torch.Tensor, torch.Tensor]:
        self.brain_encoder.eval()
        self.text_proj.eval()
        loader = make_loader(ds, self.args, shuffle=False)
        all_brain, all_text = [], []
        for batch in loader:
            brain, text = self._forward(batch)
            all_brain.append(brain.float().cpu())
            all_text.append(text.float().cpu())
        return torch.cat(all_brain), torch.cat(all_text)

    @torch.no_grad()
    def evaluate(self, ds) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        brain, text = self.collect_embeddings(ds)
        metrics = bidirectional_retrieval_metrics(text, brain)
        metrics["paper_recall_curve_auc"] = metrics["mean_auc"]
        metrics["full_recall_curve_auc_k1_to_N"] = metrics["mean_auc"]
        metrics["random_recall@10"] = metrics["mean_random_recall@10"]
        loss = self.loss_fn(brain, text)
        metrics["loss"] = float(loss.item())
        return metrics, brain, text

    def restore_best(self) -> None:
        if self.best_state is None:
            raise RuntimeError("No best checkpoint available.")
        self.brain_encoder.load_state_dict(self.best_state["brain_encoder"])
        self.text_proj.load_state_dict(self.best_state["text_proj"])

    def save_best(self) -> None:
        if self.best_state is None:
            return
        path = Path(self.args.checkpoint_dir) / "best_ale_cnn.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_state, path)

    def save_last(self) -> None:
        path = Path(self.args.checkpoint_dir) / "last_ale_cnn.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "brain_encoder": self.brain_encoder.state_dict(),
                "text_proj": self.text_proj.state_dict(),
                "history": self.history,
                "config": vars(self.args),
            },
            path,
        )


@torch.no_grad()
def recall_curve_frame(text_emb: torch.Tensor, brain_emb: torch.Tensor) -> pd.DataFrame:
    t2i, i2t = recall_curve(text_emb, brain_emb)
    n = len(t2i)
    return pd.DataFrame(
        {
            "k": np.arange(1, n + 1),
            "t2i_recall": t2i.cpu().numpy(),
            "i2t_recall": i2t.cpu().numpy(),
            "mean_recall": ((t2i + i2t) / 2).cpu().numpy(),
            "random_recall": np.arange(1, n + 1) / float(n),
        }
    )


@torch.no_grad()
def recall_curve_payload(text_emb: torch.Tensor, brain_emb: torch.Tensor) -> dict:
    """Return paper-style full recall curves as JSON-serializable arrays."""
    t2i, i2t = recall_curve(text_emb, brain_emb)
    n = len(t2i)
    return {
        "k_values": list(range(1, n + 1)),
        "text_to_brain_recall_curve": t2i.cpu().tolist(),
        "brain_to_text_recall_curve": i2t.cpu().tolist(),
        "mean_recall_curve": ((t2i + i2t) / 2).cpu().tolist(),
        "random_recall_curve": (torch.arange(1, n + 1).float() / float(n)).tolist(),
        "paper_recall_curve_auc": float(((t2i + i2t) / 2).mean().item()),
    }


@torch.no_grad()
def retrieval_diagnostics(text_emb: torch.Tensor, brain_emb: torch.Tensor, cov: pd.DataFrame) -> pd.DataFrame:
    import torch.nn.functional as F

    text_n = F.normalize(text_emb.float(), dim=1, eps=1e-8)
    brain_n = F.normalize(brain_emb.float(), dim=1, eps=1e-8)
    sim = text_n @ brain_n.T
    ranks = retrieval_ranks(sim).cpu().numpy()
    out = pd.DataFrame(
        {
            "sample_pos": np.arange(len(ranks)),
            "rank": ranks,
            "hit@10": ranks <= 10,
            "top1_pos": sim.argmax(dim=1).cpu().numpy(),
            "true_score": sim.diag().cpu().numpy(),
            "top1_score": sim.max(dim=1).values.cpu().numpy(),
        }
    )
    return out.merge(cov, on="sample_pos", how="left")


def save_plots(run_dir: Path, history: dict, curve_df: pd.DataFrame, diag_df: Optional[pd.DataFrame], brain_emb: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"])
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("InfoNCE")
    if "val_paper_recall_curve_auc" in history:
        ax[1].plot(
            history["val_epoch"],
            history["val_paper_recall_curve_auc"],
            label="paper recall-curve AUC",
        )
        for k in [1, 5, 10, 50]:
            key = f"val_mean_recall@{k}"
            if key in history:
                ax[1].plot(history["val_epoch"], history[key], label=f"r@{k}")
    ax[1].set_title("Validation Retrieval")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve_df["k"], curve_df["mean_recall"], label="model")
    ax.plot(curve_df["k"], curve_df["random_recall"], linestyle="--", label="random")
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("recall")
    ax.set_title("Full Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "recall_curve.png", dpi=160)
    plt.close(fig)

    if diag_df is None:
        return
    try:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=42)
            xy = reducer.fit_transform(brain_emb.float().numpy())
            method = "UMAP"
        except Exception:
            from sklearn.decomposition import PCA

            xy = PCA(n_components=2).fit_transform(brain_emb.float().numpy())
            method = "PCA"
    except Exception as exc:
        print(f"Skipping UMAP/PCA diagnostics: {exc}")
        return

    color_cols = [
        "n_peaks",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "hit@10",
        "total_activation_mass",
        "sparsity",
    ]
    color_cols = [c for c in color_cols if c in diag_df.columns]
    if not color_cols:
        return
    n_cols = min(3, len(color_cols))
    n_rows = int(np.ceil(len(color_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for ax, col in zip(axes.flat, color_cols):
        vals = diag_df[col]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=8, cmap="viridis")
        ax.set_title(col)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        fig.colorbar(sc, ax=ax, fraction=0.046)
    for ax in axes.flat[len(color_cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "umap_diagnostics.png", dpi=160)
    plt.close(fig)


def save_embedding_correlations(run_dir: Path, brain_emb: torch.Tensor, cov: pd.DataFrame) -> None:
    from neurovlm.gnn.ale_cnn import embedding_covariate_correlations

    try:
        corr = embedding_covariate_correlations(brain_emb, cov)
    except Exception as exc:
        print(f"Skipping embedding covariate correlations: {exc}")
        return
    corr.to_csv(run_dir / "embedding_covariate_correlations.csv", index=False)


@torch.no_grad()
def evaluate_neurovlm_baseline_by_pmids(pmids: np.ndarray, device: torch.device) -> dict[str, float]:
    from neurovlm.models import ProjHead
    from neurovlm.retrieval_resources import _load_latent_neuro, _load_latent_text

    brain_latent, brain_pmids = _load_latent_neuro()
    text_latent, text_pmids = _load_latent_text()
    brain_lookup = {str(p): i for i, p in enumerate(brain_pmids)}
    text_lookup = {str(p): i for i, p in enumerate(text_pmids)}
    pairs = [(brain_lookup[p], text_lookup[p]) for p in map(str, pmids) if p in brain_lookup and p in text_lookup]
    if not pairs:
        raise RuntimeError("No overlapping PMIDs for pretrained NeuroVLM baseline.")
    b_idx, t_idx = zip(*pairs)
    image_proj = ProjHead.from_pretrained("image_infonce").to(device).eval()
    text_proj = ProjHead.from_pretrained("text_infonce").to(device).eval()
    brain = image_proj(brain_latent[list(b_idx)].float().to(device)).cpu()
    text = text_proj(text_latent[list(t_idx)].float().to(device)).cpu()
    metrics = bidirectional_retrieval_metrics(text, brain)
    metrics["n_eval"] = float(len(pairs))
    metrics["n_requested"] = float(len(pmids))
    return metrics


def append_comparison_row(args, payload, metrics, trainer, input_shape) -> dict:
    path = Path(args.comparison_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model_name": args.model,
        "preprocessing_type": args.mode,
        "uses_difumo": "yes" if args.mode == "difumo_compatible" else "no",
        "atlas_free": "yes" if args.mode == "atlas_free" else "no",
        "input_shape": str(tuple([1, *input_shape])),
        "number_of_parameters": count_parameters(trainer.brain_encoder),
        "training_time_per_epoch": float(np.mean(trainer.history["epoch_time_sec"])),
        "peak_vram": float(max(trainer.history["peak_vram_mb"] or [0.0])),
        "paper_recall_curve_auc": metrics["paper_recall_curve_auc"],
        "full_recall_curve_auc_k1_to_N": metrics["full_recall_curve_auc_k1_to_N"],
        "full_recall_curve_auc": metrics["paper_recall_curve_auc"],
        "recall@1": metrics["mean_recall@1"],
        "recall@5": metrics["mean_recall@5"],
        "recall@10": metrics["mean_recall@10"],
        "recall@50": metrics["mean_recall@50"],
        "mrr": metrics["mean_mrr"],
        "median_rank": metrics["mean_median_rank"],
        "random_recall@10": metrics["mean_random_recall@10"],
        "checkpoint_path": str(Path(args.checkpoint_dir) / "best_ale_cnn.pt"),
        "config_path": str(Path(args.run_dir) / "config.json"),
        "cache_path": str(args.cache_file),
        "cache_shape": str(tuple(payload["metadata"]["shape"])),
    }
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    jsonl = path.with_suffix(".jsonl")
    with jsonl.open("a") as f:
        f.write(json.dumps(row) + "\n")
    with (Path(args.run_dir) / "comparison_row.json").open("w") as f:
        json.dump(row, f, indent=2)
    return row


def main() -> None:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_dir is None:
        args.run_dir = str(Path("runs") / f"{args.model}_{args.mode}_{stamp}")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.run_dir) / "checkpoints")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds, train_ds, val_ds, test_ds, payload, preprocess_config = build_dataset(args)
    brain_encoder, text_proj = build_model(args, ds.input_shape)
    device = which_device(args.device)
    if args.pin_memory is False and device.type == "cuda":
        args.pin_memory = True

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (run_dir / "preprocessing_config.json").open("w") as f:
        json.dump({**payload["config"], "metadata": payload["metadata"]}, f, indent=2)
    with (run_dir / "model_config.json").open("w") as f:
        json.dump(
            {
                "model": args.model,
                "base_channels": args.base_channels,
                "num_blocks": args.num_blocks,
                "out_dim": args.out_dim,
                "dropout": args.dropout,
                "norm": args.norm,
                "pooling": args.pooling,
                "input_shape": ds.input_shape,
                "parameters": count_parameters(brain_encoder),
            },
            f,
            indent=2,
        )

    print("\n== ALE Dataset ==")
    print(f"  mode={args.mode} cache={args.cache_file}")
    print(f"  volume shape={ds.input_shape} aligned papers={len(ds):,}")
    print(f"  split train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print("\n== Model ==")
    print(f"  {args.model} brain params={count_parameters(brain_encoder):,}")
    print(f"  text params={count_parameters(text_proj):,} train_text={not args.freeze_text_proj}")
    print(f"  device={device} amp={args.amp and device.type == 'cuda'} batch={args.batch_size}")

    trainer = ALETrainer(brain_encoder, text_proj, args, device)
    trainer.fit(train_ds, val_ds)
    trainer.restore_best()

    all_metrics = {}
    eval_payloads = {}
    for name, split_ds in [("val", val_ds), ("test", test_ds)]:
        metrics, brain_emb, text_emb = trainer.evaluate(split_ds)
        all_metrics[name] = metrics
        eval_payloads[name] = (metrics, brain_emb, text_emb)
        print(
            f"\n{name.upper()} n={len(split_ds):,} "
            f"paper_recall_curve_auc={metrics['paper_recall_curve_auc']:.4f} "
            f"r@1={metrics['mean_recall@1']:.4f} "
            f"r@5={metrics['mean_recall@5']:.4f} "
            f"r@10={metrics['mean_recall@10']:.4f} "
            f"r@50={metrics['mean_recall@50']:.4f} "
            f"MRR={metrics['mean_mrr']:.4f} "
            f"median_rank={metrics['mean_median_rank']:.1f} "
            f"random_r@10={metrics['mean_random_recall@10']:.4f}"
        )

    if args.eval_neurovlm_baseline:
        all_metrics["test_neurovlm_baseline"] = evaluate_neurovlm_baseline_by_pmids(test_ds.pmids, device)

    with (run_dir / "training_history.json").open("w") as f:
        json.dump(trainer.history, f, indent=2)
    with (run_dir / "eval_results.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    curve_frames = {}
    for split_name, (_, brain_emb, text_emb) in eval_payloads.items():
        curve_df = recall_curve_frame(text_emb, brain_emb)
        curve_frames[split_name] = curve_df
        curve_df.to_csv(run_dir / f"{split_name}_recall_curve.csv", index=False)
        with (run_dir / f"{split_name}_recall_curve.json").open("w") as f:
            json.dump(recall_curve_payload(text_emb, brain_emb), f, indent=2)

    test_metrics, test_brain, test_text = eval_payloads["test"]
    curve_df = curve_frames["test"]
    cov = test_ds.covariate_frame()
    diag_df = retrieval_diagnostics(test_text, test_brain, cov)
    diag_df.to_csv(run_dir / "test_retrieval_diagnostics.csv", index=False)
    save_embedding_correlations(run_dir, test_brain, cov)
    if args.save_plots:
        save_plots(run_dir, trainer.history, curve_df, diag_df if args.umap else None, test_brain)

    append_comparison_row(args, payload, test_metrics, trainer, ds.input_shape)
    print(f"\nArtifacts saved to {run_dir}")
    print(f"Comparison row appended to {args.comparison_file}")


if __name__ == "__main__":
    main()
