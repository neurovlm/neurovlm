#!/usr/bin/env python
"""Pretrain ALE CNN autoencoders with sparse-aware reconstruction losses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas_free_multipositive.evaluation.generation_metrics import generation_metrics
from atlas_free_multipositive.training.generation_losses import (
    GenerationLossConfig,
    combined_generation_loss,
)
from experiments.train_ale_cnn import build_dataset, which_device
from neurovlm.gnn.ale_cnn import ALE3DCNNAutoEncoder, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain ALE CNN autoencoder on atlas-free volumes.")
    p.add_argument("--mode", choices=["difumo_compatible", "atlas_free"], default="atlas_free")
    p.add_argument("--model", choices=["ale_3dcnn", "ale_3dcnn_resnet"], default="ale_3dcnn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batch-size-auto", action="store_true")
    p.add_argument("--batch-size-candidates", default="64,32,16,8,4")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=20)

    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--blocks-per-stage", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm", choices=["group", "batch", "instance", "none"], default="group")
    p.add_argument("--pooling", choices=["max", "stride"], default="max")
    p.add_argument("--use-dilation", action="store_true")
    p.add_argument("--multi-scale", action="store_true")
    p.add_argument("--global-context", choices=["none", "se", "attention"], default="none")

    p.add_argument("--kernel-fwhm-mm", type=float, default=9.0)
    p.add_argument("--resolution-mm", type=float, default=4.0)
    p.add_argument("--crop-to-brain", dest="crop_to_brain", action="store_true", default=True)
    p.add_argument("--no-crop-to-brain", dest="crop_to_brain", action="store_false")
    p.add_argument("--normalize", choices=["max", "mass", "none"], default="max")
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--cache-file", default=None)
    p.add_argument("--force-rebuild-cache", action="store_true")
    p.add_argument("--max-papers", type=int, default=None)

    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-dice", type=float, default=0.5)
    p.add_argument("--lambda-topk", type=float, default=0.5)
    p.add_argument("--lambda-corr", type=float, default=0.25)
    p.add_argument("--recon-alpha", type=float, default=10.0)
    p.add_argument("--recon-gamma", type=float, default=1.0)
    p.add_argument("--prediction-activation", choices=["sigmoid", "softplus", "none"], default="sigmoid")

    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    return p.parse_args()


def make_loader(ds, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )


def build_autoencoder(args: argparse.Namespace, output_shape: tuple[int, int, int]) -> ALE3DCNNAutoEncoder:
    return ALE3DCNNAutoEncoder(
        output_shape=output_shape,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        norm=args.norm,
        pooling=args.pooling,
        encoder_arch="resnet" if args.model == "ale_3dcnn_resnet" else "plain",
        blocks_per_stage=args.blocks_per_stage,
        use_dilation=args.use_dilation,
        multi_scale=args.multi_scale,
        global_context=args.global_context,
    )


def loss_config(args: argparse.Namespace) -> GenerationLossConfig:
    return GenerationLossConfig(
        lambda_recon=args.lambda_recon,
        lambda_dice=args.lambda_dice,
        lambda_topk=args.lambda_topk,
        lambda_corr=args.lambda_corr,
        recon_alpha=args.recon_alpha,
        recon_gamma=args.recon_gamma,
        prediction_activation=args.prediction_activation,
    )


def preflight_batch_size(
    model: nn.Module,
    train_ds,
    args: argparse.Namespace,
    device: torch.device,
    cfg: GenerationLossConfig,
) -> None:
    if not args.batch_size_auto:
        return
    candidates = [int(v) for v in args.batch_size_candidates.split(",") if v.strip()]
    for bs in candidates:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            args.batch_size = min(bs, len(train_ds))
            batch = next(iter(make_loader(train_ds, args, shuffle=False)))
            x = batch["volume"].float().to(device)
            model.train()
            pred = model(x)
            loss, _ = combined_generation_loss(pred, x, config=cfg)
            loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            print(f"Selected autoencoder batch_size={args.batch_size} preflight_peak_vram={peak:.0f}MB")
            return
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "mps backend out of memory" not in message:
                raise
            print(f"Autoencoder batch_size={bs} OOM; trying smaller.")
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(f"No autoencoder batch size fit from candidates={candidates}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    args: argparse.Namespace,
    device: torch.device,
    cfg: GenerationLossConfig,
    *,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    losses = []
    metric_rows = []
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    for batch in loader:
        x = batch["volume"].float().to(device, non_blocking=args.pin_memory)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(x)
                loss, _ = combined_generation_loss(pred, x, config=cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        metric_rows.append(generation_metrics(torch.sigmoid(pred.detach()), x.detach(), include_voxel_auroc=False))
    out = {k: float(np.mean([row[k] for row in metric_rows])) for k in metric_rows[0]} if metric_rows else {}
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


def main() -> None:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_dir is None:
        args.run_dir = str(Path("runs") / f"{args.model}_autoencoder_{args.mode}_{stamp}")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.run_dir) / "checkpoints")
    run_dir = Path(args.run_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = which_device(args.device)
    if args.pin_memory is False and device.type == "cuda":
        args.pin_memory = True

    ds, train_ds, val_ds, _test_ds, payload, _preprocess_config = build_dataset(args)
    model = build_autoencoder(args, ds.input_shape).to(device)
    cfg = loss_config(args)
    preflight_batch_size(model, train_ds, args, device, cfg)

    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_state = None
    bad_checks = 0

    config = vars(args)
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    with (run_dir / "preprocessing_config.json").open("w") as f:
        json.dump({**payload["config"], "metadata": payload["metadata"]}, f, indent=2)
    print(f"Autoencoder params={count_parameters(model):,} device={device} batch={args.batch_size}")

    for epoch in range(args.epochs):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, args, device, cfg, train=True)
        val_metrics = {}
        if epoch == 0 or epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, optimizer, scaler, args, device, cfg, train=False)
            score = val_metrics.get("top5_dice", 0.0) + val_metrics.get("spatial_corr", 0.0)
            state = {
                "encoder": deepcopy(model.encoder.state_dict()),
                "decoder": deepcopy(model.decoder.state_dict()),
                "model": deepcopy(model.state_dict()),
                "epoch": epoch,
                "metrics": val_metrics,
                "history": history,
                "config": config,
                "target_shape": ds.input_shape,
            }
            if score > best_score:
                best_score = score
                best_state = state
                torch.save(best_state, ckpt_dir / "best_cnn_autoencoder.pt")
                bad_checks = 0
            else:
                bad_checks += 1
        row = {
            "epoch": float(epoch),
            "epoch_time_sec": float(time.perf_counter() - start),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "history": history,
                "config": config,
                "target_shape": ds.input_shape,
            },
            ckpt_dir / "last_cnn_autoencoder.pt",
        )
        with (run_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)
        print(row, flush=True)
        if args.early_stopping_patience and bad_checks >= args.early_stopping_patience:
            print(f"Early stopping after {bad_checks} validation checks.")
            break

    if best_state is None:
        raise RuntimeError("No best autoencoder checkpoint was saved.")
    with (run_dir / "autoencoder_run_info.json").open("w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "best_checkpoint": str(ckpt_dir / "best_cnn_autoencoder.pt"),
                "last_checkpoint": str(ckpt_dir / "last_cnn_autoencoder.pt"),
                "best_score": best_score,
            },
            f,
            indent=2,
        )
    print(f"Best autoencoder checkpoint: {ckpt_dir / 'best_cnn_autoencoder.pt'}")


if __name__ == "__main__":
    main()
