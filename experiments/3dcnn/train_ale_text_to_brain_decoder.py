#!/usr/bin/env python
"""Train text -> CNN latent -> frozen ALE decoder generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from neurovlm.gnn.model import TextProjHead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train text projection through a frozen ALE CNN decoder.")
    p.add_argument("--mode", choices=["atlas_free", "difumo_compatible"], default="atlas_free")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-size-auto", action="store_true")
    p.add_argument("--batch-size-candidates", default="512,384,256,192,128,96,64,32,16,8,4")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=20)

    p.add_argument("--autoencoder-checkpoint", required=True)
    p.add_argument("--text-proj-init", choices=["random", "pretrained_infonce"], default="pretrained_infonce")
    p.add_argument("--text-proj-checkpoint", default=None, help="Optional best_ale_cnn.pt to initialize text_proj from.")
    p.add_argument("--freeze-decoder", action="store_true", default=True)
    p.add_argument("--unfreeze-decoder", dest="freeze_decoder", action="store_false")
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--unfreeze-encoder", dest="freeze_encoder", action="store_false")

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
    p.add_argument("--lambda-latent", type=float, default=1.0)
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
    num_workers = int(args.num_workers)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=num_workers > 0,
    )


def _autoencoder_kwargs(payload: dict, fallback_shape: tuple[int, int, int]) -> dict:
    cfg = dict(payload.get("config") or {})
    target_shape = tuple(int(v) for v in payload.get("target_shape") or cfg.get("target_shape") or fallback_shape)
    model_name = str(cfg.get("model", "ale_3dcnn"))
    return {
        "output_shape": target_shape,
        "base_channels": int(cfg.get("base_channels", 48)),
        "num_blocks": int(cfg.get("num_blocks", 4)),
        "latent_dim": int(cfg.get("latent_dim", cfg.get("out_dim", 384))),
        "dropout": float(cfg.get("dropout", 0.1)),
        "norm": str(cfg.get("norm", "group")),
        "pooling": str(cfg.get("pooling", "max")),
        "encoder_arch": "resnet" if model_name == "ale_3dcnn_resnet" or cfg.get("encoder_arch") == "resnet" else "plain",
        "blocks_per_stage": int(cfg.get("blocks_per_stage", 2)),
        "use_dilation": bool(cfg.get("use_dilation", False)),
        "multi_scale": bool(cfg.get("multi_scale", False)),
        "global_context": str(cfg.get("global_context", "none")),
    }


def load_autoencoder(checkpoint: str | Path, fallback_shape: tuple[int, int, int]) -> ALE3DCNNAutoEncoder:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kwargs = _autoencoder_kwargs(payload, fallback_shape)
    model = ALE3DCNNAutoEncoder(**kwargs)
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.encoder.load_state_dict(payload["encoder"], strict=True)
        model.decoder.load_state_dict(payload["decoder"], strict=True)
    return model


def build_text_projector(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.text_proj_init == "pretrained_infonce":
        from neurovlm.models import ProjHead

        text_proj = ProjHead.from_pretrained("text_infonce")
    else:
        text_proj = TextProjHead(in_dim=768, hidden_dim=512, out_dim=384)
    if args.text_proj_checkpoint:
        payload = torch.load(args.text_proj_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("text_proj") or payload.get("text_projector")
        if not state:
            raise KeyError("--text-proj-checkpoint must contain 'text_proj' or 'text_projector'")
        text_proj.load_state_dict(state, strict=True)
    return text_proj.to(device)


def loss_config(args: argparse.Namespace) -> GenerationLossConfig:
    return GenerationLossConfig(
        lambda_recon=args.lambda_recon,
        lambda_latent=args.lambda_latent,
        lambda_dice=args.lambda_dice,
        lambda_topk=args.lambda_topk,
        lambda_corr=args.lambda_corr,
        recon_alpha=args.recon_alpha,
        recon_gamma=args.recon_gamma,
        prediction_activation=args.prediction_activation,
    )


def preflight_batch_size(autoencoder, text_proj, train_ds, args, device, cfg) -> None:
    if not args.batch_size_auto:
        return
    candidates = [int(v.strip()) for v in str(args.batch_size_candidates).split(",") if v.strip()]
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    for bs in candidates:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            args.batch_size = min(bs, len(train_ds))
            batch = next(iter(make_loader(train_ds, args, shuffle=False)))
            x = batch["volume"].float().to(device)
            text = batch["text"].float().to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                brain_z = autoencoder.encoder(x)
                text_z = text_proj(text)
                pred = autoencoder.decoder(text_z)
                loss, _ = combined_generation_loss(pred, x, brain_z=brain_z, text_z=text_z, config=cfg)
            loss.backward()
            text_proj.zero_grad(set_to_none=True)
            autoencoder.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            print(f"Selected text-to-brain batch_size={args.batch_size} preflight_peak_vram={peak:.0f}MB", flush=True)
            return
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "mps backend out of memory" not in message:
                raise
            print(f"text-to-brain batch_size={bs} OOM; trying smaller.", flush=True)
            text_proj.zero_grad(set_to_none=True)
            autoencoder.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(f"No text-to-brain batch size fit from candidates={candidates}")


def run_epoch(autoencoder, text_proj, loader, optimizer, scaler, args, device, cfg, *, train: bool) -> dict[str, float]:
    autoencoder.eval()
    text_proj.train(train)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    losses = []
    parts_rows: list[dict[str, float]] = []
    metric_rows: list[dict[str, float]] = []
    for batch in loader:
        x = batch["volume"].float().to(device, non_blocking=args.pin_memory)
        text = batch["text"].float().to(device, non_blocking=args.pin_memory)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                with torch.no_grad() if args.freeze_encoder else torch.enable_grad():
                    brain_z = autoencoder.encoder(x)
                text_z = text_proj(text)
                pred = autoencoder.decoder(text_z)
                loss, parts = combined_generation_loss(pred, x, brain_z=brain_z, text_z=text_z, config=cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in text_proj.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        parts_rows.append({k: float(v.detach().float().cpu()) for k, v in parts.items()})
        metric_rows.append(generation_metrics(torch.sigmoid(pred.detach()), x.detach(), include_voxel_auroc=False))
    out = {k: float(np.mean([row[k] for row in metric_rows])) for k in metric_rows[0]} if metric_rows else {}
    for key in parts_rows[0] if parts_rows else []:
        out[f"loss_{key}"] = float(np.mean([row[key] for row in parts_rows]))
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


@torch.no_grad()
def save_generation_examples(autoencoder, text_proj, ds, args, device, out_dir: Path, n: int = 4) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping generation plot: {exc}", flush=True)
        return
    loader = make_loader(ds, args, shuffle=False)
    batch = next(iter(loader))
    x = batch["volume"].float().to(device)[:n]
    text = batch["text"].float().to(device)[:n]
    pred = torch.sigmoid(autoencoder.decoder(text_proj(text))).detach().cpu()
    target = x.detach().cpu()
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
    for i in range(n):
        z = target.shape[-1] // 2
        axes[i, 0].imshow(target[i, 0, :, :, z], cmap="magma")
        axes[i, 0].set_title("target")
        axes[i, 1].imshow(pred[i, 0, :, :, z], cmap="magma")
        axes[i, 1].set_title("generated")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "text_to_brain_examples.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_dir is None:
        args.run_dir = str(Path("runs") / f"ale_text_to_brain_decoder_{stamp}")
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

    ds, train_ds, val_ds, test_ds, payload, _preprocess_config = build_dataset(args)
    autoencoder = load_autoencoder(args.autoencoder_checkpoint, ds.input_shape).to(device)
    if args.freeze_encoder:
        for p in autoencoder.encoder.parameters():
            p.requires_grad_(False)
    if args.freeze_decoder:
        for p in autoencoder.decoder.parameters():
            p.requires_grad_(False)
    autoencoder.eval()

    text_proj = build_text_projector(args, device)
    cfg = loss_config(args)
    preflight_batch_size(autoencoder, text_proj, train_ds, args, device, cfg)

    params = [p for p in list(text_proj.parameters()) + list(autoencoder.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (run_dir / "preprocessing_config.json").open("w") as f:
        json.dump({**payload["config"], "metadata": payload["metadata"]}, f, indent=2)

    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_state = None
    bad_checks = 0
    for epoch in range(args.epochs):
        start = time.perf_counter()
        train_metrics = run_epoch(autoencoder, text_proj, train_loader, optimizer, scaler, args, device, cfg, train=True)
        val_metrics = {}
        if epoch == 0 or epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            val_metrics = run_epoch(autoencoder, text_proj, val_loader, optimizer, scaler, args, device, cfg, train=False)
            score = val_metrics.get("top5_dice", 0.0) + val_metrics.get("spatial_corr", 0.0)
            state = {
                "text_proj": deepcopy(text_proj.state_dict()),
                "autoencoder": deepcopy(autoencoder.state_dict()),
                "epoch": epoch,
                "metrics": val_metrics,
                "history": history,
                "config": vars(args),
                "target_shape": ds.input_shape,
            }
            if score > best_score:
                best_score = score
                best_state = state
                torch.save(state, ckpt_dir / "best_text_to_brain_decoder.pt")
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
                "text_proj": text_proj.state_dict(),
                "autoencoder": autoencoder.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "history": history,
                "config": vars(args),
                "target_shape": ds.input_shape,
            },
            ckpt_dir / "last_text_to_brain_decoder.pt",
        )
        with (run_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)
        print(row, flush=True)
        if args.early_stopping_patience and bad_checks >= args.early_stopping_patience:
            print(f"Early stopping after {bad_checks} validation checks.", flush=True)
            break

    if best_state is not None:
        text_proj.load_state_dict(best_state["text_proj"])
        autoencoder.load_state_dict(best_state["autoencoder"])
    final_metrics = {
        "model": "ale_text_to_brain_decoder",
        "autoencoder_checkpoint": str(args.autoencoder_checkpoint),
        "text_proj_checkpoint": str(args.text_proj_checkpoint or ""),
        "text_params": count_parameters(text_proj),
        "val": run_epoch(autoencoder, text_proj, val_loader, optimizer, scaler, args, device, cfg, train=False),
        "test": run_epoch(autoencoder, text_proj, test_loader, optimizer, scaler, args, device, cfg, train=False),
    }
    with (run_dir / "text_to_brain_generation_metrics.json").open("w") as f:
        json.dump(final_metrics, f, indent=2)
    rows = []
    for split, metrics in [("val", final_metrics["val"]), ("test", final_metrics["test"])]:
        rows.append({"split": split, **metrics})
    import pandas as pd

    pd.DataFrame(rows).to_csv(run_dir / "text_to_brain_generation_metrics.csv", index=False)
    save_generation_examples(autoencoder, text_proj, test_ds, args, device, run_dir / "plots")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
