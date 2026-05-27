"""Stage 2: train a CNN-specific text projection through a frozen decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from atlas_free_multipositive.evaluation.generation_metrics import generation_metrics
from atlas_free_multipositive.training.checkpointing import CheckpointManager
from atlas_free_multipositive.training.collators import MultiPositiveCollator
from atlas_free_multipositive.training.datasets import UnifiedMapTextDataset
from atlas_free_multipositive.training.generation_losses import GenerationLossConfig, combined_generation_loss
from atlas_free_multipositive.training.model_wrappers import (
    build_cnn_autoencoder,
    build_text_projection,
    load_autoencoder_checkpoint,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _target_shape(cfg: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(v) for v in cfg.get("target_shape", [36, 45, 38]))


def _loss_cfg(cfg: dict[str, Any]) -> GenerationLossConfig:
    loss = cfg.get("loss", {})
    recon = cfg.get("weighted_recon", {})
    return GenerationLossConfig(
        lambda_recon=float(loss.get("lambda_recon", 1.0)),
        lambda_latent=float(loss.get("lambda_latent", 1.0)),
        lambda_dice=float(loss.get("lambda_dice", 0.5)),
        lambda_topk=float(loss.get("lambda_topk", 0.5)),
        lambda_corr=float(loss.get("lambda_corr", 0.25)),
        recon_type=str(recon.get("type", "mse")),
        recon_alpha=float(recon.get("alpha", 10.0)),
        recon_gamma=float(recon.get("gamma", 1.0)),
        prediction_activation=str(cfg.get("prediction_activation", "sigmoid")),
    )


def _load_text_cache(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload.items()}


def _lookup(cache: dict[str, torch.Tensor], texts: list[str]) -> torch.Tensor:
    missing = [t for t in texts if t not in cache]
    if missing:
        raise KeyError(f"{len(missing)} texts missing from embedding cache; example={missing[0][:160]}")
    return torch.stack([cache[t] for t in texts])


def run_epoch(
    autoencoder,
    text_projector,
    loader,
    text_cache,
    optimizer,
    device,
    loss_cfg,
    *,
    train: bool,
    max_batches: int | None = None,
    compute_metrics: bool = True,
    metric_max_batches: int | None = None,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    metrics_device: torch.device | None = None,
    include_voxel_auroc: bool = False,
    show_progress: bool = True,
    progress_desc: str | None = None,
):
    autoencoder.eval()
    text_projector.train(train)
    losses = []
    metric_rows = []
    total = len(loader)
    if max_batches is not None:
        total = min(total, int(max_batches))
    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, total=total, desc=progress_desc or ("train" if train else "val"), unit="batch", leave=False)
    for step, batch in enumerate(iterator):
        if max_batches is not None and step >= max_batches:
            break
        x = batch["volume"].to(device)
        raw_text = _lookup(text_cache, batch["texts"]).to(device)
        owners = batch["pos_mask"].float().argmax(dim=0).to(device)
        target = x[owners]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
                brain_z = autoencoder.encoder(target)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
                text_z = text_projector(raw_text)
                pred = autoencoder.decoder(text_z)
                loss, _ = combined_generation_loss(pred, target, brain_z=brain_z, text_z=text_z, config=loss_cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(text_projector.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(text_projector.parameters(), 1.0)
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if compute_metrics and (metric_max_batches is None or len(metric_rows) < int(metric_max_batches)):
            pred_metric = torch.sigmoid(pred.detach())
            target_metric = target.detach()
            if metrics_device is not None:
                pred_metric = pred_metric.to(metrics_device, non_blocking=True)
                target_metric = target_metric.to(metrics_device, non_blocking=True)
            metric_rows.append(generation_metrics(pred_metric, target_metric, include_voxel_auroc=include_voxel_auroc))
        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{losses[-1]:.4f}")
    avg_metrics = {k: float(sum(r[k] for r in metric_rows) / max(1, len(metric_rows))) for k in metric_rows[0]} if metric_rows else {}
    avg_metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return avg_metrics


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    target_shape = _target_shape(cfg)
    model_cfg = cfg.get("model", {})
    autoencoder = build_cnn_autoencoder(
        target_shape,
        latent_dim=int(model_cfg.get("latent_dim", 384)),
        base_channels=int(model_cfg.get("base_channels", 8)),
        num_blocks=int(model_cfg.get("num_blocks", 2)),
        encoder_arch=str(model_cfg.get("encoder_arch", "plain")),
        blocks_per_stage=int(model_cfg.get("blocks_per_stage", 2)),
        use_dilation=bool(model_cfg.get("use_dilation", False)),
        multi_scale=bool(model_cfg.get("multi_scale", False)),
        global_context=str(model_cfg.get("global_context", "none")),
    ).to(device)
    load_autoencoder_checkpoint(autoencoder, cfg["autoencoder_checkpoint"])
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)
    text_projector = build_text_projection(cfg.get("text_projection_init", "random"), device=device)
    train_ds = UnifiedMapTextDataset(cfg["train_jsonl"])
    val_ds = UnifiedMapTextDataset(cfg["val_jsonl"])
    collator = MultiPositiveCollator(positives_per_map=int(cfg.get("positive_texts_per_map", 1)), target_shape=target_shape, seed=int(cfg.get("seed", 42)))
    num_workers = int(cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": bool(cfg.get("pin_memory", device.type == "cuda")),
        "persistent_workers": bool(cfg.get("persistent_workers", num_workers > 0)) if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 4)), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 4)), shuffle=False, **loader_kwargs)
    text_cache = _load_text_cache(cfg["text_embedding_cache"])
    optimizer = torch.optim.AdamW(text_projector.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    loss_cfg = _loss_cfg(cfg)
    ckpt = CheckpointManager(cfg.get("checkpoint_dir", "atlas_free_multipositive/outputs/runs/text_to_brain/checkpoints"))
    history = []
    use_amp = bool(cfg.get("amp", device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and device.type == "cuda"))
    val_interval = int(cfg.get("val_interval", 1))
    train_metric_batches = cfg.get("train_metric_batches", None)
    val_metric_batches = cfg.get("val_metric_batches", None)
    metrics_device_name = str(cfg.get("metrics_device", "cuda" if device.type == "cuda" else "cpu")).lower()
    metrics_device = torch.device(device if metrics_device_name == "cuda" and device.type == "cuda" else "cpu")
    include_voxel_auroc = bool(cfg.get("include_voxel_auroc", False))
    last_val_metrics: dict[str, float] = {}
    for epoch in range(1, int(cfg.get("epochs", 3)) + 1):
        train_metrics = run_epoch(
            autoencoder,
            text_projector,
            train_loader,
            text_cache,
            optimizer,
            device,
            loss_cfg,
            train=True,
            max_batches=cfg.get("max_train_batches"),
            compute_metrics=bool(cfg.get("compute_train_metrics", True)),
            metric_max_batches=train_metric_batches,
            use_amp=use_amp,
            scaler=scaler,
            metrics_device=metrics_device,
            include_voxel_auroc=include_voxel_auroc,
            show_progress=bool(cfg.get("progress", True)),
            progress_desc=f"epoch {epoch} train",
        )
        if epoch == 1 or epoch % val_interval == 0 or epoch == int(cfg.get("epochs", 3)):
            last_val_metrics = run_epoch(
                autoencoder,
                text_projector,
                val_loader,
                text_cache,
                optimizer,
                device,
                loss_cfg,
                train=False,
                max_batches=cfg.get("max_val_batches"),
                compute_metrics=True,
                metric_max_batches=val_metric_batches,
                use_amp=use_amp,
                metrics_device=metrics_device,
                include_voxel_auroc=include_voxel_auroc,
                show_progress=bool(cfg.get("progress", True)),
                progress_desc=f"epoch {epoch} val",
            )
        val_metrics = dict(last_val_metrics)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        payload = {"text_projector": text_projector.state_dict(), "config": cfg, "history": history, "epoch": epoch, "target_shape": target_shape}
        ckpt.save_last(payload)
        if val_metrics:
            ckpt.maybe_save_best("generation_top5_dice", val_metrics.get("top5_dice", 0.0), payload)
            ckpt.maybe_save_best("generation_spatial_correlation", val_metrics.get("spatial_corr", -1.0), payload)
        print(row)
    out_dir = Path(cfg.get("output_dir", "atlas_free_multipositive/outputs/runs/text_to_brain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    return {"history": history, "checkpoint_dir": str(ckpt.out_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="atlas_free_multipositive/configs/text_to_brain_config.yaml")
    args = p.parse_args()
    train_from_config(load_yaml(args.config))


if __name__ == "__main__":
    main()
