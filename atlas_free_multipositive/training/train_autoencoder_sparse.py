"""Stage 1: sparse-aware CNN autoencoder pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from atlas_free_multipositive.evaluation.generation_metrics import generation_metrics
from atlas_free_multipositive.training.checkpointing import CheckpointManager
from atlas_free_multipositive.training.datasets import UnifiedMapTextDataset
from atlas_free_multipositive.training.generation_losses import GenerationLossConfig, combined_generation_loss
from atlas_free_multipositive.training.model_wrappers import build_cnn_autoencoder


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _target_shape(cfg: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(v) for v in cfg.get("target_shape", [36, 45, 38]))


class VolumeCollator:
    def __init__(self, target_shape: tuple[int, int, int]):
        self.target_shape = target_shape

    def __call__(self, batch):
        vols = []
        for item in batch:
            v = item["volume"].float()
            if tuple(v.shape[-3:]) != self.target_shape:
                v = F.interpolate(v.unsqueeze(0), size=self.target_shape, mode="trilinear", align_corners=False).squeeze(0)
            vols.append(v)
        return {"volume": torch.stack(vols), "map_id": [b["map_id"] for b in batch], "metadata": [b["metadata"] for b in batch]}


def loss_config_from_dict(cfg: dict[str, Any]) -> GenerationLossConfig:
    loss = cfg.get("loss", {})
    recon = cfg.get("weighted_recon", {})
    return GenerationLossConfig(
        lambda_recon=float(loss.get("lambda_recon", 1.0)),
        lambda_dice=float(loss.get("lambda_dice", 0.5)),
        lambda_topk=float(loss.get("lambda_topk", 0.5)),
        lambda_corr=float(loss.get("lambda_corr", 0.25)),
        lambda_latent=0.0,
        recon_type=str(recon.get("type", "mse")),
        recon_alpha=float(recon.get("alpha", 10.0)),
        recon_gamma=float(recon.get("gamma", 1.0)),
        prediction_activation=str(cfg.get("prediction_activation", "sigmoid")),
    )


def run_epoch(model, loader, optimizer, device, loss_cfg, *, train: bool, max_batches: int | None = None):
    model.train(train)
    losses = []
    metric_rows = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x = batch["volume"].to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss, parts = combined_generation_loss(pred, x, config=loss_cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            metric_rows.append(generation_metrics(torch.sigmoid(pred.detach().cpu()), x.detach().cpu()))
    avg_metrics = {k: float(sum(r[k] for r in metric_rows) / max(1, len(metric_rows))) for k in metric_rows[0]} if metric_rows else {}
    avg_metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return avg_metrics


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    target_shape = _target_shape(cfg)
    train_ds = UnifiedMapTextDataset(cfg["train_jsonl"])
    val_ds = UnifiedMapTextDataset(cfg["val_jsonl"])
    collate = VolumeCollator(target_shape)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 8)), shuffle=True, num_workers=int(cfg.get("num_workers", 0)), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 8)), shuffle=False, num_workers=int(cfg.get("num_workers", 0)), collate_fn=collate)
    model_cfg = cfg.get("model", {})
    model = build_cnn_autoencoder(
        target_shape,
        latent_dim=int(model_cfg.get("latent_dim", 384)),
        base_channels=int(model_cfg.get("base_channels", 8)),
        num_blocks=int(model_cfg.get("num_blocks", 2)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    loss_cfg = loss_config_from_dict(cfg)
    ckpt = CheckpointManager(cfg.get("checkpoint_dir", "atlas_free_multipositive/outputs/runs/sparse_autoencoder/checkpoints"))
    history = []
    for epoch in range(1, int(cfg.get("epochs", 3)) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, loss_cfg, train=True, max_batches=cfg.get("max_train_batches"))
        val_metrics = run_epoch(model, val_loader, optimizer, device, loss_cfg, train=False, max_batches=cfg.get("max_val_batches"))
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        payload = {"model": model.state_dict(), "config": cfg, "history": history, "epoch": epoch, "target_shape": target_shape}
        ckpt.save_last(payload)
        ckpt.maybe_save_best("generation_top5_dice", val_metrics.get("top5_dice", 0.0), payload)
        ckpt.maybe_save_best("generation_spatial_correlation", val_metrics.get("spatial_corr", -1.0), payload)
        ckpt.maybe_save_best("validation_combined_generation_score", val_metrics.get("top5_dice", 0.0) + val_metrics.get("spatial_corr", 0.0), payload)
        print(row)
    out_dir = Path(cfg.get("output_dir", "atlas_free_multipositive/outputs/runs/sparse_autoencoder"))
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    return {"history": history, "checkpoint_dir": str(ckpt.out_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="atlas_free_multipositive/configs/autoencoder_sparse_config.yaml")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
