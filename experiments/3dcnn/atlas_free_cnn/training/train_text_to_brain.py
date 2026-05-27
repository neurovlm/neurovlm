"""Stage 4: train a text-to-brain projection through a frozen AE decoder.

This is separate from the contrastive text projection: it maps SPECTER/SPECTER2
embeddings into the AE decoder latent space.
"""

from __future__ import annotations

import argparse
import csv
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

from atlas_free_cnn.evaluation.generation_metrics import generation_metrics
from atlas_free_cnn.training.checkpointing import CheckpointManager
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset
from atlas_free_cnn.training.generation_losses import GenerationLossConfig
from atlas_free_cnn.training.generation_losses import (
    latent_alignment_loss,
    soft_dice_loss,
    spatial_correlation_loss,
    topk_overlap_loss,
    weighted_reconstruction_loss,
)
from atlas_free_cnn.training.model_wrappers import (
    build_cnn_autoencoder,
    build_text_to_brain_projection,
    load_autoencoder_checkpoint,
)
from neurovlm.gnn.ale_cnn import count_parameters


TEXT_TO_BRAIN_BATCH_CANDIDATES = [4096, 3072, 2048, 1536, 1024, 768, 512]


def canonical_source(row: dict[str, Any]) -> str:
    source = str(row.get("source", "")).lower()
    if source == "pubmed" or row.get("pmid"):
        return "pubmed"
    if source.startswith("neurovault"):
        return "neurovault"
    if source.startswith("nilearn"):
        return "nilearn"
    if source.startswith("network"):
        return "networks"
    return source or "unknown"


class PrimaryTextVolumeCollator:
    """Use exactly one primary text per map for text-to-brain training."""

    def __init__(self, target_shape: tuple[int, int, int], *, text_rank: int = 0):
        self.target_shape = target_shape
        self.text_rank = int(text_rank)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        volumes = []
        texts = []
        text_entries = []
        kept = []
        for item in batch:
            positives = item.get("positive_texts", []) or []
            if not positives:
                continue
            pos = positives[min(self.text_rank, len(positives) - 1)]
            v = item["volume"].float()
            if tuple(v.shape[-3:]) != self.target_shape:
                v = F.interpolate(v.unsqueeze(0), size=self.target_shape, mode="trilinear", align_corners=False).squeeze(0)
            volumes.append(v.clamp(0.0, 1.0))
            texts.append(pos["text"])
            text_entries.append(pos)
            kept.append(item)
        if not volumes:
            raise ValueError("Batch contains no rows with positive_texts")
        n = len(volumes)
        return {
            "volume": torch.stack(volumes),
            "map_id": [b["map_id"] for b in kept],
            "texts": texts,
            "text_entries": text_entries,
            "pos_mask": torch.eye(n, dtype=torch.bool),
            "pos_weights": torch.ones((n, n), dtype=torch.float32),
            "metadata": [b["metadata"] for b in kept],
        }


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
        lambda_dice=float(loss.get("lambda_dice", 0.0)),
        lambda_topk=float(loss.get("lambda_topk", 0.0)),
        lambda_corr=float(loss.get("lambda_corr", 0.0)),
        recon_type=str(recon.get("type", "mse")),
        recon_alpha=float(recon.get("alpha", 0.0)),
        recon_gamma=float(recon.get("gamma", 1.0)),
        prediction_activation=str(cfg.get("prediction_activation", "none")),
    )


def _load_text_cache(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload.items()}


def _lookup(cache: dict[str, torch.Tensor], texts: list[str]) -> torch.Tensor:
    missing = [t for t in texts if t not in cache]
    if missing:
        raise KeyError(f"{len(missing)} texts missing from embedding cache; example={missing[0][:160]}")
    return torch.stack([cache[t] for t in texts])


def text_to_brain_loss(pred, target, brain_z, text_z, loss_cfg: GenerationLossConfig):
    parts = {
        "latent_alignment": latent_alignment_loss(text_z, brain_z, loss_type="cosine", detach_brain_z=True),
        "recon_mse": F.mse_loss(pred, target),
    }
    total = loss_cfg.lambda_latent * parts["latent_alignment"] + loss_cfg.lambda_recon * parts["recon_mse"]
    if loss_cfg.recon_alpha > 0:
        parts["positive_weighted_mse"] = weighted_reconstruction_loss(
            pred,
            target,
            loss_type=loss_cfg.recon_type,
            alpha=loss_cfg.recon_alpha,
            gamma=loss_cfg.recon_gamma,
        )
        total = total + parts["positive_weighted_mse"]
    if loss_cfg.lambda_dice:
        parts["soft_dice"] = soft_dice_loss(pred, target)
        total = total + loss_cfg.lambda_dice * parts["soft_dice"]
    if loss_cfg.lambda_topk:
        parts["topk_overlap"] = topk_overlap_loss(pred, target)
        total = total + loss_cfg.lambda_topk * parts["topk_overlap"]
    if loss_cfg.lambda_corr:
        parts["spatial_corr"] = spatial_correlation_loss(pred, target)
        total = total + loss_cfg.lambda_corr * parts["spatial_corr"]
    parts["total"] = total
    return total, parts


def preflight_batch_size(autoencoder, text_projector, target_shape, cfg, device) -> dict[str, Any]:
    requested = int(cfg.get("batch_size", 256))
    if not bool(cfg.get("preflight_batch_size", True)) or device.type != "cuda":
        return {
            "selected_batch_size": requested,
            "peak_vram_gb": None,
            "parameter_count": count_parameters(text_projector),
        }
    candidates = sorted(set(int(v) for v in cfg.get("batch_candidates", TEXT_TO_BRAIN_BATCH_CANDIDATES)), reverse=True)
    selected = min(candidates) if candidates else requested
    selected_peak = None
    reserve_gb = float(cfg.get("preflight_vram_reserve_gb", 12.0))
    autoencoder.eval()
    text_projector.train()
    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.rand((batch_size, 1, *target_shape), device=device)
            raw_text = torch.randn((batch_size, 768), device=device)
            with torch.no_grad():
                brain_z = autoencoder.encoder(x)
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True))):
                text_z = text_projector(raw_text)
                pred = autoencoder.decoder(text_z)
                loss = F.mse_loss(pred, x) + latent_alignment_loss(text_z, brain_z)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            free, _ = torch.cuda.mem_get_info(device)
            if free / 1024**3 >= reserve_gb:
                selected = batch_size
                selected_peak = peak
                break
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            text_projector.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    return {
        "selected_batch_size": selected,
        "peak_vram_gb": selected_peak,
        "parameter_count": count_parameters(text_projector),
    }


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
                loss, _ = text_to_brain_loss(pred, target, brain_z, text_z, loss_cfg)
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
            pred_metric = pred.detach().clamp(0.0, 1.0)
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


@torch.no_grad()
def evaluate_generation_dataset(
    autoencoder,
    text_projector,
    dataset: UnifiedMapTextDataset,
    text_cache: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    loss_cfg: GenerationLossConfig,
    dataset_name: str,
    *,
    metrics_device: torch.device,
) -> list[dict[str, Any]]:
    sources = sorted({canonical_source(row) for row in dataset.rows})
    rows: list[dict[str, Any]] = []
    for source in ["all", *sources]:
        source_ds = dataset
        if source != "all":
            source_ds = UnifiedMapTextDataset(dataset.path)
            source_ds.rows = [row for row in dataset.rows if canonical_source(row) == source]
            source_ds._tensor_cache = dataset._tensor_cache
        if len(source_ds) == 0:
            continue
        collator = PrimaryTextVolumeCollator(_target_shape(cfg), text_rank=int(cfg.get("primary_text_rank", 0)))
        loader = DataLoader(
            source_ds,
            batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 256))),
            shuffle=False,
            num_workers=int(cfg.get("eval_num_workers", 0)),
            collate_fn=collator,
            pin_memory=bool(cfg.get("pin_memory", device.type == "cuda")),
        )
        metrics = run_epoch(
            autoencoder,
            text_projector,
            loader,
            text_cache,
            optimizer=None,
            device=device,
            loss_cfg=loss_cfg,
            train=False,
            max_batches=cfg.get("max_test_batches"),
            compute_metrics=True,
            metric_max_batches=cfg.get("test_metric_batches"),
            use_amp=bool(cfg.get("amp", device.type == "cuda")),
            metrics_device=metrics_device,
            include_voxel_auroc=bool(cfg.get("include_voxel_auroc", True)),
            show_progress=bool(cfg.get("progress", True)),
            progress_desc=f"{dataset_name}:{source} test",
        )
        rows.append({"dataset": dataset_name, "source": source, "n": len(source_ds), **metrics})
    return rows


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
        base_channels=int(model_cfg.get("base_channels", 48)),
        num_blocks=int(model_cfg.get("num_blocks", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        norm=str(model_cfg.get("norm", "group")),
        pooling=str(model_cfg.get("pooling", "max")),
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
    projection_cfg = cfg.get("text_to_brain_projection", {})
    text_projector = build_text_to_brain_projection(
        cfg.get("text_projection_init", "random"),
        device=device,
        hidden_dim=int(projection_cfg.get("hidden_dim", cfg.get("hidden_dim", 512))),
        depth=int(projection_cfg.get("depth", cfg.get("depth", 2))),
        dropout=float(projection_cfg.get("dropout", cfg.get("dropout", 0.1))),
        out_dim=int(model_cfg.get("latent_dim", 384)),
    )
    preflight = preflight_batch_size(autoencoder, text_projector, target_shape, cfg, device)
    cfg = dict(cfg)
    cfg["batch_size"] = int(preflight["selected_batch_size"])
    cfg["preflight"] = preflight
    train_ds = UnifiedMapTextDataset(cfg["train_jsonl"])
    val_ds = UnifiedMapTextDataset(cfg["val_jsonl"])
    collator = PrimaryTextVolumeCollator(target_shape, text_rank=int(cfg.get("primary_text_rank", 0)))
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
    ckpt = CheckpointManager(
        cfg.get("checkpoint_dir", "experiments/3dcnn/atlas_free_cnn/outputs/runs/text_to_brain/checkpoints"),
        maximize={"val_loss": False},
    )
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
            ckpt.maybe_save_best("val_loss", val_metrics.get("loss", float("inf")), payload)
            ckpt.maybe_save_best("generation_top5_dice", val_metrics.get("top5_dice", 0.0), payload)
            ckpt.maybe_save_best("generation_spatial_correlation", val_metrics.get("spatial_corr", -1.0), payload)
        print(row)
    out_dir = Path(cfg.get("output_dir", "experiments/3dcnn/atlas_free_cnn/outputs/runs/text_to_brain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / "text_to_brain_config.json", "w"), indent=2)
    json.dump(preflight, open(out_dir / "preflight.json", "w"), indent=2)
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    best_path = ckpt.out_dir / "best_val_loss.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        text_projector.load_state_dict(best_payload["text_projector"])
        text_projector.eval()
    eval_specs: dict[str, str] = {}
    if cfg.get("test_jsonl"):
        eval_specs["mixed_test"] = str(cfg["test_jsonl"])
    for key, value in (cfg.get("eval_jsonls") or {}).items():
        if value:
            eval_specs[str(key)] = str(value)
    generation_eval_rows: list[dict[str, Any]] = []
    for name, path in eval_specs.items():
        if not Path(path).exists():
            print(f"Skipping text-to-brain generation eval '{name}': missing JSONL {path}")
            continue
        eval_ds = UnifiedMapTextDataset(path)
        generation_eval_rows.extend(
            evaluate_generation_dataset(
                autoencoder,
                text_projector,
                eval_ds,
                text_cache,
                cfg,
                device,
                loss_cfg,
                name,
                metrics_device=metrics_device,
            )
        )
    if generation_eval_rows:
        with (out_dir / "generation_eval_metrics.json").open("w") as f:
            json.dump(generation_eval_rows, f, indent=2)
        with (out_dir / "generation_eval_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(generation_eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(generation_eval_rows)
    return {"history": history, "checkpoint_dir": str(ckpt.out_dir), "best_checkpoint": str(best_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/3dcnn/atlas_free_cnn/configs/text_to_brain_config.yaml")
    args = p.parse_args()
    train_from_config(load_yaml(args.config))


if __name__ == "__main__":
    main()
