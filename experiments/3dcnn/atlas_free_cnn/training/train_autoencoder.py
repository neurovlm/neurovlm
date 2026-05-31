"""Stage 1: stable atlas-free CNN autoencoder pretraining.

This is the default Stage 1 trainer. It intentionally uses the recovered
previous-good recipe: raw decoder output, plain MSE reconstruction loss, and
clamping only for evaluation/plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
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
from atlas_free_cnn.pipeline_outputs import git_info
from atlas_free_cnn.training.autoencoder_losses import AutoencoderLossConfig, reconstruction_loss
from atlas_free_cnn.training.checkpointing import CheckpointManager
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset
from atlas_free_cnn.training.model_wrappers import build_cnn_autoencoder
from atlas_free_cnn.training.source_sampling import (
    build_source_sampler,
    canonical_source,
    epoch_source_exposure,
    source_counts as count_sources,
    source_detail,
)
from neurovlm.gnn.ale_cnn import count_parameters


AUTOENCODER_BATCH_CANDIDATES = [1024, 768, 512, 384, 256, 192, 128, 96, 64]
MODEL_SIZE_PRESETS = {
    "base": {"base_channels": 48, "num_blocks": 4, "latent_dim": 384},
    "wide": {"base_channels": 64, "num_blocks": 4, "latent_dim": 384},
    "deeper": {"base_channels": 48, "num_blocks": 5, "latent_dim": 384},
}

BASELINE_RAW_MSE_RECIPE = "baseline_raw_mse"
LEGACY_BASELINE_RECIPE = "previous_good_compatible"

BASELINE_RAW_MSE_DEFAULTS = {
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "amp": True,
    "gradient_clipping": 1.0,
    "target_shape": [36, 45, 38],
    "source_sampling": "natural",
    "loss": {
        "type": "raw_mse",
        "lambda_foreground": 0.0,
        "lambda_topk": 0.0,
        "prediction_activation": "none",
    },
    "model": {
        "latent_dim": 384,
        "base_channels": 64,
        "num_blocks": 4,
        "dropout": 0.1,
        "norm": "group",
        "pooling": "max",
    },
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _target_shape(cfg: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(v) for v in cfg.get("target_shape", [36, 45, 38]))


def filter_data_mode(dataset: UnifiedMapTextDataset, data_mode: str) -> UnifiedMapTextDataset:
    if data_mode not in {"pubmed_only", "mixed", "statmaps_only"}:
        raise ValueError("DATA_MODE/data_mode must be 'pubmed_only', 'mixed', or 'statmaps_only'")
    if data_mode == "pubmed_only":
        dataset.rows = [row for row in dataset.rows if canonical_source(row) == "pubmed"]
    elif data_mode == "statmaps_only":
        dataset.rows = [row for row in dataset.rows if canonical_source(row) in {"neurovault", "nilearn"}]
    return dataset


def source_counts(dataset: UnifiedMapTextDataset) -> dict[str, int]:
    return count_sources(dataset.rows)


def apply_ae_recipe_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    recipe = str(cfg.get("ae_training_recipe", cfg.get("AE_TRAINING_RECIPE", BASELINE_RAW_MSE_RECIPE)))
    legacy_alias_used = recipe == LEGACY_BASELINE_RECIPE
    if legacy_alias_used:
        recipe = BASELINE_RAW_MSE_RECIPE
    cfg["ae_training_recipe"] = recipe
    if legacy_alias_used:
        cfg["ae_training_recipe_alias"] = LEGACY_BASELINE_RECIPE
    if recipe == BASELINE_RAW_MSE_RECIPE:
        for key, value in BASELINE_RAW_MSE_DEFAULTS.items():
            if key == "model":
                model = dict(value)
                model.update(dict(cfg.get("model") or {}))
                cfg["model"] = model
            elif key == "loss":
                loss = dict(value)
                loss.update(dict(cfg.get("loss") or {}))
                cfg["loss"] = loss
            else:
                cfg.setdefault(key, value)
        cfg.setdefault("checkpoint_selection_metric", "best_val_loss")
    elif recipe in {"mixed_balanced_raw_mse", "mixed_balanced_hybrid_loss", "mixed_hybrid"}:
        cfg.setdefault("data_mode", "mixed")
        cfg.setdefault("source_sampling", "balanced")
        if recipe == "mixed_balanced_raw_mse":
            cfg.setdefault("loss", {"type": "raw_mse"})
        else:
            cfg.setdefault(
                "loss",
                {
                    "type": "hybrid_recon",
                    "lambda_foreground": 0.10,
                    "lambda_topk": 0.05,
                    "topk_percent": 5,
                    "prediction_activation": "none",
                },
            )
        cfg.setdefault("checkpoint_selection_metric", "best_top5_dice")
    return cfg


class VolumeCollator:
    def __init__(self, target_shape: tuple[int, int, int]):
        self.target_shape = target_shape

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        vols = []
        for item in batch:
            v = item["volume"].float()
            if tuple(v.shape[-3:]) != self.target_shape:
                v = F.interpolate(
                    v.unsqueeze(0),
                    size=self.target_shape,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
            vols.append(v.clamp(0.0, 1.0))
        metadata = [b["metadata"] for b in batch]
        return {
            "volume": torch.stack(vols),
            "map_id": [b["map_id"] for b in batch],
            "metadata": metadata,
            "source": [canonical_source(m) for m in metadata],
            "source_detail": [source_detail(m) for m in metadata],
        }


def model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    size = str(cfg.get("model_size", cfg.get("MODEL_SIZE", "base"))).lower()
    if size not in MODEL_SIZE_PRESETS:
        raise ValueError("MODEL_SIZE/model_size must be base, wide, or deeper")
    out = dict(MODEL_SIZE_PRESETS[size])
    out.update(cfg.get("model", {}))
    out.setdefault("encoder_arch", "plain")
    out.setdefault("dropout", 0.1)
    out.setdefault("norm", "group")
    out.setdefault("pooling", "max")
    out.setdefault("blocks_per_stage", 2)
    out.setdefault("use_dilation", False)
    out.setdefault("multi_scale", False)
    out.setdefault("global_context", "none")
    return out


def build_model(cfg: dict[str, Any], target_shape: tuple[int, int, int], device: torch.device):
    mcfg = model_config(cfg)
    model = build_cnn_autoencoder(
        target_shape,
        latent_dim=int(mcfg["latent_dim"]),
        base_channels=int(mcfg["base_channels"]),
        num_blocks=int(mcfg["num_blocks"]),
        dropout=float(mcfg.get("dropout", 0.1)),
        norm=str(mcfg.get("norm", "group")),
        pooling=str(mcfg.get("pooling", "max")),
        encoder_arch=str(mcfg.get("encoder_arch", "plain")),
        blocks_per_stage=int(mcfg.get("blocks_per_stage", 2)),
        use_dilation=bool(mcfg.get("use_dilation", False)),
        multi_scale=bool(mcfg.get("multi_scale", False)),
        global_context=str(mcfg.get("global_context", "none")),
    ).to(device)
    return model


def preflight_batch_size(
    model,
    target_shape: tuple[int, int, int],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    requested = int(cfg.get("batch_size", 64))
    if not bool(cfg.get("preflight_batch_size", True)) or device.type != "cuda":
        return {"selected_batch_size": requested, "peak_vram_gb": None, "parameter_count": count_parameters(model)}
    reserve_gb = float(cfg.get("preflight_vram_reserve_gb", 12.0))
    candidates = [int(v) for v in cfg.get("batch_candidates", AUTOENCODER_BATCH_CANDIDATES)]
    if cfg.get("max_batch_size") is not None:
        max_batch_size = int(cfg["max_batch_size"])
        candidates = [v for v in candidates if v <= max_batch_size]
    candidates = [v for v in candidates if v >= requested] + [v for v in candidates if v < requested]
    candidates = sorted(set(candidates), reverse=True)
    param_count = count_parameters(model)
    selected = min(candidates) if candidates else requested
    selected_peak = None
    was_training = model.training
    model.train()
    for batch_size in candidates:
        x = pred = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.rand((batch_size, 1, *target_shape), device=device)
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True))):
                pred = model(x)
                loss = F.mse_loss(pred, x)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            free, total = torch.cuda.mem_get_info(device)
            free_gb = free / 1024**3
            if free_gb >= reserve_gb:
                selected = batch_size
                selected_peak = peak
                break
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            model.zero_grad(set_to_none=True)
            del x, pred, loss
            torch.cuda.empty_cache()
    model.train(was_training)
    return {"selected_batch_size": selected, "peak_vram_gb": selected_peak, "parameter_count": param_count}


def _avg_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(sum(r[k] for r in rows) / max(1, len(rows))) for k in keys}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_history_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(_json_ready(value))
        else:
            flat[key] = value
    return flat


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    use_amp: bool,
    loss_config: AutoencoderLossConfig | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
    max_batches: int | None = None,
    compute_metrics: bool = True,
    metric_max_batches: int | None = None,
    include_voxel_auroc: bool = False,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    model.train(train)
    losses: list[float] = []
    loss_parts: dict[str, list[float]] = defaultdict(list)
    metric_rows: list[dict[str, float]] = []
    source_metric_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    source_counter: Counter[str] = Counter()
    total = min(len(loader), int(max_batches)) if max_batches is not None else len(loader)
    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, total=total, desc=progress_desc or ("train" if train else "val"), unit="batch", leave=False)
    start = time.time()
    for step, batch in enumerate(iterator):
        if max_batches is not None and step >= int(max_batches):
            break
        x = batch["volume"].to(device, non_blocking=True)
        source_counter.update(batch["source"])
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
                pred = model(x)
                loss, parts = reconstruction_loss(pred, x, loss_config)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for key, value in parts.items():
            loss_parts[key].append(float(value.detach().cpu()))
        if compute_metrics and (metric_max_batches is None or len(metric_rows) < int(metric_max_batches)):
            pred_cpu = pred.detach().clamp(0.0, 1.0).cpu()
            x_cpu = x.detach().cpu()
            metric_rows.append(generation_metrics(pred_cpu, x_cpu, include_voxel_auroc=include_voxel_auroc))
            for i, src in enumerate(batch["source"]):
                source_metric_rows[src].append(
                    generation_metrics(
                        pred_cpu[i : i + 1],
                        x_cpu[i : i + 1],
                        include_voxel_auroc=include_voxel_auroc,
                    )
                )
        if show_progress and tqdm is not None:
            iterator.set_postfix(mse=f"{losses[-1]:.5f}")
    metrics = _avg_metric_rows(metric_rows)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    for key, values in loss_parts.items():
        metrics[f"loss_component_{key}"] = float(sum(values) / max(1, len(values)))
    metrics["epoch_time_sec"] = float(time.time() - start)
    metrics["source_counts"] = dict(source_counter)
    metrics["source_metrics"] = {
        src: _avg_metric_rows(rows) for src, rows in sorted(source_metric_rows.items())
    }
    if device.type == "cuda":
        metrics["peak_vram_gb"] = float(torch.cuda.max_memory_allocated(device) / 1024**3)
    return metrics


@torch.no_grad()
def evaluate_by_source(
    model,
    dataset: UnifiedMapTextDataset,
    cfg: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> list[dict[str, Any]]:
    collate = VolumeCollator(_target_shape(cfg))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 64))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=bool(cfg.get("pin_memory", device.type == "cuda")),
    )
    rows_by_key: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    model.eval()
    for batch in loader:
        x = batch["volume"].to(device, non_blocking=True)
        pred = model(x).clamp(0.0, 1.0).cpu()
        target = x.cpu()
        for i, (src, detail) in enumerate(zip(batch["source"], batch["source_detail"])):
            metrics = generation_metrics(pred[i : i + 1], target[i : i + 1], include_voxel_auroc=bool(cfg.get("include_voxel_auroc", True)))
            rows_by_key[("all", "ALL_DETAILS")].append(metrics)
            rows_by_key[(src, detail)].append(metrics)
            rows_by_key[(src, "ALL_DETAILS")].append(metrics)
    out = []
    for (src, detail), metric_rows in sorted(rows_by_key.items()):
        row = {"split": split_name, "source": src, "source_detail": detail, "n": len(metric_rows)}
        row.update(_avg_metric_rows(metric_rows))
        out.append(row)
    return out


@torch.no_grad()
def save_qualitative_plots(model, dataset, cfg: dict[str, Any], device: torch.device, out_dir: Path, source: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    rows = [i for i, row in enumerate(dataset.rows) if canonical_source(row) == source]
    if not rows:
        return
    rng = random.Random(int(cfg.get("seed", 42)))
    chosen = rng.sample(rows, k=min(int(cfg.get("plot_examples_per_source", 3)), len(rows)))
    collate = VolumeCollator(_target_shape(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for kind in ["random", "peak_centered", "mip"]:
        fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)), squeeze=False)
        for r, idx in enumerate(chosen):
            batch = collate([dataset[idx]])
            x = batch["volume"].to(device)
            pred = model(x).clamp(0.0, 1.0).cpu()[0, 0]
            target = x.cpu()[0, 0]
            if kind == "mip":
                true_img = target.max(dim=2).values
                pred_img = pred.max(dim=2).values
            else:
                peak = int(target.flatten().argmax().item()) if kind == "peak_centered" else target.numel() // 2
                z = int(torch.unravel_index(torch.tensor(peak), target.shape)[2].item())
                true_img = target[:, :, z]
                pred_img = pred[:, :, z]
            diff_img = (pred_img - true_img).abs()
            for ax, img, title in zip(axes[r], [true_img, pred_img, diff_img], ["true", "recon", "abs diff"]):
                ax.imshow(img.T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
                ax.set_title(title)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{source}_{kind}.png", dpi=150)
        plt.close(fig)


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = apply_ae_recipe_defaults(cfg)
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    target_shape = _target_shape(cfg)
    data_mode = str(cfg.get("data_mode", cfg.get("DATA_MODE", "mixed"))).lower()
    out_dir = Path(cfg.get("output_dir", "experiments/3dcnn/atlas_free_cnn/outputs/runs/autoencoder_stable"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir = out_dir / "config"
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    checkpoint_dir = Path(cfg.get("checkpoint_dir", str(out_dir / "checkpoints")))
    for path in [config_dir, metrics_dir, plots_dir, checkpoint_dir]:
        path.mkdir(parents=True, exist_ok=True)
    train_ds = filter_data_mode(UnifiedMapTextDataset(cfg["train_jsonl"]), data_mode)
    val_ds = filter_data_mode(UnifiedMapTextDataset(cfg["val_jsonl"]), data_mode)
    test_ds = filter_data_mode(UnifiedMapTextDataset(cfg["test_jsonl"]), data_mode) if cfg.get("test_jsonl") else None
    split_counts = {
        "train": source_counts(train_ds),
        "val": source_counts(val_ds),
        "test": source_counts(test_ds) if test_ds is not None else {},
    }
    with (out_dir / "source_counts_by_split.json").open("w") as f:
        json.dump(split_counts, f, indent=2)
    split_rows = []
    for split, counts in split_counts.items():
        for src, n in counts.items():
            split_rows.append({"split": split, "source": src, "n": n})
    _write_csv(metrics_dir / "source_counts_by_split.csv", split_rows)
    print({"data_mode": data_mode, "source_counts_by_split": split_counts})

    model = build_model(cfg, target_shape, device)
    preflight = preflight_batch_size(model, target_shape, cfg, device)
    batch_size = int(preflight["selected_batch_size"])
    cfg = dict(cfg)
    cfg["batch_size"] = batch_size
    loss_cfg = AutoencoderLossConfig.from_config(cfg)
    cfg["loss"] = loss_cfg.to_dict()
    cfg["prediction_activation"] = loss_cfg.prediction_activation
    cfg["model"] = model_config(cfg)
    cfg["data_mode"] = data_mode
    cfg["preflight"] = preflight
    cfg["git_info"] = git_info(Path(__file__).resolve().parents[4])
    with (out_dir / "autoencoder_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    with (config_dir / "ae_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    with (config_dir / "loss_config.json").open("w") as f:
        json.dump(loss_cfg.to_dict(), f, indent=2)
    with (out_dir / "preflight.json").open("w") as f:
        json.dump(preflight, f, indent=2)

    num_workers = int(cfg.get("num_workers", 0))
    collate = VolumeCollator(target_shape)
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": bool(cfg.get("pin_memory", device.type == "cuda")),
        "persistent_workers": bool(cfg.get("persistent_workers", num_workers > 0)) if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))
    sampler, sampler_cfg = build_source_sampler(train_ds.rows, cfg) if data_mode == "mixed" else (None, {})
    with (config_dir / "sampler_config.json").open("w") as f:
        json.dump(sampler_cfg, f, indent=2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    init_checkpoint = cfg.get("init_checkpoint") or cfg.get("ae_init_checkpoint")
    if init_checkpoint:
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model") or payload.get("autoencoder") or payload.get("state_dict")
        if state is None:
            raise KeyError("init_checkpoint must contain model, autoencoder, or state_dict")
        model.load_state_dict(state, strict=True)
        freeze = str(cfg.get("freeze_mode", "none")).lower()
        if freeze == "encoder":
            for p in model.encoder.parameters():
                p.requires_grad_(False)
        elif freeze == "decoder":
            for p in model.decoder.parameters():
                p.requires_grad_(False)
        elif freeze == "all_but_decoder":
            for p in model.encoder.parameters():
                p.requires_grad_(False)
        elif freeze in {"none", "train_all", "encoder_decoder"}:
            pass
        else:
            raise ValueError("freeze_mode must be none, encoder, decoder, all_but_decoder, train_all, or encoder_decoder")
    ckpt = CheckpointManager(
        checkpoint_dir,
        maximize={
            "val_loss": False,
            "spatial_corr": True,
            "top1_dice": True,
            "top5_dice": True,
            "foreground_mse": False,
        },
    )
    use_amp = bool(cfg.get("amp", device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and device.type == "cuda"))
    history: list[dict[str, Any]] = []
    sampling_history: list[dict[str, Any]] = []
    last_val_metrics: dict[str, Any] = {}
    max_epochs = int(cfg.get("epochs", 200))
    early_stopping = bool(cfg.get("early_stopping", True))
    early_metric = str(cfg.get("early_stopping_metric", "val_loss"))
    early_patience = int(cfg.get("early_stopping_patience", 25))
    early_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    best_early_value = float("inf")
    bad_val_checks = 0
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            use_amp=use_amp,
            loss_config=loss_cfg,
            scaler=scaler,
            grad_clip=float(cfg.get("gradient_clipping", 1.0)),
            max_batches=cfg.get("max_train_batches"),
            compute_metrics=bool(cfg.get("compute_train_metrics", True)),
            metric_max_batches=cfg.get("train_metric_batches"),
            include_voxel_auroc=bool(cfg.get("include_voxel_auroc", False)),
            show_progress=bool(cfg.get("progress", True)),
            progress_desc=f"epoch {epoch} train",
        )
        if epoch == 1 or epoch % int(cfg.get("val_interval", 1)) == 0 or epoch == max_epochs:
            last_val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                use_amp=use_amp,
                loss_config=loss_cfg,
                grad_clip=float(cfg.get("gradient_clipping", 1.0)),
                max_batches=cfg.get("max_val_batches"),
                compute_metrics=True,
                metric_max_batches=cfg.get("val_metric_batches"),
                include_voxel_auroc=bool(cfg.get("include_voxel_auroc", False)),
                show_progress=bool(cfg.get("progress", True)),
                progress_desc=f"epoch {epoch} val",
            )
        val_metrics = dict(last_val_metrics)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        sampling_history.append(epoch_source_exposure(epoch, train_metrics.get("source_counts", {}), sampler_cfg))
        payload = {
            "model": model.state_dict(),
            "config": cfg,
            "history": history,
            "epoch": epoch,
            "train_loss": train_metrics.get("loss"),
            "validation_metrics": val_metrics,
            "source_wise_validation_metrics": val_metrics.get("source_metrics", {}),
            "target_shape": target_shape,
            "model_architecture": cfg["model"],
            "checkpoint_selection_metric": cfg.get("checkpoint_selection_metric", "best_val_loss"),
            "git_info": cfg.get("git_info", {}),
            "data_mode": data_mode,
            "ae_variant": cfg.get("ae_variant", cfg.get("AE_VARIANT", cfg.get("ae_training_recipe"))),
            "source_counts_by_split": split_counts,
            "early_stopping": {
                "enabled": early_stopping,
                "metric": early_metric,
                "patience": early_patience,
                "min_delta": early_min_delta,
                "bad_val_checks": bad_val_checks,
                "best_value": best_early_value,
            },
        }
        ckpt.save_last(payload)
        ckpt.save("last_cnn_autoencoder.pt", payload)
        if val_metrics:
            ckpt.maybe_save_best("val_loss", float(val_metrics.get("loss", math.inf)), payload)
            ckpt.maybe_save_best("spatial_corr", float(val_metrics.get("spatial_corr", -1.0)), payload)
            ckpt.maybe_save_best("top1_dice", float(val_metrics.get("top1_dice", 0.0)), payload)
            ckpt.maybe_save_best("top5_dice", float(val_metrics.get("top5_dice", 0.0)), payload)
            ckpt.maybe_save_best("foreground_mse", float(val_metrics.get("foreground_mse", math.inf)), payload)
            selection = str(cfg.get("checkpoint_selection_metric", "best_val_loss"))
            metric_name = selection.removeprefix("best_")
            if metric_name == "last":
                ckpt.save("best_cnn_autoencoder.pt", payload)
            elif metric_name in ckpt.best:
                selected_value = {
                    "val_loss": float(val_metrics.get("loss", math.inf)),
                    "spatial_corr": float(val_metrics.get("spatial_corr", -1.0)),
                    "top1_dice": float(val_metrics.get("top1_dice", 0.0)),
                    "top5_dice": float(val_metrics.get("top5_dice", 0.0)),
                    "foreground_mse": float(val_metrics.get("foreground_mse", math.inf)),
                }.get(metric_name)
                if selected_value is not None and float(ckpt.best[metric_name]) == float(selected_value):
                    ckpt.save("best_cnn_autoencoder.pt", payload)
            if early_stopping:
                metric_key = early_metric[4:] if early_metric.startswith("val_") else early_metric
                current = float(val_metrics.get(metric_key, math.inf))
                if current < best_early_value - early_min_delta:
                    best_early_value = current
                    bad_val_checks = 0
                else:
                    bad_val_checks += 1
                if bad_val_checks >= early_patience:
                    stop_reason = f"early_stopping:{early_metric}"
                    print({
                        "early_stopping": True,
                        "epoch": epoch,
                        "metric": early_metric,
                        "best_value": best_early_value,
                        "current_value": current,
                        "bad_val_checks": bad_val_checks,
                    })
                    break
        print(row)
    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    _write_csv(metrics_dir / "train_history.csv", [_flatten_history_row(row) for row in history])
    _write_csv(metrics_dir / "val_history.csv", [_flatten_history_row(row) for row in history])
    _write_csv(metrics_dir / "source_sampling_history.csv", sampling_history)
    with (out_dir / "training_stop.json").open("w") as f:
        json.dump(
            {
                "stop_reason": stop_reason,
                "epochs_completed": len(history),
                "early_stopping": early_stopping,
                "early_stopping_metric": early_metric,
                "early_stopping_patience": early_patience,
                "early_stopping_min_delta": early_min_delta,
                "best_early_value": best_early_value,
            },
            f,
            indent=2,
        )

    eval_rows = []
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds is not None and bool(cfg.get("final_eval", True)):
            eval_rows.extend(evaluate_by_source(model, ds, cfg, device, split_name))
    if eval_rows:
        with (out_dir / "autoencoder_reconstruction_metrics.json").open("w") as f:
            json.dump(eval_rows, f, indent=2)
        with (out_dir / "autoencoder_reconstruction_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)
        _write_csv(metrics_dir / "reconstruction_metrics_all_rows.csv", eval_rows)
        _write_csv(metrics_dir / "reconstruction_summary_by_source.csv", eval_rows)
        checkpoint_rows = []
        for row in eval_rows:
            checkpoint_rows.append(
                {
                    "checkpoint": "last",
                    "ae_variant": cfg.get("ae_variant", cfg.get("ae_training_recipe")),
                    **row,
                }
            )
        _write_csv(metrics_dir / "reconstruction_summary_by_checkpoint_source.csv", checkpoint_rows)
    leaderboard_rows = []
    for name, value in sorted(ckpt.best.items()):
        leaderboard_rows.append(
            {
                "metric": name,
                "best_value": value,
                "checkpoint_path": str(ckpt.out_dir / f"best_{name}.pt"),
                "maximize": ckpt.maximize.get(name, True),
            }
        )
    _write_csv(metrics_dir / "checkpoint_leaderboard.csv", leaderboard_rows)
    if bool(cfg.get("save_plots", True)):
        plot_ds = test_ds or val_ds
        for src in ["pubmed", "neurovault", "nilearn"]:
            save_qualitative_plots(model, plot_ds, cfg, device, plots_dir / "recon_examples" / src, src)
    return {
        "history": history,
        "checkpoint_dir": str(ckpt.out_dir),
        "best_checkpoint": str(ckpt.out_dir / "best_cnn_autoencoder.pt"),
        "checkpoint_leaderboard": leaderboard_rows,
        "preflight": preflight,
        "source_counts_by_split": split_counts,
    }


def train_stage1b_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run domain-specific AE fine-tuning from a mixed pretraining checkpoint."""

    mode = str(cfg.get("stage1b_mode", cfg.get("STAGE1B_MODE", ""))).lower()
    if mode not in {"mixed_pretrain_to_pubmed", "mixed_pretrain_to_statmaps"}:
        raise ValueError("stage1b_mode must be mixed_pretrain_to_pubmed or mixed_pretrain_to_statmaps")
    ft_cfg = dict(cfg)
    ft_cfg["ae_training_recipe"] = cfg.get("ae_training_recipe", BASELINE_RAW_MSE_RECIPE)
    ft_cfg["init_checkpoint"] = cfg.get("init_checkpoint") or cfg.get("mixed_pretrain_checkpoint")
    if not ft_cfg.get("init_checkpoint"):
        raise ValueError("Stage 1B fine-tuning requires init_checkpoint or mixed_pretrain_checkpoint")
    ft_cfg.setdefault("lr", 1e-4)
    ft_cfg.setdefault("freeze_mode", "none")
    ft_cfg["data_mode"] = "pubmed_only" if mode == "mixed_pretrain_to_pubmed" else "statmaps_only"
    ft_cfg["ae_variant"] = cfg.get(
        "ae_variant",
        "mixed_to_pubmed" if mode == "mixed_pretrain_to_pubmed" else "mixed_to_statmaps",
    )
    ft_cfg["stage1b_mode"] = mode
    return train_from_config(ft_cfg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/3dcnn/atlas_free_cnn/configs/autoencoder_config.yaml")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if cfg.get("stage1b_mode") or cfg.get("STAGE1B_MODE"):
        train_stage1b_from_config(cfg)
    else:
        train_from_config(cfg)


if __name__ == "__main__":
    main()
