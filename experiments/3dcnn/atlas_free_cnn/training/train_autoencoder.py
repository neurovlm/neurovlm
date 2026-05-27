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
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from atlas_free_cnn.training.model_wrappers import build_cnn_autoencoder
from neurovlm.gnn.ale_cnn import count_parameters


AUTOENCODER_BATCH_CANDIDATES = [1024, 768, 512, 384, 256, 192, 128, 96, 64]
MODEL_SIZE_PRESETS = {
    "base": {"base_channels": 48, "num_blocks": 4, "latent_dim": 384},
    "wide": {"base_channels": 64, "num_blocks": 4, "latent_dim": 384},
    "deeper": {"base_channels": 48, "num_blocks": 5, "latent_dim": 384},
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _target_shape(cfg: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(v) for v in cfg.get("target_shape", [36, 45, 38]))


def canonical_source(row: dict[str, Any]) -> str:
    source = str(row.get("source", "")).lower()
    if source == "pubmed" or row.get("pmid"):
        return "pubmed"
    if source.startswith("neurovault"):
        return "neurovault"
    if source.startswith("nilearn"):
        return "nilearn"
    return source or "unknown"


def source_detail(row: dict[str, Any]) -> str:
    return str(row.get("source") or canonical_source(row))


def filter_data_mode(dataset: UnifiedMapTextDataset, data_mode: str) -> UnifiedMapTextDataset:
    if data_mode not in {"pubmed_only", "mixed"}:
        raise ValueError("DATA_MODE/data_mode must be 'pubmed_only' or 'mixed'")
    if data_mode == "pubmed_only":
        dataset.rows = [row for row in dataset.rows if canonical_source(row) == "pubmed"]
    return dataset


def source_counts(dataset: UnifiedMapTextDataset) -> dict[str, int]:
    return dict(Counter(canonical_source(row) for row in dataset.rows))


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


def build_source_sampler(dataset: UnifiedMapTextDataset, cfg: dict[str, Any]):
    mode = str(cfg.get("source_sampling", cfg.get("SOURCE_SAMPLING", "temperature"))).lower()
    alpha = float(cfg.get("source_sampling_alpha", cfg.get("SOURCE_SAMPLING_ALPHA", 0.5)))
    if mode not in {"natural", "temperature", "balanced"}:
        raise ValueError("SOURCE_SAMPLING/source_sampling must be natural, temperature, or balanced")
    if mode == "natural":
        return None
    counts = Counter(canonical_source(row) for row in dataset.rows)
    if mode == "balanced":
        source_probs = {src: 1.0 / max(1, len(counts)) for src in counts}
    else:
        denom = sum(float(n) ** alpha for n in counts.values())
        source_probs = {src: (float(n) ** alpha) / denom for src, n in counts.items()}
    weights = [source_probs[canonical_source(row)] / counts[canonical_source(row)] for row in dataset.rows]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


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


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_batches: int | None = None,
    compute_metrics: bool = True,
    metric_max_batches: int | None = None,
    include_voxel_auroc: bool = False,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    model.train(train)
    losses: list[float] = []
    metric_rows: list[dict[str, float]] = []
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
                loss = F.mse_loss(pred, x)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if compute_metrics and (metric_max_batches is None or len(metric_rows) < int(metric_max_batches)):
            metric_rows.append(generation_metrics(pred.detach().clamp(0.0, 1.0).cpu(), x.detach().cpu(), include_voxel_auroc=include_voxel_auroc))
        if show_progress and tqdm is not None:
            iterator.set_postfix(mse=f"{losses[-1]:.5f}")
    metrics = _avg_metric_rows(metric_rows)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    metrics["epoch_time_sec"] = float(time.time() - start)
    metrics["source_counts"] = dict(source_counter)
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
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    target_shape = _target_shape(cfg)
    data_mode = str(cfg.get("data_mode", cfg.get("DATA_MODE", "mixed"))).lower()
    out_dir = Path(cfg.get("output_dir", "experiments/3dcnn/atlas_free_cnn/outputs/runs/autoencoder_stable"))
    out_dir.mkdir(parents=True, exist_ok=True)
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
    print({"data_mode": data_mode, "source_counts_by_split": split_counts})

    model = build_model(cfg, target_shape, device)
    preflight = preflight_batch_size(model, target_shape, cfg, device)
    batch_size = int(preflight["selected_batch_size"])
    cfg = dict(cfg)
    cfg["batch_size"] = batch_size
    cfg["loss"] = {"type": "raw_mse"}
    cfg["prediction_activation"] = "none"
    cfg["model"] = model_config(cfg)
    cfg["data_mode"] = data_mode
    cfg["preflight"] = preflight
    with (out_dir / "autoencoder_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
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
    sampler = build_source_sampler(train_ds, cfg) if data_mode == "mixed" else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    ckpt = CheckpointManager(cfg.get("checkpoint_dir", str(out_dir / "checkpoints")), maximize={"val_loss": False})
    use_amp = bool(cfg.get("amp", device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and device.type == "cuda"))
    history: list[dict[str, Any]] = []
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
            scaler=scaler,
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
        payload = {
            "model": model.state_dict(),
            "config": cfg,
            "history": history,
            "epoch": epoch,
            "target_shape": target_shape,
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
            if ckpt.maybe_save_best("val_loss", float(val_metrics.get("loss", math.inf)), payload):
                ckpt.save("best_cnn_autoencoder.pt", payload)
            ckpt.maybe_save_best("spatial_corr", float(val_metrics.get("spatial_corr", -1.0)), payload)
            ckpt.maybe_save_best("top5_dice", float(val_metrics.get("top5_dice", 0.0)), payload)
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
    if bool(cfg.get("save_plots", True)):
        plot_ds = test_ds or val_ds
        for src in ["pubmed", "neurovault", "nilearn"]:
            save_qualitative_plots(model, plot_ds, cfg, device, out_dir / "plots", src)
    return {
        "history": history,
        "checkpoint_dir": str(ckpt.out_dir),
        "best_checkpoint": str(ckpt.out_dir / "best_cnn_autoencoder.pt"),
        "preflight": preflight,
        "source_counts_by_split": split_counts,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/3dcnn/atlas_free_cnn/configs/autoencoder_config.yaml")
    args = p.parse_args()
    train_from_config(load_yaml(args.config))


if __name__ == "__main__":
    main()
