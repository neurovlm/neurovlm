"""Stage 4: train a text-to-brain projection through a frozen AE decoder.

This is separate from the contrastive text projection: it maps SPECTER/SPECTER2
embeddings into the AE decoder latent space.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
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
from atlas_free_cnn.evaluation.stage4_semantic import evaluate_generation_semantic_loader
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
    build_brain_encoder,
    build_cnn_autoencoder,
    build_generative_text_to_ae_latent,
    build_text_projection,
    build_text_to_brain_projection,
    load_autoencoder_checkpoint,
)
from atlas_free_cnn.training.source_sampling import canonical_source
from neurovlm.gnn.ale_cnn import count_parameters


TEXT_TO_BRAIN_BATCH_CANDIDATES = [4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 96, 64]

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
        lambda_latent_cosine=float(loss.get("lambda_latent_cosine", 0.0)),
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
    if isinstance(payload, dict):
        for key in ["processed_embedding_by_text", "embedding_by_text", "text_to_embedding", "embeddings_by_text"]:
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
        else:
            if isinstance(payload.get("records"), list):
                payload = {
                    str(row.get("text", row.get("input_text", row.get("text_id", "")))): row["processed_embedding"]
                    for row in payload["records"]
                    if "processed_embedding" in row
                }
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported text embedding cache format in {path}")
    out = {}
    for k, v in payload.items():
        if torch.is_tensor(v) or isinstance(v, (list, tuple)):
            tensor = torch.as_tensor(v, dtype=torch.float32)
            if tensor.ndim == 1:
                out[str(k)] = tensor
    return out


def domain_from_source(value: str) -> str:
    source = str(value or "").lower()
    if source == "pubmed":
        return "pubmed"
    if source == "neurovault" or source.startswith("neurovault:"):
        return "neurovault"
    if source == "nilearn" or source.startswith("nilearn:"):
        return "nilearn"
    return source


def primary_pair(row: dict[str, Any]) -> tuple[str, str]:
    positives = row.get("positive_texts", []) or []
    text_id = ""
    if positives:
        first = positives[0]
        text_id = str(first.get("text_id") or first.get("id") or first.get("text") or "")
    return str(row.get("map_id", "")), text_id


def split_fingerprint(rows: list[dict[str, Any]]) -> str:
    pairs = sorted(primary_pair(row) for row in rows)
    payload = json.dumps(pairs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def filter_dataset_to_domain(dataset: UnifiedMapTextDataset, domain: str) -> dict[str, Any]:
    source_field = "source"
    before = len(dataset.rows)
    source_counts_before = Counter(str(row.get(source_field, "")) for row in dataset.rows)
    dataset.rows = [row for row in dataset.rows if domain_from_source(row.get(source_field, "")) == domain]
    source_counts_after = Counter(str(row.get(source_field, "")) for row in dataset.rows)
    if not dataset.rows:
        raise RuntimeError(f"Domain filter {domain!r} produced zero rows for {dataset.path}")
    bad = [source for source in source_counts_after if domain_from_source(source) != domain]
    if bad:
        raise RuntimeError(f"Domain filter {domain!r} left non-domain rows in {dataset.path}: {bad}")
    return {
        "path": str(dataset.path),
        "source_field": source_field,
        "rows_before_filtering": before,
        "rows_after_filtering": len(dataset.rows),
        "unique_map_ids": len({str(row.get("map_id", "")) for row in dataset.rows}),
        "unique_text_ids": len({primary_pair(row)[1] for row in dataset.rows}),
        "source_value_counts_before": dict(source_counts_before),
        "source_value_counts_after": dict(source_counts_after),
        "fingerprint": split_fingerprint(dataset.rows),
        "sample_rows": [
            {
                "map_id": row.get("map_id"),
                "source": row.get(source_field),
                "positive_text_id": primary_pair(row)[1],
            }
            for row in dataset.rows[:3]
        ],
    }


def leakage_report(train_ds: UnifiedMapTextDataset, val_ds: UnifiedMapTextDataset, test_ds: UnifiedMapTextDataset) -> dict[str, Any]:
    split_ids = {
        "train": {primary_pair(row) for row in train_ds.rows},
        "val": {primary_pair(row) for row in val_ds.rows},
        "test": {primary_pair(row) for row in test_ds.rows},
    }
    overlaps = {
        "train_val": len(split_ids["train"] & split_ids["val"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "val_test": len(split_ids["val"] & split_ids["test"]),
    }
    return {"overlap_counts": overlaps, "passed": all(value == 0 for value in overlaps.values())}


def checkpoint_architecture(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    target_shape = payload.get("target_shape") or cfg.get("target_shape") or [36, 45, 38]
    if isinstance(target_shape, tuple):
        target_shape = list(target_shape)
    return {
        "latent_dim": model_cfg.get("latent_dim", payload.get("latent_dim", 384)),
        "base_channels": model_cfg.get("base_channels", 64),
        "num_blocks": model_cfg.get("num_blocks", 4),
        "encoder_arch": model_cfg.get("encoder_arch", "plain"),
        "dropout": model_cfg.get("dropout", 0.1),
        "norm": model_cfg.get("norm", "group"),
        "pooling": model_cfg.get("pooling", "max"),
        "blocks_per_stage": model_cfg.get("blocks_per_stage", 2),
        "use_dilation": model_cfg.get("use_dilation", False),
        "multi_scale": model_cfg.get("multi_scale", False),
        "global_context": model_cfg.get("global_context", "none"),
        "target_shape": target_shape,
    }


def apply_checkpoint_architecture(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(cfg["autoencoder_checkpoint"], map_location="cpu", weights_only=False)
    arch = checkpoint_architecture(payload)
    cfg = dict(cfg)
    model_cfg = dict(cfg.get("model", {}))
    for key in [
        "latent_dim",
        "base_channels",
        "num_blocks",
        "encoder_arch",
        "dropout",
        "norm",
        "pooling",
        "blocks_per_stage",
        "use_dilation",
        "multi_scale",
        "global_context",
    ]:
        model_cfg[key] = arch[key]
    cfg["model"] = model_cfg
    cfg["target_shape"] = arch["target_shape"]
    return cfg, {"selected_ae_checkpoint_architecture": arch}


def _lookup(cache: dict[str, torch.Tensor], texts: list[str]) -> torch.Tensor:
    missing = [t for t in texts if t not in cache]
    if missing:
        raise KeyError(f"{len(missing)} texts missing from embedding cache; example={missing[0][:160]}")
    return torch.stack([cache[t] for t in texts])


def text_to_brain_loss(pred, target, brain_z, text_z, loss_cfg: GenerationLossConfig):
    latent_cosine = F.cosine_similarity(text_z, brain_z.detach(), dim=1, eps=1e-8).mean()
    parts = {
        "latent_mse": F.mse_loss(text_z, brain_z.detach()),
        "latent_cosine": latent_cosine,
        "latent_cosine_loss": 1.0 - latent_cosine,
        "reconstruction_mse": F.mse_loss(pred, target),
    }
    total = (
        loss_cfg.lambda_latent * parts["latent_mse"]
        + loss_cfg.lambda_latent_cosine * parts["latent_cosine_loss"]
        + loss_cfg.lambda_recon * parts["reconstruction_mse"]
    )
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
                loss = F.mse_loss(pred, x) + F.mse_loss(text_z, brain_z.detach())
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


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "cuda out of memory" in str(exc).lower()


def _fallback_batch_size(current: int, candidates: list[int]) -> int | None:
    smaller = sorted({int(v) for v in candidates if int(v) < int(current)}, reverse=True)
    return smaller[0] if smaller else None


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
    loss_part_rows = []
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
        target = x
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
                brain_z = autoencoder.encoder(target)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
                text_z = text_projector(raw_text)
                pred = autoencoder.decoder(text_z)
                loss, parts = text_to_brain_loss(pred, target, brain_z, text_z, loss_cfg)
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
        loss_part_rows.append({k: float(v.detach().cpu()) for k, v in parts.items() if torch.is_tensor(v)})
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
    if loss_part_rows:
        for key in loss_part_rows[0]:
            avg_metrics[key] = float(sum(row[key] for row in loss_part_rows) / max(1, len(loss_part_rows)))
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


def _stage3_encoder_arch(model_name: str | None) -> str:
    if model_name in {None, "", "ale_3dcnn"}:
        return "plain"
    if model_name == "ale_3dcnn_resnet":
        return "resnet"
    raise ValueError(f"Unknown stage3 encoder model_name: {model_name!r}")


def load_stage3_contrastive_models(stage3_checkpoint: str | Path, device: torch.device):
    payload = torch.load(stage3_checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    out_dim = int(cfg.get("out_dim", 384))
    brain_encoder = build_brain_encoder(
        out_dim=out_dim,
        encoder_arch=_stage3_encoder_arch(cfg.get("model", "ale_3dcnn")),
        base_channels=int(cfg.get("base_channels", 64)),
        num_blocks=int(cfg.get("num_blocks", 4)),
        blocks_per_stage=int(cfg.get("blocks_per_stage", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_dilation=bool(cfg.get("use_dilation", False)),
        multi_scale=bool(cfg.get("multi_scale", False)),
        global_context=str(cfg.get("global_context", "none")),
    ).to(device)
    text_proj = build_text_projection("random", device=device)
    brain_state = payload.get("brain_encoder")
    text_state = payload.get("text_proj") or payload.get("text_projection")
    if brain_state is None or text_state is None:
        raise KeyError(f"Stage 3 checkpoint must contain brain_encoder and text_proj: {stage3_checkpoint}")
    brain_encoder.load_state_dict(brain_state, strict=True)
    text_proj.load_state_dict(text_state, strict=True)
    brain_encoder.eval()
    text_proj.eval()
    for model in [brain_encoder, text_proj]:
        for param in model.parameters():
            param.requires_grad_(False)
    return brain_encoder, text_proj


@torch.no_grad()
def generation_semantic_auc(
    autoencoder,
    generative_text_to_ae_latent,
    stage3_brain_encoder,
    stage3_text_projection,
    dataset: UnifiedMapTextDataset,
    text_cache: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    collator = PrimaryTextVolumeCollator(_target_shape(cfg), text_rank=int(cfg.get("primary_text_rank", 0)))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("generation_auc_batch_size", cfg.get("eval_batch_size", cfg.get("batch_size", 256)))),
        shuffle=False,
        num_workers=int(cfg.get("eval_num_workers", 0)),
        collate_fn=collator,
        pin_memory=bool(cfg.get("pin_memory", device.type == "cuda")),
    )
    return evaluate_generation_semantic_loader(
        autoencoder,
        generative_text_to_ae_latent,
        stage3_brain_encoder,
        stage3_text_projection,
        loader,
        text_cache,
        device,
        evaluator_text_cache=text_cache,
        prefix="generation",
        include_raw=True,
        include_clamped=True,
    )


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg, architecture_report = apply_checkpoint_architecture(cfg)
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
    architecture_report.update(
        {
            "instantiated_stage4_model": model_cfg,
            "target_shape": list(target_shape),
            "strict_autoencoder_load": "passed",
        }
    )
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)
    trainable_ae = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    if trainable_ae:
        raise RuntimeError(f"Corrected Stage 4 requires frozen AE encoder/decoder; found {trainable_ae} trainable AE parameters")
    projection_cfg = cfg.get("generative_text_to_ae_latent", cfg.get("text_to_brain_projection", {}))
    projector_name = str(projection_cfg.get("name", "generative_text_to_ae_latent"))
    legacy_stage4 = bool(cfg.get("legacy_contrastive_initialized_stage4", False))
    input_dim = int(projection_cfg.get("in_dim", 768))
    hidden_dim = int(projection_cfg.get("hidden_dim", cfg.get("hidden_dim", 512)))
    latent_dim = int(model_cfg.get("latent_dim", 384))
    if input_dim != 768:
        raise RuntimeError(f"Corrected Stage 4 requires 768-d processed SPECTER2 embeddings, got {input_dim}")
    if latent_dim != 384:
        raise RuntimeError(f"Corrected Stage 4 requires AE latent_dim=384, got {latent_dim}")
    if legacy_stage4:
        text_projector = build_text_to_brain_projection(
            cfg.get("text_projection_init", "random"),
            device=device,
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=int(projection_cfg.get("depth", cfg.get("depth", 2))),
            dropout=float(projection_cfg.get("dropout", cfg.get("dropout", 0.1))),
            out_dim=latent_dim,
        )
    else:
        if cfg.get("text_projection_init", "random") not in {"random", "scratch", "fresh"}:
            raise RuntimeError(
                "Corrected Stage 4 generative_text_to_ae_latent must be initialized fresh. "
                "Set legacy_contrastive_initialized_stage4=True to reproduce the old path."
            )
        text_projector = build_generative_text_to_ae_latent(
            device=device,
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
    stage3_init_report = {
        "stage3_checkpoint": cfg.get("stage3_contrastive_checkpoint", ""),
        "loaded_tensors": 0,
        "projector_name": projector_name,
        "legacy_contrastive_initialized_stage4": legacy_stage4,
        "note": "Corrected Stage 4 initializes the generative text-to-AE-latent projector fresh and does not load Stage 3 contrastive text projection tensors.",
    }
    if legacy_stage4 and cfg.get("stage3_contrastive_checkpoint"):
        stage3_payload = torch.load(cfg["stage3_contrastive_checkpoint"], map_location="cpu", weights_only=False)
        stage3_state = stage3_payload.get("text_proj") or stage3_payload.get("text_projection") or {}
        current_state = text_projector.state_dict()
        compatible = {
            key: value
            for key, value in stage3_state.items()
            if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
        }
        current_state.update(compatible)
        text_projector.load_state_dict(current_state)
        stage3_init_report = {
            "stage3_checkpoint": cfg["stage3_contrastive_checkpoint"],
            "checkpoint_tensors": len(stage3_state),
            "loaded_tensors": len(compatible),
            "loaded_keys": sorted(compatible),
            "note": "Stage 4 projection is run-specific; compatible tensors from the matching Stage 3 text projection are used as initialization.",
        }
    architecture_report["matching_stage3_text_projection_initialization"] = stage3_init_report
    architecture_report["corrected_stage4_projector"] = {
        "projector_name": projector_name,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "fresh_initialization": not legacy_stage4,
        "loads_stage3_projection_tensors": bool(stage3_init_report.get("loaded_tensors", 0)),
        "targets": "raw frozen-autoencoder latent",
    }
    if not legacy_stage4 and stage3_init_report["loaded_tensors"] != 0:
        raise RuntimeError("Corrected Stage 4 must not load Stage 3 contrastive projector tensors")
    preflight = preflight_batch_size(autoencoder, text_projector, target_shape, cfg, device)
    cfg = dict(cfg)
    cfg["batch_size"] = int(preflight["selected_batch_size"])
    cfg["preflight"] = preflight
    train_ds = UnifiedMapTextDataset(cfg["train_jsonl"])
    val_ds = UnifiedMapTextDataset(cfg["val_jsonl"])
    domain_report = {}
    if cfg.get("domain"):
        requested_domain = str(cfg["domain"])
        domain_report = {
            "requested_domain": requested_domain,
            "train": filter_dataset_to_domain(train_ds, requested_domain),
            "val": filter_dataset_to_domain(val_ds, requested_domain),
        }
        if cfg.get("test_jsonl"):
            test_probe = UnifiedMapTextDataset(cfg["test_jsonl"], load_volumes=False)
            domain_report["test"] = filter_dataset_to_domain(test_probe, requested_domain)
            domain_report["leakage"] = leakage_report(train_ds, val_ds, test_probe)
            if not domain_report["leakage"]["passed"]:
                raise RuntimeError(f"Stage 4 split leakage detected: {domain_report['leakage']['overlap_counts']}")
    architecture_report["domain_filter_report"] = domain_report
    out_dir = Path(cfg.get("output_dir", "experiments/3dcnn/atlas_free_cnn/outputs/runs/text_to_brain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trainability_report = {
        "autoencoder_encoder_trainable_parameters": int(sum(p.numel() for p in autoencoder.encoder.parameters() if p.requires_grad)),
        "autoencoder_decoder_trainable_parameters": int(sum(p.numel() for p in autoencoder.decoder.parameters() if p.requires_grad)),
        "generative_text_to_ae_latent_trainable_parameters": int(sum(p.numel() for p in text_projector.parameters() if p.requires_grad)),
        "stage3_contrastive_projector_part_of_training": False,
        "stage3_brain_encoder_part_of_training": False,
        "status": "passed",
    }
    if trainability_report["autoencoder_encoder_trainable_parameters"] or trainability_report["autoencoder_decoder_trainable_parameters"]:
        trainability_report["status"] = "failed"
        raise RuntimeError(f"Corrected Stage 4 freeze check failed: {trainability_report}")
    with (out_dir / "stage4_trainability_report.json").open("w") as f:
        json.dump(trainability_report, f, indent=2)
    with (out_dir / "stage4_architecture_compatibility_report.json").open("w") as f:
        json.dump(architecture_report, f, indent=2)
    if domain_report:
        with (out_dir / "stage4_domain_dataset_report.json").open("w") as f:
            json.dump(domain_report, f, indent=2)
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
    def make_loaders(batch_size: int):
        train = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, **loader_kwargs)
        val = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, **loader_kwargs)
        return train, val

    train_loader, val_loader = make_loaders(int(cfg.get("batch_size", 4)))
    text_cache = _load_text_cache(cfg["text_embedding_cache"])
    if not text_cache:
        raise RuntimeError(f"Text embedding cache is empty: {cfg['text_embedding_cache']}")
    dims = {int(v.numel()) for v in text_cache.values()}
    if dims != {768}:
        raise RuntimeError(f"Corrected Stage 4 requires all text cache vectors to be 768-d, got dimensions {sorted(dims)}")
    stage3_semantic_models = None
    if cfg.get("stage3_contrastive_checkpoint"):
        stage3_semantic_models = load_stage3_contrastive_models(cfg["stage3_contrastive_checkpoint"], device)
    optimizer = torch.optim.AdamW(text_projector.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    loss_cfg = _loss_cfg(cfg)
    ckpt = CheckpointManager(
        cfg.get("checkpoint_dir", "experiments/3dcnn/atlas_free_cnn/outputs/runs/text_to_brain/checkpoints"),
        maximize={
            "val_loss": False,
            "val_latent_mse": False,
            "val_reconstruction_mse": False,
            "val_spatial_corr": True,
            "val_top5_dice": True,
            "val_generation_normalized_auc": True,
            "generation_top5_dice": True,
            "generation_spatial_correlation": True,
        },
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
    max_epochs = int(cfg.get("epochs", 3))
    generation_auc_val_interval = int(cfg.get("generation_auc_val_interval", 5))
    early_stopping = bool(cfg.get("early_stopping", True))
    early_metric = str(cfg.get("early_stopping_metric", "val_generation_normalized_auc"))
    early_mode = str(cfg.get("early_stopping_mode", "max" if "auc" in early_metric or "corr" in early_metric or "dice" in early_metric else "min"))
    early_patience = int(cfg.get("early_stopping_patience", 25))
    early_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    early_metric_key = early_metric[4:] if early_metric.startswith("val_") else early_metric
    early_metric_requires_generation_auc = "generation" in early_metric_key and "auc" in early_metric_key
    best_early_value = -float("inf") if early_mode == "max" else float("inf")
    bad_val_checks = 0
    stop_reason = "max_epochs"
    runtime_candidates = sorted(
        {int(v) for v in cfg.get("batch_candidates", TEXT_TO_BRAIN_BATCH_CANDIDATES)} | {int(cfg["batch_size"])},
        reverse=True,
    )
    runtime_batch_fallback = bool(cfg.get("runtime_batch_fallback", True))
    epoch = 1
    while epoch <= max_epochs:
        try:
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
            if epoch == 1 or epoch % val_interval == 0 or epoch == max_epochs:
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
                run_semantic_auc = (
                    epoch == 1
                    or epoch % generation_auc_val_interval == 0
                    or epoch == max_epochs
                    or (early_stopping and early_metric_requires_generation_auc)
                )
                if stage3_semantic_models is not None and run_semantic_auc:
                    stage3_brain_encoder, stage3_text_projection = stage3_semantic_models
                    last_val_metrics.update(
                        generation_semantic_auc(
                            autoencoder,
                            text_projector,
                            stage3_brain_encoder,
                            stage3_text_projection,
                            val_ds,
                            text_cache,
                            cfg,
                            device,
                        )
                    )
        except BaseException as exc:
            if not (runtime_batch_fallback and device.type == "cuda" and _is_cuda_oom(exc)):
                raise
            current_batch_size = int(cfg["batch_size"])
            next_batch_size = _fallback_batch_size(current_batch_size, runtime_candidates)
            if next_batch_size is None:
                raise
            print({
                "runtime_batch_fallback": True,
                "epoch": epoch,
                "failed_batch_size": current_batch_size,
                "next_batch_size": next_batch_size,
                "reason": "cuda_out_of_memory",
            })
            optimizer.zero_grad(set_to_none=True)
            text_projector.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            cfg["batch_size"] = int(next_batch_size)
            cfg.setdefault("runtime_batch_fallbacks", []).append(
                {"epoch": epoch, "failed_batch_size": current_batch_size, "next_batch_size": int(next_batch_size)}
            )
            train_loader, val_loader = make_loaders(int(next_batch_size))
            continue
        val_metrics = dict(last_val_metrics)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        payload = {
            "text_projector": text_projector.state_dict(),
            "generative_text_to_ae_latent": text_projector.state_dict(),
            "config": cfg,
            "history": history,
            "epoch": epoch,
            "target_shape": target_shape,
            "projector_name": projector_name,
            "early_stopping": {
                "enabled": early_stopping,
                "metric": early_metric,
                "mode": early_mode,
                "patience": early_patience,
                "min_delta": early_min_delta,
                "generation_auc_forced_each_validation": bool(early_metric_requires_generation_auc),
                "bad_val_checks": bad_val_checks,
                "best_value": best_early_value,
            },
        }
        ckpt.save_last(payload)
        if val_metrics:
            ckpt.maybe_save_best("val_loss", val_metrics.get("loss", float("inf")), payload)
            ckpt.maybe_save_best("val_latent_mse", val_metrics.get("latent_mse", float("inf")), payload)
            ckpt.maybe_save_best("val_reconstruction_mse", val_metrics.get("reconstruction_mse", float("inf")), payload)
            ckpt.maybe_save_best("val_spatial_corr", val_metrics.get("spatial_corr", -1.0), payload)
            ckpt.maybe_save_best("val_top5_dice", val_metrics.get("top5_dice", 0.0), payload)
            if "generation_mean_normalized_auc" in val_metrics:
                ckpt.maybe_save_best(
                    "val_generation_normalized_auc",
                    val_metrics.get("generation_mean_normalized_auc", 0.0),
                    payload,
                )
            ckpt.maybe_save_best("generation_top5_dice", val_metrics.get("top5_dice", 0.0), payload)
            ckpt.maybe_save_best("generation_spatial_correlation", val_metrics.get("spatial_corr", -1.0), payload)
            if early_stopping:
                if early_metric_key not in val_metrics:
                    current = float("nan")
                else:
                    current = float(val_metrics[early_metric_key])
                    improved = (
                        current > best_early_value + early_min_delta
                        if early_mode == "max"
                        else current < best_early_value - early_min_delta
                    )
                    if improved:
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
        epoch += 1
    with (out_dir / "text_to_brain_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    with (out_dir / "preflight.json").open("w") as f:
        json.dump(preflight, f, indent=2)
    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "training_stop.json").open("w") as f:
        json.dump(
            {
                "stop_reason": stop_reason,
                "epochs_completed": len(history),
                "early_stopping": early_stopping,
                "early_stopping_metric": early_metric,
                "early_stopping_metric_key": early_metric_key,
                "early_stopping_patience": early_patience,
                "early_stopping_min_delta": early_min_delta,
                "generation_auc_forced_each_validation": bool(early_metric_requires_generation_auc),
                "best_early_value": best_early_value,
                "final_batch_size": int(cfg.get("batch_size", 0)),
                "runtime_batch_fallbacks": cfg.get("runtime_batch_fallbacks", []),
            },
            f,
            indent=2,
        )
    best_path = ckpt.out_dir / "best_val_generation_normalized_auc.pt"
    if not best_path.exists():
        best_path = ckpt.out_dir / "best_val_loss.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        text_projector.load_state_dict(best_payload.get("generative_text_to_ae_latent") or best_payload["text_projector"])
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
        if cfg.get("domain"):
            filter_dataset_to_domain(eval_ds, str(cfg["domain"]))
        eval_rows = evaluate_generation_dataset(
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
        if stage3_semantic_models is not None:
            stage3_brain_encoder, stage3_text_projection = stage3_semantic_models
            semantic = generation_semantic_auc(
                autoencoder,
                text_projector,
                stage3_brain_encoder,
                stage3_text_projection,
                eval_ds,
                text_cache,
                cfg,
                device,
            )
            for row in eval_rows:
                if row.get("source") == "all":
                    row.update(semantic)
        generation_eval_rows.extend(eval_rows)
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
