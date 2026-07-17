"""Standardized training for the retained atlas-free CNN autoencoder."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from neurovlm.ale_cnn import ALE3DCNNAutoEncoder
from neurovlm.atlas_free_dataset import AtlasFreeCNNDataProvider, canonical_atlas_free_domain
from neurovlm.evaluation.spatial import reconstruction_metrics
from neurovlm.pipelines import (
    CheckpointManager,
    MetricDirection,
    MetricRecorder,
    RunConfig,
    RunContext,
    atomic_write_csv,
    atomic_write_json,
)


CNN_AUTOENCODER_PRESET = "retained_base64_v1"
CNN_AUTOENCODER_SHAPE = (36, 45, 38)
CNN_AUTOENCODER_DOMAINS = ("pubmed", "nilearn", "neurovault")


@dataclass(frozen=True)
class AutoencoderTrainConfig:
    """Typed configuration for mixed pretraining or domain fine-tuning.

    ``initialization="auto"`` means scratch initialization for the mixed
    baseline and the released mixed autoencoder for a fine-tuned run.
    Released resources remain the default; ``split_dir`` and ``volume_path``
    are explicit local overrides.
    """

    output_root: str | Path = "runs"
    run_id: str | None = None
    variant: Literal["mixed_baseline", "finetuned"] = "mixed_baseline"
    domain: Literal["pubmed", "nilearn", "neurovault"] | None = None
    seed: int = 42
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 64
    eval_batch_size: int | None = None
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float | None = 1.0
    amp: bool = True
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0
    include_voxel_auroc: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    limit: int | None = None
    split_dir: str | Path | None = None
    volume_path: str | Path | None = None
    initialization: Literal["auto", "scratch", "released_mixed"] = "auto"
    from_run: str | Path | None = None
    init_checkpoint: str | Path | None = None
    resume: str | Path | None = None
    preset: Literal["retained_base64_v1", "custom"] = CNN_AUTOENCODER_PRESET
    target_shape: tuple[int, int, int] = CNN_AUTOENCODER_SHAPE
    in_channels: int = 1
    base_channels: int = 64
    num_blocks: int = 4
    latent_dim: int = 384
    dropout: float = 0.1
    norm: Literal["group", "batch", "instance", "none"] = "group"
    pooling: Literal["max", "stride"] = "max"

    def __post_init__(self) -> None:
        if self.variant not in {"mixed_baseline", "finetuned"}:
            raise ValueError("variant must be 'mixed_baseline' or 'finetuned'")
        if self.variant == "mixed_baseline" and self.domain is not None:
            raise ValueError("mixed_baseline is domain-independent; omit domain")
        if self.variant == "finetuned" and self.domain not in CNN_AUTOENCODER_DOMAINS:
            raise ValueError("finetuned autoencoder training requires a domain")
        if self.initialization not in {"auto", "scratch", "released_mixed"}:
            raise ValueError("initialization must be auto, scratch, or released_mixed")
        explicit = sum(value is not None for value in (self.from_run, self.init_checkpoint))
        if explicit > 1:
            raise ValueError("Pass at most one of from_run and init_checkpoint")
        if explicit and self.initialization != "auto":
            raise ValueError("Explicit checkpoint initialization cannot be combined with initialization")
        if self.resume is not None and self.run_id is None:
            raise ValueError("resume requires the original run_id")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("num_workers and seed must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive or None")
        if self.max_eval_batches is not None and self.max_eval_batches < 1:
            raise ValueError("max_eval_batches must be positive or None")
        if len(self.target_shape) != 3 or any(int(value) <= 0 for value in self.target_shape):
            raise ValueError("target_shape must contain three positive dimensions")
        if self.in_channels < 1 or self.base_channels < 1 or self.latent_dim < 1:
            raise ValueError("in_channels, base_channels, and latent_dim must be positive")
        if self.preset not in {CNN_AUTOENCODER_PRESET, "custom"}:
            raise ValueError(f"Unknown CNN autoencoder preset {self.preset!r}")
        retained = {
            "target_shape": CNN_AUTOENCODER_SHAPE,
            "in_channels": 1,
            "base_channels": 64,
            "num_blocks": 4,
            "latent_dim": 384,
            "dropout": 0.1,
            "norm": "group",
            "pooling": "max",
        }
        if self.preset == CNN_AUTOENCODER_PRESET:
            changed = {
                name: getattr(self, name)
                for name, expected in retained.items()
                if getattr(self, name) != expected
            }
            if changed:
                raise ValueError(
                    f"preset={CNN_AUTOENCODER_PRESET!r} requires the retained architecture; "
                    f"use preset='custom' for overrides: {changed}"
                )

    @property
    def primary_metric(self) -> str:
        return "val_loss" if self.variant == "mixed_baseline" else "val_top5_dice"

    @property
    def metric_direction(self) -> MetricDirection:
        return MetricDirection.MIN if self.variant == "mixed_baseline" else MetricDirection.MAX

    def architecture(self) -> dict[str, Any]:
        return {
            "architecture": "ALE3DCNNAutoEncoder",
            "preset": self.preset,
            "output_shape": tuple(int(value) for value in self.target_shape),
            "in_channels": int(self.in_channels),
            "base_channels": int(self.base_channels),
            "num_blocks": int(self.num_blocks),
            "latent_dim": int(self.latent_dim),
            "dropout": float(self.dropout),
            "norm": self.norm,
            "pooling": self.pooling,
            "encoder_arch": "plain",
        }


@dataclass(frozen=True)
class AutoencoderEvaluation:
    summary: Mapping[str, float]
    by_source: tuple[Mapping[str, Any], ...]
    n: int


@dataclass(frozen=True)
class AutoencoderTrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    epochs_completed: int
    best_metric: float
    test_metrics: Mapping[str, float]
    model: ALE3DCNNAutoEncoder


class _VolumeCollator:
    def __init__(self, target_shape: tuple[int, int, int]):
        self.target_shape = target_shape

    def __call__(self, batch: list[Mapping[str, Any]]) -> dict[str, Any]:
        volumes = []
        sources = []
        for item in batch:
            volume = torch.as_tensor(item["volume"]).float()
            if volume.ndim != 4 or volume.shape[0] != 1:
                raise ValueError(f"Expected a 1 x D x H x W volume, got {tuple(volume.shape)}")
            if tuple(volume.shape[-3:]) != self.target_shape:
                volume = F.interpolate(
                    volume.unsqueeze(0),
                    size=self.target_shape,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
            volumes.append(torch.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0))
            sources.append(canonical_atlas_free_domain(item.get("metadata") or {}))
        return {
            "volume": torch.stack(volumes),
            "source": sources,
            "map_id": [str(item.get("map_id") or "") for item in batch],
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def build_autoencoder(config: AutoencoderTrainConfig) -> ALE3DCNNAutoEncoder:
    """Construct the configured retained plain 3D CNN autoencoder."""

    return ALE3DCNNAutoEncoder(
        output_shape=tuple(config.target_shape),
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        norm=config.norm,
        pooling=config.pooling,
        encoder_arch="plain",
    )


def _checkpoint_architecture(payload: Mapping[str, Any]) -> dict[str, Any]:
    architecture = payload.get("architecture")
    if isinstance(architecture, Mapping) and architecture:
        output = dict(architecture)
        if "target_shape" in output and "output_shape" not in output:
            output["output_shape"] = output.pop("target_shape")
        if "output_shape" in output:
            output["output_shape"] = tuple(int(value) for value in output["output_shape"])
        return output
    config = payload.get("config")
    if isinstance(config, Mapping):
        model = config.get("model") if isinstance(config.get("model"), Mapping) else config
        shape = payload.get("target_shape") or config.get("target_shape")
        if shape is not None:
            return {
                "architecture": "ALE3DCNNAutoEncoder",
                "preset": CNN_AUTOENCODER_PRESET,
                "output_shape": tuple(int(value) for value in shape),
                "in_channels": int(model.get("in_channels", 1)),
                "base_channels": int(model.get("base_channels", 64)),
                "num_blocks": int(model.get("num_blocks", 4)),
                "latent_dim": int(model.get("latent_dim", 384)),
                "dropout": float(model.get("dropout", 0.1)),
                "norm": str(model.get("norm", "group")),
                "pooling": str(model.get("pooling", "max")),
                "encoder_arch": str(model.get("encoder_arch", "plain")),
            }
    raise ValueError("Checkpoint does not record a reloadable autoencoder architecture")


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, Tensor]:
    for name in ("model_state_dict", "state_dict", "model", "autoencoder"):
        state = payload.get(name)
        if isinstance(state, Mapping):
            return state
    raise KeyError("Checkpoint has no model_state_dict, state_dict, model, or autoencoder")


def _validate_architecture(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = []
    for key, expected_value in expected.items():
        if key not in actual:
            continue
        actual_value = actual[key]
        if key == "output_shape":
            actual_value = tuple(actual_value)
            expected_value = tuple(expected_value)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if mismatches:
        raise ValueError("Checkpoint architecture mismatch: " + "; ".join(mismatches))


def autoencoder_from_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ALE3DCNNAutoEncoder:
    """Reconstruct an autoencoder from a standardized or legacy checkpoint."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Autoencoder checkpoint payload must be a mapping")
    arch = _checkpoint_architecture(payload)
    if arch.get("architecture") not in {None, "ALE3DCNNAutoEncoder"}:
        raise ValueError(f"Unsupported autoencoder architecture {arch.get('architecture')!r}")
    if arch.get("encoder_arch", "plain") != "plain":
        raise ValueError("Only retained plain ALE3DCNNAutoEncoder checkpoints are supported")
    model = ALE3DCNNAutoEncoder(
        output_shape=tuple(arch["output_shape"]),
        in_channels=int(arch.get("in_channels", 1)),
        base_channels=int(arch["base_channels"]),
        num_blocks=int(arch["num_blocks"]),
        latent_dim=int(arch["latent_dim"]),
        dropout=float(arch.get("dropout", 0.1)),
        norm=str(arch.get("norm", "group")),
        pooling=str(arch.get("pooling", "max")),
        encoder_arch="plain",
    )
    try:
        model.load_state_dict(_checkpoint_state(payload), strict=True)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint weights do not match recorded architecture: {error}") from error
    return model.to(device).eval()


def _load_initial_state(model: nn.Module, path: str | Path, architecture: Mapping[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Initialization checkpoint must contain a mapping")
    try:
        checkpoint_arch = _checkpoint_architecture(payload)
    except ValueError:
        checkpoint_arch = {}
    _validate_architecture(checkpoint_arch, architecture)
    try:
        model.load_state_dict(_checkpoint_state(payload), strict=True)
    except RuntimeError as error:
        raise ValueError(f"Initialization checkpoint architecture mismatch: {error}") from error


def _from_run_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = (path / "checkpoints" / "best.pt", path / "best.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No best autoencoder checkpoint found under run {path}")


def _initialization_source(config: AutoencoderTrainConfig, provided_model: bool) -> tuple[str, Path | None]:
    if provided_model:
        return "provided_model", None
    if config.from_run is not None:
        return "from_run", _from_run_checkpoint(config.from_run)
    if config.init_checkpoint is not None:
        return "checkpoint", Path(config.init_checkpoint)
    resolved = config.initialization
    if resolved == "auto":
        resolved = "scratch" if config.variant == "mixed_baseline" else "released_mixed"
    return resolved, None


def _make_loader(
    dataset: Dataset,
    *,
    config: AutoencoderTrainConfig,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=_VolumeCollator(tuple(config.target_shape)),
        pin_memory=_resolve_device(config.device).type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _mean_rows(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in keys}


@torch.no_grad()
def evaluate_autoencoder(
    model: nn.Module,
    data: Dataset | DataLoader,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 64,
    target_shape: tuple[int, int, int] = CNN_AUTOENCODER_SHAPE,
    num_workers: int = 0,
    seed: int = 42,
    include_voxel_auroc: bool = False,
    max_batches: int | None = None,
) -> AutoencoderEvaluation:
    """Evaluate raw autoencoder output and return overall/by-source summaries."""

    resolved_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    if isinstance(data, DataLoader):
        loader = data
    else:
        temporary = AutoencoderTrainConfig(
            epochs=1,
            batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            device=str(resolved_device),
            target_shape=target_shape,
            base_channels=2,
            num_blocks=1,
            latent_dim=4,
            preset="custom",
        )
        loader = _make_loader(
            data,
            config=temporary,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
    was_training = model.training
    model.to(resolved_device).eval()
    all_rows: list[dict[str, float]] = []
    source_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["volume"].to(resolved_device, non_blocking=True)
        prediction = model(target)
        for index, source in enumerate(batch.get("source", ["unknown"] * len(target))):
            row = reconstruction_metrics(
                prediction[index : index + 1],
                target[index : index + 1],
                include_voxel_auroc=include_voxel_auroc,
            )
            row["loss"] = float(F.mse_loss(prediction[index], target[index]))
            all_rows.append(row)
            source_rows[str(source)].append(row)
    model.train(was_training)
    by_source = tuple(
        {"source": source, "n": len(rows), **_mean_rows(rows)}
        for source, rows in sorted(source_rows.items())
    )
    return AutoencoderEvaluation(summary=_mean_rows(all_rows), by_source=by_source, n=len(all_rows))


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp: bool,
    gradient_clip: float | None,
    include_voxel_auroc: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    autocast_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["volume"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            prediction = model(target)
            # Retained recipe: optimize raw decoder output with plain MSE.
            loss = F.mse_loss(prediction, target)
        if autocast_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        metrics = reconstruction_metrics(
            prediction.detach(), target, include_voxel_auroc=include_voxel_auroc
        )
        metrics["loss"] = float(loss.detach())
        rows.extend([metrics] * len(target))
    if not rows:
        raise RuntimeError("Training dataset produced no batches")
    return _mean_rows(rows)


def _record_metrics(
    recorder: MetricRecorder,
    split: str,
    metrics: Mapping[str, float],
    *,
    epoch: int | None,
    n: int,
) -> None:
    for name, value in metrics.items():
        recorder.record(
            split=split,
            metric=f"{split}_{name}",
            value=float(value),
            epoch=epoch,
            n=n,
        )


def train_autoencoder(
    config: AutoencoderTrainConfig,
    *,
    provider: AtlasFreeCNNDataProvider | None = None,
    model: ALE3DCNNAutoEncoder | None = None,
) -> AutoencoderTrainResult:
    """Train and evaluate a mixed or explicitly fine-tuned CNN autoencoder."""

    if not isinstance(config, AutoencoderTrainConfig):
        raise TypeError("config must be an AutoencoderTrainConfig")
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    architecture = config.architecture()
    initialization_source, initialization_path = _initialization_source(config, model is not None)
    recorded_config = asdict(config)
    # ``resume`` is an invocation detail, not an effective training setting.
    # Omitting it keeps the original run manifest compatible when the exact
    # same run is resumed from its last checkpoint.
    recorded_config.pop("resume", None)
    run_config = RunConfig.resolve(
        family="cnn",
        task="autoencoder",
        domain=config.domain,
        variant=config.variant,
        output_root=config.output_root,
        run_id=config.run_id,
        seed=config.seed,
        device=str(device),
        primary_metric=config.primary_metric,
        metric_direction=config.metric_direction,
        data={
            "provider": "atlas_free_cnn_huggingface" if config.split_dir is None else "local_override",
            "split_dir": config.split_dir,
            "volume_path": config.volume_path,
            "domain": config.domain,
            "limit": config.limit,
        },
        resources={"dataset": "neurovlm/atlas_free_cnn_dataset"},
        initialization={
            "source": initialization_source,
            "path": initialization_path,
            "from_run": config.from_run,
        },
        requested=recorded_config,
        effective={
            **recorded_config,
            "device": str(device),
            "architecture": architecture,
            "loss": "raw_mse",
            "metric_clamp": [0.0, 1.0],
            "primary_metric": config.primary_metric,
            "metric_direction": config.metric_direction.value,
            "initialization_source": initialization_source,
        },
    )
    if provider is None:
        provider = AtlasFreeCNNDataProvider(
            domain=config.domain if config.variant == "finetuned" else None,
            limit=config.limit,
            split_dir=config.split_dir,
            volume_path=config.volume_path,
        )
    if model is None:
        model = build_autoencoder(config)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if initialization_source in {"from_run", "checkpoint"}:
        assert initialization_path is not None
        _load_initial_state(model, initialization_path, architecture)
    elif initialization_source == "released_mixed":
        from neurovlm.models import load_model

        released = load_model(family="cnn", task="autoencoder", variant="mixed_baseline")
        try:
            model.load_state_dict(released.state_dict(), strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Released mixed autoencoder architecture does not match the requested fine-tuning architecture: "
                f"{error}"
            ) from error
        del released
    elif initialization_source not in {"scratch", "provided_model"}:
        raise RuntimeError(f"Unhandled initialization source {initialization_source!r}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint_manager = CheckpointManager(
        run_config,
        expected_architecture=architecture,
    )
    start_epoch = 1
    best_metric = float("inf") if config.metric_direction is MetricDirection.MIN else -float("inf")
    early_best = best_metric
    stale_epochs = 0

    with RunContext(run_config) as artifacts:
        recorder = MetricRecorder(run_config)
        if config.resume is not None:
            resumed = checkpoint_manager.load_resume(
                config.resume,
                model=model,
                optimizer=optimizer,
                map_location=device,
            )
            start_epoch = int(resumed.epoch or 0) + 1
            if checkpoint_manager.best_value is not None:
                best_metric = checkpoint_manager.best_value
            extra = resumed.payload.get("extra") or {}
            stale_epochs = int(extra.get("stale_epochs", 0))
            early_best = float(extra.get("early_best", best_metric))

        epochs_completed = start_epoch - 1
        for epoch in range(start_epoch, config.epochs + 1):
            train_loader = _make_loader(
                provider.train,
                config=config,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            )
            val_loader = _make_loader(
                provider.val,
                config=config,
                batch_size=config.eval_batch_size or config.batch_size,
                shuffle=False,
                seed=config.seed,
            )
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                amp=config.amp,
                gradient_clip=config.gradient_clip,
                include_voxel_auroc=config.include_voxel_auroc,
                max_batches=config.max_train_batches,
            )
            validation = evaluate_autoencoder(
                model,
                val_loader,
                device=device,
                include_voxel_auroc=config.include_voxel_auroc,
                max_batches=config.max_eval_batches,
            )
            _record_metrics(
                recorder, "train", train_metrics, epoch=epoch, n=len(provider.train)
            )
            _record_metrics(
                recorder, "val", validation.summary, epoch=epoch, n=validation.n
            )
            recorder.flush()
            selected = (
                float(validation.summary["loss"])
                if config.variant == "mixed_baseline"
                else float(validation.summary["top5_dice"])
            )
            metrics = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in validation.summary.items()},
            }
            improved = checkpoint_manager.save_best(
                model,
                selected,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=architecture,
                aliases=(
                    "best_val_loss.pt"
                    if config.variant == "mixed_baseline"
                    else "best_val_top5_dice.pt",
                ),
            )
            if improved is not None:
                best_metric = selected
            early_improved = (
                selected < early_best - config.early_stopping_min_delta
                if config.metric_direction is MetricDirection.MIN
                else selected > early_best + config.early_stopping_min_delta
            )
            if early_improved:
                early_best = selected
                stale_epochs = 0
            else:
                stale_epochs += 1
            checkpoint_manager.save_last(
                model,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=architecture,
                extra={"stale_epochs": stale_epochs, "early_best": early_best},
            )
            epochs_completed = epoch
            if (
                config.early_stopping_patience is not None
                and stale_epochs >= config.early_stopping_patience
            ):
                break

        best_checkpoint = run_config.run_dir / "checkpoints" / "best.pt"
        last_checkpoint = run_config.run_dir / "checkpoints" / "last.pt"
        checkpoint_manager.load_resume(
            best_checkpoint,
            model=model,
            map_location=device,
        )
        by_source_rows: list[dict[str, Any]] = []
        test_metrics: Mapping[str, float] = {}
        for split, dataset in (("val", provider.val), ("test", provider.test)):
            evaluation = evaluate_autoencoder(
                model,
                dataset,
                device=device,
                batch_size=config.eval_batch_size or config.batch_size,
                target_shape=tuple(config.target_shape),
                num_workers=config.num_workers,
                seed=config.seed,
                include_voxel_auroc=config.include_voxel_auroc,
                max_batches=config.max_eval_batches,
            )
            _record_metrics(recorder, split, evaluation.summary, epoch=None, n=evaluation.n)
            by_source_rows.extend({"split": split, **row} for row in evaluation.by_source)
            if split == "test":
                test_metrics = evaluation.summary
                atomic_write_csv(
                    run_config.run_dir / "metrics" / "test_summary.csv",
                    [{"split": "test", "n": evaluation.n, **evaluation.summary}],
                )
        recorder.flush()
        atomic_write_csv(run_config.run_dir / "metrics" / "by_source.csv", by_source_rows)
        atomic_write_json(
            run_config.run_dir / "metrics" / "training_summary.json",
            {
                "epochs_completed": epochs_completed,
                "best_metric": best_metric,
                "primary_metric": config.primary_metric,
                "test": test_metrics,
            },
        )

    model.eval()

    return AutoencoderTrainResult(
        run_dir=run_config.run_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        epochs_completed=epochs_completed,
        best_metric=best_metric,
        test_metrics=test_metrics,
        model=model,
    )


__all__ = [
    "CNN_AUTOENCODER_PRESET",
    "CNN_AUTOENCODER_SHAPE",
    "AutoencoderEvaluation",
    "AutoencoderTrainConfig",
    "AutoencoderTrainResult",
    "autoencoder_from_checkpoint",
    "build_autoencoder",
    "evaluate_autoencoder",
    "train_autoencoder",
]
