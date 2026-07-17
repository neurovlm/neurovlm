"""Standardized Stage 4 CNN text-to-brain projector training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from neurovlm.atlas_free_dataset import AtlasFreeCNNDataProvider
from neurovlm.atlas_free_text import AtlasFreeContrastiveCollator, AtlasFreeTextEmbeddingLookup
from neurovlm.cnn import CNNTextToBrainModel, GenerativeTextToAELatent
from neurovlm.evaluation.text_to_brain import TextToBrainEvaluation, evaluate_text_to_brain
from neurovlm.pipelines import (
    CheckpointManager,
    MetricDirection,
    MetricRecorder,
    RunConfig,
    RunContext,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from neurovlm.training.autoencoder import (
    CNN_AUTOENCODER_PRESET,
    CNN_AUTOENCODER_SHAPE,
    _resolve_device,
    _seed_everything,
    _seed_worker,
    autoencoder_from_checkpoint,
)


CNN_TEXT_TO_BRAIN_DOMAINS = ("pubmed", "nilearn", "neurovault")


@dataclass(frozen=True)
class TextToBrainTrainConfig:
    """Configuration for one domain-specific Stage 4 projector branch."""

    domain: Literal["pubmed", "nilearn", "neurovault"]
    output_root: str | Path = "runs"
    run_id: str | None = None
    variant: Literal["mixed_baseline", "finetuned"] = "mixed_baseline"
    seed: int = 42
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 64
    eval_batch_size: int | None = None
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    reconstruction_weight: float = 1.0
    latent_weight: float = 1.0
    gradient_clip: float | None = 1.0
    amp: bool = True
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    generated_output_limit: int = 0
    limit: int | None = None
    split_dir: str | Path | None = None
    volume_path: str | Path | None = None
    autoencoder_from_run: str | Path | None = None
    autoencoder_checkpoint: str | Path | None = None
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
    text_in_dim: int = 768
    text_hidden_dim: int = 512

    def __post_init__(self) -> None:
        if self.domain not in CNN_TEXT_TO_BRAIN_DOMAINS:
            raise ValueError(f"domain is required and must be one of {CNN_TEXT_TO_BRAIN_DOMAINS}")
        if self.variant not in {"mixed_baseline", "finetuned"}:
            raise ValueError("variant must be 'mixed_baseline' or 'finetuned'")
        if self.autoencoder_from_run is not None and self.autoencoder_checkpoint is not None:
            raise ValueError("Pass at most one of autoencoder_from_run and autoencoder_checkpoint")
        if self.resume is not None and self.run_id is None:
            raise ValueError("resume requires the original run_id")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        if self.seed < 0 or self.num_workers < 0:
            raise ValueError("seed and num_workers must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.reconstruction_weight < 0 or self.latent_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if not self.reconstruction_weight and not self.latent_weight:
            raise ValueError("at least one loss weight must be non-zero")
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
        if self.generated_output_limit < 0:
            raise ValueError("generated_output_limit must be non-negative")
        if self.text_in_dim != 768 or self.text_hidden_dim != 512 or self.latent_dim != 384:
            raise ValueError("Retained Stage 4 requires the exact 768 -> 512 -> 384 projector")
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
        if self.preset not in {CNN_AUTOENCODER_PRESET, "custom"}:
            raise ValueError(f"Unknown CNN preset {self.preset!r}")
        changed = {
            key: getattr(self, key)
            for key, expected in retained.items()
            if getattr(self, key) != expected
        }
        if self.preset == CNN_AUTOENCODER_PRESET and changed:
            raise ValueError(
                f"preset={CNN_AUTOENCODER_PRESET!r} requires the retained architecture; "
                f"use preset='custom' for overrides: {changed}"
            )

    @property
    def internal_variant(self) -> str:
        return f"mixed_to_{self.domain}" if self.variant == "mixed_baseline" else self.domain

    @property
    def primary_metric(self) -> str:
        return "val_top5_dice"

    def architecture(self) -> dict[str, Any]:
        return {
            "architecture": "GenerativeTextToAELatent",
            "text_projection": {
                "in_dim": self.text_in_dim,
                "hidden_dim": self.text_hidden_dim,
                "latent_dim": self.latent_dim,
            },
            "autoencoder": {
                "architecture": "ALE3DCNNAutoEncoder",
                "preset": self.preset,
                "output_shape": tuple(int(value) for value in self.target_shape),
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "num_blocks": self.num_blocks,
                "latent_dim": self.latent_dim,
                "dropout": self.dropout,
                "norm": self.norm,
                "pooling": self.pooling,
                "encoder_arch": "plain",
            },
        }


@dataclass(frozen=True)
class TextToBrainTrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    epochs_completed: int
    best_metric: float
    test_metrics: Mapping[str, float]
    model: CNNTextToBrainModel


def _from_run_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path / "checkpoints" / "best.pt", path / "best.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No best autoencoder checkpoint found under run {path}")


def _autoencoder_source(config: TextToBrainTrainConfig) -> dict[str, Any]:
    path: Path | None = None
    kind: str
    if config.autoencoder_from_run is not None:
        kind, path = "from_run", _from_run_checkpoint(config.autoencoder_from_run)
    elif config.autoencoder_checkpoint is not None:
        kind, path = "checkpoint", Path(config.autoencoder_checkpoint)
    else:
        kind = "released"
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Autoencoder checkpoint does not exist: {path}")
        return {
            "kind": kind,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "variant": config.variant,
            "domain": config.domain if config.variant == "finetuned" else None,
        }
    return {
        "kind": "released",
        "family": "cnn",
        "task": "autoencoder",
        "variant": config.variant,
        "domain": config.domain if config.variant == "finetuned" else None,
        "loader_variant": "mixed" if config.variant == "mixed_baseline" else config.domain,
    }


def _validate_autoencoder(autoencoder: nn.Module, config: TextToBrainTrainConfig) -> None:
    encoder = getattr(autoencoder, "encoder", None)
    decoder = getattr(autoencoder, "decoder", None)
    if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
        raise TypeError("autoencoder must expose .encoder and .decoder torch modules")
    actual_shape = tuple(getattr(decoder, "output_shape", ()))
    actual_latent = getattr(decoder, "latent_dim", None)
    if actual_shape != tuple(config.target_shape) or actual_latent != config.latent_dim:
        raise ValueError(
            "Autoencoder architecture mismatch: "
            f"expected output_shape={tuple(config.target_shape)!r}, latent_dim={config.latent_dim}; "
            f"got output_shape={actual_shape!r}, latent_dim={actual_latent!r}"
        )


def _freeze_autoencoder(autoencoder: nn.Module) -> None:
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)


def build_text_to_brain(
    config: TextToBrainTrainConfig,
    *,
    autoencoder: nn.Module | None = None,
    text_projection: nn.Module | None = None,
) -> CNNTextToBrainModel:
    """Build a fresh projector through the exact frozen AE encoder/decoder."""

    if autoencoder is None:
        source = _autoencoder_source(config)
        if source["kind"] in {"from_run", "checkpoint"}:
            autoencoder = autoencoder_from_checkpoint(source["path"])
        else:
            from neurovlm.models import load_model

            kwargs: dict[str, Any] = {
                "family": "cnn",
                "task": "autoencoder",
                "variant": config.variant,
            }
            if config.variant == "finetuned":
                kwargs["domain"] = config.domain
            autoencoder = load_model(**kwargs)
    _validate_autoencoder(autoencoder, config)
    _freeze_autoencoder(autoencoder)
    if text_projection is None:
        text_projection = GenerativeTextToAELatent(
            in_dim=config.text_in_dim,
            hidden_dim=config.text_hidden_dim,
            latent_dim=config.latent_dim,
        )
    return CNNTextToBrainModel(text_projection, autoencoder)


def _validate_checkpoint_architecture(
    payload: Mapping[str, Any], autoencoder: nn.Module
) -> None:
    architecture = payload.get("architecture") or {}
    if architecture and architecture.get("architecture") != "GenerativeTextToAELatent":
        raise ValueError("Checkpoint is not a retained GenerativeTextToAELatent projector")
    projection = architecture.get("text_projection") or {}
    expected_projection = {"in_dim": 768, "hidden_dim": 512, "latent_dim": 384}
    mismatches = [
        f"{key}: expected {value!r}, got {projection.get(key)!r}"
        for key, value in expected_projection.items()
        if projection and projection.get(key) != value
    ]
    ae_arch = architecture.get("autoencoder") or {}
    decoder = getattr(autoencoder, "decoder", None)
    if ae_arch:
        actual_shape = tuple(getattr(decoder, "output_shape", ()))
        actual_latent = getattr(decoder, "latent_dim", None)
        if tuple(ae_arch.get("output_shape", ())) != actual_shape:
            mismatches.append(
                f"autoencoder.output_shape: expected {tuple(ae_arch.get('output_shape', ()))!r}, "
                f"got {actual_shape!r}"
            )
        if ae_arch.get("latent_dim") != actual_latent:
            mismatches.append(
                f"autoencoder.latent_dim: expected {ae_arch.get('latent_dim')!r}, got {actual_latent!r}"
            )
    if mismatches:
        raise ValueError("Text-to-brain checkpoint architecture mismatch: " + "; ".join(mismatches))


def text_to_brain_from_checkpoint(
    checkpoint: str | Path,
    *,
    autoencoder: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> CNNTextToBrainModel:
    """Reload a standardized Stage 4 projector and its recorded frozen AE."""

    from neurovlm.cnn import text_to_brain_from_payload

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Text-to-brain checkpoint payload must be a mapping")
    source = (payload.get("extra") or {}).get("autoencoder_source") or payload.get(
        "autoencoder_source"
    )
    if autoencoder is None:
        if not isinstance(source, Mapping):
            raise ValueError(
                "Checkpoint does not record an autoencoder source; pass autoencoder= for legacy payloads"
            )
        if source.get("kind") in {"from_run", "checkpoint"}:
            path = Path(str(source.get("path", "")))
            if not path.is_file():
                raise FileNotFoundError(f"Recorded autoencoder checkpoint is unavailable: {path}")
            expected_hash = source.get("sha256")
            if expected_hash and sha256_file(path) != expected_hash:
                raise ValueError("Recorded autoencoder checkpoint SHA256 mismatch")
            autoencoder = autoencoder_from_checkpoint(path)
        elif source.get("kind") == "released":
            from neurovlm.models import load_model

            kwargs: dict[str, Any] = {
                "family": "cnn",
                "task": "autoencoder",
                "variant": str(source.get("variant")),
            }
            if source.get("variant") == "finetuned":
                kwargs["domain"] = str(source.get("domain"))
            autoencoder = load_model(**kwargs)
        else:
            raise ValueError(f"Unsupported recorded autoencoder source: {source!r}")
    _validate_checkpoint_architecture(payload, autoencoder)
    return text_to_brain_from_payload(payload, autoencoder).to(device)


def text_to_brain_loss(
    prediction: Tensor,
    target: Tensor,
    brain_latent: Tensor,
    text_latent: Tensor,
    *,
    reconstruction_weight: float = 1.0,
    latent_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Raw decoder MSE plus detached-target latent alignment MSE."""

    latent_mse = F.mse_loss(text_latent, brain_latent.detach())
    reconstruction_mse = F.mse_loss(prediction, target)
    latent_cosine = F.cosine_similarity(
        text_latent, brain_latent.detach(), dim=1, eps=1e-8
    ).mean()
    total = reconstruction_weight * reconstruction_mse + latent_weight * latent_mse
    return total, {
        "loss": total,
        "total": total,
        "latent_mse": latent_mse,
        "latent_cosine": latent_cosine,
        "raw_reconstruction_mse": reconstruction_mse,
        "reconstruction_mse": reconstruction_mse,
    }


def _make_loader(
    dataset: Dataset,
    lookup: AtlasFreeTextEmbeddingLookup,
    config: TextToBrainTrainConfig,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    rows = getattr(dataset, "rows", None)
    if rows is not None:
        lookup.validate_dataset(rows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=AtlasFreeContrastiveCollator(lookup, tuple(config.target_shape)),
        pin_memory=_resolve_device(config.device).type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def _train_epoch(
    model: CNNTextToBrainModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    config: TextToBrainTrainConfig,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    model.text_projection.train()
    model.autoencoder.eval()
    autocast_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        target = batch["volume"].to(device, non_blocking=True)
        text = batch["text_embedding"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            brain_z = model.autoencoder.encoder(target)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            text_z = model.text_projection(text)
            prediction = model.autoencoder.decoder(text_z)
            loss, parts = text_to_brain_loss(
                prediction,
                target,
                brain_z,
                text_z,
                reconstruction_weight=config.reconstruction_weight,
                latent_weight=config.latent_weight,
            )
        if autocast_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.text_projection.parameters(), config.gradient_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.text_projection.parameters(), config.gradient_clip
                )
            optimizer.step()
        batch_n = len(target)
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_n
        count += batch_n
    if not count:
        raise RuntimeError("Training dataset produced no batches")
    return {name: value / count for name, value in totals.items()}, count


def _record(
    recorder: MetricRecorder,
    split: str,
    metrics: Mapping[str, float],
    *,
    epoch: int | None,
    n: int,
) -> None:
    for name, value in metrics.items():
        recorder.record(
            split=split, metric=f"{split}_{name}", value=float(value), epoch=epoch, n=n
        )


def train_text_to_brain(
    config: TextToBrainTrainConfig,
    *,
    provider: AtlasFreeCNNDataProvider | None = None,
    lookup: AtlasFreeTextEmbeddingLookup | None = None,
    model: CNNTextToBrainModel | None = None,
    semantic_evaluator: Callable[..., Mapping[str, float]] | None = None,
) -> TextToBrainTrainResult:
    """Train a fresh Stage 4 projector with a frozen, provenance-bound AE."""

    if not isinstance(config, TextToBrainTrainConfig):
        raise TypeError("config must be a TextToBrainTrainConfig")
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    architecture = config.architecture()
    source = _autoencoder_source(config)
    if model is not None:
        source = {"kind": "provided_model"}
    recorded = asdict(config)
    recorded.pop("resume", None)
    # ``epochs`` is a run-control target rather than model/data identity. It
    # remains in requested.json and the training summary, but excluding it
    # from the effective-config identity permits an interrupted run to extend
    # its target epoch while every substantive setting remains guarded by the
    # manifest hash.
    effective_recorded = dict(recorded)
    effective_recorded.pop("epochs", None)
    run_config = RunConfig.resolve(
        family="cnn",
        task="text_to_brain",
        domain=config.domain,
        variant=config.variant,
        output_root=config.output_root,
        run_id=config.run_id,
        seed=config.seed,
        device=str(device),
        primary_metric=config.primary_metric,
        metric_direction=MetricDirection.MAX,
        data={
            "provider": "atlas_free_cnn_huggingface" if config.split_dir is None else "local_override",
            "domain": config.domain,
            "split_dir": config.split_dir,
            "volume_path": config.volume_path,
            "limit": config.limit,
        },
        resources={
            "dataset": "neurovlm/atlas_free_cnn_dataset",
            "text_embeddings": "text_embeddings/specter2_stage3_stage4_emptycentered_unitnorm.pt",
        },
        initialization={"autoencoder": source, "projector": "fresh"},
        requested=recorded,
        effective={
            **effective_recorded,
            "device": str(device),
            "architecture": architecture,
            "internal_variant": config.internal_variant,
            "loss": "raw_reconstruction_mse_plus_latent_mse",
            "text_preprocessing": "first_positive_empty_string_centered_l2_unit_normalized",
            "autoencoder_frozen": True,
            "primary_metric": config.primary_metric,
        },
    )
    if provider is None:
        provider = AtlasFreeCNNDataProvider(
            domain=config.domain,
            limit=config.limit,
            split_dir=config.split_dir,
            volume_path=config.volume_path,
        )
    if lookup is None:
        lookup = AtlasFreeTextEmbeddingLookup.published()
    if model is None:
        model = build_text_to_brain(config)
    _validate_autoencoder(model.autoencoder, config)
    _freeze_autoencoder(model.autoencoder)
    model = model.to(device)
    for parameter in model.text_projection.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.text_projection.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    manager = CheckpointManager(run_config, expected_architecture=architecture)
    start_epoch = 1
    best_metric = -float("inf")
    early_best = -float("inf")
    stale_epochs = 0

    with RunContext(run_config):
        recorder = MetricRecorder(run_config)
        if config.resume is not None:
            resumed = manager.load_resume(
                config.resume,
                model=model.text_projection,
                optimizer=optimizer,
                map_location=device,
            )
            start_epoch = int(resumed.epoch or 0) + 1
            if manager.best_value is not None:
                best_metric = manager.best_value
            extra = resumed.payload.get("extra") or {}
            early_best = float(extra.get("early_best", best_metric))
            stale_epochs = int(extra.get("stale_epochs", 0))

        epochs_completed = start_epoch - 1
        for epoch in range(start_epoch, config.epochs + 1):
            train_loader = _make_loader(
                provider.train,
                lookup,
                config,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            )
            train_metrics, train_n = _train_epoch(
                model, train_loader, optimizer, config=config, device=device
            )
            validation = evaluate_text_to_brain(
                model,
                provider.val,
                lookup=lookup,
                device=device,
                batch_size=config.eval_batch_size or config.batch_size,
                target_shape=tuple(config.target_shape),
                num_workers=config.num_workers,
                seed=config.seed,
                max_batches=config.max_eval_batches,
                reconstruction_weight=config.reconstruction_weight,
                latent_weight=config.latent_weight,
                semantic_evaluator=semantic_evaluator,
            )
            _record(recorder, "train", train_metrics, epoch=epoch, n=train_n)
            _record(recorder, "val", validation.summary, epoch=epoch, n=validation.n)
            recorder.flush()
            selected = float(validation.summary["top5_dice"])
            metrics = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in validation.summary.items()},
            }
            checkpoint_extra = {
                "autoencoder_source": source,
                "early_best": early_best,
                "stale_epochs": stale_epochs,
            }
            if manager.save_best(
                model.text_projection,
                selected,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=architecture,
                extra=checkpoint_extra,
                aliases=("best_val_top5_dice.pt",),
            ) is not None:
                best_metric = selected
            if selected > early_best + config.early_stopping_min_delta:
                early_best = selected
                stale_epochs = 0
            else:
                stale_epochs += 1
            manager.save_last(
                model.text_projection,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=architecture,
                extra={
                    "autoencoder_source": source,
                    "early_best": early_best,
                    "stale_epochs": stale_epochs,
                },
            )
            epochs_completed = epoch
            if (
                config.early_stopping_patience is not None
                and stale_epochs >= config.early_stopping_patience
            ):
                break

        best_checkpoint = run_config.run_dir / "checkpoints" / "best.pt"
        last_checkpoint = run_config.run_dir / "checkpoints" / "last.pt"
        manager.load_resume(best_checkpoint, model=model.text_projection, map_location=device)
        by_source_rows: list[dict[str, Any]] = []
        by_sample_rows: list[dict[str, Any]] = []
        test_metrics: Mapping[str, float] = {}
        for split, dataset in (("val", provider.val), ("test", provider.test)):
            evaluation = evaluate_text_to_brain(
                model,
                dataset,
                lookup=lookup,
                device=device,
                batch_size=config.eval_batch_size or config.batch_size,
                target_shape=tuple(config.target_shape),
                num_workers=config.num_workers,
                seed=config.seed,
                max_batches=config.max_eval_batches,
                generated_limit=config.generated_output_limit,
                reconstruction_weight=config.reconstruction_weight,
                latent_weight=config.latent_weight,
                semantic_evaluator=semantic_evaluator,
            )
            _record(recorder, split, evaluation.summary, epoch=None, n=evaluation.n)
            by_source_rows.extend({"split": split, **row} for row in evaluation.by_source)
            by_sample_rows.extend({"split": split, **row} for row in evaluation.by_sample)
            if evaluation.generated:
                torch.save(
                    {"split": split, "items": evaluation.generated},
                    run_config.run_dir / "generated_maps" / f"{split}_predictions.pt",
                )
            if split == "test":
                test_metrics = evaluation.summary
                atomic_write_csv(
                    run_config.run_dir / "metrics" / "test_summary.csv",
                    [{"split": split, "n": evaluation.n, **evaluation.summary}],
                )
        recorder.flush()
        atomic_write_csv(run_config.run_dir / "metrics" / "by_source.csv", by_source_rows)
        atomic_write_csv(run_config.run_dir / "metrics" / "by_sample.csv", by_sample_rows)
        atomic_write_json(
            run_config.run_dir / "metrics" / "training_summary.json",
            {
                "epochs_completed": epochs_completed,
                "epochs_requested": config.epochs,
                "best_metric": best_metric,
                "primary_metric": config.primary_metric,
                "internal_variant": config.internal_variant,
                "autoencoder_source": source,
                "test": test_metrics,
            },
        )
    model.eval()
    return TextToBrainTrainResult(
        run_dir=run_config.run_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        epochs_completed=epochs_completed,
        best_metric=best_metric,
        test_metrics=test_metrics,
        model=model,
    )


__all__ = [
    "CNN_TEXT_TO_BRAIN_DOMAINS",
    "TextToBrainEvaluation",
    "TextToBrainTrainConfig",
    "TextToBrainTrainResult",
    "build_text_to_brain",
    "evaluate_text_to_brain",
    "text_to_brain_from_checkpoint",
    "text_to_brain_loss",
    "train_text_to_brain",
]
