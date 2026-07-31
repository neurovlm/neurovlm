"""Standardized Stage 3 CNN contrastive training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neurovlm.cnn.models import CNNContrastiveModel
from neurovlm.data.atlas_free_dataset import AtlasFreeCNNDataProvider
from neurovlm.data.atlas_free_text import (
    AtlasFreeContrastiveCollator,
    AtlasFreeTextEmbeddingLookup,
)
from neurovlm.evaluation.contrastive import ContrastiveEvaluation, evaluate_contrastive
from neurovlm.models.losses import InfoNCELoss
from neurovlm.pipelines import (
    CheckpointManager,
    MetricDirection,
    MetricRecorder,
    RunConfig,
    RunContext,
    atomic_write_csv,
    atomic_write_json,
)
from neurovlm.training.autoencoder import (
    CNN_AUTOENCODER_PRESET,
    CNN_AUTOENCODER_SHAPE,
    _resolve_device,
    _seed_everything,
    _seed_worker,
    autoencoder_from_checkpoint,
)


CNN_CONTRASTIVE_DOMAINS = ("pubmed", "nilearn", "neurovault")


@dataclass(frozen=True)
class ContrastiveTrainConfig:
    """Configuration for one domain-specific Stage 3 branch.

    The default ``mixed_baseline`` initializes from the released mixed AE;
    ``finetuned`` initializes from the released AE matching ``domain``.
    ``from_run`` and ``init_checkpoint`` explicitly replace that AE source.
    """

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
    brain_learning_rate: float = 1e-4
    projection_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    gradient_clip: float | None = 1.0
    amp: bool = True
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    limit: int | None = None
    split_dir: str | Path | None = None
    volume_path: str | Path | None = None
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
        if self.domain not in CNN_CONTRASTIVE_DOMAINS:
            raise ValueError(f"domain is required and must be one of {CNN_CONTRASTIVE_DOMAINS}")
        if self.variant not in {"mixed_baseline", "finetuned"}:
            raise ValueError("variant must be 'mixed_baseline' or 'finetuned'")
        if self.from_run is not None and self.init_checkpoint is not None:
            raise ValueError("Pass at most one of from_run and init_checkpoint")
        if self.resume is not None and self.run_id is None:
            raise ValueError("resume requires the original run_id")
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be positive and batch_size must be at least 2 for InfoNCE")
        if self.eval_batch_size is not None and self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        if self.seed < 0 or self.num_workers < 0:
            raise ValueError("seed and num_workers must be non-negative")
        if self.brain_learning_rate <= 0 or self.projection_learning_rate <= 0:
            raise ValueError("brain and projection learning rates must be positive")
        if self.weight_decay < 0 or self.temperature <= 0:
            raise ValueError("weight_decay must be non-negative and temperature positive")
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
        if self.latent_dim != 384:
            raise ValueError("The pretrained MLP InfoNCE text head requires latent_dim=384")
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
        return "val_mean_normalized_k_recall_curve_auc"

    def architecture(self) -> dict[str, Any]:
        return {
            "architecture": "CNNContrastiveModel",
            "brain_encoder": "ALE3DCNNEncoder",
            "model": "ale_3dcnn",
            "preset": self.preset,
            "target_shape": tuple(int(value) for value in self.target_shape),
            "in_channels": int(self.in_channels),
            "base_channels": int(self.base_channels),
            "num_blocks": int(self.num_blocks),
            "out_dim": int(self.latent_dim),
            "latent_dim": int(self.latent_dim),
            "dropout": float(self.dropout),
            "norm": self.norm,
            "pooling": self.pooling,
            "text_in_dim": 768,
            "text_hidden_dim": 512,
            "text_out_dim": 384,
            "encoder_arch": "plain",
        }


@dataclass(frozen=True)
class ContrastiveTrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    epochs_completed: int
    best_metric: float
    test_metrics: Mapping[str, float]
    model: CNNContrastiveModel


def _from_run_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path / "checkpoints" / "best.pt", path / "best.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No best autoencoder checkpoint found under run {path}")


def _ae_initialization(config: ContrastiveTrainConfig) -> tuple[str, Path | None]:
    if config.from_run is not None:
        return "from_run", _from_run_checkpoint(config.from_run)
    if config.init_checkpoint is not None:
        return "checkpoint", Path(config.init_checkpoint)
    return (
        "released_mixed_autoencoder"
        if config.variant == "mixed_baseline"
        else f"released_{config.domain}_autoencoder",
        None,
    )


def build_contrastive(
    config: ContrastiveTrainConfig,
    *,
    autoencoder: nn.Module | None = None,
    text_projection: nn.Module | None = None,
) -> CNNContrastiveModel:
    """Build the trainable composite from released or explicit AE weights."""

    if autoencoder is None:
        _, path = _ae_initialization(config)
        if path is not None:
            autoencoder = autoencoder_from_checkpoint(path)
        else:
            from neurovlm.models.base import load_model

            kwargs: dict[str, Any] = {
                "family": "cnn",
                "task": "autoencoder",
                "variant": config.variant,
            }
            if config.variant == "finetuned":
                kwargs["domain"] = config.domain
            autoencoder = load_model(**kwargs)
    encoder = getattr(autoencoder, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise TypeError("autoencoder must expose a torch module at .encoder")
    features = getattr(encoder, "features", None)
    projection = getattr(encoder, "proj", None)
    if isinstance(features, nn.Sequential) and len(features):
        first_conv = getattr(features[0], "conv", None)
        actual = {
            "in_channels": getattr(first_conv, "in_channels", None),
            "base_channels": getattr(first_conv, "out_channels", None),
            "num_blocks": len(features),
            "latent_dim": getattr(projection, "out_features", None),
        }
        expected = {
            "in_channels": config.in_channels,
            "base_channels": config.base_channels,
            "num_blocks": config.num_blocks,
            "latent_dim": config.latent_dim,
        }
        mismatches = [
            f"{key}: expected {expected[key]!r}, got {actual[key]!r}"
            for key in expected
            if actual[key] != expected[key]
        ]
        if mismatches:
            raise ValueError("Autoencoder encoder architecture mismatch: " + "; ".join(mismatches))
    if text_projection is None:
        from neurovlm.models.base import ProjHead

        text_projection = ProjHead.from_pretrained("text_infonce")
    model = CNNContrastiveModel(encoder, text_projection)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def contrastive_from_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> CNNContrastiveModel:
    """Reload a standardized or legacy Stage 3 checkpoint."""

    from neurovlm.cnn.models import contrastive_from_payload

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Contrastive checkpoint payload must be a mapping")
    return contrastive_from_payload(payload).to(device)


def _make_loader(
    dataset: Dataset,
    lookup: AtlasFreeTextEmbeddingLookup,
    config: ContrastiveTrainConfig,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    rows = getattr(dataset, "rows", None)
    if rows is not None:
        lookup.validate_dataset(rows)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # A singleton training batch has no InfoNCE negatives and produces a
        # zero-signal objective. Evaluation retains every sample.
        drop_last=shuffle,
        num_workers=config.num_workers,
        collate_fn=AtlasFreeContrastiveCollator(lookup, tuple(config.target_shape)),
        pin_memory=_resolve_device(config.device).type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _train_epoch(
    model: CNNContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    temperature: float,
    amp: bool,
    gradient_clip: float | None,
    max_batches: int | None,
) -> tuple[float, int]:
    model.train()
    loss_fn = InfoNCELoss(temperature)
    autocast_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
    weighted_loss = 0.0
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        volume = batch["volume"].to(device, non_blocking=True)
        text_embedding = batch["text_embedding"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            brain, text = model(volume, text_embedding)
            loss = loss_fn(brain, text)
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
        batch_n = len(volume)
        weighted_loss += float(loss.detach()) * batch_n
        count += batch_n
    if count == 0:
        raise RuntimeError("Training dataset produced no batches")
    return weighted_loss / count, count


def _record(
    recorder: MetricRecorder,
    split: str,
    metrics: Mapping[str, float],
    *,
    epoch: int | None,
    n: int,
) -> None:
    for name, value in metrics.items():
        recorder.record(split=split, metric=f"{split}_{name}", value=float(value), epoch=epoch, n=n)


def train_contrastive(
    config: ContrastiveTrainConfig,
    *,
    provider: AtlasFreeCNNDataProvider | None = None,
    lookup: AtlasFreeTextEmbeddingLookup | None = None,
    model: CNNContrastiveModel | None = None,
) -> ContrastiveTrainResult:
    """Train, select, and fully evaluate one CNN contrastive branch."""

    if not isinstance(config, ContrastiveTrainConfig):
        raise TypeError("config must be a ContrastiveTrainConfig")
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    architecture = config.architecture()
    initialization_source, initialization_path = _ae_initialization(config)
    if model is not None:
        initialization_source, initialization_path = "provided_model", None
    recorded = asdict(config)
    recorded.pop("resume", None)
    run_config = RunConfig.resolve(
        family="cnn",
        task="contrastive",
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
            "text_embeddings": "specter2_stage3_stage4_emptycentered_unitnorm.pt",
            "pretrained_text_projection": "proj_head_text_infonce",
        },
        initialization={
            "source": initialization_source,
            "path": initialization_path,
            "from_run": config.from_run,
        },
        requested=recorded,
        effective={
            **recorded,
            "device": str(device),
            "architecture": architecture,
            "internal_variant": config.internal_variant,
            "loss": "symmetric_infonce",
            "temperature": config.temperature,
            "text_preprocessing": "empty_string_centered_l2_unit_normalized",
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
        model = build_contrastive(config)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.brain_encoder.parameters(), "lr": config.brain_learning_rate},
            {"params": model.text_projection.parameters(), "lr": config.projection_learning_rate},
        ],
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
                config.resume, model=model, optimizer=optimizer, map_location=device
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
            train_loss, train_n = _train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                temperature=config.temperature,
                amp=config.amp,
                gradient_clip=config.gradient_clip,
                max_batches=config.max_train_batches,
            )
            validation = evaluate_contrastive(
                model,
                provider.val,
                lookup=lookup,
                device=device,
                batch_size=config.eval_batch_size or config.batch_size,
                target_shape=tuple(config.target_shape),
                num_workers=config.num_workers,
                seed=config.seed,
                temperature=config.temperature,
                max_batches=config.max_eval_batches,
            )
            _record(recorder, "train", {"loss": train_loss}, epoch=epoch, n=train_n)
            _record(recorder, "val", validation.summary, epoch=epoch, n=validation.n)
            recorder.flush()
            selected = float(validation.summary["mean_normalized_k_recall_curve_auc"])
            metrics = {"train_loss": train_loss, **{f"val_{k}": v for k, v in validation.summary.items()}}
            if manager.save_best(
                model,
                selected,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=architecture,
                aliases=("best_val_normalized_recall_auc.pt",),
            ) is not None:
                best_metric = selected
            if selected > early_best + config.early_stopping_min_delta:
                early_best = selected
                stale_epochs = 0
            else:
                stale_epochs += 1
            manager.save_last(
                model,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=architecture,
                extra={"early_best": early_best, "stale_epochs": stale_epochs},
            )
            epochs_completed = epoch
            if (
                config.early_stopping_patience is not None
                and stale_epochs >= config.early_stopping_patience
            ):
                break

        best_checkpoint = run_config.run_dir / "checkpoints" / "best.pt"
        last_checkpoint = run_config.run_dir / "checkpoints" / "last.pt"
        manager.load_resume(best_checkpoint, model=model, map_location=device)
        curve_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        test_metrics: Mapping[str, float] = {}
        for split, dataset in (("val", provider.val), ("test", provider.test)):
            evaluation = evaluate_contrastive(
                model,
                dataset,
                lookup=lookup,
                device=device,
                batch_size=config.eval_batch_size or config.batch_size,
                target_shape=tuple(config.target_shape),
                num_workers=config.num_workers,
                seed=config.seed,
                temperature=config.temperature,
                max_batches=config.max_eval_batches,
            )
            _record(recorder, split, evaluation.summary, epoch=None, n=evaluation.n)
            curve_rows.extend({"split": split, **row} for row in evaluation.recall_curves)
            source_rows.extend({"split": split, **row} for row in evaluation.by_source)
            if split == "test":
                test_metrics = evaluation.summary
                atomic_write_csv(
                    run_config.run_dir / "metrics" / "test_summary.csv",
                    [{"split": split, "n": evaluation.n, **evaluation.summary}],
                )
        recorder.flush()
        atomic_write_csv(run_config.run_dir / "metrics" / "recall_curves.csv", curve_rows)
        atomic_write_csv(run_config.run_dir / "metrics" / "by_source.csv", source_rows)
        atomic_write_json(
            run_config.run_dir / "metrics" / "training_summary.json",
            {
                "epochs_completed": epochs_completed,
                "best_metric": best_metric,
                "primary_metric": config.primary_metric,
                "internal_variant": config.internal_variant,
                "test": test_metrics,
            },
        )
    model.eval()
    return ContrastiveTrainResult(
        run_dir=run_config.run_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        epochs_completed=epochs_completed,
        best_metric=best_metric,
        test_metrics=test_metrics,
        model=model,
    )


__all__ = [
    "CNN_CONTRASTIVE_DOMAINS",
    "ContrastiveTrainConfig",
    "ContrastiveTrainResult",
    "build_contrastive",
    "contrastive_from_checkpoint",
    "train_contrastive",
]
