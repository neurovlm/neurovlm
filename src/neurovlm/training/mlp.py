"""Standardized training for the retained flat-map NeuroVLM models.

This module complements, rather than replaces, the legacy :class:`Trainer`.
The old API remains useful for small interactive fits; these entry points add
reproducible run artifacts, safe resume, best-checkpoint selection, and full
split evaluation.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from neurovlm.evaluation.mlp import (
    MLPEvaluation,
    MLPContrastiveEvaluation,
    _autoencoder_input,
    _paired_input,
    evaluate_mlp_autoencoder,
    evaluate_mlp_contrastive,
    evaluate_mlp_text_to_brain,
)
from neurovlm.models.base import NeuroAutoEncoder, ProjHead
from neurovlm.models.losses import InfoNCELoss
from neurovlm.pipelines import (
    CheckpointManager,
    MetricDirection,
    MetricRecorder,
    RunConfig,
    RunContext,
)
from neurovlm.pipelines.serialization import atomic_write_csv, atomic_write_json


MLP_RETAINED_DIMS = (28_542, 1024, 512, 384)


def _device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _splits(provider: Any) -> tuple[Any, Any, Any]:
    if isinstance(provider, Mapping):
        values = tuple(provider.get(name) for name in ("train", "val", "test"))
    else:
        values = tuple(getattr(provider, name, None) for name in ("train", "val", "test"))
    if any(value is None for value in values):
        raise ValueError("provider must expose train, val, and test datasets/loaders")
    return values  # type: ignore[return-value]


def _loader(
    data: Dataset | DataLoader,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
    )


def _n(data: Any) -> int | None:
    try:
        return len(data.dataset if isinstance(data, DataLoader) else data)
    except TypeError:
        return None


@dataclass(frozen=True)
class _BaseConfig:
    output_root: str | Path = "runs"
    run_id: str | None = None
    seed: int = 42
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 256
    eval_batch_size: int | None = None
    num_workers: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    gradient_clip: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    resume: str | Path | None = None

    def _validate_base(self) -> None:
        if self.seed < 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("seed must be non-negative and epochs/batch_size must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")


@dataclass(frozen=True)
class MLPAutoencoderTrainConfig(_BaseConfig):
    preset: str = "retained"
    dim_neuro: int = 28_542
    dim_h0: int = 1024
    dim_h1: int = 512
    dim_latent: int = 384
    primary_metric: str = "val_loss"
    metric_direction: MetricDirection = MetricDirection.MIN

    def __post_init__(self) -> None:
        self._validate_base()
        dims = (self.dim_neuro, self.dim_h0, self.dim_h1, self.dim_latent)
        if self.preset not in {"retained", "custom"}:
            raise ValueError("preset must be 'retained' or 'custom'")
        if self.preset == "retained" and dims != MLP_RETAINED_DIMS:
            raise ValueError("Changing retained MLP dimensions requires preset='custom'")
        if min(dims) < 1:
            raise ValueError("all architecture dimensions must be positive")

    def architecture(self) -> dict[str, Any]:
        return {"architecture": "NeuroAutoEncoder", "preset": self.preset, "out": "logit",
                "dim_neuro": self.dim_neuro, "dim_h0": self.dim_h0,
                "dim_h1": self.dim_h1, "dim_latent": self.dim_latent}


@dataclass(frozen=True)
class MLPTextToBrainTrainConfig(_BaseConfig):
    preset: str = "retained"
    text_dim: int = 768
    hidden_dim: int = 512
    latent_dim: int = 384
    brain_dim: int = 28_542
    primary_metric: str = "val_loss"
    metric_direction: MetricDirection = MetricDirection.MIN

    def __post_init__(self) -> None:
        self._validate_base()
        dims = (self.text_dim, self.hidden_dim, self.latent_dim, self.brain_dim)
        if self.preset not in {"retained", "custom"}:
            raise ValueError("preset must be 'retained' or 'custom'")
        if self.preset == "retained" and dims != (768, 512, 384, 28_542):
            raise ValueError("Changing retained MLP dimensions requires preset='custom'")
        if min(dims) < 1:
            raise ValueError("all architecture dimensions must be positive")

    def architecture(self) -> dict[str, Any]:
        return {"architecture": "MLPTextToBrain", "preset": self.preset,
                "text_dim": self.text_dim, "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim, "brain_dim": self.brain_dim,
                "loss": "latent_mse"}


@dataclass(frozen=True)
class MLPContrastiveTrainConfig(_BaseConfig):
    preset: str = "retained"
    text_dim: int = 768
    text_hidden_dim: int = 512
    brain_dim: int = 384
    brain_hidden_dim: int = 384
    shared_dim: int = 384
    temperature: float = 0.07
    initialize_text_from_mse: bool = True
    primary_metric: str = "val_loss"
    metric_direction: MetricDirection = MetricDirection.MIN

    def __post_init__(self) -> None:
        self._validate_base()
        dims = (self.text_dim, self.text_hidden_dim, self.brain_dim, self.brain_hidden_dim, self.shared_dim)
        if self.preset not in {"retained", "custom"}:
            raise ValueError("preset must be 'retained' or 'custom'")
        if self.preset == "retained" and dims != (768, 512, 384, 384, 384):
            raise ValueError("Changing retained MLP dimensions requires preset='custom'")
        if min(dims) < 1 or self.temperature <= 0:
            raise ValueError("dimensions and temperature must be positive")
        if self.batch_size < 2:
            raise ValueError("contrastive batch_size must be at least two for InfoNCE")

    def architecture(self) -> dict[str, Any]:
        return {"architecture": "MLPContrastiveModel", "preset": self.preset,
                "text_dim": self.text_dim, "text_hidden_dim": self.text_hidden_dim,
                "brain_dim": self.brain_dim, "brain_hidden_dim": self.brain_hidden_dim,
                "shared_dim": self.shared_dim, "temperature": self.temperature}


@dataclass(frozen=True)
class MLPBrainToTextRetrievalTrainConfig(MLPContrastiveTrainConfig):
    """Train the retained contrastive heads for brain-to-text retrieval.

    The optimization objective remains the symmetric retained InfoNCE loss.
    Only checkpoint selection changes: this task selects the image/brain-to-text
    (``i2t``) full recall-curve AUC instead of validation loss.
    """

    primary_metric: str = "val_i2t_normalized_k_recall_curve_auc"
    metric_direction: MetricDirection = MetricDirection.MAX


class MLPTextToBrainModel(nn.Module):
    """Text projection plus a frozen autoencoder used for targets/decoding."""

    def __init__(self, text_projection: nn.Module, autoencoder: nn.Module):
        super().__init__()
        self.text_projection = text_projection
        self.autoencoder = autoencoder
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)
        self.autoencoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        return self

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.text_projection(text_embedding)

    def decode(self, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decoder(self(text_embedding))


class MLPContrastiveModel(nn.Module):
    """The two trainable projection heads from the retained MLP recipe."""

    def __init__(self, brain_projection: nn.Module, text_projection: nn.Module):
        super().__init__()
        self.brain_projection = brain_projection
        self.text_projection = text_projection

    def forward(self, brain_embedding: torch.Tensor, text_embedding: torch.Tensor):
        # Retained notebooks normalize SPECTER embeddings before the text head.
        text_embedding = F.normalize(text_embedding, dim=1, eps=1e-8)
        return self.brain_projection(brain_embedding), self.text_projection(text_embedding)


@dataclass(frozen=True)
class MLPTrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    epochs_completed: int
    best_metric: float
    test_metrics: Mapping[str, float]
    model: nn.Module


def build_mlp_autoencoder(config: MLPAutoencoderTrainConfig) -> NeuroAutoEncoder:
    return NeuroAutoEncoder(seed=config.seed, out="logit", dim_neuro=config.dim_neuro,
                            dim_h0=config.dim_h0, dim_h1=config.dim_h1,
                            dim_latent=config.dim_latent)


def build_mlp_text_to_brain(
    config: MLPTextToBrainTrainConfig,
    *,
    autoencoder: nn.Module | None = None,
    text_projection: nn.Module | None = None,
) -> MLPTextToBrainModel:
    if autoencoder is None:
        if config.preset != "retained":
            raise ValueError("Custom text-to-brain dimensions require an injected autoencoder")
        from neurovlm.models.base import load_model
        autoencoder = load_model(family="mlp", task="autoencoder")
    if text_projection is None:
        text_projection = ProjHead(config.text_dim, config.hidden_dim, config.latent_dim, seed=config.seed)
    return MLPTextToBrainModel(text_projection, autoencoder)


def build_mlp_contrastive(
    config: MLPContrastiveTrainConfig,
    *,
    text_projection: nn.Module | None = None,
    brain_projection: nn.Module | None = None,
) -> MLPContrastiveModel:
    if text_projection is None:
        if config.initialize_text_from_mse:
            if config.preset != "retained":
                raise ValueError("Released MSE initialization is only compatible with preset='retained'")
            text_projection = ProjHead.from_pretrained("text_mse")
        else:
            text_projection = ProjHead(config.text_dim, config.text_hidden_dim, config.shared_dim, seed=config.seed)
    if brain_projection is None:
        brain_projection = ProjHead(config.brain_dim, config.brain_hidden_dim, config.shared_dim, seed=config.seed)
    return MLPContrastiveModel(brain_projection, text_projection)


def _payload(checkpoint: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        return checkpoint
    return torch.load(Path(checkpoint), map_location="cpu", weights_only=True)


def mlp_autoencoder_from_checkpoint(checkpoint: str | Path | Mapping[str, Any], *, device: str = "cpu") -> NeuroAutoEncoder:
    payload = _payload(checkpoint)
    arch = dict(payload.get("architecture") or {})
    if arch.get("architecture") != "NeuroAutoEncoder":
        raise ValueError("Checkpoint is not a standardized MLP autoencoder")
    model = NeuroAutoEncoder(out="logit", dim_neuro=int(arch["dim_neuro"]),
                             dim_h0=int(arch["dim_h0"]), dim_h1=int(arch["dim_h1"]),
                             dim_latent=int(arch["dim_latent"]))
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"Checkpoint state does not match its recorded MLP autoencoder architecture: {error}") from error
    return model.to(device).eval()


def mlp_text_to_brain_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any], *, autoencoder: nn.Module | None = None, device: str = "cpu"
) -> MLPTextToBrainModel:
    payload = _payload(checkpoint)
    arch = dict(payload.get("architecture") or {})
    if arch.get("architecture") != "MLPTextToBrain":
        raise ValueError("Checkpoint is not a standardized MLP text-to-brain model")
    source = dict(payload.get("extra") or {}).get("initialization_source")
    if autoencoder is None:
        if arch.get("preset") != "retained" or source not in {None, "released_autoencoder_and_scratch_projection"}:
            raise ValueError("Reloading this text-to-brain checkpoint requires autoencoder=")
        from neurovlm.models.base import load_model
        autoencoder = load_model(family="mlp", task="autoencoder")
    projection = ProjHead(int(arch["text_dim"]), int(arch["hidden_dim"]), int(arch["latent_dim"]))
    try:
        projection.load_state_dict(payload["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"Checkpoint state does not match its recorded text projection architecture: {error}") from error
    return MLPTextToBrainModel(projection, autoencoder).to(device).eval()


def mlp_contrastive_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any], *, device: str = "cpu"
) -> MLPContrastiveModel:
    payload = _payload(checkpoint)
    arch = dict(payload.get("architecture") or {})
    if arch.get("architecture") != "MLPContrastiveModel":
        raise ValueError("Checkpoint is not a standardized MLP contrastive model")
    model = MLPContrastiveModel(
        ProjHead(int(arch["brain_dim"]), int(arch["brain_hidden_dim"]), int(arch["shared_dim"])),
        ProjHead(int(arch["text_dim"]), int(arch["text_hidden_dim"]), int(arch["shared_dim"])),
    )
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"Checkpoint state does not match its recorded contrastive architecture: {error}") from error
    return model.to(device).eval()


def mlp_brain_to_text_retrieval_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any], *, device: str = "cpu"
) -> MLPContrastiveModel:
    """Reload a standardized brain-to-text retrieval checkpoint.

    Retrieval intentionally shares the exact composite model and checkpoint
    schema with MLP contrastive training.
    """

    payload = _payload(checkpoint)
    spec = dict(payload.get("model_spec") or {})
    task = spec.get("task")
    if task not in {None, "brain_to_text_retrieval"}:
        raise ValueError(f"Checkpoint task is {task!r}, not brain_to_text_retrieval")
    return mlp_contrastive_from_checkpoint(payload, device=device)


def _run_config(config: _BaseConfig, task: str, architecture: Mapping[str, Any], initialization: Mapping[str, Any]) -> RunConfig:
    values = asdict(config)
    values.pop("resume", None)
    variant = "text" if task == "contrastive" else None
    return RunConfig.resolve(
        family="mlp", task=task, variant=variant, output_root=config.output_root,
        run_id=config.run_id, seed=config.seed, device=str(_device(config.device)),
        primary_metric=getattr(config, "primary_metric"),
        metric_direction=getattr(config, "metric_direction"),
        data={"provider": "injected", "splits": ["train", "val", "test"]},
        resources={}, initialization=initialization, requested=values,
        effective={**values, "architecture": dict(architecture)},
    )


def _validation_metric(config: _BaseConfig) -> str:
    name = str(getattr(config, "primary_metric"))
    return name[4:] if name.startswith("val_") else name


def _initial_best(config: _BaseConfig) -> float:
    direction = MetricDirection(getattr(config, "metric_direction"))
    return float("inf") if direction is MetricDirection.MIN else float("-inf")


def _improved(config: _BaseConfig, value: float, best: float) -> bool:
    delta = float(config.early_stopping_min_delta)
    if MetricDirection(getattr(config, "metric_direction")) is MetricDirection.MIN:
        return value < best - delta
    return value > best + delta


def _record(recorder: MetricRecorder, split: str, values: Mapping[str, float], *, epoch: int | None, n: int | None) -> None:
    for name, value in values.items():
        recorder.record(split=split, metric=name, value=float(value), epoch=epoch, n=n)


def _finalize(
    run: RunConfig,
    recorder: MetricRecorder,
    evaluations: tuple[tuple[str, MLPEvaluation], ...],
    *,
    epochs_completed: int,
    best_metric: float,
) -> Mapping[str, float]:
    by_source: list[Mapping[str, Any]] = []
    by_sample: list[Mapping[str, Any]] = []
    test: Mapping[str, float] = {}
    for split, evaluation in evaluations:
        _record(recorder, split, evaluation.summary, epoch=None, n=evaluation.n)
        by_source.extend({"split": split, **row} for row in evaluation.by_source)
        by_sample.extend({"split": split, **row} for row in evaluation.by_sample)
        if split == "test":
            test = evaluation.summary
            atomic_write_csv(run.run_dir / "metrics/test_summary.csv", [{"split": split, "n": evaluation.n, **test}])
        if isinstance(evaluation, MLPContrastiveEvaluation):
            atomic_write_csv(run.run_dir / f"metrics/{split}_recall_curves.csv", evaluation.recall_curves)
    recorder.flush()
    atomic_write_csv(run.run_dir / "metrics/by_source.csv", by_source)
    atomic_write_csv(run.run_dir / "metrics/by_sample.csv", by_sample)
    atomic_write_json(run.run_dir / "metrics/training_summary.json",
                      {"epochs_completed": epochs_completed, "best_metric": best_metric,
                       "primary_metric": run.primary_metric, "test": test})
    return test


def _train_common(
    config: _BaseConfig,
    provider: Any,
    model: nn.Module,
    *,
    task: str,
    architecture: Mapping[str, Any],
    checkpoint_model: nn.Module,
    initialization: Mapping[str, Any],
    train_step: Any,
    evaluator: Any,
    evaluator_kwargs: Mapping[str, Any] | None = None,
) -> MLPTrainResult:
    _seed(config.seed)
    device = _device(config.device)
    train_data, val_data, test_data = _splits(provider)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in checkpoint_model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    run = _run_config(config, task, architecture, initialization)
    manager = CheckpointManager(run, expected_architecture=architecture)
    start_epoch, stale = 1, 0
    early_best = _initial_best(config)
    epochs_completed = 0
    kwargs = dict(evaluator_kwargs or {})
    with RunContext(run):
        recorder = MetricRecorder(run)
        if config.resume is not None:
            resumed = manager.load_resume(config.resume, model=checkpoint_model, optimizer=optimizer, map_location=device)
            start_epoch = int(resumed.epoch or 0) + 1
            extra = dict(resumed.payload.get("extra") or {})
            stale = int(extra.get("stale_epochs", 0))
            restored_best = manager.best_value
            early_best = float(extra.get("early_best", restored_best if restored_best is not None else _initial_best(config)))
        for epoch in range(start_epoch, config.epochs + 1):
            loader = _loader(train_data, batch_size=config.batch_size, shuffle=True,
                             seed=config.seed + epoch, num_workers=config.num_workers,
                             drop_last=task in {"contrastive", "brain_to_text_retrieval"})
            train_loss, count = train_step(model, loader, optimizer, device, config)
            validation = evaluator(model, val_data, device=device,
                                   batch_size=config.eval_batch_size or config.batch_size,
                                   num_workers=config.num_workers,
                                   max_batches=config.max_eval_batches, **kwargs)
            _record(recorder, "train", {"loss": train_loss}, epoch=epoch, n=count)
            _record(recorder, "val", validation.summary, epoch=epoch, n=validation.n)
            recorder.flush()
            metric_name = _validation_metric(config)
            if metric_name not in validation.summary:
                raise KeyError(
                    f"Validation summary has no {metric_name!r} required by "
                    f"primary_metric={config.primary_metric!r}"
                )
            selected = float(validation.summary[metric_name])
            metrics = {"train_loss": train_loss, **{f"val_{k}": v for k, v in validation.summary.items()}}
            manager.save_best(checkpoint_model, selected, epoch=epoch, metrics=metrics,
                              optimizer=optimizer, architecture=architecture,
                              extra={"initialization_source": initialization.get("source")},
                              aliases=(f"best_{config.primary_metric}.pt",))
            if _improved(config, selected, early_best):
                early_best, stale = selected, 0
            else:
                stale += 1
            manager.save_last(checkpoint_model, epoch=epoch, metrics=metrics, optimizer=optimizer,
                              architecture=architecture,
                              extra={"stale_epochs": stale, "early_best": early_best,
                                     "initialization_source": initialization.get("source")})
            epochs_completed = epoch
            if config.early_stopping_patience is not None and stale >= config.early_stopping_patience:
                break
        manager.load_resume(run.run_dir / "checkpoints/best.pt", model=checkpoint_model, map_location=device)
        evaluations = tuple(
            (split, evaluator(model, data, device=device,
                              batch_size=config.eval_batch_size or config.batch_size,
                              num_workers=config.num_workers,
                              max_batches=config.max_eval_batches, **kwargs))
            for split, data in (("val", val_data), ("test", test_data))
        )
        best = float(manager.best_value if manager.best_value is not None else early_best)
        test = _finalize(run, recorder, evaluations, epochs_completed=epochs_completed, best_metric=best)
    model.eval()
    return MLPTrainResult(run.run_dir, run.run_dir / "checkpoints/best.pt",
                          run.run_dir / "checkpoints/last.pt", epochs_completed, best, test, model)


def _ae_step(model: nn.Module, loader: DataLoader, optimizer: Any, device: torch.device, config: _BaseConfig):
    model.train()
    total = count = 0
    for index, batch in enumerate(loader):
        if config.max_train_batches is not None and index >= config.max_train_batches:
            break
        target = _autoencoder_input(batch).float().to(device)
        loss = F.binary_cross_entropy_with_logits(model(target), target)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step(); total += float(loss.detach()) * len(target); count += len(target)
    if count == 0: raise RuntimeError("Training dataset produced no batches")
    return total / count, count


def _t2b_step(model: MLPTextToBrainModel, loader: DataLoader, optimizer: Any, device: torch.device, config: _BaseConfig):
    model.train(); total = count = 0
    for index, batch in enumerate(loader):
        if config.max_train_batches is not None and index >= config.max_train_batches: break
        text, brain = _paired_input(batch, latent_brain=True)
        text, brain = text.float().to(device), brain.float().to(device)
        if brain.shape[-1] != model(text[:1]).shape[-1]:
            with torch.no_grad(): brain = model.autoencoder.encoder(brain)
        prediction = model(text)
        loss = F.mse_loss(prediction, brain)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.text_projection.parameters(), config.gradient_clip)
        optimizer.step(); total += float(loss.detach()) * len(text); count += len(text)
    if count == 0: raise RuntimeError("Training dataset produced no batches")
    return total / count, count


def _contrastive_step(model: MLPContrastiveModel, loader: DataLoader, optimizer: Any, device: torch.device, config: MLPContrastiveTrainConfig):
    model.train(); total = count = 0; loss_fn = InfoNCELoss(config.temperature)
    for index, batch in enumerate(loader):
        if config.max_train_batches is not None and index >= config.max_train_batches: break
        text, brain = _paired_input(batch, latent_brain=True)
        if len(text) < 2: continue
        brain_p, text_p = model(brain.float().to(device), text.float().to(device))
        loss = loss_fn(brain_p, text_p)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step(); total += float(loss.detach()) * len(text); count += len(text)
    if count == 0: raise RuntimeError("Contrastive training requires at least one batch with two pairs")
    return total / count, count


def train_mlp_autoencoder(config: MLPAutoencoderTrainConfig, *, provider: Any, model: NeuroAutoEncoder | None = None) -> MLPTrainResult:
    provided = model is not None
    model = model or build_mlp_autoencoder(config)
    return _train_common(config, provider, model, task="autoencoder", architecture=config.architecture(),
                         checkpoint_model=model, initialization={"source": "provided_model" if provided else "scratch"},
                         train_step=_ae_step, evaluator=evaluate_mlp_autoencoder)


def train_mlp_text_to_brain(
    config: MLPTextToBrainTrainConfig, *, provider: Any, model: MLPTextToBrainModel | None = None,
    autoencoder: nn.Module | None = None, text_projection: nn.Module | None = None,
) -> MLPTrainResult:
    provided = model is not None or autoencoder is not None
    model = model or build_mlp_text_to_brain(config, autoencoder=autoencoder, text_projection=text_projection)
    return _train_common(config, provider, model, task="text_to_brain", architecture=config.architecture(),
                         checkpoint_model=model.text_projection,
                         initialization={"source": "provided_model" if provided else "released_autoencoder_and_scratch_projection"},
                         train_step=_t2b_step, evaluator=evaluate_mlp_text_to_brain)


def train_mlp_contrastive(
    config: MLPContrastiveTrainConfig, *, provider: Any, model: MLPContrastiveModel | None = None,
    text_projection: nn.Module | None = None, brain_projection: nn.Module | None = None,
) -> MLPTrainResult:
    provided = model is not None
    model = model or build_mlp_contrastive(config, text_projection=text_projection, brain_projection=brain_projection)
    return _train_common(config, provider, model, task="contrastive", architecture=config.architecture(),
                         checkpoint_model=model,
                         initialization={"source": "provided_model" if provided else ("released_text_mse" if config.initialize_text_from_mse else "scratch")},
                         train_step=_contrastive_step, evaluator=evaluate_mlp_contrastive,
                         evaluator_kwargs={"temperature": config.temperature})


def build_mlp_brain_to_text_retrieval(
    config: MLPBrainToTextRetrievalTrainConfig,
    *,
    text_projection: nn.Module | None = None,
    brain_projection: nn.Module | None = None,
) -> MLPContrastiveModel:
    """Build the shared retained contrastive model for B2T retrieval."""

    return build_mlp_contrastive(
        config,
        text_projection=text_projection,
        brain_projection=brain_projection,
    )


def train_mlp_brain_to_text_retrieval(
    config: MLPBrainToTextRetrievalTrainConfig,
    *,
    provider: Any,
    model: MLPContrastiveModel | None = None,
    text_projection: nn.Module | None = None,
    brain_projection: nn.Module | None = None,
) -> MLPTrainResult:
    """Train symmetric InfoNCE while selecting the brain-to-text direction."""

    if not isinstance(config, MLPBrainToTextRetrievalTrainConfig):
        raise TypeError("config must be an MLPBrainToTextRetrievalTrainConfig")
    provided = model is not None
    model = model or build_mlp_brain_to_text_retrieval(
        config,
        text_projection=text_projection,
        brain_projection=brain_projection,
    )
    return _train_common(
        config,
        provider,
        model,
        task="brain_to_text_retrieval",
        architecture=config.architecture(),
        checkpoint_model=model,
        initialization={
            "source": "provided_model"
            if provided
            else ("released_text_mse" if config.initialize_text_from_mse else "scratch")
        },
        train_step=_contrastive_step,
        evaluator=evaluate_mlp_contrastive,
        evaluator_kwargs={"temperature": config.temperature},
    )


__all__ = [
    "MLP_RETAINED_DIMS", "MLPAutoencoderTrainConfig", "MLPBrainToTextRetrievalTrainConfig", "MLPContrastiveModel",
    "MLPContrastiveTrainConfig", "MLPTextToBrainModel", "MLPTextToBrainTrainConfig",
    "MLPTrainResult", "build_mlp_autoencoder", "build_mlp_brain_to_text_retrieval", "build_mlp_contrastive",
    "build_mlp_text_to_brain", "mlp_autoencoder_from_checkpoint",
    "mlp_brain_to_text_retrieval_from_checkpoint", "mlp_contrastive_from_checkpoint", "mlp_text_to_brain_from_checkpoint",
    "train_mlp_autoencoder", "train_mlp_brain_to_text_retrieval", "train_mlp_contrastive", "train_mlp_text_to_brain",
]
