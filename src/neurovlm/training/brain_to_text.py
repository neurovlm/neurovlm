"""Standardized MLP brain-to-text retrieval and Q-Former generation training."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from neurovlm.evaluation.brain_to_text import (
    BrainToTextBatch,
    BrainToTextGenerationEvaluation,
    brain_to_text_lm_forward,
    evaluate_brain_to_text_generation,
    parse_brain_to_text_batch,
)
from neurovlm.pipelines import (
    CheckpointManager,
    MetricDirection,
    MetricRecorder,
    RunConfig,
    RunContext,
)
from neurovlm.pipelines.serialization import atomic_write_csv, atomic_write_json
from neurovlm.models.qformer import NeuroQFormer
from neurovlm.resources.loaders import NEURO_QFORMER_REPO_ID, NEURO_QWEN_REPO_ID


QFORMER_RETAINED_ARCHITECTURE = (384, 384, 1024, 32, 512, 8, 6)


@dataclass(frozen=True)
class BrainToTextGenerationTrainConfig:
    output_root: str | Path = "runs"
    run_id: str | None = None
    seed: int = 42
    device: str = "auto"
    epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float | None = 1.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    resume: str | Path | None = None
    initialization: str = "released"
    qformer_checkpoint: str | Path | None = None
    qformer_resource: str = NEURO_QFORMER_REPO_ID
    lm_resource: str = NEURO_QWEN_REPO_ID
    preset: str = "retained"
    image_dim: int = 384
    semantic_dim: int = 384
    lm_dim: int = 1024
    num_queries: int = 32
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.05
    projection_temp: float | None = 0.05
    canonical_basis: str = "all"
    use_canonical_projection: bool = True
    train_image_projection: bool = False
    pad_token_id: int | None = None
    generated_samples_limit: int = 0
    primary_metric: str = "val_loss"
    metric_direction: MetricDirection = MetricDirection.MIN

    def __post_init__(self) -> None:
        if self.seed < 0 or self.epochs < 1 or min(self.batch_size, self.eval_batch_size) < 1:
            raise ValueError("seed must be non-negative and epochs/batch sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.generated_samples_limit < 0:
            raise ValueError("generated_samples_limit must be non-negative")
        if self.initialization not in {"released", "scratch", "from_run"}:
            raise ValueError("initialization must be released, scratch, or from_run")
        if self.initialization == "from_run" and self.qformer_checkpoint is None:
            raise ValueError("initialization='from_run' requires qformer_checkpoint")
        if self.initialization != "from_run" and self.qformer_checkpoint is not None:
            raise ValueError("qformer_checkpoint is only valid with initialization='from_run'")
        if self.preset not in {"retained", "custom"}:
            raise ValueError("preset must be retained or custom")
        dims = (
            self.image_dim,
            self.semantic_dim,
            self.lm_dim,
            self.num_queries,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
        )
        if self.preset == "retained" and dims != QFORMER_RETAINED_ARCHITECTURE:
            raise ValueError("Changing retained Q-Former dimensions requires preset='custom'")
        if min(dims) < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("Q-Former dimensions must be positive and hidden_dim divisible by num_heads")

    def architecture(self) -> dict[str, Any]:
        return {
            "architecture": "NeuroQFormer",
            "preset": self.preset,
            "image_dim": self.image_dim,
            "semantic_dim": self.semantic_dim,
            "lm_dim": self.lm_dim,
            "num_queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "projection_temp": self.projection_temp,
            "canonical_basis": self.canonical_basis,
            "use_canonical_projection": self.use_canonical_projection,
            "train_image_projection": self.train_image_projection,
            "lm_resource": self.lm_resource,
        }


@dataclass(frozen=True)
class BrainToTextGenerationTrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    epochs_completed: int
    best_metric: float
    test_metrics: Mapping[str, float]
    qformer: NeuroQFormer
    causal_lm: nn.Module


class BrainToTextCollator:
    """Pad token IDs while preserving optional semantic latents and metadata."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        if not rows or not all(isinstance(row, Mapping) for row in rows):
            raise TypeError("BrainToTextCollator expects non-empty mapping rows")
        parsed = [parse_brain_to_text_batch(row, pad_token_id=self.pad_token_id) for row in rows]
        if any(len(item.raw_brain) != 1 for item in parsed):
            raise ValueError("BrainToTextCollator dataset rows must describe one sample")
        has_semantic = [item.semantic_brain is not None for item in parsed]
        if any(has_semantic) and not all(has_semantic):
            raise ValueError("semantic brain latents must be present for every row or none")
        input_ids = pad_sequence(
            [item.input_ids[0] for item in parsed],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention = input_ids.ne(self.pad_token_id).long()
        output: dict[str, Any] = {
            "brain_embedding": torch.cat([item.raw_brain for item in parsed]),
            "input_ids": input_ids,
            "attention_mask": attention,
            "source": [item.sources[0] for item in parsed],
            "sample_id": [item.sample_ids[0] for item in parsed],
            "reference_text": [item.references[0] for item in parsed],
        }
        if all(has_semantic):
            output["semantic_embedding"] = torch.cat(
                [item.semantic_brain for item in parsed if item.semantic_brain is not None]
            )
        return output


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
    collator: Callable[[Sequence[Any]], Any] | None,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )


def _freeze_models(
    qformer: NeuroQFormer, causal_lm: nn.Module, *, train_image_projection: bool
) -> None:
    for parameter in causal_lm.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    causal_lm.eval()
    for parameter in qformer.parameters():
        parameter.requires_grad_(False)
    for parameter in qformer.qformer.parameters():
        parameter.requires_grad_(True)
    if train_image_projection:
        for parameter in qformer.proj_head_image.parameters():
            parameter.requires_grad_(True)


def _load_released_lm(resource: str, device: torch.device) -> nn.Module:
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(
        resource, torch_dtype=dtype, device_map=None
    ).to(device)


def brain_to_text_generation_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any], *, device: str | torch.device = "cpu"
) -> NeuroQFormer:
    """Standalone reload of a standardized or legacy NeuroQFormer payload."""

    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    architecture = dict(payload.get("architecture") or payload.get("config") or {})
    name = architecture.get("architecture")
    if name not in {None, "NeuroQFormer"}:
        raise ValueError(f"Checkpoint architecture is {name!r}, not NeuroQFormer")
    try:
        return NeuroQFormer.from_state_dict_payload(payload, map_location=device)
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint state does not match its recorded NeuroQFormer architecture: {error}"
        ) from error


def build_brain_to_text_generation(
    config: BrainToTextGenerationTrainConfig,
    *,
    qformer: NeuroQFormer | None = None,
    causal_lm: nn.Module | None = None,
    qformer_loader: Callable[[], NeuroQFormer] | None = None,
    lm_loader: Callable[[], nn.Module] | None = None,
    proj_head_image: nn.Module | None = None,
    canonical_banks: dict[str, torch.Tensor] | None = None,
) -> tuple[NeuroQFormer, nn.Module]:
    """Build the Q-Former and frozen LM without requiring network when injected."""

    device = _device(config.device)
    if qformer is None:
        if config.initialization == "from_run":
            qformer = brain_to_text_generation_from_checkpoint(
                config.qformer_checkpoint, device=device  # type: ignore[arg-type]
            )
        elif config.initialization == "released":
            if qformer_loader is not None:
                qformer = qformer_loader()
            else:
                from neurovlm.models.base import load_model

                qformer = load_model(
                    family="mlp", task="brain_to_text_generation", variant="qformer"
                )
        else:
            qformer = NeuroQFormer(
                proj_head_image=proj_head_image,
                canonical_banks=canonical_banks,
                image_dim=config.image_dim,
                semantic_dim=config.semantic_dim,
                lm_dim=config.lm_dim,
                num_queries=config.num_queries,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                projection_temp=config.projection_temp,
                canonical_basis=config.canonical_basis,
                use_canonical_projection=config.use_canonical_projection,
            )
    if causal_lm is None:
        causal_lm = lm_loader() if lm_loader is not None else _load_released_lm(config.lm_resource, device)
    qformer.to(device)
    causal_lm.to(device)
    actual = qformer.architecture_config()
    for key in (
        "image_dim",
        "semantic_dim",
        "lm_dim",
        "num_queries",
        "hidden_dim",
        "num_heads",
        "num_layers",
    ):
        if actual[key] != config.architecture()[key]:
            raise ValueError(
                f"Injected/initialized Q-Former {key}={actual[key]!r} does not match config "
                f"value {config.architecture()[key]!r}"
            )
    embeddings = causal_lm.get_input_embeddings()
    if int(embeddings.weight.shape[1]) != config.lm_dim:
        raise ValueError(
            f"Causal LM embedding dim {embeddings.weight.shape[1]} does not match lm_dim={config.lm_dim}"
        )
    _freeze_models(qformer, causal_lm, train_image_projection=config.train_image_projection)
    return qformer, causal_lm


def _record(
    recorder: MetricRecorder,
    split: str,
    values: Mapping[str, float],
    *,
    epoch: int | None,
    n: int | None,
) -> None:
    for name, value in values.items():
        recorder.record(split=split, metric=name, value=float(value), epoch=epoch, n=n)


def _train_epoch(
    qformer: NeuroQFormer,
    causal_lm: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: BrainToTextGenerationTrainConfig,
) -> tuple[float, int, int]:
    qformer.train()
    qformer.proj_head_image.eval() if not config.train_image_projection else qformer.proj_head_image.train()
    causal_lm.eval()
    loss_sum = 0.0
    token_count = 0
    samples = 0
    for index, raw_batch in enumerate(loader):
        if config.max_train_batches is not None and index >= config.max_train_batches:
            break
        batch = parse_brain_to_text_batch(raw_batch, pad_token_id=config.pad_token_id)
        output = brain_to_text_lm_forward(
            qformer, causal_lm, batch, device=device, pad_token_id=config.pad_token_id
        )
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        if config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(
                [parameter for parameter in qformer.parameters() if parameter.requires_grad],
                config.gradient_clip,
            )
        optimizer.step()
        tokens = int(output.token_mask.sum())
        loss_sum += float(output.token_losses[output.token_mask].sum().detach())
        token_count += tokens
        samples += len(batch.raw_brain)
    if not samples or not token_count:
        raise RuntimeError("Brain-to-text training produced no supervised tokens")
    return loss_sum / token_count, samples, token_count


def train_brain_to_text_generation(
    config: BrainToTextGenerationTrainConfig,
    *,
    provider: Any,
    qformer: NeuroQFormer | None = None,
    causal_lm: nn.Module | None = None,
    qformer_loader: Callable[[], NeuroQFormer] | None = None,
    lm_loader: Callable[[], nn.Module] | None = None,
    collator: Callable[[Sequence[Any]], Any] | None = None,
    generation_callback: Callable[[nn.Module, nn.Module, BrainToTextBatch], Sequence[str]] | None = None,
    semantic_metric_callback: Callable[[Sequence[str], Sequence[str | None], Sequence[Mapping[str, Any]]], Mapping[str, float]] | None = None,
) -> BrainToTextGenerationTrainResult:
    """Train only Q-Former parameters against a frozen causal LM."""

    if not isinstance(config, BrainToTextGenerationTrainConfig):
        raise TypeError("config must be a BrainToTextGenerationTrainConfig")
    _seed(config.seed)
    device = _device(config.device)
    supplied_qformer, supplied_lm = qformer is not None, causal_lm is not None
    qformer, causal_lm = build_brain_to_text_generation(
        config,
        qformer=qformer,
        causal_lm=causal_lm,
        qformer_loader=qformer_loader,
        lm_loader=lm_loader,
    )
    train_data, val_data, test_data = _splits(provider)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in qformer.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    values = asdict(config)
    values.pop("resume", None)
    initialization_source = (
        "provided_qformer" if supplied_qformer else config.initialization
    )
    resources = {
        "qformer": "injected" if supplied_qformer else config.qformer_resource,
        "causal_lm": "injected" if supplied_lm else config.lm_resource,
    }
    run = RunConfig.resolve(
        family="mlp",
        task="brain_to_text_generation",
        variant="qformer",
        output_root=config.output_root,
        run_id=config.run_id,
        seed=config.seed,
        device=str(device),
        primary_metric=config.primary_metric,
        metric_direction=config.metric_direction,
        data={"provider": "injected", "splits": ["train", "val", "test"]},
        resources=resources,
        initialization={
            "source": initialization_source,
            "checkpoint": str(config.qformer_checkpoint) if config.qformer_checkpoint else None,
        },
        requested=values,
        effective={**values, "architecture": config.architecture()},
    )
    manager = CheckpointManager(run, expected_architecture=config.architecture())
    start_epoch, stale, epochs_completed = 1, 0, 0
    early_best = float("inf")
    with RunContext(run):
        recorder = MetricRecorder(run)
        if config.resume is not None:
            resumed = manager.load_resume(
                config.resume,
                model=qformer,
                optimizer=optimizer,
                map_location=device,
            )
            start_epoch = int(resumed.epoch or 0) + 1
            extra = dict(resumed.payload.get("extra") or {})
            stale = int(extra.get("stale_epochs", 0))
            early_best = float(
                extra.get(
                    "early_best",
                    manager.best_value if manager.best_value is not None else float("inf"),
                )
            )
        for epoch in range(start_epoch, config.epochs + 1):
            train_loader = _loader(
                train_data,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
                num_workers=config.num_workers,
                collator=collator,
            )
            train_loss, train_n, train_tokens = _train_epoch(
                qformer, causal_lm, train_loader, optimizer, device, config
            )
            validation = evaluate_brain_to_text_generation(
                qformer,
                causal_lm,
                val_data,
                device=device,
                batch_size=config.eval_batch_size,
                num_workers=config.num_workers,
                max_batches=config.max_eval_batches,
                pad_token_id=config.pad_token_id,
            )
            _record(
                recorder,
                "train",
                {"loss": train_loss, "token_count": float(train_tokens)},
                epoch=epoch,
                n=train_n,
            )
            _record(recorder, "val", validation.summary, epoch=epoch, n=validation.n)
            recorder.flush()
            selected = float(validation.summary["loss"])
            metrics = {
                "train_loss": train_loss,
                **{f"val_{key}": value for key, value in validation.summary.items()},
            }
            checkpoint_extra = {
                "stale_epochs": stale,
                "early_best": early_best,
                "initialization_source": initialization_source,
                "resources": resources,
            }
            manager.save_best(
                qformer,
                selected,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=config.architecture(),
                extra=checkpoint_extra,
                aliases=("best_val_loss.pt",),
            )
            if selected < early_best - config.early_stopping_min_delta:
                early_best, stale = selected, 0
            else:
                stale += 1
            manager.save_last(
                qformer,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                architecture=config.architecture(),
                extra={
                    **checkpoint_extra,
                    "stale_epochs": stale,
                    "early_best": early_best,
                },
            )
            epochs_completed = epoch
            if (
                config.early_stopping_patience is not None
                and stale >= config.early_stopping_patience
            ):
                break
        manager.load_resume(
            run.run_dir / "checkpoints/best.pt", model=qformer, map_location=device
        )
        evaluations: list[tuple[str, BrainToTextGenerationEvaluation]] = []
        for split, data in (("val", val_data), ("test", test_data)):
            evaluation = evaluate_brain_to_text_generation(
                qformer,
                causal_lm,
                data,
                device=device,
                batch_size=config.eval_batch_size,
                num_workers=config.num_workers,
                max_batches=config.max_eval_batches,
                pad_token_id=config.pad_token_id,
                generated_samples_limit=(
                    config.generated_samples_limit if split == "test" else 0
                ),
                generation_callback=generation_callback,
                semantic_metric_callback=semantic_metric_callback,
            )
            evaluations.append((split, evaluation))
            _record(recorder, split, evaluation.summary, epoch=None, n=evaluation.n)
        recorder.flush()
        by_source = [
            {"split": split, **row}
            for split, evaluation in evaluations
            for row in evaluation.by_source
        ]
        by_sample = [
            {"split": split, **row}
            for split, evaluation in evaluations
            for row in evaluation.by_sample
        ]
        test = evaluations[-1][1]
        atomic_write_csv(run.run_dir / "metrics/by_source.csv", by_source)
        atomic_write_csv(run.run_dir / "metrics/by_sample.csv", by_sample)
        atomic_write_csv(
            run.run_dir / "metrics/test_summary.csv",
            [{"split": "test", "n": test.n, **test.summary}],
        )
        if test.generated:
            atomic_write_csv(
                run.run_dir / "metrics/generated_text.csv", test.generated
            )
        best = float(manager.best_value if manager.best_value is not None else early_best)
        atomic_write_json(
            run.run_dir / "metrics/training_summary.json",
            {
                "epochs_completed": epochs_completed,
                "best_metric": best,
                "primary_metric": config.primary_metric,
                "test": test.summary,
            },
        )
    qformer.eval()
    causal_lm.eval()
    return BrainToTextGenerationTrainResult(
        run.run_dir,
        run.run_dir / "checkpoints/best.pt",
        run.run_dir / "checkpoints/last.pt",
        epochs_completed,
        best,
        test.summary,
        qformer,
        causal_lm,
    )


__all__ = [
    "BrainToTextCollator",
    "BrainToTextGenerationTrainConfig",
    "BrainToTextGenerationTrainResult",
    "QFORMER_RETAINED_ARCHITECTURE",
    "brain_to_text_generation_from_checkpoint",
    "build_brain_to_text_generation",
    "train_brain_to_text_generation",
]
