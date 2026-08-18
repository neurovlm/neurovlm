"""Task-oriented, tensor-level inference for packaged NeuroVLM models.

The legacy :class:`neurovlm.core.NeuroVLM` API remains unchanged.  This
module is the smaller inference surface used when callers want to select a
model family, task, domain, or training run explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neurovlm.cnn.models import ATLAS_FREE_VOLUME_SHAPE, MLP_MASKER_VOXEL_COUNT
from neurovlm.models.base import load_model
from neurovlm.models.registry import ModelFamily, ModelSpec, ModelTask, resolve_model_spec


TextEncoder = Callable[[Any], Tensor]


def _checkpoint_from_run(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    for checkpoint in (candidate / "checkpoints" / "best.pt", candidate / "best.pt"):
        if checkpoint.is_file():
            return checkpoint
    raise FileNotFoundError(f"No best checkpoint found under run {candidate}")


def _batch(value: Tensor, expected_ndim: int) -> Tensor:
    value = torch.as_tensor(value)
    return value.unsqueeze(0) if value.ndim == expected_ndim - 1 else value


def _freeze(model: nn.Module, device: torch.device) -> nn.Module:
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _first_linear_input(module: nn.Module | None) -> int | None:
    if module is None:
        return None
    return next(
        (layer.in_features for layer in module.modules() if isinstance(layer, nn.Linear)),
        None,
    )


@dataclass(frozen=True)
class RuntimeMetadata:
    """Resolved inference selection, suitable for logs and manifests."""

    canonical_name: str
    family: str
    task: str
    domain: str | None
    variant: str
    loader_variant: str | None
    source: str
    checkpoint: str | None
    device: str
    brain_space: str
    text_space: str | None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class NeuroVLMRuntime:
    """A resolved model pipeline with consistent tensor-level methods.

    Use :func:`load_pipeline` instead of constructing this class directly.
    Methods that do not apply to the selected task raise a precise error.
    """

    def __init__(
        self,
        *,
        spec: ModelSpec,
        model: nn.Module,
        device: torch.device,
        source: str,
        checkpoint: Path | None = None,
        autoencoder: nn.Module | None = None,
        brain_projection: nn.Module | None = None,
        text_projection: nn.Module | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> None:
        self.spec = spec
        self.device = device
        self.model = _freeze(model, device)
        self.autoencoder = None if autoencoder is None else _freeze(autoencoder, device)
        self.brain_projection = (
            None if brain_projection is None else _freeze(brain_projection, device)
        )
        self.text_projection = (
            None if text_projection is None else _freeze(text_projection, device)
        )
        self._text_encoder = text_encoder
        self._lazy_text_encoder: Any | None = None
        self._brain_dim = None
        self._volume_shape = None
        if spec.family is ModelFamily.CNN:
            decoder = getattr(model, "decoder", None)
            output_shape = getattr(decoder, "output_shape", None)
            self._volume_shape = None if output_shape is None else tuple(output_shape)
            declared = self._volume_shape or ("D", "H", "W")
            brain_space = f"atlas_free_volume:B,1,{','.join(map(str, declared))}"
        else:
            brain_model = autoencoder if autoencoder is not None else model
            self._brain_dim = _first_linear_input(getattr(brain_model, "encoder", None))
            self._brain_dim = self._brain_dim or MLP_MASKER_VOXEL_COUNT
            brain_space = f"mlp_masker_flatmap:B,{self._brain_dim}"
        self.metadata = RuntimeMetadata(
            canonical_name=spec.canonical_name,
            family=spec.family.value,
            task=spec.task.value,
            domain=None if spec.domain is None else spec.domain.value,
            variant=spec.variant.value,
            loader_variant=spec.loader_variant,
            source=source,
            checkpoint=None if checkpoint is None else str(checkpoint.resolve()),
            device=str(device),
            brain_space=brain_space,
            text_space=(
                "specter_embedding:B,768"
                if spec.task in {ModelTask.CONTRASTIVE, ModelTask.TEXT_TO_BRAIN,
                                 ModelTask.BRAIN_TO_TEXT_RETRIEVAL}
                else None
            ),
        )

    def _require(self, *tasks: ModelTask) -> None:
        if self.spec.task not in tasks:
            allowed = ", ".join(task.value for task in tasks)
            raise RuntimeError(
                f"{self.spec.canonical_name} does not support this method; "
                f"load task {allowed}"
            )

    def _brain(self, value: Tensor) -> Tensor:
        if self.spec.family is ModelFamily.CNN:
            value = _batch(value, 5).float()
            expected = None if self._volume_shape is None else (1, *self._volume_shape)
            if value.ndim != 5 or value.shape[1] != 1 or (
                expected is not None and tuple(value.shape[1:]) != expected
            ):
                label = "(B, 1, D, H, W)" if expected is None else f"(B, {expected})"
                raise ValueError(f"CNN brain input must have shape {label}, got {tuple(value.shape)}")
        else:
            value = _batch(value, 2).float()
            if value.ndim != 2 or value.shape[1] != self._brain_dim:
                raise ValueError(
                    f"MLP brain input must have shape (B, {self._brain_dim}), "
                    f"got {tuple(value.shape)}"
                )
        return value.to(self.device)

    def _text(self, value: Any) -> Tensor:
        if torch.is_tensor(value):
            embedding = _batch(value, 2).float()
        else:
            encoder = self._text_encoder
            if encoder is None:
                if self._lazy_text_encoder is None:
                    self._lazy_text_encoder = load_model(
                        family="mlp", task="text_encoder", variant="specter"
                    )
                encoder = self._lazy_text_encoder
            embedding = torch.as_tensor(encoder(value)).float()
            embedding = _batch(embedding, 2)
        if embedding.ndim != 2:
            raise ValueError(f"Text embeddings must be a 2D batch, got {tuple(embedding.shape)}")
        return embedding.to(self.device)

    @torch.inference_mode()
    def encode(self, brain: Tensor) -> Tensor:
        self._require(ModelTask.AUTOENCODER)
        return self.model.encoder(self._brain(brain))

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        self._require(ModelTask.AUTOENCODER)
        latent = _batch(latent, 2).float().to(self.device)
        output = self.model.decoder(latent)
        if self.spec.family is ModelFamily.MLP and not any(
            isinstance(layer, nn.Sigmoid) for layer in self.model.decoder.modules()
        ):
            output = torch.sigmoid(output)
        return output

    @torch.inference_mode()
    def reconstruct(self, brain: Tensor) -> Tensor:
        self._require(ModelTask.AUTOENCODER)
        output = self.model(self._brain(brain))
        # Standardized MLP training stores logits; released MLP and every CNN
        # decoder already return their intended map representation.
        if self.spec.family is ModelFamily.MLP and not any(
            isinstance(layer, nn.Sigmoid) for layer in self.model.decoder.modules()
        ):
            output = torch.sigmoid(output)
        return output

    @torch.inference_mode()
    def encode_brain(self, brain: Tensor) -> Tensor:
        self._require(ModelTask.CONTRASTIVE, ModelTask.BRAIN_TO_TEXT_RETRIEVAL)
        value = self._brain(brain)
        if self.spec.family is ModelFamily.CNN:
            embedding = self.model.encode_brain(value)
        else:
            if self.autoencoder is None or self.brain_projection is None:
                raise RuntimeError("MLP contrastive runtime is missing its brain pipeline")
            embedding = self.brain_projection(self.autoencoder.encoder(value))
        return F.normalize(embedding, dim=-1, eps=1e-8)

    @torch.inference_mode()
    def encode_text(self, text_or_embedding: Any) -> Tensor:
        self._require(ModelTask.CONTRASTIVE, ModelTask.BRAIN_TO_TEXT_RETRIEVAL)
        value = self._text(text_or_embedding)
        if self.spec.family is ModelFamily.CNN:
            embedding = self.model.encode_text(value)
        else:
            if self.text_projection is None:
                raise RuntimeError("MLP contrastive runtime is missing its text pipeline")
            embedding = self.text_projection(F.normalize(value, dim=-1, eps=1e-8))
        return F.normalize(embedding, dim=-1, eps=1e-8)

    @torch.inference_mode()
    def similarity(self, brain: Tensor, text_or_embedding: Any) -> Tensor:
        """Return the full brain-by-text cosine-similarity matrix."""

        return self.encode_brain(brain) @ self.encode_text(text_or_embedding).T

    retrieve = similarity

    @torch.inference_mode()
    def generate(self, text_or_embedding: Any) -> Tensor:
        self._require(ModelTask.TEXT_TO_BRAIN)
        value = self._text(text_or_embedding)
        if self.spec.family is ModelFamily.CNN:
            return self.model(value)
        if hasattr(self.model, "decode"):
            output = self.model.decode(value)
        else:
            if self.autoencoder is None:
                raise RuntimeError("MLP text-to-brain runtime is missing its autoencoder")
            output = self.autoencoder.decoder(self.model(value))
        if self.autoencoder is not None and not any(
            isinstance(layer, nn.Sigmoid) for layer in self.autoencoder.decoder.modules()
        ):
            output = torch.sigmoid(output)
        return output


def _released_mlp(spec: ModelSpec) -> tuple[nn.Module, nn.Module | None, nn.Module | None, nn.Module | None]:
    if spec.task is ModelTask.AUTOENCODER:
        model = load_model(family="mlp", task="autoencoder")
        return model, None, None, None
    if spec.task in {ModelTask.CONTRASTIVE, ModelTask.BRAIN_TO_TEXT_RETRIEVAL}:
        autoencoder = load_model(family="mlp", task="autoencoder")
        brain = load_model(family="mlp", task="contrastive", variant="brain")
        text = load_model(family="mlp", task="contrastive", variant="text")
        return nn.Identity(), autoencoder, brain, text
    if spec.task is ModelTask.TEXT_TO_BRAIN:
        if spec.variant.value != "mse":
            raise ValueError("The tensor-level runtime supports the MLP text-to-brain MSE model")
        autoencoder = load_model(family="mlp", task="autoencoder")
        projection = load_model(family="mlp", task="text_to_brain", variant="mse")
        return projection, autoencoder, None, projection
    raise ValueError(f"Tensor-level runtime does not support task {spec.task.value!r}")


def load_pipeline(
    *,
    family: str = "mlp",
    task: str = "autoencoder",
    domain: str | None = None,
    variant: str | None = None,
    checkpoint: str | Path | None = None,
    from_run: str | Path | None = None,
    device: str | torch.device = "cpu",
    text_encoder: TextEncoder | None = None,
) -> NeuroVLMRuntime:
    """Resolve and load an inference pipeline.

    CNN domain tasks default to ``mixed_baseline``.  Passing
    ``variant='finetuned'`` is the only way to select specialized CNN weights.
    ``from_run`` accepts a run directory or its checkpoint; ``checkpoint`` is
    the explicit-file spelling.  They are mutually exclusive.
    """

    if checkpoint is not None and from_run is not None:
        raise ValueError("Pass at most one of checkpoint and from_run")
    spec = resolve_model_spec(
        family=family, task=task, domain=domain, variant=variant
    )
    resolved_device = torch.device(device)
    selected = None
    if from_run is not None:
        selected = _checkpoint_from_run(from_run)
    elif checkpoint is not None:
        selected = Path(checkpoint).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {selected}")

    autoencoder = brain_projection = text_projection = None
    if selected is None:
        if spec.family is ModelFamily.CNN:
            model = load_model(
                family=spec.family,
                task=spec.task,
                domain=spec.domain,
                variant=spec.variant,
            )
        else:
            model, autoencoder, brain_projection, text_projection = _released_mlp(spec)
        source = "released"
    else:
        if spec.family is ModelFamily.CNN:
            if spec.task is ModelTask.AUTOENCODER:
                from neurovlm.training.autoencoder import autoencoder_from_checkpoint
                model = autoencoder_from_checkpoint(selected, device=resolved_device)
            elif spec.task is ModelTask.CONTRASTIVE:
                from neurovlm.training.contrastive import contrastive_from_checkpoint
                model = contrastive_from_checkpoint(selected, device=resolved_device)
            elif spec.task is ModelTask.TEXT_TO_BRAIN:
                from neurovlm.training.text_to_brain import text_to_brain_from_checkpoint
                model = text_to_brain_from_checkpoint(selected, device=resolved_device)
            else:
                raise ValueError(f"CNN from-run reload does not support task {spec.task.value!r}")
        else:
            from neurovlm.training.mlp import (
                mlp_autoencoder_from_checkpoint,
                mlp_brain_to_text_retrieval_from_checkpoint,
                mlp_contrastive_from_checkpoint,
                mlp_text_to_brain_from_checkpoint,
            )
            if spec.task is ModelTask.AUTOENCODER:
                model = mlp_autoencoder_from_checkpoint(selected, device=str(resolved_device))
            elif spec.task is ModelTask.CONTRASTIVE:
                model = mlp_contrastive_from_checkpoint(selected, device=str(resolved_device))
                brain_projection = model.brain_projection
                text_projection = model.text_projection
                # Run checkpoints consume the same released AE latent space.
                autoencoder = load_model(family="mlp", task="autoencoder")
            elif spec.task is ModelTask.BRAIN_TO_TEXT_RETRIEVAL:
                model = mlp_brain_to_text_retrieval_from_checkpoint(selected, device=str(resolved_device))
                brain_projection = model.brain_projection
                text_projection = model.text_projection
                autoencoder = load_model(family="mlp", task="autoencoder")
            elif spec.task is ModelTask.TEXT_TO_BRAIN:
                model = mlp_text_to_brain_from_checkpoint(selected, device=str(resolved_device))
                autoencoder = model.autoencoder
                text_projection = model.text_projection
            else:
                raise ValueError(f"MLP from-run reload does not support task {spec.task.value!r}")
        source = "from_run" if from_run is not None else "checkpoint"

    return NeuroVLMRuntime(
        spec=spec,
        model=model,
        device=resolved_device,
        source=source,
        checkpoint=selected,
        autoencoder=autoencoder,
        brain_projection=brain_projection,
        text_projection=text_projection,
        text_encoder=text_encoder,
    )


__all__ = ["NeuroVLMRuntime", "RuntimeMetadata", "load_pipeline"]
