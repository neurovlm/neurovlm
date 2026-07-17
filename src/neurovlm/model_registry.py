"""Canonical model identifiers and resolution rules.

This module deliberately contains no model imports or checkpoint loading.  It
is therefore safe to use from configuration, training, inference, and docs
code without initializing PyTorch models or contacting Hugging Face.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class _StringEnum(str, Enum):
    """A Python 3.10-compatible string enum."""

    def __str__(self) -> str:
        return self.value


class ModelFamily(_StringEnum):
    """Supported model architecture families."""

    MLP = "mlp"
    CNN = "cnn"


class ModelTask(_StringEnum):
    """Public task identifiers shared by training and inference code."""

    AUTOENCODER = "autoencoder"
    CONTRASTIVE = "contrastive"
    TEXT_TO_BRAIN = "text_to_brain"
    BRAIN_TO_TEXT_RETRIEVAL = "brain_to_text_retrieval"
    BRAIN_TO_TEXT_GENERATION = "brain_to_text_generation"
    TEXT_ENCODER = "text_encoder"


class ModelDomain(_StringEnum):
    """Datasets with released domain-specific CNN checkpoints."""

    PUBMED = "pubmed"
    NILEARN = "nilearn"
    NEUROVAULT = "neurovault"


class ModelVariant(_StringEnum):
    """Canonical variants of released model artifacts."""

    DEFAULT = "default"
    MIXED_BASELINE = "mixed_baseline"
    FINETUNED = "finetuned"
    TEXT = "text"
    BRAIN = "brain"
    MSE = "mse"
    ADAPTER = "adapter"
    QFORMER = "qformer"
    SPECTER = "specter"


class ModelLoader(_StringEnum):
    """Internal loader dispatch keys.

    Keeping these keys in the registry makes resolution testable without
    loading a checkpoint, while :mod:`neurovlm.models` owns the actual imports.
    """

    MLP_AUTOENCODER = "mlp_autoencoder"
    MLP_TEXT_INFONCE = "mlp_text_infonce"
    MLP_IMAGE_INFONCE = "mlp_image_infonce"
    MLP_TEXT_MSE = "mlp_text_mse"
    MLP_SPECTER = "mlp_specter"
    MLP_NEURO_QFORMER = "mlp_neuro_qformer"
    MLP_NEURO_ADAPTER = "mlp_neuro_adapter"
    CNN_AUTOENCODER = "cnn_autoencoder"
    CNN_CONTRASTIVE = "cnn_contrastive"
    CNN_TEXT_TO_BRAIN = "cnn_text_to_brain"


@dataclass(frozen=True)
class ModelSpec:
    """An immutable, fully resolved model artifact specification."""

    family: ModelFamily
    task: ModelTask
    variant: ModelVariant
    loader: ModelLoader
    domain: ModelDomain | None = None
    loader_variant: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def canonical_name(self) -> str:
        """Return a stable, human-readable identifier for manifests."""

        parts = [self.family.value, self.task.value]
        if self.domain is not None:
            parts.append(self.domain.value)
        parts.append(self.variant.value)
        return ":".join(parts)


def _mlp_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.AUTOENCODER,
            ModelVariant.DEFAULT,
            ModelLoader.MLP_AUTOENCODER,
            aliases=("autoencoder",),
        ),
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.CONTRASTIVE,
            ModelVariant.TEXT,
            ModelLoader.MLP_TEXT_INFONCE,
            aliases=("proj_head_text_infonce",),
        ),
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.CONTRASTIVE,
            ModelVariant.BRAIN,
            ModelLoader.MLP_IMAGE_INFONCE,
            aliases=("proj_head_image_infonce",),
        ),
        # Brain-to-text retrieval projects the brain latent into the shared
        # contrastive space.  It intentionally aliases the same released
        # image/brain projection head as the contrastive ``brain`` component.
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.BRAIN_TO_TEXT_RETRIEVAL,
            ModelVariant.DEFAULT,
            ModelLoader.MLP_IMAGE_INFONCE,
        ),
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.TEXT_TO_BRAIN,
            ModelVariant.MSE,
            ModelLoader.MLP_TEXT_MSE,
            aliases=("proj_head_text_mse",),
        ),
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.TEXT_TO_BRAIN,
            ModelVariant.ADAPTER,
            ModelLoader.MLP_NEURO_ADAPTER,
            aliases=("neuro_adapter",),
        ),
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.BRAIN_TO_TEXT_GENERATION,
            ModelVariant.QFORMER,
            ModelLoader.MLP_NEURO_QFORMER,
            aliases=("neuro_qformer",),
        ),
        ModelSpec(
            ModelFamily.MLP,
            ModelTask.TEXT_ENCODER,
            ModelVariant.SPECTER,
            ModelLoader.MLP_SPECTER,
            aliases=("specter",),
        ),
    ]


def _cnn_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            ModelFamily.CNN,
            ModelTask.AUTOENCODER,
            ModelVariant.MIXED_BASELINE,
            ModelLoader.CNN_AUTOENCODER,
            loader_variant="mixed",
            aliases=("autoencoder_cnn", "autoencoder_cnn_mixed"),
        )
    ]
    for domain in ModelDomain:
        specs.append(
            ModelSpec(
                ModelFamily.CNN,
                ModelTask.AUTOENCODER,
                ModelVariant.FINETUNED,
                ModelLoader.CNN_AUTOENCODER,
                domain=domain,
                loader_variant=domain.value,
                aliases=(f"autoencoder_cnn_{domain.value}",),
            )
        )
        for task, loader, alias_prefix in (
            (ModelTask.CONTRASTIVE, ModelLoader.CNN_CONTRASTIVE, "contrastive_cnn"),
            (ModelTask.TEXT_TO_BRAIN, ModelLoader.CNN_TEXT_TO_BRAIN, "text_to_brain_cnn"),
        ):
            specs.extend(
                [
                    ModelSpec(
                        ModelFamily.CNN,
                        task,
                        ModelVariant.MIXED_BASELINE,
                        loader,
                        domain=domain,
                        loader_variant=f"mixed_to_{domain.value}",
                        aliases=(f"{alias_prefix}_mixed_to_{domain.value}",),
                    ),
                    ModelSpec(
                        ModelFamily.CNN,
                        task,
                        ModelVariant.FINETUNED,
                        loader,
                        domain=domain,
                        loader_variant=domain.value,
                        aliases=(f"{alias_prefix}_{domain.value}",),
                    ),
                ]
            )
    return specs


MODEL_SPECS: tuple[ModelSpec, ...] = tuple(_mlp_specs() + _cnn_specs())


def _unique_mapping(items: list[tuple[object, ModelSpec]], label: str) -> Mapping:
    output = {}
    for key, spec in items:
        if key in output:
            raise RuntimeError(f"Duplicate {label} {key!r} in the model registry")
        output[key] = spec
    return MappingProxyType(output)


MODEL_REGISTRY: Mapping[str, ModelSpec] = _unique_mapping(
    [(spec.canonical_name, spec) for spec in MODEL_SPECS], "canonical model name"
)
MODEL_ALIASES: Mapping[str, ModelSpec] = _unique_mapping(
    [(alias, spec) for spec in MODEL_SPECS for alias in spec.aliases], "model alias"
)
_STRUCTURED_REGISTRY: Mapping[
    tuple[ModelFamily, ModelTask, ModelDomain | None, ModelVariant], ModelSpec
] = _unique_mapping(
    [
        ((spec.family, spec.task, spec.domain, spec.variant), spec)
        for spec in MODEL_SPECS
    ],
    "structured model key",
)


def _coerce_enum(value, enum_type, field: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"Unknown {field} {value!r}; expected one of: {choices}") from error


def _default_variant(family: ModelFamily, task: ModelTask) -> ModelVariant:
    if family is ModelFamily.CNN:
        return ModelVariant.MIXED_BASELINE
    defaults = {
        ModelTask.AUTOENCODER: ModelVariant.DEFAULT,
        ModelTask.CONTRASTIVE: ModelVariant.TEXT,
        ModelTask.TEXT_TO_BRAIN: ModelVariant.MSE,
        ModelTask.BRAIN_TO_TEXT_RETRIEVAL: ModelVariant.DEFAULT,
        ModelTask.BRAIN_TO_TEXT_GENERATION: ModelVariant.QFORMER,
        ModelTask.TEXT_ENCODER: ModelVariant.SPECTER,
    }
    return defaults[task]


def resolve_model_spec(
    name: str | None = None,
    *,
    family: ModelFamily | str = ModelFamily.MLP,
    task: ModelTask | str | None = None,
    domain: ModelDomain | str | None = None,
    variant: ModelVariant | str | None = None,
) -> ModelSpec:
    """Resolve a legacy alias or structured selection to a canonical spec.

    CNN contrastive and text-to-brain selections require a domain and default
    to the released mixed-autoencoder baseline for that domain.  A
    domain-specialized CNN is selected only by explicitly passing
    ``variant="finetuned"``.  Legacy names remain explicit aliases to their
    historical artifacts.
    """

    if name is not None:
        if task is not None or domain is not None or variant is not None or family != ModelFamily.MLP:
            raise ValueError("Pass either a legacy model name or structured model fields, not both")
        try:
            return MODEL_ALIASES[name]
        except KeyError as error:
            valid = ", ".join(sorted(MODEL_ALIASES))
            raise ValueError(f"Unknown model name {name!r}; expected one of: {valid}") from error

    resolved_family = _coerce_enum(family, ModelFamily, "model family")
    if task is None:
        raise ValueError("task is required when loading a model by structured fields")
    resolved_task = _coerce_enum(task, ModelTask, "model task")
    resolved_domain = None if domain is None else _coerce_enum(domain, ModelDomain, "model domain")
    resolved_variant = (
        _default_variant(resolved_family, resolved_task)
        if variant is None
        else _coerce_enum(variant, ModelVariant, "model variant")
    )

    if resolved_family is ModelFamily.MLP and resolved_domain is not None:
        raise ValueError("MLP model selections do not accept a domain")

    if resolved_family is ModelFamily.CNN:
        if resolved_task in (ModelTask.CONTRASTIVE, ModelTask.TEXT_TO_BRAIN):
            if resolved_domain is None:
                raise ValueError(f"domain is required for CNN {resolved_task.value} models")
        elif resolved_task is ModelTask.AUTOENCODER:
            if resolved_variant is ModelVariant.MIXED_BASELINE and resolved_domain is not None:
                raise ValueError(
                    "The mixed-baseline CNN autoencoder is domain-independent; omit domain, "
                    "or explicitly use variant='finetuned'"
                )
            if resolved_variant is ModelVariant.FINETUNED and resolved_domain is None:
                raise ValueError("domain is required for a fine-tuned CNN autoencoder")

    key = (resolved_family, resolved_task, resolved_domain, resolved_variant)
    try:
        return _STRUCTURED_REGISTRY[key]
    except KeyError as error:
        fields = (
            f"family={resolved_family.value!r}, task={resolved_task.value!r}, "
            f"domain={None if resolved_domain is None else resolved_domain.value!r}, "
            f"variant={resolved_variant.value!r}"
        )
        raise ValueError(f"No released model matches {fields}") from error


__all__ = [
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "MODEL_SPECS",
    "ModelDomain",
    "ModelFamily",
    "ModelLoader",
    "ModelSpec",
    "ModelTask",
    "ModelVariant",
    "resolve_model_spec",
]
