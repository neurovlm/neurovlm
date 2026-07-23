"""Shared MLP/CNN model-comparison entry points.

The default data source is :class:`AtlasFreeCNNDataProvider`, whose published
split rows and shared volume tensor are resolved from Hugging Face.  Historical
JSONL path fields are never consulted.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from neurovlm.atlas_free_dataset import AtlasFreeCNNDataProvider, canonical_atlas_free_domain
from neurovlm.atlas_free_text import (
    AtlasFreeContrastiveCollator,
    AtlasFreeTextEmbeddingLookup,
    primary_positive_text,
    primary_positive_text_id,
)
from neurovlm.cnn import atlas_free_volume_to_mlp_flat
from neurovlm.evaluation.contrastive import evaluate_contrastive
from neurovlm.evaluation.spatial import reconstruction_metrics
from neurovlm.model_registry import ModelDomain, ModelFamily, ModelTask, ModelVariant
from neurovlm.runtime import NeuroVLMRuntime, load_pipeline


@dataclass(frozen=True)
class ComparisonSelection:
    family: str
    task: str
    domain: str | None = None
    variant: str | None = None
    checkpoint: str | Path | None = None
    from_run: str | Path | None = None
    evaluation_domain: str | None = None

    @property
    def model_id(self) -> str:
        parts = [self.family, self.task]
        if self.family == "cnn":
            parts.append(self.variant or "mixed_baseline")
        if self.domain:
            parts.append(self.domain)
        if self.evaluation_domain:
            parts.extend(("eval", self.evaluation_domain))
        return "_".join(parts)

    def load_kwargs(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "task": self.task,
            "domain": self.domain,
            "variant": self.variant,
            "checkpoint": self.checkpoint,
            "from_run": self.from_run,
        }


@dataclass(frozen=True)
class ComparisonResult:
    summary: tuple[Mapping[str, Any], ...]
    by_source: tuple[Mapping[str, Any], ...]
    by_sample: tuple[Mapping[str, Any], ...]
    recall_curves: tuple[Mapping[str, Any], ...]
    manifest: tuple[Mapping[str, Any], ...]


def default_comparison_matrix(
    task: str,
    *,
    domains: Iterable[str] = ("pubmed", "nilearn", "neurovault"),
    include_finetuned: bool = False,
) -> tuple[ComparisonSelection, ...]:
    """Return MLP plus mixed-baseline CNN selections for a task.

    Every domain receives an MLP row and a mixed-baseline CNN row evaluated on
    the exact same subset. Fine-tuned CNN branches are added only when
    explicitly requested.
    """

    resolved_task = ModelTask(task)
    if resolved_task not in {
        ModelTask.AUTOENCODER,
        ModelTask.CONTRASTIVE,
        ModelTask.TEXT_TO_BRAIN,
    }:
        raise ValueError(f"Comparison task {task!r} is not supported")
    rows = []
    for domain in domains:
        value = ModelDomain(domain).value
        rows.append(ComparisonSelection("mlp", resolved_task.value, evaluation_domain=value))
        if resolved_task is ModelTask.AUTOENCODER:
            rows.append(ComparisonSelection(
                "cnn", task, variant="mixed_baseline", evaluation_domain=value
            ))
        else:
            rows.append(ComparisonSelection(
                "cnn", task, domain=value, variant="mixed_baseline", evaluation_domain=value
            ))
        if include_finetuned:
            rows.append(ComparisonSelection(
                "cnn", task, domain=value, variant="finetuned", evaluation_domain=value
            ))
    return tuple(rows)


def _status(error: Exception) -> str:
    if isinstance(error, (FileNotFoundError, OSError)):
        return "missing_checkpoint"
    if isinstance(error, (ValueError, NotImplementedError)):
        return "unsupported"
    return "error"


def resolve_comparison_manifest(
    selections: Sequence[ComparisonSelection],
    *,
    device: str | torch.device = "cpu",
    loader: Callable[..., NeuroVLMRuntime] = load_pipeline,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve every selection independently; missing artifacts never abort the matrix."""

    output = []
    for selection in selections:
        try:
            runtime = loader(**selection.load_kwargs(), device=device)
            output.append({
                "model_id": selection.model_id,
                "evaluation_domain": selection.evaluation_domain,
                "status": "resolved",
                "error": None,
                **runtime.metadata.as_dict(),
            })
        except Exception as error:  # matrix resolution intentionally isolates failures
            output.append({
                "model_id": selection.model_id,
                "evaluation_domain": selection.evaluation_domain,
                "family": selection.family,
                "task": selection.task,
                "domain": selection.domain,
                "variant": selection.variant or ("mixed_baseline" if selection.family == "cnn" else None),
                "status": _status(error),
                "error": str(error),
                "checkpoint": None,
            })
    return tuple(output)


def write_comparison_manifest(
    path: str | Path,
    selections: Sequence[ComparisonSelection],
    **kwargs: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolve_comparison_manifest(selections, **kwargs), indent=2) + "\n")
    return path


def _data(data: Dataset | None, provider: AtlasFreeCNNDataProvider | None, split: str) -> Dataset:
    if data is not None and provider is not None:
        raise ValueError("Pass either data or provider, not both")
    if data is not None:
        return data
    return (provider or AtlasFreeCNNDataProvider()).split(split)


def _source(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        return str(metadata.get("source") or canonical_atlas_free_domain(metadata))
    return str(row.get("source") or "unknown")


def _domain(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    return canonical_atlas_free_domain(metadata if isinstance(metadata, Mapping) else row)


class _DomainView(Dataset):
    """A metadata-only split view; volume tensors remain owned by the source dataset."""

    def __init__(self, dataset: Dataset, domain: str | None):
        self.dataset = dataset
        self.domain = domain
        self.indices = [
            index for index in range(len(dataset))
            if domain is None or _domain(dataset[index]) == domain
        ]
        self.rows = [dataset[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.dataset[self.indices[index]]


def _volume_collate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "volume": torch.stack([torch.as_tensor(row["volume"]).float() for row in rows]),
        "map_id": [str(row.get("map_id", "")) for row in rows],
        "source": [_source(row) for row in rows],
    }


def _volume_loader(data: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=_volume_collate)


def _base(selection: ComparisonSelection, runtime: NeuroVLMRuntime) -> dict[str, Any]:
    return {
        "model_id": selection.model_id,
        "family": runtime.metadata.family,
        "task": runtime.metadata.task,
        "domain": runtime.metadata.domain,
        "variant": runtime.metadata.variant,
        "evaluation_domain": selection.evaluation_domain,
        "status": "resolved",
    }


def _space(runtime: NeuroVLMRuntime) -> str:
    return "mlp_masker_flatmap" if runtime.metadata.family == "mlp" else "native_atlas_free_volume"


def _comparison_base(
    selection: ComparisonSelection,
    runtime: NeuroVLMRuntime,
    *,
    text_preprocessing: str | None = None,
) -> dict[str, Any]:
    base = {
        **_base(selection, runtime),
        "comparison_protocol": "paired_atlas_free",
        "comparison_space": _space(runtime),
    }
    if text_preprocessing is not None:
        base["text_preprocessing"] = text_preprocessing
    return base


def _mean_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = {
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float)) and key != "n"
    }
    output = {}
    for name in sorted(names):
        values = [float(row[name]) for row in rows if name in row and math.isfinite(float(row[name]))]
        output[name] = float(sum(values) / len(values)) if values else float("nan")
    return output


def _group(rows: Sequence[Mapping[str, Any]], base: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    return tuple(
        {**base, "source": source, "n": len(values), **_mean_rows(values)}
        for source, values in sorted(grouped.items())
    )


def _load_runtimes(
    selections: Sequence[ComparisonSelection], device: str | torch.device
) -> tuple[list[tuple[ComparisonSelection, NeuroVLMRuntime]], tuple[Mapping[str, Any], ...]]:
    loaded = []
    manifest = []
    for selection in selections:
        try:
            runtime = load_pipeline(**selection.load_kwargs(), device=device)
            loaded.append((selection, runtime))
            manifest.append({"model_id": selection.model_id, "status": "resolved", "error": None,
                             "evaluation_domain": selection.evaluation_domain,
                             **runtime.metadata.as_dict()})
        except Exception as error:
            manifest.append({"model_id": selection.model_id, "family": selection.family,
                             "task": selection.task, "domain": selection.domain,
                             "evaluation_domain": selection.evaluation_domain,
                             "variant": selection.variant, "status": _status(error), "error": str(error)})
    return loaded, tuple(manifest)


def _skipped(manifest: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        {"model_id": row["model_id"], "family": row.get("family"), "task": row.get("task"),
         "domain": row.get("domain"), "variant": row.get("variant"), "status": row["status"],
         "evaluation_domain": row.get("evaluation_domain"),
         "reason": row.get("error"), "n": 0}
        for row in manifest if row["status"] != "resolved"
    ]


def evaluate_reconstruction_comparison(
    *,
    selections: Sequence[ComparisonSelection] | None = None,
    data: Dataset | None = None,
    provider: AtlasFreeCNNDataProvider | None = None,
    split: str = "test",
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    include_finetuned: bool = False,
) -> ComparisonResult:
    """Compare AE reconstruction in each family's declared spatial space."""

    chosen = tuple(selections or default_comparison_matrix("autoencoder", include_finetuned=include_finetuned))
    if any(item.task != "autoencoder" for item in chosen):
        raise ValueError("All reconstruction selections must use task='autoencoder'")
    dataset = _data(data, provider, split)
    loaded, manifest = _load_runtimes(chosen, device)
    summary: list[Mapping[str, Any]] = _skipped(manifest)
    by_source: list[Mapping[str, Any]] = []
    by_sample: list[Mapping[str, Any]] = []
    for selection, runtime in loaded:
        rows = []
        selected_data = _DomainView(dataset, selection.evaluation_domain)
        if not len(selected_data):
            base = _comparison_base(selection, runtime)
            summary.append({**base, "status": "unsupported_dataset", "reason":
                            f"No samples for evaluation domain {selection.evaluation_domain!r}", "n": 0})
            continue
        for batch in _volume_loader(selected_data, batch_size):
            target = batch["volume"]
            if runtime.metadata.family == "mlp":
                target = atlas_free_volume_to_mlp_flat(target, binarize=True)
            prediction = runtime.reconstruct(target).cpu()
            target = target.cpu()
            for index in range(len(target)):
                metrics = reconstruction_metrics(prediction[index:index + 1], target[index:index + 1])
                rows.append({"sample_id": batch["map_id"][index], "source": batch["source"][index], **metrics})
        base = _comparison_base(selection, runtime)
        summary.append({**base, "n": len(rows), **_mean_rows(rows)})
        by_source.extend(_group(rows, base))
        by_sample.extend({**base, **row} for row in rows)
    return ComparisonResult(tuple(summary), tuple(by_source), tuple(by_sample), (), manifest)


class _RuntimeContrastive(nn.Module):
    def __init__(self, runtime: NeuroVLMRuntime):
        super().__init__()
        self.runtime = runtime

    def forward(self, volume: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        brain = volume
        if self.runtime.metadata.family == "mlp":
            brain = atlas_free_volume_to_mlp_flat(volume, binarize=True)
        return self.runtime.encode_brain(brain), self.runtime.encode_text(text)


def _family_text_lookup(
    data: _DomainView,
    encoder: Callable[[Sequence[str]], Tensor],
    *,
    batch_size: int,
) -> AtlasFreeTextEmbeddingLookup:
    """Encode paired rows in bounded batches with the released MLP convention."""

    rows = data.rows
    text_ids = [primary_positive_text_id(row) for row in rows]
    texts = [primary_positive_text(row) for row in rows]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches = []
    for start in range(0, len(texts), batch_size):
        text_batch = texts[start:start + batch_size]
        encoded = torch.as_tensor(
            encoder(text_batch), dtype=torch.float32
        ).detach().cpu()
        if encoded.ndim != 2 or encoded.shape != (len(text_batch), 768):
            raise ValueError(
                "MLP text encoder must return one 768-dimensional embedding per input; "
                f"got {tuple(encoded.shape)} for {len(text_batch)} texts"
            )
        batches.append(encoded)
    embeddings = torch.cat(batches)
    embeddings = F.normalize(embeddings, dim=1)
    return AtlasFreeTextEmbeddingLookup(
        embeddings,
        text_ids,
        {
            "source": "family_native_runtime_encoding",
            "text_preprocessing": "specter2_adhoc_query_orthogonalized_then_l2",
        },
    )


def _default_mlp_text_encoder(device: str | torch.device) -> Callable[[Sequence[str]], Tensor]:
    """Construct the MLP family's released SPECTER2 query encoder lazily."""

    from neurovlm.models import Specter

    return Specter(device=str(device))


def evaluate_contrastive_comparison(
    *,
    selections: Sequence[ComparisonSelection] | None = None,
    data: Dataset | None = None,
    provider: AtlasFreeCNNDataProvider | None = None,
    lookup: AtlasFreeTextEmbeddingLookup | None = None,
    mlp_text_encoder: Callable[[Sequence[str]], Tensor] | None = None,
    split: str = "test",
    domains: Iterable[str] = ("pubmed", "nilearn", "neurovault"),
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    include_finetuned: bool = False,
) -> ComparisonResult:
    """Evaluate paired full-split retrieval with family-native text preprocessing.

    Every family sees the same atlas-free map/text rows. CNN models consume the
    immutable published normalized cache; MLP models re-encode the raw positive
    text with their released SPECTER2 ``adhoc_query`` convention. This paired
    protocol is intentionally distinct from historical family-native benchmarks
    that used different sample cohorts.
    """

    chosen = tuple(selections or default_comparison_matrix(
        "contrastive", domains=domains, include_finetuned=include_finetuned
    ))
    if any(item.task != "contrastive" for item in chosen):
        raise ValueError("All retrieval selections must use task='contrastive'")
    dataset = _data(data, provider, split)
    cnn_lookup = lookup
    loaded, manifest = _load_runtimes(chosen, device)
    summary: list[Mapping[str, Any]] = _skipped(manifest)
    by_source: list[Mapping[str, Any]] = []
    by_sample: list[Mapping[str, Any]] = []
    curves: list[Mapping[str, Any]] = []
    resolved_mlp_text_encoder = mlp_text_encoder
    for selection, runtime in loaded:
        selected_data = _DomainView(dataset, selection.evaluation_domain)
        text_preprocessing = (
            "specter2_adhoc_query_orthogonalized_then_l2"
            if runtime.metadata.family == "mlp"
            else "empty_string_centered_l2_unit_normalized"
        )
        if not len(selected_data):
            base = _comparison_base(
                selection, runtime, text_preprocessing=text_preprocessing
            )
            summary.append({**base, "status": "unsupported_dataset", "reason":
                            f"No samples for evaluation domain {selection.evaluation_domain!r}", "n": 0})
            continue
        if runtime.metadata.family == "mlp":
            if resolved_mlp_text_encoder is None:
                resolved_mlp_text_encoder = _default_mlp_text_encoder(device)
            runtime_lookup = _family_text_lookup(
                selected_data,
                resolved_mlp_text_encoder,
                batch_size=batch_size,
            )
        else:
            if cnn_lookup is None:
                cnn_lookup = AtlasFreeTextEmbeddingLookup.published()
            runtime_lookup = cnn_lookup
        result = evaluate_contrastive(
            _RuntimeContrastive(runtime),
            selected_data,
            lookup=runtime_lookup,
            device=device,
            batch_size=batch_size,
        )
        base = _comparison_base(
            selection, runtime, text_preprocessing=text_preprocessing
        )
        summary.append({**base, "n": result.n, **result.summary})
        by_source.extend({**base, **row} for row in result.by_source)
        curves.extend({**base, **row} for row in result.recall_curves)
        paired = (F.normalize(result.brain_embeddings, dim=1) *
                  F.normalize(result.text_embeddings, dim=1)).sum(1)
        # Collator ordering equals dataset ordering; map/source metadata are
        # read from validated rows without consulting their legacy paths.
        for index, value in enumerate(paired):
            row = selected_data[index]
            by_sample.append({**base, "sample_id": str(row.get("map_id", index)),
                              "source": _source(row), "paired_cosine_similarity": float(value)})
    return ComparisonResult(tuple(summary), tuple(by_source), tuple(by_sample), tuple(curves), manifest)


def evaluate_text_to_brain_comparison(
    *,
    selections: Sequence[ComparisonSelection] | None = None,
    data: Dataset | None = None,
    provider: AtlasFreeCNNDataProvider | None = None,
    lookup: AtlasFreeTextEmbeddingLookup | None = None,
    mlp_text_encoder: Callable[[Sequence[str]], Tensor] | None = None,
    split: str = "test",
    domains: Iterable[str] = ("pubmed", "nilearn", "neurovault"),
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    include_finetuned: bool = False,
) -> ComparisonResult:
    """Compare generated maps with paired rows and family-native text inputs."""

    chosen = tuple(selections or default_comparison_matrix(
        "text_to_brain", domains=domains, include_finetuned=include_finetuned
    ))
    if any(item.task != "text_to_brain" for item in chosen):
        raise ValueError("All generation selections must use task='text_to_brain'")
    dataset = _data(data, provider, split)
    cnn_lookup = lookup
    loaded, manifest = _load_runtimes(chosen, device)
    summary: list[Mapping[str, Any]] = _skipped(manifest)
    by_source: list[Mapping[str, Any]] = []
    by_sample: list[Mapping[str, Any]] = []
    resolved_mlp_text_encoder = mlp_text_encoder
    for selection, runtime in loaded:
        rows = []
        selected_data = _DomainView(dataset, selection.evaluation_domain)
        text_preprocessing = (
            "specter2_adhoc_query_orthogonalized_then_l2"
            if runtime.metadata.family == "mlp"
            else "empty_string_centered_l2_unit_normalized"
        )
        if not len(selected_data):
            base = _comparison_base(
                selection, runtime, text_preprocessing=text_preprocessing
            )
            summary.append({**base, "status": "unsupported_dataset", "reason":
                            f"No samples for evaluation domain {selection.evaluation_domain!r}", "n": 0})
            continue
        if runtime.metadata.family == "mlp":
            if resolved_mlp_text_encoder is None:
                resolved_mlp_text_encoder = _default_mlp_text_encoder(device)
            runtime_lookup = _family_text_lookup(
                selected_data,
                resolved_mlp_text_encoder,
                batch_size=batch_size,
            )
        else:
            if cnn_lookup is None:
                cnn_lookup = AtlasFreeTextEmbeddingLookup.published()
            runtime_lookup = cnn_lookup
        loader = DataLoader(selected_data, batch_size=batch_size, shuffle=False,
                            collate_fn=AtlasFreeContrastiveCollator(runtime_lookup, (36, 45, 38)))
        for batch in loader:
            target = batch["volume"]
            if runtime.metadata.family == "mlp":
                target = atlas_free_volume_to_mlp_flat(target, binarize=True)
            prediction = runtime.generate(batch["text_embedding"]).cpu()
            target = target.cpu()
            for index in range(len(target)):
                metrics = reconstruction_metrics(prediction[index:index + 1], target[index:index + 1])
                rows.append({"sample_id": str(batch["map_id"][index]),
                             "text_id": str(batch["text_id"][index]),
                             "source": str(batch["source"][index]), **metrics})
        base = _comparison_base(
            selection, runtime, text_preprocessing=text_preprocessing
        )
        summary.append({**base, "n": len(rows), **_mean_rows(rows)})
        by_source.extend(_group(rows, base))
        by_sample.extend({**base, **row} for row in rows)
    return ComparisonResult(tuple(summary), tuple(by_source), tuple(by_sample), (), manifest)


__all__ = [
    "ComparisonResult",
    "ComparisonSelection",
    "default_comparison_matrix",
    "evaluate_contrastive_comparison",
    "evaluate_reconstruction_comparison",
    "evaluate_text_to_brain_comparison",
    "resolve_comparison_manifest",
    "write_comparison_manifest",
]
