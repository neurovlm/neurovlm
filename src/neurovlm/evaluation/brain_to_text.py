"""Evaluation primitives for MLP brain-to-text generation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class BrainToTextBatch:
    """Canonical batch consumed by Q-Former/causal-LM training."""

    raw_brain: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    semantic_brain: torch.Tensor | None
    sources: tuple[str, ...]
    sample_ids: tuple[str, ...]
    references: tuple[str | None, ...]


@dataclass(frozen=True)
class BrainToTextLMOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    token_losses: torch.Tensor
    token_mask: torch.Tensor


@dataclass(frozen=True)
class BrainToTextGenerationEvaluation:
    summary: Mapping[str, float]
    by_source: tuple[Mapping[str, Any], ...]
    by_sample: tuple[Mapping[str, Any], ...]
    generated: tuple[Mapping[str, Any], ...]
    n: int


def _mapping_tensor(batch: Mapping[str, Any], names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _strings(value: Any, n: int, *, default: Callable[[int], str | None]) -> tuple[str | None, ...]:
    if value is None:
        return tuple(default(i) for i in range(n))
    if isinstance(value, (str, int)):
        return tuple(str(value) for _ in range(n))
    values = list(value)
    if len(values) != n:
        raise ValueError(f"Metadata has {len(values)} rows for batch size {n}")
    return tuple(None if item is None else str(item) for item in values)


def parse_brain_to_text_batch(
    batch: Any,
    *,
    pad_token_id: int | None = None,
    offset: int = 0,
) -> BrainToTextBatch:
    """Normalize a mapping or retained tuple batch.

    Mapping batches require a 2-D brain latent and ``input_ids``.  Supported
    tuples are ``(brain, input_ids, attention_mask)`` and
    ``(brain, semantic_brain, input_ids, attention_mask)``.
    """

    semantic: torch.Tensor | None = None
    if isinstance(batch, Mapping):
        raw = _mapping_tensor(
            batch,
            ("brain_embedding", "brain_latent", "raw_brain", "image_embedding", "brain"),
        )
        semantic = _mapping_tensor(
            batch, ("semantic_embedding", "semantic_brain", "brain_semantic")
        )
        input_ids = _mapping_tensor(batch, ("input_ids",))
        attention = _mapping_tensor(batch, ("attention_mask", "attn_mask"))
    elif isinstance(batch, (tuple, list)) and len(batch) == 3:
        raw, input_ids, attention = batch
    elif isinstance(batch, (tuple, list)) and len(batch) == 4:
        raw, semantic, input_ids, attention = batch
    else:
        raise TypeError(
            "Brain-to-text batches must be mappings or (brain, input_ids, attention_mask) "
            "or (brain, semantic_brain, input_ids, attention_mask) tuples"
        )
    if not all(isinstance(value, torch.Tensor) for value in (raw, input_ids)):
        raise KeyError("Brain-to-text batches require tensor brain latents and input_ids")
    raw = raw if raw.ndim == 2 else raw.reshape(1, -1) if raw.ndim == 1 else raw
    input_ids = input_ids if input_ids.ndim == 2 else input_ids.reshape(1, -1) if input_ids.ndim == 1 else input_ids
    if raw.ndim != 2 or input_ids.ndim != 2 or len(raw) != len(input_ids):
        raise ValueError("brain latents and input_ids must be aligned 2-D tensors")
    if semantic is not None:
        semantic = semantic if semantic.ndim == 2 else semantic.reshape(1, -1) if semantic.ndim == 1 else semantic
        if semantic.ndim != 2 or len(semantic) != len(raw):
            raise ValueError("semantic brain latents must be an aligned 2-D tensor")
    if attention is None:
        if pad_token_id is None:
            attention = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention = input_ids.ne(int(pad_token_id)).long()
    if not isinstance(attention, torch.Tensor) or attention.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    n = len(raw)
    if isinstance(batch, Mapping):
        sources = _strings(batch.get("source"), n, default=lambda _: "unknown")
        sample_ids = _strings(
            batch.get("sample_id", batch.get("id")),
            n,
            default=lambda i: str(offset + i),
        )
        references = _strings(
            batch.get("reference_text", batch.get("text")), n, default=lambda _: None
        )
    else:
        sources = tuple("unknown" for _ in range(n))
        sample_ids = tuple(str(offset + i) for i in range(n))
        references = tuple(None for _ in range(n))
    return BrainToTextBatch(
        raw.float(),
        input_ids.long(),
        attention.long(),
        None if semantic is None else semantic.float(),
        tuple(str(v) for v in sources),
        tuple(str(v) for v in sample_ids),
        references,
    )


def brain_to_text_lm_forward(
    qformer: nn.Module,
    causal_lm: nn.Module,
    batch: BrainToTextBatch | Any,
    *,
    device: str | torch.device = "cpu",
    pad_token_id: int | None = None,
) -> BrainToTextLMOutput:
    """Apply the retained visual-prefix causal-LM objective."""

    resolved = torch.device(device)
    parsed = batch if isinstance(batch, BrainToTextBatch) else parse_brain_to_text_batch(
        batch, pad_token_id=pad_token_id
    )
    raw = parsed.raw_brain.to(resolved)
    semantic = None if parsed.semantic_brain is None else parsed.semantic_brain.to(resolved)
    input_ids = parsed.input_ids.to(resolved)
    text_mask = parsed.attention_mask.to(resolved)
    visual = qformer(raw, semantic) if semantic is not None else qformer(raw)
    with torch.no_grad():
        text_embeddings = causal_lm.get_input_embeddings()(input_ids)
    visual = visual.to(dtype=text_embeddings.dtype)
    inputs_embeds = torch.cat((visual, text_embeddings), dim=1)
    visual_mask = torch.ones(
        len(raw), visual.shape[1], dtype=text_mask.dtype, device=resolved
    )
    full_mask = torch.cat((visual_mask, text_mask), dim=1)
    visual_labels = torch.full(
        (len(raw), visual.shape[1]), -100, dtype=torch.long, device=resolved
    )
    # Unlike the retained exploratory notebook, padding must not contribute to
    # the causal objective merely because its attention value is zero.
    text_labels = input_ids.masked_fill(text_mask.eq(0), -100)
    labels = torch.cat((visual_labels, text_labels), dim=1)
    output = causal_lm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels,
    )
    if not hasattr(output, "loss") or not hasattr(output, "logits"):
        raise TypeError("causal_lm must return an object with loss and logits")
    logits = output.logits
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    token_mask = shifted_labels.ne(-100)
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100, reduction="none"
    )
    return BrainToTextLMOutput(
        output.loss, logits, labels, full_mask, token_losses, token_mask
    )


def _loader(data: Dataset | DataLoader, batch_size: int, num_workers: int) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    tokens = sum(int(row["token_count"]) for row in rows)
    if tokens == 0:
        raise RuntimeError("Brain-to-text evaluation produced no supervised tokens")
    loss = sum(float(row["loss_sum"]) for row in rows) / tokens
    correct = sum(int(row["correct_tokens"]) for row in rows)
    return {
        "loss": float(loss),
        "perplexity": float(math.exp(min(loss, 80.0))),
        "token_accuracy": float(correct / tokens),
        "token_count": float(tokens),
    }


@torch.no_grad()
def evaluate_brain_to_text_generation(
    qformer: nn.Module,
    causal_lm: nn.Module,
    data: Dataset | DataLoader,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    max_batches: int | None = None,
    pad_token_id: int | None = None,
    generated_samples_limit: int = 0,
    generation_callback: Callable[[nn.Module, nn.Module, BrainToTextBatch], Sequence[str]] | None = None,
    semantic_metric_callback: Callable[[Sequence[str], Sequence[str | None], Sequence[Mapping[str, Any]]], Mapping[str, float]] | None = None,
) -> BrainToTextGenerationEvaluation:
    """Evaluate loss/token metrics and optional bounded generated text."""

    if batch_size < 1 or generated_samples_limit < 0:
        raise ValueError("batch_size must be positive and generated_samples_limit non-negative")
    if generated_samples_limit and generation_callback is None:
        raise ValueError("generated_samples_limit requires generation_callback")
    resolved = torch.device(device)
    qformer_was_training, lm_was_training = qformer.training, causal_lm.training
    qformer.to(resolved).eval()
    causal_lm.to(resolved).eval()
    sample_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    offset = 0
    for index, raw_batch in enumerate(_loader(data, batch_size, num_workers)):
        if max_batches is not None and index >= max_batches:
            break
        batch = parse_brain_to_text_batch(raw_batch, pad_token_id=pad_token_id, offset=offset)
        output = brain_to_text_lm_forward(
            qformer, causal_lm, batch, device=resolved, pad_token_id=pad_token_id
        )
        predicted = output.logits[:, :-1].argmax(-1)
        for row_index in range(len(batch.raw_brain)):
            mask = output.token_mask[row_index]
            count = int(mask.sum())
            loss_sum = float(output.token_losses[row_index][mask].sum())
            correct = int(
                predicted[row_index][mask].eq(output.labels[row_index, 1:][mask]).sum()
            )
            sample_rows.append(
                {
                    "sample_id": batch.sample_ids[row_index],
                    "source": batch.sources[row_index],
                    "loss": loss_sum / max(count, 1),
                    "perplexity": math.exp(min(loss_sum / max(count, 1), 80.0)),
                    "token_accuracy": correct / max(count, 1),
                    "token_count": count,
                    "loss_sum": loss_sum,
                    "correct_tokens": correct,
                }
            )
        remaining = generated_samples_limit - len(generated)
        if remaining > 0 and generation_callback is not None:
            values = list(generation_callback(qformer, causal_lm, batch))
            if len(values) != len(batch.raw_brain):
                raise ValueError("generation_callback must return one string per batch row")
            for row_index, value in enumerate(values[:remaining]):
                generated.append(
                    {
                        "sample_id": batch.sample_ids[row_index],
                        "source": batch.sources[row_index],
                        "generated_text": str(value),
                        "reference_text": batch.references[row_index],
                    }
                )
        offset += len(batch.raw_brain)
    qformer.train(qformer_was_training)
    causal_lm.train(lm_was_training)
    if not sample_rows:
        raise RuntimeError("Brain-to-text evaluation dataset produced no batches")
    summary = _aggregate(sample_rows)
    positions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        positions[str(row["source"])].append(row)
    by_source = tuple(
        {"source": source, "n": len(rows), **_aggregate(rows)}
        for source, rows in sorted(positions.items())
    )
    if semantic_metric_callback is not None and generated:
        semantic = semantic_metric_callback(
            [str(row["generated_text"]) for row in generated],
            [row.get("reference_text") for row in generated],
            generated,
        )
        summary.update({str(name): float(value) for name, value in semantic.items()})
    public_sample_rows = tuple(
        {key: value for key, value in row.items() if key not in {"loss_sum", "correct_tokens"}}
        for row in sample_rows
    )
    return BrainToTextGenerationEvaluation(
        summary, by_source, public_sample_rows, tuple(generated), len(sample_rows)
    )


__all__ = [
    "BrainToTextBatch",
    "BrainToTextGenerationEvaluation",
    "BrainToTextLMOutput",
    "brain_to_text_lm_forward",
    "evaluate_brain_to_text_generation",
    "parse_brain_to_text_batch",
]
