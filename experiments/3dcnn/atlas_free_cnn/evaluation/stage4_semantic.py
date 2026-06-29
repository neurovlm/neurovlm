"""Semantic retrieval evaluation for Stage 4 generated brain maps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atlas_free_cnn.evaluation.generation_metrics import generation_metrics
from atlas_free_cnn.evaluation.metrics import ranking_metrics
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset
from atlas_free_cnn.training.model_wrappers import (
    build_brain_encoder,
    build_cnn_autoencoder,
    build_generative_text_to_ae_latent,
    build_text_projection,
    build_text_to_brain_projection,
    load_autoencoder_checkpoint,
)
from atlas_free_cnn.training.source_sampling import canonical_source


def normalize_text_key(text: str | None) -> str:
    return " ".join(str(text or "").split()).casefold()


def primary_text(row: dict[str, Any], *, text_rank: int = 0) -> dict[str, str]:
    positives = row.get("positive_texts", []) or []
    if positives:
        pos = positives[min(int(text_rank), len(positives) - 1)]
        return {
            "text": str(pos.get("text") or pos.get("title") or ""),
            "text_id": str(pos.get("text_id") or pos.get("id") or pos.get("text") or ""),
        }
    return {
        "text": str(row.get("primary_text") or row.get("text") or row.get("_primary_text") or ""),
        "text_id": str(row.get("primary_text_id") or row.get("text_id") or row.get("_primary_text_id") or ""),
    }


def publication_or_group_key(row: dict[str, Any]) -> str:
    source = canonical_source(row)
    nested = row.get("negative_sampling_groups")
    candidates: list[tuple[str, Any]] = []
    for key in ["publication_id", "pmid", "doi", "collection_id", "group_id", "study_id", "atlas", "source_detail"]:
        candidates.append((key, row.get(key)))
    if isinstance(nested, dict):
        for key in ["publication_id", "pmid", "collection_id", "group_id", "atlas", "map_type", "source"]:
            candidates.append((f"negative_sampling_groups.{key}", nested.get(key)))
    for pos in row.get("positive_texts", []) or []:
        if isinstance(pos, dict):
            for key in ["publication_id", "pmid", "doi", "collection_id", "group_id"]:
                candidates.append((f"positive_texts.{key}", pos.get(key)))
    for key, value in candidates:
        if value not in {None, ""}:
            return f"{source}:{key}:{value}"
    return f"{source}:map_id:{row.get('map_id') or row.get('_map_id') or ''}"


def _mask_from_keys(keys: list[str], *, identity_fallback: bool = True) -> torch.Tensor:
    n = len(keys)
    mask = torch.zeros((n, n), dtype=torch.bool)
    grouped: dict[str, list[int]] = {}
    for idx, key in enumerate(keys):
        if key:
            grouped.setdefault(key, []).append(idx)
    for idxs in grouped.values():
        idx = torch.tensor(idxs, dtype=torch.long)
        mask[idx[:, None], idx[None, :]] = True
    if identity_fallback:
        mask |= torch.eye(n, dtype=torch.bool)
    return mask


def semantic_positive_masks(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    map_keys = [str(row.get("map_id") or row.get("_map_id") or i) for i, row in enumerate(records)]
    text_keys = [normalize_text_key(str(row.get("_primary_text") or primary_text(row)["text"])) for row in records]
    group_keys = [publication_or_group_key(row) for row in records]
    return {
        "strict_map": _mask_from_keys(map_keys),
        "same_text_group": _mask_from_keys(text_keys),
        "publication_group": _mask_from_keys(group_keys),
    }


def semantic_group_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    text_counts = Counter(normalize_text_key(str(row.get("_primary_text") or primary_text(row)["text"])) for row in records)
    group_counts = Counter(publication_or_group_key(row) for row in records)
    return {
        "semantic_eval_n": float(len(records)),
        "same_text_duplicate_groups": float(sum(1 for count in text_counts.values() if count > 1)),
        "same_text_max_group_size": float(max(text_counts.values(), default=0)),
        "publication_group_duplicate_groups": float(sum(1 for count in group_counts.values() if count > 1)),
        "publication_group_max_group_size": float(max(group_counts.values(), default=0)),
    }


def _copy_rank_metrics(out: dict[str, float], base: str, t2g: dict[str, float], g2t: dict[str, float]) -> None:
    t_auc = float(t2g.get("normalized_recall_curve_auc", float("nan")))
    g_auc = float(g2t.get("normalized_recall_curve_auc", float("nan")))
    out[f"{base}_text_to_brain_normalized_auc"] = t_auc
    out[f"{base}_brain_to_text_normalized_auc"] = g_auc
    out[f"{base}_mean_normalized_auc"] = (t_auc + g_auc) / 2.0
    for k in (1, 5, 10, 50):
        t_key = f"recall@{k}"
        if t_key in t2g and t_key in g2t:
            out[f"{base}_recall@{k}"] = (float(t2g[t_key]) + float(g2t[t_key])) / 2.0
    if "mrr" in t2g and "mrr" in g2t:
        out[f"{base}_mrr"] = (float(t2g["mrr"]) + float(g2t["mrr"])) / 2.0


def _cosine_summary(text_embeddings: torch.Tensor, generated_embeddings: torch.Tensor, *, prefix: str) -> dict[str, float]:
    text_n = F.normalize(text_embeddings.float(), dim=1, eps=1e-8)
    gen_n = F.normalize(generated_embeddings.float(), dim=1, eps=1e-8)
    sim = text_n @ gen_n.T
    matched = sim.diag()
    if sim.shape[0] > 1:
        perm = torch.randperm(sim.shape[0])
        shuffled = sim[torch.arange(sim.shape[0]), perm]
        shuffled_mean = float(shuffled.mean().item())
    else:
        shuffled_mean = float("nan")
    return {
        f"{prefix}_matched_contrastive_cosine": float(matched.mean().item()),
        f"{prefix}_shuffled_contrastive_cosine": shuffled_mean,
    }


def semantic_retrieval_metrics(
    text_embeddings: torch.Tensor,
    generated_embeddings: torch.Tensor,
    records: list[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, float]:
    text_n = F.normalize(text_embeddings.float(), dim=1, eps=1e-8)
    gen_n = F.normalize(generated_embeddings.float(), dim=1, eps=1e-8)
    text_to_generated = text_n @ gen_n.T
    generated_to_text = gen_n @ text_n.T
    masks = semantic_positive_masks(records)
    out = semantic_group_summary(records)
    for name, mask in masks.items():
        t2g = ranking_metrics(text_to_generated, mask, ks=(1, 5, 10, 50))
        g2t = ranking_metrics(generated_to_text, mask, ks=(1, 5, 10, 50))
        _copy_rank_metrics(out, f"{prefix}_{name}", t2g, g2t)
    out.update(_cosine_summary(text_embeddings, generated_embeddings, prefix=prefix))
    return out


def stage4_metric_aliases(metrics: dict[str, Any], *, prefix: str = "generation") -> dict[str, Any]:
    """Return stable notebook/report column names for Stage 4 semantic metrics."""

    return {
        "raw_strict_auc": metrics.get(f"{prefix}_raw_strict_map_mean_normalized_auc", ""),
        "clamped_strict_auc": metrics.get(f"{prefix}_clamped_strict_map_mean_normalized_auc", metrics.get(f"{prefix}_mean_normalized_auc", "")),
        "same_text_group_auc": metrics.get(f"{prefix}_clamped_same_text_group_mean_normalized_auc", ""),
        "publication_group_auc": metrics.get(f"{prefix}_clamped_publication_group_mean_normalized_auc", ""),
        "matched_cosine": metrics.get(f"{prefix}_clamped_matched_contrastive_cosine", metrics.get(f"{prefix}_matched_contrastive_cosine", "")),
        "shuffled_cosine": metrics.get(f"{prefix}_clamped_shuffled_contrastive_cosine", metrics.get(f"{prefix}_shuffled_contrastive_cosine", "")),
    }


def stack_text_cache(text_cache: dict[str, torch.Tensor], texts: Iterable[str]) -> torch.Tensor:
    text_list = [str(text) for text in texts]
    missing = [text for text in text_list if text not in text_cache]
    if missing:
        raise KeyError(f"{len(missing)} texts missing from embedding cache; example={missing[0][:160]}")
    return torch.stack([text_cache[text] for text in text_list])


def records_from_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    text_entries = batch.get("text_entries") or [{} for _ in batch["texts"]]
    for idx, text in enumerate(batch["texts"]):
        row = dict(batch.get("metadata", [{}])[idx] or {})
        row["_map_id"] = str(batch.get("map_id", [""])[idx])
        row.setdefault("map_id", row["_map_id"])
        row["_primary_text"] = str(text)
        entry = text_entries[idx] if idx < len(text_entries) and isinstance(text_entries[idx], dict) else {}
        row["_primary_text_id"] = str(entry.get("text_id") or entry.get("id") or entry.get("text") or text)
        records.append(row)
    return records


@torch.no_grad()
def evaluate_generation_semantic_loader(
    autoencoder,
    text_to_latent,
    stage3_brain_encoder,
    stage3_text_projection,
    loader: DataLoader,
    generation_text_cache: dict[str, torch.Tensor],
    device: torch.device,
    *,
    evaluator_text_cache: dict[str, torch.Tensor] | None = None,
    prefix: str = "generation",
    include_raw: bool = True,
    include_clamped: bool = True,
) -> dict[str, float]:
    evaluator_text_cache = evaluator_text_cache or generation_text_cache
    generated_by_variant: dict[str, list[torch.Tensor]] = {}
    if include_raw:
        generated_by_variant["raw"] = []
    if include_clamped:
        generated_by_variant["clamped"] = []
    text_embeddings = []
    records: list[dict[str, Any]] = []
    raw_value_min = float("inf")
    raw_value_max = -float("inf")
    raw_below_zero = 0.0
    raw_above_one = 0.0
    n_voxels = 0.0
    for batch in loader:
        gen_text = stack_text_cache(generation_text_cache, batch["texts"]).to(device)
        eval_text = stack_text_cache(evaluator_text_cache, batch["texts"]).to(device)
        text_latent = text_to_latent(gen_text)
        pred = autoencoder.decoder(text_latent).float()
        raw_value_min = min(raw_value_min, float(pred.min().item()))
        raw_value_max = max(raw_value_max, float(pred.max().item()))
        raw_below_zero += float((pred < 0).sum().item())
        raw_above_one += float((pred > 1).sum().item())
        n_voxels += float(pred.numel())
        if include_raw:
            generated_by_variant["raw"].append(stage3_brain_encoder(pred).cpu())
        if include_clamped:
            generated_by_variant["clamped"].append(stage3_brain_encoder(pred.clamp(0.0, 1.0)).cpu())
        text_embeddings.append(stage3_text_projection(eval_text.float()).cpu())
        records.extend(records_from_batch(batch))
    text = torch.cat(text_embeddings, dim=0)
    out: dict[str, float] = {
        f"{prefix}_raw_pred_min": raw_value_min,
        f"{prefix}_raw_pred_max": raw_value_max,
        f"{prefix}_raw_pred_fraction_below_0": raw_below_zero / max(1.0, n_voxels),
        f"{prefix}_raw_pred_fraction_above_1": raw_above_one / max(1.0, n_voxels),
    }
    for variant, chunks in generated_by_variant.items():
        generated = torch.cat(chunks, dim=0)
        out.update(semantic_retrieval_metrics(text, generated, records, prefix=f"{prefix}_{variant}"))
    default = f"{prefix}_clamped_strict_map"
    if f"{default}_mean_normalized_auc" in out:
        out[f"{prefix}_text_to_brain_normalized_auc"] = out[f"{default}_text_to_brain_normalized_auc"]
        out[f"{prefix}_brain_to_text_normalized_auc"] = out[f"{default}_brain_to_text_normalized_auc"]
        out[f"{prefix}_mean_normalized_auc"] = out[f"{default}_mean_normalized_auc"]
        out[f"{prefix}_normalized_auc"] = out[f"{default}_mean_normalized_auc"]
        out[f"{prefix}_matched_contrastive_cosine"] = out[f"{prefix}_clamped_matched_contrastive_cosine"]
        out[f"{prefix}_shuffled_contrastive_cosine"] = out[f"{prefix}_clamped_shuffled_contrastive_cosine"]
        for k in (1, 5, 10, 50):
            key = f"{default}_recall@{k}"
            if key in out:
                out[f"{prefix}_recall@{k}"] = out[key]
    return out


def primary_text_volume_collate(batch: list[dict[str, Any]], target_shape: tuple[int, int, int], *, text_rank: int = 0) -> dict[str, Any]:
    volumes = []
    texts = []
    text_entries = []
    kept = []
    for item in batch:
        positives = item.get("positive_texts", []) or []
        if not positives:
            continue
        pos = positives[min(int(text_rank), len(positives) - 1)]
        v = item["volume"].float()
        if tuple(v.shape[-3:]) != target_shape:
            v = F.interpolate(v.unsqueeze(0), size=target_shape, mode="trilinear", align_corners=False).squeeze(0)
        volumes.append(v.clamp(0.0, 1.0))
        texts.append(str(pos["text"]))
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
    for key, value in payload.items():
        if torch.is_tensor(value) or isinstance(value, (list, tuple)):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim == 1:
                out[str(key)] = tensor
    return out


def _autoencoder_arch(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    target_shape = payload.get("target_shape") or cfg.get("target_shape") or [36, 45, 38]
    return {
        "latent_dim": int(model_cfg.get("latent_dim", payload.get("latent_dim", 384))),
        "base_channels": int(model_cfg.get("base_channels", 64)),
        "num_blocks": int(model_cfg.get("num_blocks", 4)),
        "encoder_arch": str(model_cfg.get("encoder_arch", "plain")),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "norm": str(model_cfg.get("norm", "group")),
        "pooling": str(model_cfg.get("pooling", "max")),
        "blocks_per_stage": int(model_cfg.get("blocks_per_stage", 2)),
        "use_dilation": bool(model_cfg.get("use_dilation", False)),
        "multi_scale": bool(model_cfg.get("multi_scale", False)),
        "global_context": str(model_cfg.get("global_context", "none")),
        "target_shape": tuple(int(v) for v in target_shape),
    }


def load_autoencoder_for_eval(checkpoint: str | Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = _autoencoder_arch(payload)
    model = build_cnn_autoencoder(
        arch["target_shape"],
        latent_dim=arch["latent_dim"],
        base_channels=arch["base_channels"],
        num_blocks=arch["num_blocks"],
        dropout=arch["dropout"],
        norm=arch["norm"],
        pooling=arch["pooling"],
        encoder_arch=arch["encoder_arch"],
        blocks_per_stage=arch["blocks_per_stage"],
        use_dilation=arch["use_dilation"],
        multi_scale=arch["multi_scale"],
        global_context=arch["global_context"],
    ).to(device)
    load_autoencoder_checkpoint(model, checkpoint, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, arch


def _stage3_encoder_arch(model_name: str | None) -> str:
    if model_name in {None, "", "ale_3dcnn"}:
        return "plain"
    if model_name == "ale_3dcnn_resnet":
        return "resnet"
    raise ValueError(f"Unknown stage3 encoder model_name: {model_name!r}")


def load_stage3_evaluator(checkpoint: str | Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    brain = build_brain_encoder(
        out_dim=int(cfg.get("out_dim", 384)),
        encoder_arch=_stage3_encoder_arch(cfg.get("model", "ale_3dcnn")),
        base_channels=int(cfg.get("base_channels", 64)),
        num_blocks=int(cfg.get("num_blocks", 4)),
        blocks_per_stage=int(cfg.get("blocks_per_stage", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_dilation=bool(cfg.get("use_dilation", False)),
        multi_scale=bool(cfg.get("multi_scale", False)),
        global_context=str(cfg.get("global_context", "none")),
    ).to(device)
    text = build_text_projection("random", device=device)
    brain.load_state_dict(payload["brain_encoder"], strict=True)
    text.load_state_dict(payload.get("text_proj") or payload["text_projection"], strict=True)
    for model in [brain, text]:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return brain, text


def _state_dict_from_stage4_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["generative_text_to_ae_latent", "text_projector", "text_projection", "model", "state_dict"]:
        state = payload.get(key)
        if isinstance(state, dict) and state and all(torch.is_tensor(v) for v in state.values()):
            return state
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise KeyError("Stage 4 checkpoint does not contain a recognized projector state dict")


def load_stage4_projector(checkpoint: str | Path, device: torch.device, *, latent_dim: int = 384):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    state = _state_dict_from_stage4_payload(payload)
    projection_cfg = cfg.get("generative_text_to_ae_latent", cfg.get("text_to_brain_projection", {})) if isinstance(cfg, dict) else {}
    hidden_dim = int(projection_cfg.get("hidden_dim", cfg.get("hidden_dim", 512) if isinstance(cfg, dict) else 512))
    if any(key.startswith("net.") for key in state):
        model = build_generative_text_to_ae_latent(device=device, in_dim=768, hidden_dim=hidden_dim, latent_dim=latent_dim)
    else:
        model = build_text_to_brain_projection(
            "random",
            device=device,
            in_dim=int(projection_cfg.get("in_dim", 768)),
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            depth=int(projection_cfg.get("depth", cfg.get("depth", 2) if isinstance(cfg, dict) else 2)),
            dropout=float(projection_cfg.get("dropout", cfg.get("dropout", 0.1) if isinstance(cfg, dict) else 0.1)),
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


@torch.no_grad()
def evaluate_stage4_checkpoint(
    *,
    stage4_checkpoint: str | Path,
    autoencoder_checkpoint: str | Path,
    stage3_evaluator_checkpoint: str | Path,
    test_jsonl: str | Path,
    generation_text_cache_path: str | Path,
    evaluator_text_cache_path: str | Path | None = None,
    domain: str | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    include_spatial: bool = True,
) -> dict[str, float]:
    device = torch.device(device)
    autoencoder, ae_arch = load_autoencoder_for_eval(autoencoder_checkpoint, device)
    projector, _ = load_stage4_projector(stage4_checkpoint, device, latent_dim=ae_arch["latent_dim"])
    stage3_brain, stage3_text = load_stage3_evaluator(stage3_evaluator_checkpoint, device)
    generation_cache = _load_text_cache(generation_text_cache_path)
    evaluator_cache = _load_text_cache(evaluator_text_cache_path) if evaluator_text_cache_path else generation_cache
    dataset = UnifiedMapTextDataset(test_jsonl)
    if domain:
        dataset.rows = [row for row in dataset.rows if canonical_source(row) == str(domain)]
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=lambda batch: primary_text_volume_collate(batch, ae_arch["target_shape"]),
    )
    out = evaluate_generation_semantic_loader(
        autoencoder,
        projector,
        stage3_brain,
        stage3_text,
        loader,
        generation_cache,
        device,
        evaluator_text_cache=evaluator_cache,
    )
    if include_spatial:
        spatial_rows = []
        for batch in loader:
            raw = stack_text_cache(generation_cache, batch["texts"]).to(device)
            target = batch["volume"].to(device)
            pred = autoencoder.decoder(projector(raw)).detach().cpu()
            spatial_rows.append(generation_metrics(pred, target.detach().cpu(), include_voxel_auroc=False))
        if spatial_rows:
            for key in spatial_rows[0]:
                out[key] = float(sum(row[key] for row in spatial_rows) / len(spatial_rows))
    return out


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_cross_eval(config: dict[str, Any]) -> list[dict[str, Any]]:
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for stage4 in config.get("stage4_models", []):
        for evaluator in config.get("stage3_evaluators", []):
            metrics = evaluate_stage4_checkpoint(
                stage4_checkpoint=stage4["checkpoint"],
                autoencoder_checkpoint=stage4["autoencoder_checkpoint"],
                stage3_evaluator_checkpoint=evaluator["checkpoint"],
                test_jsonl=stage4.get("test_jsonl", config["test_jsonl"]),
                generation_text_cache_path=stage4["generation_text_cache"],
                evaluator_text_cache_path=evaluator.get("text_cache") or stage4.get("evaluator_text_cache"),
                domain=stage4.get("domain", config.get("domain")),
                batch_size=int(config.get("batch_size", 64)),
                device=device,
                include_spatial=bool(config.get("include_spatial", True)),
            )
            rows.append(
                {
                    "domain": stage4.get("domain", config.get("domain", "")),
                    "branch": stage4.get("branch", stage4.get("branch_kind", "")),
                    "generation_model_type": stage4.get("generation_model_type", stage4["name"]),
                    "stage3_evaluator_type": evaluator.get("stage3_evaluator_type", evaluator["name"]),
                    "text_preprocessing": evaluator.get("text_preprocessing", stage4.get("text_preprocessing", "")),
                    **stage4_metric_aliases(metrics),
                    "n_eval": metrics.get("semantic_eval_n", ""),
                    "checkpoint_paths": json.dumps(
                        {
                            "stage4_checkpoint": stage4["checkpoint"],
                            "stage3_evaluator_checkpoint": evaluator["checkpoint"],
                            "autoencoder_checkpoint": stage4["autoencoder_checkpoint"],
                        },
                        sort_keys=True,
                    ),
                    "stage4_model": stage4["name"],
                    "stage3_evaluator": evaluator["name"],
                    "stage4_checkpoint": stage4["checkpoint"],
                    "stage3_evaluator_checkpoint": evaluator["checkpoint"],
                    **metrics,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 4 generated maps with raw/clamped and duplicate-aware semantic AUC.")
    parser.add_argument("--cross-eval-config", help="JSON config with stage4_models and stage3_evaluators for 2x2 cross-eval.")
    parser.add_argument("--stage4-checkpoint")
    parser.add_argument("--autoencoder-checkpoint")
    parser.add_argument("--stage3-evaluator-checkpoint")
    parser.add_argument("--test-jsonl")
    parser.add_argument("--generation-text-cache")
    parser.add_argument("--evaluator-text-cache")
    parser.add_argument("--domain")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cross_eval_config:
        with Path(args.cross_eval_config).open() as f:
            rows = run_cross_eval(json.load(f))
    else:
        required = [
            args.stage4_checkpoint,
            args.autoencoder_checkpoint,
            args.stage3_evaluator_checkpoint,
            args.test_jsonl,
            args.generation_text_cache,
        ]
        if any(not value for value in required):
            raise SystemExit("Single evaluation requires --stage4-checkpoint, --autoencoder-checkpoint, --stage3-evaluator-checkpoint, --test-jsonl, and --generation-text-cache")
        rows = [
            evaluate_stage4_checkpoint(
                stage4_checkpoint=args.stage4_checkpoint,
                autoencoder_checkpoint=args.autoencoder_checkpoint,
                stage3_evaluator_checkpoint=args.stage3_evaluator_checkpoint,
                test_jsonl=args.test_jsonl,
                generation_text_cache_path=args.generation_text_cache,
                evaluator_text_cache_path=args.evaluator_text_cache,
                domain=args.domain,
                batch_size=args.batch_size,
                device=args.device,
            )
        ]
    out = Path(args.output)
    if out.suffix.lower() == ".csv":
        _write_table(out, rows)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} Stage 4 semantic eval row(s) to {out}")


if __name__ == "__main__":
    main()
