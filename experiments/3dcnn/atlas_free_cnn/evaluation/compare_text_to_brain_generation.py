"""Compare MLP NeuroVLM and atlas-free CNN text-to-brain generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atlas_free_cnn import notebook_utils
from atlas_free_cnn.evaluation import model_comparison_adapters as adapters
from atlas_free_cnn.evaluation import model_comparison_registry as registry
from atlas_free_cnn.evaluation.generation_metrics import generation_metrics
from atlas_free_cnn.evaluation.stage1_checkpoint_evaluation import discover_split_dir
from atlas_free_cnn.evaluation.stage4_semantic import stack_text_cache
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset
from atlas_free_cnn.training.source_sampling import canonical_source, source_detail
from atlas_free_cnn.training.train_autoencoder import filter_data_mode
from neurovlm.text_to_brain_metrics import (
    add_random_correlation_baseline,
    build_surface_eligibility_mask,
    dice_percentile,
    evaluate_t2b_sample,
    generated_image_text_retrieval_curve,
    pearson_correlation,
    sensitivity_rows_for_df,
)

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - scipy is present in normal eval envs
    spearmanr = None


DATASETS = ("pubmed", "nilearn", "neurovault")
T2B_MODEL_IDS = (
    "mlp_neurovlm",
    "cnn_t2b_mixed",
    "cnn_t2b_mixed_to_pubmed",
    "cnn_t2b_mixed_to_nilearn",
    "cnn_t2b_mixed_to_neurovault",
    "cnn_t2b_mixed_pubmed",
    "cnn_t2b_mixed_nilearn",
    "cnn_t2b_mixed_neurovault",
    "cnn_t2b_pubmed",
    "cnn_t2b_nilearn",
    "cnn_t2b_neurovault",
)
DOMAIN_TO_DATA_MODE = {
    "pubmed": "pubmed_only",
    "nilearn": "nilearn_only",
    "neurovault": "neurovault_only",
}
TARGET_SHAPE = (36, 45, 38)
DEFAULT_OUTPUT_DIR = Path("experiments/3dcnn/atlas_free_cnn/outputs/model_comparison")
BY_SAMPLE_FILENAME = "text_to_brain_by_sample.csv"
SUMMARY_FILENAME = "text_to_brain_summary.csv"
RANDOM_BASELINE_FILENAME = "text_to_brain_random_baseline.csv"
DICE_SENSITIVITY_FILENAME = "text_to_brain_dice_sensitivity.csv"

SUMMARY_METRIC_KEYS = (
    "pearson_r",
    "pearson_minus_random",
    "pearson_random_percentile",
    "spearman_rho",
    "spearman_minus_random",
    "spearman_random_percentile",
    "dice_pct90",
    "generated_image_text_normalized_auc",
    "mse",
    "foreground_mse",
    "spatial_corr",
    "top1_dice",
    "top5_dice",
    "top10_dice",
    "voxel_auroc",
)
INTERNAL_ARRAY_COLUMNS = ("_brain_pred", "_brain_true", "_pred_lh", "_pred_rh", "_true_lh", "_true_rh")
_PROVENANCE_CACHE: dict[str, dict[str, Any]] = {}
RANDOM_BASELINE_COLUMNS = (
    "dataset",
    "model_family",
    "model_id",
    "sample_id",
    "comparison_space",
    "pearson_baseline_actual",
    "pearson_random_mean",
    "pearson_random_std",
    "pearson_minus_random",
    "pearson_random_percentile",
    "spearman_baseline_actual",
    "spearman_random_mean",
    "spearman_random_std",
    "spearman_minus_random",
    "spearman_random_percentile",
    "random_baseline_n",
    "random_baseline_n_voxels",
    "random_baseline_group_dataset",
    "random_baseline_group_model_family",
)
DICE_SENSITIVITY_COLUMNS = (
    "dataset",
    "model_id",
    "comparison_space",
    "sample_id",
    "status",
    "pct",
    "top_fraction",
    "dice",
    "spin_p_value",
    "spin_significant",
    "method",
    "skip_reason",
)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        if not keys:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_json_ready(value), f, indent=2, sort_keys=True)


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append({key: value for key, value in row.items() if key not in INTERNAL_ARRAY_COLUMNS})
    return out


def resolve_test_jsonl(test_jsonl: str | Path | None) -> Path:
    if test_jsonl is not None:
        return Path(test_jsonl).expanduser()
    return discover_split_dir() / "test.jsonl"


def resolve_text_cache_path(text_embedding_cache: str | Path | None) -> tuple[Path, dict[str, Any]]:
    spec = notebook_utils.resolve_text_embedding_cache()
    if text_embedding_cache is not None:
        spec["local_cache_path"] = str(Path(text_embedding_cache).expanduser())
        spec["local_cache_override_env"] = "cli"
    return Path(spec["local_cache_path"]).expanduser(), spec


def normalize_model_id_for_dataset(model_id: str, dataset_name: str) -> tuple[str, str | None]:
    """Return registry model id and CNN variant for a dataset-specific request."""

    if model_id == "mlp_neurovlm":
        return model_id, None
    if model_id == "cnn_t2b_mixed":
        variant = f"mixed_to_{dataset_name}"
        return f"cnn_t2b_{variant}", variant
    prefix = "cnn_t2b_mixed_"
    if model_id.startswith(prefix) and not model_id.startswith("cnn_t2b_mixed_to_"):
        domain = model_id.removeprefix(prefix)
        variant = f"mixed_to_{domain}"
        return f"cnn_t2b_{variant}", variant
    if model_id.startswith("cnn_t2b_"):
        variant = model_id.removeprefix("cnn_t2b_")
        return model_id, variant
    raise ValueError(f"Unknown text-to-brain model id {model_id!r}")


def provenance_for_model(model_id: str, dataset_name: str) -> dict[str, Any]:
    resolved_model_id, variant = normalize_model_id_for_dataset(model_id, dataset_name)
    spec = registry.MODEL_SPECS.get(resolved_model_id)
    out: dict[str, Any] = {
        "requested_model_id": model_id,
        "model_id": resolved_model_id,
        "model_family": spec.family if spec else "",
        "model_domain": spec.domain if spec else "",
        "model_branch": spec.branch if spec else "",
        "cnn_variant": variant or "",
    }
    if spec is None:
        out.update({"status": "unknown_model", "checkpoint": "", "checkpoint_error": resolved_model_id})
        return out
    if spec.family == "mlp":
        out.update({"status": "resolved", "checkpoint": "", "checkpoint_error": ""})
        return out
    if resolved_model_id not in _PROVENANCE_CACHE:
        _PROVENANCE_CACHE[resolved_model_id] = registry.resolve_model_registry((resolved_model_id,))[resolved_model_id]
    row = _PROVENANCE_CACHE[resolved_model_id]
    out.update(
        {
            "status": row.get("status", ""),
            "checkpoint": row.get("checkpoint_path") or "",
            "checkpoint_error": row.get("error") or "",
        }
    )
    return out


def skipped_row(*, dataset_name: str, model_id: str, provenance: dict[str, Any], status: str, reason: str, sample_id: str = "__dataset__") -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        **provenance,
        "model_id": provenance.get("model_id", model_id),
        "status": status,
        "skip_reason": reason,
        "supported": False,
        "sample_id": sample_id,
        "comparison_space": "",
    }


def _as_2d_float_tensor(value: Any) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def _text_from_parts(*parts: Any) -> str:
    return " ".join(str(part) for part in parts if str(part or "").strip()).strip()


def load_mlp_t2b_records(dataset_name: str, *, limit: int | None) -> tuple[list[dict[str, Any]], str]:
    from neurovlm.data import load_dataset, load_latent
    from neurovlm.evaluation_notebook_utils import build_neurovault_t2b_eval

    if dataset_name == "nilearn":
        raise NotImplementedError("main_neurovlm_flat_nilearn_resource_unavailable")
    if dataset_name == "pubmed":
        df_pubs = load_dataset("pubmed_text")
        if "test" in df_pubs.columns:
            df_pubs = df_pubs[df_pubs["test"].fillna(False).astype(bool)].reset_index(drop=True)
            split_strategy = "main_pubmed_official_test_split"
        else:
            split_strategy = "main_pubmed_all_rows"
        pmid_col = "pmid" if "pmid" in df_pubs.columns else df_pubs.columns[0]
        title_col = "name" if "name" in df_pubs.columns else "title"
        abstract_col = "description" if "description" in df_pubs.columns else "abstract"
        pubmed_latents, pubmed_pmids = load_latent("pubmed_images")
        pubmed_pmids = np.asarray(pubmed_pmids)
        pmid_to_row = df_pubs.set_index(pmid_col)
        records = []
        for latent_pos, pmid in enumerate(pubmed_pmids):
            if pmid not in pmid_to_row.index:
                continue
            row = pmid_to_row.loc[pmid]
            text = _text_from_parts(row.get(title_col, ""), row.get(abstract_col, ""))
            if not text:
                continue
            records.append(
                {
                    "sample_id": str(pmid),
                    "map_id": str(pmid),
                    "text_id": str(pmid),
                    "text": text,
                    "true_latent": pubmed_latents[latent_pos],
                    "candidate_latents": None,
                    "candidate_indices": None,
                    "source": "pubmed",
                    "source_detail": split_strategy,
                    "tensor_index": latent_pos,
                }
            )
            if limit is not None and len(records) >= int(limit):
                break
        return records, split_strategy
    if dataset_name == "neurovault":
        raw = build_neurovault_t2b_eval(max_samples=limit)
        records = []
        for i, item in enumerate(raw):
            text = _text_from_parts(item.get("short_gt", ""), item.get("long_gt", ""))
            records.append(
                {
                    "sample_id": str(item.get("doi", f"neurovault_{i}")),
                    "map_id": str(item.get("doi", f"neurovault_{i}")),
                    "text_id": str(item.get("doi", f"neurovault_{i}")),
                    "text": text,
                    "true_latent": item["latent"],
                    "candidate_latents": item.get("candidate_latents"),
                    "candidate_indices": item.get("candidate_image_indices"),
                    "source": "neurovault",
                    "source_detail": "main_neurovault_publication_group",
                    "tensor_index": i,
                }
            )
        return records, "main_neurovault_publication_group"
    raise ValueError(f"Unknown dataset {dataset_name!r}")


def build_atlas_free_dataset(
    dataset_name: str,
    *,
    test_jsonl: str | Path | None,
    text_cache: dict[str, torch.Tensor],
    limit: int | None,
) -> UnifiedMapTextDataset:
    if dataset_name not in DOMAIN_TO_DATA_MODE:
        raise ValueError(f"Unknown dataset {dataset_name!r}; expected one of {DATASETS}")
    dataset = UnifiedMapTextDataset(resolve_test_jsonl(test_jsonl))
    filter_data_mode(dataset, DOMAIN_TO_DATA_MODE[dataset_name])
    rows = []
    for row in dataset.rows:
        positives = row.get("positive_texts", []) or []
        text = str((positives[0] if positives else {}).get("text", ""))
        if text and text in text_cache:
            rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    dataset.rows = rows
    return dataset


def _collate_cnn_t2b(batch: list[dict[str, Any]], target_shape: tuple[int, int, int]) -> dict[str, Any]:
    volumes = []
    texts = []
    text_entries = []
    kept = []
    for item in batch:
        positives = item.get("positive_texts", []) or []
        if not positives:
            continue
        pos = positives[0]
        volume = item["volume"].float()
        if tuple(volume.shape[-3:]) != target_shape:
            volume = F.interpolate(volume.unsqueeze(0), size=target_shape, mode="trilinear", align_corners=False).squeeze(0)
        volumes.append(volume.clamp(0.0, 1.0))
        texts.append(str(pos["text"]))
        text_entries.append(pos)
        kept.append(item)
    if not volumes:
        raise ValueError("Batch contains no rows with positive_texts")
    return {
        "volume": torch.stack(volumes),
        "map_id": [item["map_id"] for item in kept],
        "texts": texts,
        "text_entries": text_entries,
        "metadata": [item["metadata"] for item in kept],
    }


def _flat_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().reshape(-1).numpy()


def normalize_dice_column(row: dict[str, Any], dice_pct: float) -> dict[str, Any]:
    stable = f"dice_pct{dice_pct:g}"
    raw = f"dice_pct{dice_pct}"
    if raw in row and stable not in row:
        row[stable] = row[raw]
    return row


def evaluate_cnn_generated_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_name: str,
    dataset_name: str,
    dice_pct: float,
) -> dict[str, Any]:
    pred_flat = _flat_np(pred)
    target_flat = _flat_np(target)
    spearman_rho = float("nan")
    if spearmanr is not None:
        spearman_rho = float(spearmanr(target_flat, pred_flat).statistic)
    return normalize_dice_column({
        "name": sample_name,
        "pearson_r": float(pearson_correlation(target_flat, pred_flat)),
        "spearman_rho": spearman_rho,
        f"dice_pct{dice_pct:g}": float(dice_percentile(pred_flat, target_flat, pct=dice_pct)),
        "dice_method": "native_volume_percentile",
        "spin_p_value": float("nan"),
        "spin_significant": False,
        "spin_method": "not_run_native_volume",
        "surface_metric_eligible": True,
        "surface_metric_skip_reason": "not_applicable_native_volume",
        "cortical_top_mass_fraction": float("nan"),
        "surface_eligibility_method": "not_applicable_native_volume",
        "neurovault_selection_mode": "not_applicable",
        "n_candidate_images": 1,
        "selected_candidate_position": 0,
        "selected_candidate_index": 0,
        "selected_candidate_pearson": float("nan"),
        "candidate_pearson_max": float("nan"),
        "candidate_pearson_mean": float("nan"),
        "_brain_pred": pred_flat,
        "_brain_true": target_flat,
        "_pred_lh": None,
        "_pred_rh": None,
        "_true_lh": None,
        "_true_rh": None,
    }, dice_pct)


def row_base(*, dataset_name: str, provenance: dict[str, Any], sample_id: str, comparison_space: str, device: str, limit: int | None) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        **provenance,
        "checkpoint": provenance.get("checkpoint", ""),
        "sample_id": sample_id,
        "comparison_space": comparison_space,
        "device": device,
        "limit": limit if limit is not None else "",
        "supported": True,
        "skip_reason": "",
    }


def evaluate_mlp_dataset(
    *,
    dataset_name: str,
    provenance: dict[str, Any],
    device: str,
    limit: int | None,
    dice_pct: float,
    skip_spin: bool,
    spin_test_n_perm: int,
    random_seed: int,
    include_voxel_auroc: bool,
) -> list[dict[str, Any]]:
    from neurovlm.core import NeuroVLM

    records, split_strategy = load_mlp_t2b_records(dataset_name, limit=limit)
    if not records:
        return [
            skipped_row(
                dataset_name=dataset_name,
                model_id="mlp_neurovlm",
                provenance=provenance,
                status="empty_dataset",
                reason="no_mlp_t2b_records_after_filtering",
            )
        ]
    nvlm = NeuroVLM(device=device)
    nvlm._ensure_masker()
    surface_mask, surface_method = build_surface_eligibility_mask(nvlm._masker)
    rows: list[dict[str, Any]] = []
    pred_flatmaps = []
    texts = []
    for i, record in enumerate(records):
        metric_row = evaluate_t2b_sample(
            nvlm=nvlm,
            masker=nvlm._masker,
            name=str(record["sample_id"]),
            text_input=str(record["text"]),
            true_latent=record["true_latent"],
            dataset_name=dataset_name,
            candidate_latents=record.get("candidate_latents"),
            candidate_indices=record.get("candidate_indices"),
            dice_pct=dice_pct,
            surface_mask=surface_mask,
            surface_eligibility_method=surface_method,
            spin_use_neuromaps=not skip_spin,
            spin_require_neuromaps=False,
            spin_test_n_perm=spin_test_n_perm,
            spin_test_random_state=random_seed,
        )
        if metric_row is None:
            rows.append(
                {
                    **row_base(
                        dataset_name=dataset_name,
                        provenance=provenance,
                        sample_id=str(record["sample_id"]),
                        comparison_space="mlp_masker_flatmap",
                        device=device,
                        limit=limit,
                    ),
                    "status": "error",
                    "supported": False,
                    "skip_reason": "evaluate_t2b_sample_failed",
                    "split_strategy": split_strategy,
                }
            )
            continue
        metric_row = normalize_dice_column(metric_row, dice_pct)
        brain_pred = torch.as_tensor(metric_row["_brain_pred"]).float()
        brain_true = torch.as_tensor(metric_row["_brain_true"]).float()
        pred_flatmaps.append(brain_pred)
        texts.append(str(record["text"]))
        # `generation_metrics` (spatial_corr/foreground_mse/topK dice) operates
        # on flattened batches (see `spatial_correlation_loss`/`hard_topk_dice`,
        # both `.flatten(1)` internally), so it applies directly to the MLP's
        # flat masker-space vectors -- no 3D reshape needed. This gives the MLP
        # rows the same native-style metrics already computed for CNN rows.
        native = generation_metrics(
            brain_pred.unsqueeze(0),
            brain_true.unsqueeze(0),
            include_voxel_auroc=include_voxel_auroc,
        )
        rows.append(
            {
                **row_base(
                    dataset_name=dataset_name,
                    provenance=provenance,
                    sample_id=str(record["sample_id"]),
                    comparison_space="mlp_masker_flatmap",
                    device=device,
                    limit=limit,
                ),
                "status": "ok",
                "sample_index": i,
                "map_id": record["map_id"],
                "text_id": record["text_id"],
                "source": record["source"],
                "source_detail": record["source_detail"],
                "tensor_index": record["tensor_index"],
                "split_strategy": split_strategy,
                **metric_row,
                **native,
            }
        )
    if len(pred_flatmaps) >= 2:
        try:
            auc, _, rank_df = generated_image_text_retrieval_curve(nvlm, torch.stack(pred_flatmaps), texts, seed=random_seed)
            ok_positions = [idx for idx, row in enumerate(rows) if row.get("status") == "ok"]
            for pos, rank_row in enumerate(rank_df.to_dict("records")):
                rows[ok_positions[pos]].update(
                    {
                        "generated_image_text_normalized_auc": float(auc),
                        "generated_image_text_rank": int(rank_row["rank"]),
                        "matched_contrastive_sim": float(rank_row["matched_contrastive_sim"]),
                        "null_contrastive_sim": float(rank_row["null_contrastive_sim"]),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - spatial metrics are still useful
            for row in rows:
                if row.get("status") == "ok":
                    row["generated_image_text_retrieval_error"] = str(exc)
    return rows


@torch.no_grad()
def evaluate_cnn_dataset(
    *,
    dataset_name: str,
    model_id: str,
    provenance: dict[str, Any],
    device: str,
    batch_size: int,
    limit: int | None,
    test_jsonl: str | Path | None,
    text_cache: dict[str, torch.Tensor],
    dice_pct: float,
    include_voxel_auroc: bool,
) -> list[dict[str, Any]]:
    _, variant = normalize_model_id_for_dataset(model_id, dataset_name)
    if variant is None:
        raise ValueError(f"{model_id!r} is not a CNN text-to-brain model id")
    adapter = adapters.CNNTextToBrainAdapter(variant, device=device)
    stage3 = adapters.CNNContrastiveAdapter(variant, device=device)
    dataset = build_atlas_free_dataset(dataset_name=dataset_name, test_jsonl=test_jsonl, text_cache=text_cache, limit=limit)
    if len(dataset) == 0:
        return [
            skipped_row(
                dataset_name=dataset_name,
                model_id=model_id,
                provenance=provenance,
                status="empty_dataset",
                reason="no_atlas_free_rows_with_cached_text_embeddings",
            )
        ]
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=lambda batch: _collate_cnn_t2b(batch, TARGET_SHAPE),
    )
    rows: list[dict[str, Any]] = []
    generated_shared = []
    text_shared = []
    sample_offset = 0
    for batch in loader:
        text_embeddings = stack_text_cache(text_cache, batch["texts"]).to(adapter.device)
        pred = adapter.generate(text_embeddings).detach().cpu()
        target = batch["volume"].detach().cpu()
        if pred.shape[-3:] != target.shape[-3:]:
            pred = F.interpolate(pred, size=target.shape[-3:], mode="trilinear", align_corners=False)
        generated_shared.append(stage3.encode_brain_to_shared(pred.to(stage3.device)).detach().cpu())
        text_shared.append(stage3.encode_text_to_shared(text_embeddings).detach().cpu())
        for i, map_id in enumerate(batch["map_id"]):
            metric_row = evaluate_cnn_generated_sample(
                pred[i : i + 1],
                target[i : i + 1],
                sample_name=str(map_id),
                dataset_name=dataset_name,
                dice_pct=dice_pct,
            )
            native = generation_metrics(pred[i : i + 1], target[i : i + 1], include_voxel_auroc=include_voxel_auroc)
            metadata = batch["metadata"][i]
            text_entry = batch["text_entries"][i] if i < len(batch["text_entries"]) else {}
            rows.append(
                {
                    **row_base(
                        dataset_name=dataset_name,
                        provenance=provenance,
                        sample_id=str(map_id),
                        comparison_space="native_atlas_free_volume",
                        device=device,
                        limit=limit,
                    ),
                    "status": "ok",
                    "sample_index": sample_offset + i,
                    "map_id": str(map_id),
                    "text_id": str(text_entry.get("text_id") or text_entry.get("id") or text_entry.get("text") or ""),
                    "source": canonical_source(metadata),
                    "source_detail": source_detail(metadata),
                    "tensor_index": metadata.get("tensor_index"),
                    "split_strategy": "atlas_free_unified_test_split",
                    **metric_row,
                    **native,
                }
            )
        sample_offset += len(batch["map_id"])
    if len(rows) >= 2:
        text_all = torch.cat(text_shared, dim=0)
        gen_all = torch.cat(generated_shared, dim=0)
        scores = gen_all @ text_all.T
        matched = scores.diag()
        ranks = 1 + (scores > matched[:, None]).sum(dim=1)
        hit_counts = torch.bincount(ranks - 1, minlength=len(rows)).float()
        recall_curve = torch.cumsum(hit_counts, dim=0) / float(len(rows))
        from neurovlm.retrieval_metrics import normalized_recall_curve_auc

        auc = normalized_recall_curve_auc(recall_curve)
        for row, rank, sim in zip(rows, ranks.tolist(), matched.tolist(), strict=True):
            row["generated_image_text_normalized_auc"] = float(auc)
            row["generated_image_text_rank"] = int(rank)
            row["matched_contrastive_sim"] = float(sim)
    return rows


def add_grouped_random_baselines(rows: list[dict[str, Any]], *, n_random: int, seed: int, max_voxels: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply main random-baseline helper per dataset and model family."""

    ok_rows = [row for row in rows if row.get("status") == "ok" and "_brain_pred" in row and "_brain_true" in row]
    if not ok_rows:
        return rows, []

    baseline_rows: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        groups[(str(row.get("dataset", "")), str(row.get("model_family", "")))].append(row)

    for group_seed_offset, ((dataset_name, model_family), group) in enumerate(sorted(groups.items())):
        df = pd.DataFrame(group)
        df = add_random_correlation_baseline(
            df,
            n_random=n_random,
            seed=seed + group_seed_offset,
            max_voxels=max_voxels,
        )
        for record in df.to_dict("records"):
            record["random_baseline_group_dataset"] = dataset_name
            record["random_baseline_group_model_family"] = model_family
            enriched.append(record)
            baseline_rows.append(
                {
                    key: record.get(key)
                    for key in RANDOM_BASELINE_COLUMNS
                }
            )

    by_key = {(row.get("dataset"), row.get("model_id"), row.get("sample_id")): row for row in enriched}
    merged = []
    for row in rows:
        merged.append(by_key.get((row.get("dataset"), row.get("model_id"), row.get("sample_id")), row))
    return merged, baseline_rows


def add_dice_sensitivity(rows: list[dict[str, Any]], *, pcts: Iterable[float], skip_spin: bool, spin_test_n_perm: int, random_seed: int, n_jobs: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and "_brain_pred" in row and "_brain_true" in row:
            groups[(str(row.get("dataset", "")), str(row.get("model_id", "")), str(row.get("comparison_space", "")))].append(row)
    for (dataset_name, model_id, comparison_space), group in sorted(groups.items()):
        df = pd.DataFrame(group)
        try:
            sens = sensitivity_rows_for_df(
                df,
                dataset_name,
                pcts=pcts,
                spin_test_n_perm=spin_test_n_perm,
                spin_test_random_state=random_seed,
                spin_fsaverage_density="41k",
                spin_use_neuromaps=not skip_spin,
                spin_require_neuromaps=False,
                n_jobs=n_jobs,
            )
        except Exception as exc:  # noqa: BLE001 - do not fail the whole comparison on sensitivity diagnostics
            out.append(
                {
                    "dataset": dataset_name,
                    "model_id": model_id,
                    "comparison_space": comparison_space,
                    "status": "error",
                    "skip_reason": str(exc),
                }
            )
            continue
        for row in sens:
            sample_id = str(row.pop("sample", ""))
            row.update(
                {
                    "dataset": dataset_name,
                    "model_id": model_id,
                    "comparison_space": comparison_space,
                    "sample_id": sample_id,
                    "status": "ok",
                }
            )
            out.append(row)
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset", "")), str(row.get("model_id", "")), str(row.get("comparison_space", "")))].append(row)

    summary = []
    for (dataset_name, model_id, comparison_space), group in sorted(groups.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        skipped = [row for row in group if row.get("status") != "ok"]
        first = group[0] if group else {}
        out: dict[str, Any] = {
            "dataset": dataset_name,
            "model_id": model_id,
            "model_family": first.get("model_family", ""),
            "model_domain": first.get("model_domain", ""),
            "model_branch": first.get("model_branch", ""),
            "checkpoint": first.get("checkpoint", ""),
            "comparison_space": comparison_space,
            "status": "ok" if ok else str(first.get("status", "")),
            "n_samples": len(group),
            "n_ok": len(ok),
            "n_skipped": len(skipped),
            "skip_reasons": "; ".join(sorted({str(row.get("skip_reason", "")) for row in skipped if row.get("skip_reason")})),
        }
        for key in SUMMARY_METRIC_KEYS:
            values = [float(row[key]) for row in ok if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
            if values:
                mean = sum(values) / len(values)
                var = sum((value - mean) ** 2 for value in values) / len(values)
                out[f"{key}_mean"] = mean
                out[f"{key}_std"] = math.sqrt(var)
        summary.append(out)
    return summary


def evaluate_model_dataset(
    *,
    dataset_name: str,
    model_id: str,
    device: str,
    batch_size: int,
    limit: int | None,
    test_jsonl: str | Path | None,
    text_cache: dict[str, torch.Tensor] | None,
    text_cache_spec: dict[str, Any],
    dice_pct: float,
    skip_spin: bool,
    spin_test_n_perm: int,
    random_seed: int,
    include_voxel_auroc: bool,
) -> list[dict[str, Any]]:
    provenance = provenance_for_model(model_id, dataset_name)
    if provenance["status"] == "missing_checkpoint":
        return [
            skipped_row(
                dataset_name=dataset_name,
                model_id=model_id,
                provenance=provenance,
                status="missing_checkpoint",
                reason=provenance.get("checkpoint_error", ""),
            )
        ]
    if provenance["status"] not in {"resolved", ""}:
        return [
            skipped_row(
                dataset_name=dataset_name,
                model_id=model_id,
                provenance=provenance,
                status=str(provenance["status"]),
                reason=provenance.get("checkpoint_error", ""),
            )
        ]
    try:
        if model_id == "mlp_neurovlm":
            return evaluate_mlp_dataset(
                dataset_name=dataset_name,
                provenance=provenance,
                device=device,
                limit=limit,
                dice_pct=dice_pct,
                skip_spin=skip_spin,
                spin_test_n_perm=spin_test_n_perm,
                random_seed=random_seed,
                include_voxel_auroc=include_voxel_auroc,
            )
        if text_cache is None:
            try:
                # Local cache first (fast path once populated); falls back to
                # downloading from Hugging Face (neurovlm/atlas_free_cnn_dataset)
                # if the local repo cache dir hasn't been populated yet, mirroring
                # how the unified split JSONLs are resolved.
                text_cache = notebook_utils.load_or_download_text_embedding_cache(text_cache_spec)
            except Exception as exc:
                return [
                    skipped_row(
                        dataset_name=dataset_name,
                        model_id=model_id,
                        provenance=provenance,
                        status="missing_text_embedding_cache",
                        reason=str(exc),
                    )
                ]
        return evaluate_cnn_dataset(
            dataset_name=dataset_name,
            model_id=model_id,
            provenance=provenance,
            device=device,
            batch_size=batch_size,
            limit=limit,
            test_jsonl=test_jsonl,
            text_cache=text_cache,
            dice_pct=dice_pct,
            include_voxel_auroc=include_voxel_auroc,
        )
    except NotImplementedError as exc:
        return [skipped_row(dataset_name=dataset_name, model_id=model_id, provenance=provenance, status="unsupported_dataset", reason=str(exc))]
    except FileNotFoundError as exc:
        status = "missing_resource" if model_id == "mlp_neurovlm" else "missing_checkpoint"
        return [skipped_row(dataset_name=dataset_name, model_id=model_id, provenance=provenance, status=status, reason=str(exc))]
    except Exception as exc:  # noqa: BLE001 - comparison should continue across model/dataset failures
        return [skipped_row(dataset_name=dataset_name, model_id=model_id, provenance=provenance, status="error", reason=str(exc))]


def run_comparison(
    *,
    datasets: Iterable[str],
    models: Iterable[str],
    limit: int | None,
    device: str,
    output_dir: str | Path,
    test_jsonl: str | Path | None,
    text_embedding_cache: str | Path | None,
    batch_size: int,
    dice_pct: float,
    dice_sensitivity_pcts: Iterable[float],
    skip_spin: bool,
    spin_test_n_perm: int,
    random_baseline_n: int,
    random_baseline_max_voxels: int,
    random_seed: int,
    include_voxel_auroc: bool,
    sensitivity_jobs: int,
) -> dict[str, Any]:
    datasets = tuple(datasets)
    models = tuple(models)
    text_cache_path, text_cache_spec = resolve_text_cache_path(text_embedding_cache)
    text_cache_spec["local_cache_path"] = str(text_cache_path)

    by_sample_rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for model_id in models:
            rows = evaluate_model_dataset(
                dataset_name=dataset_name,
                model_id=model_id,
                device=device,
                batch_size=batch_size,
                limit=limit,
                test_jsonl=test_jsonl,
                text_cache=None,
                text_cache_spec=text_cache_spec,
                dice_pct=dice_pct,
                skip_spin=skip_spin,
                spin_test_n_perm=spin_test_n_perm,
                random_seed=random_seed,
                include_voxel_auroc=include_voxel_auroc,
            )
            by_sample_rows.extend(rows)

    by_sample_rows, random_rows = add_grouped_random_baselines(
        by_sample_rows,
        n_random=random_baseline_n,
        seed=random_seed,
        max_voxels=random_baseline_max_voxels,
    )
    dice_rows = add_dice_sensitivity(
        by_sample_rows,
        pcts=dice_sensitivity_pcts,
        skip_spin=skip_spin,
        spin_test_n_perm=spin_test_n_perm,
        random_seed=random_seed,
        n_jobs=sensitivity_jobs,
    )
    summary_rows = summarize_rows(by_sample_rows)

    output_dir = Path(output_dir)
    by_sample_path = output_dir / BY_SAMPLE_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    random_path = output_dir / RANDOM_BASELINE_FILENAME
    dice_path = output_dir / DICE_SENSITIVITY_FILENAME
    manifest_path = output_dir / "text_to_brain_manifest.json"
    write_csv(by_sample_path, public_rows(by_sample_rows))
    write_csv(summary_path, summary_rows)
    write_csv(random_path, random_rows, fieldnames=RANDOM_BASELINE_COLUMNS)
    write_csv(dice_path, dice_rows, fieldnames=DICE_SENSITIVITY_COLUMNS)
    write_json(
        manifest_path,
        {
            "datasets": list(datasets),
            "models": list(models),
            "limit": limit,
            "device": device,
            "test_jsonl": str(resolve_test_jsonl(test_jsonl)) if test_jsonl is not None else "",
            "text_embedding_cache": str(text_cache_path),
            "dice_pct": dice_pct,
            "dice_sensitivity_pcts": list(dice_sensitivity_pcts),
            "skip_spin": skip_spin,
            "by_sample_csv": by_sample_path,
            "summary_csv": summary_path,
            "random_baseline_csv": random_path,
            "dice_sensitivity_csv": dice_path,
        },
    )
    return {
        "by_sample_path": by_sample_path,
        "summary_path": summary_path,
        "random_baseline_path": random_path,
        "dice_sensitivity_path": dice_path,
        "manifest_path": manifest_path,
        "by_sample": by_sample_rows,
        "summary": summary_rows,
        "random_baseline": random_rows,
        "dice_sensitivity": dice_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=T2B_MODEL_IDS, default=["mlp_neurovlm", "cnn_t2b_mixed", "cnn_t2b_pubmed", "cnn_t2b_nilearn", "cnn_t2b_neurovault"])
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset/source after deterministic filtering.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-jsonl", default=None, help="Path to unified atlas-free test.jsonl. Defaults to local/HF split discovery.")
    parser.add_argument("--text-embedding-cache", default=None, help="Normalized SPECTER2 cache. Defaults to notebook_utils resolver/env overrides.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dice-pct", type=float, default=90.0)
    parser.add_argument("--dice-sensitivity-pcts", nargs="+", type=float, default=[80.0, 85.0, 90.0, 95.0])
    parser.add_argument("--skip-spin", action="store_true", default=False, help="Disable neuromaps spin tests.")
    parser.add_argument("--spin-test-n-perm", type=int, default=1000)
    parser.add_argument("--random-baseline-n", type=int, default=25)
    parser.add_argument("--random-baseline-max-voxels", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--skip-voxel-auroc", action="store_true", help="Skip voxel AUROC for faster smoke runs.")
    parser.add_argument("--sensitivity-jobs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_comparison(
            datasets=args.datasets,
            models=args.models,
            limit=args.limit,
            device=args.device,
            output_dir=args.output_dir,
            test_jsonl=args.test_jsonl,
            text_embedding_cache=args.text_embedding_cache,
            batch_size=args.batch_size,
            dice_pct=args.dice_pct,
            dice_sensitivity_pcts=args.dice_sensitivity_pcts,
            skip_spin=args.skip_spin,
            spin_test_n_perm=args.spin_test_n_perm,
            random_baseline_n=args.random_baseline_n,
            random_baseline_max_voxels=args.random_baseline_max_voxels,
            random_seed=args.random_seed,
            include_voxel_auroc=not args.skip_voxel_auroc,
            sensitivity_jobs=args.sensitivity_jobs,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should report cache/HF issues clearly
        print(f"Text-to-brain generation comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote by-sample CSV to {result['by_sample_path']}")
    print(f"Wrote summary CSV to {result['summary_path']}")
    print(f"Wrote random baseline CSV to {result['random_baseline_path']}")
    print(f"Wrote Dice sensitivity CSV to {result['dice_sensitivity_path']}")
    print(f"Wrote manifest JSON to {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
