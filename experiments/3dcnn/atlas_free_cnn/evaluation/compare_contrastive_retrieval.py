"""Compare MLP and atlas-free CNN contrastive retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atlas_free_cnn import notebook_utils
from atlas_free_cnn.evaluation import model_comparison_adapters as adapters
from atlas_free_cnn.evaluation import model_comparison_registry as registry
from atlas_free_cnn.evaluation.artifacts import resolve_test_jsonl, write_csv, write_json
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset
from atlas_free_cnn.training.source_sampling import canonical_source, source_detail
from atlas_free_cnn.training.train_ale_cnn import UnifiedContrastiveDataset
from atlas_free_cnn.training.train_autoencoder import VolumeCollator, filter_data_mode
from neurovlm.retrieval_metrics import (
    bidirectional_retrieval_metrics,
    normalized_k_values,
    recall_curve,
    retrieval_ranks,
)


DATASETS = ("pubmed", "nilearn", "neurovault")
CONTRASTIVE_MODEL_IDS = (
    "mlp_neurovlm",
    "cnn_contrastive_mixed",
    "cnn_contrastive_mixed_to_pubmed",
    "cnn_contrastive_mixed_to_nilearn",
    "cnn_contrastive_mixed_to_neurovault",
    "cnn_contrastive_mixed_pubmed",
    "cnn_contrastive_mixed_nilearn",
    "cnn_contrastive_mixed_neurovault",
    "cnn_contrastive_pubmed",
    "cnn_contrastive_nilearn",
    "cnn_contrastive_neurovault",
)
DOMAIN_TO_DATA_MODE = {
    "pubmed": "pubmed_only",
    "nilearn": "nilearn_only",
    "neurovault": "neurovault_only",
}
TARGET_SHAPE = (36, 45, 38)
DEFAULT_OUTPUT_DIR = Path("experiments/3dcnn/atlas_free_cnn/outputs/model_comparison")
SUMMARY_FILENAME = "contrastive_retrieval_summary.csv"
CURVES_FILENAME = "contrastive_retrieval_curves.csv"
EXAMPLES_FILENAME = "contrastive_retrieval_examples.csv"


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
    if model_id == "cnn_contrastive_mixed":
        variant = f"mixed_to_{dataset_name}"
        return f"cnn_contrastive_{variant}", variant
    prefix = "cnn_contrastive_mixed_"
    if model_id.startswith(prefix) and not model_id.startswith("cnn_contrastive_mixed_to_"):
        domain = model_id.removeprefix(prefix)
        variant = f"mixed_to_{domain}"
        return f"cnn_contrastive_{variant}", variant
    if model_id.startswith("cnn_contrastive_"):
        variant = model_id.removeprefix("cnn_contrastive_")
        return model_id, variant
    raise ValueError(f"Unknown contrastive model id {model_id!r}")


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
        out.update({"status": "unknown_model", "checkpoint_path": "", "checkpoint_error": resolved_model_id})
        return out
    if spec.family == "mlp":
        out.update({"status": "resolved", "checkpoint_path": "", "checkpoint_error": ""})
        return out
    manifest = registry.resolve_model_registry((resolved_model_id,))
    row = manifest[resolved_model_id]
    out.update(
        {
            "status": row.get("status", ""),
            "checkpoint_path": row.get("checkpoint_path") or "",
            "checkpoint_error": row.get("error") or "",
        }
    )
    return out


def build_atlas_free_dataset(
    dataset_name: str,
    *,
    test_jsonl: str | Path | None,
    text_cache: dict[str, torch.Tensor],
    limit: int | None,
    target_shape: tuple[int, int, int] = TARGET_SHAPE,
) -> UnifiedContrastiveDataset:
    if dataset_name not in DOMAIN_TO_DATA_MODE:
        raise ValueError(f"Unknown dataset {dataset_name!r}; expected one of {DATASETS}")
    jsonl_path = resolve_test_jsonl(test_jsonl)
    base = UnifiedMapTextDataset(jsonl_path, load_volumes=False)
    filter_data_mode(base, DOMAIN_TO_DATA_MODE[dataset_name])
    rows = []
    for row in base.rows:
        positives = row.get("positive_texts", []) or []
        if positives and str(positives[0].get("text", "")) in text_cache:
            rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    return UnifiedContrastiveDataset(jsonl_path, rows, text_cache, target_shape=target_shape)


@torch.no_grad()
def collect_cnn_embeddings(
    *,
    adapter: adapters.CNNContrastiveAdapter,
    dataset: UnifiedContrastiveDataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_text: list[torch.Tensor] = []
    all_brain: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for batch in loader:
        text = adapter.encode_text_to_shared(batch["text"]).detach().cpu()
        brain = adapter.encode_brain_to_shared(batch["volume"]).detach().cpu()
        all_text.append(text)
        all_brain.append(brain)
        for i, map_id in enumerate(batch["map_id"]):
            idx = int(batch["paper_idx"][i].item())
            row = dataset.rows[idx]
            positives = row.get("positive_texts", []) or [{}]
            records.append(
                {
                    "sample_index": len(records),
                    "map_id": str(map_id),
                    "text_id": str(batch["text_id"][i]),
                    "text": str(positives[0].get("text", "")),
                    "source": canonical_source(row),
                    "source_detail": source_detail(row),
                    "tensor_index": row.get("tensor_index"),
                }
            )
    if not all_text:
        raise ValueError("No paired text/brain rows were available after filtering.")
    return torch.cat(all_text, dim=0), torch.cat(all_brain, dim=0), records


def _as_2d_float_tensor(value: Any) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def _align_by_id(
    left: torch.Tensor,
    left_ids: Iterable[Any],
    right: torch.Tensor,
    right_ids: Iterable[Any],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    right_pos = {str(value): idx for idx, value in enumerate(right_ids)}
    left_indices: list[int] = []
    right_indices: list[int] = []
    ids: list[str] = []
    for idx, value in enumerate(left_ids):
        key = str(value)
        if key in right_pos:
            left_indices.append(idx)
            right_indices.append(right_pos[key])
            ids.append(key)
    if not ids:
        raise RuntimeError("No overlapping IDs between brain and text resources.")
    return left[left_indices], right[right_indices], ids


def _pubmed_test_positions(pmids: list[str]) -> torch.Tensor | None:
    try:
        from neurovlm.data import load_dataset
        from neurovlm.semantic_evaluation import official_split_positions

        pubmed_df = load_dataset("pubmed_text")
        positions = official_split_positions(pubmed_df, pmids)["test"]
        if len(positions) == 0:
            return None
        return torch.as_tensor(positions, dtype=torch.long)
    except Exception:
        return None


def load_mlp_nilearn_pairs(*, test_jsonl: str | Path | None, limit: int | None) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], str]:
    """Build MLP-space (text, brain) pairs for Nilearn.

    No Nilearn resource exists anywhere in the main NeuroVLM package (flat or
    otherwise). Nilearn atlas maps are only available via the atlas-free CNN's
    packed test volumes, which crop the exact same MNI152 4mm grid the MLP
    masker uses -- see `mlp_masker_bridge` -- so the brain side converts
    directly with no resampling, then is binarized like NeuroVault (Nilearn
    atlas maps are continuous/probabilistic, not the MLP's binary PubMed
    training distribution). Text is embedded on the fly with the same
    SPECTER2 model+adapter used to build the MLP's own text latents
    (`neurovlm.models.Specter`, adapter="adhoc_query", see
    `docs/03_evaluation/12_neurovault_decoding.ipynb`) rather than the CNN's
    differently-preprocessed (empty-centered, unit-normed) SPECTER2 cache,
    since the MLP text projection head expects the former convention.
    """
    from atlas_free_cnn.evaluation.mlp_masker_bridge import atlas_free_volume_to_mlp_flat
    from neurovlm.models import Specter

    jsonl_path = resolve_test_jsonl(test_jsonl)
    dataset = UnifiedMapTextDataset(jsonl_path)
    filter_data_mode(dataset, "nilearn_only")
    dataset.rows = [
        row for row in dataset.rows if str((row.get("positive_texts") or [{}])[0].get("text", "")).strip()
    ]
    if limit is not None:
        dataset.rows = dataset.rows[: int(limit)]
    if not dataset.rows:
        raise ValueError("No Nilearn rows with positive_texts were available after filtering.")

    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=VolumeCollator(TARGET_SHAPE))
    volume_chunks: list[torch.Tensor] = []
    texts: list[str] = []
    ids: list[str] = []
    for batch in loader:
        volume_chunks.append(batch["volume"])
        for i, map_id in enumerate(batch["map_id"]):
            positives = batch["metadata"][i].get("positive_texts") or [{}]
            texts.append(str(positives[0].get("text", "")))
            ids.append(str(map_id))

    flat = atlas_free_volume_to_mlp_flat(torch.cat(volume_chunks, dim=0), binarize=True)
    specter = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query")
    raw_text = _as_2d_float_tensor(specter(texts))
    split_strategy = "atlas_free_unified_test_split_via_mlp_masker_crop"
    records = [
        {
            "sample_index": i,
            "map_id": ids[i],
            "text_id": ids[i],
            "text": texts[i],
            "source": "nilearn",
            "source_detail": split_strategy,
            "tensor_index": i,
        }
        for i in range(len(ids))
    ]
    return raw_text, flat, records, split_strategy


def load_mlp_raw_pairs(dataset_name: str, *, limit: int | None, test_jsonl: str | Path | None = None) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], str]:
    from neurovlm.data import load_dataset, load_latent

    if dataset_name == "nilearn":
        return load_mlp_nilearn_pairs(test_jsonl=test_jsonl, limit=limit)
    if dataset_name == "pubmed":
        images, image_pmids = load_dataset("pubmed_images")
        text, text_pmids = load_latent("pubmed_text")
        flat, raw_text, ids = _align_by_id(
            _as_2d_float_tensor(images),
            image_pmids,
            _as_2d_float_tensor(text),
            text_pmids,
        )
        split_strategy = "main_pubmed_official_test_split"
        positions = _pubmed_test_positions(ids)
        if positions is not None:
            flat = flat.index_select(0, positions)
            raw_text = raw_text.index_select(0, positions)
            ids = [ids[int(i)] for i in positions.tolist()]
        else:
            split_strategy = "main_pubmed_all_aligned_pairs"
    elif dataset_name == "neurovault":
        # The MLP autoencoder/encoder were trained on binary PubMed activation
        # masks. NeuroVault maps are continuous statistic images, so they must
        # be binarized before encoding -- the same conversion used for MLP
        # contrastive evaluation on NeuroVault in
        # docs/03_evaluation/12_neurovault_decoding.ipynb (`neuro_clust > 0`)
        # and docs/03_evaluation/11_autoencoder.ipynb (`(X_nv > 0).float()`).
        flat = (_as_2d_float_tensor(load_dataset("neurovault_images")) > 0).float()
        raw_text = _as_2d_float_tensor(load_latent("neurovault_text"))
        n = min(int(flat.shape[0]), int(raw_text.shape[0]))
        flat = flat[:n]
        raw_text = raw_text[:n]
        ids = [f"neurovault_{i}" for i in range(n)]
        split_strategy = "main_neurovault_all_aligned_pairs_binarized"
    else:
        raise ValueError(f"Unknown dataset {dataset_name!r}")

    if limit is not None:
        flat = flat[: int(limit)]
        raw_text = raw_text[: int(limit)]
        ids = ids[: int(limit)]
    records = [
        {
            "sample_index": i,
            "map_id": ids[i],
            "text_id": ids[i],
            "text": "",
            "source": dataset_name,
            "source_detail": split_strategy,
            "tensor_index": i,
        }
        for i in range(len(ids))
    ]
    return raw_text, flat, records, split_strategy


@torch.no_grad()
def collect_mlp_embeddings(
    *,
    adapter: adapters.MLPContrastiveAdapter,
    dataset_name: str,
    limit: int | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], str]:
    raw_text, flat, records, split_strategy = load_mlp_raw_pairs(dataset_name, limit=limit)
    all_text: list[torch.Tensor] = []
    all_brain: list[torch.Tensor] = []
    for start in range(0, int(flat.shape[0]), batch_size):
        all_text.append(adapter.encode_text_to_shared(raw_text[start : start + batch_size]).detach().cpu())
        all_brain.append(adapter.encode_brain_to_shared(flat[start : start + batch_size]).detach().cpu())
    if not all_text:
        raise ValueError("No paired MLP rows were available after filtering.")
    return torch.cat(all_text, dim=0), torch.cat(all_brain, dim=0), records, split_strategy


def _auc_aliases(metrics: dict[str, float]) -> dict[str, float]:
    t2i = float(metrics["t2i_normalized_k_recall_curve_auc"])
    i2t = float(metrics["i2t_normalized_k_recall_curve_auc"])
    return {
        "normalized_k_recall_curve_auc": (t2i + i2t) / 2.0,
        "t2i_normalized_k_recall_curve_auc": t2i,
        "i2t_normalized_k_recall_curve_auc": i2t,
        "mrr": float(metrics["mean_mrr"]),
        "t2i_mrr": float(metrics["t2i_mrr"]),
        "i2t_mrr": float(metrics["i2t_mrr"]),
        "median_rank": float(metrics["mean_median_rank"]),
        "t2i_median_rank": float(metrics["t2i_median_rank"]),
        "i2t_median_rank": float(metrics["i2t_median_rank"]),
    }


def retrieval_summary_row(
    text_embeddings: torch.Tensor,
    brain_embeddings: torch.Tensor,
    *,
    base: dict[str, Any],
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, Any]:
    metrics = bidirectional_retrieval_metrics(text_embeddings, brain_embeddings, ks=ks)
    return {
        **base,
        "status": "ok",
        "n_pairs": int(text_embeddings.shape[0]),
        **_auc_aliases(metrics),
        **metrics,
    }


def recall_curve_rows(
    text_embeddings: torch.Tensor,
    brain_embeddings: torch.Tensor,
    *,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    t2i, i2t = recall_curve(text_embeddings, brain_embeddings)
    n = int(t2i.numel())
    normalized_k = normalized_k_values(n)
    rows = []
    for i in range(n):
        rows.append(
            {
                **base,
                "status": "ok",
                "n_pairs": n,
                "k": i + 1,
                "normalized_k": float(normalized_k[i].item()),
                "t2i_recall": float(t2i[i].item()),
                "i2t_recall": float(i2t[i].item()),
                "mean_recall": float(((t2i[i] + i2t[i]) / 2).item()),
                "random_recall": float(normalized_k[i].item()),
            }
        )
    return rows


def retrieval_example_rows(
    text_embeddings: torch.Tensor,
    brain_embeddings: torch.Tensor,
    records: list[dict[str, Any]],
    *,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    text_n = F.normalize(text_embeddings.float(), dim=1, eps=1e-8)
    brain_n = F.normalize(brain_embeddings.float(), dim=1, eps=1e-8)
    sim = text_n @ brain_n.T
    t2i_ranks = retrieval_ranks(sim)
    i2t_ranks = retrieval_ranks(sim.T)
    t2i_top1 = sim.argmax(dim=1)
    i2t_top1 = sim.T.argmax(dim=1)
    rows = []
    for i, record in enumerate(records):
        rows.append(
            {
                **base,
                "status": "ok",
                "sample_index": record.get("sample_index", i),
                "map_id": record.get("map_id", ""),
                "text_id": record.get("text_id", ""),
                "source": record.get("source", ""),
                "source_detail": record.get("source_detail", ""),
                "tensor_index": record.get("tensor_index", ""),
                "t2i_rank": int(t2i_ranks[i].item()),
                "i2t_rank": int(i2t_ranks[i].item()),
                "t2i_top1_map_id": records[int(t2i_top1[i].item())].get("map_id", ""),
                "i2t_top1_text_id": records[int(i2t_top1[i].item())].get("text_id", ""),
                "matched_similarity": float(sim[i, i].item()),
                "t2i_top1_similarity": float(sim[i].max().item()),
                "i2t_top1_similarity": float(sim[:, i].max().item()),
            }
        )
    return rows


def skipped_summary_row(*, dataset_name: str, model_id: str, provenance: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        **provenance,
        "status": status,
        "skip_reason": reason,
        "n_pairs": 0,
    }


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    provenance = provenance_for_model(model_id, dataset_name)
    base = {
        "dataset": dataset_name,
        **provenance,
        "device": device,
        "limit": limit if limit is not None else "",
        "text_embedding_convention": text_cache_spec.get("convention", ""),
        "text_embedding_preprocessing": text_cache_spec.get("preprocessing", ""),
        "text_embedding_cache_path": text_cache_spec.get("local_cache_path", ""),
    }
    if provenance["status"] == "missing_checkpoint":
        return [skipped_summary_row(dataset_name=dataset_name, model_id=model_id, provenance=base, status="missing_checkpoint", reason=provenance.get("checkpoint_error", ""))], [], []
    if provenance["status"] not in {"resolved", ""}:
        return [skipped_summary_row(dataset_name=dataset_name, model_id=model_id, provenance=base, status=str(provenance["status"]), reason=provenance.get("checkpoint_error", ""))], [], []

    try:
        if model_id == "mlp_neurovlm":
            raw_text, flat, records, split_strategy = load_mlp_raw_pairs(dataset_name, limit=limit, test_jsonl=test_jsonl)
            adapter = adapters.MLPContrastiveAdapter(device=device)
            all_text: list[torch.Tensor] = []
            all_brain: list[torch.Tensor] = []
            for start in range(0, int(flat.shape[0]), batch_size):
                all_text.append(adapter.encode_text_to_shared(raw_text[start : start + batch_size]).detach().cpu())
                all_brain.append(adapter.encode_brain_to_shared(flat[start : start + batch_size]).detach().cpu())
            if not all_text:
                raise ValueError("No paired MLP rows were available after filtering.")
            text_embeddings = torch.cat(all_text, dim=0)
            brain_embeddings = torch.cat(all_brain, dim=0)
            base["split_strategy"] = split_strategy
            base["comparison_space"] = "mlp_masker_flatmap"
        else:
            if text_cache is None:
                try:
                    # Local cache first (fast path once populated); falls back to
                    # downloading from Hugging Face (neurovlm/atlas_free_cnn_dataset)
                    # if the local repo cache dir hasn't been populated yet, mirroring
                    # how the unified split JSONLs are resolved.
                    text_cache = notebook_utils.load_or_download_text_embedding_cache(text_cache_spec)
                except Exception as exc:
                    return [
                        skipped_summary_row(
                            dataset_name=dataset_name,
                            model_id=model_id,
                            provenance=base,
                            status="missing_text_embedding_cache",
                            reason=str(exc),
                        )
                    ], [], []
            _, variant = normalize_model_id_for_dataset(model_id, dataset_name)
            if variant is None:
                raise ValueError(f"{model_id!r} is not a CNN contrastive model id")
            adapter = adapters.CNNContrastiveAdapter(variant, device=device)
            dataset = build_atlas_free_dataset(
                dataset_name=dataset_name,
                test_jsonl=test_jsonl,
                text_cache=text_cache,
                limit=limit,
            )
            text_embeddings, brain_embeddings, records = collect_cnn_embeddings(
                adapter=adapter,
                dataset=dataset,
                batch_size=batch_size,
            )
            base["split_strategy"] = "atlas_free_unified_test_split"
            base["comparison_space"] = "native_atlas_free_volume"
    except NotImplementedError as exc:
        return [skipped_summary_row(dataset_name=dataset_name, model_id=model_id, provenance=base, status="unsupported_dataset", reason=str(exc))], [], []
    except FileNotFoundError as exc:
        status = "missing_checkpoint" if model_id != "mlp_neurovlm" else "missing_resource"
        return [skipped_summary_row(dataset_name=dataset_name, model_id=model_id, provenance=base, status=status, reason=str(exc))], [], []
    except Exception as exc:  # noqa: BLE001 - comparison should continue across model/dataset failures
        return [skipped_summary_row(dataset_name=dataset_name, model_id=model_id, provenance=base, status="error", reason=str(exc))], [], []

    summary = [retrieval_summary_row(text_embeddings, brain_embeddings, base=base)]
    curves = recall_curve_rows(text_embeddings, brain_embeddings, base=base)
    examples = retrieval_example_rows(text_embeddings, brain_embeddings, records, base=base)
    return summary, curves, examples


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
) -> dict[str, Any]:
    datasets = tuple(datasets)
    models = tuple(models)
    text_cache_path, text_cache_spec = resolve_text_cache_path(text_embedding_cache)
    text_cache_spec["local_cache_path"] = str(text_cache_path)

    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for model_id in models:
            summary, curves, examples = evaluate_model_dataset(
                dataset_name=dataset_name,
                model_id=model_id,
                device=device,
                batch_size=batch_size,
                limit=limit,
                test_jsonl=test_jsonl,
                text_cache=None,
                text_cache_spec=text_cache_spec,
            )
            summary_rows.extend(summary)
            curve_rows.extend(curves)
            example_rows.extend(examples)

    output_dir = Path(output_dir)
    summary_path = output_dir / SUMMARY_FILENAME
    curves_path = output_dir / CURVES_FILENAME
    examples_path = output_dir / EXAMPLES_FILENAME
    manifest_path = output_dir / "contrastive_retrieval_manifest.json"
    write_csv(summary_path, summary_rows)
    write_csv(curves_path, curve_rows)
    write_csv(examples_path, example_rows)
    write_json(
        manifest_path,
        {
            "datasets": list(datasets),
            "models": list(models),
            "limit": limit,
            "device": device,
            "test_jsonl": str(resolve_test_jsonl(test_jsonl)) if test_jsonl is not None else "",
            "text_embedding_cache": str(text_cache_path),
            "summary_csv": summary_path,
            "curves_csv": curves_path,
            "examples_csv": examples_path,
        },
    )
    return {
        "summary_path": summary_path,
        "curves_path": curves_path,
        "examples_path": examples_path,
        "manifest_path": manifest_path,
        "summary": summary_rows,
        "curves": curve_rows,
        "examples": example_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=CONTRASTIVE_MODEL_IDS, default=list(CONTRASTIVE_MODEL_IDS[:2] + CONTRASTIVE_MODEL_IDS[-3:]))
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset/source after deterministic filtering.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-jsonl", default=None, help="Path to unified atlas-free test.jsonl. Defaults to local/HF split discovery.")
    parser.add_argument("--text-embedding-cache", default=None, help="Normalized SPECTER2 cache. Defaults to notebook_utils resolver/env overrides.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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
        )
    except Exception as exc:  # noqa: BLE001 - CLI should report cache/HF issues clearly
        print(f"Contrastive retrieval comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote summary CSV to {result['summary_path']}")
    print(f"Wrote recall curves to {result['curves_path']}")
    print(f"Wrote examples CSV to {result['examples_path']}")
    print(f"Wrote manifest JSON to {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
