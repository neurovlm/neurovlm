"""Shared semantic evaluation hooks for NeuroVLM experiment notebooks/scripts."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nilearn.image import resample_img

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurovlm.semantic_evaluation import (
    DEFAULT_MESH_RANKING_NODE_TYPES,
    align_network_ground_truth,
    align_network_term_ground_truth,
    build_mesh_candidate_corpus,
    build_network_label_corpus,
    encode_texts_with_specter,
    evaluate_mesh_term_ranking,
    evaluate_network_labeling,
    evaluate_network_term_ranking,
    evaluate_semantic_neighbor_retrieval,
    exact_pmid_retrieval_metrics,
    exact_recall_curve,
    find_default_mesh_json,
    load_network_label_table,
    load_network_maps,
    load_network_term_corpus,
    load_pmid_mesh_map,
    mesh_node_type_lookup_from_dataframe,
    preprocess_network_maps,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _resource_candidates(resource_dir: str | Path | None, relative: str) -> list[Path]:
    candidates: list[Path] = []
    if resource_dir is not None:
        candidates.append(Path(resource_dir) / relative)
    candidates.extend(
        [
            Path.cwd() / relative,
            REPO_ROOT / relative,
            Path("/content/drive/MyDrive/neurovlm_evaluation_resources") / relative,
        ]
    )
    return candidates


def resolve_resource_file(
    resource_dir: str | Path | None,
    relative: str,
    fallback: str | Path | None = None,
) -> Path | None:
    if fallback is not None and Path(fallback).exists():
        return Path(fallback)
    for candidate in _resource_candidates(resource_dir, relative):
        if candidate.exists():
            return candidate
    return None


@torch.no_grad()
def project_raw_text_embeddings(
    raw_text_embeddings: torch.Tensor,
    text_projector: torch.nn.Module,
    *,
    device: torch.device | str,
    batch_size: int = 4096,
) -> torch.Tensor:
    device = torch.device(device)
    text_projector = text_projector.to(device).eval()
    chunks = []
    for start in range(0, len(raw_text_embeddings), batch_size):
        batch = raw_text_embeddings[start : start + batch_size].float().to(device)
        chunks.append(text_projector(batch).float().detach().cpu())
    return F.normalize(torch.cat(chunks, dim=0), dim=1, eps=1e-8)


def run_embedding_semantic_evaluations(
    *,
    model_name: str,
    brain_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    raw_text_embeddings: torch.Tensor,
    pmids: Sequence[str],
    text_projector: torch.nn.Module,
    out_dir: str | Path,
    device: torch.device | str,
    resource_dir: str | Path | None = None,
    mesh_json: str | Path | None = None,
    resource_use: dict[str, Any] | None = None,
    extra_summary: dict[str, Any] | None = None,
    run_mesh: bool = True,
    run_semantic_neighbors: bool = True,
    semantic_neighbors: int = 10,
) -> dict[str, Any]:
    """Run exact PMID, MeSH, semantic-neighbor, and summary-row evaluations."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pmids = np.asarray(pmids).astype(str)
    brain = F.normalize(brain_embeddings.float().cpu(), dim=1, eps=1e-8)
    text = F.normalize(text_embeddings.float().cpu(), dim=1, eps=1e-8)
    raw_text = torch.as_tensor(raw_text_embeddings).float().cpu()

    if len(brain) != len(text) or len(brain) != len(pmids):
        raise ValueError(
            f"Expected aligned brain/text/PMID lengths, got brain={len(brain)} text={len(text)} pmids={len(pmids)}"
        )

    sim = brain @ text.T
    exact_b2p = exact_pmid_retrieval_metrics(sim)
    exact_p2b = exact_pmid_retrieval_metrics(sim.T)
    exact_payload = {
        "n_test": int(len(pmids)),
        "brain_to_paper": exact_b2p,
        "paper_to_brain": exact_p2b,
        "interpretation": "exact_pmid_retrieval is a strict diagnostic, not the main success metric.",
    }
    with (out_path / "exact_pmid_retrieval_metrics.json").open("w") as f:
        json.dump(_jsonable(exact_payload), f, indent=2)
    pd.DataFrame(
        {
            "k": np.arange(1, len(pmids) + 1),
            "brain_to_paper_recall_curve": exact_recall_curve(sim).cpu().numpy(),
            "paper_to_brain_recall_curve": exact_recall_curve(sim.T).cpu().numpy(),
            "random_recall_curve": np.arange(1, len(pmids) + 1) / float(len(pmids)),
        }
    ).to_csv(out_path / "exact_pmid_retrieval_curves.csv", index=False)

    mesh_metrics: dict[str, Any] | None = None
    if run_mesh:
        mesh_path = resolve_resource_file(resource_dir, "mesh_kg/mesh_annotations.json", mesh_json)
        if mesh_path is None:
            mesh_path = find_default_mesh_json()
        if mesh_path is None:
            warnings.warn("No PMID->MeSH JSON found; writing skipped mesh metrics.")
            mesh_metrics = {"skipped": True, "reason": "missing PMID->MeSH JSON"}
            with (out_path / "mesh_term_ranking_metrics.json").open("w") as f:
                json.dump(mesh_metrics, f, indent=2)
        else:
            pmid_mesh = load_pmid_mesh_map(mesh_path)
            definition_lookup = {}
            node_type_lookup = {}
            try:
                from neurovlm.retrieval_resources import _load_kg_mesh_dataset
                from neurovlm.semantic_evaluation import definition_lookup_from_dataframe

                definition_lookup = definition_lookup_from_dataframe(_load_kg_mesh_dataset())
            except Exception as exc:
                warnings.warn(f"Could not load MeSH definitions; using term-only candidates where needed: {exc}")
            mesh_nodes_path = resolve_resource_file(resource_dir, "mesh_kg/mesh_kg_nodes.parquet")
            if mesh_nodes_path is not None:
                try:
                    node_type_lookup = mesh_node_type_lookup_from_dataframe(pd.read_parquet(mesh_nodes_path))
                except Exception as exc:
                    warnings.warn(f"Could not load MeSH node types from {mesh_nodes_path}; leaving MeSH candidates unfiltered: {exc}")
            else:
                warnings.warn("No mesh_kg_nodes.parquet found; leaving MeSH candidates unfiltered by node_type.")
            mesh_corpus = build_mesh_candidate_corpus(
                pmid_mesh,
                list(pmid_mesh.keys()),
                definition_lookup=definition_lookup,
                node_type_lookup=node_type_lookup or None,
                allowed_node_types=DEFAULT_MESH_RANKING_NODE_TYPES if node_type_lookup else None,
            )
            from neurovlm.models import Specter

            specter = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query", device=str(device))
            mesh_embeddings = encode_texts_with_specter(
                mesh_corpus["text"].tolist(),
                specter,
                text_projector,
                batch_size=64,
                device=device,
            )
            mesh_metrics, _ = evaluate_mesh_term_ranking(
                brain,
                pmids,
                mesh_embeddings,
                mesh_corpus,
                pmid_mesh,
                out_dir=out_path,
            )

    semantic_metrics: dict[str, Any] | None = None
    if run_semantic_neighbors:
        semantic_metrics, _ = evaluate_semantic_neighbor_retrieval(
            brain,
            text,
            pmids,
            neighbor_text_embeddings=raw_text,
            n_neighbors=semantic_neighbors,
            out_dir=out_path,
        )

    summary = {
        "model": model_name,
        "n_test_pmids": int(len(pmids)),
        "exact_pmid_paper_recall_curve_auc": exact_b2p["paper_recall_curve_auc"],
        "exact_pmid_recall@1": exact_b2p["recall@1"],
        "exact_pmid_recall@5": exact_b2p["recall@5"],
        "exact_pmid_recall@10": exact_b2p["recall@10"],
        "exact_pmid_recall@50": exact_b2p["recall@50"],
        "exact_pmid_mrr": exact_b2p["mrr"],
        "exact_pmid_median_rank": exact_b2p["median_rank"],
    }
    if mesh_metrics and not mesh_metrics.get("skipped"):
        summary.update(
            {
                "mesh_recall@5": mesh_metrics.get("recall@5"),
                "mesh_recall@10": mesh_metrics.get("recall@10"),
                "mesh_map": mesh_metrics.get("map"),
                "mesh_mrr": mesh_metrics.get("mrr"),
                "mesh_median_best_true_term_rank": mesh_metrics.get("median_best_true_term_rank"),
            }
        )
    if semantic_metrics:
        summary.update(semantic_metrics)
    if extra_summary:
        summary.update(extra_summary)

    pd.DataFrame([summary]).to_csv(out_path / "main_comparison_summary_row.csv", index=False)
    with (out_path / "main_comparison_summary_row.json").open("w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    manifest = {
        "model_name": model_name,
        "resource_dir": str(resource_dir) if resource_dir is not None else None,
        "resource_use": resource_use or {},
        "outputs": {
            "main_comparison_summary_row": str(out_path / "main_comparison_summary_row.csv"),
            "exact_pmid_retrieval_metrics": str(out_path / "exact_pmid_retrieval_metrics.json"),
            "mesh_term_ranking_metrics": str(out_path / "mesh_term_ranking_metrics.json"),
            "semantic_neighbor_retrieval_metrics": str(out_path / "semantic_neighbor_retrieval_metrics.json"),
            "network_labeling_metrics": str(out_path / "network_labeling_metrics.json"),
            "network_term_ranking_metrics": str(out_path / "network_term_ranking_metrics.json"),
        },
    }
    with (out_path / "semantic_evaluation_manifest.json").open("w") as f:
        json.dump(_jsonable(manifest), f, indent=2)
    return summary


def _network_label_inputs(resource_dir: str | Path | None):
    labels_csv = resolve_resource_file(resource_dir, "networks_labels/network_test_set_labels.csv")
    labels_df = load_network_label_table(labels_csv)
    label_corpus = build_network_label_corpus(labels_df)
    term_corpus_path = resolve_resource_file(resource_dir, "networks_labels/network_terms_with_definitions.csv")
    term_corpus = load_network_term_corpus(term_corpus_path) if term_corpus_path is not None else None
    return labels_df, label_corpus, term_corpus


def _save_network_skipped(out_dir: str | Path, reason: str) -> None:
    payload = {"skipped": True, "reason": reason}
    with (Path(out_dir) / "network_labeling_metrics.json").open("w") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame([payload]).to_csv(Path(out_dir) / "network_labeling_predictions.csv", index=False)
    _save_network_term_skipped(out_dir, reason)


def _save_network_term_skipped(out_dir: str | Path, reason: str) -> None:
    payload = {"skipped": True, "reason": reason}
    with (Path(out_dir) / "network_term_ranking_metrics.json").open("w") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame([payload]).to_csv(Path(out_dir) / "network_term_predictions.csv", index=False)


@torch.no_grad()
def run_network_labeling_from_embeddings(
    *,
    model_name: str,
    network_embeddings: torch.Tensor,
    network_records: Sequence[dict[str, Any]],
    text_projector: torch.nn.Module,
    out_dir: str | Path,
    device: torch.device | str,
    resource_dir: str | Path | None = None,
) -> dict[str, Any]:
    labels_df, label_corpus, term_corpus = _network_label_inputs(resource_dir)
    from neurovlm.models import Specter

    specter = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query", device=str(device))
    label_embeddings = encode_texts_with_specter(
        label_corpus["text"].tolist(),
        specter,
        text_projector,
        batch_size=64,
        device=device,
    )
    truth = align_network_ground_truth(network_records, labels_df)
    metrics, _ = evaluate_network_labeling(
        network_embeddings,
        label_embeddings,
        truth,
        label_corpus,
        out_dir=out_dir,
    )
    summary_updates = {
        "network_accuracy": metrics.get("accuracy"),
        "network_top_2_accuracy": metrics.get("top_2_accuracy"),
        "network_macro_auc": metrics.get("macro_auc"),
    }

    if term_corpus is not None and len(term_corpus):
        term_embeddings = encode_texts_with_specter(
            term_corpus["text"].tolist(),
            specter,
            text_projector,
            batch_size=64,
            device=device,
        )
        term_truth = align_network_term_ground_truth(network_records, labels_df, term_corpus)
        term_metrics, _ = evaluate_network_term_ranking(
            network_embeddings,
            term_embeddings,
            term_truth,
            term_corpus,
            out_dir=out_dir,
        )
        metrics.update({f"term_{key}": value for key, value in term_metrics.items()})
        summary_updates.update(
            {
                "network_term_recall@5": term_metrics.get("recall@5"),
                "network_term_recall@10": term_metrics.get("recall@10"),
                "network_term_recall@20": term_metrics.get("recall@20"),
                "network_term_map": term_metrics.get("map"),
                "network_term_mrr": term_metrics.get("mrr"),
                "network_term_ndcg@10": term_metrics.get("ndcg@10"),
                "network_term_median_best_true_term_rank": term_metrics.get("median_best_true_term_rank"),
                "network_term_n_candidate_terms": term_metrics.get("n_candidate_terms"),
            }
        )
    else:
        _save_network_term_skipped(
            out_dir,
            "network_terms_with_definitions.csv not found; run experiments/create_network_term_definition_corpus.ipynb.",
        )

    summary_path = Path(out_dir) / "main_comparison_summary_row.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary.update(summary_updates)
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2))
        pd.DataFrame([summary]).to_csv(Path(out_dir) / "main_comparison_summary_row.csv", index=False)
    metrics.update(summary_updates)
    return metrics


def ale_network_volume_records(preprocess_config, masker) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Convert raw network maps to ALE volume tensors for the current run."""

    from neurovlm.gnn.ale_dataset import _brain_crop, _get_mask_img_for_resolution, _mask_data_and_affine, _normalize_volume

    records = preprocess_network_maps(load_network_maps(), masker)
    vols = []
    if preprocess_config.mode == "difumo_compatible":
        mask, affine = _mask_data_and_affine(None)
        crop = _brain_crop(mask) if preprocess_config.crop_to_brain else (slice(None), slice(None), slice(None))
        crop_mask = mask[crop]
        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
        native_shape = tuple(int(s.stop - s.start) for s in crop)
        target_resolution = float(preprocess_config.resolution_mm)
        target_shape = native_shape
        needs_resample = target_resolution > 0 and not np.allclose(voxel_sizes, target_resolution, atol=0.05)
        if needs_resample:
            target_shape = tuple(max(1, int(round(native_shape[i] * float(voxel_sizes[i]) / target_resolution))) for i in range(3))
        for rec in records:
            flat = masker.transform(rec["image"]).astype(np.float32)
            vol = np.zeros(native_shape, dtype=np.float32)
            vol[crop_mask] = flat.reshape(-1)
            vol = _normalize_volume(vol, preprocess_config.normalize, preprocess_config.clamp)
            vol_t = torch.from_numpy(vol).view(1, 1, *native_shape)
            if needs_resample:
                vol_t = F.interpolate(vol_t, size=target_shape, mode="trilinear", align_corners=False)
            vols.append(vol_t.squeeze(0).squeeze(0))
    else:
        mask_img = _get_mask_img_for_resolution(float(preprocess_config.resolution_mm))
        mask = np.asarray(mask_img.get_fdata() > 0)
        crop = _brain_crop(mask) if preprocess_config.crop_to_brain else (slice(None), slice(None), slice(None))
        for rec in records:
            img = resample_img(
                rec["image"],
                target_affine=mask_img.affine,
                target_shape=mask.shape,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
            vol = np.asarray(img.get_fdata(), dtype=np.float32)
            if vol.ndim == 4 and vol.shape[-1] == 1:
                vol = vol[..., 0]
            if vol.shape != mask.shape:
                raise ValueError(
                    "Atlas-free network map shape does not match mask shape after resampling: "
                    f"map={vol.shape}, mask={mask.shape}"
                )
            vol *= mask.astype(np.float32)
            vol = _normalize_volume(vol, preprocess_config.normalize, preprocess_config.clamp)
            vols.append(torch.from_numpy(vol[crop]))
    return records, torch.stack(vols).float()


@torch.no_grad()
def run_ale_network_labeling(
    *,
    trainer,
    preprocess_config,
    masker,
    out_dir: str | Path,
    device: torch.device | str,
    resource_dir: str | Path | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    records, volumes = ale_network_volume_records(preprocess_config, masker)
    trainer.brain_encoder.eval()
    chunks = []
    device = torch.device(device)
    for start in range(0, len(volumes), batch_size):
        batch = volumes[start : start + batch_size].unsqueeze(1).to(device)
        chunks.append(trainer.brain_encoder(batch).float().detach().cpu())
    net_emb = F.normalize(torch.cat(chunks, dim=0), dim=1, eps=1e-8)
    return run_network_labeling_from_embeddings(
        model_name="ale",
        network_embeddings=net_emb,
        network_records=records,
        text_projector=trainer.text_proj,
        out_dir=out_dir,
        device=device,
        resource_dir=resource_dir,
    )


@torch.no_grad()
def run_difumo_network_labeling(
    *,
    trainer,
    args,
    data,
    edge_index,
    edge_attr,
    extra_node_feats,
    out_dir: str | Path,
    device: torch.device | str,
    resource_dir: str | Path | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    from neurovlm.data import load_dataset, load_masker
    from neurovlm.gnn.atlas import compute_difumo_coefficients, load_difumo_components

    masker = load_masker()
    records = preprocess_network_maps(load_network_maps(), masker)
    flat = np.vstack([masker.transform(rec["image"]).astype(np.float32) for rec in records])
    components = load_difumo_components(dimension=512)
    coeffs = flat @ components.T

    if getattr(args, "difumo_normalize_coeffs", True):
        # Match the training cache convention as closely as possible: stats are
        # computed over the aligned PubMed coefficient population.
        images_data = load_dataset("pubmed_images")
        brain_flat_tensor = images_data[0] if isinstance(images_data, (tuple, list)) else images_data
        if hasattr(brain_flat_tensor, "detach"):
            brain_flat_np = brain_flat_tensor.detach().cpu().numpy().astype(np.float32)
        else:
            brain_flat_np = np.asarray(brain_flat_tensor, dtype=np.float32)
        train_raw = brain_flat_np @ components.T
        mu = train_raw.mean(axis=0, keepdims=True)
        sigma = train_raw.std(axis=0, keepdims=True) + 1e-8
        coeffs = (coeffs - mu) / sigma
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32)

    trainer.brain_encoder.eval()
    device = torch.device(device)
    chunks = []
    if args.model == "mlp":
        for start in range(0, len(coeffs_t), batch_size):
            chunks.append(trainer.brain_encoder(coeffs_t[start : start + batch_size].to(device)).float().detach().cpu())
    else:
        from torch_geometric.data import Batch, Data

        for start in range(0, len(coeffs_t), batch_size):
            rows = coeffs_t[start : start + batch_size]
            samples = []
            for row in rows:
                x = row.unsqueeze(1)
                if extra_node_feats is not None:
                    x = torch.cat([x, extra_node_feats.cpu()], dim=1)
                d = Data(x=x, edge_index=edge_index.cpu())
                if edge_attr is not None:
                    d.edge_attr = edge_attr.cpu()
                samples.append(d)
            b = Batch.from_data_list(samples).to(device)
            chunks.append(trainer.brain_encoder(b.x, b.edge_index, getattr(b, "edge_attr", None), b.batch).float().detach().cpu())
    net_emb = F.normalize(torch.cat(chunks, dim=0), dim=1, eps=1e-8)
    return run_network_labeling_from_embeddings(
        model_name="difumo",
        network_embeddings=net_emb,
        network_records=records,
        text_projector=trainer.text_proj,
        out_dir=out_dir,
        device=device,
        resource_dir=resource_dir,
    )


def mark_network_labeling_skipped(out_dir: str | Path, reason: str) -> None:
    _save_network_skipped(out_dir, reason)
