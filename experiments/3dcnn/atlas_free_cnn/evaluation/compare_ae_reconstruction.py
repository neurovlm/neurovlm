"""Compare MLP and atlas-free CNN autoencoder reconstruction quality."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from atlas_free_cnn.evaluation import model_comparison_adapters as adapters
from atlas_free_cnn.evaluation.generation_metrics import generation_metrics
from atlas_free_cnn.evaluation.stage1_checkpoint_evaluation import discover_split_dir
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset
from atlas_free_cnn.training.source_sampling import canonical_source, source_detail
from atlas_free_cnn.training.train_autoencoder import VolumeCollator, filter_data_mode
from neurovlm.metrics import compute_ae_performance


DATASETS = ("pubmed", "nilearn", "neurovault")
AE_MODEL_IDS = ("mlp_neurovlm", "cnn_ae_mixed", "cnn_ae_pubmed", "cnn_ae_nilearn", "cnn_ae_neurovault")
CNN_AE_MODEL_TO_DOMAIN = {
    "cnn_ae_mixed": "mixed",
    "cnn_ae_pubmed": "pubmed",
    "cnn_ae_nilearn": "nilearn",
    "cnn_ae_neurovault": "neurovault",
}
DOMAIN_TO_DATA_MODE = {
    "pubmed": "pubmed_only",
    "nilearn": "nilearn_only",
    "neurovault": "neurovault_only",
}
TARGET_SHAPE = (36, 45, 38)
DEFAULT_OUTPUT_DIR = Path("experiments/3dcnn/atlas_free_cnn/outputs/model_comparison")
METRIC_KEYS = (
    "mse",
    "reconstruction_mse",
    "mae",
    "foreground_mse",
    "spatial_corr",
    "top1_dice",
    "top5_dice",
    "top10_dice",
    "top1_overlap",
    "top5_overlap",
    "top10_overlap",
    "target_nonzero_fraction",
    "pred_nonzero_fraction",
    "pred_mean",
    "pred_max",
    "voxel_auroc",
    "mlp_bpp_pct_improvement",
    "mlp_batch_roc_auc",
)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_ready(value), f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
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


def resolve_test_jsonl(test_jsonl: str | Path | None) -> Path:
    if test_jsonl is not None:
        return Path(test_jsonl).expanduser()
    return discover_split_dir() / "test.jsonl"


def build_atlas_free_test_datasets(
    datasets: Iterable[str],
    *,
    test_jsonl: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, UnifiedMapTextDataset]:
    jsonl_path = resolve_test_jsonl(test_jsonl)
    out: dict[str, UnifiedMapTextDataset] = {}
    for dataset_name in datasets:
        if dataset_name not in DOMAIN_TO_DATA_MODE:
            raise ValueError(f"Unknown dataset {dataset_name!r}; expected one of {DATASETS}")
        ds = UnifiedMapTextDataset(jsonl_path)
        filter_data_mode(ds, DOMAIN_TO_DATA_MODE[dataset_name])
        if limit is not None:
            ds.rows = ds.rows[: int(limit)]
        out[dataset_name] = ds
    return out


def _row_base(
    *,
    dataset_name: str,
    model_id: str,
    model_family: str,
    comparison_space: str,
    supported: bool,
    unsupported_reason: str = "",
    row: dict[str, Any] | None = None,
    map_id: str | None = None,
    sample_index: int | None = None,
) -> dict[str, Any]:
    row = row or {}
    return {
        "dataset": dataset_name,
        "model_id": model_id,
        "model_family": model_family,
        "comparison_space": comparison_space,
        "supported": bool(supported),
        "unsupported_reason": unsupported_reason,
        "sample_index": sample_index,
        "map_id": map_id or row.get("map_id"),
        "source": canonical_source(row) if row else dataset_name,
        "source_detail": source_detail(row) if row else dataset_name,
        "tensor_index": row.get("tensor_index"),
    }


@torch.no_grad()
def evaluate_mlp_on_atlas_free_dataset(
    *,
    dataset_name: str,
    dataset: UnifiedMapTextDataset,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    target_shape: tuple[int, int, int] = TARGET_SHAPE,
    include_voxel_auroc: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate the MLP autoencoder on atlas-free CNN volumes (Nilearn/NeuroVault).

    The atlas-free CNN's (36, 45, 38) volumes are crops of the exact MNI152
    4mm grid the MLP masker uses (see `mlp_masker_bridge`), so they can be
    converted to/from the MLP's 28542-dim masker-flat space with a boolean
    index -- no resampling. This is what makes an MLP-vs-CNN reconstruction
    comparison possible on Nilearn and NeuroVault at all, since neither has a
    dedicated flat resource in the main NeuroVLM package the way PubMed does.
    Targets are binarized before encoding, matching the established
    convention for feeding continuous statistic maps into the MLP's
    binary-trained encoder (docs/03_evaluation/11_autoencoder.ipynb,
    12_neurovault_decoding.ipynb).
    """
    from atlas_free_cnn.evaluation.mlp_masker_bridge import (
        atlas_free_volume_to_mlp_flat,
        mlp_flat_to_atlas_free_volume,
    )

    adapter = adapters.MLPAutoencoderAdapter(device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=VolumeCollator(target_shape))
    rows: list[dict[str, Any]] = []
    sample_offset = 0
    for batch in loader:
        flat_target = atlas_free_volume_to_mlp_flat(batch["volume"], binarize=True).to(adapter.device)
        flat_pred = adapter.decode(adapter.encode(flat_target)).detach().cpu()
        target_volume = mlp_flat_to_atlas_free_volume(flat_target.detach().cpu())
        pred_volume = mlp_flat_to_atlas_free_volume(flat_pred)
        for i, map_id in enumerate(batch["map_id"]):
            metrics = generation_metrics(
                pred_volume[i : i + 1],
                target_volume[i : i + 1],
                include_voxel_auroc=include_voxel_auroc,
            )
            rows.append(
                {
                    **_row_base(
                        dataset_name=dataset_name,
                        model_id="mlp_neurovlm",
                        model_family="mlp",
                        comparison_space="atlas_free_volume_via_mlp_masker_crop",
                        supported=True,
                        row=batch["metadata"][i],
                        map_id=str(map_id),
                        sample_index=sample_offset + i,
                    ),
                    **metrics,
                }
            )
        sample_offset += len(batch["map_id"])
    return rows


def _match_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape == target.shape:
        return pred
    if pred.ndim == target.ndim and pred.ndim >= 4 and pred.shape[-3:] != target.shape[-3:]:
        return torch.nn.functional.interpolate(pred, size=target.shape[-3:], mode="trilinear", align_corners=False)
    return pred.reshape_as(target)


@torch.no_grad()
def evaluate_cnn_ae_on_dataset(
    *,
    model_id: str,
    dataset_name: str,
    dataset: UnifiedMapTextDataset,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    target_shape: tuple[int, int, int] = TARGET_SHAPE,
    include_voxel_auroc: bool = True,
) -> list[dict[str, Any]]:
    if model_id not in CNN_AE_MODEL_TO_DOMAIN:
        raise ValueError(f"{model_id!r} is not a CNN AE model id")
    domain = CNN_AE_MODEL_TO_DOMAIN[model_id]
    adapter = adapters.CNNAutoencoderAdapter(domain, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=VolumeCollator(target_shape))
    rows: list[dict[str, Any]] = []
    sample_offset = 0
    for batch in loader:
        target = batch["volume"].to(adapter.device).float()
        recon = adapter.decode(adapter.encode(target))
        recon = _match_shape(recon, target).detach().cpu()
        target_cpu = target.detach().cpu()
        for i, map_id in enumerate(batch["map_id"]):
            metrics = generation_metrics(
                recon[i : i + 1],
                target_cpu[i : i + 1],
                include_voxel_auroc=include_voxel_auroc,
            )
            rows.append(
                {
                    **_row_base(
                        dataset_name=dataset_name,
                        model_id=model_id,
                        model_family="cnn_ae",
                        comparison_space="native_atlas_free_volume",
                        supported=True,
                        row=batch["metadata"][i],
                        map_id=str(map_id),
                        sample_index=sample_offset + i,
                    ),
                    "model_domain": domain,
                    **metrics,
                }
            )
        sample_offset += len(batch["map_id"])
    return rows


def _as_2d_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def _load_flat_mlp_dataset(dataset_name: str) -> tuple[torch.Tensor, list[str]]:
    from neurovlm.data import load_dataset

    if dataset_name == "pubmed":
        images, pmids = load_dataset("pubmed_images")
        return _as_2d_float_tensor(images), [str(v) for v in pmids]
    raise ValueError(f"No binary-compatible main NeuroVLM flat-image resource is available for {dataset_name!r}.")


@torch.no_grad()
def evaluate_mlp_flat_dataset(
    *,
    dataset_name: str,
    device: str | torch.device = "cpu",
    limit: int | None = None,
    batch_size: int = 64,
    include_voxel_auroc: bool = True,
) -> list[dict[str, Any]]:
    if dataset_name != "pubmed":
        # PubMed is the only dataset with a dedicated main-package flat
        # resource (`load_dataset("pubmed_images")`). Nilearn/NeuroVault are
        # evaluated separately via `evaluate_mlp_on_atlas_free_dataset`,
        # which converts the shared atlas-free CNN test volumes into MLP
        # masker-flat space instead (comparison_space
        # "atlas_free_volume_via_mlp_masker_crop").
        return [
            _row_base(
                dataset_name=dataset_name,
                model_id="mlp_neurovlm",
                model_family="mlp",
                comparison_space="mlp_masker_flatmap",
                supported=False,
                unsupported_reason="handled_via_atlas_free_volume_via_mlp_masker_crop_instead",
                map_id="__dataset__",
            )
        ]
    try:
        flatmaps, ids = _load_flat_mlp_dataset(dataset_name)
        if limit is not None:
            flatmaps = flatmaps[: int(limit)]
            ids = ids[: int(limit)]
        adapter = adapters.MLPAutoencoderAdapter(device=device)
    except Exception as exc:  # noqa: BLE001 - callers need a row explaining missing local/HF assets
        return [
            _row_base(
                dataset_name=dataset_name,
                model_id="mlp_neurovlm",
                model_family="mlp",
                comparison_space="mlp_masker_flatmap",
                supported=False,
                unsupported_reason=f"mlp_flat_resource_unavailable: {exc}",
                map_id="__dataset__",
            )
        ]

    rows: list[dict[str, Any]] = []
    for start in range(0, int(flatmaps.shape[0]), batch_size):
        target = flatmaps[start : start + batch_size].to(adapter.device)
        logits = adapter.autoencoder.decoder(adapter.autoencoder.encoder(target))
        pred = torch.sigmoid(logits).detach().cpu()
        target_cpu = target.detach().cpu()
        try:
            _, _, bpp_pct, roc_auc = compute_ae_performance(target_cpu, logits.detach().cpu())
        except Exception:
            bpp_pct = [float("nan")] * int(target_cpu.shape[0])
            roc_auc = float("nan")
        for i in range(int(target_cpu.shape[0])):
            metrics = generation_metrics(
                pred[i : i + 1],
                target_cpu[i : i + 1],
                include_voxel_auroc=include_voxel_auroc,
            )
            rows.append(
                {
                    **_row_base(
                        dataset_name=dataset_name,
                        model_id="mlp_neurovlm",
                        model_family="mlp",
                        comparison_space="mlp_masker_flatmap",
                        supported=True,
                        map_id=ids[start + i],
                        sample_index=start + i,
                    ),
                    **metrics,
                    "mlp_bpp_pct_improvement": float(bpp_pct[i]),
                    "mlp_batch_roc_auc": float(roc_auc),
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["model_id"]), str(row["comparison_space"]))].append(row)

    summary = []
    for (dataset_name, model_id, comparison_space), group in sorted(groups.items()):
        supported = [row for row in group if bool(row.get("supported"))]
        unsupported = [row for row in group if not bool(row.get("supported"))]
        out: dict[str, Any] = {
            "dataset": dataset_name,
            "model_id": model_id,
            "comparison_space": comparison_space,
            "n_samples": len(group),
            "n_supported": len(supported),
            "n_unsupported": len(unsupported),
            "unsupported_reasons": "; ".join(sorted({str(row.get("unsupported_reason", "")) for row in unsupported if row.get("unsupported_reason")})),
        }
        for key in METRIC_KEYS:
            values = []
            for row in supported:
                value = row.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if values:
                mean = sum(values) / len(values)
                var = sum((value - mean) ** 2 for value in values) / len(values)
                out[f"{key}_mean"] = mean
                out[f"{key}_std"] = math.sqrt(var)
        summary.append(out)
    return summary


def run_comparison(
    *,
    datasets: Iterable[str],
    models: Iterable[str],
    limit: int | None,
    device: str,
    output_dir: str | Path,
    test_jsonl: str | Path | None,
    batch_size: int,
    include_voxel_auroc: bool,
    include_mlp_flat: bool,
) -> dict[str, Path | list[dict[str, Any]]]:
    datasets = tuple(datasets)
    models = tuple(models)
    atlas_free = build_atlas_free_test_datasets(datasets, test_jsonl=test_jsonl, limit=limit)
    by_sample: list[dict[str, Any]] = []
    for dataset_name, dataset in atlas_free.items():
        for model_id in models:
            if model_id == "mlp_neurovlm":
                if dataset_name == "pubmed":
                    if include_mlp_flat:
                        by_sample.extend(
                            evaluate_mlp_flat_dataset(
                                dataset_name=dataset_name,
                                device=device,
                                limit=limit,
                                batch_size=batch_size,
                                include_voxel_auroc=include_voxel_auroc,
                            )
                        )
                else:
                    # Nilearn/NeuroVault: no dedicated main-package flat
                    # resource exists, but the atlas-free CNN's packed
                    # volumes share the MLP masker's exact MNI grid, so they
                    # can be converted and evaluated directly.
                    by_sample.extend(
                        evaluate_mlp_on_atlas_free_dataset(
                            dataset_name=dataset_name,
                            dataset=dataset,
                            device=device,
                            batch_size=batch_size,
                            include_voxel_auroc=include_voxel_auroc,
                        )
                    )
            else:
                by_sample.extend(
                    evaluate_cnn_ae_on_dataset(
                        model_id=model_id,
                        dataset_name=dataset_name,
                        dataset=dataset,
                        device=device,
                        batch_size=batch_size,
                        include_voxel_auroc=include_voxel_auroc,
                    )
                )

    summary = summarize_rows(by_sample)
    output_dir = Path(output_dir)
    by_sample_path = output_dir / "ae_reconstruction_by_sample.csv"
    summary_csv_path = output_dir / "ae_reconstruction_summary.csv"
    summary_json_path = output_dir / "ae_reconstruction_summary.json"
    write_csv(by_sample_path, by_sample)
    write_csv(summary_csv_path, summary)
    write_json(
        summary_json_path,
        {
            "datasets": list(datasets),
            "models": list(models),
            "limit": limit,
            "device": device,
            "by_sample_csv": by_sample_path,
            "summary_csv": summary_csv_path,
            "summary": summary,
        },
    )
    return {
        "by_sample_path": by_sample_path,
        "summary_csv_path": summary_csv_path,
        "summary_json_path": summary_json_path,
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=AE_MODEL_IDS, default=list(AE_MODEL_IDS))
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset/source after deterministic filtering.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-jsonl", default=None, help="Path to unified atlas-free test.jsonl. Defaults to local/HF split discovery.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-voxel-auroc", action="store_true", help="Skip voxel AUROC for faster smoke runs.")
    parser.add_argument(
        "--skip-mlp-flat",
        action="store_true",
        help="Only write unsupported atlas-free MLP rows; do not load main NeuroVLM flat PubMed/NeuroVault resources.",
    )
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
            batch_size=args.batch_size,
            include_voxel_auroc=not args.skip_voxel_auroc,
            include_mlp_flat=not args.skip_mlp_flat,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should report cache/HF issues clearly
        print(f"AE reconstruction comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote by-sample metrics to {result['by_sample_path']}")
    print(f"Wrote summary CSV to {result['summary_csv_path']}")
    print(f"Wrote summary JSON to {result['summary_json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
