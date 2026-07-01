"""Evaluation-only comparison for Stage 1A/1B autoencoder checkpoints.

This runner compares saved autoencoder checkpoints on fixed held-out test
splits. It never constructs optimizers, performs backpropagation, resumes
training, or writes checkpoint files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_DIR = Path(__file__).resolve().parents[4]
THREEDCNN_DIR = Path(__file__).resolve().parents[2]
DRIVE_ROOT = Path(os.environ.get("NEUROVLM_DRIVE_ROOT", "/content/drive/MyDrive/neurovlm"))
for path in [REPO_DIR / "src", THREEDCNN_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from atlas_free_cnn.evaluation.generation_metrics import generation_metrics
from atlas_free_cnn.training.autoencoder_losses import AutoencoderLossConfig
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset
from atlas_free_cnn.training.generation_losses import apply_prediction_activation
from atlas_free_cnn.training.checkpointing import metric_direction, metric_higher_is_better
from atlas_free_cnn.training.source_sampling import canonical_source, source_detail
from atlas_free_cnn.training.train_autoencoder import (
    VolumeCollator,
    apply_ae_recipe_defaults,
    build_model,
    filter_data_mode,
    model_config,
)
from neurovlm.gnn.ale_cnn import count_parameters


TARGET_SHAPE = (36, 45, 38)
DEFAULT_LATENT_DIM = 384
CHECKPOINT_FILENAMES = [
    "best_val_loss.pt",
    "best_spatial_corr.pt",
    "best_top1_dice.pt",
    "best_top5_dice.pt",
    "best_foreground_mse.pt",
    "best_top10_dice.pt",
    "best_cnn_autoencoder.pt",
    "last.pt",
    "last_cnn_autoencoder.pt",
    "best_generation_spatial_correlation.pt",
    "best_generation_top5_dice.pt",
]
STAGE1A_SELECTION_METRIC = "held_out_source_normalized_spatial_corr"
STAGE1A_SELECTION_POLICY = (
    "Rank within each Stage 1A recipe by source-normalized spatial correlation, "
    "then source-normalized top-5 Dice, then source-normalized foreground MSE, "
    "then source-normalized reconstruction MSE."
)
STAGE1A_RECIPE_BEST_COLUMNS = [
    "recipe",
    "best_checkpoint_name",
    "best_checkpoint_path",
    "selection_metric",
    "selection_metric_value",
    "mse",
    "foreground_mse",
    "spatial_corr",
    "top1_dice",
    "top5_dice",
    "top10_dice",
    "epoch",
    "heldout_split_fingerprint",
]
DOMAIN_TO_DATA_MODE = {
    "mixed": "mixed",
    "pubmed": "pubmed_only",
    "nilearn": "nilearn_only",
    "neurovault": "neurovault_only",
}
METRIC_COLUMNS = [
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
    "raw_pred_min",
    "raw_pred_max",
    "clamped_pred_min",
    "clamped_pred_max",
]


def _default_run_root() -> Path:
    return Path(os.environ.get("NEUROVLM_AE_ABLATION_RUN_DIR", "")).expanduser()


AE_RUN_ROOT = _default_run_root()
AE_RUN_REGISTRY: dict[str, dict[str, Any]] = {
    "mixed_baseline_raw_mse": {
        "run_dir": str(AE_RUN_ROOT / "01_stage1_ae_pretraining/mixed_baseline_raw_mse"),
        "stage": "stage1a",
        "training_domain": "mixed",
        "test_domains": ["mixed", "pubmed", "nilearn", "neurovault"],
    },
    "mixed_balanced_raw_mse": {
        "run_dir": str(AE_RUN_ROOT / "01_stage1_ae_pretraining/mixed_balanced_raw_mse"),
        "stage": "stage1a",
        "training_domain": "mixed",
        "test_domains": ["mixed", "pubmed", "nilearn", "neurovault"],
    },
    "mixed_balanced_hybrid_loss": {
        "run_dir": str(AE_RUN_ROOT / "01_stage1_ae_pretraining/mixed_balanced_hybrid_loss"),
        "stage": "stage1a",
        "training_domain": "mixed",
        "test_domains": ["mixed", "pubmed", "nilearn", "neurovault"],
    },
    "mixed_to_pubmed": {
        "run_dir": str(AE_RUN_ROOT / "02_stage1b_ae_finetuning/pubmed"),
        "stage": "stage1b",
        "training_domain": "pubmed",
        "test_domains": ["pubmed"],
        "cross_domain_test_domains": ["mixed", "nilearn", "neurovault"],
    },
    "mixed_to_nilearn": {
        "run_dir": str(AE_RUN_ROOT / "02_stage1b_ae_finetuning/nilearn"),
        "stage": "stage1b",
        "training_domain": "nilearn",
        "test_domains": ["nilearn"],
        "cross_domain_test_domains": ["mixed", "pubmed", "neurovault"],
    },
    "mixed_to_neurovault": {
        "run_dir": str(AE_RUN_ROOT / "02_stage1b_ae_finetuning/neurovault"),
        "stage": "stage1b",
        "training_domain": "neurovault",
        "test_domains": ["neurovault"],
        "cross_domain_test_domains": ["mixed", "pubmed", "nilearn"],
    },
}


@dataclass
class EvaluationConfig:
    registry: dict[str, dict[str, Any]]
    output_root: Path
    test_jsonl: Path | None = None
    device: str = "auto"
    eval_batch_size: int = 32
    batch_size_floor: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    amp: bool = True
    include_voxel_auroc: bool = True
    overwrite: bool = False
    bootstrap_samples: int = 500
    bootstrap_seed: int = 20260624
    qualitative_seed: int = 20260624
    qualitative_examples_per_domain: int = 4
    make_qualitative_plots: bool = True
    evaluate_stage1b_cross_domain: bool = True
    progress: bool = True


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
        json.dump(json_ready(value), f, indent=2)


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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def split_dir_has_jsonl(path: Path) -> bool:
    return all((path / name).exists() for name in ["train.jsonl", "val.jsonl", "test.jsonl"])


def hf_download_first_available(filenames: list[str], local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    repo_id = os.environ.get("NEUROVLM_ATLAS_FREE_HF_REPO", "neurovlm/atlas_free_cnn_dataset")
    local_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for filename in filenames:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            return Path(path)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    raise FileNotFoundError("Could not download any candidate from HF:\n" + "\n".join(errors))


def ensure_hf_unified_splits() -> Path:
    """Mirror notebook 6's fixed split fallback."""

    repo_id = os.environ.get("NEUROVLM_ATLAS_FREE_HF_REPO", "neurovlm/atlas_free_cnn_dataset")
    local_unified_cache_dir = REPO_DIR / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl_rebuild"
    local_split_dir = local_unified_cache_dir / "splits"
    local_pack_dir = REPO_DIR / "experiments/3dcnn/atlas_free_cnn/cache/hf_atlas_free_cnn_rebuild"
    print(f"Downloading atlas-free CNN split JSONLs from Hugging Face: {repo_id}")
    local_split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        downloaded = hf_download_first_available(
            [f"splits/{split}.jsonl", f"unified_jsonl_rebuild/splits/{split}.jsonl", f"{split}.jsonl"],
            local_unified_cache_dir,
        )
        target = local_split_dir / f"{split}.jsonl"
        if downloaded.resolve() != target.resolve():
            shutil.copy2(downloaded, target)
    for name in ["train_map_ids.json", "val_map_ids.json", "test_map_ids.json"]:
        try:
            downloaded = hf_download_first_available(
                [f"splits/{name}", f"unified_jsonl_rebuild/splits/{name}", name],
                local_unified_cache_dir,
            )
            target = local_split_dir / name
            if downloaded.resolve() != target.resolve():
                shutil.copy2(downloaded, target)
        except Exception as exc:
            print(f"Optional split sidecar not downloaded ({name}): {exc}")
    try:
        downloaded_volume = hf_download_first_available(
            [
                "atlas_free_cnn_volumes.pt",
                "hf_atlas_free_cnn/atlas_free_cnn_volumes.pt",
                "hf_atlas_free_cnn_rebuild/atlas_free_cnn_volumes.pt",
            ],
            local_pack_dir,
        )
        target_volume = local_pack_dir / "atlas_free_cnn_volumes.pt"
        if downloaded_volume.resolve() != target_volume.resolve():
            try:
                if target_volume.exists() or target_volume.is_symlink():
                    target_volume.unlink()
                os.symlink(downloaded_volume, target_volume)
            except Exception:
                shutil.copy2(downloaded_volume, target_volume)
        print("Volume tensor available at:", target_volume)
    except Exception as exc:
        print("WARNING: split JSONLs downloaded, but volume tensor was not prepared:", exc)
        print("Evaluation will fail unless tensor_path values inside JSONL resolve to an accessible tensor file.")
    return local_split_dir


def discover_split_dir() -> Path:
    override = os.environ.get("NEUROVLM_UNIFIED_SPLIT_DIR")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            REPO_DIR / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
            REPO_DIR / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl/splits",
            DRIVE_ROOT / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
            DRIVE_ROOT / "experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl/splits",
            DRIVE_ROOT / "atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
            DRIVE_ROOT / "atlas_free_cnn/cache/unified_jsonl/splits",
            DRIVE_ROOT / "cache/unified_jsonl_rebuild/splits",
            DRIVE_ROOT / "cache/unified_jsonl/splits",
            DRIVE_ROOT / "data_atlas_free_cnn/unified_jsonl_rebuild/splits",
            DRIVE_ROOT / "data_atlas_free_cnn/unified_jsonl/splits",
            DRIVE_ROOT / "data_atlas_free_cnn/cache/unified_jsonl_rebuild/splits",
            DRIVE_ROOT / "data_atlas_free_cnn/cache/unified_jsonl/splits",
            DRIVE_ROOT / "data_ale_3dcnn/unified_jsonl_rebuild/splits",
            DRIVE_ROOT / "data_ale_3dcnn/unified_jsonl/splits",
        ]
    )
    for candidate in candidates:
        if split_dir_has_jsonl(candidate):
            return candidate
    try:
        hf_split_dir = ensure_hf_unified_splits()
        if split_dir_has_jsonl(hf_split_dir):
            return hf_split_dir
    except Exception as exc:
        hf_error = exc
    else:
        hf_error = None
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find fixed unified split JSONLs locally, and Hugging Face fallback did not produce them. "
        "Set NEUROVLM_UNIFIED_SPLIT_DIR or pass --test-jsonl. Expected train.jsonl, val.jsonl, and test.jsonl in one of:\n"
        f"{checked}\n\n"
        f"HF dataset repo tried: {os.environ.get('NEUROVLM_ATLAS_FREE_HF_REPO', 'neurovlm/atlas_free_cnn_dataset')}\n"
        f"HF fallback error: {hf_error}"
    )


def resolve_test_jsonl(config: EvaluationConfig) -> Path:
    if config.test_jsonl:
        return Path(config.test_jsonl).expanduser()
    return discover_split_dir() / "test.jsonl"


def safe_stem(name: str) -> str:
    return Path(name).stem.replace("/", "_").replace(" ", "_")


def variant_output_dir(output_root: Path, variant: str, spec: dict[str, Any]) -> Path:
    if spec["stage"] == "stage1a":
        return output_root / "01_stage1a" / variant
    domain = str(spec["training_domain"])
    return output_root / "02_stage1b" / domain


def checkpoint_dir_for_run(run_dir: Path) -> Path:
    if (run_dir / "checkpoints").is_dir():
        return run_dir / "checkpoints"
    return run_dir


def discover_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    ckpt_dir = checkpoint_dir_for_run(run_dir)
    rows = []
    seen = set()
    for filename in CHECKPOINT_FILENAMES:
        path = ckpt_dir / filename
        if path.exists() and path not in seen:
            rows.append({"checkpoint_name": filename, "checkpoint_path": path})
            seen.add(path)
    return rows


def extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        state = payload.get("model") or payload.get("autoencoder") or payload.get("state_dict")
    else:
        state = payload
    if state is None or not isinstance(state, dict):
        raise KeyError("checkpoint does not contain model, autoencoder, or state_dict")
    keys = set(state.keys())
    if not any(str(k).startswith("encoder.") for k in keys) or not any(str(k).startswith("decoder.") for k in keys):
        raise KeyError("checkpoint does not contain a complete encoder+decoder autoencoder state")
    return state


def model_state_checksum(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def load_json_if_exists(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return {}


def checkpoint_metric_name(checkpoint_name: str, payload: dict[str, Any]) -> str:
    stem = Path(checkpoint_name).stem
    legacy = {
        "best_generation_spatial_correlation": "spatial_corr",
        "best_generation_top5_dice": "top5_dice",
    }
    if stem in legacy:
        return legacy[stem]
    if stem == "best_cnn_autoencoder":
        selection = str(payload.get("checkpoint_selection_metric", "best_val_loss"))
        return selection.removeprefix("best_")
    if stem.startswith("best_"):
        return stem.removeprefix("best_")
    return "last"


def selection_sort_value(row: dict[str, Any], metric_name: str, default: float = float("nan")) -> float:
    value = safe_float(row.get(metric_name), default)
    if metric_higher_is_better(metric_name):
        return value
    return -value


def comparison_higher_is_better(metric_name: str) -> bool:
    try:
        return metric_higher_is_better(metric_name)
    except ValueError:
        return True


def saved_val_score(metric_name: str, val_metrics: dict[str, Any]) -> Any:
    aliases = {
        "val_loss": "loss",
        "loss": "loss",
        "spatial_corr": "spatial_corr",
        "top1_dice": "top1_dice",
        "top5_dice": "top5_dice",
        "top10_dice": "top10_dice",
        "foreground_mse": "foreground_mse",
    }
    key = aliases.get(metric_name, metric_name)
    return val_metrics.get(key)


def build_checkpoint_manifest(registry: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for variant, spec in registry.items():
        run_dir = Path(spec["run_dir"]).expanduser()
        checkpoints = discover_checkpoints(run_dir) if run_dir.exists() else []
        if not checkpoints:
            failures.append(
                {
                    "variant": variant,
                    "stage": spec["stage"],
                    "run_dir": str(run_dir),
                    "load_status": "missing_run_or_checkpoints",
                    "error_message": f"No requested checkpoint files found under {checkpoint_dir_for_run(run_dir)}",
                }
            )
            continue
        checksum_to_canonical: dict[str, str] = {}
        for item in checkpoints:
            checkpoint_path = Path(item["checkpoint_path"])
            row = {
                "variant": variant,
                "stage": spec["stage"],
                "training_domain": spec["training_domain"],
                "checkpoint_name": item["checkpoint_name"],
                "checkpoint_path": str(checkpoint_path.resolve()),
                "run_dir": str(run_dir.resolve()),
            }
            try:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state = extract_model_state(payload)
                checksum = model_state_checksum(state)
                epoch = payload.get("epoch") if isinstance(payload, dict) else None
                val_metrics = payload.get("validation_metrics", {}) if isinstance(payload, dict) else {}
                metric_name = checkpoint_metric_name(item["checkpoint_name"], payload if isinstance(payload, dict) else {})
                canonical = checksum_to_canonical.setdefault(checksum, item["checkpoint_name"])
                row.update(
                    {
                        "checkpoint_epoch": epoch,
                        "model_state_checksum": checksum,
                        "canonical_checkpoint_name": canonical,
                        "is_alias": item["checkpoint_name"] != canonical,
                        "alias_of": "" if item["checkpoint_name"] == canonical else canonical,
                        "saved_checkpoint_selection_metric": metric_name,
                        "saved_checkpoint_selection_direction": "n/a" if metric_name == "last" else metric_direction(metric_name),
                        "saved_validation_score": saved_val_score(metric_name, val_metrics),
                        "saved_validation_metrics": json.dumps(json_ready(val_metrics)),
                        "load_status": "manifest_ok",
                        "error_message": "",
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "checkpoint_epoch": None,
                        "model_state_checksum": "",
                        "canonical_checkpoint_name": item["checkpoint_name"],
                        "is_alias": False,
                        "alias_of": "",
                        "saved_checkpoint_selection_metric": checkpoint_metric_name(item["checkpoint_name"], {}),
                        "saved_checkpoint_selection_direction": "n/a",
                        "saved_validation_score": None,
                        "saved_validation_metrics": "{}",
                        "load_status": "manifest_failed",
                        "error_message": repr(exc),
                    }
                )
                failures.append(row)
            manifest.append(row)
    return manifest, failures


def load_config_for_checkpoint(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(payload.get("config") or {})
    if not cfg:
        cfg = load_json_if_exists(
            [
                run_dir / "autoencoder_config.json",
                run_dir / "config" / "ae_config.json",
            ]
        )
    cfg = apply_ae_recipe_defaults(cfg)
    if "model" not in cfg and payload.get("model_architecture"):
        cfg["model"] = payload["model_architecture"]
    if "target_shape" not in cfg and payload.get("target_shape"):
        cfg["target_shape"] = payload["target_shape"]
    cfg.setdefault("target_shape", list(TARGET_SHAPE))
    return cfg


def validate_checkpoint_config(cfg: dict[str, Any]) -> list[str]:
    warnings = []
    target_shape = tuple(int(v) for v in cfg.get("target_shape", TARGET_SHAPE))
    if target_shape != TARGET_SHAPE:
        warnings.append(f"target_shape {target_shape} != expected {TARGET_SHAPE}")
    mcfg = model_config(cfg)
    latent_dim = int(mcfg.get("latent_dim", DEFAULT_LATENT_DIM))
    if latent_dim != DEFAULT_LATENT_DIM:
        warnings.append(f"latent_dim {latent_dim} != default {DEFAULT_LATENT_DIM}; using checkpoint config")
    for key in ["base_channels", "num_blocks"]:
        if key not in mcfg:
            warnings.append(f"model config missing {key}")
    return warnings


def load_autoencoder_for_eval(
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload is not a dict")
    state = extract_model_state(payload)
    cfg = load_config_for_checkpoint(run_dir, payload)
    warnings = validate_checkpoint_config(cfg)
    target_shape = tuple(int(v) for v in cfg.get("target_shape", TARGET_SHAPE))
    model = build_model(cfg, target_shape, device)
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    compatibility_conversion = ""
    try:
        result = model.load_state_dict(state, strict=True)
        missing_keys = list(result.missing_keys)
        unexpected_keys = list(result.unexpected_keys)
    except RuntimeError as exc:
        stripped = {}
        for key, value in state.items():
            stripped[key.removeprefix("module.")] = value
        if stripped != state:
            result = model.load_state_dict(stripped, strict=True)
            missing_keys = list(result.missing_keys)
            unexpected_keys = list(result.unexpected_keys)
            compatibility_conversion = "stripped module. prefix"
            warnings.append(compatibility_conversion)
        else:
            raise exc
    model.eval()
    load_info = {
        "architecture": model.__class__.__name__,
        "model_architecture": json.dumps(json_ready(model_config(cfg))),
        "parameter_count": count_parameters(model, trainable_only=False),
        "checkpoint_epoch": payload.get("epoch"),
        "saved_validation_metrics": payload.get("validation_metrics", {}),
        "loss_recipe": cfg.get("loss", {}),
        "prediction_activation": AutoencoderLossConfig.from_config(cfg).prediction_activation,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "warnings": warnings,
        "compatibility_conversion": compatibility_conversion,
    }
    return model, cfg, load_info


def build_test_datasets(test_jsonl: Path) -> dict[str, UnifiedMapTextDataset]:
    base = UnifiedMapTextDataset(test_jsonl)
    datasets = {}
    for domain, data_mode in DOMAIN_TO_DATA_MODE.items():
        ds = UnifiedMapTextDataset(test_jsonl)
        datasets[domain] = filter_data_mode(ds, data_mode)
    if len(datasets["mixed"]) != len(base):
        raise RuntimeError("mixed test dataset unexpectedly differs from unfiltered test JSONL")
    return datasets


def split_fingerprint(dataset: UnifiedMapTextDataset, split_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h = hashlib.sha256()
    counts = Counter()
    for idx, row in enumerate(dataset.rows):
        src = canonical_source(row)
        counts[src] += 1
        record = {
            "ordinal": idx,
            "map_id": row.get("map_id"),
            "tensor_path": row.get("tensor_path"),
            "tensor_index": row.get("tensor_index"),
            "source": src,
            "source_detail": source_detail(row),
        }
        h.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    info = {
        "split": split_name,
        "n": len(dataset),
        "fingerprint": h.hexdigest(),
        "source_counts": dict(sorted(counts.items())),
    }
    rows = [{"split": split_name, "source": src, "n": n, "fingerprint": info["fingerprint"]} for src, n in sorted(counts.items())]
    return info, rows


def prediction_for_metrics(raw_pred: torch.Tensor, loss_cfg: AutoencoderLossConfig) -> torch.Tensor:
    if loss_cfg.prediction_activation in {"none", "raw", "identity", ""}:
        activated = raw_pred
    else:
        activated = apply_prediction_activation(raw_pred, loss_cfg.prediction_activation)
    return activated.float().clamp(0.0, 1.0)


def batch_to_per_example_rows(
    raw_pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, Any],
    loss_cfg: AutoencoderLossConfig,
    include_voxel_auroc: bool,
) -> list[dict[str, Any]]:
    raw_cpu = raw_pred.detach().float().cpu()
    target_cpu = target.detach().float().cpu()
    pred_cpu = prediction_for_metrics(raw_cpu, loss_cfg)
    rows = []
    for i, map_id in enumerate(batch["map_id"]):
        metrics = generation_metrics(
            pred_cpu[i : i + 1],
            target_cpu[i : i + 1],
            include_voxel_auroc=include_voxel_auroc,
        )
        rows.append(
            {
                "map_id": map_id,
                "source": batch["source"][i],
                "source_detail": batch["source_detail"][i],
                "tensor_index": batch["metadata"][i].get("tensor_index"),
                "raw_pred_min": float(raw_cpu[i].min().item()),
                "raw_pred_max": float(raw_cpu[i].max().item()),
                "clamped_pred_min": float(pred_cpu[i].min().item()),
                "clamped_pred_max": float(pred_cpu[i].max().item()),
                **metrics,
            }
        )
    return rows


def valid_per_example_file(path: Path, expected_n: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        rows = read_csv_rows(path)
    except Exception:
        return False
    return len(rows) == expected_n and bool(rows) and "map_id" in rows[0]


def evaluate_checkpoint_on_split(
    model: Any,
    cfg: dict[str, Any],
    dataset: UnifiedMapTextDataset,
    device: torch.device,
    out_path: Path,
    eval_cfg: EvaluationConfig,
) -> list[dict[str, Any]]:
    if valid_per_example_file(out_path, len(dataset)) and not eval_cfg.overwrite:
        return read_csv_rows(out_path)
    loss_cfg = AutoencoderLossConfig.from_config(cfg)
    batch_size = int(eval_cfg.eval_batch_size)
    last_oom = None
    while batch_size >= int(eval_cfg.batch_size_floor):
        try:
            rows: list[dict[str, Any]] = []
            collate = VolumeCollator(tuple(int(v) for v in cfg.get("target_shape", TARGET_SHAPE)))
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(eval_cfg.num_workers),
                collate_fn=collate,
                pin_memory=bool(eval_cfg.pin_memory and device.type == "cuda"),
            )
            iterator = loader
            if eval_cfg.progress and tqdm is not None:
                iterator = tqdm(loader, desc=f"eval bs={batch_size}", unit="batch", leave=False)
            model.eval()
            with torch.inference_mode():
                for batch in iterator:
                    x = batch["volume"].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=bool(eval_cfg.amp and device.type == "cuda")):
                        raw_pred = model(x)
                    rows.extend(
                        batch_to_per_example_rows(
                            raw_pred,
                            x,
                            batch,
                            loss_cfg,
                            include_voxel_auroc=eval_cfg.include_voxel_auroc,
                        )
                    )
                    del x, raw_pred
            write_csv(out_path, rows)
            return rows
        except torch.cuda.OutOfMemoryError as exc:
            last_oom = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size //= 2
    raise RuntimeError(f"CUDA OOM even at batch size floor {eval_cfg.batch_size_floor}: {last_oom}")


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        try:
            value = float(row[key])
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return vals


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not values or samples <= 0 or np is None:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(samples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate_rows(rows: list[dict[str, Any]], eval_cfg: EvaluationConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"test_sample_count": len(rows)}
    for key in METRIC_COLUMNS:
        vals = numeric_values(rows, key)
        if not vals:
            out[key] = float("nan")
            out[f"{key}_median"] = float("nan")
            out[f"{key}_std"] = float("nan")
            out[f"{key}_ci95_low"] = float("nan")
            out[f"{key}_ci95_high"] = float("nan")
            continue
        lo, hi = bootstrap_ci(vals, samples=eval_cfg.bootstrap_samples, seed=eval_cfg.bootstrap_seed)
        out[key] = float(sum(vals) / len(vals))
        out[f"{key}_median"] = float(np.median(vals) if np is not None else sorted(vals)[len(vals) // 2])
        out[f"{key}_std"] = float(np.std(vals, ddof=1) if np is not None and len(vals) > 1 else 0.0)
        out[f"{key}_ci95_low"] = lo
        out[f"{key}_ci95_high"] = hi
    return out


def augment_rows(rows: list[dict[str, Any]], prefix: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**row, **prefix} for row in rows]


def copy_alias_results(
    canonical_path: Path,
    alias_path: Path,
    alias_prefix: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = read_csv_rows(canonical_path)
    alias_rows = augment_rows(rows, alias_prefix)
    write_csv(alias_path, alias_rows)
    return alias_rows


def manifest_lookup(manifest: list[dict[str, Any]], variant: str, checkpoint_name: str) -> dict[str, Any] | None:
    for row in manifest:
        if row["variant"] == variant and row["checkpoint_name"] == checkpoint_name:
            return row
    return None


def run_checkpoint_evaluation(
    eval_cfg: EvaluationConfig,
    output_root: Path,
    manifest: list[dict[str, Any]],
    fingerprints: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_jsonl = resolve_test_jsonl(eval_cfg)
    datasets = build_test_datasets(test_jsonl)
    device_name = eval_cfg.device
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    comparison_rows: list[dict[str, Any]] = []
    load_failures: list[dict[str, Any]] = []
    for variant, spec in eval_cfg.registry.items():
        run_dir = Path(spec["run_dir"]).expanduser()
        if not run_dir.exists():
            continue
        variant_dir = variant_output_dir(output_root, variant, spec)
        per_example_dir = variant_dir / "per_example"
        summary_dir = variant_dir / "summaries"
        per_example_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        domain_list = list(spec.get("test_domains", []))
        if spec["stage"] == "stage1b" and eval_cfg.evaluate_stage1b_cross_domain:
            domain_list.extend([d for d in spec.get("cross_domain_test_domains", []) if d not in domain_list])
        canonical_cache: dict[tuple[str, str], Path] = {}
        canonical_load_info: dict[str, dict[str, Any]] = {}
        for checkpoint_row in [r for r in manifest if r["variant"] == variant]:
            checkpoint_name = checkpoint_row["checkpoint_name"]
            checkpoint_path = Path(checkpoint_row["checkpoint_path"])
            if checkpoint_row["load_status"] != "manifest_ok":
                load_failures.append(checkpoint_row)
                continue
            canonical_name = checkpoint_row["canonical_checkpoint_name"]
            is_alias = bool(str(checkpoint_row["is_alias"]).lower() in {"true", "1"} or checkpoint_row["is_alias"] is True)
            load_info: dict[str, Any] = {}
            model = None
            cfg = {}
            if not is_alias:
                try:
                    model, cfg, load_info = load_autoencoder_for_eval(checkpoint_path, run_dir, device)
                    canonical_load_info[checkpoint_name] = load_info
                except Exception as exc:
                    failed = dict(checkpoint_row)
                    failed.update({"load_status": "load_failed", "error_message": repr(exc)})
                    load_failures.append(failed)
                    for test_domain in domain_list:
                        comparison_rows.append(
                            {
                                **base_comparison_prefix(spec, checkpoint_row, test_domain, fingerprints),
                                "load_status": "load_failed",
                                "error_message": repr(exc),
                            }
                        )
                    continue
            for test_domain in domain_list:
                if test_domain not in datasets:
                    continue
                dataset = datasets[test_domain]
                eval_scope = "primary" if test_domain in spec.get("test_domains", []) else "cross_domain"
                base_prefix = base_comparison_prefix(spec, checkpoint_row, test_domain, fingerprints)
                base_prefix["eval_scope"] = eval_scope
                per_example_path = per_example_dir / f"{safe_stem(checkpoint_name)}__{test_domain}_per_example.csv"
                if is_alias:
                    load_info = canonical_load_info.get(canonical_name, {})
                    canonical_path = canonical_cache.get((canonical_name, test_domain))
                    if canonical_path and canonical_path.exists():
                        rows = copy_alias_results(
                            canonical_path,
                            per_example_path,
                            {**base_prefix, **load_info, "load_status": "alias_copied"},
                        )
                    else:
                        rows = []
                        base_prefix["warnings"] = append_warning(base_prefix.get("warnings", ""), "alias canonical output missing")
                else:
                    rows_raw = evaluate_checkpoint_on_split(
                        model,
                        cfg,
                        dataset,
                        device,
                        per_example_path,
                        eval_cfg,
                    )
                    rows = augment_rows(rows_raw, {**base_prefix, **load_info, "load_status": "loaded"})
                    write_csv(per_example_path, rows)
                    canonical_cache[(checkpoint_name, test_domain)] = per_example_path
                agg = aggregate_rows(rows, eval_cfg) if rows else {"test_sample_count": 0}
                comparison_rows.append(
                    {
                        **base_prefix,
                        **load_info,
                        **agg,
                        "load_status": "alias_copied" if is_alias else base_prefix.get("load_status", "loaded"),
                        "error_message": base_prefix.get("error_message", ""),
                    }
                )
            if model is not None:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        variant_rows = [r for r in comparison_rows if r["variant"] == variant]
        primary_rows = [r for r in variant_rows if r.get("eval_scope") == "primary"]
        cross_rows = [r for r in variant_rows if r.get("eval_scope") == "cross_domain"]
        write_csv(summary_dir / f"{variant}_checkpoint_comparison.csv", primary_rows)
        if cross_rows:
            write_csv(summary_dir / f"{variant}_cross_domain_checkpoint_comparison.csv", cross_rows)
    return comparison_rows, load_failures


def ensure_expected_output_files(output_root: Path, registry: dict[str, dict[str, Any]]) -> None:
    for variant, spec in registry.items():
        base_dir = variant_output_dir(output_root, variant, spec)
        (base_dir / "per_example").mkdir(parents=True, exist_ok=True)
        (base_dir / "summaries").mkdir(parents=True, exist_ok=True)
        (base_dir / "plots").mkdir(parents=True, exist_ok=True)
        comparison = base_dir / "summaries" / f"{variant}_checkpoint_comparison.csv"
        if not comparison.exists():
            write_csv(comparison, [])
        if spec["stage"] == "stage1b":
            cross = base_dir / "summaries" / f"{variant}_cross_domain_checkpoint_comparison.csv"
            if not cross.exists():
                write_csv(cross, [])
    for domain in ["pubmed", "nilearn", "neurovault"]:
        path = output_root / "02_stage1b" / domain / f"{domain}_stage1b_checkpoint_selection.csv"
        if not path.exists():
            write_csv(path, [])
    stage1a_path = output_root / "01_stage1a/mixed_stage1a_checkpoint_selection.csv"
    if not stage1a_path.exists():
        write_csv(stage1a_path, [])
    for filename in ["stage1a_all_checkpoint_eval.csv", "stage1a_recipe_best_checkpoint_comparison.csv"]:
        path = output_root / "01_stage1a" / filename
        if not path.exists():
            write_csv(path, [])


def append_warning(existing: str, warning: str) -> str:
    if not existing:
        return warning
    return f"{existing}; {warning}"


def base_comparison_prefix(
    spec: dict[str, Any],
    checkpoint_row: dict[str, Any],
    test_domain: str,
    fingerprints: dict[str, Any],
) -> dict[str, Any]:
    warnings = checkpoint_row.get("warnings", "")
    return {
        "variant": checkpoint_row["variant"],
        "stage": spec["stage"],
        "training_domain": spec["training_domain"],
        "test_domain": test_domain,
        "checkpoint_name": checkpoint_row["checkpoint_name"],
        "checkpoint_path": checkpoint_row["checkpoint_path"],
        "canonical_checkpoint_name": checkpoint_row["canonical_checkpoint_name"],
        "alias_status": "alias" if checkpoint_row.get("is_alias") else "canonical",
        "checkpoint_epoch": checkpoint_row.get("checkpoint_epoch"),
        "saved_checkpoint_selection_metric": checkpoint_row.get("saved_checkpoint_selection_metric"),
        "saved_checkpoint_selection_direction": checkpoint_row.get("saved_checkpoint_selection_direction"),
        "saved_validation_score": checkpoint_row.get("saved_validation_score"),
        "test_split_fingerprint": fingerprints[test_domain]["fingerprint"],
        "model_state_checksum": checkpoint_row.get("model_state_checksum"),
        "warnings": warnings,
    }


def stage1a_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("variant", "")), str(row.get("checkpoint_name", "")), str(row.get("test_domain", "")))


def minmax_normalize(rows: list[dict[str, Any]], key: str, *, higher_is_better: bool | None = None) -> dict[tuple[str, str, str], float]:
    if higher_is_better is None:
        higher_is_better = metric_higher_is_better(key)
    vals = []
    row_keys = []
    for row in rows:
        try:
            val = float(row[key])
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
            row_keys.append(stage1a_row_key(row))
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    out = {}
    for row_key, val in zip(row_keys, vals):
        norm = 0.5 if hi == lo else (val - lo) / (hi - lo)
        if not higher_is_better:
            norm = 1.0 - norm
        out[row_key] = float(norm)
    return out


def stage1a_variants(registry: dict[str, dict[str, Any]] | None, comparison_rows: list[dict[str, Any]]) -> list[str]:
    variants: list[str] = []
    if registry:
        variants.extend([variant for variant, spec in registry.items() if spec.get("stage") == "stage1a"])
    for row in comparison_rows:
        if row.get("stage") == "stage1a" and row.get("variant") not in variants:
            variants.append(str(row.get("variant")))
    return variants


def stage1a_selection_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        safe_float(row.get("mean_source_normalized_spatial_corr"), -1),
        safe_float(row.get("mean_source_normalized_top5_dice"), -1),
        safe_float(row.get("mean_source_normalized_foreground_mse"), -1),
        safe_float(row.get("mean_source_normalized_reconstruction_mse"), -1),
    )


def stage1a_mixed_row(group: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in group if r.get("test_domain") == "mixed"), group[0])


def stage1a_checkpoint_eval_row(
    recipe: str,
    checkpoint_name: str,
    group: list[dict[str, Any]],
    *,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    if not group:
        return {
            "recipe": recipe,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": "",
            "selection_metric": STAGE1A_SELECTION_METRIC,
            "selection_metric_value": "",
            "mse": "",
            "foreground_mse": "",
            "spatial_corr": "",
            "top1_dice": "",
            "top5_dice": "",
            "top10_dice": "",
            "epoch": "",
            "heldout_split_fingerprint": "",
            "status": status,
            "error_message": error_message,
        }
    mixed = stage1a_mixed_row(group)
    return {
        "recipe": recipe,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": mixed.get("checkpoint_path", ""),
        "selection_metric": STAGE1A_SELECTION_METRIC,
        "selection_metric_value": mixed.get("mean_source_normalized_spatial_corr", ""),
        "mse": mixed.get("reconstruction_mse", ""),
        "foreground_mse": mixed.get("foreground_mse", ""),
        "spatial_corr": mixed.get("spatial_corr", ""),
        "top1_dice": mixed.get("top1_dice", ""),
        "top5_dice": mixed.get("top5_dice", ""),
        "top10_dice": mixed.get("top10_dice", ""),
        "epoch": mixed.get("checkpoint_epoch", ""),
        "heldout_split_fingerprint": mixed.get("test_split_fingerprint", ""),
        "status": status,
        "error_message": error_message,
    }


def stage1a_recipe_best_row(recipe: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipe": recipe,
        "best_checkpoint_name": row.get("checkpoint_name", ""),
        "best_checkpoint_path": row.get("checkpoint_path", ""),
        "selection_metric": STAGE1A_SELECTION_METRIC,
        "selection_metric_value": row.get("mean_source_normalized_spatial_corr", ""),
        "mse": row.get("mse", ""),
        "foreground_mse": row.get("foreground_mse", ""),
        "spatial_corr": row.get("spatial_corr", ""),
        "top1_dice": row.get("top1_dice", ""),
        "top5_dice": row.get("top5_dice", ""),
        "top10_dice": row.get("top10_dice", ""),
        "epoch": row.get("epoch", ""),
        "heldout_split_fingerprint": row.get("heldout_split_fingerprint", ""),
        "selection_policy": STAGE1A_SELECTION_POLICY,
        "rank_within_recipe": row.get("rank_within_recipe", ""),
    }


def stage1a_manifest_missing_rows(
    variants: list[str],
    manifest: list[dict[str, Any]] | None,
    loaded_groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = manifest or []
    for item in manifest:
        if item.get("stage") != "stage1a":
            continue
        variant = str(item.get("variant", ""))
        if variant not in variants:
            continue
        status = str(item.get("load_status", ""))
        if status in {"manifest_ok", "loaded", "alias_copied"}:
            continue
        rows.append(
            stage1a_checkpoint_eval_row(
                variant,
                str(item.get("checkpoint_name") or "missing_checkpoints"),
                [],
                status=status or "missing",
                error_message=str(item.get("error_message", "")),
            )
        )
    for variant in variants:
        has_loaded = any(key[0] == variant for key in loaded_groups)
        has_missing = any(row["recipe"] == variant for row in rows)
        if not has_loaded and not has_missing:
            rows.append(
                stage1a_checkpoint_eval_row(
                    variant,
                    "missing_checkpoints",
                    [],
                    status="missing_run_or_checkpoints",
                    error_message="No loadable Stage 1A checkpoints were evaluated for this recipe.",
                )
            )
    return rows


def create_stage1a_selection(
    comparison_rows: list[dict[str, Any]],
    output_root: Path,
    registry: dict[str, dict[str, Any]] | None = None,
    manifest: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    variants = stage1a_variants(registry, comparison_rows)
    rows = [
        r for r in comparison_rows
        if r.get("stage") == "stage1a"
        and r.get("eval_scope") == "primary"
        and r.get("load_status") in {"loaded", "alias_copied"}
    ]
    if not rows:
        missing_rows = stage1a_manifest_missing_rows(variants, manifest, {})
        write_csv(output_root / "01_stage1a/stage1a_all_checkpoint_eval.csv", missing_rows)
        write_csv(output_root / "01_stage1a/stage1a_recipe_best_checkpoint_comparison.csv", [])
        write_csv(output_root / "01_stage1a/mixed_stage1a_checkpoint_selection.csv", [])
        return [], None

    spatial_norm = minmax_normalize(rows, "spatial_corr")
    top5_norm = minmax_normalize(rows, "top5_dice")
    fg_norm = minmax_normalize(rows, "foreground_mse")
    mse_norm = minmax_normalize(rows, "reconstruction_mse")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["checkpoint_name"]))].append(row)

    selection_rows: list[dict[str, Any]] = []
    all_eval_rows: list[dict[str, Any]] = []
    for (recipe, checkpoint_name), group in grouped.items():
        key_vals = [stage1a_row_key(r) for r in group]
        mean_spatial_norm = mean([spatial_norm.get(k, float("nan")) for k in key_vals])
        mean_top5_norm = mean([top5_norm.get(k, float("nan")) for k in key_vals])
        mean_fg_norm = mean([fg_norm.get(k, float("nan")) for k in key_vals])
        mean_mse_norm = mean([mse_norm.get(k, float("nan")) for k in key_vals])
        mixed = stage1a_mixed_row(group)
        selection_row = {
            "recipe": recipe,
            "variant": recipe,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": mixed.get("checkpoint_path", group[0].get("checkpoint_path", "")),
            "canonical_checkpoint_name": group[0].get("canonical_checkpoint_name", ""),
            "checkpoint_epoch": group[0].get("checkpoint_epoch", ""),
            "mean_source_normalized_spatial_corr": mean_spatial_norm,
            "mean_source_normalized_top5_dice": mean_top5_norm,
            "mean_source_normalized_foreground_mse": mean_fg_norm,
            "mean_source_normalized_reconstruction_mse": mean_mse_norm,
            "generalization_score": mean_spatial_norm + mean_top5_norm + 0.25 * mean_fg_norm + 0.10 * mean_mse_norm,
            "mean_spatial_corr": mean_float(group, "spatial_corr"),
            "mean_top5_dice": mean_float(group, "top5_dice"),
            "mean_foreground_mse": mean_float(group, "foreground_mse"),
            "mean_reconstruction_mse": mean_float(group, "reconstruction_mse"),
            "mixed_spatial_corr": metric_for_domain(group, "mixed", "spatial_corr"),
            "pubmed_spatial_corr": metric_for_domain(group, "pubmed", "spatial_corr"),
            "nilearn_spatial_corr": metric_for_domain(group, "nilearn", "spatial_corr"),
            "neurovault_spatial_corr": metric_for_domain(group, "neurovault", "spatial_corr"),
            "mixed_top5_dice": metric_for_domain(group, "mixed", "top5_dice"),
            "pubmed_top5_dice": metric_for_domain(group, "pubmed", "top5_dice"),
            "nilearn_top5_dice": metric_for_domain(group, "nilearn", "top5_dice"),
            "neurovault_top5_dice": metric_for_domain(group, "neurovault", "top5_dice"),
            "expected_candidate": checkpoint_name == "best_spatial_corr.pt",
            "selection_metric": STAGE1A_SELECTION_METRIC,
            "selection_metric_value": mean_spatial_norm,
            "mse": mixed.get("reconstruction_mse", ""),
            "foreground_mse": mixed.get("foreground_mse", ""),
            "spatial_corr": mixed.get("spatial_corr", ""),
            "top1_dice": mixed.get("top1_dice", ""),
            "top5_dice": mixed.get("top5_dice", ""),
            "top10_dice": mixed.get("top10_dice", ""),
            "epoch": mixed.get("checkpoint_epoch", ""),
            "heldout_split_fingerprint": mixed.get("test_split_fingerprint", ""),
            "status": "alias" if any(r.get("alias_status") == "alias" for r in group) else "evaluated",
            "selection_policy": STAGE1A_SELECTION_POLICY,
        }
        eval_row = stage1a_checkpoint_eval_row(recipe, checkpoint_name, group, status=selection_row["status"])
        eval_row.update(
            {
                "mean_source_normalized_spatial_corr": mean_spatial_norm,
                "mean_source_normalized_top5_dice": mean_top5_norm,
                "mean_source_normalized_foreground_mse": mean_fg_norm,
                "mean_source_normalized_reconstruction_mse": mean_mse_norm,
                "generalization_score": selection_row["generalization_score"],
                "selection_policy": STAGE1A_SELECTION_POLICY,
            }
        )
        all_eval_rows.append(eval_row)
        if selection_row["status"] != "alias":
            selection_rows.append(selection_row)

    all_eval_rows.extend(stage1a_manifest_missing_rows(variants, manifest, grouped))
    selection_by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selection_rows:
        selection_by_recipe[row["recipe"]].append(row)

    recipe_best_rows = []
    for recipe in variants:
        recipe_rows = selection_by_recipe.get(recipe, [])
        recipe_rows.sort(key=stage1a_selection_sort_key, reverse=True)
        for rank, row in enumerate(recipe_rows, 1):
            row["rank_within_recipe"] = rank
            best_eval = next(
                r for r in all_eval_rows
                if r["recipe"] == recipe and r["checkpoint_name"] == row["checkpoint_name"]
            )
            best_eval["rank_within_recipe"] = rank
            best_eval["is_recipe_best"] = rank == 1
            if rank == 1:
                recipe_best_rows.append(stage1a_recipe_best_row(recipe, best_eval))
    for row in all_eval_rows:
        row.setdefault("rank_within_recipe", "")
        row.setdefault("is_recipe_best", False)

    recipe_best_rows.sort(key=lambda r: safe_float(r.get("selection_metric_value"), -1), reverse=True)
    for rank, row in enumerate(recipe_best_rows, 1):
        row["recipe_comparison_rank"] = rank
    all_eval_rows.sort(key=lambda r: (str(r.get("recipe")), safe_float(r.get("rank_within_recipe"), 1e9), str(r.get("checkpoint_name"))))

    write_csv(output_root / "01_stage1a/stage1a_all_checkpoint_eval.csv", all_eval_rows)
    write_csv(output_root / "01_stage1a/stage1a_recipe_best_checkpoint_comparison.csv", recipe_best_rows)

    baseline_rows = selection_by_recipe.get("mixed_baseline_raw_mse", [])
    path = output_root / "01_stage1a/mixed_stage1a_checkpoint_selection.csv"
    write_csv(path, baseline_rows)
    return selection_rows, baseline_rows[0] if baseline_rows else None


def create_stage1b_selection(
    comparison_rows: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for domain, variant in [
        ("pubmed", "mixed_to_pubmed"),
        ("nilearn", "mixed_to_nilearn"),
        ("neurovault", "mixed_to_neurovault"),
    ]:
        rows = [
            r for r in comparison_rows
            if r["variant"] == variant
            and r["test_domain"] == domain
            and r.get("eval_scope") == "primary"
            and r.get("alias_status") != "alias"
            and r.get("load_status") in {"loaded", "alias_copied"}
        ]
        rows.sort(
            key=lambda r: (
                selection_sort_value(r, "top5_dice", -1),
                selection_sort_value(r, "spatial_corr", -1),
                selection_sort_value(r, "foreground_mse", 1e9),
                selection_sort_value(r, "reconstruction_mse", 1e9),
            ),
            reverse=True,
        )
        selection_rows = []
        expected = next((r for r in rows if r["checkpoint_name"] == "best_top5_dice.pt"), None)
        for rank, row in enumerate(rows, 1):
            selection_rows.append(
                {
                    "rank": rank,
                    "domain": domain,
                    "variant": variant,
                    "checkpoint_name": row["checkpoint_name"],
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_epoch": row["checkpoint_epoch"],
                    "top5_dice": row.get("top5_dice"),
                    "spatial_corr": row.get("spatial_corr"),
                    "foreground_mse": row.get("foreground_mse"),
                    "reconstruction_mse": row.get("reconstruction_mse"),
                    "expected_candidate": row["checkpoint_name"] == "best_top5_dice.pt",
                    "beats_expected_top5": (
                        expected is not None
                        and safe_float(row.get("top5_dice"), -1) > safe_float(expected.get("top5_dice"), -1)
                    ),
                }
            )
        path = output_root / "02_stage1b" / domain / f"{domain}_stage1b_checkpoint_selection.csv"
        write_csv(path, selection_rows)
        out[domain] = selection_rows
    return out


def mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return mean([safe_float(r.get(key)) for r in rows])


def metric_for_domain(rows: list[dict[str, Any]], domain: str, key: str) -> float:
    for row in rows:
        if row.get("test_domain") == domain:
            return safe_float(row.get(key))
    return float("nan")


def selected_stage1b_row(domain: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    empirical = rows[0]
    expected = next((r for r in rows if r["checkpoint_name"] == "best_top5_dice.pt"), None)
    if expected is None:
        return empirical
    expected_top5 = safe_float(expected.get("top5_dice"), -1)
    empirical_top5 = safe_float(empirical.get("top5_dice"), -1)
    if empirical["checkpoint_name"] != expected["checkpoint_name"] and empirical_top5 > expected_top5 + 1e-6:
        return empirical
    return expected


def per_example_path_for(output_root: Path, variant: str, spec: dict[str, Any], checkpoint_name: str, domain: str) -> Path:
    return variant_output_dir(output_root, variant, spec) / "per_example" / f"{safe_stem(checkpoint_name)}__{domain}_per_example.csv"


def create_baseline_vs_specialized(
    output_root: Path,
    registry: dict[str, dict[str, Any]],
    stage1a_selected: dict[str, Any] | None,
    stage1b_selected: dict[str, list[dict[str, Any]]],
    eval_cfg: EvaluationConfig,
) -> list[dict[str, Any]]:
    rows_written = []
    if stage1a_selected is None:
        return rows_written
    for domain, variant in [
        ("pubmed", "mixed_to_pubmed"),
        ("nilearn", "mixed_to_nilearn"),
        ("neurovault", "mixed_to_neurovault"),
    ]:
        spec_a = registry["mixed_baseline_raw_mse"]
        spec_b = registry[variant]
        b_row = selected_stage1b_row(domain, stage1b_selected.get(domain, []))
        if b_row is None:
            write_csv(output_root / "03_baseline_vs_specialized" / f"{domain}_ae_baseline_vs_specialized.csv", [])
            continue
        baseline_path = per_example_path_for(output_root, "mixed_baseline_raw_mse", spec_a, stage1a_selected["checkpoint_name"], domain)
        specialized_path = per_example_path_for(output_root, variant, spec_b, b_row["checkpoint_name"], domain)
        if not baseline_path.exists() or not specialized_path.exists():
            write_csv(output_root / "03_baseline_vs_specialized" / f"{domain}_ae_baseline_vs_specialized.csv", [])
            continue
        baseline = {r["map_id"]: r for r in read_csv_rows(baseline_path)}
        specialized = {r["map_id"]: r for r in read_csv_rows(specialized_path)}
        common_ids = [mid for mid in baseline if mid in specialized]
        comp_rows = []
        for metric in METRIC_COLUMNS:
            base_vals = [safe_float(baseline[mid].get(metric)) for mid in common_ids]
            spec_vals = [safe_float(specialized[mid].get(metric)) for mid in common_ids]
            pairs = [(b, s) for b, s in zip(base_vals, spec_vals) if math.isfinite(b) and math.isfinite(s)]
            if not pairs:
                continue
            b_mean = sum(b for b, _ in pairs) / len(pairs)
            s_mean = sum(s for _, s in pairs) / len(pairs)
            diff = s_mean - b_mean
            rel = diff / abs(b_mean) * 100.0 if b_mean else float("nan")
            higher = comparison_higher_is_better(metric)
            winner = "specialized" if (s_mean > b_mean if higher else s_mean < b_mean) else "baseline"
            ci_low, ci_high = paired_bootstrap_diff_ci(
                [b for b, _ in pairs],
                [s for _, s in pairs],
                samples=eval_cfg.bootstrap_samples,
                seed=eval_cfg.bootstrap_seed,
            )
            comp_rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "n_paired_maps": len(pairs),
                    "baseline_variant": "mixed_baseline_raw_mse",
                    "baseline_checkpoint": stage1a_selected["checkpoint_name"],
                    "specialized_variant": variant,
                    "specialized_checkpoint": b_row["checkpoint_name"],
                    "baseline_value": b_mean,
                    "specialized_value": s_mean,
                    "absolute_difference": diff,
                    "relative_percentage_difference": rel,
                    "paired_bootstrap_diff_ci95_low": ci_low,
                    "paired_bootstrap_diff_ci95_high": ci_high,
                    "winner": winner,
                }
            )
        out_path = output_root / "03_baseline_vs_specialized" / f"{domain}_ae_baseline_vs_specialized.csv"
        write_csv(out_path, comp_rows)
        rows_written.extend(comp_rows)
    return rows_written


def paired_bootstrap_diff_ci(baseline: list[float], specialized: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not baseline or samples <= 0 or np is None:
        return (float("nan"), float("nan"))
    b = np.array(baseline, dtype=np.float64)
    s = np.array(specialized, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(samples, len(b)))
    diffs = (s[idx] - b[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def create_selected_manifest(
    output_root: Path,
    stage1a_selected: dict[str, Any] | None,
    stage1a_rows: list[dict[str, Any]],
    stage1b_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    expected_stage1a = next(
        (
            r for r in stage1a_rows
            if r.get("recipe", r.get("variant")) == "mixed_baseline_raw_mse"
            and r["checkpoint_name"] == "best_spatial_corr.pt"
        ),
        None,
    )
    selected: dict[str, Any] = {}
    if stage1a_selected:
        selected["mixed_stage1a"] = {
            "checkpoint_path": stage1a_selected["checkpoint_path"],
            "checkpoint_name": stage1a_selected["checkpoint_name"],
            "expected_checkpoint": expected_stage1a["checkpoint_path"] if expected_stage1a else None,
            "empirical_best_checkpoint": stage1a_selected["checkpoint_path"],
            "selection_reason": "Ranked by source-normalized spatial correlation, then source-normalized top-5 Dice, with foreground MSE and reconstruction MSE secondary.",
            "held_out_metrics": {
                "mean_spatial_corr": stage1a_selected.get("mean_spatial_corr"),
                "mean_top5_dice": stage1a_selected.get("mean_top5_dice"),
                "mean_foreground_mse": stage1a_selected.get("mean_foreground_mse"),
                "mean_reconstruction_mse": stage1a_selected.get("mean_reconstruction_mse"),
            },
        }
        if expected_stage1a and expected_stage1a["checkpoint_name"] != stage1a_selected["checkpoint_name"]:
            selected["mixed_stage1a"]["reason_expected_differs"] = "Expected best_spatial_corr.pt was not rank 1 under held-out source-normalized policy."
    for domain, key in [
        ("pubmed", "mixed_to_pubmed_stage1b"),
        ("nilearn", "mixed_to_nilearn_stage1b"),
        ("neurovault", "mixed_to_neurovault_stage1b"),
    ]:
        domain_rows = stage1b_rows.get(domain, [])
        chosen = selected_stage1b_row(domain, domain_rows)
        expected = next((r for r in domain_rows if r["checkpoint_name"] == "best_top5_dice.pt"), None)
        empirical = domain_rows[0] if domain_rows else None
        if not chosen:
            selected[key] = {
                "checkpoint_path": None,
                "selection_reason": "No loadable checkpoint evaluated for this domain.",
                "held_out_metrics": {},
            }
            continue
        selected[key] = {
            "checkpoint_path": chosen["checkpoint_path"],
            "checkpoint_name": chosen["checkpoint_name"],
            "expected_checkpoint": expected["checkpoint_path"] if expected else None,
            "empirical_best_checkpoint": empirical["checkpoint_path"] if empirical else None,
            "selection_reason": "Uses best_top5_dice.pt unless another checkpoint shows a strictly better held-out top-5 Dice.",
            "held_out_metrics": {
                "top5_dice": chosen.get("top5_dice"),
                "spatial_corr": chosen.get("spatial_corr"),
                "foreground_mse": chosen.get("foreground_mse"),
                "reconstruction_mse": chosen.get("reconstruction_mse"),
            },
        }
        if expected and empirical and expected["checkpoint_name"] != empirical["checkpoint_name"]:
            selected[key]["reason_expected_differs"] = "Empirical rank-1 checkpoint differs from expected best_top5_dice.pt; inspect selection CSV before changing Stage 2/3."
    write_json(output_root / "04_final_selection/selected_stage2_checkpoints.json", selected)
    return selected


def create_all_leaderboard(output_root: Path, comparison_rows: list[dict[str, Any]]) -> None:
    rows = [r for r in comparison_rows if r.get("load_status") in {"loaded", "alias_copied"}]
    rows.sort(
        key=lambda r: (
            str(r.get("stage")),
            str(r.get("variant")),
            str(r.get("test_domain")),
            -safe_float(r.get("spatial_corr"), -1),
            -safe_float(r.get("top5_dice"), -1),
        )
    )
    write_csv(output_root / "04_final_selection/all_checkpoint_leaderboard.csv", rows)


def discover_unregistered_variants(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    registered = {Path(v["run_dir"]).expanduser().resolve() for v in registry.values() if v.get("run_dir")}
    roots = set()
    for spec in registry.values():
        run_dir = Path(spec["run_dir"]).expanduser()
        if run_dir.parent.exists():
            roots.add(run_dir.parent)
    rows = []
    for root in roots:
        for ckpt_dir in root.glob("*/checkpoints"):
            run_dir = ckpt_dir.parent.resolve()
            if run_dir in registered:
                continue
            if any((ckpt_dir / name).exists() for name in CHECKPOINT_FILENAMES):
                rows.append({"run_dir": str(run_dir), "checkpoint_dir": str(ckpt_dir), "variant": run_dir.name})
    return rows


def load_model_for_plot(checkpoint_path: Path, run_dir: Path, device: torch.device):
    model, cfg, _ = load_autoencoder_for_eval(checkpoint_path, run_dir, device)
    model.eval()
    return model, cfg


def create_qualitative_plots(
    output_root: Path,
    registry: dict[str, dict[str, Any]],
    selected_manifest: dict[str, Any],
    eval_cfg: EvaluationConfig,
) -> list[dict[str, Any]]:
    if not eval_cfg.make_qualitative_plots:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    test_jsonl = resolve_test_jsonl(eval_cfg)
    datasets = build_test_datasets(test_jsonl)
    device_name = eval_cfg.device
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    manifest_rows = []
    rng = random.Random(eval_cfg.qualitative_seed)
    for domain, stage1b_key, stage1b_variant in [
        ("pubmed", "mixed_to_pubmed_stage1b", "mixed_to_pubmed"),
        ("nilearn", "mixed_to_nilearn_stage1b", "mixed_to_nilearn"),
        ("neurovault", "mixed_to_neurovault_stage1b", "mixed_to_neurovault"),
    ]:
        dataset = datasets[domain]
        if len(dataset) == 0:
            continue
        indices = list(range(len(dataset)))
        chosen = rng.sample(indices, k=min(eval_cfg.qualitative_examples_per_domain, len(indices)))
        model_specs = []
        mixed_sel = selected_manifest.get("mixed_stage1a", {})
        stage1b_sel = selected_manifest.get(stage1b_key, {})
        if mixed_sel.get("checkpoint_path"):
            model_specs.append(("selected_mixed_stage1a", "mixed_baseline_raw_mse", mixed_sel["checkpoint_path"]))
        if stage1b_sel.get("checkpoint_path"):
            model_specs.append(("selected_stage1b", stage1b_variant, stage1b_sel["checkpoint_path"]))
        last_path = checkpoint_dir_for_run(Path(registry[stage1b_variant]["run_dir"]).expanduser()) / "last.pt"
        if last_path.exists() and str(last_path.resolve()) not in {str(Path(p).expanduser().resolve()) for _, _, p in model_specs}:
            model_specs.append(("last_stage1b", stage1b_variant, str(last_path)))
        models = []
        for label, variant, path in model_specs:
            try:
                model, cfg = load_model_for_plot(Path(path).expanduser(), Path(registry[variant]["run_dir"]).expanduser(), device)
                models.append((label, model, cfg))
            except Exception:
                continue
        if not models:
            continue
        collate = VolumeCollator(TARGET_SHAPE)
        out_dir = output_root / "02_stage1b" / domain / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in chosen:
            item = dataset[idx]
            batch = collate([item])
            target = batch["volume"][0, 0].float()
            map_id = batch["map_id"][0]
            manifest_rows.append({"domain": domain, "map_id": map_id, "dataset_index": idx, "seed": eval_cfg.qualitative_seed})
            recons = []
            for label, model, cfg in models:
                loss_cfg = AutoencoderLossConfig.from_config(cfg)
                with torch.inference_mode():
                    raw = model(batch["volume"].to(device))[0, 0].detach().cpu()
                recons.append((label, prediction_for_metrics(raw, loss_cfg)))
            save_example_plots(out_dir, domain, map_id, target, recons, plt)
        for _, model, _ in models:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output_root / "qualitative_example_manifest.csv", manifest_rows)
    return manifest_rows


def save_example_plots(out_dir: Path, domain: str, map_id: str, target: torch.Tensor, recons: list[tuple[str, torch.Tensor]], plt: Any) -> None:
    middle_z = int(target.shape[2] // 2)
    peak = int(target.flatten().argmax().item())
    peak_z = int(torch.unravel_index(torch.tensor(peak), target.shape)[2].item())
    plot_specs = [
        ("middle_slices", lambda v: v[:, :, middle_z]),
        ("peak_centered_slices", lambda v: v[:, :, peak_z]),
        ("maximum_intensity_projections", lambda v: v.max(dim=2).values),
    ]
    for kind, selector in plot_specs:
        panels = [("target", selector(target))]
        panels.extend((label, selector(recon)) for label, recon in recons)
        panels.extend((f"{label}_abs_diff", (selector(recon) - selector(target)).abs()) for label, recon in recons)
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3), squeeze=False)
        for ax, (title, image) in zip(axes[0], panels):
            ax.imshow(image.T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{domain}_{safe_stem(str(map_id))}_{kind}.png", dpi=150)
        plt.close(fig)


def write_readme(output_root: Path) -> None:
    text = """# Stage 1 Checkpoint Evaluation

This directory contains evaluation-only outputs for saved Stage 1A and Stage 1B autoencoder checkpoints.

Inspect these first:

1. `04_final_selection/selected_stage2_checkpoints.json`
2. `01_stage1a/stage1a_recipe_best_checkpoint_comparison.csv`
3. `01_stage1a/stage1a_all_checkpoint_eval.csv`
4. `01_stage1a/mixed_stage1a_checkpoint_selection.csv`
5. `02_stage1b/pubmed/pubmed_stage1b_checkpoint_selection.csv`
6. `02_stage1b/nilearn/nilearn_stage1b_checkpoint_selection.csv`
7. `02_stage1b/neurovault/neurovault_stage1b_checkpoint_selection.csv`
8. `03_baseline_vs_specialized/*_ae_baseline_vs_specialized.csv`
9. `00_metadata/checkpoint_manifest.csv` for aliases, load failures, epochs, and checksums.

Each Stage 1A recipe was first checkpoint-selected on the same held-out split. The table compares the best checkpoint from each recipe.

Ranking policy:

- Stage 1A prioritizes source-normalized spatial correlation, then source-normalized top-5 Dice.
- Foreground MSE and reconstruction MSE are secondary tie-breakers so background-dominated MSE does not control selection.
- `01_stage1a/stage1a_all_checkpoint_eval.csv` keeps the per-recipe checkpoint diagnostics, including last-checkpoint rows when present.
- Stage 1B keeps `best_top5_dice.pt` unless another checkpoint shows strictly better held-out top-5 Dice on the matching domain.

The evaluator runs under `model.eval()` and `torch.inference_mode()`. It does not create optimizers, call backward, modify checkpoint files, or save trained weights.
"""
    path = output_root / "04_final_selection/README_WHAT_TO_LOOK_AT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def print_final_summary(
    output_root: Path,
    registry: dict[str, dict[str, Any]],
    manifest: list[dict[str, Any]],
    load_failures: list[dict[str, Any]],
    selected: dict[str, Any],
    baseline_vs_specialized: list[dict[str, Any]],
) -> None:
    unique = {r.get("model_state_checksum") for r in manifest if r.get("model_state_checksum")}
    aliases = [r for r in manifest if r.get("is_alias")]
    print("\nEvaluation complete")
    print(f"Variants configured: {len(registry)}")
    print(f"Unique checkpoints discovered: {len(unique)}")
    print(f"Aliases detected: {len(aliases)}")
    print(f"Checkpoints that failed to load/evaluate: {len(load_failures)}")
    for key, label in [
        ("mixed_stage1a", "Best Stage 1A checkpoint"),
        ("mixed_to_pubmed_stage1b", "Best PubMed Stage 1B checkpoint"),
        ("mixed_to_nilearn_stage1b", "Best Nilearn Stage 1B checkpoint"),
        ("mixed_to_neurovault_stage1b", "Best NeuroVault Stage 1B checkpoint"),
    ]:
        item = selected.get(key, {})
        print(f"{label}: {item.get('checkpoint_path')}")
    for domain in ["pubmed", "nilearn", "neurovault"]:
        domain_rows = [r for r in baseline_vs_specialized if r.get("domain") == domain and r.get("metric") == "top5_dice"]
        improved = domain_rows and domain_rows[0].get("winner") == "specialized"
        print(f"Stage 1B improved {domain} top-5 Dice: {bool(improved)}")
    print("Selected checkpoint manifest:", output_root / "04_final_selection/selected_stage2_checkpoints.json")
    print("Upload for review:")
    for path in [
        output_root / "00_metadata/checkpoint_manifest.csv",
        output_root / "00_metadata/test_split_fingerprints.json",
        output_root / "01_stage1a/stage1a_recipe_best_checkpoint_comparison.csv",
        output_root / "01_stage1a/stage1a_all_checkpoint_eval.csv",
        output_root / "01_stage1a/mixed_stage1a_checkpoint_selection.csv",
        output_root / "02_stage1b/pubmed/pubmed_stage1b_checkpoint_selection.csv",
        output_root / "02_stage1b/nilearn/nilearn_stage1b_checkpoint_selection.csv",
        output_root / "02_stage1b/neurovault/neurovault_stage1b_checkpoint_selection.csv",
        output_root / "03_baseline_vs_specialized/pubmed_ae_baseline_vs_specialized.csv",
        output_root / "03_baseline_vs_specialized/nilearn_ae_baseline_vs_specialized.csv",
        output_root / "03_baseline_vs_specialized/neurovault_ae_baseline_vs_specialized.csv",
        output_root / "04_final_selection/selected_stage2_checkpoints.json",
        output_root / "04_final_selection/README_WHAT_TO_LOOK_AT.md",
    ]:
        print("-", path)


def run_evaluation(eval_cfg: EvaluationConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = eval_cfg.output_root / f"stage1_checkpoint_evaluation_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metadata_dir = output_root / "00_metadata"
    test_jsonl = resolve_test_jsonl(eval_cfg)
    write_json(metadata_dir / "evaluation_config.json", {"registry": eval_cfg.registry, "test_jsonl": test_jsonl, **eval_cfg.__dict__})
    datasets = build_test_datasets(test_jsonl)
    fingerprints = {}
    count_rows = []
    for domain, dataset in datasets.items():
        fp, rows = split_fingerprint(dataset, domain)
        fingerprints[domain] = fp
        count_rows.extend(rows)
    write_json(metadata_dir / "test_split_fingerprints.json", fingerprints)
    write_csv(metadata_dir / "test_split_source_counts.csv", count_rows)
    unregistered = discover_unregistered_variants(eval_cfg.registry)
    write_csv(metadata_dir / "unregistered_ae_variants.csv", unregistered)
    manifest, manifest_failures = build_checkpoint_manifest(eval_cfg.registry)
    write_csv(metadata_dir / "checkpoint_manifest.csv", manifest)
    comparison_rows, load_failures = run_checkpoint_evaluation(eval_cfg, output_root, manifest, fingerprints)
    ensure_expected_output_files(output_root, eval_cfg.registry)
    load_failures.extend(manifest_failures)
    manifest_for_selection = manifest + [r for r in manifest_failures if r not in manifest]
    stage1a_rows, stage1a_selected = create_stage1a_selection(
        comparison_rows,
        output_root,
        eval_cfg.registry,
        manifest_for_selection,
    )
    stage1b_rows = create_stage1b_selection(comparison_rows, output_root)
    baseline_vs_specialized = create_baseline_vs_specialized(output_root, eval_cfg.registry, stage1a_selected, stage1b_rows, eval_cfg)
    selected = create_selected_manifest(output_root, stage1a_selected, stage1a_rows, stage1b_rows)
    create_all_leaderboard(output_root, comparison_rows)
    create_qualitative_plots(output_root, eval_cfg.registry, selected, eval_cfg)
    write_readme(output_root)
    write_json(
        metadata_dir / "run_status.json",
        {
            "status": "complete",
            "elapsed_sec": time.time() - start,
            "variants_configured": len(eval_cfg.registry),
            "checkpoint_rows": len(manifest),
            "unique_model_states": len({r.get("model_state_checksum") for r in manifest if r.get("model_state_checksum")}),
            "aliases_detected": len([r for r in manifest if r.get("is_alias")]),
            "load_failures": load_failures,
            "unregistered_variants": unregistered,
        },
    )
    print_final_summary(output_root, eval_cfg.registry, manifest, load_failures, selected, baseline_vs_specialized)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=os.environ.get("NEUROVLM_AE_ABLATION_RUN_DIR", ""))
    parser.add_argument("--test-jsonl", default=os.environ.get("NEUROVLM_TEST_JSONL", ""))
    parser.add_argument("--output-root", default=os.environ.get("NEUROVLM_AE_EVAL_OUTPUT_ROOT", "experiments/3dcnn"))
    parser.add_argument("--device", default=os.environ.get("NEUROVLM_DEVICE", "auto"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("NEUROVLM_AE_EVAL_BATCH_SIZE", "32")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NEUROVLM_EVAL_NUM_WORKERS", "0")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-cross-domain-stage1b", action="store_true")
    return parser.parse_args()


def registry_from_root(run_root: str) -> dict[str, dict[str, Any]]:
    registry = json.loads(json.dumps(AE_RUN_REGISTRY))
    root = Path(run_root).expanduser() if run_root else AE_RUN_ROOT
    if str(root):
        replacements = {
            "mixed_to_pubmed": [
                root / "02_stage1b_ae_finetuning/pubmed",
                root / "02_stage1b_domain_finetune/pubmed",
                root / "02_stage1b_domain_finetune/pubmed/mixed_baseline_to_pubmed",
                root / "02_stage1b_domain_finetune/pubmed/mixed_to_pubmed",
            ],
            "mixed_to_nilearn": [
                root / "02_stage1b_ae_finetuning/nilearn",
                root / "02_stage1b_domain_finetune/nilearn",
                root / "02_stage1b_domain_finetune/nilearn/mixed_baseline_to_nilearn",
                root / "02_stage1b_domain_finetune/nilearn/mixed_to_nilearn",
            ],
            "mixed_to_neurovault": [
                root / "02_stage1b_ae_finetuning/neurovault",
                root / "02_stage1b_domain_finetune/neurovault",
                root / "02_stage1b_domain_finetune/neurovault/mixed_baseline_to_neurovault",
                root / "02_stage1b_domain_finetune/neurovault/mixed_to_neurovault",
            ],
        }
        for variant in ["mixed_baseline_raw_mse", "mixed_balanced_raw_mse", "mixed_balanced_hybrid_loss"]:
            registry[variant]["run_dir"] = str(root / "01_stage1_ae_pretraining" / variant)
        for variant, candidates in replacements.items():
            chosen = next((p for p in candidates if p.exists()), candidates[0])
            registry[variant]["run_dir"] = str(chosen)
    return registry


def main() -> None:
    args = parse_args()
    registry = registry_from_root(args.run_root)
    test_jsonl = Path(args.test_jsonl).expanduser() if args.test_jsonl else None
    cfg = EvaluationConfig(
        registry=registry,
        output_root=Path(args.output_root).expanduser(),
        test_jsonl=test_jsonl,
        device=args.device,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        make_qualitative_plots=not args.no_plots,
        evaluate_stage1b_cross_domain=not args.no_cross_domain_stage1b,
    )
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
