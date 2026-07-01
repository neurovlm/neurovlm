#!/usr/bin/env python
"""Train ALE 3D CNN experiments for NeuroVLM.

Examples
--------
Atlas-free default:

    python experiments/train_ale_cnn.py --mode atlas_free --epochs 50 --batch-size 16

DiFuMo-compatible reconstruction from existing NeuroVLM flatmaps:

    python experiments/train_ale_cnn.py --mode difumo_compatible --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[4]
THREEDCNN_DIR = Path(__file__).resolve().parents[2]
for path in [THREEDCNN_DIR, REPO_DIR / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neurovlm.gnn.ale_cnn import ALE3DCNNEncoder, ALEFlatMLPEncoder, ALEResNet3DEncoder, count_parameters
from neurovlm.gnn.ale_dataset import ALEPreprocessConfig, ALEVolumeDataset, build_or_load_ale_cache
from neurovlm.gnn.model import TextProjHead
from neurovlm.loss import InfoNCELoss
from neurovlm.metrics import (
    bidirectional_retrieval_metrics,
    normalized_k_values,
    normalized_recall_curve_auc,
    recall_curve,
    retrieval_ranks,
)

from atlas_free_cnn.pipeline_outputs import select_ae_checkpoint
from atlas_free_cnn.training.datasets import UnifiedMapTextDataset

SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ALE 3D CNN on NeuroVLM PubMed pairs.")
    p.add_argument("--mode", choices=["difumo_compatible", "atlas_free"], default="atlas_free")
    p.add_argument("--model", choices=["ale_3dcnn", "ale_3dcnn_resnet", "ale_flat_mlp"], default="ale_3dcnn")
    p.add_argument("--train-jsonl", default=None, help="Unified multi-source train split JSONL.")
    p.add_argument("--val-jsonl", default=None, help="Unified multi-source validation split JSONL.")
    p.add_argument("--test-jsonl", default=None, help="Unified multi-source test split JSONL.")
    p.add_argument("--text-embedding-cache", default=None, help="Torch cache mapping primary text strings to SPECTER embeddings.")
    p.add_argument("--domain", choices=["pubmed", "nilearn", "neurovault"], default=None, help="Domain filter for unified multi-source JSONLs.")
    p.add_argument("--target-shape", default="36,45,38", help="Target 3D shape for unified JSONL volumes, comma-separated.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batch-size-auto", action="store_true")
    p.add_argument(
        "--batch-size-candidates",
        default="2048,1536,1024,768,512,384,256,192,128,96,64,32,16,8,4",
        help=(
            "Comma-separated microbatch sizes to try when --batch-size-auto is set. "
            "The trainer starts from the largest value and falls back only after "
            "an actual preflight OOM."
        ),
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients across this many microbatches. This reduces "
            "optimizer-step memory pressure, but InfoNCE negatives still come "
            "only from each microbatch unless --batch-size itself is increased."
        ),
    )
    p.add_argument("--lr-cnn", type=float, default=1e-4)
    p.add_argument("--lr-proj", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--monitor-metric", default="paper_recall_curve_auc")

    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--blocks-per-stage", type=int, default=2)
    p.add_argument("--out-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm", choices=["group", "batch", "instance", "none"], default="group")
    p.add_argument("--pooling", choices=["max", "stride"], default="max")
    p.add_argument("--use-dilation", action="store_true")
    p.add_argument("--multi-scale", action="store_true")
    p.add_argument("--global-context", choices=["none", "se", "attention"], default="none")
    p.add_argument("--mlp-hidden-dim", type=int, default=1024)

    p.add_argument("--kernel-fwhm-mm", type=float, default=9.0)
    p.add_argument("--resolution-mm", type=float, default=4.0)
    p.add_argument("--crop-to-brain", dest="crop_to_brain", action="store_true", default=True)
    p.add_argument("--no-crop-to-brain", dest="crop_to_brain", action="store_false")
    p.add_argument("--normalize", choices=["max", "mass", "none"], default="max")
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--cache-file", default=None)
    p.add_argument("--force-rebuild-cache", action="store_true")
    p.add_argument(
        "--build-cache-only",
        action="store_true",
        help="Build or refresh the packed ALE cache, then exit before training.",
    )
    p.add_argument("--max-papers", type=int, default=None, help="Smoke-test subset size.")

    p.add_argument("--encoder-init", choices=["random", "autoencoder_pretrained"], default="random")
    p.add_argument("--autoencoder-checkpoint", default=None)
    p.add_argument(
        "--ae-init-variant",
        default=None,
        help="Named AE initialization variant, e.g. mixed_baseline, mixed_balanced, mixed_hybrid, mixed_to_pubmed, mixed_to_statmaps.",
    )
    p.add_argument(
        "--ae-checkpoint-selection",
        default="best_val_loss",
        choices=["best_val_loss", "best_spatial_corr", "best_top1_dice", "best_top5_dice", "best_foreground_mse", "last"],
    )
    p.add_argument("--ae-ckpt-path", default=None, help="Explicit AE checkpoint path; overrides variant/selection.")
    p.add_argument("--ae-checkpoint-dir", default=None, help="Directory containing Stage 1 AE checkpoint files.")
    p.add_argument("--text-proj-init", choices=["random", "pretrained_infonce"], default="pretrained_infonce")
    p.add_argument("--freeze-text-proj", action="store_true")
    p.add_argument(
        "--strict-controlled-recipe",
        action="store_true",
        help=(
            "Require the final controlled Stage 3 recipe: AE-pretrained encoder, "
            "trainable pretrained InfoNCE text projection, and 384-d shared embedding."
        ),
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)

    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument(
        "--resume-from",
        default=None,
        help="Resume contrastive training from a last.pt checkpoint.",
    )
    p.add_argument("--comparison-file", default="runs/ale_model_comparison.csv")
    p.add_argument("--save-plots", action="store_true", default=False)
    p.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    p.add_argument("--save-plots-during-training", dest="save_plots", action="store_true")
    p.add_argument("--umap", action="store_true", help="Save UMAP/PCA diagnostics.")
    p.add_argument("--run-umap-during-training", dest="umap", action="store_true")
    p.add_argument("--run-covariate-correlations", action="store_true", default=False)
    p.add_argument("--eval-neurovlm-baseline", action="store_true")
    p.add_argument("--semantic-eval", action="store_true", default=False)
    p.add_argument("--eval-resource-dir", default=None)
    p.add_argument("--mesh-json", default=None)
    p.add_argument("--run-train-sanity-eval", action="store_true", default=False)
    p.add_argument("--final-test-eval", dest="final_test_eval", action="store_true", default=True)
    p.add_argument("--no-final-test-eval", dest="final_test_eval", action="store_false")
    p.add_argument("--train-sanity-n", type=int, default=512,
                   help="Evaluate retrieval on this many training examples after training when --run-train-sanity-eval is set.")
    return p.parse_args()


def which_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    import math

    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(ds, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    num_workers = int(args.num_workers)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=num_workers > 0,
    )


def parse_batch_size_candidates(raw: str, n_examples: int) -> list[int]:
    values: list[int] = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Batch-size candidates must be positive, got {value}")
        values.append(min(value, int(n_examples)))
    values = sorted(set(values), reverse=True)
    return [value for value in values if value > 0]


def parse_target_shape(raw: str) -> tuple[int, int, int]:
    values = [int(item.strip()) for item in str(raw).replace("x", ",").split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError(f"--target-shape must have three dimensions, got {raw!r}")
    return tuple(values)


def domain_from_source(value: str) -> str:
    source = str(value or "").lower()
    if source == "pubmed":
        return "pubmed"
    if source == "neurovault" or source.startswith("neurovault:"):
        return "neurovault"
    if source == "nilearn" or source.startswith("nilearn:"):
        return "nilearn"
    return source


def primary_pair(row: dict) -> tuple[str, str]:
    positives = row.get("positive_texts", []) or []
    if not positives:
        return str(row.get("map_id", "")), ""
    first = positives[0]
    text_id = first.get("text_id") or first.get("id") or first.get("text") or ""
    return str(row.get("map_id", "")), str(text_id)


def split_fingerprint(rows: list[dict]) -> str:
    pairs = sorted(primary_pair(row) for row in rows)
    payload = json.dumps(pairs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_source_field(rows: list[dict]) -> str:
    candidates = ["source", "domain", "image_source", "dataset_source"]
    present = [field for field in candidates if any(field in row and row.get(field) for row in rows)]
    if not present:
        raise KeyError(f"Could not find an authoritative source field among {candidates}")
    return present[0]


def filter_rows_for_domain(rows: list[dict], domain: str, source_field: str) -> list[dict]:
    return [row for row in rows if domain_from_source(row.get(source_field, "")) == domain]


def leakage_report(splits: dict[str, list[dict]]) -> dict:
    ids = {name: {primary_pair(row) for row in rows} for name, rows in splits.items()}
    overlaps = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = ids[left] & ids[right]
        overlaps[f"{left}_{right}"] = len(overlap)
    return {"overlap_counts": overlaps, "passed": all(value == 0 for value in overlaps.values())}


def write_domain_dataset_report(args: argparse.Namespace, splits: dict[str, list[dict]], source_field: str) -> dict:
    before_counts = {name: count for name, count in getattr(args, "_domain_rows_before", {}).items()}
    leakage = leakage_report(splits)
    report = {
        "requested_domain": args.domain,
        "detected_source_field": source_field,
        "rows_before_filtering": before_counts,
        "rows_after_filtering": {name: len(rows) for name, rows in splits.items()},
        "train_count": len(splits["train"]),
        "validation_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "unique_map_ids": {name: len({str(row.get("map_id", "")) for row in rows}) for name, rows in splits.items()},
        "unique_text_ids": {name: len({primary_pair(row)[1] for row in rows}) for name, rows in splits.items()},
        "source_value_counts": {
            name: dict(Counter(str(row.get(source_field, "")) for row in rows))
            for name, rows in splits.items()
        },
        "sample_rows": {
            name: [
                {
                    "map_id": row.get("map_id"),
                    "source": row.get(source_field),
                    "positive_text_id": primary_pair(row)[1],
                }
                for row in rows[:3]
            ]
            for name, rows in splits.items()
        },
        "split_fingerprints": {name: split_fingerprint(rows) for name, rows in splits.items()},
        "leakage": leakage,
    }
    if any(len(rows) == 0 for rows in splits.values()):
        raise RuntimeError(f"Domain filter produced an empty split: {report['rows_after_filtering']}")
    if not leakage["passed"]:
        raise RuntimeError(f"Domain split leakage detected: {leakage['overlap_counts']}")
    for split_name, counts in report["source_value_counts"].items():
        bad = [source for source in counts if domain_from_source(source) != args.domain]
        if bad:
            raise RuntimeError(f"{args.domain} {split_name} split contains non-domain sources: {bad}")
    run_dir = Path(args.run_dir)
    with (run_dir / "domain_dataset_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    return report


class UnifiedContrastiveDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        rows: list[dict],
        text_cache: dict[str, torch.Tensor],
        *,
        target_shape: tuple[int, int, int],
    ):
        self.base = UnifiedMapTextDataset(jsonl_path)
        self.base.rows = rows
        self.rows = rows
        self.text_cache = text_cache
        self.target_shape = target_shape
        self.input_shape = target_shape
        self.pmids = np.array([str(row.get("pmid", row.get("map_id", ""))) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        positives = item.get("positive_texts", []) or []
        if not positives:
            raise ValueError(f"Row {item.get('map_id')} has no positive_texts")
        text = positives[0]["text"]
        if text not in self.text_cache:
            text_id = positives[0].get("text_id", "")
            raise KeyError(f"Primary text missing from embedding cache for map_id={item.get('map_id')} text_id={text_id}")
        volume = item["volume"].float()
        if tuple(volume.shape[-3:]) != self.target_shape:
            volume = F.interpolate(
                volume.unsqueeze(0),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return {
            "volume": volume.clamp(0.0, 1.0),
            "text": self.text_cache[text].float(),
            "paper_idx": torch.tensor(idx, dtype=torch.long),
            "map_id": item["map_id"],
            "text_id": positives[0].get("text_id", ""),
        }

    def split_by_index(self, train_idx, val_idx, test_idx):
        raise NotImplementedError("UnifiedContrastiveDataset receives fixed split JSONLs; do not random-split it.")

    def covariate_frame(self) -> pd.DataFrame:
        rows = []
        for idx, row in enumerate(self.rows):
            rows.append(
                {
                    "sample_pos": idx,
                    "map_id": row.get("map_id", ""),
                    "source": row.get("source", ""),
                    "source_detail": row.get("source_detail", ""),
                    "quality_score": row.get("quality_score", ""),
                }
            )
        return pd.DataFrame(rows)


def load_text_embedding_cache(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ["embedding_by_text", "processed_embedding_by_text", "text_to_embedding", "embeddings_by_text"]:
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
        else:
            if torch.is_tensor(payload.get("embeddings")) and isinstance(payload.get("texts"), list):
                embeddings = payload["embeddings"].float().cpu()
                texts = [str(text) for text in payload["texts"]]
                payload = {text: embeddings[i] for i, text in enumerate(texts)}
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported text embedding cache format in {path}")
    return {
        str(key): torch.as_tensor(value, dtype=torch.float32).flatten()
        for key, value in payload.items()
        if torch.is_tensor(value) or isinstance(value, (list, tuple))
    }


def build_unified_dataset(args: argparse.Namespace):
    required = {
        "--train-jsonl": args.train_jsonl,
        "--val-jsonl": args.val_jsonl,
        "--test-jsonl": args.test_jsonl,
        "--text-embedding-cache": args.text_embedding_cache,
        "--domain": args.domain,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Unified multi-source training requires {', '.join(missing)}")

    split_paths = {"train": Path(args.train_jsonl), "val": Path(args.val_jsonl), "test": Path(args.test_jsonl)}
    raw_rows = {}
    for name, path in split_paths.items():
        with path.open() as f:
            raw_rows[name] = [json.loads(line) for line in f if line.strip()]
    source_field = detect_source_field([row for rows in raw_rows.values() for row in rows[:100]])
    args._domain_rows_before = {name: len(rows) for name, rows in raw_rows.items()}
    filtered = {name: filter_rows_for_domain(rows, args.domain, source_field) for name, rows in raw_rows.items()}
    domain_report = write_domain_dataset_report(args, filtered, source_field)
    text_cache = load_text_embedding_cache(args.text_embedding_cache)
    target_shape = parse_target_shape(args.target_shape)
    train_ds = UnifiedContrastiveDataset(split_paths["train"], filtered["train"], text_cache, target_shape=target_shape)
    val_ds = UnifiedContrastiveDataset(split_paths["val"], filtered["val"], text_cache, target_shape=target_shape)
    test_ds = UnifiedContrastiveDataset(split_paths["test"], filtered["test"], text_cache, target_shape=target_shape)
    train_ds.domain_report = domain_report
    payload = {
        "config": {
            "dataset_kind": "unified_jsonl",
            "train_jsonl": str(split_paths["train"]),
            "val_jsonl": str(split_paths["val"]),
            "test_jsonl": str(split_paths["test"]),
            "text_embedding_cache": str(args.text_embedding_cache),
            "domain": args.domain,
            "source_field": source_field,
        },
        "metadata": {
            "shape": target_shape,
            "n_volumes": sum(len(rows) for rows in filtered.values()),
            "domain_report": domain_report,
        },
    }
    return train_ds, train_ds, val_ds, test_ds, payload, payload["config"]


def build_dataset(args: argparse.Namespace):
    if args.train_jsonl or args.val_jsonl or args.test_jsonl or args.domain:
        return build_unified_dataset(args)

    print("Preparing ALE preprocessing config ...", flush=True)
    config = ALEPreprocessConfig(
        mode=args.mode,
        kernel_fwhm_mm=args.kernel_fwhm_mm,
        resolution_mm=args.resolution_mm,
        crop_to_brain=args.crop_to_brain,
        normalize=args.normalize,
        clamp=not args.no_clamp,
        cache_dtype=args.cache_dtype,
        max_papers=args.max_papers,
    )
    if args.cache_file is None:
        args.cache_file = default_cache_file(args)
    payload = build_or_load_ale_cache(
        args.cache_file, config, force_rebuild=args.force_rebuild_cache
    )
    print("Constructing aligned ALE dataset ...", flush=True)
    ds = ALEVolumeDataset.from_cache(payload)
    print("Creating train/val/test split ...", flush=True)
    try:
        from neurovlm.data import load_dataset
        from neurovlm.semantic_evaluation import official_split_positions

        pubmed_df = load_dataset("pubmed_text")
        split_pos = official_split_positions(
            pubmed_df,
            ds.pmids,
            out_dir=args.run_dir,
            random_state=args.seed,
            random_val_frac=args.val_frac,
            random_test_frac=args.test_frac,
        )
        train_ds, val_ds, test_ds = ds.split_by_index(
            split_pos["train"].tolist(),
            split_pos["val"].tolist(),
            split_pos["test"].tolist(),
        )
        print("Using PubMed dataframe split columns for train/val/test.", flush=True)
    except Exception as exc:
        print(f"WARNING: official PubMed split failed ({exc}); falling back to random split.", flush=True)
        train_ds, val_ds, test_ds = ds.split(args.val_frac, args.test_frac, seed=args.seed)
    return ds, train_ds, val_ds, test_ds, payload, config


def default_cache_file(args: argparse.Namespace) -> str:
    name = (
        f"{args.mode}_ale_{int(args.resolution_mm)}mm_"
        f"fwhm{str(args.kernel_fwhm_mm).replace('.', 'p')}_"
        f"{'crop' if args.crop_to_brain else 'full'}_{args.cache_dtype}.pt"
    )
    return str(Path("data/ale_caches") / name)


def build_cache_only(args: argparse.Namespace) -> None:
    config = ALEPreprocessConfig(
        mode=args.mode,
        kernel_fwhm_mm=args.kernel_fwhm_mm,
        resolution_mm=args.resolution_mm,
        crop_to_brain=args.crop_to_brain,
        normalize=args.normalize,
        clamp=not args.no_clamp,
        cache_dtype=args.cache_dtype,
        max_papers=args.max_papers,
    )
    if args.cache_file is None:
        args.cache_file = default_cache_file(args)
    payload = build_or_load_ale_cache(
        args.cache_file,
        config,
        force_rebuild=args.force_rebuild_cache,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "cache_metadata.json").open("w") as f:
        json.dump(
            {
                "cache_file": args.cache_file,
                "config": payload.get("config", {}),
                "metadata": payload.get("metadata", {}),
            },
            f,
            indent=2,
        )
    print("\nCache build complete.")
    print(f"  cache_file: {args.cache_file}")
    print(f"  n_volumes : {payload['metadata']['n_volumes']:,}")
    print(f"  shape     : {tuple(payload['metadata']['shape'])}")
    print(f"  metadata  : {run_dir / 'cache_metadata.json'}")


PREVIOUS_PUBMED_REFERENCE_ARCHITECTURE = {
    "model": "ale_3dcnn",
    "base_channels": 48,
    "num_blocks": 4,
    "out_dim": 384,
    "dropout": 0.1,
    "norm": "group",
    "pooling": "max",
    "encoder_arch": "plain",
    "blocks_per_stage": 2,
    "use_dilation": False,
    "multi_scale": False,
    "global_context": "none",
    "target_shape": [36, 45, 38],
}


CONTROLLED_MULTISOURCE_ARCHITECTURE = {
    **PREVIOUS_PUBMED_REFERENCE_ARCHITECTURE,
    "base_channels": 64,
}


def nested_get(mapping: dict, *keys, default=None):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_checkpoint_architecture(payload: dict) -> dict:
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    target_shape = (
        payload.get("target_shape")
        or cfg.get("target_shape")
        or nested_get(payload, "model_architecture", "target_shape")
        or CONTROLLED_MULTISOURCE_ARCHITECTURE["target_shape"]
    )
    if isinstance(target_shape, tuple):
        target_shape = list(target_shape)
    return {
        "model": cfg.get("model_name", "ale_3dcnn"),
        "base_channels": model_cfg.get("base_channels", nested_get(payload, "model_architecture", "base_channels")),
        "num_blocks": model_cfg.get("num_blocks", nested_get(payload, "model_architecture", "num_blocks")),
        "latent_dim": model_cfg.get("latent_dim", payload.get("latent_dim")),
        "out_dim": model_cfg.get("latent_dim", payload.get("latent_dim")),
        "dropout": model_cfg.get("dropout", nested_get(payload, "model_architecture", "dropout")),
        "norm": model_cfg.get("norm", nested_get(payload, "model_architecture", "norm")),
        "pooling": model_cfg.get("pooling", nested_get(payload, "model_architecture", "pooling")),
        "encoder_arch": model_cfg.get("encoder_arch", nested_get(payload, "model_architecture", "encoder_arch", default="plain")),
        "blocks_per_stage": model_cfg.get("blocks_per_stage", nested_get(payload, "model_architecture", "blocks_per_stage", default=2)),
        "use_dilation": model_cfg.get("use_dilation", nested_get(payload, "model_architecture", "use_dilation", default=False)),
        "multi_scale": model_cfg.get("multi_scale", nested_get(payload, "model_architecture", "multi_scale", default=False)),
        "global_context": model_cfg.get("global_context", nested_get(payload, "model_architecture", "global_context", default="none")),
        "target_shape": target_shape,
    }


def stage3_architecture_from_args(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "base_channels": args.base_channels,
        "num_blocks": args.num_blocks,
        "out_dim": args.out_dim,
        "dropout": args.dropout,
        "norm": args.norm,
        "pooling": args.pooling,
        "encoder_arch": "plain" if args.model == "ale_3dcnn" else args.model,
        "blocks_per_stage": args.blocks_per_stage,
        "use_dilation": args.use_dilation,
        "multi_scale": args.multi_scale,
        "global_context": args.global_context,
        "target_shape": list(parse_target_shape(args.target_shape)) if args.train_jsonl else list(CONTROLLED_MULTISOURCE_ARCHITECTURE["target_shape"]),
    }


def compare_architecture(selected: dict, instantiated: dict) -> tuple[dict, dict]:
    expected_differences = {
        "base_channels": {
            "previous_pubmed": 48,
            "multisource_required": 64,
            "reason": "Selected mixed-source Stage 1 AE checkpoints were trained with base_channels=64.",
        }
    }
    unexpected = {}
    for key, expected_value in CONTROLLED_MULTISOURCE_ARCHITECTURE.items():
        selected_value = selected.get("latent_dim") if key == "out_dim" else selected.get(key)
        instantiated_value = instantiated.get(key)
        if key == "base_channels":
            if selected_value != 64 or instantiated_value != 64:
                unexpected[key] = {"selected_checkpoint": selected_value, "instantiated_stage3": instantiated_value, "expected": 64}
            continue
        if selected_value is not None and selected_value != expected_value:
            unexpected[f"selected_{key}"] = {"observed": selected_value, "expected": expected_value}
        if instantiated_value != expected_value:
            unexpected[f"stage3_{key}"] = {"observed": instantiated_value, "expected": expected_value}
    return expected_differences, unexpected


def encoder_state_from_autoencoder_payload(payload: dict) -> dict:
    state = (
        payload.get("encoder")
        or payload.get("brain_encoder")
        or payload.get("model", {})
    )
    if state and any(str(k).startswith("encoder.") for k in state):
        state = {str(k).removeprefix("encoder."): v for k, v in state.items() if str(k).startswith("encoder.")}
    if not state:
        raise KeyError(
            "Autoencoder checkpoint must contain 'encoder', 'brain_encoder', "
            "or a 'model' state_dict with encoder.* keys"
        )
    return state


def _load_encoder_from_autoencoder_checkpoint(
    brain_encoder: nn.Module,
    checkpoint_path: str | None,
    args: argparse.Namespace,
) -> dict:
    if not checkpoint_path:
        raise ValueError("--encoder-init autoencoder_pretrained requires --autoencoder-checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = encoder_state_from_autoencoder_payload(payload)
    model_state = brain_encoder.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatches = {
        key: {"checkpoint": list(value.shape), "model": list(model_state[key].shape)}
        for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    }
    selected_arch = extract_checkpoint_architecture(payload)
    instantiated_arch = stage3_architecture_from_args(args)
    expected_differences, unexpected_differences = compare_architecture(selected_arch, instantiated_arch)
    report = {
        "previous_pubmed_reference_architecture": PREVIOUS_PUBMED_REFERENCE_ARCHITECTURE,
        "selected_ae_checkpoint_architecture": selected_arch,
        "instantiated_stage3_architecture": instantiated_arch,
        "expected_differences": expected_differences,
        "unexpected_differences": unexpected_differences,
        "state_dict_missing_keys": missing,
        "state_dict_unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "strict_load_success": False,
        "final_compatibility_status": "failed",
    }
    if unexpected_differences or missing or unexpected or shape_mismatches:
        with (Path(args.run_dir) / "architecture_compatibility_report.json").open("w") as f:
            json.dump(report, f, indent=2, default=str)
        raise RuntimeError(f"Stage 3 architecture compatibility failed: {json.dumps(report, indent=2, default=str)}")
    brain_encoder.load_state_dict(state, strict=True)
    report["strict_load_success"] = True
    report["final_compatibility_status"] = "passed"
    with (Path(args.run_dir) / "architecture_compatibility_report.json").open("w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def resolve_autoencoder_initialization(args: argparse.Namespace) -> dict:
    if args.encoder_init != "autoencoder_pretrained":
        return {"encoder_init": args.encoder_init, "ae_checkpoint_path": None}
    explicit = args.ae_ckpt_path or args.autoencoder_checkpoint
    selection = args.ae_checkpoint_selection
    if explicit:
        selected = select_ae_checkpoint(explicit_path=explicit, selection=selection)
    else:
        if not args.ae_checkpoint_dir:
            raise ValueError(
                "--encoder-init autoencoder_pretrained requires --autoencoder-checkpoint, "
                "--ae-ckpt-path, or --ae-checkpoint-dir"
            )
        selected = select_ae_checkpoint(
            checkpoint_dir=args.ae_checkpoint_dir,
            selection=selection,
        )
    if not selected["exists"]:
        raise FileNotFoundError(f"Selected AE checkpoint does not exist: {selected['ae_checkpoint_path']}")
    args.autoencoder_checkpoint = selected["ae_checkpoint_path"]
    payload = torch.load(args.autoencoder_checkpoint, map_location="cpu", weights_only=False)
    report = {
        "encoder_init": args.encoder_init,
        "ae_init_variant": args.ae_init_variant or payload.get("ae_variant") or payload.get("config", {}).get("ae_variant", ""),
        "ae_checkpoint_selection": selected["ae_checkpoint_selection"],
        "ae_checkpoint_path": args.autoencoder_checkpoint,
        "ae_checkpoint_epoch": payload.get("epoch"),
        "ae_training_config": payload.get("config", {}),
        "ae_source_wise_reconstruction_metrics": payload.get("source_wise_validation_metrics", {}),
        "ae_validation_metrics": payload.get("validation_metrics", payload.get("metrics", {})),
    }
    print("AE encoder initialization:", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in {"ae_training_config"}}, indent=2, default=str), flush=True)
    return report


def build_model(args: argparse.Namespace, input_shape: tuple[int, ...]):
    print("Building ALE model ...", flush=True)
    if args.model == "ale_3dcnn":
        brain_encoder = ALE3DCNNEncoder(
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            out_dim=args.out_dim,
            dropout=args.dropout,
            norm=args.norm,
            pooling=args.pooling,
        )
    elif args.model == "ale_3dcnn_resnet":
        brain_encoder = ALEResNet3DEncoder(
            base_channels=args.base_channels,
            num_stages=args.num_blocks,
            blocks_per_stage=args.blocks_per_stage,
            out_dim=args.out_dim,
            dropout=args.dropout,
            norm=args.norm,
            use_dilation=args.use_dilation,
            multi_scale=args.multi_scale,
            global_context=args.global_context,
        )
    else:
        brain_encoder = ALEFlatMLPEncoder(
            hidden_dim=args.mlp_hidden_dim,
            out_dim=args.out_dim,
            dropout=args.dropout,
        )
    architecture_report = {}
    if args.encoder_init == "autoencoder_pretrained":
        if args.model == "ale_flat_mlp":
            raise ValueError("autoencoder_pretrained is only valid for CNN models")
        architecture_report = _load_encoder_from_autoencoder_checkpoint(brain_encoder, args.autoencoder_checkpoint, args)

    if args.text_proj_init == "pretrained_infonce":
        if args.out_dim != 384:
            raise ValueError("pretrained_infonce text projector requires --out-dim 384")
        from neurovlm.models import ProjHead

        text_proj = ProjHead.from_pretrained("text_infonce")
    else:
        text_proj = TextProjHead(in_dim=768, hidden_dim=512, out_dim=args.out_dim)

    with torch.no_grad():
        dummy = torch.zeros(2, 1, *input_shape)
        out = brain_encoder(dummy)
        assert out.shape == (2, args.out_dim), out.shape
    return brain_encoder, text_proj, architecture_report


class ALETrainer:
    def __init__(
        self,
        brain_encoder: nn.Module,
        text_proj: nn.Module,
        args: argparse.Namespace,
        device: torch.device,
    ):
        self.brain_encoder = brain_encoder.to(device)
        self.text_proj = text_proj.to(device)
        self.args = args
        self.device = device
        self.loss_fn = InfoNCELoss(args.temperature)
        if args.freeze_text_proj:
            for p in self.text_proj.parameters():
                p.requires_grad_(False)
        groups = [{"params": self.brain_encoder.parameters(), "lr": args.lr_cnn}]
        if not args.freeze_text_proj:
            groups.append({"params": self.text_proj.parameters(), "lr": args.lr_proj})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.grad_accum_steps = max(1, int(args.grad_accum_steps))
        self.use_amp = args.amp and device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.best_state: Optional[dict] = None
        self.best_score = -float("inf")
        self._printed_contrastive_sanity = False
        self.preflight_peak_vram_mb = 0.0
        self.history: dict[str, list] = {
            "train_loss": [],
            "epoch_time_sec": [],
            "peak_vram_mb": [],
            "lr_brain": [],
            "lr_proj": [],
            "val_epoch": [],
        }
        self.timing_profile: dict[str, object] = {
            "data_cache_load_sec": 0.0,
            "preflight_batch_size_sec": 0.0,
            "train_epoch_sec": [],
            "validation_eval_sec": [],
            "generation_auc_eval_sec": [],
            "artifact_save_sec": [],
            "final_eval_sec": [],
            "total_branch_sec": 0.0,
        }

        n_text_trainable = sum(p.numel() for p in self.text_proj.parameters() if p.requires_grad)
        print(
            "Contrastive objective: symmetric InfoNCE over projected ALE-brain "
            "and projected SPECTER-text embeddings (not reconstruction/MSE).",
            flush=True,
        )
        print(
            f"text_proj trainable={n_text_trainable > 0} "
            f"trainable_text_params={n_text_trainable:,}",
            flush=True,
        )
        if self.grad_accum_steps > 1:
            print(
                f"Gradient accumulation enabled: grad_accum_steps={self.grad_accum_steps}. "
                "Optimizer sees a larger effective batch for gradients, but InfoNCE "
                "in-batch negatives are still limited to each microbatch. Increase "
                "--batch-size for more contrastive negatives.",
                flush=True,
            )

    def _load_resume_payload(self) -> Optional[dict]:
        resume_from = getattr(self.args, "resume_from", None)
        if not resume_from:
            return None
        path = Path(resume_from)
        if not path.exists():
            raise FileNotFoundError(f"--resume-from checkpoint does not exist: {path}")
        return torch.load(path, map_location=self.device, weights_only=False)

    def load_resume_model_weights(self) -> None:
        payload = self._load_resume_payload()
        if not payload:
            return
        if "brain_encoder" not in payload or "text_proj" not in payload:
            raise KeyError("--resume-from checkpoint must contain brain_encoder and text_proj")
        self.brain_encoder.load_state_dict(payload["brain_encoder"], strict=True)
        self.text_proj.load_state_dict(payload["text_proj"], strict=True)
        self.history = payload.get("history", self.history)
        self.best_state = payload.get("best_state", self.best_state)
        self.best_score = float(payload.get("best_score", self.best_score))
        print(
            f"Loaded contrastive resume weights from {self.args.resume_from} "
            f"(last completed epoch={payload.get('epoch', 'unknown')}).",
            flush=True,
        )

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        volume = batch["volume"].to(self.device, non_blocking=self.args.pin_memory)
        text = batch["text"].to(self.device, non_blocking=self.args.pin_memory)
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                brain = self.brain_encoder(volume)
                text_emb = self.text_proj(text)
        else:
            volume = volume.float()
            brain = self.brain_encoder(volume)
            text_emb = self.text_proj(text)
        return brain, text_emb

    @torch.no_grad()
    def _print_contrastive_sanity(
        self,
        brain_emb: torch.Tensor,
        text_emb: torch.Tensor,
        batch: dict,
    ) -> None:
        """Print one explicit contrastive-batch alignment check.

        The evaluation/query matrix is normalized projected text embeddings
        multiplied by normalized CNN brain embeddings. Because each dataset
        item returns the matched text and brain example together, diagonal
        entries are the true pairs even when the DataLoader shuffles examples.
        """
        if self._printed_contrastive_sanity:
            return
        text_n = F.normalize(text_emb.float(), dim=1, eps=1e-8)
        brain_n = F.normalize(brain_emb.float(), dim=1, eps=1e-8)
        sim_text_brain = text_n @ brain_n.T
        assert sim_text_brain.shape[0] == sim_text_brain.shape[1], (
            "InfoNCE requires square batch similarity matrices with paired "
            "text/brain rows on the diagonal."
        )
        diag = sim_text_brain.diag()
        paper_idx = batch.get("paper_idx")
        if torch.is_tensor(paper_idx):
            paper_idx = paper_idx[: min(5, len(paper_idx))].cpu().tolist()
        print("First-batch contrastive sanity check:", flush=True)
        print(
            f"  projected_text shape={tuple(text_emb.shape)} "
            f"projected_brain shape={tuple(brain_emb.shape)}",
            flush=True,
        )
        print(
            "  similarity matrix = normalize(text_proj(SPECTER)) @ "
            "normalize(brain_encoder(ALE)).T",
            flush=True,
        )
        print(f"  sim shape={tuple(sim_text_brain.shape)}", flush=True)
        print(
            "  diagonal entries are true paired text-brain examples from the "
            "same dataset row",
            flush=True,
        )
        print(
            f"  first paper_idx values={paper_idx} "
            f"diag cosine mean={float(diag.mean()):.4f}",
            flush=True,
        )
        self._printed_contrastive_sanity = True

    def preflight_batch_size(self, train_ds) -> None:
        """Select the largest real contrastive batch that fits on this device."""

        if not self.args.batch_size_auto:
            return
        candidates = parse_batch_size_candidates(self.args.batch_size_candidates, len(train_ds))
        if not candidates:
            raise ValueError("--batch-size-auto needs at least one positive --batch-size-candidates value")
        if self.device.type != "cuda":
            self.args.batch_size = min(self.args.batch_size, len(train_ds))
            print(
                f"Batch-size auto-preflight is CUDA-only; using batch_size={self.args.batch_size}.",
                flush=True,
            )
            return

        print(
            "Preflighting contrastive batch size on CUDA "
            f"(largest first): {','.join(str(v) for v in candidates)}",
            flush=True,
        )
        self.brain_encoder.train()
        self.text_proj.train()
        self.optimizer.zero_grad(set_to_none=True)
        last_oom = None
        for candidate in candidates:
            loader = DataLoader(
                train_ds,
                batch_size=candidate,
                shuffle=True,
                num_workers=0,
                pin_memory=self.args.pin_memory,
            )
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(self.device)
                batch = next(iter(loader))
                brain, text = self._forward(batch)
                loss = self.loss_fn(brain, text)
                loss.backward()
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    self.preflight_peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
                    torch.cuda.empty_cache()
                self.args.batch_size = int(candidate)
                setattr(self.args, "selected_batch_size", int(candidate))
                setattr(self.args, "preflight_peak_vram_mb", float(self.preflight_peak_vram_mb))
                print(
                    f"Selected contrastive batch_size={candidate} "
                    f"(preflight peak VRAM {self.preflight_peak_vram_mb:.0f}MB).",
                    flush=True,
                )
                return
            except RuntimeError as exc:
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                message = str(exc).lower()
                if "out of memory" not in message and "cuda error" not in message:
                    raise
                last_oom = exc
                print(f"  batch_size={candidate} did not fit; trying smaller.", flush=True)
                continue
        raise RuntimeError(
            "No --batch-size-candidates value fit during CUDA preflight."
        ) from last_oom

    def fit(self, train_ds, val_ds) -> None:
        loader = make_loader(train_ds, self.args, shuffle=True)
        steps_per_epoch = max(1, int(np.ceil(len(loader) / self.grad_accum_steps)))
        total_steps = max(1, steps_per_epoch * self.args.epochs)
        warmup_steps = max(1, steps_per_epoch * self.args.warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_warmup(step, warmup_steps, total_steps),
        )

        bad_checks = 0
        start_epoch = 0
        resume_payload = self._load_resume_payload()
        if resume_payload:
            if "optimizer" in resume_payload:
                self.optimizer.load_state_dict(resume_payload["optimizer"])
            if "scheduler" in resume_payload and resume_payload["scheduler"] is not None:
                self.scheduler.load_state_dict(resume_payload["scheduler"])
            if "scaler" in resume_payload and resume_payload["scaler"] is not None:
                self.scaler.load_state_dict(resume_payload["scaler"])
            self.history = resume_payload.get("history", self.history)
            self.best_state = resume_payload.get("best_state", self.best_state)
            self.best_score = float(resume_payload.get("best_score", self.best_score))
            bad_checks = int(resume_payload.get("bad_checks", 0))
            start_epoch = int(resume_payload.get("epoch", -1)) + 1
            print(
                f"Resuming contrastive training at epoch {start_epoch}/{self.args.epochs}.",
                flush=True,
            )

        last_completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, self.args.epochs):
            should_stop = False
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
            losses = []
            self.brain_encoder.train()
            self.text_proj.train()
            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                brain, text = self._forward(batch)
                self._print_contrastive_sanity(brain, text, batch)
                # Symmetric InfoNCE. The loss normalizes both sides internally,
                # builds a batch x batch cosine-similarity matrix over matched
                # projected brain/text pairs, and optimizes both directions.
                loss = self.loss_fn(brain, text)
                loss_for_backward = loss / float(self.grad_accum_steps)
                self.scaler.scale(loss_for_backward).backward()
                should_step = (
                    (step + 1) % self.grad_accum_steps == 0
                    or (step + 1) == len(loader)
                )
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.brain_encoder.parameters()) + list(self.text_proj.parameters()),
                        max_norm=1.0,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                losses.append(float(loss.detach().cpu()))

            epoch_time = time.perf_counter() - start
            peak_vram = (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
                if self.device.type == "cuda"
                else 0.0
            )
            self.history["train_loss"].append(float(np.mean(losses)))
            self.history["epoch_time_sec"].append(float(epoch_time))
            self.timing_profile["train_epoch_sec"].append({"epoch": int(epoch), "seconds": float(epoch_time)})
            print(f"[timing] train_epoch epoch={epoch} seconds={epoch_time:.2f}", flush=True)
            self.history["peak_vram_mb"].append(float(peak_vram))
            self.history["lr_brain"].append(float(self.optimizer.param_groups[0]["lr"]))
            self.history["lr_proj"].append(
                0.0 if self.args.freeze_text_proj else float(self.optimizer.param_groups[1]["lr"])
            )

            if epoch % self.args.val_interval == 0:
                val_start = time.perf_counter()
                metrics, _, _ = self.evaluate(val_ds)
                val_time = time.perf_counter() - val_start
                self.timing_profile["validation_eval_sec"].append({"epoch": int(epoch), "seconds": float(val_time)})
                print(f"[timing] validation_eval epoch={epoch} seconds={val_time:.2f}", flush=True)
                self.history["val_epoch"].append(epoch)
                for k, v in metrics.items():
                    self.history.setdefault(f"val_{k}", []).append(float(v))
                score = float(metrics[self.args.monitor_metric])
                print(
                    f"Epoch {epoch:03d}/{self.args.epochs} "
                    f"loss={self.history['train_loss'][-1]:.4f} "
                    f"val_paper_recall_curve_auc={metrics['paper_recall_curve_auc']:.4f} "
                    f"val_r@10={metrics['mean_recall@10']:.4f} "
                    f"time={epoch_time:.1f}s vram={peak_vram:.0f}MB"
                )
                if score > self.best_score:
                    self.best_score = score
                    self.best_state = {
                        "brain_encoder": deepcopy(self.brain_encoder.state_dict()),
                        "text_proj": deepcopy(self.text_proj.state_dict()),
                        "epoch": epoch,
                        "metrics": metrics,
                        "config": vars(self.args),
                    }
                    bad_checks = 0
                else:
                    bad_checks += 1
                    if (
                        self.args.early_stopping_patience is not None
                        and bad_checks >= self.args.early_stopping_patience
                    ):
                        print(f"Early stopping after {bad_checks} validation checks.")
                        should_stop = True
            else:
                print(
                    f"Epoch {epoch:03d}/{self.args.epochs} "
                    f"loss={self.history['train_loss'][-1]:.4f} time={epoch_time:.1f}s"
                )
            last_completed_epoch = epoch
            save_start = time.perf_counter()
            self.save_last(epoch=epoch, bad_checks=bad_checks)
            save_time = time.perf_counter() - save_start
            self.timing_profile["artifact_save_sec"].append({"epoch": int(epoch), "artifact": "last_checkpoint", "seconds": float(save_time)})
            print(f"[timing] artifact_save epoch={epoch} artifact=last_checkpoint seconds={save_time:.2f}", flush=True)
            if should_stop:
                break

        save_start = time.perf_counter()
        self.save_best()
        self.timing_profile["artifact_save_sec"].append({"artifact": "best_checkpoint", "seconds": float(time.perf_counter() - save_start)})
        save_start = time.perf_counter()
        self.save_last(epoch=last_completed_epoch, bad_checks=bad_checks)
        self.timing_profile["artifact_save_sec"].append({"artifact": "final_last_checkpoint", "seconds": float(time.perf_counter() - save_start)})

    @torch.no_grad()
    def collect_embeddings(self, ds) -> tuple[torch.Tensor, torch.Tensor]:
        self.brain_encoder.eval()
        self.text_proj.eval()
        loader = make_loader(ds, self.args, shuffle=False)
        all_brain, all_text = [], []
        for batch in loader:
            brain, text = self._forward(batch)
            all_brain.append(brain.float().cpu())
            all_text.append(text.float().cpu())
        return torch.cat(all_brain), torch.cat(all_text)

    @torch.no_grad()
    def evaluate(self, ds) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        # Evaluation ranks projected SPECTER embeddings against projected ALE
        # embeddings: text_proj(SPECTER) vs brain_encoder(ALE volume), never
        # raw SPECTER vectors.
        brain, text = self.collect_embeddings(ds)
        metrics = bidirectional_retrieval_metrics(text, brain)
        metrics["paper_recall_curve_auc"] = metrics["mean_normalized_k_recall_curve_auc"]
        metrics["normalized_k_recall_curve_auc"] = metrics["mean_normalized_k_recall_curve_auc"]
        metrics["full_recall_curve_auc_k1_to_N"] = metrics["paper_recall_curve_auc"]
        metrics["random_recall@10"] = metrics["mean_random_recall@10"]
        loss = self.loss_fn(brain, text)
        metrics["loss"] = float(loss.item())
        return metrics, brain, text

    def restore_best(self) -> None:
        if self.best_state is None:
            raise RuntimeError("No best checkpoint available.")
        self.brain_encoder.load_state_dict(self.best_state["brain_encoder"])
        self.text_proj.load_state_dict(self.best_state["text_proj"])

    def save_best(self) -> None:
        if self.best_state is None:
            return
        ckpt_dir = Path(self.args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        canonical = ckpt_dir / "best_val_normalized_recall_auc.pt"
        torch.save(self.best_state, canonical)
        if SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES:
            shutil.copy2(canonical, ckpt_dir / "best_ale_cnn.pt")

    def save_last(self, epoch: Optional[int] = None, bad_checks: int = 0) -> None:
        ckpt_dir = Path(self.args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "last.pt"
        torch.save(
            {
                "brain_encoder": self.brain_encoder.state_dict(),
                "text_proj": self.text_proj.state_dict(),
                "history": self.history,
                "best_state": self.best_state,
                "best_score": self.best_score,
                "bad_checks": bad_checks,
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict(),
                "config": vars(self.args),
            },
            path,
        )


def trainability_report(trainer: ALETrainer, args: argparse.Namespace) -> dict:
    brain_trainable = sum(p.numel() for p in trainer.brain_encoder.parameters() if p.requires_grad)
    brain_frozen = sum(p.numel() for p in trainer.brain_encoder.parameters() if not p.requires_grad)
    text_trainable = sum(p.numel() for p in trainer.text_proj.parameters() if p.requires_grad)
    text_frozen = sum(p.numel() for p in trainer.text_proj.parameters() if not p.requires_grad)
    report = {
        "encoder_init": args.encoder_init,
        "cnn_encoder_trainable": brain_trainable > 0,
        "text_proj_init": args.text_proj_init,
        "text_projection_pretrained": args.text_proj_init == "pretrained_infonce",
        "text_projection_trainable": text_trainable > 0,
        "loss": "symmetric InfoNCE",
        "shared_embedding_dimension": args.out_dim,
        "strict_controlled_recipe": bool(getattr(args, "strict_controlled_recipe", False)),
        "cnn_encoder_trainable_parameter_count": brain_trainable,
        "text_projection_trainable_parameter_count": text_trainable,
        "brain_projection_trainable_parameter_count": 0,
        "frozen_parameter_count_by_component": {
            "cnn_encoder": brain_frozen,
            "text_projection": text_frozen,
        },
        "optimizer_parameter_groups": [
            {
                "index": idx,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "parameter_count": sum(p.numel() for p in group.get("params", [])),
            }
            for idx, group in enumerate(trainer.optimizer.param_groups)
        ],
    }
    failures = []
    if report["encoder_init"] != "autoencoder_pretrained":
        failures.append("encoder_init is not autoencoder_pretrained")
    if not report["cnn_encoder_trainable"]:
        failures.append("CNN encoder is frozen")
    if not report["text_projection_pretrained"]:
        failures.append("text projection is not initialized from pretrained InfoNCE")
    if not report["text_projection_trainable"]:
        failures.append("text projection is frozen")
    if report["shared_embedding_dimension"] != 384:
        failures.append("shared embedding dimension is not 384")
    report["failures"] = failures
    report["status"] = "passed" if not failures else "failed"
    with (Path(args.run_dir) / "stage3_trainability_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    if failures:
        message = f"Stage 3 controlled-recipe verification failed: {failures}"
        if getattr(args, "strict_controlled_recipe", False):
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        print({"stage3_trainability_warning": failures, "strict_controlled_recipe": False}, flush=True)
    return report


PREVIOUS_PUBMED_REFERENCE_TRAINING = {
    "recipe": "ae_pretrained_finetune_cnn_pretrained_text_trainable",
    "lr_cnn": 1e-4,
    "lr_proj": 1e-5,
    "weight_decay": 1e-4,
    "temperature": 0.07,
    "warmup_epochs": 5,
    "batch_size": 512,
    "batch_size_auto": False,
    "epochs": 150,
    "early_stopping_patience": 25,
    "monitor_metric": "paper_recall_curve_auc",
    "seed": 42,
    "base_channels": 48,
    "num_blocks": 4,
    "out_dim": 384,
    "dropout": 0.1,
    "norm": "group",
    "pooling": "max",
    "encoder_init": "autoencoder_pretrained",
    "text_proj_init": "pretrained_infonce",
    "freeze_text_proj": False,
}


def write_training_diff_report(args: argparse.Namespace) -> dict:
    current = {
        "lr_cnn": args.lr_cnn,
        "lr_proj": args.lr_proj,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "batch_size_auto": args.batch_size_auto,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "monitor_metric": args.monitor_metric,
        "seed": args.seed,
        "base_channels": args.base_channels,
        "num_blocks": args.num_blocks,
        "out_dim": args.out_dim,
        "dropout": args.dropout,
        "norm": args.norm,
        "pooling": args.pooling,
        "encoder_init": args.encoder_init,
        "text_proj_init": args.text_proj_init,
        "freeze_text_proj": args.freeze_text_proj,
    }
    diffs = {}
    for key, previous in PREVIOUS_PUBMED_REFERENCE_TRAINING.items():
        if key == "recipe":
            continue
        value = current.get(key)
        if value != previous:
            required = key == "base_channels"
            if required:
                reason = "Required because the selected mixed-source Stage 1 AE checkpoints were trained with base_channels=64."
            elif key == "val_interval":
                reason = "Intentional six-run workflow change: validation is cheap for these runs, so validating every epoch gives tighter checkpointing and earlier stopping."
            else:
                reason = "Configured multi-source launch setting; not required by architecture compatibility."
            diffs[key] = {
                "previous_value": previous,
                "current_value": value,
                "reason_for_change": reason,
                "required": required,
                "applies_identically_to_all_six_runs": True,
            }
    report = {
        "previous_pubmed_reference": PREVIOUS_PUBMED_REFERENCE_TRAINING,
        "current_multisource_training": current,
        "differences": diffs,
    }
    with (Path(args.run_dir) / "pubmed_reference_vs_multisource_training_diff.json").open("w") as f:
        json.dump(report, f, indent=2)
    return report


@torch.no_grad()
def recall_curve_frame(text_emb: torch.Tensor, brain_emb: torch.Tensor) -> pd.DataFrame:
    t2i, i2t = recall_curve(text_emb, brain_emb)
    n = len(t2i)
    normalized_k = normalized_k_values(n).cpu().numpy()
    return pd.DataFrame(
        {
            "k": np.arange(1, n + 1),
            "normalized_k": normalized_k,
            "t2i_recall": t2i.cpu().numpy(),
            "i2t_recall": i2t.cpu().numpy(),
            "mean_recall": ((t2i + i2t) / 2).cpu().numpy(),
            "random_recall": normalized_k,
        }
    )


@torch.no_grad()
def recall_curve_payload(text_emb: torch.Tensor, brain_emb: torch.Tensor) -> dict:
    """Return paper-style full recall curves as JSON-serializable arrays."""
    t2i, i2t = recall_curve(text_emb, brain_emb)
    n = len(t2i)
    mean_curve = (t2i + i2t) / 2
    normalized_k = normalized_k_values(n)
    auc = normalized_recall_curve_auc(mean_curve)
    return {
        "k_values": list(range(1, n + 1)),
        "normalized_k_values": normalized_k.cpu().tolist(),
        "text_to_brain_recall_curve": t2i.cpu().tolist(),
        "brain_to_text_recall_curve": i2t.cpu().tolist(),
        "mean_recall_curve": mean_curve.cpu().tolist(),
        "random_recall_curve": normalized_k.cpu().tolist(),
        "paper_recall_curve_auc": auc,
        "normalized_k_recall_curve_auc": auc,
    }


@torch.no_grad()
def retrieval_diagnostics(text_emb: torch.Tensor, brain_emb: torch.Tensor, cov: pd.DataFrame) -> pd.DataFrame:
    import torch.nn.functional as F

    text_n = F.normalize(text_emb.float(), dim=1, eps=1e-8)
    brain_n = F.normalize(brain_emb.float(), dim=1, eps=1e-8)
    sim = text_n @ brain_n.T
    ranks = retrieval_ranks(sim).cpu().numpy()
    out = pd.DataFrame(
        {
            "sample_pos": np.arange(len(ranks)),
            "rank": ranks,
            "hit@10": ranks <= 10,
            "top1_pos": sim.argmax(dim=1).cpu().numpy(),
            "true_score": sim.diag().cpu().numpy(),
            "top1_score": sim.max(dim=1).values.cpu().numpy(),
        }
    )
    return out.merge(cov, on="sample_pos", how="left")


def save_plots(run_dir: Path, history: dict, curve_df: pd.DataFrame, diag_df: Optional[pd.DataFrame], brain_emb: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"])
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("InfoNCE")
    if "val_paper_recall_curve_auc" in history:
        ax[1].plot(
            history["val_epoch"],
            history["val_paper_recall_curve_auc"],
            label="paper recall-curve AUC",
        )
        for k in [1, 5, 10, 50]:
            key = f"val_mean_recall@{k}"
            if key in history:
                ax[1].plot(history["val_epoch"], history[key], label=f"r@{k}")
    ax[1].set_title("Validation Retrieval")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve_df["k"], curve_df["mean_recall"], label="model")
    ax.plot(curve_df["k"], curve_df["random_recall"], linestyle="--", label="random")
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("recall")
    ax.set_title("Full Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "recall_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = curve_df["normalized_k"] if "normalized_k" in curve_df else curve_df["k"] / float(curve_df["k"].max())
    ax.plot(x, curve_df["mean_recall"], label="model")
    ax.plot(x, curve_df["random_recall"], linestyle="--", label="random")
    ax.set_xlabel("normalized k (k / n)")
    ax.set_ylabel("recall")
    ax.set_title("Paper-Style Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "recall_curve_normalized_k.png", dpi=160)
    plt.close(fig)

    if diag_df is None:
        return
    try:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=42)
            xy = reducer.fit_transform(brain_emb.float().numpy())
            method = "UMAP"
        except Exception:
            from sklearn.decomposition import PCA

            xy = PCA(n_components=2).fit_transform(brain_emb.float().numpy())
            method = "PCA"
    except Exception as exc:
        print(f"Skipping UMAP/PCA diagnostics: {exc}")
        return

    color_cols = [
        "n_peaks",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "hit@10",
        "total_activation_mass",
        "sparsity",
    ]
    color_cols = [c for c in color_cols if c in diag_df.columns]
    if not color_cols:
        return
    n_cols = min(3, len(color_cols))
    n_rows = int(np.ceil(len(color_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for ax, col in zip(axes.flat, color_cols):
        vals = diag_df[col]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=8, cmap="viridis")
        ax.set_title(col)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        fig.colorbar(sc, ax=ax, fraction=0.046)
    for ax in axes.flat[len(color_cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "umap_diagnostics.png", dpi=160)
    plt.close(fig)


def save_embedding_correlations(run_dir: Path, brain_emb: torch.Tensor, cov: pd.DataFrame) -> None:
    from neurovlm.gnn.ale_cnn import embedding_covariate_correlations

    try:
        corr = embedding_covariate_correlations(brain_emb, cov)
    except Exception as exc:
        print(f"Skipping embedding covariate correlations: {exc}")
        return
    corr.to_csv(run_dir / "embedding_covariate_correlations.csv", index=False)


@torch.no_grad()
def evaluate_neurovlm_baseline_by_pmids(pmids: np.ndarray, device: torch.device) -> dict[str, float]:
    from neurovlm.models import ProjHead
    from neurovlm.retrieval_resources import _load_latent_neuro, _load_latent_text

    brain_latent, brain_pmids = _load_latent_neuro()
    text_latent, text_pmids = _load_latent_text()
    brain_lookup = {str(p): i for i, p in enumerate(brain_pmids)}
    text_lookup = {str(p): i for i, p in enumerate(text_pmids)}
    pairs = [(brain_lookup[p], text_lookup[p]) for p in map(str, pmids) if p in brain_lookup and p in text_lookup]
    if not pairs:
        raise RuntimeError("No overlapping PMIDs for pretrained NeuroVLM baseline.")
    b_idx, t_idx = zip(*pairs)
    image_proj = ProjHead.from_pretrained("image_infonce").to(device).eval()
    text_proj = ProjHead.from_pretrained("text_infonce").to(device).eval()
    brain = image_proj(brain_latent[list(b_idx)].float().to(device)).cpu()
    text = text_proj(text_latent[list(t_idx)].float().to(device)).cpu()
    metrics = bidirectional_retrieval_metrics(text, brain)
    metrics["n_eval"] = float(len(pairs))
    metrics["n_requested"] = float(len(pmids))
    return metrics


def append_comparison_row(args, payload, metrics, trainer, input_shape) -> dict:
    path = Path(args.comparison_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model_name": args.model,
        "preprocessing_type": args.mode,
        "uses_difumo": "yes" if args.mode == "difumo_compatible" else "no",
        "atlas_free": "yes" if args.mode == "atlas_free" else "no",
        "input_shape": str(tuple([1, *input_shape])),
        "number_of_parameters": count_parameters(trainer.brain_encoder),
        "encoder_init": args.encoder_init,
        "autoencoder_checkpoint": args.autoencoder_checkpoint or "",
        "blocks_per_stage": args.blocks_per_stage,
        "use_dilation": args.use_dilation,
        "multi_scale": args.multi_scale,
        "global_context": args.global_context,
        "batch_size": args.batch_size,
        "batch_size_auto": args.batch_size_auto,
        "batch_size_candidates": args.batch_size_candidates,
        "preflight_peak_vram_mb": float(getattr(args, "preflight_peak_vram_mb", 0.0)),
        "training_time_per_epoch": float(np.mean(trainer.history["epoch_time_sec"])),
        "peak_vram": float(max(trainer.history["peak_vram_mb"] or [0.0])),
        "paper_recall_curve_auc": metrics["paper_recall_curve_auc"],
        "full_recall_curve_auc_k1_to_N": metrics["full_recall_curve_auc_k1_to_N"],
        "full_recall_curve_auc": metrics["paper_recall_curve_auc"],
        "recall@1": metrics["mean_recall@1"],
        "recall@5": metrics["mean_recall@5"],
        "recall@10": metrics["mean_recall@10"],
        "recall@50": metrics["mean_recall@50"],
        "mrr": metrics["mean_mrr"],
        "median_rank": metrics["mean_median_rank"],
        "random_recall@10": metrics["mean_random_recall@10"],
        "checkpoint_path": str(Path(args.checkpoint_dir) / "best_val_normalized_recall_auc.pt"),
        "config_path": str(Path(args.run_dir) / "config.json"),
        "cache_path": str(args.cache_file),
        "cache_shape": str(tuple(payload["metadata"]["shape"])),
    }
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    jsonl = path.with_suffix(".jsonl")
    with jsonl.open("a") as f:
        f.write(json.dumps(row) + "\n")
    with (Path(args.run_dir) / "comparison_row.json").open("w") as f:
        json.dump(row, f, indent=2)
    return row


def evaluate_train_subset_sanity(
    trainer: ALETrainer,
    train_ds,
    n: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate retrieval on a small training subset after training.

    If train-subset retrieval is also low, suspect loss/projection/alignment or
    model capacity. If train retrieval is high but val/test retrieval is low,
    the model is likely overfitting or the exact-paper retrieval task is noisy.
    """
    n_eval = min(int(n), len(train_ds))
    subset = Subset(train_ds, list(range(n_eval)))
    metrics, brain_emb, text_emb = trainer.evaluate(subset)
    metrics["n_eval"] = float(n_eval)
    print(
        f"\nTRAIN sanity subset n={n_eval:,} "
        f"paper_recall_curve_auc={metrics['paper_recall_curve_auc']:.4f} "
        f"r@10={metrics['mean_recall@10']:.4f} "
        f"MRR={metrics['mean_mrr']:.4f}",
        flush=True,
    )
    return metrics, brain_emb, text_emb


def main() -> None:
    branch_start = time.perf_counter()
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_dir is None:
        args.run_dir = str(Path("runs") / f"{args.model}_{args.mode}_{stamp}")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.run_dir) / "checkpoints")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timing_profile: dict[str, object] = {
        "data_cache_load_sec": 0.0,
        "preflight_batch_size_sec": 0.0,
        "train_epoch_sec": [],
        "validation_eval_sec": [],
        "generation_auc_eval_sec": [],
        "artifact_save_sec": [],
        "final_eval_sec": [],
        "total_branch_sec": 0.0,
    }

    if args.build_cache_only:
        with (run_dir / "training_config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)
        data_start = time.perf_counter()
        build_cache_only(args)
        timing_profile["data_cache_load_sec"] = float(time.perf_counter() - data_start)
        timing_profile["total_branch_sec"] = float(time.perf_counter() - branch_start)
        with (run_dir / "timing_profile.json").open("w") as f:
            json.dump(timing_profile, f, indent=2)
        return

    data_start = time.perf_counter()
    ae_init_report = resolve_autoencoder_initialization(args)
    ds, train_ds, val_ds, test_ds, payload, preprocess_config = build_dataset(args)
    brain_encoder, text_proj, architecture_report = build_model(args, ds.input_shape)
    timing_profile["data_cache_load_sec"] = float(time.perf_counter() - data_start)
    print(f"[timing] data_cache_load seconds={timing_profile['data_cache_load_sec']:.2f}", flush=True)
    device = which_device(args.device)
    print(f"Selected device: {device}", flush=True)
    if args.pin_memory is False and device.type == "cuda":
        args.pin_memory = True

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (run_dir / "selected_ae_checkpoint.json").open("w") as f:
        json.dump(ae_init_report, f, indent=2, default=str)
    with (run_dir / "encoder_init_report.json").open("w") as f:
        json.dump(ae_init_report, f, indent=2, default=str)
    stage2_dir = run_dir.parent / "03_stage2_encoder_initialization" if run_dir.name == "04_stage3_contrastive" else None
    if stage2_dir is not None:
        stage2_dir.mkdir(parents=True, exist_ok=True)
        with (stage2_dir / "selected_ae_checkpoint.json").open("w") as f:
            json.dump(ae_init_report, f, indent=2, default=str)
        with (stage2_dir / "encoder_init_report.json").open("w") as f:
            json.dump(ae_init_report, f, indent=2, default=str)
    with (run_dir / "preprocessing_config.json").open("w") as f:
        json.dump({**payload["config"], "metadata": payload["metadata"]}, f, indent=2)
    with (run_dir / "model_config.json").open("w") as f:
        json.dump(
            {
                "model": args.model,
                "base_channels": args.base_channels,
                "num_blocks": args.num_blocks,
                "blocks_per_stage": args.blocks_per_stage,
                "out_dim": args.out_dim,
                "dropout": args.dropout,
                "norm": args.norm,
                "pooling": args.pooling,
                "use_dilation": args.use_dilation,
                "multi_scale": args.multi_scale,
                "global_context": args.global_context,
                "encoder_init": args.encoder_init,
                "autoencoder_checkpoint": args.autoencoder_checkpoint,
                "input_shape": ds.input_shape,
                "parameters": count_parameters(brain_encoder),
            },
            f,
            indent=2,
        )

    print("\n== ALE Dataset ==")
    print(f"  mode={args.mode} cache={args.cache_file}")
    print(f"  volume shape={ds.input_shape} aligned papers={len(ds):,}")
    print(f"  split train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print("\n== Model ==")
    print(f"  {args.model} brain params={count_parameters(brain_encoder):,}")
    print(f"  text params={count_parameters(text_proj):,} train_text={not args.freeze_text_proj}")
    batch_label = "auto" if args.batch_size_auto else str(args.batch_size)
    print(f"  device={device} amp={args.amp and device.type == 'cuda'} batch={batch_label}")

    trainer = ALETrainer(brain_encoder, text_proj, args, device)
    trainer.timing_profile.update(timing_profile)
    trainability_report(trainer, args)
    write_training_diff_report(args)
    trainer.load_resume_model_weights()
    preflight_start = time.perf_counter()
    trainer.preflight_batch_size(train_ds)
    trainer.timing_profile["preflight_batch_size_sec"] = float(time.perf_counter() - preflight_start)
    print(f"[timing] preflight_batch_size seconds={trainer.timing_profile['preflight_batch_size_sec']:.2f}", flush=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  selected batch_size={args.batch_size}", flush=True)
    trainer.fit(train_ds, val_ds)
    trainer.restore_best()

    all_metrics = {}
    eval_payloads = {}
    if args.run_train_sanity_eval and args.train_sanity_n and args.train_sanity_n > 0:
        eval_start = time.perf_counter()
        train_metrics, train_brain, train_text = evaluate_train_subset_sanity(
            trainer, train_ds, args.train_sanity_n
        )
        eval_time = time.perf_counter() - eval_start
        trainer.timing_profile["final_eval_sec"].append({"split": "train_subset", "seconds": float(eval_time)})
        print(f"[timing] final_eval split=train_subset seconds={eval_time:.2f}", flush=True)
        all_metrics["train_subset"] = train_metrics
        eval_payloads["train_subset"] = (train_metrics, train_brain, train_text)

    final_splits = [("val", val_ds)]
    if args.final_test_eval:
        final_splits.append(("test", test_ds))
    for name, split_ds in final_splits:
        eval_start = time.perf_counter()
        metrics, brain_emb, text_emb = trainer.evaluate(split_ds)
        eval_time = time.perf_counter() - eval_start
        trainer.timing_profile["final_eval_sec"].append({"split": name, "seconds": float(eval_time)})
        print(f"[timing] final_eval split={name} seconds={eval_time:.2f}", flush=True)
        all_metrics[name] = metrics
        eval_payloads[name] = (metrics, brain_emb, text_emb)
        print(
            f"\n{name.upper()} n={len(split_ds):,} "
            f"paper_recall_curve_auc={metrics['paper_recall_curve_auc']:.4f} "
            f"r@1={metrics['mean_recall@1']:.4f} "
            f"r@5={metrics['mean_recall@5']:.4f} "
            f"r@10={metrics['mean_recall@10']:.4f} "
            f"r@50={metrics['mean_recall@50']:.4f} "
            f"MRR={metrics['mean_mrr']:.4f} "
            f"median_rank={metrics['mean_median_rank']:.1f} "
            f"random_r@10={metrics['mean_random_recall@10']:.4f}"
        )

    if args.eval_neurovlm_baseline:
        eval_start = time.perf_counter()
        all_metrics["test_neurovlm_baseline"] = evaluate_neurovlm_baseline_by_pmids(test_ds.pmids, device)
        eval_time = time.perf_counter() - eval_start
        trainer.timing_profile["final_eval_sec"].append({"split": "test_neurovlm_baseline", "seconds": float(eval_time)})
        print(f"[timing] final_eval split=test_neurovlm_baseline seconds={eval_time:.2f}", flush=True)

    artifact_start = time.perf_counter()
    with (run_dir / "training_history.json").open("w") as f:
        json.dump(trainer.history, f, indent=2)
    with (run_dir / "eval_results.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)
    metrics_dir = run_dir / "metrics"
    config_dir = run_dir / "config"
    ckpt_stage_dir = run_dir / "checkpoints"
    for path in [metrics_dir, config_dir, ckpt_stage_dir]:
        path.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "test_metrics.json").open("w") as f:
        json.dump(all_metrics.get("test", {}), f, indent=2)
    with (config_dir / "contrastive_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (config_dir / "ae_init_config.json").open("w") as f:
        json.dump(ae_init_report, f, indent=2, default=str)
    best_src = Path(args.checkpoint_dir) / "best_val_normalized_recall_auc.pt"
    last_src = Path(args.checkpoint_dir) / "last.pt"
    if SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES and best_src.exists():
        shutil.copy2(best_src, ckpt_stage_dir / "best_contrastive.pt")
        shutil.copy2(best_src, ckpt_stage_dir / "best_ale_cnn.pt")
    if SAVE_LEGACY_STAGE3_CHECKPOINT_ALIASES and last_src.exists():
        shutil.copy2(last_src, ckpt_stage_dir / "last_ale_cnn.pt")
        shutil.copy2(last_src, ckpt_stage_dir / "last_contrastive.pt")
    artifact_time = time.perf_counter() - artifact_start
    trainer.timing_profile["artifact_save_sec"].append({"artifact": "final_core_artifacts", "seconds": float(artifact_time)})
    print(f"[timing] artifact_save artifact=final_core_artifacts seconds={artifact_time:.2f}", flush=True)

    diagnostics_start = time.perf_counter()
    curve_frames = {}
    for split_name, (_, brain_emb, text_emb) in eval_payloads.items():
        curve_df = recall_curve_frame(text_emb, brain_emb)
        curve_frames[split_name] = curve_df
        curve_df.to_csv(run_dir / f"{split_name}_recall_curve.csv", index=False)
        with (run_dir / f"{split_name}_recall_curve.json").open("w") as f:
            json.dump(recall_curve_payload(text_emb, brain_emb), f, indent=2)

    if args.final_test_eval and "test" in eval_payloads:
        test_metrics, test_brain, test_text = eval_payloads["test"]
        curve_df = curve_frames["test"]
        cov = test_ds.covariate_frame()
        diag_df = retrieval_diagnostics(test_text, test_brain, cov)
        diag_df.to_csv(run_dir / "test_retrieval_diagnostics.csv", index=False)
        if args.run_covariate_correlations:
            save_embedding_correlations(run_dir, test_brain, cov)
        if args.save_plots:
            save_plots(run_dir, trainer.history, curve_df, diag_df if args.umap else None, test_brain)
    else:
        test_metrics = all_metrics.get("val", {})
    diagnostics_time = time.perf_counter() - diagnostics_start
    trainer.timing_profile["artifact_save_sec"].append({"artifact": "final_diagnostics_artifacts", "seconds": float(diagnostics_time)})
    print(f"[timing] artifact_save artifact=final_diagnostics_artifacts seconds={diagnostics_time:.2f}", flush=True)

    if getattr(args, "semantic_eval", False) and args.final_test_eval and "test" in eval_payloads:
        semantic_start = time.perf_counter()
        try:
            from experiments.semantic_model_eval import (
                _add_macro_retrieval_normalized_k_auc,
                run_ale_network_labeling,
                run_embedding_semantic_evaluations,
            )
            from neurovlm.data import load_masker

            print("\nRunning standard semantic evaluation suite ...", flush=True)
            semantic_summary = run_embedding_semantic_evaluations(
                model_name=f"{args.model}_{args.mode}",
                brain_embeddings=test_brain,
                text_embeddings=test_text,
                raw_text_embeddings=test_ds.raw_text_embeddings,
                pmids=test_ds.pmids,
                text_projector=trainer.text_proj,
                out_dir=run_dir,
                device=device,
                resource_dir=getattr(args, "eval_resource_dir", None),
                mesh_json=getattr(args, "mesh_json", None),
                resource_use={
                    "network_label_csv": "networks_labels/network_test_set_labels.csv",
                    "network_term_corpus_csv": "networks_labels/network_terms_with_definitions.csv",
                    "pmid_mesh_json": "mesh_kg/mesh_annotations.json",
                    "mesh_node_types": "mesh_kg/mesh_kg_nodes.parquet",
                },
                extra_summary={
                    "params": count_parameters(trainer.brain_encoder),
                    "peak_vram": float(max(trainer.history["peak_vram_mb"] or [0.0])),
                    "training_time_per_epoch": float(np.mean(trainer.history["epoch_time_sec"])),
                    "preprocessing_type": args.mode,
                },
            )
            network_metrics = run_ale_network_labeling(
                trainer=trainer,
                preprocess_config=preprocess_config,
                masker=load_masker(),
                out_dir=run_dir,
                device=device,
                resource_dir=getattr(args, "eval_resource_dir", None),
            )
            semantic_summary.update(
                {
                    "network_accuracy": network_metrics.get("accuracy"),
                    "network_top_2_accuracy": network_metrics.get("top_2_accuracy"),
                    "network_macro_auc": network_metrics.get("macro_auc"),
                }
            )
            semantic_summary.update({key: value for key, value in network_metrics.items() if key.startswith("network_term_")})
            _add_macro_retrieval_normalized_k_auc(semantic_summary)
            with (run_dir / "main_comparison_summary_row.json").open("w") as f:
                json.dump(semantic_summary, f, indent=2)
            pd.DataFrame([semantic_summary]).to_csv(run_dir / "main_comparison_summary_row.csv", index=False)
        except Exception as exc:
            print(f"WARNING: semantic evaluation suite failed: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
        semantic_time = time.perf_counter() - semantic_start
        trainer.timing_profile["final_eval_sec"].append({"split": "semantic_eval", "seconds": float(semantic_time)})
        print(f"[timing] final_eval split=semantic_eval seconds={semantic_time:.2f}", flush=True)
    elif getattr(args, "semantic_eval", False):
        print("Skipping semantic evaluation because final_test_eval=False.", flush=True)

    comparison_start = time.perf_counter()
    append_comparison_row(args, payload, test_metrics, trainer, ds.input_shape)
    trainer.timing_profile["artifact_save_sec"].append({"artifact": "comparison_row", "seconds": float(time.perf_counter() - comparison_start)})
    trainer.timing_profile["total_branch_sec"] = float(time.perf_counter() - branch_start)
    print(f"[timing] total_branch seconds={trainer.timing_profile['total_branch_sec']:.2f}", flush=True)
    with (run_dir / "timing_profile.json").open("w") as f:
        json.dump(trainer.timing_profile, f, indent=2)
    print(f"\nArtifacts saved to {run_dir}")
    print(f"Comparison row appended to {args.comparison_file}")


if __name__ == "__main__":
    main()
