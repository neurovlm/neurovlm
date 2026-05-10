#!/usr/bin/env python
"""Stronger DiFuMo GAT / control experiments for Colab A100.

This script is the executable backend for ``experiments/difumo_gat_colab.ipynb``.
It keeps the DiFuMo representation fixed at 512 coefficients and compares
GAT/GATv2 graph encoders against coefficient-only controls using the same
retrieval metrics as the ALE and CoordGNN experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurovlm.data import load_dataset, load_latent
from neurovlm.gnn.atlas import (
    DIFUMO_DIM,
    compute_difumo_coefficients,
    get_component_centroids,
    load_difumo_components,
)
from neurovlm.gnn.graph import compute_fc_from_coefficients, load_fc_matrix
from neurovlm.gnn.model import TextProjHead
from neurovlm.loss import InfoNCELoss
from neurovlm.metrics import (
    bidirectional_retrieval_metrics,
    recall_curve,
    retrieval_ranks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate stronger DiFuMo GAT controls.")
    p.add_argument("--model", choices=["gat", "gcn", "sage", "mlp", "deepset"], default="gat")
    p.add_argument("--conv", choices=["gat", "gatv2"], default="gatv2")
    p.add_argument(
        "--graph-type",
        choices=["coactivation", "hcp_fc", "spatial", "combined", "no_edge", "shuffled"],
        default="coactivation",
    )
    p.add_argument(
        "--shuffle-graph-base",
        choices=["coactivation", "hcp_fc", "spatial", "combined"],
        default="coactivation",
    )
    p.add_argument("--hcp-fc-path", default=None)
    p.add_argument("--edge-threshold", type=float, default=95.0)
    p.add_argument("--positive-only", action="store_true", default=True)
    p.add_argument("--keep-negative-fc", dest="positive_only", action="store_false")
    p.add_argument("--spatial-sigma-mm", type=float, default=35.0)
    p.add_argument("--combined-fc-weight", type=float, default=0.5)
    p.add_argument("--no-edge-attr", action="store_true")
    p.add_argument("--add-centroids", action="store_true")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--residual", action="store_true", default=True)
    p.add_argument("--no-residual", dest="residual", action="store_false")
    p.add_argument("--layer-norm", action="store_true", default=True)
    p.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    p.add_argument("--out-dim", type=int, default=384)
    p.add_argument("--mlp-hidden-dim", type=int, default=1024)

    p.add_argument("--lr-gat", type=float, default=3e-4)
    p.add_argument("--lr-proj", type=float, default=3e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument(
        "--early-stopping-monitor",
        choices=["train_loss", "val_metric", "none"],
        default="train_loss",
        help=(
            "What triggers early stopping. 'train_loss' stops after the training "
            "loss stops decreasing for N consecutive epochs; 'val_metric' stops "
            "after the monitored validation metric stops improving; 'none' runs "
            "all epochs. Best checkpoint is still selected by validation metric."
        ),
    )
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--monitor-metric", default="mean_auc")
    p.add_argument("--text-proj-init", choices=["random", "pretrained_infonce"], default="random")
    p.add_argument("--freeze-text-proj", action="store_true")

    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-papers", type=int, default=None)

    p.add_argument(
        "--coeff-source",
        choices=["flatmap", "ale_coordinates"],
        default="flatmap",
        help=(
            "Source for DiFuMo node coefficients. 'flatmap' projects existing "
            "pubmed_images flatmaps; 'ale_coordinates' builds Gaussian ALE maps "
            "from sparse MNI coordinates before DiFuMo projection."
        ),
    )
    p.add_argument("--ale-kernel-fwhm-mm", type=float, default=9.0)
    p.add_argument("--ale-normalize", choices=["max", "mass", "none"], default="max")
    p.add_argument("--ale-clamp", action="store_true", default=True)
    p.add_argument("--ale-no-clamp", dest="ale_clamp", action="store_false")
    p.add_argument("--difumo-normalize-coeffs", action="store_true", default=True)
    p.add_argument("--no-difumo-normalize-coeffs", dest="difumo_normalize_coeffs", action="store_false")
    p.add_argument("--coeff-cache-file", default="data/difumo/difumo512_pubmed_coeffs.npz")
    p.add_argument("--force-rebuild-cache", action="store_true")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--comparison-file", default="runs/difumo_gat_comparison.csv")
    p.add_argument("--save-plots", action="store_true", default=True)
    p.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    p.add_argument("--umap", action="store_true")
    p.add_argument("--semantic-eval", action="store_true", default=False)
    p.add_argument("--eval-resource-dir", default=None)
    p.add_argument("--mesh-json", default=None)
    return p.parse_args()


def which_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_step(message: str) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class DifumoData:
    coeffs: torch.Tensor
    text: torch.Tensor
    pmids: np.ndarray
    centroids: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor


class DifumoGraphDataset:
    def __init__(
        self,
        coeffs: torch.Tensor,
        text: torch.Tensor,
        pmids: np.ndarray,
        indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        extra_node_feats: Optional[torch.Tensor] = None,
    ):
        self.coeffs = coeffs.float()
        self.text = text.float()
        self.pmids = pmids
        self.indices = indices.long()
        self.edge_index = edge_index.long()
        self.edge_attr = None if edge_attr is None else edge_attr.float()
        self.extra_node_feats = extra_node_feats

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, pos: int):
        from torch_geometric.data import Data

        idx = int(self.indices[pos])
        x = self.coeffs[idx].unsqueeze(1)
        if self.extra_node_feats is not None:
            x = torch.cat([x, self.extra_node_feats], dim=1)
        data = Data(x=x, edge_index=self.edge_index, y=self.text[idx].unsqueeze(0))
        if self.edge_attr is not None:
            data.edge_attr = self.edge_attr
        data.sample_idx = torch.tensor([idx], dtype=torch.long)
        return data

    def flat_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (node_feats, text) tensors for all split samples without PyG overhead.

        node_feats: [N, n_nodes, feat_dim]  text: [N, text_dim]
        """
        x = self.coeffs[self.indices].unsqueeze(-1)  # [N, 512, 1]
        if self.extra_node_feats is not None:
            extra = self.extra_node_feats.unsqueeze(0).expand(len(self.indices), -1, -1)
            x = torch.cat([x, extra], dim=-1)  # [N, 512, 1+extra_dim]
        return x, self.text[self.indices]

    def covariate_frame(self) -> pd.DataFrame:
        rows = []
        coeffs = self.coeffs[self.indices].float()
        for sample_pos, row in enumerate(coeffs):
            abs_row = row.abs()
            mass = float(abs_row.sum())
            p = abs_row / (abs_row.sum() + 1e-8)
            entropy = float(-(p * torch.log(p + 1e-8)).sum())
            rows.append(
                {
                    "sample_pos": sample_pos,
                    "paper_idx": int(self.indices[sample_pos]),
                    "pmid": str(self.pmids[int(self.indices[sample_pos])]),
                    "total_activation_mass": mass,
                    "mean_activation": float(row.mean()),
                    "std_activation": float(row.std(unbiased=False)),
                    "l2_norm": float(row.norm()),
                    "sparsity_abs_lt_1e-3": float((abs_row < 1e-3).float().mean()),
                    "max_abs_activation": float(abs_row.max()),
                    "argmax_abs_component": int(abs_row.argmax()),
                    "activation_entropy": entropy,
                }
            )
        return pd.DataFrame(rows)


def make_tensor_dataset(data: DifumoData, indices: torch.Tensor) -> TensorDataset:
    return TensorDataset(data.coeffs[indices].float(), data.text[indices].float())


def split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _pmids_to_numpy(pmids_raw, n: int) -> np.ndarray:
    if pmids_raw is None:
        return np.arange(n).astype(str)
    if hasattr(pmids_raw, "detach"):
        pmids_raw = pmids_raw.detach().cpu().numpy()
    return np.asarray(pmids_raw).astype(str)


def _coeff_cache_config(args: argparse.Namespace) -> dict:
    return {
        "coeff_source": args.coeff_source,
        "difumo_dim": DIFUMO_DIM,
        "difumo_normalize_coeffs": bool(args.difumo_normalize_coeffs),
        "max_papers": args.max_papers,
        "ale_kernel_fwhm_mm": float(args.ale_kernel_fwhm_mm),
        "ale_normalize": args.ale_normalize,
        "ale_clamp": bool(args.ale_clamp),
        "version": 2,
    }


def _cache_key(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_flat_volume(vol: np.ndarray, normalize: str, clamp: bool) -> np.ndarray:
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if clamp:
        vol = np.clip(vol, 0.0, None)
    if normalize == "max":
        mx = float(vol.max())
        if mx > 0:
            vol = vol / mx
    elif normalize == "mass":
        total = float(vol.sum())
        if total > 0:
            vol = vol / total
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of {'max', 'mass', 'none'}")
    if clamp:
        vol = np.clip(vol, 0.0, 1.0)
    return vol.astype(np.float32)


def _normalize_coefficients(coeffs: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return coeffs.astype(np.float32)
    mu = coeffs.mean(axis=0, keepdims=True)
    sigma = coeffs.std(axis=0, keepdims=True) + 1e-8
    return ((coeffs - mu) / sigma).astype(np.float32)


def _build_flatmap_difumo_coefficients(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    print("Loading PubMed brain flatmaps...")
    images_data = load_dataset("pubmed_images")
    if isinstance(images_data, (tuple, list)):
        brain_flat_tensor, pmids_raw = images_data[0], images_data[1]
    else:
        brain_flat_tensor, pmids_raw = images_data, None
    brain_flat = brain_flat_tensor.numpy().astype(np.float32)
    pmids = _pmids_to_numpy(pmids_raw, len(brain_flat))
    if args.max_papers is not None:
        brain_flat = brain_flat[: args.max_papers]
        pmids = pmids[: args.max_papers]

    print(f"Loading DiFuMo {DIFUMO_DIM} atlas...")
    components = load_difumo_components(dimension=DIFUMO_DIM)
    print("Projecting flatmaps into DiFuMo coefficient space...")
    coeffs = compute_difumo_coefficients(
        brain_flat,
        components,
        normalize=args.difumo_normalize_coeffs,
    ).astype(np.float32)
    meta = {
        "source": "pubmed_images flatmaps projected into DiFuMo space",
        "n_papers": int(len(pmids)),
        "n_voxels": int(brain_flat.shape[1]),
    }
    return coeffs, pmids, meta


def _build_ale_coordinate_difumo_coefficients(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build sparse-coordinate -> Gaussian ALE -> DiFuMo coefficients."""
    from scipy.ndimage import gaussian_filter
    from neurovlm.data import load_masker
    from neurovlm.retrieval_resources import _load_pubmed_coordinates

    print("Loading PubMed MNI peak coordinates...")
    coords_df = _load_pubmed_coordinates().copy()
    coords_df["pmid"] = coords_df["pmid"].astype(str)
    grouped = list(coords_df.groupby("pmid", sort=False))
    if args.max_papers is not None:
        grouped = grouped[: args.max_papers]

    masker = load_masker()
    mask_img = masker.mask_img_
    mask = np.asarray(mask_img.get_fdata() > 0)
    affine = np.asarray(mask_img.affine, dtype=np.float32)
    inv_affine = np.linalg.inv(affine)
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
    sigma_mm = float(args.ale_kernel_fwhm_mm) / 2.354820045
    sigma_vox = sigma_mm / voxel_sizes

    print(f"Loading DiFuMo {DIFUMO_DIM} atlas...")
    components = load_difumo_components(dimension=DIFUMO_DIM)
    if int(mask.sum()) != int(components.shape[1]):
        raise ValueError(
            f"Mask has {int(mask.sum())} voxels but DiFuMo components have "
            f"{int(components.shape[1])} masked voxels."
        )

    coeff_rows: list[np.ndarray] = []
    pmids: list[str] = []
    skipped_empty = 0
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x  # type: ignore[assignment]

    print(
        "Building Gaussian ALE maps from coordinates and projecting to DiFuMo "
        f"(FWHM={args.ale_kernel_fwhm_mm:g}mm)..."
    )
    for pmid, grp in tqdm(grouped, desc="ALE->DiFuMo", unit="paper"):
        impulse = np.zeros(mask.shape, dtype=np.float32)
        coords = grp[["x", "y", "z"]].to_numpy(dtype=np.float32)
        for xyz in np.unique(coords, axis=0):
            vox = inv_affine @ np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float32)
            ijk = np.rint(vox[:3]).astype(int)
            if np.any(ijk < 0) or np.any(ijk >= np.asarray(mask.shape)):
                continue
            idx = tuple(ijk.tolist())
            if not mask[idx]:
                continue
            impulse[idx] += 1.0
        if impulse.sum() == 0:
            skipped_empty += 1
            continue
        vol = gaussian_filter(impulse, sigma=sigma_vox, mode="constant")
        vol *= mask.astype(np.float32)
        vol = _normalize_flat_volume(vol, args.ale_normalize, args.ale_clamp)
        flat = vol[mask].astype(np.float32)
        coeff_rows.append(flat @ components.T)
        pmids.append(str(pmid))

    if not coeff_rows:
        raise RuntimeError("ALE coordinate DiFuMo builder produced no non-empty papers.")
    coeffs = np.stack(coeff_rows).astype(np.float32)
    coeffs = _normalize_coefficients(coeffs, args.difumo_normalize_coeffs)
    meta = {
        "source": "pubmed MNI coordinates -> Gaussian ALE maps -> DiFuMo coefficients",
        "n_papers": int(len(pmids)),
        "skipped_empty_papers": int(skipped_empty),
        "kernel_fwhm_mm": float(args.ale_kernel_fwhm_mm),
        "sigma_vox": sigma_vox.tolist(),
        "voxel_sizes_mm": voxel_sizes.tolist(),
        "ale_normalize": args.ale_normalize,
        "ale_clamp": bool(args.ale_clamp),
    }
    return coeffs, np.asarray(pmids).astype(str), meta


def build_or_load_coefficients(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    cache = Path(args.coeff_cache_file)
    config = _coeff_cache_config(args)
    key = _cache_key(config)
    if cache.exists() and not args.force_rebuild_cache:
        payload = np.load(cache, allow_pickle=True)
        cached_key = str(payload["config_key"]) if "config_key" in payload.files else None
        if cached_key == key:
            coeffs = payload["coeffs"].astype(np.float32)
            pmids = payload["pmids"].astype(str)
            print(f"Loaded cached DiFuMo coefficients: {coeffs.shape} from {cache}")
            return coeffs, pmids
        print("DiFuMo coefficient cache config changed; rebuilding cache.")

    if args.coeff_source == "flatmap":
        coeffs, pmids, meta = _build_flatmap_difumo_coefficients(args)
    elif args.coeff_source == "ale_coordinates":
        coeffs, pmids, meta = _build_ale_coordinate_difumo_coefficients(args)
    else:
        raise ValueError(f"Unknown coefficient source: {args.coeff_source}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        coeffs=coeffs,
        pmids=pmids,
        config_key=key,
        config_json=json.dumps(config, sort_keys=True),
        metadata_json=json.dumps(meta, sort_keys=True),
    )
    print(f"Saved coefficient cache: {cache}")
    return coeffs, pmids


def align_text(coeffs: np.ndarray, brain_pmids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_latents = load_latent("pubmed_text")
    pubmed_df = load_dataset("pubmed_text")
    if isinstance(text_latents, tuple):
        text_tensor, text_pmids = text_latents
        text = text_tensor.detach().cpu().numpy() if hasattr(text_tensor, "detach") else np.asarray(text_tensor)
        text_pmids = np.asarray(text_pmids).astype(str)
    elif isinstance(text_latents, dict):
        text_pmids = np.asarray(list(text_latents.keys())).astype(str)
        text = np.stack([np.asarray(text_latents[p], dtype=np.float32) for p in text_latents])
    else:
        text_tensor = text_latents if isinstance(text_latents, torch.Tensor) else torch.tensor(text_latents)
        text = text_tensor.detach().cpu().numpy().astype(np.float32)
        if "pmid" in pubmed_df.columns:
            text_pmids = pubmed_df["pmid"].astype(str).values[: len(text)]
        else:
            text_pmids = np.arange(len(text)).astype(str)

    text_lookup = {str(p): i for i, p in enumerate(text_pmids)}
    brain_rows, text_rows, keep_pmids = [], [], []
    for i, pmid in enumerate(brain_pmids.astype(str)):
        if pmid in text_lookup:
            brain_rows.append(i)
            text_rows.append(text_lookup[pmid])
            keep_pmids.append(pmid)

    if not brain_rows:
        n = min(len(coeffs), len(text))
        print(f"No PMID overlap found; falling back to row alignment for {n} pairs.")
        return coeffs[:n], text[:n], np.arange(n).astype(str)

    aligned_coeffs = coeffs[brain_rows].astype(np.float32)
    aligned_text = text[text_rows].astype(np.float32)
    print(f"Aligned {len(keep_pmids):,} DiFuMo/text pairs by PMID.")
    return aligned_coeffs, aligned_text, np.asarray(keep_pmids).astype(str)


def load_difumo_data(args: argparse.Namespace) -> DifumoData:
    coeffs_np, brain_pmids = build_or_load_coefficients(args)
    coeffs_np, text_np, pmids = align_text(coeffs_np, brain_pmids)
    graph_base = args.shuffle_graph_base if args.graph_type == "shuffled" else args.graph_type
    needs_centroids = bool(args.add_centroids or graph_base in {"spatial", "combined"})
    if needs_centroids:
        centroids = get_component_centroids(dimension=DIFUMO_DIM).astype(np.float32)
        centroids = centroids / 150.0
    else:
        # MLP/coactivation/no-edge runs do not use atlas centroids. Avoid loading
        # the full DiFuMo atlas a second time in memory-constrained Colab jobs.
        centroids = np.zeros((DIFUMO_DIM, 3), dtype=np.float32)
    try:
        from neurovlm.data import load_dataset
        from neurovlm.semantic_evaluation import official_split_positions

        split_pos = official_split_positions(
            load_dataset("pubmed_text"),
            pmids,
            out_dir=args.run_dir,
            random_state=args.seed,
            random_val_frac=args.val_frac,
            random_test_frac=args.test_frac,
        )
        train_idx = torch.tensor(split_pos["train"], dtype=torch.long)
        val_idx = torch.tensor(split_pos["val"], dtype=torch.long)
        test_idx = torch.tensor(split_pos["test"], dtype=torch.long)
        print("Using PubMed dataframe split columns for train/val/test.")
    except Exception as exc:
        print(f"WARNING: official PubMed split failed ({exc}); falling back to random split.")
        train_idx, val_idx, test_idx = split_indices(len(coeffs_np), args.val_frac, args.test_frac, args.seed)
    return DifumoData(
        coeffs=torch.tensor(coeffs_np, dtype=torch.float32),
        text=torch.tensor(text_np, dtype=torch.float32),
        pmids=pmids,
        centroids=torch.tensor(centroids, dtype=torch.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def normalize_unit(mat: np.ndarray) -> np.ndarray:
    x = mat.astype(np.float32).copy()
    finite = np.isfinite(x)
    x[~finite] = 0.0
    np.fill_diagonal(x, 0.0)
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs
    return x


def spatial_matrix(centroids: np.ndarray, sigma_mm: float) -> np.ndarray:
    xyz = centroids.astype(np.float32)
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    weights = np.exp(-(dist * dist) / (2.0 * sigma_mm * sigma_mm)).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    return weights


def threshold_matrix(mat: np.ndarray, percentile: float, positive_only: bool) -> np.ndarray:
    adj = mat.astype(np.float32).copy()
    adj[~np.isfinite(adj)] = 0.0
    np.fill_diagonal(adj, 0.0)
    if positive_only:
        adj = np.clip(adj, 0.0, None)
    scores = np.abs(adj)
    nonzero = scores[scores > 0]
    if nonzero.size == 0:
        return np.zeros_like(adj, dtype=np.float32)
    cutoff = np.percentile(nonzero, percentile)
    adj[scores < cutoff] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def adjacency_to_edges(adj: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = np.nonzero(adj)
    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float32)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(adj[rows, cols], dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def connected_components(edge_index: torch.Tensor, n_nodes: int) -> int:
    if edge_index.numel() == 0:
        return n_nodes
    neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        neighbors[s].append(d)
        neighbors[d].append(s)
    seen = [False] * n_nodes
    n_components = 0
    for start in range(n_nodes):
        if seen[start]:
            continue
        n_components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nxt in neighbors[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
    return n_components


def graph_stats(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor], n_nodes: int = DIFUMO_DIM) -> dict[str, float]:
    n_edges = int(edge_index.shape[1])
    stats: dict[str, float] = {
        "number_of_nodes": float(n_nodes),
        "number_of_edges": float(n_edges),
        "average_degree": float(n_edges / max(1, n_nodes)),
        "average_degree_directed": float(n_edges / max(1, n_nodes)),
        "connected_components": float(connected_components(edge_index, n_nodes)),
    }
    if edge_attr is None or edge_attr.numel() == 0:
        stats.update(
            {
                "edge_weight_min": 0.0,
                "edge_weight_max": 0.0,
                "edge_weight_mean": 0.0,
                "edge_weight_std": 0.0,
            }
        )
    else:
        w = edge_attr.float().view(-1)
        stats.update(
            {
                "edge_weight_min": float(w.min()),
                "edge_weight_max": float(w.max()),
                "edge_weight_mean": float(w.mean()),
                "edge_weight_std": float(w.std(unbiased=False)),
            }
        )
    return stats


def build_weight_matrix(graph_type: str, data: DifumoData, args: argparse.Namespace) -> np.ndarray:
    coeffs = data.coeffs.numpy()
    centroids_mm = data.centroids.numpy() * 150.0
    if graph_type == "coactivation":
        fc = compute_fc_from_coefficients(coeffs[data.train_idx.numpy()])
        print(
            "Coactivation FC raw stats:",
            f"min={fc.min():.4f}",
            f"max={fc.max():.4f}",
            f"mean={fc.mean():.4f}",
            f"std={fc.std():.4f}",
        )
        return fc
    if graph_type == "hcp_fc":
        if args.hcp_fc_path is None:
            raise ValueError("--hcp-fc-path is required for graph-type=hcp_fc")
        fc = load_fc_matrix(args.hcp_fc_path)
        if fc.shape != (DIFUMO_DIM, DIFUMO_DIM):
            raise ValueError(f"Expected HCP FC shape {(DIFUMO_DIM, DIFUMO_DIM)}, got {fc.shape}")
        return fc
    if graph_type == "spatial":
        return spatial_matrix(centroids_mm, args.spatial_sigma_mm)
    if graph_type == "combined":
        if args.hcp_fc_path:
            fc = load_fc_matrix(args.hcp_fc_path)
        else:
            fc = compute_fc_from_coefficients(coeffs[data.train_idx.numpy()])
        sp = spatial_matrix(centroids_mm, args.spatial_sigma_mm)
        w = float(args.combined_fc_weight)
        return w * normalize_unit(fc) + (1.0 - w) * normalize_unit(sp)
    raise ValueError(f"Unsupported weight matrix graph type: {graph_type}")


def build_graph(data: DifumoData, args: argparse.Namespace) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, float]]:
    if args.graph_type == "no_edge":
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = None if args.no_edge_attr else torch.empty((0, 1), dtype=torch.float32)
        stats = graph_stats(edge_index, edge_attr)
        return edge_index, edge_attr, stats

    base_type = args.shuffle_graph_base if args.graph_type == "shuffled" else args.graph_type
    weights = build_weight_matrix(base_type, data, args)
    adj = threshold_matrix(weights, percentile=args.edge_threshold, positive_only=args.positive_only)
    edge_index, edge_attr = adjacency_to_edges(adj)

    if args.graph_type == "shuffled" and edge_index.numel() > 0:
        gen = torch.Generator().manual_seed(args.seed + 17)
        n_edges = edge_index.shape[1]
        src = edge_index[0]
        dst = torch.randint(0, DIFUMO_DIM, (n_edges,), generator=gen)
        self_mask = dst.eq(src)
        dst[self_mask] = (dst[self_mask] + 1) % DIFUMO_DIM
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = edge_attr[torch.randperm(n_edges, generator=gen)]

    if args.no_edge_attr:
        edge_attr = None
    stats = graph_stats(edge_index, edge_attr)
    return edge_index, edge_attr, stats


class DifumoGATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        heads: int,
        layers: int,
        out_dim: int,
        dropout: float,
        conv: str,
        use_edge_attr: bool,
        residual: bool,
        layer_norm: bool,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv, GATv2Conv
        except ImportError as exc:
            raise ImportError("Install PyTorch Geometric: pip install torch-geometric") from exc
        conv_cls = GATv2Conv if conv == "gatv2" else GATConv
        self.dropout = dropout
        self.residual = residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        dims = [in_dim] + [hidden * heads] * layers
        for layer_idx in range(layers):
            edge_dim = 1 if use_edge_attr else None
            self.convs.append(
                conv_cls(
                    dims[layer_idx],
                    hidden,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )
            self.norms.append(nn.LayerNorm(dims[layer_idx + 1]) if layer_norm else nn.Identity())
            if residual and dims[layer_idx] != dims[layer_idx + 1]:
                self.res_proj.append(nn.Linear(dims[layer_idx], dims[layer_idx + 1], bias=False))
            else:
                self.res_proj.append(nn.Identity())
        self.proj = nn.Sequential(
            nn.Linear(dims[-1] * 2, dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims[-1], out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        from torch_geometric.nn import global_add_pool, global_mean_pool

        for conv, norm, res_proj in zip(self.convs, self.norms, self.res_proj):
            prev = x
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual:
                x = x + res_proj(prev)
            x = norm(x)
        pooled = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        return self.proj(pooled)


class DifumoMLPEncoder(nn.Module):
    def __init__(self, in_dim: int = DIFUMO_DIM, hidden_dim: int = 1024, out_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self.net(coeffs)


class DifumoDeepSetEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        from torch_geometric.nn import global_max_pool, global_mean_pool

        h = self.phi(x)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.rho(pooled)


class DifumoGraphConvEncoder(nn.Module):
    """Faster message-passing baselines over the shared DiFuMo graph."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        layers: int,
        out_dim: int,
        dropout: float,
        conv: str,
        residual: bool,
        layer_norm: bool,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv, SAGEConv
        except ImportError as exc:
            raise ImportError("Install PyTorch Geometric: pip install torch-geometric") from exc
        if conv not in {"gcn", "sage"}:
            raise ValueError("conv must be 'gcn' or 'sage'")
        self.conv_kind = conv
        self.dropout = dropout
        self.residual = residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for layer_idx in range(layers):
            if conv == "gcn":
                self.convs.append(GCNConv(dims[layer_idx], hidden, add_self_loops=False))
            else:
                self.convs.append(SAGEConv(dims[layer_idx], hidden))
            self.norms.append(nn.LayerNorm(hidden) if layer_norm else nn.Identity())
            if residual and dims[layer_idx] != hidden:
                self.res_proj.append(nn.Linear(dims[layer_idx], hidden, bias=False))
            else:
                self.res_proj.append(nn.Identity())
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        from torch_geometric.nn import global_add_pool, global_mean_pool

        for conv, norm, res_proj in zip(self.convs, self.norms, self.res_proj):
            prev = x
            if self.conv_kind == "gcn":
                edge_weight = edge_attr.view(-1) if edge_attr is not None and edge_attr.numel() > 0 else None
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual:
                x = x + res_proj(prev)
            x = norm(x)
        pooled = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        return self.proj(pooled)


def build_model(args: argparse.Namespace, in_dim: int, use_edge_attr: bool) -> tuple[nn.Module, nn.Module]:
    if args.model == "gat":
        brain = DifumoGATEncoder(
            in_dim=in_dim,
            hidden=args.hidden,
            heads=args.heads,
            layers=args.layers,
            out_dim=args.out_dim,
            dropout=args.dropout,
            conv=args.conv,
            use_edge_attr=use_edge_attr,
            residual=args.residual,
            layer_norm=args.layer_norm,
        )
    elif args.model in {"gcn", "sage"}:
        brain = DifumoGraphConvEncoder(
            in_dim=in_dim,
            hidden=args.hidden,
            layers=args.layers,
            out_dim=args.out_dim,
            dropout=args.dropout,
            conv=args.model,
            residual=args.residual,
            layer_norm=args.layer_norm,
        )
    elif args.model == "mlp":
        brain = DifumoMLPEncoder(
            in_dim=DIFUMO_DIM,
            hidden_dim=args.mlp_hidden_dim,
            out_dim=args.out_dim,
            dropout=args.dropout,
        )
    else:
        brain = DifumoDeepSetEncoder(in_dim=in_dim, hidden=args.hidden * args.heads, out_dim=args.out_dim, dropout=args.dropout)

    if args.text_proj_init == "pretrained_infonce":
        if args.out_dim != 384:
            raise ValueError("pretrained_infonce text projector requires --out-dim 384")
        from neurovlm.models import ProjHead

        text_proj = ProjHead.from_pretrained("text_infonce")
    else:
        text_proj = TextProjHead(in_dim=768, hidden_dim=512, out_dim=args.out_dim)
    return brain, text_proj


class DifumoTrainer:
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
        groups = [{"params": self.brain_encoder.parameters(), "lr": args.lr_gat}]
        if not args.freeze_text_proj:
            groups.append({"params": self.text_proj.parameters(), "lr": args.lr_proj})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.use_amp = args.amp and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.best_state: Optional[dict] = None
        self.best_score = -float("inf")
        self._batch_templates: dict[tuple[int, int], object] = {}
        self._graph_cache: Optional[dict] = None
        self.history: dict[str, list] = {
            "train_loss": [],
            "epoch_time_sec": [],
            "peak_vram_mb": [],
            "lr_brain": [],
            "lr_proj": [],
            "val_epoch": [],
        }

    def _build_template(self, batch_size: int, n_nodes: int, feat_dim: int) -> object:
        from torch_geometric.data import Data, Batch
        from torch_geometric.utils import add_self_loops as pyg_add_self_loops

        gc = self._graph_cache
        # Pre-add self-loops so the graph is static and GATv2Conv (add_self_loops=False)
        # doesn't need to allocate new tensors on every forward pass.
        ei = gc["edge_index"].cpu()
        ea = gc["edge_attr"].cpu() if gc["edge_attr"] is not None else None
        ei, ea = pyg_add_self_loops(ei, ea, num_nodes=n_nodes)
        samples = [
            Data(
                x=torch.zeros(n_nodes, feat_dim),
                edge_index=ei,
                **({} if ea is None else {"edge_attr": ea}),
            )
            for _ in range(batch_size)
        ]
        return Batch.from_data_list(samples).to(self.device)

    def _get_template(self, batch_size: int, feat_dim: int) -> object:
        key = (batch_size, feat_dim)
        if key not in self._batch_templates:
            self._batch_templates[key] = self._build_template(
                batch_size, self._graph_cache["n_nodes"], feat_dim
            )
        return self._batch_templates[key]

    def make_loader(self, ds, shuffle: bool):
        if self.args.model == "mlp":
            return DataLoader(
                ds,
                batch_size=self.args.batch_size,
                shuffle=shuffle,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                persistent_workers=self.args.num_workers > 0,
            )
        # For graph models (gat, deepset): avoid per-batch PyG collation by using a
        # pre-built template batch whose edge_index/batch vector never changes.
        if self._graph_cache is None:
            self._graph_cache = {
                "edge_index": ds.edge_index.to(self.device),
                "edge_attr": ds.edge_attr.to(self.device) if ds.edge_attr is not None else None,
                "n_nodes": int(ds.coeffs.shape[1]),
            }
        node_feats, text = ds.flat_tensors()
        flat_ds = TensorDataset(node_feats, text)
        return DataLoader(
            flat_ds,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            persistent_workers=self.args.num_workers > 0,
            drop_last=shuffle,
        )

    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.args.model == "mlp":
            coeffs, text = batch
            coeffs = coeffs.to(self.device, non_blocking=self.args.pin_memory)
            text = text.to(self.device, non_blocking=self.args.pin_memory)
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    brain = self.brain_encoder(coeffs)
                    text_emb = self.text_proj(text)
            else:
                brain = self.brain_encoder(coeffs)
                text_emb = self.text_proj(text)
            return brain, text_emb

        # Graph models: batch is (node_feats [B, N, F], text [B, D]) from flat loader.
        x_batch, text_batch = batch
        B, N, F = x_batch.shape
        x_batch = x_batch.to(self.device, non_blocking=self.args.pin_memory)
        text_batch = text_batch.to(self.device, non_blocking=self.args.pin_memory)
        tmpl = self._get_template(B, F)
        tmpl.x = x_batch.reshape(B * N, F)
        edge_attr = getattr(tmpl, "edge_attr", None)
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                brain = self.brain_encoder(tmpl.x, tmpl.edge_index, edge_attr, tmpl.batch)
                text_emb = self.text_proj(text_batch)
        else:
            brain = self.brain_encoder(tmpl.x, tmpl.edge_index, edge_attr, tmpl.batch)
            text_emb = self.text_proj(text_batch)
        return brain, text_emb

    def fit(self, train_ds, val_ds) -> None:
        loader = self.make_loader(train_ds, shuffle=True)
        total_steps = max(1, len(loader) * self.args.epochs)
        warmup_steps = max(1, len(loader) * self.args.warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_warmup(step, warmup_steps, total_steps),
        )
        bad_val_checks = 0
        bad_loss_epochs = 0
        best_train_loss = float("inf")
        for epoch in range(self.args.epochs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
            losses = []
            self.brain_encoder.train()
            self.text_proj.train()
            for batch in loader:
                brain, text = self._forward(batch)
                loss = self.loss_fn(brain, text)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                params = list(self.brain_encoder.parameters()) + list(self.text_proj.parameters())
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                losses.append(float(loss.detach().cpu()))

            epoch_time = time.perf_counter() - start
            peak_vram = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
            self.history["train_loss"].append(float(np.mean(losses)))
            mean_loss = self.history["train_loss"][-1]
            self.history["epoch_time_sec"].append(float(epoch_time))
            self.history["peak_vram_mb"].append(float(peak_vram))
            self.history["lr_brain"].append(float(self.optimizer.param_groups[0]["lr"]))
            self.history["lr_proj"].append(0.0 if self.args.freeze_text_proj else float(self.optimizer.param_groups[1]["lr"]))

            if mean_loss < best_train_loss - self.args.early_stopping_min_delta:
                best_train_loss = mean_loss
                bad_loss_epochs = 0
            else:
                bad_loss_epochs += 1

            if epoch % self.args.val_interval == 0:
                metrics, _, _ = self.evaluate(val_ds)
                self.history["val_epoch"].append(epoch)
                for key, value in metrics.items():
                    self.history.setdefault(f"val_{key}", []).append(float(value))
                score = float(metrics[self.args.monitor_metric])
                print(
                    f"Epoch {epoch:03d}/{self.args.epochs} "
                    f"loss={mean_loss:.4f} "
                    f"val_auc={metrics['mean_auc']:.4f} "
                    f"val_r@10={metrics['mean_recall@10']:.4f} "
                    f"time={epoch_time:.1f}s vram={peak_vram:.0f}MB"
                )
                if score > self.best_score + self.args.early_stopping_min_delta:
                    self.best_score = score
                    self.best_state = {
                        "brain_encoder": deepcopy(self.brain_encoder.state_dict()),
                        "text_proj": deepcopy(self.text_proj.state_dict()),
                        "epoch": epoch,
                        "metrics": metrics,
                        "config": vars(self.args),
                    }
                    bad_val_checks = 0
                else:
                    bad_val_checks += 1
            else:
                print(f"Epoch {epoch:03d}/{self.args.epochs} loss={mean_loss:.4f} time={epoch_time:.1f}s")

            if self.args.early_stopping_patience is not None:
                if (
                    self.args.early_stopping_monitor == "train_loss"
                    and bad_loss_epochs >= self.args.early_stopping_patience
                ):
                    print(
                        "Early stopping after "
                        f"{bad_loss_epochs} epochs without training-loss improvement "
                        f"(min_delta={self.args.early_stopping_min_delta:g})."
                    )
                    break
                if (
                    self.args.early_stopping_monitor == "val_metric"
                    and bad_val_checks >= self.args.early_stopping_patience
                ):
                    print(
                        "Early stopping after "
                        f"{bad_val_checks} validation checks without "
                        f"{self.args.monitor_metric} improvement "
                        f"(min_delta={self.args.early_stopping_min_delta:g})."
                    )
                    break

        if self.best_state is None:
            self.best_state = {
                "brain_encoder": deepcopy(self.brain_encoder.state_dict()),
                "text_proj": deepcopy(self.text_proj.state_dict()),
                "epoch": len(self.history["train_loss"]) - 1,
                "metrics": {},
                "config": vars(self.args),
            }
        self.save_checkpoint("best_difumo_gat.pt")
        self.save_last()

    @torch.no_grad()
    def collect_embeddings(self, ds) -> tuple[torch.Tensor, torch.Tensor]:
        self.brain_encoder.eval()
        self.text_proj.eval()
        loader = self.make_loader(ds, shuffle=False)
        all_brain, all_text = [], []
        for batch in loader:
            brain, text = self._forward(batch)
            all_brain.append(brain.float().cpu())
            all_text.append(text.float().cpu())
        return torch.cat(all_brain), torch.cat(all_text)

    @torch.no_grad()
    def evaluate(self, ds) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        brain, text = self.collect_embeddings(ds)
        metrics = bidirectional_retrieval_metrics(text, brain)
        metrics["loss"] = float(self.loss_fn(brain, text).item())
        return metrics, brain, text

    def restore_best(self) -> None:
        if self.best_state is None:
            raise RuntimeError("No best checkpoint available.")
        self.brain_encoder.load_state_dict(self.best_state["brain_encoder"])
        self.text_proj.load_state_dict(self.best_state["text_proj"])

    def save_checkpoint(self, filename: str) -> None:
        if self.best_state is None:
            return
        path = Path(self.args.checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_state, path)

    def save_last(self) -> None:
        path = Path(self.args.checkpoint_dir) / "last_difumo_gat.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "brain_encoder": self.brain_encoder.state_dict(),
                "text_proj": self.text_proj.state_dict(),
                "history": self.history,
                "config": vars(self.args),
            },
            path,
        )


@torch.no_grad()
def recall_curve_frame(text_emb: torch.Tensor, brain_emb: torch.Tensor) -> pd.DataFrame:
    t2i, i2t = recall_curve(text_emb, brain_emb)
    n = len(t2i)
    return pd.DataFrame(
        {
            "k": np.arange(1, n + 1),
            "t2i_recall": t2i.cpu().numpy(),
            "i2t_recall": i2t.cpu().numpy(),
            "mean_recall": ((t2i + i2t) / 2).cpu().numpy(),
            "random_recall": np.arange(1, n + 1) / float(n),
        }
    )


@torch.no_grad()
def retrieval_diagnostics(text_emb: torch.Tensor, brain_emb: torch.Tensor, cov: pd.DataFrame) -> pd.DataFrame:
    text_n = F.normalize(text_emb.float(), dim=1, eps=1e-8)
    brain_n = F.normalize(brain_emb.float(), dim=1, eps=1e-8)
    sim = text_n @ brain_n.T
    ranks = retrieval_ranks(sim).cpu().numpy()
    out = pd.DataFrame(
        {
            "sample_pos": np.arange(len(ranks)),
            "rank": ranks,
            "hit@1": ranks <= 1,
            "hit@5": ranks <= 5,
            "hit@10": ranks <= 10,
            "hit@50": ranks <= 50,
            "top1_pos": sim.argmax(dim=1).cpu().numpy(),
            "true_score": sim.diag().cpu().numpy(),
            "top1_score": sim.max(dim=1).values.cpu().numpy(),
        }
    )
    return out.merge(cov, on="sample_pos", how="left")


def save_embedding_correlations(run_dir: Path, brain_emb: torch.Tensor, cov: pd.DataFrame) -> None:
    from neurovlm.gnn.coord_diagnostics import embedding_covariate_correlations

    try:
        corr = embedding_covariate_correlations(brain_emb, cov)
    except Exception as exc:
        print(f"Skipping embedding covariate correlations: {exc}")
        return
    corr.to_csv(run_dir / "embedding_covariate_correlations.csv", index=False)


def save_plots(run_dir: Path, history: dict, curve_df: pd.DataFrame, diag_df: pd.DataFrame, brain_emb: torch.Tensor, use_umap: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"])
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("InfoNCE")
    if "val_mean_auc" in history:
        ax[1].plot(history["val_epoch"], history["val_mean_auc"], label="AUC")
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

    if not use_umap:
        return
    try:
        try:
            import umap  # type: ignore

            xy = umap.UMAP(n_components=2, random_state=42).fit_transform(brain_emb.float().numpy())
            method = "UMAP"
        except Exception:
            from sklearn.decomposition import PCA

            xy = PCA(n_components=2).fit_transform(brain_emb.float().numpy())
            method = "PCA"
    except Exception as exc:
        print(f"Skipping UMAP/PCA diagnostics: {exc}")
        return

    color_cols = [
        "rank",
        "hit@10",
        "total_activation_mass",
        "std_activation",
        "l2_norm",
        "activation_entropy",
        "sparsity_abs_lt_1e-3",
    ]
    color_cols = [c for c in color_cols if c in diag_df.columns]
    n_cols = min(3, len(color_cols))
    n_rows = int(np.ceil(len(color_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for ax, col in zip(axes.flat, color_cols):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=diag_df[col], s=8, cmap="viridis")
        ax.set_title(col)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        fig.colorbar(sc, ax=ax, fraction=0.046)
    for ax in axes.flat[len(color_cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "umap_diagnostics.png", dpi=160)
    plt.close(fig)


def append_comparison_row(
    args: argparse.Namespace,
    metrics: dict[str, float],
    graph_info: dict[str, float],
    trainer: DifumoTrainer,
) -> dict:
    path = Path(args.comparison_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model_name": f"difumo_{args.model}",
        "preprocessing_type": "difumo_coefficients",
        "coefficient_source": args.coeff_source,
        "ale_kernel_fwhm_mm": args.ale_kernel_fwhm_mm if args.coeff_source == "ale_coordinates" else "",
        "uses_difumo": "yes",
        "atlas_free": "no",
        "graph_type": args.graph_type if args.model in {"gat", "gcn", "sage"} else "none",
        "conv": args.conv if args.model == "gat" else (args.model if args.model in {"gcn", "sage"} else ""),
        "input_shape": "(512,)",
        "number_of_parameters": count_parameters(trainer.brain_encoder),
        "training_time_per_epoch": float(np.mean(trainer.history["epoch_time_sec"])),
        "peak_vram": float(max(trainer.history["peak_vram_mb"] or [0.0])),
        "validation_paper_recall_curve_auc": trainer.best_state.get("metrics", {}).get("mean_auc") if trainer.best_state else "",
        "validation_recall@10": trainer.best_state.get("metrics", {}).get("mean_recall@10") if trainer.best_state else "",
        "paper_recall_curve_auc": metrics["mean_auc"],
        "full_recall_curve_auc": metrics["mean_auc"],
        "recall@1": metrics["mean_recall@1"],
        "recall@5": metrics["mean_recall@5"],
        "recall@10": metrics["mean_recall@10"],
        "recall@50": metrics["mean_recall@50"],
        "mrr": metrics["mean_mrr"],
        "median_rank": metrics["mean_median_rank"],
        "random_recall@10": metrics["mean_random_recall@10"],
        "n_edges": graph_info.get("number_of_edges", 0.0),
        "avg_degree": graph_info.get("average_degree_directed", 0.0),
        "checkpoint_path": str(Path(args.checkpoint_dir) / "best_difumo_gat.pt"),
        "config_path": str(Path(args.run_dir) / "config.json"),
    }
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with path.with_suffix(".jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    with (Path(args.run_dir) / "comparison_row.json").open("w") as f:
        json.dump(row, f, indent=2)
    return row


def main() -> None:
    args = parse_args()
    log_step("Starting DiFuMo GAT/MLP run")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_dir is None:
        args.run_dir = str(Path("runs") / f"difumo_{args.model}_{args.graph_type}_{stamp}")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.run_dir) / "checkpoints")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = which_device(args.device)
    if device.type == "cuda" and not args.pin_memory:
        args.pin_memory = True
    log_step(f"Resolved device={device}; amp={args.amp and device.type == 'cuda'}")

    log_step(f"Loading/building DiFuMo coefficients from source={args.coeff_source}")
    data = load_difumo_data(args)
    log_step(f"Loaded aligned data: n={len(data.coeffs):,}; dim={data.coeffs.shape[1]}")
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr: Optional[torch.Tensor] = None
    graph_info: dict[str, float] = graph_stats(edge_index, edge_attr)
    if args.model in {"gat", "gcn", "sage", "deepset"}:
        log_step(f"Building graph type={args.graph_type}")
        edge_index, edge_attr, graph_info = build_graph(data, args)
        log_step(
            f"Graph ready: edges={int(graph_info['number_of_edges']):,}; "
            f"avg_degree={graph_info['average_degree']:.1f}; "
            f"components={int(graph_info['connected_components'])}"
        )
    use_edge_attr = edge_attr is not None and edge_attr.numel() > 0 and args.model in {"gat", "gcn"}
    in_dim = 1 + (3 if args.add_centroids else 0)
    extra = data.centroids if args.add_centroids else None

    if args.model == "mlp":
        train_ds = make_tensor_dataset(data, data.train_idx)
        val_ds = make_tensor_dataset(data, data.val_idx)
        test_ds = make_tensor_dataset(data, data.test_idx)
        test_graph_ds = DifumoGraphDataset(data.coeffs, data.text, data.pmids, data.test_idx, edge_index, edge_attr, extra)
    else:
        train_ds = DifumoGraphDataset(data.coeffs, data.text, data.pmids, data.train_idx, edge_index, edge_attr, extra)
        val_ds = DifumoGraphDataset(data.coeffs, data.text, data.pmids, data.val_idx, edge_index, edge_attr, extra)
        test_ds = DifumoGraphDataset(data.coeffs, data.text, data.pmids, data.test_idx, edge_index, edge_attr, extra)
        test_graph_ds = test_ds

    log_step(f"Building model={args.model}; conv={args.conv}; batch_size={args.batch_size}")
    brain_encoder, text_proj = build_model(args, in_dim=in_dim, use_edge_attr=use_edge_attr)
    config_payload = vars(args).copy()
    config_payload.update(
        {
            "device_resolved": str(device),
            "amp_enabled": bool(args.amp and device.type == "cuda"),
            "n_aligned_pairs": int(len(data.coeffs)),
            "split_train": int(len(data.train_idx)),
            "split_val": int(len(data.val_idx)),
            "split_test": int(len(data.test_idx)),
            "brain_parameters": count_parameters(brain_encoder),
            "text_parameters": count_parameters(text_proj),
        }
    )
    with (run_dir / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)
    with (run_dir / "graph_stats.json").open("w") as f:
        json.dump(graph_info, f, indent=2)

    print("\n== DiFuMo Dataset ==")
    print(f"  aligned pairs={len(data.coeffs):,} train={len(data.train_idx):,} val={len(data.val_idx):,} test={len(data.test_idx):,}")
    print("\n== Graph ==")
    print(json.dumps(graph_info, indent=2))
    print("\n== Model ==")
    print(f"  model={args.model} conv={args.conv} out_dim={args.out_dim}")
    print(f"  brain params={count_parameters(brain_encoder):,} text params={count_parameters(text_proj):,}")
    print(f"  device={device} amp={args.amp and device.type == 'cuda'} batch={args.batch_size}")

    log_step("Starting training loop")
    trainer = DifumoTrainer(brain_encoder, text_proj, args, device)
    trainer.fit(train_ds, val_ds)
    log_step("Training finished; restoring best checkpoint from RAM")
    trainer.restore_best()

    all_metrics: dict[str, dict[str, float]] = {}
    eval_payloads = {}
    log_step("Evaluating validation and test splits")
    for name, split_ds in [("val", val_ds), ("test", test_ds)]:
        metrics, brain_emb, text_emb = trainer.evaluate(split_ds)
        all_metrics[name] = metrics
        eval_payloads[name] = (metrics, brain_emb, text_emb)
        print(
            f"\n{name.upper()} n={len(split_ds):,} "
            f"AUC={metrics['mean_auc']:.4f} "
            f"r@1={metrics['mean_recall@1']:.4f} "
            f"r@5={metrics['mean_recall@5']:.4f} "
            f"r@10={metrics['mean_recall@10']:.4f} "
            f"r@50={metrics['mean_recall@50']:.4f} "
            f"MRR={metrics['mean_mrr']:.4f} "
            f"median_rank={metrics['mean_median_rank']:.1f} "
            f"random_r@10={metrics['mean_random_recall@10']:.4f}"
        )

    with (run_dir / "training_history.json").open("w") as f:
        json.dump(trainer.history, f, indent=2)
    with (run_dir / "eval_results.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    test_metrics, test_brain, test_text = eval_payloads["test"]
    curve_df = recall_curve_frame(test_text, test_brain)
    curve_df.to_csv(run_dir / "test_recall_curve.csv", index=False)
    cov = test_graph_ds.covariate_frame()
    cov.to_csv(run_dir / "test_difumo_covariates.csv", index=False)
    diag_df = retrieval_diagnostics(test_text, test_brain, cov)
    diag_df.to_csv(run_dir / "test_retrieval_diagnostics.csv", index=False)
    save_embedding_correlations(run_dir, test_brain, cov)
    if args.save_plots:
        log_step("Saving plots and diagnostics")
        save_plots(run_dir, trainer.history, curve_df, diag_df, test_brain, args.umap)

    semantic_summary = None
    if getattr(args, "semantic_eval", False):
        try:
            from experiments.semantic_model_eval import (
                run_difumo_network_labeling,
                run_embedding_semantic_evaluations,
            )

            log_step("Running standard semantic evaluation suite")
            semantic_summary = run_embedding_semantic_evaluations(
                model_name=f"difumo_{args.model}_{args.graph_type}",
                brain_embeddings=test_brain,
                text_embeddings=test_text,
                raw_text_embeddings=data.text[data.test_idx],
                pmids=data.pmids[data.test_idx.numpy()],
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
                    "preprocessing_type": "difumo_coefficients",
                    "graph_type": args.graph_type if args.model in {"gat", "gcn", "sage"} else "none",
                },
            )
            network_metrics = run_difumo_network_labeling(
                trainer=trainer,
                args=args,
                data=data,
                edge_index=edge_index,
                edge_attr=edge_attr,
                extra_node_feats=extra,
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
            with (run_dir / "main_comparison_summary_row.json").open("w") as f:
                json.dump(semantic_summary, f, indent=2)
            pd.DataFrame([semantic_summary]).to_csv(run_dir / "main_comparison_summary_row.csv", index=False)
        except Exception as exc:
            print(f"WARNING: semantic evaluation suite failed: {exc}", flush=True)

    log_step("Writing final manifests and comparison rows")
    comparison_row = append_comparison_row(args, test_metrics, graph_info, trainer)
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(Path(args.checkpoint_dir)),
        "best_checkpoint": str(Path(args.checkpoint_dir) / "best_difumo_gat.pt"),
        "last_checkpoint": str(Path(args.checkpoint_dir) / "last_difumo_gat.pt"),
        "config": str(run_dir / "config.json"),
        "graph_stats": str(run_dir / "graph_stats.json"),
        "training_history": str(run_dir / "training_history.json"),
        "eval_results": str(run_dir / "eval_results.json"),
        "test_recall_curve": str(run_dir / "test_recall_curve.csv"),
        "test_covariates": str(run_dir / "test_difumo_covariates.csv"),
        "test_retrieval_diagnostics": str(run_dir / "test_retrieval_diagnostics.csv"),
        "embedding_covariate_correlations": str(run_dir / "embedding_covariate_correlations.csv"),
        "training_curves_plot": str(run_dir / "training_curves.png"),
        "recall_curve_plot": str(run_dir / "recall_curve.png"),
        "umap_diagnostics_plot": str(run_dir / "umap_diagnostics.png"),
        "comparison_file": str(Path(args.comparison_file)),
        "comparison_jsonl": str(Path(args.comparison_file).with_suffix(".jsonl")),
        "comparison_row": str(run_dir / "comparison_row.json"),
        "coefficient_cache": str(Path(args.coeff_cache_file)),
        "test_metrics": test_metrics,
        "comparison_row_payload": comparison_row,
        "semantic_summary": semantic_summary,
    }
    with (run_dir / "artifacts_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nArtifacts saved to {run_dir}")
    print(f"Comparison row appended to {args.comparison_file}")
    print(f"Artifact manifest saved to {run_dir / 'artifacts_manifest.json'}")


if __name__ == "__main__":
    main()
