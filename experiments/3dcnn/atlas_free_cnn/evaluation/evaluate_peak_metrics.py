"""Peak/coordinate metrics for generated PubMed ALE maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def top_voxel_mni(pred: torch.Tensor, affine: np.ndarray, *, n_peaks: int = 20) -> np.ndarray:
    arr = pred.squeeze().detach().cpu().float()
    flat_idx = torch.topk(arr.flatten(), min(n_peaks, arr.numel())).indices.cpu().numpy()
    ijk = np.column_stack(np.unravel_index(flat_idx, arr.shape))
    homog = np.c_[ijk, np.ones(len(ijk))]
    return (affine @ homog.T).T[:, :3]


def peak_recall(pred_peaks_mni: np.ndarray, true_coords_mni: np.ndarray, radius_mm: float) -> float:
    if len(true_coords_mni) == 0 or len(pred_peaks_mni) == 0:
        return float("nan")
    d = np.linalg.norm(true_coords_mni[:, None, :] - pred_peaks_mni[None, :, :], axis=-1)
    return float((d.min(axis=1) <= radius_mm).mean())


def load_pubmed_coordinates() -> dict[str, np.ndarray]:
    from neurovlm.retrieval_resources import _load_pubmed_coordinates

    df = _load_pubmed_coordinates().copy()
    df["pmid"] = df["pmid"].astype(str)
    return {pmid: grp[["x", "y", "z"]].to_numpy(float) for pmid, grp in df.groupby("pmid")}


def evaluate_peak_metrics(pred: torch.Tensor, pmids: list[str], affine: np.ndarray, *, n_peaks: int = 20) -> dict[str, float]:
    coord_lookup = load_pubmed_coordinates()
    recalls = {10: [], 15: [], 20: []}
    false_positive_counts = []
    for i, pmid in enumerate(pmids):
        true_coords = coord_lookup.get(str(pmid), np.empty((0, 3)))
        pred_peaks = top_voxel_mni(pred[i], affine, n_peaks=n_peaks)
        for radius in recalls:
            recalls[radius].append(peak_recall(pred_peaks, true_coords, radius))
        if len(true_coords):
            d = np.linalg.norm(pred_peaks[:, None, :] - true_coords[None, :, :], axis=-1)
            false_positive_counts.append(float((d.min(axis=1) > 20.0).sum()))
    out = {f"peak_recall_{r}mm": float(np.nanmean(v)) for r, v in recalls.items()}
    out["false_positive_peak_count"] = float(np.nanmean(false_positive_counts)) if false_positive_counts else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--pmids-json", required=True)
    p.add_argument("--affine-json", required=True)
    p.add_argument("--output", default="experiments/3dcnn/atlas_free_cnn/outputs/eval/peak_metrics.json")
    args = p.parse_args()
    pred = torch.load(args.pred, map_location="cpu", weights_only=False)
    pmids = json.load(open(args.pmids_json))
    affine = np.asarray(json.load(open(args.affine_json)), dtype=float)
    metrics = evaluate_peak_metrics(pred, pmids, affine)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(args.output, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
