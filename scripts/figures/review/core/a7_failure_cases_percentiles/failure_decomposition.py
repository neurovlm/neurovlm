#!/usr/bin/env python
"""Failure decomposition (2.12): data failures versus model failures among the test maps whose caption fails.

A failure is a test map whose stage-2 caption has a negative contrastive cosine with the paper's title [SEP] summary
(b2t_text_cos < 0 in outputs/pubmed_items.parquet, written by run_analysis.py). Each failure, and each map of a random
control of as many non-failures (seed 0), gets two data flags:
  a  poor input map: the Fig. 3 rule fails on the input map (>= 2 clusters, half the mass in <= 3 pieces, <= 10 % of it
     in specks; top 5 % of voxels, 26-connected, pieces of >= 20 voxels) or the coordinate table has <= 3 foci.
  b  text-map mismatch in the data: the paper's own title [SEP] abstract and its own map have a contrastive cosine below
     the test-set 10th percentile (stored SPECTER2 embedding and autoencoder latent through the two contrastive heads,
     the space of the retrieval analysis; nothing is generated).
Categories: a only, b only, a+b, and c = neither (clean map, consistent pair, caption still negative: a model miss).
Check that uses no part of NeuroVLM: Neurosynth's map for the same title and abstract (a1_text_to_brain_baselines)
against the paper's map, Pearson r below its test-set 10th percentile.

    .venv/bin/python scripts/figures/review/core/a7_failure_cases_percentiles/failure_decomposition.py   # CPU, ~15 s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _hf_pins import hf_file, load_masker  # noqa: E402
from _review_common import DEFAULT_MODEL_DIR, component, heldout_pmids, load_pubmed_maps, run_file, set_model_dir, shared_brain, shared_text  # noqa: E402

A1 = HERE.parent / "a1_text_to_brain_baselines" / "outputs" / "per_query_metrics.parquet"
TOP, MIN_CLUSTER = 0.05, 20  # Fig. 3: drawn voxels (top 5 %) and the smallest piece that is not a speck
CLEAN = {"clusters": 2, "k50": 3, "speckle": 0.10}
MAX_FEW_FOCI = 3
PCT = 10  # mismatch: true-pair cosine below this test-set percentile
CATS = [("a", "Poor input map only"), ("b", "Text-map mismatch only"), ("a+b", "Both"), ("c", "Neither (model miss)")]


def map_clusters(v: np.ndarray, mask3d: np.ndarray) -> tuple[int, int, float]:
    """Fig. 3 rule on one map: clusters >= MIN_CLUSTER voxels, pieces holding half the mass, mass share of specks."""
    from scipy import ndimage

    keep = (v >= np.quantile(v, 1.0 - TOP)) & (v > 0)
    vol = np.zeros(mask3d.shape, np.float32)
    vol[mask3d] = np.where(keep, v, 0.0)
    lab, n = ndimage.label(vol > 0, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, 0, 0.0
    mass = np.asarray(ndimage.sum_labels(vol, lab, index=np.arange(1, n + 1)))
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    share = np.sort(mass)[::-1].cumsum() / mass.sum()
    return int((size >= MIN_CLUSTER).sum()), int(np.searchsorted(share, 0.5) + 1), float(mass[size < MIN_CLUSTER].sum() / mass.sum())


def true_pair_cosine(pmids: np.ndarray) -> np.ndarray:
    """Contrastive cosine of each paper's title [SEP] abstract with its own map."""
    import torch

    lat = torch.load(run_file("pmids_split").parent / "latent_neuro.pt", map_location="cpu", weights_only=False)
    spe = torch.load(hf_file("neurovlm/embedded_text", "latent_specter2_adhoc.pt"), map_location="cpu", weights_only=False)
    pos_l = {int(p): i for i, p in enumerate(np.asarray(lat["pmid"]))}
    pos_s = {int(p): i for i, p in enumerate(np.asarray(spe["pmid"]))}
    z_t = shared_text(component("proj_head_text_infonce", "cpu"), spe["latent"][[pos_s[p] for p in pmids]], "cpu")
    z_m = shared_brain(component("proj_head_image_infonce", "cpu"), lat["latent"][[pos_l[p] for p in pmids]], "cpu")
    return (z_t * z_m).sum(1).numpy()


def table(d: pd.DataFrame, groups: dict) -> pd.DataFrame:
    rows = []
    for key, label in CATS + [("ab", "Any data flag (a, b or both)"), ("b_any", "Text-map mismatch (b or both)"),
                              ("a_any", "Poor input map (a or both)"), ("ns", "Check: Neurosynth r below its 10th percentile")]:
        sel = {"ab": d["category"] != "c", "b_any": d["mismatch"], "a_any": d["poor_map"], "ns": d["neurosynth_low"]}.get(key, d["category"] == key)
        row = {"category": key, "label": label}
        for g, m in groups.items():
            row[f"{g}_n"], row[f"{g}_pct"] = int((sel & m).sum()), 100 * (sel & m).sum() / m.sum()
        row["test_maps_n"], row["test_maps_failing_pct"] = int(sel.sum()), 100 * d.loc[sel, "failure"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    set_model_dir(args.model_dir)

    d = pd.read_parquet(args.out / "pubmed_items.parquet", columns=["pmid", "title", "b2t_text_cos", "b2t_caption_map_cos"])
    pmids = d["pmid"].to_numpy(np.int64)
    assert np.array_equal(pmids, heldout_pmids("test", exclusions=True, log=print)[0])
    d["failure"] = d["b2t_text_cos"] < 0

    coords = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "coordinates.parquet"), columns=["pmid"])
    d["n_foci"] = coords.groupby("pmid").size().reindex(pmids).to_numpy()
    mask3d = np.asarray(load_masker().mask_img_.dataobj).astype(bool)
    X = load_pubmed_maps(pmids).numpy()
    d[["clusters", "k50", "speckle"]] = [map_clusters(v, mask3d) for v in X]
    d["clean_map"] = (d["clusters"] >= CLEAN["clusters"]) & (d["k50"] <= CLEAN["k50"]) & (d["speckle"] <= CLEAN["speckle"])
    d["poor_map"] = ~d["clean_map"] | (d["n_foci"] <= MAX_FEW_FOCI)

    d["true_pair_cos"] = true_pair_cosine(pmids)
    d["mismatch"] = d["true_pair_cos"] < np.percentile(d["true_pair_cos"], PCT)
    d["category"] = np.select([d["poor_map"] & d["mismatch"], d["poor_map"], d["mismatch"]], ["a+b", "a", "b"], "c")

    ns = pd.read_parquet(A1).query("method == 'neurosynth'").set_index("pmid")["pearson_raw"].reindex(pmids).to_numpy()
    d["neurosynth_r"] = ns
    d["neurosynth_low"] = ns < np.nanpercentile(ns, PCT)

    rest = np.flatnonzero(~d["failure"].to_numpy())
    d["control"] = False
    d.loc[np.sort(np.random.default_rng(args.seed).choice(rest, int(d["failure"].sum()), replace=False)), "control"] = True
    tab = table(d, {"failures": d["failure"], "control": d["control"]})

    d.to_parquet(args.out / "failure_decomposition_items.parquet", index=False)
    tab.to_csv(args.out / "failure_decomposition.csv", index=False, float_format="%.1f")
    nf, nc = int(d["failure"].sum()), int(d["control"].sum())
    md = [f"# Failure decomposition (test set, n = {len(d):,})", "",
          f"Failures: stage-2 caption with a negative contrastive cosine to the paper's text ({nf} maps). Control: {nc} random "
          f"non-failing maps (seed {args.seed}). a = poor input map (Fig. 3 rule fails or <= {MAX_FEW_FOCI} foci), b = the paper's "
          f"own text and map below the test-set {PCT}th percentile of true-pair cosine "
          f"({np.percentile(d['true_pair_cos'], PCT):.3f}), c = neither. Last column: share of all test maps of that kind whose "
          "caption fails.", "",
          "| | Failures | Control | Test maps failing |", "|---|---|---|---|"]
    md += [f"| {r.label} | {r.failures_n} ({r.failures_pct:.1f} %) | {r.control_n} ({r.control_pct:.1f} %) | "
           f"{r.test_maps_failing_pct:.1f} % of {r.test_maps_n:,} |" for r in tab.itertuples()]
    med = d.groupby(["failure", "category"])["b2t_caption_map_cos"].median().unstack()
    md += ["", "Median contrastive cosine of the caption with its own map:", "",
           "| | " + " | ".join(k for k, _ in CATS) + " |", "|---" * (len(CATS) + 1) + "|"]
    md += [f"| {name} | " + " | ".join(f"{med.loc[f, k]:.3f}" for k, _ in CATS) + " |" for f, name in ((True, "Failures"), (False, "Non-failures"))]
    (args.out / "failure_decomposition.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
