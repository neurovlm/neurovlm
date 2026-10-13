#!/usr/bin/env python
"""A7 figure (outputs/a7_ood.{png,svg}) and tables from run_analysis.py's outputs.

Figure and outputs/a7_ood_table.{csv,tex}: cosine scores (higher = more like the training data) and the share of inputs flagged below
the 5th percentile of the held-out score.  outputs/a7_ood_table_sparsity.{csv,tex}: the same map flags with the sparsity controls.  CPU, ~2 s.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths
from run_analysis import MAP_SETS, TEXT_SETS  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

MAP_SCORE, TEXT_SCORE = "recon", "contrastive_maha"  # previous detectors (sparsity table)
COS = {"map": "cos_recon", "text": "max_cos_specter2"}  # figure scores: cos(map, its reconstruction); text: nearest training abstract
ID = {"map": "id_test", "text": "id_test_text"}
NAME = {"id_test": "held-out test maps", "rot90": "rotated 90°", "rotation": "random rotation", "spin": "surface spin",
        "shuffle": "voxel shuffle", "noise": "white noise", "x100": "map × 100", "x0.01": "map × 0.01",
        "id_test_text": "held-out test abstracts", "wikitext": "Wikipedia (WikiText)", "recipes": "cooking recipes"}
# blue = in-distribution (held out), gray = OOD
BLUE, GRAY = "#2a78d6", "#c3c2b7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#a9a7a1", "axes.linewidth": 0.8,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})


def detector_rows(det: pd.DataFrame) -> pd.DataFrame:
    """One row per OOD set: reconstruction error for maps, text-embedding Mahalanobis for text."""
    d = pd.concat([det[(det["modality"] == "map") & (det["score"] == MAP_SCORE)],
                   det[(det["modality"] == "text") & (det["score"] == TEXT_SCORE)]])
    return d.set_index("set").loc[list(MAP_SETS + TEXT_SETS)].reset_index()


def gate_figure(df: pd.DataFrame, flag: pd.DataFrame, out: Path, paper: bool = False) -> None:
    """a: cosine score per input set (blue = held-out, gray = OOD), dashed = flag threshold; b: share flagged per set."""
    groups = [("map", (ID["map"], *MAP_SETS)), ("text", (ID["text"], *TEXT_SETS))]
    ypos, y = {}, 0.0
    for _, sets in groups:
        for s in sets:
            ypos[s], y = y, y - 1
        y -= 0.5
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_FULL_W, 3.7) if paper else (FIG_LETTER_W, 3.4), sharey=True, layout="constrained",
                                     gridspec_kw={"width_ratios": [1.25 if paper else 1.15, 1]})
    note = 7 if paper else 8  # font size of the two notes
    for modality, sets in groups:
        f = flag[flag["score"] == COS[modality]].set_index("set")
        for s in sets:
            v = df.loc[df["set"] == s, COS[modality]].to_numpy()
            colour = BLUE if s == ID[modality] else GRAY
            ax_a.boxplot(v, positions=[ypos[s]], orientation="horizontal", widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                         boxprops={"facecolor": colour, "edgecolor": INK2, "linewidth": 0.7}, medianprops={"color": INK, "linewidth": 1.2},
                         whiskerprops={"color": INK2, "linewidth": 0.7}, capprops={"color": INK2, "linewidth": 0.7})
            pct = 100 * f.loc[s, "flagged"]
            ax_b.barh(ypos[s], pct, height=0.6, color=colour, zorder=2)
            ax_b.text(max(pct, 5) + 1.5, ypos[s], f"{pct:.0f}%", ha="left", va="center", fontsize=8, color=INK2)
        thr = f["threshold_id5"].iloc[0]
        assert np.isclose(thr, np.quantile(df.loc[df["set"] == ID[modality], COS[modality]], 0.05), atol=1e-4)
        ax_a.plot([thr, thr], [ypos[sets[-1]] - 0.45, ypos[sets[0]] + 0.45], ls="--", color=INK, lw=0.9, zorder=4)
    top = ypos[ID["map"]] + 0.5
    map_thr = flag.loc[flag["set"] == ID["map"], "threshold_id5"].iloc[0]
    ax_a.text(map_thr + 0.015, top + 0.05,
              "flag threshold: 5th\npercentile of the\nheld-out inputs" if paper else "flag threshold: 5th percentile\nof the held-out inputs",
              ha="left", va="bottom", fontsize=note, color=INK2)
    ax_a.set_xlim(-0.05, 1.05)  # cosine range [0, 1] with matplotlib's default 5 % margins
    if paper:
        ax_a.set_xticks(np.linspace(0, 1, 6))  # the letter's ticks
    ax_a.set_xlabel("cosine similarity\n(higher = more like training data)\nmaps: to own reconstruction;\ntext: to nearest training abstract" if paper
                    else "cosine similarity (higher = more like training data)\nmaps: to own reconstruction; text: to nearest training abstract")
    ax_a.set_title("a  Score per input", loc="left", fontweight="bold")
    rows = [s for _, sets in groups for s in sets]
    ax_a.set_yticks([ypos[s] for s in rows], [NAME[s] for s in rows])
    ax_a.tick_params(axis="y", length=0)
    ax_b.axvline(5, ls="--", color=INK, lw=0.9, zorder=3)
    ax_b.text(6.5, top + 0.05, "5% = false alarms\non held-out data", ha="left", va="bottom", fontsize=note, color=INK2)
    ax_b.set_xlim(0, 110)
    ax_b.set_xticks(np.arange(0, 101, 20))
    ax_b.set_xlabel("inputs flagged (%)")
    ax_b.set_title("b  Inputs flagged", loc="left", fontweight="bold")
    ax_b.tick_params(axis="y", length=0)
    ax_a.set_ylim(min(ypos.values()) - 0.55, top + (2.0 if paper else 1.6))  # room for the notes below the titles
    if paper:
        fig.savefig(PAPER_FIG_DIR / "supp_ood.png")
        return
    for fmt in ("png", "svg"):
        fig.savefig(out / f"a7_ood.{fmt}")
    plt.close(fig)
    print("wrote", out / "a7_ood.png")


def cosine_table(flag: pd.DataFrame, out: Path) -> None:
    """outputs/a7_ood_table.{csv,tex}: the figure's numbers."""
    f = pd.concat([flag[flag["score"] == COS[m]] for m in ("map", "text")])
    tab = pd.DataFrame({
        "input set": [NAME[s] for s in f["set"]], "n": f["n"].astype(int), "median cosine": [f"{v:.2f}" for v in f["median"]],
        "flagged [95% CI]": [f"{100 * r.flagged:.0f}%" + ("" if np.isnan(r.flagged_lo) else f" [{100 * r.flagged_lo:.0f}, {100 * r.flagged_hi:.0f}]")
                             for r in f.itertuples()],
        "AUROC vs held-out": ["--" if np.isnan(v) else f"{v:.2f}" for v in f["auroc"]]})
    tab.to_csv(out / "a7_ood_table.csv", index=False)
    thr = f.groupby("modality")["threshold_id5"].first()
    caption = ("Out-of-distribution warning with cosine scores (final NeuroVLM models; higher = more like the training data). Maps: cosine "
               "between the map and its autoencoder reconstruction. Text: cosine of the SPECTER2 embedding to the nearest of the 23,894 "
               "training abstracts. An input is flagged below the 5th percentile of the held-out score (maps "
               f"{thr['map']:.2f}, text {thr['text']:.2f}; 5\\% false alarms by construction); 95\\% bootstrap CI with held-out and OOD "
               "inputs resampled. The same text score in the shared text-brain space flags fewer off-topic texts ("
               + ", ".join(f"{NAME[r.set]} {100 * r.flagged:.0f}\\%" for r in flag[flag["score"] == "max_cos_shared"].itertuples() if r.set in TEXT_SETS)
               + ").")
    cols = ["Input set", "$n$", "Median cosine", "Flagged [95\\% CI]", "AUROC"]
    tex = ["\\begin{table}[ht]", "\\centering", "\\small", f"\\caption{{{caption}}}", "\\label{tab:a7_ood}", "\\begin{tabular}{lrrlr}",
           "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    for i, r in enumerate(tab.itertuples(index=False)):
        if i == len(MAP_SETS) + 1:
            tex.append("\\midrule")
        tex.append(" & ".join(str(v).replace("°", "$^\\circ$").replace("×", "$\\times$").replace("%", "\\%") for v in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "a7_ood_table.tex").write_text("\n".join(tex))
    print("wrote", out / "a7_ood_table.csv")


def gate_table(df: pd.DataFrame, det: pd.DataFrame, det_all: pd.DataFrame, out: Path) -> None:
    n_id_maps, n_id_text = int((df["set"] == "id_test").sum()), int((df["set"] == "id_test_text").sum())
    nz = df[df["modality"] == "map"].groupby("set")["nonzero"].median()
    rate = det_all[det_all["modality"] == "map"].pivot(index="set", columns="score", values="detection_at_id95")
    is_map = det["modality"] == "map"

    def pct(v):
        return f"{100 * v:.0f}%" if np.isfinite(v) else "--"

    def map_col(score):
        return [pct(rate.loc[s, score]) if mp else "--" for s, mp in zip(det["set"], is_map)]

    tab = pd.DataFrame({
        "OOD set": [NAME[s] for s in det["set"]],
        "detector": ["AE reconstruction error" if mp else "text-embedding Mahalanobis" for mp in is_map],
        "n": det["n_ood"].astype(int),
        "non-zero voxels (median)": [f"{nz[s]:,.0f}" if mp else "--" for s, mp in zip(det["set"], is_map)],
        "flagged at ID 95th pct [95% CI]": [f"{pct(r.detection_at_id95)} [{100 * r.det_lo:.0f}, {100 * r.det_hi:.0f}]" for r in det.itertuples()],
        "sparsity-matched": [pct(v) for v in det["detection_sparsity_matched"]],
        "SSE / non-zero voxels": map_col("recon_per_nz"),
        "SSE / map energy": map_col("recon_per_energy"),
        "error on non-zero voxels": map_col("recon_support"),
        "AE latent norm": map_col("latent_norm_score"),
        "AUROC [95% CI]": [f"{r.auroc:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]" for r in det.itertuples()]})
    tab.to_csv(out / "a7_ood_table_sparsity.csv", index=False)
    thr = det.groupby("modality")["id95_threshold"].first()
    caption = ("Warning signal for out-of-distribution inputs (final NeuroVLM models). Primary number: share of OOD inputs flagged above the "
               f"95th percentile of the held-out test score (maps n = {n_id_maps:,}, threshold {thr['map']:.2f}; abstracts n = {n_id_text:,}, "
               f"threshold {thr['text']:.1f}; 5\\% false alarms by construction), with a 95\\% bootstrap CI (test and OOD inputs resampled). "
               "Maps are scored by the autoencoder reconstruction error, 1 $-$ cos(map, reconstruction); texts by the Mahalanobis distance "
               "(Ledoit--Wolf) of their shared-space embedding to the 23,894 training abstracts. Sparsity controls for maps (non-zero voxel = "
               f"above 0.1\\% of the map maximum; test-map median {nz['id_test']:,.0f}): sparsity-matched = each map against the 95th percentile "
               "of the test maps in its own quintile of non-zero-voxel count (white noise is denser than every test map and is compared with "
               "the densest quintile); the reconstruction error recomputed as squared error (SSE) per non-zero voxel, as SSE per map energy "
               "$\\|x\\|^2$, and as 1 $-$ cos on the map's non-zero voxels only. AE latent norm = $|\\log(\\|z\\| / \\mathrm{median}_"
               "\\mathrm{train}\\|z\\|)|$ of the autoencoder latent. AUROC is secondary. Rescaled maps also fail a value-range check: every "
               "training map lies in [0, 1] and peaks at 1.")
    cols = ["OOD set", "$n$", "non-zero vox.", "flagged [95\\% CI]", "sparsity-matched", "SSE / non-zero", "SSE / energy", "non-zero only",
            "latent norm", "AUROC [95\\% CI]"]
    tex = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", "\\label{tab:a7_ood_sparsity}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{lrrlllllll}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule",
           "\\multicolumn{10}{l}{\\emph{Maps: autoencoder reconstruction error}} \\\\"]
    for i, r in tab.drop(columns="detector").iterrows():
        if i == len(MAP_SETS):
            tex.append("\\multicolumn{10}{l}{\\emph{Text: Mahalanobis distance of the text embedding}} \\\\")
        cells = [str(v).replace("°", "$^\\circ$").replace("×", "$\\times$").replace("%", "\\%") for v in r.values]
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    (out / "a7_ood_table_sparsity.tex").write_text("\n".join(tex))
    print("wrote", out / "a7_ood_table_sparsity.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Directory with run_analysis.py's outputs.")
    p.add_argument("--paper", action="store_true", help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_ood.png.")
    args = p.parse_args()
    out = args.out
    df = pd.read_parquet(out / "scores.parquet")
    det_all = pd.read_csv(out / "detection.csv")
    flag = pd.read_csv(out / "flagged_cosine.csv")
    if args.paper:
        gate_figure(df, flag, out, paper=True)
        return 0
    gate_figure(df, flag, out)
    cosine_table(flag, out)
    gate_table(df, detector_rows(det_all), det_all, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
