#!/usr/bin/env python
"""A7 figure (outputs/a7_ood.{png,svg}) and tables from run_analysis.py's outputs.

Figure and outputs/a7_ood_table.{csv,tex}: the two map scores and the text score per input set, and the share of inputs flagged at
thresholds fixed on the validation split.  outputs/a7_ood_table_sparsity.{csv,tex}: the map flags with the sparsity controls.  CPU, ~2 s.
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
sys.path.insert(0, str(HERE.parents[3] / "pipeline"))
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths
from run_analysis import FLAG_Q, MAP_SETS, TEXT_SETS  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

SIM = {"map": "corr_recon", "text": "max_cos_specter2"}  # panel a: map vs its reconstruction (Pearson r); text vs nearest training abstract
DIST = "encoder_maha"  # panel b: Mahalanobis distance of the map's autoencoder latent from the training latents
ID = {"map": "id_test", "text": "id_test_text"}
VAL = {"map": "id_val", "text": "id_val_text"}
NAME = {"id_val": "validation maps", "id_test": "held-out test maps", "rot90": "rotated 90°", "rotation": "random rotation",
        "spin": "surface spin", "shuffle": "voxel shuffle", "noise": "white noise", "x100": "map × 100", "x0.01": "map × 0.01",
        "id_val_text": "validation abstracts", "id_test_text": "held-out test abstracts", "wikitext": "Wikipedia (WikiText)",
        "recipes": "cooking recipes"}
SHORT = {"id_test": "test maps", "id_test_text": "test abstracts", "wikitext": "WikiText", "recipes": "recipes"}  # --paper (5.15 in)
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


def box(ax, v, y, colour) -> None:
    ax.boxplot(v, positions=[y], orientation="horizontal", widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
               boxprops={"facecolor": colour, "edgecolor": INK2, "linewidth": 0.7}, medianprops={"color": INK, "linewidth": 1.2},
               whiskerprops={"color": INK2, "linewidth": 0.7}, capprops={"color": INK2, "linewidth": 0.7})


def gate_figure(df: pd.DataFrame, flag: pd.DataFrame, path: Path, paper: bool = False) -> None:
    """a: similarity per input set, b: latent distance per map set, c: share flagged (either map score, or the text score).

    Blue = held-out test inputs, gray = OOD; dashed = thresholds fixed on the validation split; boxes 5th-95th percentile whiskers."""
    groups = [("map", (ID["map"], *MAP_SETS)), ("text", (ID["text"], *TEXT_SETS))]
    ypos, y = {}, 0.0
    for _, sets in groups:
        for s in sets:
            ypos[s], y = y, y - 1
        y -= 0.5
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(FIG_FULL_W, 4.3) if paper else (FIG_LETTER_W, 3.5), sharey=True, layout="constrained",
                                           gridspec_kw={"width_ratios": [1.15, 1, 0.85]})
    f = flag.set_index("set")
    for modality, sets in groups:
        for s in sets:
            colour = BLUE if s == ID[modality] else GRAY
            box(ax_a, df.loc[df["set"] == s, SIM[modality]].to_numpy(), ypos[s], colour)
            if modality == "map":
                box(ax_b, df.loc[df["set"] == s, DIST].to_numpy(), ypos[s], colour)
            pct = 100 * f.loc[s, "flagged"]
            ax_c.barh(ypos[s], pct, height=0.6, color=colour, zorder=2)
            ax_c.text(max(pct, 5) + 3, ypos[s], f"{pct:.0f}%", ha="left", va="center", fontsize=8, color=INK2)
        span = [ypos[sets[-1]] - 0.45, ypos[sets[0]] + 0.45]
        ax_a.plot([f.loc[VAL[modality], f"threshold_{SIM[modality]}"]] * 2, span, ls="--", color=INK, lw=0.9, zorder=4)
        if modality == "map":
            ax_b.plot([f.loc[VAL[modality], f"threshold_{DIST}"]] * 2, span, ls="--", color=INK, lw=0.9, zorder=4)
    top = ypos[ID["map"]] + 0.5
    ax_a.text(-0.1, top + 0.05, "dashed: thresholds\nfrom validation" if paper else "dashed: flag thresholds,\nset on validation inputs",
              ha="left", va="bottom", fontsize=8, color=INK2)
    ax_a.set_xlim(-0.15, 1.05)
    ax_a.set_xticks(np.linspace(0, 1, 6) if not paper else [0, 0.5, 1])
    ax_a.set_xlabel("similarity\n(maps: r with own\nreconstruction; text:\ncosine to nearest\nabstract)" if paper else
                    "similarity to training data\nmaps: r with own reconstruction\ntext: cosine to nearest abstract")
    ax_a.set_title("a  Similarity", loc="left", fontweight="bold")
    rows = [s for _, sets in groups for s in sets]
    ax_a.set_yticks([ypos[s] for s in rows], [(SHORT.get(s, NAME[s]) if paper else NAME[s]) for s in rows])
    ax_a.tick_params(axis="y", length=0)
    ax_b.set_xscale("log")
    ax_b.set_xlim(5, 6000)
    ax_b.set_xticks([10, 100, 1000], ["10", "100", "1,000"])
    ax_b.minorticks_off()
    ax_b.set_xlabel("latent distance\n(Mahalanobis,\nlog scale)" if paper else "Mahalanobis distance of\nthe map's latent from the\ntraining latents (log scale)")
    ax_b.set_title("b  Distance" if paper else "b  Latent distance", loc="left", fontweight="bold")
    ax_b.tick_params(axis="y", length=0)
    ax_c.axvline(5, ls="--", color=INK, lw=0.9, zorder=3)
    ax_c.text(8, top + 0.05, "5% of\nvalidation" if paper else "5% of validation\ninputs", ha="left", va="bottom", fontsize=8, color=INK2)
    ax_c.set_xlim(0, 125)
    ax_c.set_xticks(np.arange(0, 101, 50))
    ax_c.set_xlabel("flagged (%)\n(maps: low r\nor far latent;\ntext: low cosine)" if paper else "inputs flagged (%)\nmaps: low r or far latent\ntext: low cosine")
    ax_c.set_title("c  Flagged", loc="left", fontweight="bold")
    ax_c.tick_params(axis="y", length=0)
    ax_a.set_ylim(min(ypos.values()) - 0.55, top + 1.6)  # room for the notes below the titles
    if paper:
        fig.savefig(path)
        print("wrote", path)
        return
    for fmt in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print("wrote", path.with_suffix(".png"))


def detector_table(flag: pd.DataFrame, out: Path) -> None:
    """outputs/a7_ood_table.{csv,tex}: the figure's numbers, validation rows included."""
    def num(v, fmt):
        return "--" if not np.isfinite(v) else format(v, fmt)

    tab = pd.DataFrame({
        "input set": [NAME[s] for s in flag["set"]], "n": flag["n"].astype(int),
        "median similarity": [f"{v:.2f}" for v in np.where(flag["modality"] == "map", flag["median_corr_recon"], flag["median_max_cos_specter2"])],
        "median latent distance": [num(v, ".1f") for v in flag["median_encoder_maha"]],
        "flagged [95% CI]": [f"{100 * r.flagged:.0f}%" + ("" if np.isnan(r.flagged_lo) else f" [{100 * r.flagged_lo:.0f}, {100 * r.flagged_hi:.0f}]")
                             for r in flag.itertuples()],
        "AUROC vs test": [num(v, ".2f") for v in flag["auroc"]]})
    tab.to_csv(out / "a7_ood_table.csv", index=False)
    t = flag.set_index("set")
    caption = ("Out-of-distribution warning (final NeuroVLM models), thresholds fixed on the validation split. A map is flagged when "
               f"its Pearson correlation with its autoencoder reconstruction is below {t.loc['id_val', 'threshold_corr_recon']:.2f} or the "
               "Mahalanobis distance of its autoencoder latent from the 23,894 training latents is above "
               f"{t.loc['id_val', 'threshold_encoder_maha']:.1f} (each flags {100 * FLAG_Q:g}\\% of validation maps). A text is flagged "
               "when the cosine of its SPECTER2 embedding to the nearest training abstract is below "
               f"{t.loc['id_val_text', 'threshold_max_cos_specter2']:.2f} (5\\% of validation abstracts). 95\\% bootstrap CI with validation, "
               "test, and OOD inputs resampled. AUROC of the larger validation tail share of the scores, test inputs against each OOD set.")
    cols = ["Input set", "$n$", "Median similarity", "Median latent distance", "Flagged [95\\% CI]", "AUROC"]
    tex = ["\\begin{table}[ht]", "\\centering", "\\small", f"\\caption{{{caption}}}", "\\label{tab:a7_ood}", "\\begin{tabular}{lrrrlr}",
           "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    for i, r in enumerate(tab.itertuples(index=False)):
        if i == len(MAP_SETS) + 2:
            tex.append("\\midrule")
        tex.append(" & ".join(str(v).replace("°", "$^\\circ$").replace("×", "$\\times$").replace("%", "\\%") for v in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "a7_ood_table.tex").write_text("\n".join(tex))
    print("wrote", out / "a7_ood_table.csv")


def gate_table(df: pd.DataFrame, det: pd.DataFrame, out: Path) -> None:
    """outputs/a7_ood_table_sparsity.{csv,tex}: the map similarity score with sparsity controls and other reconstruction errors."""
    nz = df[df["modality"] == "map"].groupby("set")["nonzero"].median()
    rate = det[det["modality"] == "map"].pivot(index="set", columns="score", values="flagged")
    d = pd.concat([det[(det["modality"] == "map") & (det["score"] == SIM["map"])], det[(det["modality"] == "text") & (det["score"] == SIM["text"])]])
    d = d.set_index("set").loc[list(MAP_SETS + TEXT_SETS)].reset_index()
    is_map = d["modality"] == "map"

    def pct(v):
        return f"{100 * v:.0f}%" if np.isfinite(v) else "--"

    def map_col(score):
        return [pct(rate.loc[s, score]) if mp else "--" for s, mp in zip(d["set"], is_map)]

    tab = pd.DataFrame({
        "OOD set": [NAME[s] for s in d["set"]],
        "n": d["n_ood"].astype(int),
        "non-zero voxels (median)": [f"{nz[s]:,.0f}" if mp else "--" for s, mp in zip(d["set"], is_map)],
        "flagged at validation 5th pct [95% CI]": [f"{pct(r.flagged)} [{100 * r.flagged_lo:.0f}, {100 * r.flagged_hi:.0f}]" for r in d.itertuples()],
        "sparsity-matched": [pct(v) for v in d["flagged_sparsity_matched"]],
        "1 - cos (previous)": map_col("recon"),
        "SSE / non-zero voxels": map_col("recon_per_nz"),
        "SSE / map energy": map_col("recon_per_energy"),
        "error on non-zero voxels": map_col("recon_support"),
        "latent Mahalanobis": map_col("encoder_maha"),
        "AUROC [95% CI]": [f"{r.auroc:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]" for r in d.itertuples()]})
    tab.to_csv(out / "a7_ood_table_sparsity.csv", index=False)
    thr = d.groupby("modality")["threshold_val"].first()
    caption = ("Single scores at thresholds fixed on the validation split (final NeuroVLM models). Primary column: share of OOD inputs "
               f"below the 5th percentile of the validation score (maps: Pearson $r$ with the autoencoder reconstruction, threshold {thr['map']:.2f}; "
               f"abstracts: cosine to the nearest training abstract, threshold {thr['text']:.2f}; 5\\% of validation inputs), with a 95\\% "
               "bootstrap CI (validation, test, and OOD inputs resampled). Sparsity-matched = each map against the validation threshold of "
               f"its own quintile of non-zero-voxel count (non-zero = above 0.1\\% of the map maximum; test-map median {nz['id_test']:,.0f}). "
               "The other map columns flag above the validation 95th percentile of 1 $-$ cos(map, reconstruction) (the previous score), "
               "the squared reconstruction error (SSE) per non-zero voxel, SSE per map energy $\\|x\\|^2$, 1 $-$ cos on the map's non-zero "
               "voxels, and the Mahalanobis distance of the autoencoder latent. AUROC: held-out test inputs against each OOD set.")
    cols = ["OOD set", "$n$", "non-zero vox.", "flagged [95\\% CI]", "sparsity-matched", "1 $-$ cos", "SSE / non-zero", "SSE / energy",
            "non-zero only", "latent Mahal.", "AUROC [95\\% CI]"]
    tex = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", "\\label{tab:a7_ood_sparsity}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{lrrllllllll}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule",
           "\\multicolumn{11}{l}{\\emph{Maps: correlation with the autoencoder reconstruction}} \\\\"]
    for i, r in tab.iterrows():
        if i == len(MAP_SETS):
            tex.append("\\multicolumn{11}{l}{\\emph{Text: cosine to the nearest training abstract}} \\\\")
        tex.append(" & ".join(str(v).replace("°", "$^\\circ$").replace("×", "$\\times$").replace("%", "\\%") for v in r.values) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    (out / "a7_ood_table_sparsity.tex").write_text("\n".join(tex))
    print("wrote", out / "a7_ood_table_sparsity.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Directory with run_analysis.py's outputs.")
    p.add_argument("--paper", action="store_true", help="Draw at the paper's text width (FIG_FULL_W) and write only <paper-dir>/supp_ood.png.")
    p.add_argument("--paper-dir", type=Path, default=PAPER_FIG_DIR, help="Directory of the --paper figure (default: paper/figures).")
    args = p.parse_args()
    out = args.out
    df = pd.read_parquet(out / "scores.parquet")
    flag = pd.read_csv(out / "flagged.csv")
    if args.paper:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        gate_figure(df, flag, args.paper_dir / "supp_ood.png", paper=True)
        return 0
    gate_figure(df, flag, out / "a7_ood")
    detector_table(flag, out)
    gate_table(df, pd.read_csv(out / "detection.csv"), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
