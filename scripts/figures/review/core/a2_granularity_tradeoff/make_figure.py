#!/usr/bin/env python
"""Reviewer 3.2 figure from the outputs of run_analysis.py: a) captions with study-specific details, b) retrieval vs generation
(cosine to the paper's own text; dashed = another paper's text), c) closeness to the training texts (collapse check).

Writes outputs/a2_granularity_tradeoff.{png,svg}. CPU, a few seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "pipeline"))
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths
GRAY, BLUE, MID_BLUE, PALE_BLUE, INK, INK2, GRID = "#898781", "#2a78d6", "#7fb0ea", "#c4dbf6", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#a9a7a1", "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK2, "pdf.fonttype": 42, "savefig.dpi": 300,
})
DOMAINS = {"pubmed": "PubMed", "neurovault": "NeuroVault"}
# caption source -> (legend label, colour); gray = retrieval baseline, blues = generated (darkest = the paper's captions)
STYLE = {"retrieved": ("nearest training caption (retrieval)", GRAY), "stage1": ("stage 1", PALE_BLUE),
         "stage2_raw": ("stage 2 without projection", MID_BLUE), "stage2": ("stage 2 (paper)", BLUE),
         "real": ("real held-out text", "white")}


def positions(n: int, width: float) -> list[list[float]]:
    """x of each of ``n`` adjacent boxes or bars in every domain group (groups are 1 apart)."""
    return [[g + (j - (n - 1) / 2) * width * 1.12 for j in range(n)] for g in range(len(DOMAINS))]


def boxes(ax, keys, values, width):
    for xs, vals in zip(positions(len(keys), width), values):
        for x, k, v in zip(xs, keys, vals):
            bp = ax.boxplot([v], positions=[x], widths=width, patch_artist=True, showfliers=False, whis=(5, 95),
                            medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8),
                            boxprops=dict(lw=0.8))
            bp["boxes"][0].set_facecolor(STYLE[k][1])
            bp["boxes"][0].set_edgecolor(INK if k == "real" else INK2)
    ax.set_xticks(range(len(DOMAINS)), list(DOMAINS.values()))
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--paper", action="store_true",
                   help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_granularity.png.")
    args = p.parse_args()
    out = args.out
    per = {(d, m): pd.read_parquet(out / f"per_text_{d}_{m}.parquet") for d in DOMAINS for m in ("stage1", "stage2", "stage2_raw", "retrieved")}
    coll = {d: pd.read_parquet(out / f"collapse_{d}.parquet") for d in DOMAINS}
    res = json.loads((out / "metrics.json").read_text())["results"]
    if args.paper:
        plt.rcParams["xtick.labelsize"] = 7  # PubMed and NeuroVault fit side by side
    fig, axes = plt.subplots(1, 3, figsize=(FIG_FULL_W, 3.4) if args.paper else (FIG_LETTER_W, 3.05), layout="constrained",
                             gridspec_kw={"width_ratios": [0.95, 1.1, 1.2] if args.paper else [1.2, 0.95, 1.1]})

    # a: share of captions with >= 1 study-specific detail (population, age group, task, sample size, ...)
    ax, keys, w = axes[0], ["retrieved", "stage1", "stage2_raw", "stage2"], 0.2
    for xs, d in zip(positions(len(keys), w), DOMAINS):
        for x, k in zip(xs, keys):
            v = 100 * res[d][k]["lex_any_frac"]
            ax.bar(x, v, width=w, color=STYLE[k][1], zorder=2)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7 if args.paper else 8, color=INK2)
    ax.set_xticks(range(len(DOMAINS)), list(DOMAINS.values()))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 70)
    ax.set_ylabel("captions naming a study detail (%)")
    ax.set_title("a  Study-specific\n    details", loc="left", fontweight="bold")

    # b: retrieval vs generation, cosine to the paper's own text; dashed = mean cosine to another paper's text
    ax, keys, w = axes[1], ["retrieved", "stage2"], 0.36
    boxes(ax, keys, [[per[(d, k)]["minilm_cos_truth"].to_numpy() for k in keys] for d in DOMAINS], w)
    for xs, d in zip(positions(len(keys), w), DOMAINS):
        for x, k in zip(xs, keys):
            ax.hlines(per[(d, k)]["minilm_cos_truth_null"].mean(), x - w / 2, x + w / 2, colors=INK, linestyles=(0, (2, 1.5)), lw=1.2, zorder=5)
    ax.set_ylabel("cosine to the paper's own text")
    ax.set_title("b  Retrieval vs\n    generation", loc="left", fontweight="bold")

    # c: closeness to the training texts (collapse check)
    ax, keys = axes[2], ["stage1", "stage2_raw", "stage2", "real"]
    boxes(ax, keys, [[coll[d].loc[coll[d]["model"] == k, "nn_cos"].to_numpy() for k in keys] for d in DOMAINS], 0.19)
    ax.set_ylabel("cosine to the nearest training text")
    ax.set_title("c  Closeness to\n    training texts", loc="left", fontweight="bold")

    handles = [Patch(facecolor=c, edgecolor=INK if k == "real" else "none", label=lab) for k, (lab, c) in STYLE.items()]
    handles.append(Line2D([], [], color=INK, ls=(0, (2, 1.5)), lw=1.2, label="another paper's text (b, null)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2 if args.paper else 3, handlelength=1.4, columnspacing=1.6)
    if args.paper:
        fig.savefig(PAPER_FIG_DIR / "supp_granularity.png")
        return
    for ext in ("png", "svg"):
        fig.savefig(out / f"a2_granularity_tradeoff.{ext}")
    plt.close(fig)
    print(f"[figure] {out / 'a2_granularity_tradeoff.png'}")


if __name__ == "__main__":
    main()
