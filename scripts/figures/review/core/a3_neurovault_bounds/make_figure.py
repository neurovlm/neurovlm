#!/usr/bin/env python
"""NeuroVault bounds figure (upper bound, lower bound, multi-target) and per-map-type table from run_analysis.py.

    .venv/bin/python scripts/figures/review/core/a3_neurovault_bounds/make_figure.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "savefig.dpi": 300,
})
VARIANT = "pos"  # positive tail, p-maps inverted
RULES = {  # rule -> (label, colour, definition per direction)
    "upper": ("best map per paper (upper bound)", BLUE, {"text_to_brain": "paper_upper", "brain_to_text": "paper_upper"}),
    "lower": ("worst map per paper (lower bound)", ORANGE, {"text_to_brain": "worst_image", "brain_to_text": "worst_image"}),
    "multi": ("every map of a paper is a target", AQUA, {"text_to_brain": "oracle_any_true", "brain_to_text": "per_image"}),
}
DIRECTIONS = {"text_to_brain": "a  Text → brain", "brain_to_text": "b  Brain → text"}
MAP_TYPE_SHORT = {"univariate-beta map": "beta (univariate)", "multivariate-beta map": "beta (multivariate)",
                  "P map (given null hypothesis)": "p map", "1-P map (\"inverted\" probability)": "1 − p map"}


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def figure(curves: pd.DataFrame, out: Path, paper: bool = False) -> None:
    """Recall curve per target rule (solid) and its chance curve (dashed, same colour): texts shuffled across papers, then the same
    best- / worst-map selection, which is why chance is above the diagonal for the upper bound and below it for the lower bound."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_FULL_W, 3.15) if paper else (FIG_LETTER_W, 3.9), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)
    for ax, (direction, title) in zip(axes, DIRECTIONS.items()):
        c = curves[(curves["variant"] == VARIANT) & (curves["direction"] == direction)]
        for rule, (label, color, keys) in RULES.items():
            d = c[c["definition"] == keys[direction]].sort_values("normalized_k")
            ax.plot(d["normalized_k"], d["recall"], color=color, lw=1.8, label=label)
            ax.plot(d["normalized_k"], d["chance"], color=color, lw=1.2, ls=(0, (4, 2.5)))
        ax.set_box_aspect(1)  # square panel; default data margins
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel("k/n (fraction of all candidates)")
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel("recall (correct item in the top k)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (4, 2.5))))
    labels.append("chance (texts shuffled, same target rule)")
    fig.legend(handles, labels, loc="outside lower center", ncol=2, handlelength=2.4, columnspacing=1.0 if paper else 2.0,
               fontsize=7 if paper else None)
    if paper:
        fig.savefig(PAPER_FIG_DIR / "supp_neurovault_bounds.png")
        return
    for ext in ("png", "svg"):
        fig.savefig(out / f"neurovault_bounds.{ext}")
    plt.close(fig)
    print(f"[figure] {out / 'neurovault_bounds.png'}")


def map_type_table(per_type: pd.DataFrame, counts: pd.DataFrame, out: Path) -> None:
    pt = per_type[per_type["variant"] == VARIANT].pivot_table(index="map_type", columns="direction", values="recall_curve_auc")
    c = counts.set_index("map_type").sort_values("n_images", ascending=False)
    pt = pt.reindex(index=c.index, columns=["text_to_brain", "brain_to_text"])
    df = pd.DataFrame({
        "Map type": [MAP_TYPE_SHORT.get(t, t) for t in c.index],
        "Images": c["n_images"].to_numpy(),
        "Papers": c["n_publications"].to_numpy(),
        "Preprocessing before the threshold": c["preprocessing"].str.replace(" | |value| | sign split", "", regex=False).to_numpy(),
        "AUC text → brain": ["–" if pd.isna(pt.loc[t, 'text_to_brain']) else f"{pt.loc[t, 'text_to_brain']:.2f}" for t in c.index],
        "AUC brain → text": ["–" if pd.isna(pt.loc[t, 'brain_to_text']) else f"{pt.loc[t, 'brain_to_text']:.2f}" for t in c.index],
    })
    note = ("Every image is its own target (chance AUC 0.50). Parcellations, anatomical maps, ROI masks and variance maps "
            "are excluded before evaluation; 'other' = maps without a declared type.")
    df.to_csv(out / "map_type_table.csv", index=False)
    (out / "map_type_table.md").write_text(markdown(df) + "\n" + note + "\n")
    tex = ["% written by a3_neurovault_bounds/make_figure.py", "\\begin{table}[ht]", "\\centering", "\\footnotesize",
           f"\\caption{{NeuroVault retrieval per map type. {note}}}", "\\label{tab:neurovault_map_types}",
           "\\begin{tabular}{lrrlcc}", "\\toprule", " & ".join(df.columns).replace("→", "$\\rightarrow$") + " \\\\", "\\midrule"]
    tex += [" & ".join(str(v) for v in row).replace("−", "$-$") + " \\\\" for row in df.itertuples(index=False)]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "map_type_table.tex").write_text("\n".join(tex))
    print(markdown(df))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--paper", action="store_true",
                   help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_neurovault_bounds.png.")
    args = p.parse_args()
    out = args.out
    if args.paper:
        figure(pd.read_csv(out / "recall_curves.csv"), out, paper=True)
        return
    metrics = pd.read_csv(out / "bounds_metrics.csv")
    figure(pd.read_csv(out / "recall_curves.csv"), out)
    map_type_table(pd.read_csv(out / "per_map_type_metrics.csv"), pd.read_csv(out / "map_type_counts.csv"), out)
    shown = {(k[d], d): rule for rule, (_, _, k) in RULES.items() for d in DIRECTIONS}
    metrics = metrics[[(d, r) in shown for d, r in zip(metrics["definition"], metrics["direction"])]].copy()
    metrics.insert(0, "rule", [shown[(d, r)] for d, r in zip(metrics["definition"], metrics["direction"])])
    cols = ["rule", "variant", "direction", "definition", "n_queries", "mean_pool_size", "recall@1", "recall@5", "recall@10", "mrr",
            "median_rank", "recall_curve_auc", "chance_recall_curve_auc"]
    metrics[cols].to_csv(out / "bounds_table.csv", index=False, float_format="%.3f")
    bounds_tex(metrics[metrics["variant"] == VARIANT], out)


def bounds_tex(m: pd.DataFrame, out: Path) -> None:
    """outputs/bounds_table.tex: the three target rules of the figure, both directions."""
    names = {"upper": "upper bound: best image per paper", "lower": "lower bound: worst image per paper",
             "multi": "multi-target: every image a target"}
    rows = []
    for rule in RULES:
        for d in DIRECTIONS:
            r = m[(m["rule"] == rule) & (m["direction"] == d)].iloc[0]
            rows.append([names[rule], d.replace("_to_", " $\\rightarrow$ "), f"{int(r['n_queries']):,}", f"{r['mean_pool_size']:,.0f}",
                         *(f"{r[k]:.3f}" for k in ("recall@1", "recall@5", "recall@10", "mrr")), f"{r['median_rank']:.0f}",
                         f"{r['recall_curve_auc']:.3f}", f"{r['chance_recall_curve_auc']:.3f}"])
    tex = ["% written by a3_neurovault_bounds/make_figure.py", "\\begin{table}[ht]", "\\centering", "\\scriptsize",
           "\\caption{NeuroVault retrieval (305 papers, 3,154 maps) with three choices of target per paper. Positive-tail "
           "preprocessing with p-maps inverted; pool = mean number of candidates per query; chance: texts shuffled across papers, "
           "then the same best- or worst-map selection. Every preprocessing variant is in bounds\\_table.csv.}", "\\label{tab:neurovault_bounds}",
           "\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
           "Target & Direction & Queries & Pool & R@1 & R@5 & R@10 & MRR & Median rank & AUC & Chance AUC \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "bounds_table.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    main()
