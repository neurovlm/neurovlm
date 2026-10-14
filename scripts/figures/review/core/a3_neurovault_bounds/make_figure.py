#!/usr/bin/env python
"""NeuroVault bounds figure (upper bound, lower bound, multi-target) and per-map-type table from run_analysis.py.

    .venv/bin/python scripts/figures/review/core/a3_neurovault_bounds/make_figure.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "pipeline"))
from common import FIG_LETTER_W  # noqa: E402  response letter text width (6.5 in)
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
MAP_TYPE_SHORT = {"T map": "T", "Z map": "Z", "F map": "F", "univariate-beta map": "beta, univariate",
                  "multivariate-beta map": "beta, multivariate", "P map (given null hypothesis)": "p",
                  "1-P map (\"inverted\" probability)": "1 − p", "other: t, z, or F": "Other, t, z, or F",
                  "other: effect estimate": "Other, effect estimate", "other: component or gradient": "Other, component or gradient"}
HANDLING = {"positive tail (paper)": "positive tail", "inverted (1 - p) before normalisation": "inverted (1 − p)",
            "used directly": "used as is"}


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def figure(curves: pd.DataFrame, out: Path) -> None:
    """Recall curve per target rule (solid) and its chance curve (dashed, same colour): each paper's text given to another
    paper (20 derangements), then the same best- / worst- / every-map rule. The best of several maps scores high even for an
    unrelated text (chance above the diagonal), the worst scores low (below); every map has no selection, so chance is the
    diagonal from brain to text and above it from text to brain, where any of the paper's maps is a hit. The paper shows the
    same curves in Fig. 5b (scripts/figures/main/fig5_perf_quant.py), so this figure is the letter's only."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_LETTER_W, 3.9), sharey=True, layout="constrained")
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
    fig.legend(handles, labels, loc="outside lower center", ncol=2, handlelength=2.4, columnspacing=2.0)
    for ext in ("png", "svg"):
        fig.savefig(out / f"neurovault_bounds.{ext}")
    plt.close(fig)
    print(f"[figure] {out / 'neurovault_bounds.png'}")


def map_type_table(per_type: pd.DataFrame, counts: pd.DataFrame, metrics: pd.DataFrame, cfg: dict, out: Path) -> None:
    """Per-map-type AUCs of the per-map evaluation (every map its own target): declared NeuroVault types first, then the
    maps labeled Other by their inferred kind. Chance = the same evaluation with texts shuffled across papers."""
    pt = per_type[per_type["variant"] == VARIANT].pivot_table(index="map_type", columns="direction", values="recall_curve_auc")
    c = counts.assign(other=counts["map_type"].str.startswith("other")).sort_values(["other", "n_images"], ascending=[True, False]).set_index("map_type")
    pt = pt.reindex(index=c.index, columns=["text_to_brain", "brain_to_text"])
    chance = metrics[(metrics["variant"] == VARIANT) & (metrics["definition"] == "per_image")].set_index("direction")["chance_recall_curve_auc"]
    df = pd.DataFrame({
        "Map type": [MAP_TYPE_SHORT.get(t, t) for t in c.index],
        "Maps": [f"{v:,}" for v in c["n_images"]],
        "Papers": [f"{v:,}" for v in c["n_publications"]],
        "Handling": [HANDLING.get(h.split(" | ")[0], h.split(" | ")[0]) for h in c["preprocessing"]],
        "AUC text → brain": ["–" if pd.isna(pt.loc[t, 'text_to_brain']) else f"{pt.loc[t, 'text_to_brain']:.2f}" for t in c.index],
        "AUC brain → text": ["–" if pd.isna(pt.loc[t, 'brain_to_text']) else f"{pt.loc[t, 'brain_to_text']:.2f}" for t in c.index],
    })
    note = (f"AUC of the final model when every map is a target, scored per map. From text to brain, a paper's text ranks each "
            f"of its maps among the maps of other papers. From brain to text, each map ranks the {cfg['n_publications']} texts, as in "
            f"the bounds figure. Chance is this setting's null, texts shuffled across papers ({chance['text_to_brain']:.2f} and "
            f"{chance['brain_to_text']:.2f}). Other marks maps their uploaders labeled Other, typed from file names, descriptions, "
            "and values. Masks, parcellations, anatomical and variance maps, mean images, and untyped maps were excluded.")
    df.to_csv(out / "map_type_table.csv", index=False)
    (out / "map_type_table.md").write_text(markdown(df) + "\n" + note + "\n")
    tex = ["% written by a3_neurovault_bounds/make_figure.py", "\\begin{table}[ht]", "\\centering", "\\small",
           f"\\caption{{\\textbf{{NeuroVault results by map type.}} {note}}}", "\\label{tab:neurovault_map_types}",
           "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{3.6cm}rrlcc}", "\\toprule",
           " & & & & \\multicolumn{2}{c}{\\textbf{AUC}} \\\\\\cmidrule(lr){5-6}",
           "\\textbf{Map type} & \\textbf{Maps} & \\textbf{Papers} & \\textbf{Handling} & \\textbf{text $\\rightarrow$ brain} & "
           "\\textbf{brain $\\rightarrow$ text} \\\\", "\\midrule"]
    tex += [" & ".join(str(v) for v in row).replace("−", "$-$") + " \\\\" for row in df.itertuples(index=False)]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "map_type_table.tex").write_text("\n".join(tex))
    print(markdown(df))
    print(f"[caption] {note}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    args = p.parse_args()
    out = args.out
    metrics = pd.read_csv(out / "bounds_metrics.csv")
    cfg = json.loads((out / "config.json").read_text())
    figure(pd.read_csv(out / "recall_curves.csv"), out)
    map_type_table(pd.read_csv(out / "per_map_type_metrics.csv"), pd.read_csv(out / "map_type_counts.csv"), metrics, cfg, out)
    shown = {(k[d], d): rule for rule, (_, _, k) in RULES.items() for d in DIRECTIONS}
    metrics = metrics[[(d, r) in shown for d, r in zip(metrics["definition"], metrics["direction"])]].copy()
    metrics.insert(0, "rule", [shown[(d, r)] for d, r in zip(metrics["definition"], metrics["direction"])])
    cols = ["rule", "variant", "direction", "definition", "n_queries", "mean_pool_size", "recall@1", "recall@5", "recall@10", "mrr",
            "median_rank", "recall_curve_auc", "chance_recall_curve_auc"]
    metrics[cols].to_csv(out / "bounds_table.csv", index=False, float_format="%.3f")
    bounds_tex(metrics[metrics["variant"] == VARIANT], cfg, out)


def bounds_tex(m: pd.DataFrame, cfg: dict, out: Path) -> None:
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
           f"\\caption{{NeuroVault retrieval ({cfg['n_publications']} papers, {cfg['n_images']:,} maps) with three choices of target per paper. Positive-tail "
           "preprocessing with p-maps inverted; pool = mean number of candidates per query; chance: texts shuffled across papers, "
           "then the same best- or worst-map selection. Every preprocessing variant is in bounds\\_table.csv.}", "\\label{tab:neurovault_bounds}",
           "\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
           "Target & Direction & Queries & Pool & R@1 & R@5 & R@10 & MRR & Median rank & AUC & Chance AUC \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "bounds_table.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    main()
