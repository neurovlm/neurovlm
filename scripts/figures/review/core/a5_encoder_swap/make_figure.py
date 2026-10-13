#!/usr/bin/env python
"""SPECTER2 vs mean word2vec, heads retrained identically (reviewer 3.3): <out>/metrics.json ->
<out>/a5_encoder_swap.{png,svg} (retrieval AUC; text-to-brain Dice and Pearson r) and <out>/a5_encoder_swap_table.{csv,tex}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # scripts/
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths
ENCODERS = {"specter2": ("SPECTER2", "#2a78d6"), "word2vec": ("word2vec", "#898781")}  # blue = paper, gray = baseline
LEGEND = {"specter2": "SPECTER2 language model (paper)", "word2vec": "mean word2vec (context-free baseline)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK2,
    "xtick.major.size": 0, "savefig.dpi": 300,
})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT, help="Directory with metrics.json (run_analysis.py); outputs are written here.")
    p.add_argument("--paper", action="store_true",
                   help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_encoder_swap.png.")
    args = p.parse_args()
    out = args.out
    res = json.loads((out / "metrics.json").read_text())["results"]
    assert all(res[k]["text_to_brain"]["dice_pct"] == 70 for k in ENCODERS), "Dice must use the paper definition (70th pct)"
    vals = {k: {"AUC text→brain": res[k]["retrieval"]["t2i_auc"], "AUC brain→text": res[k]["retrieval"]["i2t_auc"],
                "Dice": res[k]["text_to_brain"]["dice_matched"], "Pearson r": res[k]["text_to_brain"]["pearson_matched"]}
            for k in ENCODERS}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_FULL_W, 2.5) if args.paper else (FIG_LETTER_W, 2.6), layout="constrained")
    panels = [("a  Retrieval", ["AUC text→brain", "AUC brain→text"], ["text → brain", "brain → text"], "retrieval, recall-curve AUC", 1.0),
              ("b  Text-to-brain generation", ["Dice", "Pearson r"], ["Dice overlap", "Pearson r"], "agreement with the true map", 0.6)]
    w = 0.36
    for ax, (title, cols, ticks, ylabel, ymax) in zip(axes, panels):
        x = np.arange(len(cols))
        for j, (k, (name, color)) in enumerate(ENCODERS.items()):
            v = [vals[k][c] for c in cols]
            xs = x + (j - 0.5) * (w + 0.02)
            ax.bar(xs, v, width=w, color=color, label=LEGEND[k])
            for xi, vi in zip(xs, v):
                ax.text(xi, vi + ymax * 0.015, f"{vi:.2f}", ha="center", va="bottom", color=INK)
        ax.set_xticks(x, ticks)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].axhline(0.5, color=INK2, lw=0.8, ls="--", zorder=0.5)
    axes[0].text(1.95 if args.paper else 1.8, 0.51, "chance", ha="right", va="bottom", color=INK2)
    axes[0].set_xlim(-0.55, 1.97 if args.paper else 1.82)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, handlelength=1.2,
               columnspacing=1.0 if args.paper else 2.0)
    if args.paper:
        fig.savefig(PAPER_FIG_DIR / "supp_encoder_swap.png")
        return
    for ext in ("png", "svg"):
        fig.savefig(out / f"a5_encoder_swap.{ext}")
    plt.close(fig)

    table = pd.DataFrame([{"Encoder": ENCODERS[k][0], **{c: f"{v:.3f}" for c, v in vals[k].items()}} for k in ENCODERS])
    table.to_csv(out / "a5_encoder_swap_table.csv", index=False)
    tex = ["% written by a5_encoder_swap/make_figure.py", "\\begin{table}[ht]", "\\centering",
           "\\caption{SPECTER2 vs a context-free baseline (mean word2vec embeddings of title + abstract). The text encoder is "
           "frozen and the projection heads are retrained for each on the final model's autoencoder latents with its recipe "
           "(text-to-brain head v2). PubMed test split ($n = 2{,}971$). "
           "AUC: recall-curve AUC of contrastive retrieval. Dice (both maps thresholded at their 70th percentile) and voxel "
           "Pearson $r$: generated vs autoencoder-reconstructed true map.}", "\\label{tab:context_free_baseline}",
           "\\begin{tabular}{lrrrr}", "\\toprule",
           "Encoder & AUC text$\\rightarrow$brain & AUC brain$\\rightarrow$text & Dice & Pearson $r$ \\\\", "\\midrule"]
    tex += [" & ".join(row) + " \\\\" for row in table.astype(str).itertuples(index=False)]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "a5_encoder_swap_table.tex").write_text("\n".join(tex))
    print(f"[figure] {out / 'a5_encoder_swap.png'}\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
