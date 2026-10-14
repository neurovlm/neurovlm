#!/usr/bin/env python
"""Figure for reviewer 3.5: share of correct caption titles per token and map set, with chance (titles shuffled).

Reads outputs/correctness_table.csv from run_correctness.py (CPU, ~2 s):
  .venv/bin/python scripts/figures/review/core/a6_steering_tokens/plot_correctness.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _a6_common as C  # noqa: E402

sys.path.insert(0, str(HERE.parents[3] / "pipeline"))  # scripts/pipeline/
from common import FIG_LETTER_W  # noqa: E402  response letter text width (6.5 in)

TOKENS = {"[NETWORK]": ("a", "names the map's network"), "[REGION]": ("b", "names a region in the map"),
          "[FUNCTION]": ("c", "matches a decoded function")}
SETS = {"networks": "atlas\nnetworks", "ica": "ICA\ncomponents", "pubmed": "PubMed\ntest maps"}
BLUE = "#2a78d6"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="run_correctness.py outputs; the figure is written there.")
    out = p.parse_args().out
    t = pd.read_csv(out / "correctness_table.csv")
    C.apply_style(8.0)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(FIG_LETTER_W, 2.6), sharey=True, layout="constrained")
    x, width = np.arange(len(SETS)), 0.62
    for ax, (token, (letter, desc)) in zip(axes, TOKENS.items()):
        g = t[t["token"] == token].set_index("set").loc[list(SETS)]
        ax.bar(x, 100 * g["hit_rate"], width, color=BLUE, zorder=2)
        for xi, h, ch in zip(x, g["hit_rate"], g["chance"]):
            ax.text(xi, 100 * h + 1.5, f"{100 * h:.0f}", ha="center", va="bottom", fontsize=8, color=C.INK)
            ax.plot([xi - width / 2, xi + width / 2], [100 * ch, 100 * ch], color=C.INK, lw=1.2, ls=(0, (2, 1.5)), zorder=3)
        ax.set_xticks(x, [f"{lab}\n(n = {int(n)})" for lab, n in zip(SETS.values(), g["n"])])
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{letter}  {token}", loc="left", fontweight="bold", fontsize=9, pad=14)
        ax.text(0.0, 1.02, desc, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=C.INK)
    axes[0].set_ylabel("captions judged correct (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    fig.legend([plt.Line2D([], [], color=C.INK, lw=1.2, ls=(0, (2, 1.5)))], ["chance (captions shuffled across maps)"],
               loc="outside lower center", handlelength=2.0)
    plt.rcParams["savefig.bbox"] = "standard"  # no tight crop: saved at exactly FIG_LETTER_W
    for fmt in ("png", "svg"):
        fig.savefig(out / f"steering_correctness.{fmt}", dpi=300)
    print("wrote", out / "steering_correctness.png")


if __name__ == "__main__":
    main()
