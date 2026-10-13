#!/usr/bin/env python
"""Table-caption audit (comment 3.7): the numbers of the response (outputs/caption_numbers.{md,csv}) and the figure
outputs/caption_usability.{png,svg}: (a) caption categories per split, (b) which text predicts the table's dominant region.

Reads the outputs of run_analysis.py and caption_usability.py.  CPU, < 2 s.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from common import FIG_LETTER_W  # noqa: E402  response letter text width (6.5 in)
BLUE, GRAY = "#2a78d6", "#898781"  # blue = the paper's input (title + abstract), gray = everything else
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
CATS = [("contrast/condition", "a contrast or condition"), ("other", "other (measure, method)"), ("boilerplate", "boilerplate"),
        ("region list", "region names only")]


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "xtick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def ci(r) -> str:
    return f"{r['macro_f1']:.3f} [{r['macro_f1_lo']:.3f}–{r['macro_f1_hi']:.3f}]"


def numbers(out: Path, info: pd.DataFrame, dec: pd.DataFrame, js: dict) -> pd.DataFrame:
    cov = pd.read_csv(out / "coverage_summary.csv").set_index("split").loc["all"]
    summ = json.loads((out / "caption_summary.json").read_text())
    cap = summ["tables"]["coordinate tables (pubget)"]
    a = info.set_index("split").loc["all"]
    d = dec.set_index("input")
    w = js["decodability_info"]["within_paper_pair_accuracy_caption"]
    rows = [
        ("Corpus papers with open-access full text (PMC-OA; the only papers whose tables pubget can read)",
         f"{cov['pct_pmc_oa']:.1f} % ({int(cov['n_pmc_oa']):,} of {int(cov['n_pmids']):,})"),
        ("Coordinate tables in those papers / with a caption", f"{int(cov['n_coord_tables']):,} / {cov['pct_coord_tables_captioned']:.1f} %"),
        ("Corpus papers with ≥ 1 captioned coordinate table", f"{cov['pct_corpus_with_captioned_coord_table_observed']:.1f} %"),
        ("Caption length, median words [IQR]",
         f"{cap['caption_words_median']:.0f} [{cap['caption_words_p25']:.0f}–{cap['caption_words_p75']:.0f}] (abstract: {summ['abstract_words_median']:.0f})"),
        ("Captions naming a contrast / condition (rule-based)", f"{a['pct_has_contrast']:.0f} %"),
        ("Captions that are boilerplate (no specific word, or ≤ 3 words and no contrast)", f"{a['pct_is_boilerplate']:.0f} %"),
        ("Captions naming any brain region", f"{a['pct_names_region']:.0f} %"),
        ("Captions naming one of their own top-3 atlas regions (chance)",
         f"{a['pct_region_match_fine']:.1f} % ({a['pct_region_match_fine_chance']:.1f} %)"),
        ("Captions sharing ≥ 2 specific words with their own abstract (chance)", f"{a['pct_task_match']:.0f} % ({a['pct_task_match_chance']:.0f} %)"),
        ("Dominant region from text, macro-F1: caption / title + abstract / chance",
         f"{ci(d.loc['caption'])} / {ci(d.loc['title + abstract'])} / {ci(d.loc['chance (training prior)'])}"),
        ("Two tables of one paper: caption assigns each its own region (chance 0.5)", f"{w[0]:.2f} [{w[1]:.2f}–{w[2]:.2f}]"),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def figure(info: pd.DataFrame, dec: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_LETTER_W, 2.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.6]})
    # (a) caption categories, all coordinate tables (the splits are within 1 point of each other: caption_informativeness.csv)
    a = info.set_index("split").loc["all"]
    y = np.arange(len(CATS))[::-1]
    v = np.array([a[f"pct_cat_{key}"] for key, _ in CATS])
    ax1.barh(y, v, height=0.6, color=GRAY)
    for yi, vi in zip(y, v):
        ax1.text(vi + 1.5, yi, f"{vi:.0f}%", ha="left", va="center", color=INK2)
    ax1.set_yticks(y, [lab for _, lab in CATS])
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel(f"table captions (%, n = {int(a['n_tables']):,})")
    ax1.set_title("a  What table captions say", loc="left", fontweight="bold")
    # (b) share of test-split tables whose dominant atlas region is predicted from each text; dashed = chance (training prior)
    rows = [("caption", "caption", GRAY), ("title + abstract", "title + abstract", BLUE),
            ("caption + title + abstract", "caption + title\n+ abstract", GRAY)]
    d = dec.set_index("input")
    yy = np.arange(len(rows))[::-1]
    vals = np.array([100 * d.loc[k, "top1_acc"] for k, _, _ in rows])
    lo = np.array([100 * d.loc[k, "top1_acc_lo"] for k, _, _ in rows])
    hi = np.array([100 * d.loc[k, "top1_acc_hi"] for k, _, _ in rows])
    ax2.barh(yy, vals, height=0.6, color=[c for _, _, c in rows])
    ax2.errorbar(vals, yy, xerr=[vals - lo, hi - vals], fmt="none", ecolor=INK, elinewidth=0.8, capsize=0)
    for yi, hv, v_ in zip(yy, hi, vals):
        ax2.text(hv + 0.8, yi, f"{v_:.0f}%", ha="left", va="center", color=INK2)
    chance = 100 * d.loc["chance (training prior)", "top1_acc"]
    ax2.axvline(chance, color=INK, lw=0.9, ls="--", zorder=3)
    ax2.text(chance + 0.6, yy[0] + 0.45, f"chance {chance:.0f}%", ha="left", va="bottom", color=INK2)
    ax2.set_ylim(-0.5, len(rows) - 0.2)
    ax2.set_yticks(yy, [lab for _, lab, _ in rows])
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(0, 35)
    ax2.set_xlabel("test tables, main region correct (%)")
    ax2.set_title("b  Text predicts the main region", loc="left", fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"[figure] {path.with_suffix('.png')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory (also the input).")
    out = p.parse_args().out
    style()
    info = pd.read_csv(out / "caption_informativeness.csv")
    dec = pd.read_csv(out / "caption_decodability.csv")
    dec = dec[dec["target"] == "dominant"]
    js = json.loads((out / "caption_usability.json").read_text())
    df = numbers(out, info, dec, js)
    df.to_csv(out / "caption_numbers.csv", index=False)
    (out / "caption_numbers.md").write_text(markdown(df))
    print(markdown(df))
    figure(info, dec, out / "caption_usability")


if __name__ == "__main__":
    main()
