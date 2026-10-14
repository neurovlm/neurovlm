#!/usr/bin/env python
"""Text-to-brain baselines: figure, tables and paired tests from the run_text_to_brain.py scores.

NeuroQuery, Text2Brain and Neurosynth are used as published (not retrained); their corpora include test papers, which can only
favor them, so every method is scored again on the papers outside those corpora (text_to_brain_baselines_subsets.csv).  No
dagger marks: the overlap is stated in the captions (overlap_note(), also the table caption).

    .venv/bin/python scripts/figures/review/core/a1_text_to_brain_baselines/make_figure.py
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
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "pipeline"))
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths
BLUE, GRAY, LIGHT = "#2a78d6", "#7d7b75", "#b9b7b0"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
FIG_METHODS = [  # (method key, label, colour), top to bottom
    ("neurovlm", "NeuroVLM", BLUE),
    ("text2brain", "Text2Brain", GRAY),
    ("neuroquery", "NeuroQuery", GRAY),
    ("neurosynth", "Neurosynth", GRAY),
    ("ridge", "Linear regression (ridge)", GRAY),
    ("atlas", "Atlas-based labeling", GRAY),
    ("mean_domain", "Domain mean map", GRAY),
    ("mean_train", "Global mean map", GRAY),
    ("null_shuffled", "Shuffled null", LIGHT),
]
LABELS = {k: lab for k, lab, _ in FIG_METHODS} | {"neurovlm_v1": "NeuroVLM, notebook head (replaced)",
                                                  "nn_text": "Nearest training paper (SPECTER2)"}
REF = "neurovlm"
# methods whose training corpus includes test papers -> per-paper flag column of per_query_metrics.parquet
TRAINED_ON = {"text2brain": "in_neuroquery_corpus", "neuroquery": "in_neuroquery_corpus", "neurosynth": "in_neurosynth_db"}
SUBSETS = [("all", None), ("outside NeuroQuery corpus", "in_neuroquery_corpus"),
           ("outside NeuroQuery corpus and Neurosynth", ("in_neuroquery_corpus", "in_neurosynth_db"))]
MAIN_METRICS = [("paper_dice", "Dice"), ("paper_pearson", "Pearson r")]
ALL_METRICS = MAIN_METRICS + [("dice_top10_raw", "Dice top 10 % (raw map)"), ("pearson_raw", "Pearson r (raw map)")]
PANELS = [("paper_dice", "a", "Dice"), ("paper_pearson", "b", "Pearson r")]
N_BOOT, SEED = 2000, 0


def overlap_note(w: dict, sep: str = ",") -> str:
    """Caption sentences on the test papers in the published baselines' corpora (figure and table caption; no daggers)."""
    nq, ns = (int(w[REF][c].sum()) for c in ("in_neuroquery_corpus", "in_neurosynth_db"))
    neither = int((~(w[REF]["in_neuroquery_corpus"].to_numpy(bool) | w[REF]["in_neurosynth_db"].to_numpy(bool))).sum())
    keep = ~(w[REF]["in_neuroquery_corpus"].to_numpy(bool) | w[REF]["in_neurosynth_db"].to_numpy(bool))
    for col, _ in MAIN_METRICS:  # the claim "NeuroVLM still beats all three" on the papers in neither corpus
        assert all(w[REF][col].to_numpy(float)[keep].mean() > w[m][col].to_numpy(float)[keep].mean() for m in TRAINED_ON), col
    f = lambda v: f"{v:,}".replace(",", sep)
    return (f"Neurosynth, NeuroQuery and Text2Brain were used as published; their corpora include {f(ns)} (Neurosynth) or {f(nq)} "
            f"(NeuroQuery, Text2Brain) test papers, which can only favor them. Instead of retraining (Text2Brain's training split is "
            f"unreleased), we scored the {f(neither)} papers in neither corpus: NeuroVLM still beats all three.")


def boot_ci(v: np.ndarray) -> tuple[float, float]:
    idx = np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order, m, out, run = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if not np.any(a != b):
        return 1.0, 1.0
    return float(stats.ttest_rel(a, b).pvalue), float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)


def table(w: dict, methods: list[str], metrics) -> pd.DataFrame:
    """Mean [95% CI] per method and metric, paired p-values vs NeuroVLM (Holm over the other methods)."""
    recs = []
    for m in methods:
        rec = {"method": m, "label": LABELS.get(m, m), "n": len(w[m])}
        for col, _ in metrics:
            v = w[m][col].to_numpy(float)
            rec[col] = float(v.mean())
            rec[f"{col}_ci_lo"], rec[f"{col}_ci_hi"] = boot_ci(v)
        recs.append(rec)
    tab = pd.DataFrame(recs)
    tab["test_papers_in_training_corpus"] = [int(w[m][TRAINED_ON[m]].sum()) if m in TRAINED_ON else 0 for m in methods]
    others = [m for m in methods if m != REF]
    for col, _ in metrics:
        ref = w[REF][col].to_numpy(float)
        pt = [paired(ref, w[m][col].to_numpy(float)) for m in others]
        for kind, j in (("t", 0), ("wilcoxon", 1)):
            tab[f"{col}_p_{kind}_vs_neurovlm"] = tab["method"].map(dict(zip(others, [x[j] for x in pt])))
            tab[f"{col}_p_{kind}_vs_neurovlm_holm"] = tab["method"].map(dict(zip(others, holm([x[j] for x in pt]))))
    return tab


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "savefig.dpi": 300,
    })


def plot(w: dict, path: Path, n: int, paper: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_FULL_W if paper else FIG_LETTER_W, 3.4), sharey=True, layout="constrained")
    y = np.arange(len(FIG_METHODS))[::-1]
    for ax, (col, title, xlabel) in zip(axes, PANELS):
        for yi, (m, _, colour) in zip(y, FIG_METHODS):
            v = w[m][col].to_numpy(float)
            lo, hi = boot_ci(v)
            ax.errorbar(v.mean(), yi, xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=5, color=colour, mec="white", mew=0.8,
                        elinewidth=1.6, capsize=0, zorder=3)
            ax.annotate(f"{v.mean():.3f}", (v.mean(), yi), xytext=(0, 5.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK if m == REF else INK2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(y.min() - 0.5, y.max() + 0.75)
    axes[0].set_yticks(y, [lab for _, lab, _ in FIG_METHODS])  # corpus overlap: stated in the caption (overlap_note), no daggers
    for ext in ("png",) if paper else ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"[figure] {path.with_suffix('.png')} (PubMed test, n = {n:,})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--paper", action="store_true",
                   help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_text_to_brain_baselines.png.")
    args = p.parse_args()
    out = args.out
    style()
    per = pd.read_parquet(out / "per_query_metrics.parquet")
    cfg = json.loads((out / "run_config.json").read_text())
    n = int(cfg["n_test"])
    w = {m: g.set_index("pmid") for m, g in per.groupby("method", sort=False)}
    assert all(w[m].index.equals(w[REF].index) for m in w)
    if args.paper:
        plot(w, PAPER_FIG_DIR / "supp_text_to_brain_baselines", n, paper=True)
        return
    plot(w, out / "text_to_brain_baselines", n)

    keys = [m for m, _, _ in FIG_METHODS]
    tab = table(w, keys, MAIN_METRICS)
    tab.to_csv(out / "text_to_brain_baselines_table.csv", index=False, float_format="%.6g")
    full = table(w, keys + [m for m in w if m not in keys], ALL_METRICS)
    full.to_csv(out / "results_table_full.csv", index=False, float_format="%.6g")
    subs = []
    for name, cols in SUBSETS:
        keep = np.ones(n, bool) if cols is None else ~np.logical_or.reduce([w[REF][c].to_numpy(bool) for c in np.atleast_1d(cols)])
        subs.append(table({m: g[keep] for m, g in w.items()}, keys, MAIN_METRICS).assign(subset=name))
    subs = pd.concat(subs, ignore_index=True)
    subs.to_csv(out / "text_to_brain_baselines_subsets.csv", index=False, float_format="%.6g")

    def tex_p(v: float) -> str:
        if v < 1e-16:
            return "$p < 10^{-16}$"
        if v < 1e-3:
            mant, ex = f"{v:.1e}".split("e")
            return f"$p = {mant}\\times10^{{{int(ex)}}}$"
        return f"$p = {v:.3f}$"

    def cell(r, col):
        v = r[f"{col}_p_t_vs_neurovlm_holm"]
        p = f" ({tex_p(v)})" if np.isfinite(v) else ""
        return f"{r[col]:.3f} [{r[f'{col}_ci_lo']:.3f}, {r[f'{col}_ci_hi']:.3f}]{p}"

    nn = f"{n:,}".replace(",", "{,}")
    lines = [
        "% auto-generated by a1_text_to_brain_baselines/make_figure.py",
        "\\begin{table}[ht]", "\\centering", "\\small",
        f"\\caption{{Text-to-brain generation on the PubMed test set ($n = {nn}$ papers), scored against the autoencoder "
        "reconstruction of each paper's coordinate map. Mean [95\\% bootstrap CI]; $p$ = paired $t$-test against NeuroVLM, "
        f"Holm-adjusted over the {['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'][len(keys) - 1]} other methods. "
        f"{overlap_note(w, '{,}')}}}",
        "\\label{tab:t2b_baselines}", "\\begin{tabular}{lcc}", "\\toprule", "Method & Dice & Pearson $r$ \\\\", "\\midrule",
    ]
    for _, r in tab.iterrows():
        lines.append(" & ".join([r["label"]] + [cell(r, c) for c, _ in MAIN_METRICS]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "text_to_brain_baselines_table.tex").write_text("\n".join(lines))

    ords = []
    for col, name in ALL_METRICS[:3]:
        block = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                va, vb = w[a][col].to_numpy(float), w[b][col].to_numpy(float)
                pt, pw = paired(va, vb)
                block.append({"metric": col, "metric_name": name, "a": a, "b": b, "mean_a": va.mean(), "mean_b": vb.mean(),
                              "diff_a_minus_b": va.mean() - vb.mean(), "frac_papers_a_gt_b": float((va > vb).mean()), "p_t": pt, "p_wilcoxon": pw})
        bd = pd.DataFrame(block)
        bd["p_t_holm"], bd["p_wilcoxon_holm"] = holm(bd["p_t"]), holm(bd["p_wilcoxon"])
        ords.append(bd)
    pd.concat(ords, ignore_index=True).to_csv(out / "ordering_tests.csv", index=False, float_format="%.6g")
    show = ["label"] + [c for col, _ in ALL_METRICS for c in (col, f"{col}_p_t_vs_neurovlm_holm")]
    print(full[show].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(subs[["subset", "label", "n", "test_papers_in_training_corpus", "paper_dice", "paper_pearson"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"[caption] {overlap_note(w)}")


if __name__ == "__main__":
    main()
