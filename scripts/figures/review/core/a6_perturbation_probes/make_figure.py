#!/usr/bin/env python
"""Perturbation-probe figure: (a) one substitution pair as brain maps, (b) substitution and (c) composition as paired
boxplots; plus table_negation_context.{md,csv}. Reads outputs/ of run_analysis.py. CPU, ~5 s.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _a6_common as C  # noqa: E402

sys.path.insert(0, str(HERE.parents[3] / "pipeline"))  # scripts/pipeline/
from common import FIG_LETTER_W  # noqa: E402  response letter text width (6.5 in)

BLUE, GRAY, INK, INK2, GRID, LINE = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": LINE, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK2,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
PCT = 95.0  # top 5% of each map shown, as in the latent-dimension figure
EXAMPLE = ("auditory/visual", "auditory", "visual")  # substitution pair shown in panel a, Neurosynth terms A, B
# one-sided maps: 'hot' without its black and white ends; differences: blue-red, symmetric around 0
HOT = LinearSegmentedColormap.from_list("hot_mid", plt.get_cmap("hot")(np.linspace(0.22, 0.88, 256)))
DIV = plt.get_cmap("RdBu_r")


def stat_map(ax, vec: np.ndarray, *, z: float, cmap, vmin: float, vmax: float, threshold: float):
    """One axial slice of a mask-space vector on the MNI152 template."""
    from nilearn.plotting import plot_stat_map

    ax.set_axis_off()
    disp = plot_stat_map(C.unmask(vec), axes=ax, display_mode="z", cut_coords=[z], cmap=cmap, vmin=vmin, vmax=vmax,
                         threshold=threshold, symmetric_cbar=vmin < 0, colorbar=False, annotate=False,
                         draw_cross=False, black_bg=False)
    disp.annotate(left_right=True, positions=False, size=8)
    return disp


def peak_z(vec: np.ndarray) -> float:
    img = C.unmask(vec)
    a = np.squeeze(img.get_fdata())
    i, j, k = np.unravel_index(np.argmax(a), a.shape)
    return float((img.affine @ np.array([i, j, k, 1.0]))[2])


def colorbar(fig, rect, cmap, vmin, vmax, ticks, ticklabels, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks(ticks, labels=ticklabels)
    cb.outline.set_visible(False)
    cax.tick_params(length=2, pad=1.5, colors=INK2)
    cb.set_label(label, labelpad=2, color=INK)
    return cb


def fmt_p(p: float) -> str:
    """'p = 1.8 × 10^-12' (mathtext) for small p, else three decimals."""
    if p >= 1e-3:
        return f"p = {p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"p = {mant} × 10$^{{{int(exp)}}}$"


def paired_box(ax, ctrl: np.ndarray, match: np.ndarray, labels: tuple[str, str], p: float, n_note: str) -> None:
    """Gray control vs blue match, one line per item, p-value on top."""
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.06, 0.06, len(ctrl))
    for a, b, j in zip(ctrl, match, jit):
        ax.plot([0 + j, 1 + j], [a, b], color=LINE, lw=0.5, alpha=0.8, zorder=1)
    for x, vals, col in ((0, ctrl, GRAY), (1, match, BLUE)):
        bp = ax.boxplot([vals], positions=[x], widths=0.42, patch_artist=True, showfliers=False, whis=(0, 100),
                        medianprops=dict(color=col, lw=1.6), boxprops=dict(facecolor="none", edgecolor=col, lw=1.0),
                        whiskerprops=dict(color=col, lw=1.0), capprops=dict(color=col, lw=1.0), zorder=2)
        bp["boxes"][0].set_facecolor(matplotlib.colors.to_rgba(col, 0.14))
        ax.scatter(x + jit, vals, s=9, color=col, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xticks([0, 1], labels)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.6, 1.6)
    top = max(ctrl.max(), match.max())
    lo = min(ctrl.min(), match.min())
    h = 0.06 * (top - lo)
    ax.plot([0, 0, 1, 1], [top + h, top + 1.6 * h, top + 1.6 * h, top + h], color=INK2, lw=0.7)
    ax.text(0.5, top + 1.9 * h, fmt_p(p), ha="center", va="bottom", color=INK)
    ax.text(0.5, -0.30, n_note, transform=ax.transAxes, ha="center", va="top", color=INK2)


def negation_context_table(out: Path) -> None:
    neg = pd.read_csv(out / "negation_items.csv")
    ctx = pd.read_csv(out / "context_summary.csv").set_index("model")
    corpus = pd.read_csv(out / "corpus_negation_counts.csv").set_index("pattern")
    tests = json.loads((out / "results.json").read_text())["context_tests"]
    rows = [
        {"Probe": "Negation", "Items": f"{len(neg)} sentences vs their negation",
         "Measure": "r between the two maps (1 = negation ignored)",
         "NeuroVLM": f"{neg['r_aff_neg'].mean():.2f} (range {neg['r_aff_neg'].min():.2f}–{neg['r_aff_neg'].max():.2f})",
         "Comparison": (f"training abstracts: {100 * corpus.loc['negated activation', 'frac_abstracts']:.1f} % negate an activation, "
                        f"{100 * corpus.loc['affirmative activation', 'frac_abstracts']:.0f} % affirm one")},
        {"Probe": "Context", "Items": f"{int(ctx.loc['neurovlm', 'n'])} phrases (5 keywords, 3–5 contexts each)",
         "Measure": "specificity: r with own Neurosynth term − r with the other contexts' terms",
         "NeuroVLM": f"{ctx.loc['neurovlm', 'specificity']:.2f}",
         "Comparison": (f"SPECTER2 ridge {ctx.loc['specter_ridge', 'specificity']:.2f} "
                        f"(p = {tests['specificity_neurovlm_vs_specter_ridge']['p_wilcoxon']:.2f}); word2vec ridge "
                        f"{ctx.loc['w2v_ridge', 'specificity']:.2f} (p = {tests['specificity_neurovlm_vs_w2v_ridge']['p_wilcoxon']:.3f})")},
    ]
    df = pd.DataFrame(rows)
    note = "p: paired Wilcoxon signed-rank test over the 18 phrases, NeuroVLM vs the control."
    df.to_csv(out / "table_negation_context.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (out / "table_negation_context.md").write_text("\n".join(md) + "\n\n" + note + "\n")
    print("\n".join(md) + "\n\n" + note)


def figure(out: Path) -> None:
    d = np.load(out / "maps.npz", allow_pickle=True)
    keys, M = [str(k) for k in d["keys"]], d["maps"].astype(np.float32)
    get = lambda k: M[keys.index(k)]  # noqa: E731
    pair, ta, tb = EXAMPLE
    A, B = get(f"sub|{pair}|A"), get(f"sub|{pair}|B")
    dn, dr = A - B, get(f"ns|{ta}") - get(f"ns|{tb}")
    dn, dr = dn / np.abs(dn).max(), dr / np.abs(dr).max()  # each difference scaled to its own max |value|
    z = round((peak_z(A) + peak_z(B)) / 2)  # one axial slice through both peaks (auditory cortex and visual cortex)
    sub = pd.read_csv(out / "substitution_per_frame.csv")
    comp = pd.read_csv(out / "composition_per_frame.csv")
    tests = json.loads((out / "results.json").read_text())
    r_ex = sub.loc[sub["pair"] == pair, "diffmap_r_ns"].mean()

    W, H = FIG_LETTER_W, 5.45
    fig = plt.figure(figsize=(W, H))
    box = lambda x, top, w, h: fig.add_axes((x / W, (H - top - h) / H, w / W, h / H))  # noqa: E731  (top = inches from top)
    text = lambda x, top, s, **kw: fig.text(x / W, (H - top) / H, s, **kw)  # noqa: E731

    # a: one substitution pair as maps
    text(0.05, 0.05, "a  Substitution example: one word swapped in otherwise identical sentences", fontweight="bold", fontsize=9,
         va="top")
    text(0.2, 0.24, f"NeuroVLM maps averaged over 8 sentence frames; top 5% of voxels of each map shown; axial slice z = {z} mm",
         color=INK2, va="top")
    vmax = float(max(A.max(), B.max()))
    # each map thresholded at its own top 5 %: a threshold pooled over both maps showed 17 % of the broader visual map
    # and 3 % of the auditory map, and the sparse auditory map's top 10 % reaches down to 3 % of its peak (insula, putamen)
    panels = [
        (A, "query → NeuroVLM map", "'auditory stimulation'", HOT, 0.0, vmax, float(np.percentile(A, PCT))),
        (B, "query → NeuroVLM map", "'visual stimulation'", HOT, 0.0, vmax, float(np.percentile(B, PCT))),
        (dn, "NeuroVLM difference", "auditory map − visual map", DIV, -1.0, 1.0, float(np.percentile(np.abs(dn), PCT))),
        (dr, "reference: Neurosynth", "'auditory' − 'visual'", DIV, -1.0, 1.0, float(np.percentile(np.abs(dr), PCT))),
    ]
    centres = [0.85, 2.44, 4.07, 5.65]
    for xc, (vec, l1, l2, cmap, lo, hi, thr) in zip(centres, panels):
        stat_map(box(xc - 0.79, 0.8, 1.58, 1.3), vec, z=z, cmap=cmap, vmin=lo, vmax=hi, threshold=thr)
        text(xc, 0.47, l1, ha="center", va="top", fontweight="bold")
        text(xc, 0.62, l2, ha="center", va="top", color=INK2)
    text(4.86, 2.12, f"r between the two differences = {r_ex:.2f} (8 sentences)", ha="center", va="top")
    ticks = [t for t in (0.0, 0.25, 0.5, 0.75, 1.0) if t <= vmax + 1e-9]
    colorbar(fig, (0.97 / W, (H - 2.47) / H, 1.35 / W, 0.07 / H), HOT, 0, vmax, ticks, [f"{t:g}" for t in ticks],
             "predicted activation (÷ map maximum)")
    colorbar(fig, (4.19 / W, (H - 2.47) / H, 1.35 / W, 0.07 / H), DIV, -1, 1, [-1, 0, 1], ["visual > auditory", "0", "auditory > visual"],
             "difference (÷ max |difference|)")

    # b: substitution, paired
    text(0.05, 3.05, "b  Substitution: 5 word pairs × 8 sentences", fontweight="bold", fontsize=9, va="top")
    ax = box(0.85, 3.45, 2.1, 1.3)
    s = tests["substitution_tests"]["diffmap_matched_vs_otherpairs"]
    paired_box(ax, sub["diffmap_r_ns_otherpairs"].to_numpy(), sub["diffmap_r_ns"].to_numpy(),
               ("other\nword pairs", "matching\nword pair"), s["p_wilcoxon"], f"n = {s['n']} sentences, paired Wilcoxon test")
    ax.axhline(0, color=LINE, lw=0.8, zorder=0)
    ax.set_ylabel("match with the Neurosynth\ndifference map (r)")

    # c: composition, paired
    text(3.4, 3.05, "c  Composition: 'A and B' vs A or B alone", fontweight="bold", fontsize=9, va="top")
    ax = box(4.3, 3.45, 2.1, 1.3)
    c = tests["composition_tests"]["dice_union_AB_vs_best_single"]
    best = comp[["dice_union_A", "dice_union_B"]].max(axis=1).to_numpy()
    paired_box(ax, best, comp["dice_union_AB"].to_numpy(), ("A or B alone\n(the better one)", "'A and B'"),
               c["p_wilcoxon"], f"n = {c['n']} sentences, paired Wilcoxon test")
    ax.set_ylabel("overlap with the A and B\nmaps combined (Dice)")

    for ext in ("png", "svg"):
        fig.savefig(out / f"perturbation_probes.{ext}")
    plt.close(fig)
    print(f"wrote {out / 'perturbation_probes.png'} (axial slice z = {z} mm)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    out = p.parse_args().out
    C.set_offline_env()
    negation_context_table(out)
    figure(out)


if __name__ == "__main__":
    main()
