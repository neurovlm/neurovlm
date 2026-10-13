#!/usr/bin/env python
"""Data-flow diagram (comments 1.2, 2.2): data sources | preprocessing | models in training order | held-out sets.
Every number comes from outputs/dataflow_counts.json; arrows follow scripts/models/ (08 -> 09 -> 12 / 13 -> 14).

    .venv/bin/python scripts/figures/review/core/a4_dataflow_diagram/make_figure.py   # outputs/a4_dataflow.{png,svg}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
ARROW, HELD = "#7d7c77", "#eb6834"  # training-data arrows; held-out (evaluation-only) arrows
STYLE = {  # box kind -> (fill, edge, linestyle)
    "data": ("#e6f0fc", "#2a78d6", "-"),
    "prep": ("#e3f5ee", "#1baf7a", "-"),
    "model": ("#ffffff", "#52514e", "-"),
    "frozen": ("#f1f0ed", "#8a8984", "--"),
    "eval": ("#fdeee7", "#eb6834", "-"),
}
W, H = FIG_LETTER_W, 4.92
FS = 8.0  # every text (pt)
LH = FS * 1.25 / 72  # line height (inches)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "savefig.dpi": 300})


def k(v: int) -> str:
    return f"{v:,}"


def main() -> None:
    global W, H, FS, LH
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    p.add_argument("--paper", action="store_true",
                   help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_dataflow.png.")
    args = p.parse_args()
    out, paper = args.out, args.paper
    if paper:  # 7 pt, narrower columns, long lines broken
        W, H, FS = FIG_FULL_W, 4.52, 7.0
        LH = FS * 1.25 / 72
    c = json.loads((out / "dataflow_counts.json").read_text())
    ex = c["eval_exclusions"]

    fig = plt.figure(figsize=(W, H), dpi=300)  # same dpi as savefig, for the fit check
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    texts: list[tuple[object, dict]] = []  # (text artist, box) pairs for the fit check

    def box(x, top, w, kind, title, lines, step=None, h=None):
        """Rounded box with its top-left corner at (x, top) in inches; bold title, then plain lines."""
        n_title = title.count("\n") + 1
        h = h or 0.1 + (n_title + len(lines)) * LH + 0.08
        fill, edge, ls = STYLE[kind]
        ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fill, ec=edge, lw=0.8, ls=ls))
        b = {"l": x, "r": x + w, "t": top, "b": top - h, "cx": x + w / 2, "cy": top - h / 2}
        texts.append((ax.text(x + w / 2, top - 0.07, title, ha="center", va="top", fontsize=FS, fontweight="bold", color=INK,
                              linespacing=1.1), b))
        if lines:
            texts.append((ax.text(x + w / 2, top - 0.08 - n_title * LH, "\n".join(lines), ha="center", va="top", fontsize=FS,
                                  color=INK2, linespacing=1.25), b))
        if step is not None:  # training-order badge on the top-left corner
            ax.text(x + 0.01, top - 0.01, str(step), ha="center", va="center", fontsize=FS, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.15", fc=INK2, ec="none"), zorder=5)
        return b

    def arrow(pts, held=False, label="", lpos=None, lha="center", lbg=False):
        """Poly-line arrow through ``pts`` (inches); the head is on the last segment."""
        col, ls = (HELD, (0, (3, 2))) if held else (ARROW, "-")
        for a, b_ in zip(pts[:-2], pts[1:-1]):
            ax.plot([a[0], b_[0]], [a[1], b_[1]], color=col, lw=0.9, ls=ls, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, lw=0.9, ls=ls, shrinkA=0, shrinkB=0.5,
                                     color=col))
        if label:
            ax.text(*lpos, label, ha=lha, va="center", fontsize=FS, color=HELD if held else INK2, linespacing=1.15,
                    bbox=dict(fc="white", ec="none", pad=0.6) if lbg else None, zorder=4)

    # columns (inches): data | models, two lanes | held out
    xs, ws = 0.03, 1.38
    xl, wl = 1.77, 1.16
    xr, wr = 3.29, 1.40
    xe, we = 5.05, 1.42
    pitch, r1 = 0.86, H - 0.8
    y_test, y_sum = H - 0.33, H - 0.56  # routing channels above the models
    if paper:
        (xs, ws), (xl, wl), (xr, wr), (xe, we) = (0.02, 1.08), (1.34, 0.98), (2.56, 1.23), (4.11, 1.01)
        pitch, r1 = 0.76, H - 0.71
        y_test, y_sum = H - 0.29, H - 0.5
    for x, w, head in ((xs, ws, "Data"), (xl, xr + wr - xl, "Models, in training order"), (xe, we, "Held out")):
        ax.text(x + w / 2, H - 0.04, head, ha="center", va="top", fontsize=FS, fontweight="bold", color=INK)

    tv = lambda tr, va: [f"train {k(tr)}", f"val {k(va)}"]  # noqa: E731
    L = [r1 - i * pitch for i in range(3)]
    R = [r1 - (i + 0.5) * pitch for i in range(3)]
    ae = box(xl, L[0], wl, "model", "Autoencoder", tv(c["split_train"], c["split_val"]), step=1)
    heads = box(xl, L[1], wl, "model", "Projection heads", tv(c["split_train"], c["split_val"]), step=2)
    anat = box(xl, L[2], wl, "model", "Anatomical head", tv(c["adapter_train"], c["adapter_val"]), step=3)
    st1 = box(xr, R[0], wr, "model", "Stage-1 projector", tv(c["qformer_pubmed_train"], c["qformer_pubmed_val"]), step=4)
    st2 = box(xr, R[1], wr, "model", "Stage-2 projector", tv(c["stage2_train"], c["stage2_val"]), step=5)
    lm = box(xr, R[2], wr, "frozen", "NeuroQwen3-0.6B", ["frozen language model,", f"{c['qwen_dapt_records'] / 1e6:.2f} M titles + abstracts",
                                                        "evaluation papers", "excluded"])

    pub = box(xs, L[0], ws, "data", "PubMed", [f"{k(c['pubmed_publications'])} papers", "split by paper:", f"train {k(c['split_train'])}",
                                               f"val {k(c['split_val'])}", f"test {k(c['split_test'])}"])
    synth = box(xs, anat["t"] + 0.08, ws, "prep", "Synthetic pairs", ["atlas regions and", "training papers:",
                                                                      f"{k(c['synth_rows'])} text–map pairs"])
    ext = box(xs, lm["b"] - 0.1, ws, "data", "NeuroVault +\nnetwork atlases",
              ["group maps;", f"{c['network_atlases']} atlases"] if paper else [f"group maps; {c['network_atlases']} atlases"])
    ev = box(xe, L[0], we, "eval", "Held-out sets", ["never used", "for training"] if paper else ["never used for training"], h=L[0] - ext["b"])

    def section(top, name, lines):
        texts.append((ax.text(ev["cx"], top, name, ha="center", va="top", fontsize=FS, color=INK, fontweight="bold"), ev))
        texts.append((ax.text(ev["cx"], top - LH, "\n".join(lines), ha="center", va="top", fontsize=FS, color=INK2, linespacing=1.25), ev))

    section(st1["t"] - 0.05 - (LH if paper else 0), "PubMed test",
            [f"{k(c['split_test_eval'])} papers", f"({ex['pubmed_test_dropped']} near-duplicates", "removed)"])
    net = [f"{k(c['network_maps_labelled'])} atlas + {k(c['network_ica_components'])} ICA"]
    nv = [f"{k(c['neurovault_publications_eval'])} papers, {k(c['neurovault_maps_eval'])} maps"]
    if paper:
        net = [f"{k(c['network_maps_labelled'])} atlas +", f"{k(c['network_ica_components'])} ICA"]
        nv = [f"{k(c['neurovault_publications_eval'])} papers,", f"{k(c['neurovault_maps_eval'])} maps"]
    net_top = ev["b"] + 0.1 + (1 + len(net)) * LH
    section(net_top, "Network maps", net)
    section(net_top + (1 + len(nv)) * LH + 0.12, "NeuroVault", nv)

    # arrows: data -> models
    arrow([(pub["r"], ae["cy"]), (ae["l"], ae["cy"])])
    arrow([(pub["r"], heads["t"] - 0.12), (heads["l"], heads["t"] - 0.12)])
    arrow([(synth["r"], anat["cy"]), (anat["l"], anat["cy"])])
    xa = pub["l"] + 0.2
    arrow([(xa, pub["b"]), (xa, synth["t"])])
    # model -> model
    arrow([(ae["cx"], ae["b"]), (heads["cx"], heads["t"])])
    arrow([(heads["cx"], heads["b"]), (anat["cx"], anat["t"])], label="init", lpos=(heads["cx"] + 0.06, (heads["b"] + anat["t"]) / 2), lha="left")
    arrow([(ae["r"], ae["cy"] - 0.04), (st1["l"], st1["t"] - 0.12)])
    arrow([(heads["r"], heads["cy"] + 0.08), (st1["l"], st1["b"] + 0.12)])
    arrow([(heads["r"], heads["cy"] - 0.08), (st2["l"], st2["t"] - 0.12)])
    arrow([(anat["r"], anat["cy"] + 0.04), (st2["l"], st2["b"] + 0.12)])
    arrow([(st1["cx"], st1["b"]), (st2["cx"], st2["t"])], label="init", lpos=(st1["cx"] + 0.06, (st1["b"] + st2["t"]) / 2), lha="left")
    arrow([(lm["cx"], lm["t"]), (st2["cx"], st2["b"])], label="used by 4 and 5", lpos=(lm["cx"] + 0.06, (lm["t"] + st2["b"]) / 2), lha="left")
    # routed above the models: stage-1 targets, and the test split straight to the held-out sets
    xs1 = pub["l"] + (0.7 if paper else 1.0)
    arrow([(xs1, pub["t"]), (xs1, y_sum), (st1["cx"] - 0.3, y_sum), (st1["cx"] - 0.3, st1["t"])],
          label="LLM summaries of the training papers", lpos=((xs1 + st1["cx"] - 0.3) / 2 + (0 if paper else 0.2), y_sum), lbg=True)
    xt = pub["l"] + 0.35
    arrow([(xt, pub["t"]), (xt, y_test), (ev["cx"], y_test), (ev["cx"], ev["t"])], held=True,
          label="test papers: evaluation only", lpos=((xl + xr + wr) / 2, y_test), lbg=True)
    arrow([(ext["r"], ext["cy"]), (ev["l"], ext["cy"])], held=True, label="evaluation only", lpos=((xl + xr + wr) / 2, ext["cy"]), lbg=True)

    leg = "Gray arrows: training data or weights. Orange dashed: evaluation only.\nCounts are papers, text–map pairs, or texts."
    leg_t = ax.text(xs, ext["b"] - 0.08, leg, ha="left", va="top", fontsize=FS, color=INK2)

    # fit check: every text block inside its box, legend inside the figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for t, b in texts:
        (x0, y0), (x1, y1) = inv.transform(t.get_window_extent(r).get_points())
        if x0 < b["l"] + 0.02 or x1 > b["r"] - 0.02 or y0 < b["b"] + 0.02 or y1 > b["t"] - 0.02:
            bad.append((t.get_text().splitlines()[0], round(x0 - b["l"], 3), round(b["r"] - x1, 3), round(y0 - b["b"], 3)))
    (x0, y0), (x1, y1) = inv.transform(leg_t.get_window_extent(r).get_points())
    if x1 > W - 0.03 or y0 < 0.02:
        bad.append(("legend", round(x1, 3), round(y0, 3)))
    for b in bad:
        print("[fit] text outside its box:", b)
    paths = [PAPER_FIG_DIR / "supp_dataflow.png"] if paper else [out / f"a4_dataflow.{ext_}" for ext_ in ("png", "svg")]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {paths[0]} ({'fit OK' if not bad else f'{len(bad)} fit problems'})")


if __name__ == "__main__":
    main()
