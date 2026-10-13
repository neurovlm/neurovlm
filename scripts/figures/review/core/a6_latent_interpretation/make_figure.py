#!/usr/bin/env python
"""Latent-dimension figure: (a) the six dimensions from select_dimensions.py as decoded maps with their top terms
(two rows of three),
(b) text- vs image-latent term profiles for all 384 dimensions vs shuffled. Also writes table_selected_dimensions.csv
and table_agreement.csv. Reads outputs/ of run_analysis.py and select_dimensions.py. CPU, ~10 s.
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

sys.path.insert(0, str(HERE.parents[3]))  # scripts/
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths

BLUE, GRAY, INK, INK2, LINE = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#c3c2b7"
PCT = 95.0  # top 5 % of voxels shown in every map
N_TERMS = 3


def tables(out: Path) -> None:
    sel = pd.read_csv(out / "selected_dimensions_neurosynth.csv")
    pd.DataFrame({
        "Dim": sel["dim"].astype(int), "Top terms (text latent)": sel["top3_terms"].str.replace("; ", ", "),
        "Neurosynth term": sel["neurosynth_term"], "r, decoded map vs Neurosynth map": sel["agreement_r"].round(2),
        "Top DiFuMo region of the decoded map": sel["top_difumo_shown"]}).to_csv(out / "table_selected_dimensions.csv", index=False)
    ag = C.read_json(out / "results.json")["agreement"]
    rows = [("DiFuMo top-5 overlap, decoder effect vs encoder correlation", ag["difumo_top5_overlap_decoder_vs_encoder"]),
            ("Term-association r, text latent vs image latent", ag["term_assoc_text_vs_image_latent_r"]),
            ("Term-association r, text latent vs term embeddings", ag["term_assoc_text_vs_term_embedding_r"])]
    pd.DataFrame([{"Statistic": name, "Observed": a["mean"], "Dimensions shuffled": a["null_mean"], "p": a["null_p"]}
                  for name, a in rows]).to_csv(out / "table_agreement.csv", index=False, float_format="%.3g")


def hot_cmap():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("hot_mid", plt.get_cmap("hot")(np.linspace(0.22, 0.88, 256)))


def stat_map(ax, vec: np.ndarray, *, cmap, threshold: float, size: float = 8):
    """Axial slice through the peak of a mask-space vector scaled to [0, 1]."""
    from nilearn.plotting import plot_stat_map

    img = C.unmask(vec)
    a = np.squeeze(img.get_fdata())
    i, j, k = np.unravel_index(np.argmax(a), a.shape)
    z = float((img.affine @ np.array([i, j, k, 1.0]))[2])
    ax.set_axis_off()
    disp = plot_stat_map(img, axes=ax, display_mode="z", cut_coords=[z], cmap=cmap, vmin=0.0, vmax=1.0, threshold=threshold,
                         symmetric_cbar=False, colorbar=False, annotate=False, draw_cross=False, black_bg=False)
    disp.annotate(left_right=True, positions=True, size=size)
    return z


def figure(out: Path, paper: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_rgba

    fs = 7 if paper else 8  # --paper: the same layout scaled down to FIG_FULL_W
    C.apply_style(fs)
    plt.rcParams.update({"axes.titlesize": fs, "savefig.bbox": "standard"})  # no tight crop: saved at exactly the figure width
    z = np.load(out / "associations.npz", allow_pickle=True)
    model = C.read_json(out / "results.json")["config"]["model_dir"]  # cache of select_dimensions.py
    delta = np.load(out / "cache" / f"decoder_effects_all_{Path(model).name}.npz")["delta"]
    sel = pd.read_csv(out / "selected_dimensions_neurosynth.csv")
    ag = C.read_json(out / "results.json")["agreement"]["term_assoc_text_vs_image_latent_r"]
    cmap = hot_cmap()

    W, H = FIG_LETTER_W, 6.6
    fig = plt.figure(figsize=(FIG_FULL_W, H * FIG_FULL_W / W) if paper else (W, H))
    box = lambda x, top, w, h: fig.add_axes((x / W, (H - top - h) / H, w / W, h / H))  # noqa: E731  (top = inches from top)
    text = lambda x, top, s, **kw: fig.text(x / W, (H - top) / H, s, **kw)  # noqa: E731

    text(0.05, 0.05, "a  Six latent dimensions: decoded map and top terms", fontweight="bold", fontsize=fs + 1, va="top")
    text(0.2, 0.27, "Map: what the decoder adds as the dimension increases (top 5% of voxels). Terms: abstract words",
         color=INK2, va="top")
    text(0.2, 0.42, "that score high on this dimension and on few others (it is among their 5% strongest). r: match with",
         color=INK2, va="top")
    text(0.2, 0.57, "the Neurosynth map of the first term (6 best of 384 dimensions).", color=INK2, va="top")
    step, x_first, row_h, top0 = 1.9, 1.0, 2.15, 0.85  # two rows of three
    for i, row in enumerate(sel.itertuples()):
        xc, y0 = x_first + (i % 3) * step, top0 + (i // 3) * row_h
        v = np.clip(row.shown_sign * np.asarray(delta[int(row.dim)], np.float32), 0, None)
        v = v / v.max()
        text(xc, y0, f"dimension {int(row.dim)}", ha="center", va="top", fontweight="bold")
        stat_map(box(xc - 0.75, y0 + 0.17, 1.5, 1.2), v, cmap=cmap, threshold=float(np.percentile(v, PCT)), size=fs)
        for k, term in enumerate(row.top3_terms.split("; ")[:N_TERMS]):  # one text per line: even spacing
            text(xc, y0 + 1.42 + k * 0.15, term, ha="center", va="top", color=INK)
        text(xc, y0 + 1.42 + N_TERMS * 0.15 + 0.04, f"r = {row.agreement_r:.2f}" + (" (flipped)" if row.flipped else ""),
             ha="center", va="top", color=INK2)
    cax = fig.add_axes((5.95 / W, (H - 2.2) / H, 0.07 / W, 0.85 / H))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_ticks([0, 1], labels=["0", "max"])
    cb.outline.set_visible(False)
    cax.tick_params(length=2, pad=1.5, colors=INK2)
    cb.set_label("decoded rise", labelpad=1, color=INK, rotation=90)

    text(0.05, 5.2, "b  Text and brain sides give the same terms (all 384 dimensions)", fontweight="bold", fontsize=fs + 1, va="top")
    ax = box(1.3, 5.48, 3.4, 0.62)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(z["agree_terms"]))
    null = np.mean(C.zrows(z["A_txt"]) * C.zrows(z["A_imgtxt"][perm]), axis=1)
    for y, vals, col in ((1, null, GRAY), (0, z["agree_terms"], BLUE)):
        bp = ax.boxplot([vals], positions=[y], widths=0.55, orientation="horizontal", patch_artist=True, showfliers=False,
                        whis=(0, 100), medianprops=dict(color=col, lw=1.6), boxprops=dict(edgecolor=col, lw=1.0),
                        whiskerprops=dict(color=col, lw=1.0), capprops=dict(color=col, lw=1.0), zorder=2)
        bp["boxes"][0].set_facecolor(to_rgba(col, 0.14))
        ax.scatter(vals, y + rng.uniform(-0.17, 0.17, len(vals)), s=3, color=col, alpha=0.45, linewidths=0, zorder=3)
    ax.set_yticks([1, 0], ["dimensions shuffled", "same dimension"])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.axvline(0, color=LINE, lw=0.8, zorder=0)
    ax.set_xlabel("agreement of the dimension's terms, text side vs brain side (r)")
    text(4.85, 5.57, f"mean r = {ag['mean']:.2f}", va="top", color=INK)
    text(4.85, 5.72, f"shuffled: {abs(ag['null_mean']) if abs(ag['null_mean']) < 0.005 else ag['null_mean']:.2f}", va="top", color=INK2)
    text(4.85, 5.87, f"p = {ag['null_p']:.3f} (200 shuffles)", va="top", color=INK2)
    paths = [PAPER_FIG_DIR / "supp_latent_dimensions.png"] if paper else [out / f"latent_interpretation.{e}" for e in ("png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {paths[0]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    p.add_argument("--paper", action="store_true",
                   help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_latent_dimensions.png.")
    args = p.parse_args()
    out = args.out
    C.set_offline_env()
    if args.paper:
        figure(out, paper=True)
        return
    tables(out)
    figure(out)


if __name__ == "__main__":
    main()
