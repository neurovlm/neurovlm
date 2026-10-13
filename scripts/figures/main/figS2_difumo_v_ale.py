#!/usr/bin/env python
"""Fig. S2: ALE kernel maps (top of each pair) vs their DiFuMo-512 reconstruction (bottom)
(docs/02_data/01_coordinate.ipynb).

    .venv/bin/python scripts/figures/main/figS2_difumo_v_ale.py
"""
from __future__ import annotations

import numpy as np
from _shared import (
    FIG_FULL_W, banner, brain_views_mm, cache_file, cut_coords_for, data_dir, draw_brain_views, fig_parser, finalize,
    finish, get_logger, load_pt, save_pt, timed,
)

STEM = "figS2_difumo_v_ale"
log = get_logger(STEM)
PER_ROW = 4  # examples per row
VIEWS = "xz"  # sagittal and axial


def parse_args():
    p = fig_parser("Fig. S2: ALE kernel maps vs DiFuMo reconstruction.")
    p.add_argument("--n-examples", type=int, default=8, help="Examples (<= 20), four per row.")
    p.add_argument("--threshold", type=float, default=0.1, help="Display threshold (the notebook's vmin).")
    return finalize(p.parse_args())


def compute(args):
    """Example rows of neuro_vectors(_ale).pt (scripts/data/01_coordinate.py), else the PNGs in data_dir/difumo_examples/."""
    cache = cache_file(args, "figS2_examples")
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)
    import pandas as pd
    import torch

    dd = data_dir()
    a, b = dd / "neuro_vectors.pt", dd / "neuro_vectors_ale.pt"
    n = min(20, args.n_examples, args.max_samples or 20)
    if a.exists() and b.exists():
        with timed("load example rows", log, args.cache_dir, stage=STEM):
            pa = torch.load(a, map_location="cpu", weights_only=False, mmap=True)
            pb = torch.load(b, map_location="cpu", weights_only=False, mmap=True)
            pm, pm_ale = np.asarray(pa["pmid"]), np.asarray(pb["pmid"])
            mask = pd.Series(pm_ale).isin(pm).to_numpy()
            rows_ale = np.flatnonzero(mask)
            if not np.array_equal(pm_ale[mask], pm):
                raise RuntimeError("neuro_vectors_ale.pt and neuro_vectors.pt pmids do not align")
            torch.manual_seed(0)  # the notebook's 20 examples
            inds = torch.randperm(len(pm))[:20].numpy()[:n]
            difumo = np.asarray(pa["neuro_vectors"][torch.as_tensor(inds)], dtype="float32")
            ale = np.asarray(pb["neuro_vectors"][torch.as_tensor(rows_ale[inds])], dtype="float32")
        payload = {"mode": "data", "difumo": difumo, "ale": ale, "pmids": pm[inds].tolist()}
    else:
        png_dir = dd / "difumo_examples"
        pairs = [(png_dir / f"{i:02d}_ale.png", png_dir / f"{i:02d}_difumo.png") for i in range(n)]
        if not all(p.exists() and q.exists() for p, q in pairs):
            raise FileNotFoundError(f"Neither {a}/{b} nor the cached PNGs in {png_dir} are available; "
                                    "run scripts/data/01_coordinate.py first.")
        log.warning("neuro_vectors(_ale).pt not found: composing the figure from cached PNGs in %s", png_dir)
        payload = {"mode": "png", "pairs": [(str(p), str(q)) for p, q in pairs]}
    save_pt(cache, payload)
    return payload


def _crop_white(arr: np.ndarray) -> np.ndarray:
    gray = arr[..., :3].mean(axis=2)
    mask = gray < 250
    if not mask.any():
        return arr
    ys, xs = np.where(mask)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def render(args, payload):
    """Examples in rows of PER_ROW, each a framed pair: ALE kernel above DiFuMo, sagittal and axial views at the
    DiFuMo map's cut coordinates."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["savefig.bbox"] = "standard"  # the saved figure is exactly FIG_FULL_W wide
    fs = args.font_size
    n = len(payload["difumo"]) if payload["mode"] == "data" else len(payload["pairs"])
    n_rows = -(-n // PER_ROW)
    left_in, gap_in, pad = 0.68, 0.06, 0.03
    cell_w = (FIG_FULL_W - left_in - 0.02 - gap_in * (PER_ROW - 1)) / PER_ROW
    wmm, hmm = brain_views_mm(VIEWS)
    brain_h = (cell_w - 2 * pad - 0.02) * hmm / wmm
    block_h = 2 * brain_h + 3 * pad
    fig_h = n_rows * block_h + (n_rows - 1) * gap_in + 0.04
    fig = plt.figure(figsize=(FIG_FULL_W, fig_h))
    W, H = FIG_FULL_W, fig_h
    if payload["mode"] == "data":
        from _shared import load_masker

        masker = load_masker()
    for c in range(n):
        row, col = divmod(c, PER_ROW)
        x0 = left_in + col * (cell_w + gap_in)
        y_top = H - 0.02 - row * (block_h + gap_in)
        if col == 0:
            for r, lab in enumerate(("ALE kernel", "DiFuMo")):
                fig.text((left_in - 0.06) / W, (y_top - pad - r * (brain_h + pad) - brain_h / 2) / H, lab, ha="right",
                         va="center", fontsize=fs + 1)
        if payload["mode"] == "data":
            img_d = masker.inverse_transform(payload["difumo"][c])
            img_a = masker.inverse_transform(payload["ale"][c])
            coords = cut_coords_for(img_d, activation_threshold=args.threshold)
        for r in range(2):
            box = (x0 + pad, y_top - pad - r * (brain_h + pad) - brain_h, cell_w - 2 * pad, brain_h)
            if payload["mode"] == "data":
                draw_brain_views(fig, (img_a, img_d)[r], box, views=VIEWS, cut_coords=coords, threshold=args.threshold,
                                 vmin=0.0, vmax=1.0)
            else:
                ax = fig.add_axes([box[0] / W, box[1] / H, box[2] / W, box[3] / H])
                ax.set_axis_off()
                ax.imshow(_crop_white(plt.imread(payload["pairs"][c][r])))
        fig.add_artist(Rectangle((x0 / W, (y_top - block_h) / H), cell_w / W, block_h / H, transform=fig.transFigure,
                                 fill=False, edgecolor="0.25", lw=0.6))
    return fig


def main():
    args = parse_args()
    banner(log, args, STEM)
    payload = compute(args)
    with timed("render", log, args.cache_dir, stage=STEM):
        fig = render(args, payload)
        # no trained model: --model-dir only changes the cache tag
        finish(fig, STEM, args, log, text_dump={"mode": payload["mode"],
                                                "pmids": payload.get("pmids"), "pairs": payload.get("pairs")})


if __name__ == "__main__":
    main()
