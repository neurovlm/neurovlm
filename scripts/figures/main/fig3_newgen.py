#!/usr/bin/env python
"""Fig. 3: NeuroVault text-to-brain and brain-to-text examples (docs/figures/12_neurovault_decoding.ipynb), in the
layout of the paper's original figure, on two pages (fig3_newgen, fig3_newgen_p2) with one row of each group each.
Quality = the two cosines of a row; a row is High / Medium / Low when both rank in the top / middle / bottom third of
all NeuroVault papers (the pool), whose densities are drawn on the left with the page's rows marked.

    .venv/bin/python scripts/figures/main/fig3_newgen.py
"""
from __future__ import annotations

import numpy as np
from _shared import (
    FIG_FULL_W, MAP_DISPLAY_RULE, NULL_COLOR, STATUS_BAD, STATUS_GOOD, STATUS_WARN, banner, brain_views_mm, cache_file,
    column_header, display_vector, draw_brain_views, encode_texts, fig_parser, finalize, finish, free_gpu,
    generate_brain_maps, get_logger, load_models, load_pt, make_nvlm, map_cal_tag, map_factor, neurovault_keep, save_pt,
    select_device, split_generated, text_width_in, timed, wrap_measured,
)

STEM = "fig3_newgen"
log = get_logger(STEM)

KEEP_TEXT_IDXS = [189, 149, 124, 152, 147, 167]  # the paper's hand-picked neurovault_text rows, two per group
GROUPS = [("High", STATUS_GOOD), ("Medium", STATUS_WARN), ("Low", STATUS_BAD)]  # two rows each
TERCILE = {"High": (200 / 3, 100.0), "Medium": (100 / 3, 200 / 3), "Low": (0.0, 100 / 3)}  # pool percentiles
QUALITY = {"cross": ("image_cos_cross", "text_cos_cross"), "same": ("image_cos_same", "text_cos_same")}
QUALITY_NOTE = {
    "cross": "Image = cosine of the paper text with the generated map, Text = cosine of the generated text with the "
             "true map (shared space)",
    "same": "Image = cosine of the generated with the true map, Text = cosine of the generated with the paper text "
            "(shared space)",
}
COS_KEYS = ("image_cos_cross", "text_cos_cross", "image_cos_same", "text_cos_same", "text_cos_specter", "true_pair_cos")
GEN_KW = dict(basis="all", num_beams=3, projection_temp=0.035, repetition_penalty=1.5, seed=0)
GEN_BATCH = 6  # bf16 generation depends on the batch size, so every paper is generated in a batch of six, like the rows
BATCH = 128
MARKER_PT = 4.5  # density markers, diameter in pt
LINE_SPACING = 1.0
ABSTRACT_LINES = 3  # true abstract: its first lines, cut with '...' (as the original)
# column widths, inches: each density strip, the shared cosine axis between them, the true text, each map column
# (sagittal above axial); the generated text takes the rest of FIG_FULL_W and sets each row's height
STRIP_W, AXIS_W, TRUE_W, MAP_W = 0.13, 0.19, 0.85, 0.5
MAP_VIEWS = "xz"
PAGES = 2  # rows i, i + PAGES, ... on page i: one row of each group per page
COMPONENTS = ("specter", "proj_head_text_mse", "autoencoder", "proj_head_text_infonce", "proj_head_image_infonce",
              "qformer_canonical", "canonical_semantic_banks", "neuroqwen")


def parse_args():
    p = fig_parser("Fig. 3: NeuroVault text-to-brain and brain-to-text examples.")
    p.add_argument("--quality", choices=["cross", "same"], default="cross",
                   help="cross: text vs generated map and generated text vs true map (notebook); same: same modality.")
    p.add_argument("--t2b-model", choices=["head", "adapter"], default="head",
                   help="head: the text-to-brain head; adapter: the NeuroAdapter (the notebook's choice).")
    return finalize(p.parse_args())


def compute(args, device):
    """Quality cosines and generated text of every NeuroVault paper (on its best-matching same-DOI map), plus the maps
    of the rows select_rows picks under either quality definition."""
    cache = cache_file(args, f"fig3_pool_{args.t2b_model}" + map_cal_tag(args), excl=True)
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)

    import torch
    import torch.nn.functional as F
    from _shared import load_dataset

    with timed("neurovault data", log, args.cache_dir, stage=STEM):
        df_text = load_dataset("neurovault_text").reset_index(drop=True)
        images = load_dataset("neurovault_images").float()
        images = (images / images.quantile(0.99, dim=1).unsqueeze(1)).clamp(0, 1)
        meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
        ok = torch.where(~images.isnan().any(dim=1))[0]
        images = images[ok]
        meta = meta.iloc[ok.numpy()].reset_index(drop=True)
        # drop the excluded papers and their images; texts keep their row index
        text_ok = neurovault_keep(args, df_text["doi"].tolist(), key="NeuroVault publications", log=log)
        img_ok = neurovault_keep(args, meta["doi"].tolist(), key="NeuroVault images", log=log, image_ids=meta["id"].to_numpy())
        images, meta = images[torch.from_numpy(img_ok)], meta[img_ok].reset_index(drop=True)
        texts = (df_text["title"].astype(str) + " [SEP] " + df_text["abstract"].astype(str)).tolist()

    with timed("contrastive embeddings", log, args.cache_dir, stage=STEM):
        t2b = "neuro_adapter" if args.t2b_model == "adapter" else "proj_head_text_mse"
        models = load_models(["specter", t2b, "proj_head_text_infonce", "proj_head_image_infonce",
                              "autoencoder"], device, args.model_dir, log)
        enc = models["autoencoder"].encoder

        def image_c(X):
            return torch.cat([F.normalize(models["proj_head_image_infonce"](enc(X[s:s + BATCH].to(device))), dim=1).cpu()
                              for s in range(0, len(X), BATCH)])

        e = encode_texts(models["specter"], texts, batch_size=32)
        with torch.no_grad():
            e_dev = e.to(device)
            im_pred = generate_brain_maps(e_dev, models, kind=args.t2b_model, calibration=args.map_calibration,
                                          factor=map_factor(args, args.t2b_model, device), batch_size=BATCH)
            text_c = F.normalize(models["proj_head_text_infonce"](e_dev), dim=1).cpu()
            gen_img_c, true_img_c = image_c(im_pred), image_c(images)
        # pair each paper with its same-DOI image of highest contrastive similarity
        sim = (text_c @ true_img_c.T).numpy()
        meta_doi = meta["doi"].to_numpy()
        idxs = np.full(len(df_text), -1)
        for i, doi in enumerate(df_text["doi"].to_numpy()):
            cand = np.flatnonzero(meta_doi == doi)
            if len(cand):
                idxs[i] = int(cand[np.argmax(sim[i, cand])])
        del models
        free_gpu()

    pool = [t for t in range(len(df_text)) if text_ok[t] and idxs[t] >= 0]
    dropped = [t for t in KEEP_TEXT_IDXS if t not in pool]
    if dropped:
        log.warning("hand-picked rows %s are excluded (NeuroVault paper in PubMed train) and are left out", dropped)
    if args.max_samples:  # preview: the hand-picked rows plus the first other papers
        pool = sorted(set(pool) & set(KEEP_TEXT_IDXS) | set([t for t in pool if t not in KEEP_TEXT_IDXS][: args.max_samples]))
    p, j = torch.as_tensor(pool), torch.as_tensor(idxs[pool])

    with timed("brain-to-text generation", log, args.cache_dir, stage=STEM, n=len(pool)):
        nvlm = make_nvlm(device, args.model_dir, log)
        X = images[j].clone()
        X[X < 0.25] = 0.0  # notebook recipe
        generated = []
        for s in range(0, len(X), GEN_BATCH):
            xb = X[s:s + GEN_BATCH]
            k = len(xb)
            xb = torch.cat([xb, xb[-1:].repeat(GEN_BATCH - k, 1)])  # the last batch is padded to six
            generated += list(nvlm.brain(xb).generate_text(**GEN_KW))[:k]
            log.info("generated %d / %d", len(generated), len(X))
        del nvlm
        free_gpu()

    with timed("brain-to-text scoring", log, args.cache_dir, stage=STEM):
        models = load_models(["specter", "proj_head_text_infonce"], device, args.model_dir, log)
        g = encode_texts(models["specter"], generated, batch_size=16)
        with torch.no_grad():
            gen_text_c = torch.cat([F.normalize(models["proj_head_text_infonce"](g[s:s + BATCH].to(device)), dim=1).cpu()
                                    for s in range(0, len(g), BATCH)])
        del models
        free_gpu()

    def cos(a, b):
        return (a * b).sum(dim=1).numpy()

    scores = {"text_idx": np.asarray(pool), "image_idx": idxs[pool],
              "image_cos_cross": cos(text_c[p], gen_img_c[p]), "text_cos_cross": cos(gen_text_c, true_img_c[j]),
              "image_cos_same": cos(gen_img_c[p], true_img_c[j]), "text_cos_same": cos(gen_text_c, text_c[p]),
              "text_cos_specter": cos(g, e[p]), "true_pair_cos": cos(text_c[p], true_img_c[j])}
    shown = {t for q in QUALITY for _, t in select_rows(scores, q)[0]}
    payload = {"pool": scores, "generated_text": generated, "doi": df_text.loc[pool, "doi"].astype(str).tolist(),
               "title": df_text.loc[pool, "title"].astype(str).tolist(),
               "abstract": df_text.loc[pool, "abstract"].astype(str).tolist(),
               "maps": {t: {"true_map": images[idxs[t]].numpy(), "generated_map": im_pred[t].numpy()} for t in shown},
               "n_texts": int(text_ok.sum()), "n_images": int(len(images)), "excluded_examples": dropped,
               "map_calibration": args.map_calibration, "generation": {**GEN_KW, "batch": GEN_BATCH}}
    save_pt(cache, payload)
    return payload


def percentiles(v) -> np.ndarray:
    """Percentile rank of every value within v: 100 x (#below + 0.5 #ties) / n."""
    s = np.sort(v)
    lo, hi = np.searchsorted(s, v, "left"), np.searchsorted(s, v, "right")
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(v)


def select_rows(pool: dict, quality: str):
    """KEEP_TEXT_IDXS, each row whose two percentiles are not both in its group's tercile replaced by the pool paper of
    that tercile closest to the tercile's centre. Returns [(slot, text_idx)] and the percentiles [n_pool, 2]."""
    pct = np.stack([percentiles(pool[k]) for k in QUALITY[quality]], axis=1)
    idx = [int(t) for t in pool["text_idx"]]

    def inside(i, group):
        lo, hi = TERCILE[group]
        return bool(((pct[i] >= lo) & (pct[i] <= hi)).all())

    rows = []
    for slot, t in enumerate(KEEP_TEXT_IDXS):
        group = GROUPS[slot // 2][0]
        if t in idx and inside(idx.index(t), group):
            rows.append((slot, t))
            continue
        used = KEEP_TEXT_IDXS + [r for _, r in rows]
        cand = [i for i in range(len(idx)) if inside(i, group) and idx[i] not in used]
        if cand:
            centre = np.mean(TERCILE[group])
            rows.append((slot, idx[min(cand, key=lambda i: np.hypot(*(pct[i] - centre)))]))
        elif t in idx:  # preview pools too small for the rule
            rows.append((slot, t))
    return rows, pct


def build_rows(args, payload) -> list[dict]:
    """The selected rows: texts, cosines, pool percentiles, group and maps."""
    pool = payload["pool"]
    sel, pct = select_rows(pool, args.quality)
    idx = [int(t) for t in pool["text_idx"]]
    rows = []
    for slot, t in sel:
        if t not in payload["maps"]:
            raise SystemExit(f"no cached maps for text_idx {t}; rerun with --force")
        i = idx.index(t)
        rows.append({"text_idx": t, "image_idx": int(pool["image_idx"][i]), "doi": payload["doi"][i],
                     "title": payload["title"][i], "abstract": payload["abstract"][i],
                     "generated_text": payload["generated_text"][i], **{k: float(pool[k][i]) for k in COS_KEYS},
                     "group": GROUPS[slot // 2][0], "hand_picked": t == KEEP_TEXT_IDXS[slot],
                     "replaces": None if t == KEEP_TEXT_IDXS[slot] else KEEP_TEXT_IDXS[slot],
                     "image_percentile": float(pct[i, 0]), "text_percentile": float(pct[i, 1]), **payload["maps"][t]})
    return rows


def shared_cuts(imgs, thresholds) -> list[float]:
    """One cut per view (sagittal x, coronal y, axial z), in mm, for all maps of a row: the slice that maximises the
    product over the maps of their drawn activation summed over that slice (values above the display threshold, each
    map scaled to max 1), so every map shows activation at the same coordinates."""
    profiles = []
    for img, thr in zip(imgs, thresholds):
        v = np.nan_to_num(np.asarray(img.get_fdata(), dtype=float))
        v = np.where(v > thr, v, 0.0) / max(float(v.max()), 1e-12)
        profiles.append([v.sum(axis=tuple(a for a in range(3) if a != axis)) for axis in range(3)])
    aff = imgs[0].affine  # diagonal (the masker's MNI grid)
    cuts = []
    for axis in range(3):
        score = np.prod([p[axis] for p in profiles], axis=0)
        if score.max() <= 0:  # no slice holds both maps: the slice with the most activation of all maps
            score = np.sum([p[axis] / max(p[axis].max(), 1e-12) for p in profiles], axis=0)
        cuts.append(float(aff[axis, axis] * int(np.argmax(score)) + aff[axis, 3]))
    return cuts


def draw_density(axes, pool: dict, rows: list[dict], keys, fs, axis_in: float):
    """Two strips back to back around their shared cosine axis (tick labels in the axis_in wide gap between them): the
    density of each quality cosine over the pool (grey; Image rises to the left, Text to the right), the pool's thirds
    (dashed) and one marker per row in its group colour next to the axis; markers that would overlap move outwards."""
    from matplotlib.transforms import blended_transform_factory
    from scipy.stats import gaussian_kde

    vals = [np.asarray(pool[k], dtype=float) for k in keys]
    lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
    lo, hi = lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)
    colour = dict(GROUPS)
    ticks = np.arange(np.ceil(lo / 0.2), np.floor(hi / 0.2) + 1) * 0.2
    for ax, v, key, side in zip(axes, vals, keys, ("right", "left")):  # side = the strip's edge on the axis
        grid = np.linspace(v.min(), v.max(), 300)  # the curve stops at the pool's extremes
        d = gaussian_kde(v)(grid)
        ax.fill_betweenx(grid, 0, 0.92 * d / d.max(), color=NULL_COLOR, lw=0)
        for q in np.percentile(v, [100 / 3, 200 / 3]):
            ax.plot([0, 1], [q, q], color="0.2", lw=0.6, ls=(0, (2.5, 1.5)))
        ax.set_xlim(*((1, 0) if side == "right" else (0, 1)))  # x = distance from the axis
        ax.set_ylim(lo, hi)
        pos = ax.get_position()
        w_in, h_in = pos.width * ax.figure.get_figwidth(), pos.height * ax.figure.get_figheight()
        disc = MARKER_PT / 72.0  # marker diameter, inches
        placed = []  # (y, x) in inches
        for r in sorted(rows, key=lambda r: r[key]):
            y_in = (r[key] - lo) / (hi - lo) * h_in
            x_in = disc / 2
            while any(abs(y_in - py) < disc and abs(x_in - px) < disc for py, px in placed):
                x_in += disc
            placed.append((y_in, x_in))
            ax.plot([x_in / w_in], [r[key]], marker="o", ms=MARKER_PT, color=colour[r["group"]], mec="white", mew=0.5,
                    clip_on=False, zorder=3)
        ax.set_xticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(s == side)
        ax.spines[side].set_linewidth(0.6)
        ax.set_yticks(ticks)
        ax.yaxis.set_ticks_position(side)
        ax.tick_params(axis="y", direction="in", length=2.5, width=0.6, labelleft=False, labelright=False)
    tr = blended_transform_factory(axes[1].transAxes, axes[1].transData)
    x_lab = -axis_in / 2 / (axes[1].get_position().width * axes[1].figure.get_figwidth())
    for t in ticks:
        axes[1].text(x_lab, t, f"{t + 0.0:.1f}", transform=tr, ha="center", va="center", fontsize=fs)
    axes[0].yaxis.set_label_position("left")
    axes[0].set_ylabel(f"Cosine, all {len(vals[0])} papers", fontsize=fs, labelpad=2.0)


def true_text(title: str, abstract: str, n_lines: int, width_in: float, fs: float) -> tuple[list[str], list[str]]:
    """Complete bold title and the abstract's first lines (ABSTRACT_LINES at most, at least one, n_lines in all when the
    title leaves room); a cut abstract ends in '...'."""
    tl = wrap_measured(title, width_in, fs, weight="bold", hyphens=True)
    al = wrap_measured(abstract, width_in, fs, hyphens=True)
    k = max(1, min(ABSTRACT_LINES, n_lines - len(tl)))
    if len(al) > k:
        al = al[:k]
        words = al[-1].split()
        while len(words) > 1 and text_width_in(" ".join(words) + "...", fs) > width_in:
            words.pop()
        al[-1] = " ".join(words).rstrip(".,;:") + "..."
    return tl, al


def text_block(fig, x_in: float, top_in: float, bold: list[str], lines: list[str], fs: float, pitch_in: float) -> int:
    """Bold lines then regular lines, left-aligned at x_in, one baseline every pitch_in from the top_in edge down
    (inches from the figure's bottom left). Returns the line count."""
    fw, fh = fig.get_size_inches()
    y = top_in - 0.76 * fs / 72.0  # first baseline: ascenders touch top_in
    for k, line in enumerate(bold + lines):
        fig.text(x_in / fw, (y - k * pitch_in) / fh, line, ha="left", va="baseline", fontsize=fs,
                 fontweight="bold" if k < len(bold) else "normal")
    return len(bold) + len(lines)


def render(args, rows: list[dict], payload, page: int):
    """One page: the rows of build_rows on this page (the density strips mark them). Returns the figure, the page's
    rows for the JSON and its layout."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from _shared import load_masker

    plt.rcParams["savefig.bbox"] = "standard"  # the saved figure is exactly FIG_FULL_W wide
    n = len(rows)
    masker = load_masker()
    fs = args.font_size
    keys = QUALITY[args.quality]
    fig_w = FIG_FULL_W

    # (width, gap after) in inches: density Image | density Text (the shared cosine axis is the gap between them) | group
    # bar | Image | Text | true text | generated maps | true maps | generated text (the rest; it sets the row heights);
    # the density's y-label sits left of the first strip
    widths = [(STRIP_W, AXIS_W), (STRIP_W, 0.03), (0.15, 0.02), (0.3, 0.03), (0.29, 0.04), (TRUE_W, 0.05),
              (MAP_W, 0.04), (MAP_W, 0.06)]
    cols, x = [], 0.15
    for w, gap in widths:
        cols.append((x, w))
        x += w + gap
    cols.append((x, fig_w - 0.03 - x))
    lh_in = fs / 72.0 * LINE_SPACING
    pad, header_h = 0.045, 0.41
    gen = []  # (bold title lines, body lines) of the generated text, complete
    for r in rows:
        title, body = split_generated(r["generated_text"])
        gen.append((wrap_measured(title, cols[8][1], fs, weight="bold", hyphens=True) if title else [],
                    wrap_measured(body, cols[8][1], fs, hyphens=True) if body else []))
    title_lines = [len(wrap_measured(r["title"], cols[5][1], fs, weight="bold", hyphens=True)) for r in rows]
    n_lines = [max(len(gt) + len(gb), k + 1) for (gt, gb), k in zip(gen, title_lines)]  # + one abstract line
    map_wmm, map_hmm = brain_views_mm(MAP_VIEWS, vertical=True)
    map_h = MAP_W * map_hmm / map_wmm + 0.02  # the two stacked views and their gap
    row_h = [max(k * lh_in, map_h) + pad for k in n_lines]
    fig_h = header_h + sum(row_h) + 0.02
    fig = plt.figure(figsize=(fig_w, fig_h))
    row_top = [fig_h - header_h - sum(row_h[:i]) for i in range(n)]  # inches from the bottom
    rows_bot = row_top[-1] - row_h[-1]

    def fx(x_in):
        return x_in / fig_w

    def fy(y_in):
        return y_in / fig_h

    y_top, y_sub = fy(fig_h - 0.16), fy(fig_h - 0.35)
    column_header(fig, fx(cols[0][0] - 0.09), fx(sum(cols[4])), y_top, "Quality", size_delta=1)
    column_header(fig, fx(cols[5][0]), fx(sum(cols[6])), y_top, "Text-to-Brain", size_delta=1)
    column_header(fig, fx(cols[7][0]), fx(sum(cols[8])), y_top, "Brain-to-Text", size_delta=1)
    # the strips' headers are centred on the strips; the labels are wider than the strips
    column_header(fig, fx(cols[0][0] - 0.09), fx(sum(cols[0]) + 0.05), y_sub, "Image", size_delta=0.0)
    column_header(fig, fx(cols[1][0] - 0.05), fx(sum(cols[1]) + 0.04), y_sub, "Text", size_delta=0.0)
    for ci, lab in ((3, "Image"), (4, "Text"), (5, "True"), (6, "Generation"), (7, "True"), (8, "Generation")):
        column_header(fig, fx(cols[ci][0]), fx(sum(cols[ci])), y_sub, lab, size_delta=0.0)

    axes = [fig.add_axes([fx(cols[c][0]), fy(rows_bot + 0.04), fx(cols[c][1]), fy(row_top[0] - rows_bot - 0.08)])
            for c in (0, 1)]
    draw_density(axes, payload["pool"], rows, keys, fs, AXIS_W)

    colour = dict(GROUPS)
    out = []
    for i, r in enumerate(rows):
        top, h = row_top[i], row_h[i]
        y_mid = top - h / 2
        # group bar and name, the row's height
        fig.add_artist(Rectangle((fx(cols[2][0] + 0.1), fy(top - h + pad / 2)), fx(0.05), fy(h - pad),
                                 transform=fig.transFigure, facecolor=colour[r["group"]], edgecolor="none"))
        fig.text(fx(cols[2][0] + 0.04), fy(y_mid), r["group"], rotation=90, ha="center", va="center", fontsize=fs + 1)
        for ci, key in ((3, keys[0]), (4, keys[1])):
            fig.text(fx(cols[ci][0] + cols[ci][1] / 2), fy(y_mid), f"{r[key]:.2f}", ha="center", va="center",
                     fontsize=fs + 1)
        tl, al = true_text(r["title"], r["abstract"], n_lines[i], cols[5][1], fs)
        text_block(fig, cols[5][0], top, tl, al, fs, lh_in)
        shown = {k: display_vector(r[f"{k}_map"]) for k in ("generated", "true")}  # (drawn map, threshold)
        imgs = {k: masker.inverse_transform(v) for k, (v, _) in shown.items()}
        # the generated (text-to-brain) and the true (brain-to-text) map share the row's cuts
        coords = shared_cuts((imgs["generated"], imgs["true"]), (shown["generated"][1], shown["true"][1]))
        for ci, k in ((6, "generated"), (7, "true")):
            draw_brain_views(fig, imgs[k], (cols[ci][0], top - h + pad / 2, cols[ci][1], h - pad), views=MAP_VIEWS,
                             cut_coords=coords, vertical=True, threshold=shown[k][1], vmin=0.0, vmax=None)
        n_gen = text_block(fig, cols[8][0], top, *gen[i], fs, lh_in)
        out.append({k: v for k, v in r.items() if not k.endswith("_map")} |
                   {"page": page + 1, "true_text_in_figure": " ".join(tl + al), "abstract_lines_in_figure": len(al),
                    "generated_lines_in_figure": n_gen, "cut_coords": coords,
                    "map_display": {k: {"threshold": round(thr, 4), "max": round(float(np.max(r[f"{k}_map"])), 4),
                                        "voxels_drawn": int((v > 0).sum())} for k, (v, thr) in shown.items()}})
    names = ("density_image", "density_text", "group_bar", "quality_image", "quality_text", "true_text",
             "generated_maps", "true_maps", "generated_text")
    layout = {"width_in": fig_w, "height_in": round(fig_h, 2), "font_pt": fs,
              "column_widths_in": {k: round(w, 3) for k, (_, w) in zip(names, cols)}, "density_axis_gap_in": AXIS_W,
              "line_spacing": LINE_SPACING, "row_heights_in": [round(h, 3) for h in row_h], "row_lines": n_lines}
    return fig, out, layout


def text_dump(args, payload) -> dict:
    """Everything behind the figure except the rows (added per page)."""
    keys = QUALITY[args.quality]
    pool = payload["pool"]
    return {"quality_definition": args.quality, "quality_columns": QUALITY_NOTE[args.quality],
            "text_to_brain_model": args.t2b_model, "map_calibration": payload["map_calibration"],
            "map_display": MAP_DISPLAY_RULE,
            "cut_coords_rule": "per row and view (sagittal x, coronal y, axial z, mm): the slice maximising the product "
                               "of the generated and the true map's drawn activation summed over that slice (values "
                               "above the display threshold, each map scaled to max 1); both maps are drawn at the "
                               "same cuts, sagittal above axial (rows[].cut_coords)",
            "pages": f"{PAGES} pages, rows i, i + {PAGES}, ... on page i (one row of each group per page); the "
                     "density strips of a page mark its rows (rows[].page)",
            "text_in_figure": "generated text complete; true text = complete title + the abstract's first lines "
                              f"(at most {ABSTRACT_LINES}, within the row height), cut with '...' (the complete "
                              "abstract is in rows[].abstract)",
            "density": "left strips, back to back around their shared cosine axis: Gaussian KDE (Scott bandwidth) of "
                       "each quality cosine over the pool, cut at its min / max; dashed = pool thirds; one marker per "
                       "row in its group colour",
            "pool": {"n": len(pool["text_idx"]), "quality_keys": {"image": keys[0], "text": keys[1]},
                     "percentile": "100 x (#below + 0.5 #ties) / n among all n pool papers (the row's own paper included)",
                     "groups": "High / Medium / Low: both percentiles in [66.7, 100] / [33.3, 66.7] / [0, 33.3]",
                     "selection": "a hand-picked row is kept when both percentiles lie in its group's third, else it is "
                                  "replaced by the pool paper of that third closest to its centre (percentile space)",
                     "hand_picked": KEEP_TEXT_IDXS, "generation": payload["generation"],
                     "thirds_cosine": {lab: [float(q) for q in np.percentile(pool[k], [100 / 3, 200 / 3])]
                                       for lab, k in zip(("image", "text"), keys)},
                     "scores": {"text_idx": [int(t) for t in pool["text_idx"]],
                                "image_idx": [int(j) for j in pool["image_idx"]],
                                **{k: [float(v) for v in pool[k]] for k in COS_KEYS},  # unrounded: rows rank exactly
                                **{f"{lab}_percentile": [round(float(v), 4) for v in percentiles(pool[k])]
                                   for lab, k in zip(("image", "text"), keys)}}},
            "rows": [], "layout": {}}


def main():
    args = parse_args()
    banner(log, args, STEM)
    device = select_device(args.device)
    components = COMPONENTS
    if args.t2b_model == "adapter":
        components = tuple("neuro_adapter" if c == "proj_head_text_mse" else c for c in COMPONENTS)
    payload = compute(args, device)
    rows = build_rows(args, payload)
    with timed("render", log, args.cache_dir, stage=STEM):
        dump = text_dump(args, payload)
        figs = []
        for page in range(PAGES):
            fig, page_rows, layout = render(args, rows[page::PAGES], payload, page)
            stem = STEM if page == 0 else f"{STEM}_p{page + 1}"
            dump["rows"] += page_rows
            dump["layout"][stem] = layout
            figs.append((fig, stem))
        order = [r["text_idx"] for r in rows]
        dump["rows"].sort(key=lambda r: order.index(r["text_idx"]))  # slot order, as build_rows
        for fig, stem in figs[1:]:
            finish(fig, stem, args, log, components=components)
        finish(figs[0][0], STEM, args, log, text_dump=dump, components=components)


if __name__ == "__main__":
    main()
