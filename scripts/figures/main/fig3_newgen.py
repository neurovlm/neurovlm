#!/usr/bin/env python
"""Fig. 3: NeuroVault text-to-brain and brain-to-text examples (docs/figures/12_neurovault_decoding.ipynb), six rows in
the layout of the paper's original figure. Quality = the two cosines of a row; a row is High / Medium / Low when both
rank in the top / middle / bottom third of all NeuroVault papers (the pool), whose densities are drawn on the left with
the rows marked. select_rows picks the rows by a fixed rule: High and Medium rows have a title that names one domain and
a clean true map, Low rows are the papers nearest the centre of their third and are labelled by low_check.

    .venv/bin/python scripts/figures/main/fig3_newgen.py
"""
from __future__ import annotations

import re

import numpy as np
from _shared import (
    FIG_FULL_W, MAP_DISPLAY_RULE, NULL_COLOR, STATUS_BAD, STATUS_GOOD, STATUS_WARN, banner, brain_views_mm, cache_file,
    column_header, display_vector, display_vmin, draw_brain_views, encode_texts, fig_parser, finalize, finish, free_gpu,
    generate_brain_maps, get_logger, load_models, load_pt, make_nvlm, map_cal_tag, map_factor, neurovault_keep, save_pt,
    select_device, split_generated, text_width_in, timed, wrap_measured,
)

STEM = "fig3_newgen"
log = get_logger(STEM)

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
# title words of the domains with a textbook map; a title is clear when it names exactly one domain
DOMAINS = {
    "motor": r"motor|movements?|manual|fingers?|hands?|grasp(?:ing)?|reach(?:ing)?|saccad\w*",
    "touch and pain": r"touch|tactile|somatosensory|propriocept\w*|pain|painful|nocicept\w*",
    "visual": r"visual|vision|faces?|retinotop\w*",
    "auditory": r"auditory|sounds?|music|musical|pitch|hearing|listening",
    "language": r"language|speech|spoken|sentences?|reading|words?|semantic|verbal|phonolog\w*|lexical|syntax|syntactic",
    "memory": r"episodic|(?<!working )memory|memories|recollection|hippocamp\w*",
    "working memory": r"working memory",
    "reward": r"rewards?",
    "emotion": r"emotions?|emotional|fear|amygdala",
    "taste and smell": r"taste|gustatory|olfactory|odou?rs?",
}
MIN_CLUSTER = 20  # voxels (160 mm3); smaller pieces of a drawn map are specks
CLEAN = {"clusters": 2, "k50": 3, "speckle": 0.10}  # >= 2 clusters, half the mass in <= 3 pieces, <= 10 % in specks
MARKER_PT = 4.5  # density markers, diameter in pt
LINE_SPACING = 1.0
ABSTRACT_LINES = 3  # true abstract: at most its first lines, where the row has room, cut with '...' (as the original)
# column widths, inches: density y-label, each density strip, the shared cosine axis between them, group bar, cosines
# (Image above Text), true text, each map column (sagittal above axial); the generated text takes the rest of FIG_FULL_W
YLAB_W, STRIP_W, AXIS_W, BAR_W, COS_W, TRUE_W, MAP_W = 0.12, 0.1, 0.16, 0.15, 0.28, 1.15, 0.4
MAP_VIEWS = "xz"
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
    """Quality cosines, generated text, true map and generated map of every NeuroVault paper (on its best-matching
    same-DOI map). The generated maps are kept unclipped (each divided by its maximum) for drawing; the cosines use the
    calibrated maps."""
    cache = cache_file(args, f"fig3_pool_maps_{args.t2b_model}" + map_cal_tag(args), excl=True)
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
            im_shape = generate_brain_maps(e_dev, models, kind=args.t2b_model, calibration="max", batch_size=BATCH)
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
    if args.max_samples:  # preview: the first papers
        pool = pool[: max(args.max_samples, 6)]
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
    payload = {"pool": scores, "generated_text": generated, "doi": df_text.loc[pool, "doi"].astype(str).tolist(),
               "title": df_text.loc[pool, "title"].astype(str).tolist(),
               "abstract": df_text.loc[pool, "abstract"].astype(str).tolist(),
               "maps": {"true": images[j].numpy().astype(np.float16), "generated": im_shape[p].numpy().astype(np.float16)},
               "n_texts": int(text_ok.sum()), "n_images": int(len(images)),
               "map_calibration": args.map_calibration, "generation": {**GEN_KW, "batch": GEN_BATCH}}
    save_pt(cache, payload)
    return payload


def percentiles(v) -> np.ndarray:
    """Percentile rank of every value within v: 100 x (#below + 0.5 #ties) / n."""
    s = np.sort(v)
    lo, hi = np.searchsorted(s, v, "left"), np.searchsorted(s, v, "right")
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(v)


def title_domains(title: str) -> list[str]:
    return [d for d, rx in DOMAINS.items() if re.search(rf"\b(?:{rx})\b", title, flags=re.I)]


def map_clusters(vec, mask3d) -> dict:
    """The drawn map (display_vector) cut into 26-connected pieces: clusters = pieces of at least MIN_CLUSTER voxels,
    k50 = pieces (largest mass first) that hold half of the drawn mass, top3 = mass share of the three largest pieces,
    speckle = mass share of the pieces under MIN_CLUSTER voxels."""
    from scipy import ndimage

    v, _ = display_vector(vec)
    vol = np.zeros(mask3d.shape, np.float32)
    vol[mask3d] = v
    lab, n = ndimage.label(vol > 0, structure=np.ones((3, 3, 3)))
    if n == 0:
        return {"clusters": 0, "k50": 0, "top3": 0.0, "speckle": 0.0, "pieces": 0}
    mass = np.asarray(ndimage.sum_labels(vol, lab, index=np.arange(1, n + 1)))
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    share = np.sort(mass)[::-1].cumsum() / mass.sum()
    return {"clusters": int((size >= MIN_CLUSTER).sum()), "k50": int(np.searchsorted(share, 0.5) + 1),
            "top3": round(float(share[min(2, n - 1)]), 4), "speckle": round(float(mass[size < MIN_CLUSTER].sum() / mass.sum()), 4),
            "pieces": int(n)}


def is_clean(c: dict) -> bool:
    return c["clusters"] >= CLEAN["clusters"] and c["k50"] <= CLEAN["k50"] and c["speckle"] <= CLEAN["speckle"]


def mask_3d() -> np.ndarray:
    from _shared import load_masker

    return np.asarray(load_masker().mask_img_.dataobj).astype(bool)


def pool_checks(payload) -> list[dict]:
    """Per pool paper: the domains its title names and the clusters of its true map."""
    mask3d = mask_3d()
    return [{"domains": title_domains(t), **map_clusters(payload["maps"]["true"][i].astype(np.float32), mask3d)}
            for i, t in enumerate(payload["title"])]


def low_check(c: dict, true_pair_pct: float) -> str:
    """Why a Low row scores low: a poor input map (not clean), else a surprising pair (the paper's text and its own map
    in the bottom third of the pool by their cosine), else a model miss."""
    if not is_clean(c):
        return "Poor input map"
    return "Surprising pair" if true_pair_pct < 100 / 3 else "Model miss"


def select_rows(pool: dict, checks: list[dict], quality: str):
    """Two rows per third, nearest the third's centre (percentile space) first. High and Medium: only papers whose
    title names one domain and whose true map is clean, the second row of another domain. Low: no filter.
    Returns [(group, pool position)], the percentiles [n, 2] and the candidates per group (nearest first)."""
    pct = np.stack([percentiles(pool[k]) for k in QUALITY[quality]], axis=1)
    rows, cands = [], {}
    for group, _ in GROUPS:
        lo, hi = TERCILE[group]
        inside = np.flatnonzero(((pct >= lo) & (pct <= hi)).all(axis=1))
        order = sorted(inside.tolist(), key=lambda i: float(np.hypot(*(pct[i] - (lo + hi) / 2))))
        ok = order if group == "Low" else [i for i in order if len(checks[i]["domains"]) == 1 and is_clean(checks[i])]
        cands[group] = ok
        picked, seen = [], set()
        for i in ok:
            d = tuple(checks[i]["domains"])
            if group == "Low" or d not in seen:
                picked.append(i)
                seen.add(d)
            if len(picked) == 2:
                break
        if len(picked) < 2:  # preview pools too small for the rule
            log.warning("%s: %d papers pass the rule; filled with the nearest others", group, len(picked))
            picked += [i for i in order if i not in picked][: 2 - len(picked)]
        rows += [(group, i) for i in picked]
    return rows, pct, cands


def build_rows(args, payload) -> tuple[list[dict], dict]:
    """The selected rows (texts, cosines, pool percentiles, group, checks, maps) and the selection log."""
    pool = payload["pool"]
    checks = pool_checks(payload)
    sel, pct, cands = select_rows(pool, checks, args.quality)
    tp = percentiles(pool["true_pair_cos"])
    mask3d = mask_3d()
    rows = []
    for group, i in sel:
        gen, true = (payload["maps"][k][i].astype(np.float32) for k in ("generated", "true"))
        c = {k: v for k, v in checks[i].items() if k != "domains"}
        rows.append({"text_idx": int(pool["text_idx"][i]), "image_idx": int(pool["image_idx"][i]), "doi": payload["doi"][i],
                     "title": payload["title"][i], "abstract": payload["abstract"][i],
                     "generated_text": payload["generated_text"][i], **{k: float(pool[k][i]) for k in COS_KEYS},
                     "group": group, "image_percentile": float(pct[i, 0]), "text_percentile": float(pct[i, 1]),
                     "checks": {"title_domains": checks[i]["domains"], "true_map": c, "true_map_clean": is_clean(c),
                                "generated_map": map_clusters(gen, mask3d), "true_pair_percentile": round(float(tp[i]), 2),
                                "low_check": low_check(c, tp[i]) if group == "Low" else None},
                     "true_map": true, "generated_map": gen})
    picked = {i for _, i in sel}
    log_ = {g: [{"text_idx": int(pool["text_idx"][i]), "title": payload["title"][i], "picked": i in picked,
                 "image_percentile": round(float(pct[i, 0]), 2), "text_percentile": round(float(pct[i, 1]), 2),
                 "title_domains": checks[i]["domains"], **{k: v for k, v in checks[i].items() if k != "domains"},
                 "true_pair_percentile": round(float(tp[i]), 2)} for i in ids[: 40 if g != "Low" else 10]]
            for g, ids in cands.items()}
    counts = {}
    for group, _ in GROUPS:
        lo, hi = TERCILE[group]
        inside = np.flatnonzero(((pct >= lo) & (pct <= hi)).all(axis=1))
        clear = [i for i in inside if len(checks[i]["domains"]) == 1]
        clean = [i for i in inside if is_clean(checks[i])]
        counts[group] = {"in_third": len(inside), "clear_title": len(clear), "clean_true_map": len(clean),
                         "both": len(set(clear) & set(clean))}
    return rows, {"counts": counts, "nearest_first": log_}


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
    axes[0].set_ylabel(f"Cosine, all {len(vals[0])} papers", fontsize=fs, labelpad=1.5)


def true_text(title: str, abstract: str, n_lines: int, width_in: float, fs: float) -> tuple[list[str], list[str]]:
    """Complete bold title and the abstract's first lines where the row has room (ABSTRACT_LINES at most, n_lines in
    all); a cut abstract ends in '...'."""
    tl = wrap_measured(title, width_in, fs, weight="bold", hyphens=True)
    al = wrap_measured(abstract, width_in, fs, hyphens=True)
    k = max(0, min(ABSTRACT_LINES, n_lines - len(tl)))
    if len(al) > k:
        al = al[:k]
        if al:
            words = al[-1].split()
            while len(words) > 1 and text_width_in(" ".join(words) + "...", fs) > width_in:
                words.pop()
            al[-1] = " ".join(words).rstrip(".,;:") + "..."
    return tl, al


def text_block(fig, x_in: float, top_in: float, parts: list[tuple[list[str], dict]], fs: float, pitch_in: float) -> int:
    """Lines left-aligned at x_in, one baseline every pitch_in from the top_in edge down (inches from the figure's
    bottom left); parts = [(lines, text kwargs)], e.g. bold title lines then regular body lines. Returns the line count."""
    fw, fh = fig.get_size_inches()
    y = top_in - 0.76 * fs / 72.0  # first baseline: ascenders touch top_in
    k = 0
    for lines, kw in parts:
        for line in lines:
            fig.text(x_in / fw, (y - k * pitch_in) / fh, line, ha="left", va="baseline", fontsize=fs, **kw)
            k += 1
    return k


def render(args, rows: list[dict], payload):
    """The figure: header, density strips, then per row the group bar, cosines, true text, generated and true map (one
    cut per view, shared) and the complete generated text. Returns the figure, the rows for the JSON and the layout."""
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
    # bar | cosines | true text | generated maps | true maps | generated text (the rest; it sets the row heights)
    widths = [(STRIP_W, AXIS_W), (STRIP_W, 0.03), (BAR_W, 0.02), (COS_W, 0.03), (TRUE_W, 0.05), (MAP_W, 0.04),
              (MAP_W, 0.06)]
    cols, x = [], YLAB_W
    for w, gap in widths:
        cols.append((x, w))
        x += w + gap
    cols.append((x, fig_w - 0.02 - x))
    lh_in = fs / 72.0 * LINE_SPACING
    pad, group_gap, header_h = 0.04, 0.04, 0.47
    gen = []  # (bold title lines, body lines) of the generated text, complete
    for r in rows:
        title, body = split_generated(r["generated_text"])
        gen.append((wrap_measured(title, cols[7][1], fs, weight="bold", hyphens=True) if title else [],
                    wrap_measured(body, cols[7][1], fs, hyphens=True) if body else []))
    notes = [[r["checks"]["low_check"]] if r["checks"]["low_check"] else [] for r in rows]
    title_lines = [len(wrap_measured(r["title"], cols[4][1], fs, weight="bold", hyphens=True)) for r in rows]
    n_lines = [max(len(gt) + len(gb), k + len(m)) for (gt, gb), k, m in zip(gen, title_lines, notes)]
    map_wmm, map_hmm = brain_views_mm(MAP_VIEWS, vertical=True)
    map_h = MAP_W * map_hmm / map_wmm + 0.02  # the two stacked views and their gap
    row_h = [max(k * lh_in, map_h) + pad for k in n_lines]
    gaps = [group_gap if i and rows[i]["group"] != rows[i - 1]["group"] else 0.0 for i in range(n)]
    fig_h = header_h + sum(row_h) + sum(gaps) + 0.02
    fig = plt.figure(figsize=(fig_w, fig_h))
    row_top = [fig_h - header_h - sum(row_h[:i]) - sum(gaps[: i + 1]) for i in range(n)]  # inches from the bottom
    rows_bot = row_top[-1] - row_h[-1]

    def fx(x_in):
        return x_in / fig_w

    def fy(y_in):
        return y_in / fig_h

    y_top, y_sub = fy(fig_h - 0.15), fy(fig_h - 0.43)
    column_header(fig, fx(0.02), fx(sum(cols[3])), y_top, "Quality", size_delta=1)
    column_header(fig, fx(cols[4][0]), fx(sum(cols[5])), y_top, "Text-to-Brain", size_delta=1)
    column_header(fig, fx(cols[6][0]), fx(sum(cols[7])), y_top, "Brain-to-Text", size_delta=1)
    # sub-headers at the text size; the strips' labels are wider than the strips
    column_header(fig, fx(0.02), fx(sum(cols[0]) + 0.06), y_sub, "Image", size_delta=-1.0)
    column_header(fig, fx(cols[1][0] - 0.02), fx(sum(cols[1]) + 0.05), y_sub, "Text", size_delta=-1.0)
    column_header(fig, fx(cols[3][0]), fx(sum(cols[3])), y_sub, "Image\nText", size_delta=-1.0)  # the stacked cosines
    for ci, lab in ((4, "True"), (5, "Generation"), (6, "True"), (7, "Generation")):
        column_header(fig, fx(cols[ci][0]), fx(sum(cols[ci])), y_sub, lab, size_delta=-1.0)

    axes = [fig.add_axes([fx(cols[c][0]), fy(rows_bot + 0.04), fx(cols[c][1]), fy(row_top[0] - rows_bot - 0.08)])
            for c in (0, 1)]
    draw_density(axes, payload["pool"], rows, keys, fs, AXIS_W)

    for group, col in GROUPS:  # one bar and name per group, over its rows
        idx = [i for i, r in enumerate(rows) if r["group"] == group]
        if not idx:
            continue
        top, bot = row_top[idx[0]] - pad / 2, row_top[idx[-1]] - row_h[idx[-1]] + pad / 2
        fig.add_artist(Rectangle((fx(cols[2][0] + 0.1), fy(bot)), fx(0.05), fy(top - bot), transform=fig.transFigure,
                                 facecolor=col, edgecolor="none"))
        fig.text(fx(cols[2][0] + 0.04), fy((top + bot) / 2), group, rotation=90, ha="center", va="center", fontsize=fs + 1)
    out = []
    for i, r in enumerate(rows):
        top, h = row_top[i], row_h[i]
        y_mid = top - h / 2
        x_cos = fx(cols[3][0] + cols[3][1] / 2)
        fig.text(x_cos, fy(y_mid + 0.6 * lh_in), f"{r[keys[0]]:.2f}", ha="center", va="center", fontsize=fs)
        fig.text(x_cos, fy(y_mid - 0.6 * lh_in), f"{r[keys[1]]:.2f}", ha="center", va="center", fontsize=fs)
        tl, al = true_text(r["title"], r["abstract"], n_lines[i] - len(notes[i]), cols[4][1], fs)
        text_block(fig, cols[4][0], top, [(tl, {"fontweight": "bold"}), (notes[i], {"fontstyle": "italic"}), (al, {})],
                   fs, lh_in)
        shown = {k: display_vector(r[f"{k}_map"]) for k in ("generated", "true")}  # (drawn map, threshold)
        imgs = {k: masker.inverse_transform(v) for k, (v, _) in shown.items()}
        # the generated (text-to-brain) and the true (brain-to-text) map share the row's cuts
        coords = shared_cuts((imgs["generated"], imgs["true"]), (shown["generated"][1], shown["true"][1]))
        for ci, k in ((5, "generated"), (6, "true")):
            draw_brain_views(fig, imgs[k], (cols[ci][0], top - h + pad / 2, cols[ci][1], h - pad), views=MAP_VIEWS,
                             cut_coords=coords, vertical=True, threshold=shown[k][1],
                             vmin=display_vmin(shown[k][1], float(shown[k][0].max())), vmax=float(shown[k][0].max()))
        n_gen = text_block(fig, cols[7][0], top, [(gen[i][0], {"fontweight": "bold"}), (gen[i][1], {})], fs, lh_in)
        out.append({k: v for k, v in r.items() if not k.endswith("_map")} |
                   {"true_text_in_figure": " ".join(tl + al), "abstract_lines_in_figure": len(al),
                    "generated_lines_in_figure": n_gen, "cut_coords": coords,
                    "map_display": {k: {"threshold": round(thr, 4), "voxels_drawn": int((v > 0).sum())}
                                    for k, (v, thr) in shown.items()}})
    names = ("density_image", "density_text", "group_bar", "cosines", "true_text", "generated_maps", "true_maps",
             "generated_text")
    layout = {"width_in": fig_w, "height_in": round(fig_h, 2), "font_pt": fs,
              "column_widths_in": {k: round(w, 3) for k, (_, w) in zip(names, cols)}, "density_axis_gap_in": AXIS_W,
              "line_spacing": LINE_SPACING, "row_heights_in": [round(h, 3) for h in row_h], "row_lines": n_lines}
    return fig, out, layout


def text_dump(args, payload, selection: dict) -> dict:
    """Everything behind the figure except the rows and the layout (added by main)."""
    keys = QUALITY[args.quality]
    pool = payload["pool"]
    return {"quality_definition": args.quality, "quality_columns": QUALITY_NOTE[args.quality],
            "text_to_brain_model": args.t2b_model, "map_calibration": payload["map_calibration"],
            "map_display": MAP_DISPLAY_RULE + "; generated maps are drawn unclipped (each divided by its maximum), the "
                                              "quality cosines use the calibrated maps",
            "cut_coords_rule": "per row and view (sagittal x, coronal y, axial z, mm): the slice maximising the product "
                               "of the generated and the true map's drawn activation summed over that slice (values "
                               "above the display threshold, each map scaled to max 1); both maps are drawn at the "
                               "same cuts, sagittal above axial (rows[].cut_coords)",
            "text_in_figure": "generated text complete; true text = complete title + the abstract's first lines where "
                              f"the row has room (at most {ABSTRACT_LINES}), cut with '...' (the complete abstract is in "
                              "rows[].abstract); Low rows add their check (rows[].checks.low_check)",
            "density": "left strips, back to back around their shared cosine axis: Gaussian KDE (Scott bandwidth) of "
                       "each quality cosine over the pool, cut at its min / max; dashed = pool thirds; one marker per "
                       "row in its group colour; the cosines column gives Image above Text",
            "selection": {
                "rule": "per group, the papers whose two percentiles both lie in the group's third, nearest the third's "
                        "centre (percentile space) first. High and Medium: only papers whose title names exactly one "
                        "domain (title_domains) and whose true map is clean; the second row must name another domain. "
                        "Low: the two nearest papers, no filter, each labelled by its check",
                "domains": DOMAINS,
                "clusters": f"the true map as drawn, cut into 26-connected pieces; clusters = pieces of >= {MIN_CLUSTER} "
                            "voxels (2 mm), k50 = pieces (largest mass first) holding half of the drawn mass, top3 = "
                            f"mass share of the three largest, speckle = mass share of pieces < {MIN_CLUSTER} voxels",
                "clean": f"clusters >= {CLEAN['clusters']}, k50 <= {CLEAN['k50']}, speckle <= {CLEAN['speckle']}",
                "low_check": "Poor input map = the true map is not clean; else Surprising pair = the paper's text and "
                             "its own true map in the bottom third of the pool by their cosine (true_pair_cos); else "
                             "Model miss",
                "candidates": selection},
            "pool": {"n": len(pool["text_idx"]), "quality_keys": {"image": keys[0], "text": keys[1]},
                     "percentile": "100 x (#below + 0.5 #ties) / n among all n pool papers (the row's own paper included)",
                     "groups": "High / Medium / Low: both percentiles in [66.7, 100] / [33.3, 66.7] / [0, 33.3]",
                     "generation": payload["generation"],
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
    rows, selection = build_rows(args, payload)
    with timed("render", log, args.cache_dir, stage=STEM):
        dump = text_dump(args, payload, selection)
        fig, dump["rows"], dump["layout"] = render(args, rows, payload)
        finish(fig, STEM, args, log, text_dump=dump, components=components)


if __name__ == "__main__":
    main()
