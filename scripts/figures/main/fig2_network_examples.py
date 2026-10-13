#!/usr/bin/env python
"""Fig. 2: atlas networks, their text-to-brain maps and brain-to-text captions. One page: the true (atlas) map and the
map generated from the description of each of the eight networks (sagittal, coronal and axial), the complete generated
Network, Anatomy and Cognition texts of the two with the cleanest generations (pick_full) and the three generated titles
of the other six. The complete descriptions and generated texts of all eight go to a supplementary table,
fig2_texts_table.{tex,md}; --text-pages also draws them as figure pages 2+.

    .venv/bin/python scripts/figures/main/fig2_network_examples.py
"""
from __future__ import annotations

import re

import numpy as np
from _shared import (
    BLUE, FIG_FULL_W, FIG_MAX_H, MAP_DISPLAY_RULE, PREFIX_RE, banner, brain_views_mm, cache_file, cut_coords_for,
    display_vector, draw_brain_views, encode_texts, fig_parser, finalize, finish, free_gpu, generate_brain_maps,
    get_logger, load_models, load_pt, make_nvlm, map_cal_tag, map_factor, mask_vector, save_pt, select_device,
    text_width_in, timed, wrap_measured,
)

STEM = "fig2_network_examples"
log = get_logger(STEM)

BASES = ("network", "region", "function")
COMPONENTS = ("specter", "autoencoder", "proj_head_text_mse", "qformer_canonical", "canonical_semantic_banks", "neuroqwen")
BASIS_LABEL = {"network": "Network", "region": "Anatomy", "function": "Cognition"}  # the three generated titles
VIEWS = "xyz"  # sagittal, coronal and axial
N_FULL = 2  # networks printed with their complete texts
LINE_SPACING = 1.1  # text lines, in font sizes
MAP_COL, COL_GAP = 1.1, 0.09  # page 1, inches: maps column of the complete examples; gap between its columns
GUTTER, CELL_GAP = 0.5, 0.1  # page 1, inches: label gutter of the other six; gap between their cells
TRUE, T2B, B2T = "True", "Generated (text-to-brain)", "Generated (brain-to-text)"  # map and text labels
LABEL_GRAY, RULE_GRAY = "#5f5e5a", "#b4b2ab"

# (atlas, map name, display title): the reference rows of the paper figure
REFS = [
    ("YeoLab", "VisualA", "Visual"),
    ("Glasser", "Language", "Language"),
    ("Shirer", "Auditory", "Auditory"),
    ("WashU", "LateralSM", "Somatomotor"),
    ("Laird", "Emo/Interoception4", "Interoception"),
    ("Shen", "default mode", "Default Mode Network"),
    ("Du", "FPN-A", "Frontoparietal Control Network"),
    ("Du", "SAL/PMN", "Salience Network"),
]
NAMES = {"Default Mode Network": "Default mode network", "Frontoparietal Control Network": "Frontoparietal control",
         "Salience Network": "Salience"}  # names on page 1 (the table keeps the titles)
REF_SHORT = {"Emo/Interoception1": "Interoception", "Emo/Interoception2": "Interoception",
             "Emo/Interoception3": "Interoception", "Emo/Interoception4": "Interoception", "default mode": "Default mode"}

# text-to-brain inputs as printed in the paper figure (not stored anywhere else)
INPUT_TEXT = {
    "Visual": (
        "The visual network supports perception and interpretation of visual information. It is centered on "
        "primary visual cortex, extrastriate occipital cortex, and lateral occipital cortex. It supports visual "
        "feature processing, object perception, motion perception, spatial layout, and visual attention."
    ),
    "Language": (
        "The language network supports comprehension and production of spoken and written language. It is mainly "
        "left-lateralized, involving inferior frontal cortex, posterior superior temporal cortex, middle temporal "
        "cortex, and angular gyrus. It supports semantic access, syntax, phonology, speech comprehension, reading, "
        "and verbal expression."
    ),
    "Auditory": (
        "The auditory network supports perception and interpretation of sound. It is centered on Heschl's gyrus, "
        "primary auditory cortex, planum temporale, and superior temporal gyrus. It supports pitch, loudness, rhythm, "
        "timbre, speech-sound processing, and auditory scene analysis."
    ),
    "Somatomotor": (
        "The somatomotor network supports movement and body sensation. It spans precentral gyrus, postcentral gyrus, "
        "supplementary motor area, and paracentral lobule. It supports voluntary movement, touch, proprioception, "
        "motor planning, and sensorimotor integration."
    ),
    "Interoception": (
        "The interoception network supports awareness of internal bodily state. It is centered on the insula and "
        "anterior cingulate cortex, with contributions from midcingulate cortex. It supports visceral sensation, "
        "autonomic monitoring, bodily awareness, pain appraisal, and affective feeling states."
    ),
    "Default Mode Network": (
        "The default mode network supports internally oriented cognition. It includes medial prefrontal cortex, "
        "posterior cingulate cortex, precuneus, angular gyrus, and medial temporal lobe. It supports autobiographical "
        "memory, self-referential thought, future simulation, social inference, and narrative reasoning."
    ),
    "Frontoparietal Control Network": (
        "The frontoparietal control network supports flexible cognitive control. It links lateral prefrontal cortex, "
        "inferior frontal junction, intraparietal sulcus, and inferior parietal lobule. It supports task goals, "
        "working memory, planning, rule use, attentional control, and adaptive decision-making."
    ),
    "Salience Network": (
        "The salience network supports detection of behaviorally relevant events. It is centered on anterior insula, "
        "frontal operculum, dorsal anterior cingulate cortex, and midcingulate cortex. It supports salience detection, "
        "interoceptive appraisal, attentional orienting, conflict monitoring, and response selection."
    ),
}

# slice planes of the source notebook for maps whose automatic cut misses the network
CUT_OVERRIDES = {"Default Mode Network": [0, -56, 34], "Salience Network": [-38, 18, 2]}


def parse_args():
    p = fig_parser("Fig. 2: network text-to-brain maps and brain-to-text captions.")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--laird-map", default="Emo/Interoception4", help="Laird component used as the Interoception reference.")
    p.add_argument("--text-pages", action="store_true",
                   help="Also draw the complete texts as figure pages 2+ (they are always in fig2_texts_table.{tex,md}).")
    return finalize(p.parse_args())


def compute(args, device):
    cache = cache_file(args, "fig2_panels" + map_cal_tag(args))
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)

    import nibabel as nib
    import torch
    from nilearn.image import resample_to_img
    from _shared import load_dataset, load_masker

    refs = [(a, (args.laird_map if a == "Laird" else m), t) for a, m, t in REFS]
    if args.max_samples:
        refs = refs[: max(1, args.max_samples)]

    with timed("reference maps", log, args.cache_dir, stage=STEM):
        masker = load_masker()
        nets = load_dataset("networks")
        true = []
        for atlas, name, _ in refs:  # array / max -> nearest resample onto the masker grid (notebook 14)
            payload = nets[atlas][name]
            arr = np.asarray(payload["array"], dtype="float32")
            img = nib.Nifti1Image(arr / max(arr.max(), 1e-8), payload["affine"])
            res = resample_to_img(img, masker.mask_img, interpolation="nearest", force_resample=True, copy_header=True)
            true.append(mask_vector(masker, res))
        true = np.stack(true)

    with timed("text-to-brain", log, args.cache_dir, stage=STEM):  # "<title> [SEP] <description>"
        models = load_models(["specter", "autoencoder", "proj_head_text_mse"], device, args.model_dir, log)
        texts = [f"{title} [SEP] {INPUT_TEXT[title]}" for _, _, title in refs]
        e = encode_texts(models["specter"], texts).to(device)
        pred = generate_brain_maps(e, models, kind="head", calibration=args.map_calibration,
                                   factor=map_factor(args, "head", device)).numpy()
        del models
        free_gpu()

    with timed("brain-to-text generation", log, args.cache_dir, stage=STEM, num_beams=args.num_beams):
        nvlm = make_nvlm(device, args.model_dir, log)
        generated = {}
        X = torch.from_numpy(true)
        for i, basis in enumerate(BASES):
            out = nvlm.generate_text(
                X, basis=basis, prefix_text=f"[{basis.upper()}]", num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens, do_sample=False, seed=i, projection_temp=0.05,
            )
            generated[basis] = [out] if isinstance(out, str) else list(out)
        del nvlm
        free_gpu()

    payload = {"refs": refs, "texts": texts, "true": true, "pred": pred, "generated": generated, "map_calibration": args.map_calibration,
               "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens}
    save_pt(cache, payload)
    return payload


def split_caption(text: str) -> tuple[str | None, str]:
    """'[NETWORK]Visual Network\\nThe visual ...' -> ('Visual Network', 'The visual ...'); also 'Title. Body'."""
    text = PREFIX_RE.sub("", str(text)).strip()
    if "\n" in text:
        title, body = text.split("\n", 1)
        return title.strip(), body.strip()
    m = re.match(r"^([A-Z][^.]{2,60}?)\.\s+(.*)$", text, flags=re.S)
    if m and len(m.group(1).split()) <= 6:
        return m.group(1).strip(), m.group(2).strip()
    return None, text


def map_cuts(masker, true_vec, pred_vec, title):
    """Shared cuts of the drawn atlas map and the drawn generated map: through their overlap unless overridden."""
    coords = CUT_OVERRIDES.get(title)
    if coords is None:
        inter = true_vec * (pred_vec > 0)
        coords = cut_coords_for(masker.inverse_transform(inter)) if inter.sum() > 0 else None
    return coords or cut_coords_for(masker.inverse_transform(true_vec))


def pick_full(refs, captions, n=N_FULL):
    """The n networks with the cleanest generations, in figure order: most of the three generated titles named verbatim
    in the input description, ties broken by the share of the description's word pairs that the generated texts repeat.
    Returns (indices, per-network scores)."""
    def pairs(s):
        words = re.findall(r"[a-z']+", s.lower())
        return set(zip(words, words[1:]))

    scores = []
    for (_, _, title), caps in zip(refs, captions):
        desc = INPUT_TEXT[title]
        named = sum(bool(t) and t.lower() in desc.lower() for t, _ in caps)
        overlap = len(pairs(desc) & pairs(" ".join(str(b) for _, b in caps))) / len(pairs(desc))
        scores.append((named, overlap))
    order = sorted(range(len(refs)), key=lambda i: scores[i], reverse=True)
    return sorted(order[:n]), [{"titles_named": a, "word_pair_overlap": round(b, 3)} for a, b in scores]


def column_widths(captions, total, fs):
    """Widths of the Network | Anatomy | Cognition columns (summing to total, inches) that need the fewest lines for the
    longest of the complete texts, ties broken by the fewest lines in all (1 % steps)."""
    best = None
    for a in range(30, 46):  # percent of total for Network, then Anatomy
        for b in range(24, 35):
            ws = (a * total / 100, b * total / 100, (100 - a - b) * total / 100)
            n = [len(wrap_measured(t or "(no title)", w, fs, weight="bold")) + len(wrap_measured(body, w, fs))
                 for caps in captions for (t, body), w in zip(caps, ws)]
            if best is None or (max(n), sum(n)) < best[0]:
                best = ((max(n), sum(n)), ws)
    return best[1]


def map_height(width):
    """Height of draw_brain_views' VIEWS side by side in a box ``width`` inches wide."""
    w_mm, h_mm = brain_views_mm(VIEWS)
    return h_mm * (width - 0.02 * (len(VIEWS) - 1)) / w_mm


def overview_page(refs, shown, coords, captions, full, masker, fs):
    """Page 1. Top: the networks in ``full``, one row each, the network's name, its true (atlas) map and the map generated
    from its description in the left column, the complete generated texts in Network | Anatomy | Cognition columns.
    Below: the other six in two rows of three cells, each the name, the two maps side by side and the three generated
    titles. ``shown``: per network the two drawn maps and their thresholds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    W, lh, name_fs = FIG_FULL_W, fs * LINE_SPACING / 72, fs + 1
    ops = []  # ("text", x, y, line, size, weight, color, ha) | ("rule", x0, x1, y) | ("map", i, k, x, y, w, h); y from the top

    def put(x, y, lines, size=fs, weight="normal", color="black", ha="left"):
        """Lines from y down; returns the y below them."""
        for k, line in enumerate(lines):
            ops.append(("text", x, y + k * size * LINE_SPACING / 72, line, size, weight, color, ha))
        return y + len(lines) * size * LINE_SPACING / 72

    def wrap(s, w, size=fs, weight="normal"):
        return wrap_measured(s, w, size, weight=weight)

    def name(i):
        return NAMES.get(refs[i][2], refs[i][2])

    # complete examples: maps column | three text columns under one header
    widths = column_widths([captions[i] for i in full], W - MAP_COL - 3 * COL_GAP, fs)
    xs = [MAP_COL + COL_GAP + sum(widths[:k]) + k * COL_GAP for k in range(3)]
    y = put((xs[0] + W) / 2, 0.01, [B2T], color=LABEL_GRAY, ha="center")
    ops.append(("rule", xs[0], W, y + 0.01))
    y = max(put(x, y + 0.03, [BASIS_LABEL[b]], color=LABEL_GRAY) for x, b in zip(xs, BASES))
    ops += [("rule", x, x + w, y + 0.01) for x, w in zip(xs, widths)]
    y += 0.05
    h_big = map_height(MAP_COL)
    for i in full:
        y0 = y
        yl = put(0, y0, wrap(name(i), MAP_COL, name_fs, "bold"), name_fs, "bold") + 0.02
        yl = put(0, yl, [TRUE], color=LABEL_GRAY)
        ops.append(("map", i, 0, 0, yl, MAP_COL, h_big))
        yl = put(0, yl + h_big + 0.04, wrap(T2B, MAP_COL), color=LABEL_GRAY)
        ops.append(("map", i, 1, 0, yl, MAP_COL, h_big))
        yl += h_big
        for x, w, (t, body) in zip(xs, widths, captions[i]):
            yt = put(x, y0, wrap(t or "(no title)", w, weight="bold"), weight="bold")
            yl = max(yl, put(x, yt, wrap(body, w)))
        y = yl + 0.05
        ops.append(("rule", 0, W, y))
        y += 0.06

    # the others: gutter of labels | three cells per row
    cell = (W - GUTTER - 2 * CELL_GAP) / 3
    mw = (cell - 0.04) / 2
    h_small = map_height(mw)
    rest = [i for i in range(len(refs)) if i not in full]
    for r0 in range(0, len(rest), 3):
        row = rest[r0:r0 + 3]
        cx = [GUTTER + j * (cell + CELL_GAP) for j in range(len(row))]
        y = max(put(x, y, wrap(name(i), cell, name_fs, "bold"), name_fs, "bold") for x, i in zip(cx, row)) + 0.01
        labels = [wrap(TRUE, mw), wrap(T2B, mw)]
        n = max(map(len, labels))
        for x in cx:
            for k, lab in enumerate(labels):
                put(x + k * (mw + 0.04), y + (n - len(lab)) * lh, lab, color=LABEL_GRAY)
        y += n * lh
        ops += [("map", i, k, x + k * (mw + 0.04), y, mw, h_small) for x, i in zip(cx, row) for k in range(2)]
        y = put(0, y + h_small + 0.03, [B2T], color=LABEL_GRAY)
        for k, b in enumerate(BASES):
            put(0, y, [BASIS_LABEL[b]], color=LABEL_GRAY)
            y = max(put(x, y, wrap(captions[i][k][0] or "(no title)", cell, weight="bold"), weight="bold")
                    for x, i in zip(cx, row))
        y += 0.05
        if r0 + 3 < len(rest):
            ops.append(("rule", 0, W, y))
        y += 0.06
    H = y - 0.06

    fig = plt.figure(figsize=(W, H))
    for op in ops:
        if op[0] == "text":
            _, x, yt, line, size, weight, color, ha = op
            fig.text(x / W, 1 - (yt + 0.8 * size / 72) / H, line, fontsize=size, fontweight=weight, color=color,
                     ha=ha, va="baseline")
        elif op[0] == "rule":
            _, x0, x1, yr = op
            fig.add_artist(Line2D([x0 / W, x1 / W], [1 - yr / H] * 2, transform=fig.transFigure, color=RULE_GRAY, lw=0.5))
        else:
            _, i, k, x, yt, w, h = op
            vec, thr = shown[i][k]
            draw_brain_views(fig, masker.inverse_transform(vec), (x, H - yt - h, w, h), views=VIEWS,
                             cut_coords=coords[i], threshold=thr, vmin=0.0, vmax=None)
    return fig, round(H, 2)


STYLE = {"name": dict(fontweight="bold", color=BLUE), "label": dict(color="#52514e"), "title": dict(fontweight="bold"),
         "body": {}, "head": dict(fontweight="bold", size=1)}  # size: pt above fs


def _font(st, fs):
    """(size, weight, fig.text keywords) of a token style."""
    style = dict(STYLE[st])
    return fs + style.pop("size", 0), style.get("fontweight", "normal"), style


def rich_lines(tokens, width, fs):
    """Greedy wrap of (word, style) tokens to width (inches, or a function of the line index); each line is a list of
    (word, style)."""
    width_of = width if callable(width) else (lambda k: width)
    space = text_width_in(" ", fs)
    lines, cur, used = [], [], 0.0
    for word, st in tokens:
        w = text_width_in(word, *_font(st, fs)[:2])
        if cur and used + space + w > width_of(len(lines)):
            lines.append(cur)
            cur, used = [], 0.0
        used += (space if cur else 0.0) + w
        cur.append((word, st))
    return lines + [cur] if cur else lines


def draw_rich_line(fig, x, y, line, fs, W, H):
    """One wrapped line as runs of one style each."""
    runs = []
    for word, st in line:
        if runs and runs[-1][1] == st:
            runs[-1][0].append(word)
        else:
            runs.append(([word], st))
    for k, (words, st) in enumerate(runs):
        text = " ".join(words) + (" " if k < len(runs) - 1 else "")
        size, weight, style = _font(st, fs)
        fig.text(x / W, y / H, text, fontsize=size, va="top", **style)
        x += text_width_in(text, size, weight)


def text_pages(refs, captions, fs):
    """Pages 2+: the complete description of every network and its three complete generated texts, in two columns
    of run-in paragraphs ("Network: <title>. <text>")."""
    import matplotlib.pyplot as plt

    W, left, gap, max_h = FIG_FULL_W, 0.02, 0.16, FIG_MAX_H
    col_w = (W - 2 * left - gap) / 2
    lh, sep = fs * 1.1 / 72, 0.4  # line height; gap between networks, in lines
    head = "Complete input descriptions and generated texts"
    head_lines = wrap_measured(head, W - 2 * left, fs + 1, weight="bold")
    top0 = max_h - 0.02 - len(head_lines) * (fs + 1) * 1.2 / 72 - 0.08
    stream = []  # wrapped lines of every network, None between networks
    for i, (atlas, name, title) in enumerate(refs):
        if stream:
            stream.append(None)
        paras = [(f"{title} ({atlas} {REF_SHORT.get(name, name)}).", "Description:", None, INPUT_TEXT[title])]
        paras += [("", f"{BASIS_LABEL[b]}:", t, body) for b, (t, body) in zip(BASES, captions[i])]
        for head_words, label, t, body in paras:
            tokens = [(w, "name") for w in head_words.split()] + [(label, "label")]
            tokens += [(w, "title") for w in (f"{t}." if t else "").split()] + [(w, "body") for w in str(body).split()]
            stream += rich_lines(tokens, col_w, fs)

    def col_h(col):
        return sum(sep * lh if line is None else lh for line in col)

    cols, cur = [], []
    for line in stream:
        if line is None and not cur:
            continue
        starts = line is not None and line[0][1] == "name"  # a network starts a column if fewer than 3 lines are left
        if col_h(cur) + (3 if starts else 1) * lh > top0 + 1e-6:
            cols.append(cur)
            cur = []
            if line is None:
                continue
        cur.append(line)
    cols.append(cur)
    figs = []
    for p in range(0, len(cols), 2):
        H = max_h - top0 + max(col_h(c) for c in cols[p:p + 2])
        fig = plt.figure(figsize=(W, H))
        fig.text(left / W, 1 - 0.02 / H, "\n".join(head_lines), fontsize=fs + 1, fontweight="bold", va="top",
                 linespacing=1.2)
        for j, col in enumerate(cols[p:p + 2]):
            x, y = left + j * (col_w + gap), H - (max_h - top0)
            for line in col:
                if line is not None:
                    draw_rich_line(fig, x, y, line, fs, W, H)
                y -= sep * lh if line is None else lh
        figs.append((fig, round(H, 2)))
    return figs

def render(args, payload):
    """([page figures], text dump); every text is printed in full (page 1 the titles, pages 2+ the complete texts)."""
    import matplotlib.pyplot as plt
    from _shared import load_masker

    plt.rcParams["savefig.bbox"] = "standard"  # pages exactly FIG_FULL_W wide
    refs, true, pred, gen = payload["refs"], payload["true"], payload["pred"], payload["generated"]
    masker = load_masker()
    fs = args.font_size
    captions = [[split_caption(gen[b][i]) for b in BASES] for i in range(len(refs))]
    shown = [(display_vector(t), display_vector(p)) for t, p in zip(true, pred)]  # (atlas, generated): (map, threshold)
    coords = [map_cuts(masker, a[0], g[0], refs[i][2]) for i, (a, g) in enumerate(shown)]
    full, scores = pick_full(refs, captions)
    fig1, h1 = overview_page(refs, shown, coords, captions, full, masker, fs)
    pages = [(fig1, h1)] + (text_pages(refs, captions, fs) if args.text_pages else [])
    text_dump = {"text_in_figure": "page 1: the complete generated texts of the full examples, the three generated titles "
                                   "of the others; complete texts of all in fig2_texts_table.{tex,md}"
                                   + (" and on pages 2+" if args.text_pages else ""),
                 "labels": {"true map": TRUE, "generated map": T2B, "generated texts": B2T},
                 "full_examples": {"networks": [refs[i][2] for i in full],
                                   "rule": "the two networks with the most of their three generated titles (network, "
                                           "anatomy, cognition) named verbatim in the input description, ties broken by "
                                           "the share of the description's word pairs that the three generated texts "
                                           "repeat (pick_full)",
                                   "scores": {refs[i][2]: sc for i, sc in enumerate(scores)}},
                 "views": "sagittal, coronal and axial (VIEWS = 'xyz'); full examples: true map above the map generated "
                          "from the description; the others: true map left, generated map right",
                 "map_display": MAP_DISPLAY_RULE, "map_calibration": payload["map_calibration"], "pages": {}, "rows": []}
    for k, (_, h) in enumerate(pages):
        text_dump["pages"][f"{STEM if k == 0 else f'{STEM}_p{k + 1}'}.png"] = {"height_in": h}
    for i, (atlas, name, title) in enumerate(refs):
        text_dump["rows"].append({
            "atlas": atlas, "map": name, "title": title, "input_text": INPUT_TEXT[title],
            "cut_coords": [float(c) for c in coords[i]],
            "map_display": {k: {"threshold": round(thr, 4), "max": round(float(np.max(m)), 4), "voxels_drawn": int((v > 0).sum())}
                            for k, (v, thr), m in zip(("atlas", "generated"), shown[i], (true[i], pred[i]))},
            "generated": {b: {"raw": gen[b][i], "title": captions[i][k][0], "body": captions[i][k][1]}
                          for k, b in enumerate(BASES)}})
    text_dump["generation"] = {"num_beams": payload["num_beams"], "max_new_tokens": payload["max_new_tokens"],
                               "projection_temp": 0.05, "do_sample": False, "seed_per_basis": list(range(len(BASES)))}
    return [f for f, _ in pages], text_dump


def tex_escape(s: str) -> str:
    for c in "&%$#_{}":
        s = s.replace(c, "\\" + c)
    return s


def write_text_table(dump: dict, out) -> list:
    """Supplementary table: per network a block of four rows, its input description and the three complete generated
    texts, as LaTeX (longtable, booktabs, array; p columns filling the sn-jnl text width, 372 pt) and Markdown."""
    from pathlib import Path

    widths = (1.05, 3.98)  # inches; + 2 x 4 pt column gap = 5.14 in
    # ragged right; \hspace{0pt} lets TeX hyphenate a cell's first word (long bold titles in the narrow column)
    cols = "@{}" + "".join(f">{{\\raggedright\\arraybackslash\\hspace{{0pt}}}}p{{{w:.2f}in}}" for w in widths) + "@{}"
    header = "Text (generated title) & Complete text \\\\"
    tex = ["% written by fig2_network_examples.py from fig2_network_examples_text.json; needs longtable, booktabs, array",
           "{\\small\\setlength{\\tabcolsep}{4pt}",
           f"\\begin{{longtable}}{{{cols}}}",
           "\\caption{Complete texts of Fig.~\\ref{fig:gen}. The input description is the text the text-to-brain head turns "
           "into a map. The stage~2 projector writes the generated titles and texts for the atlas map.}"
           "\\label{tab:fig2_texts}\\\\",
           "\\toprule", header, "\\endfirsthead",
           "\\multicolumn{2}{@{}l}{\\footnotesize Table~\\thetable{} (continued)} \\\\", "\\toprule", header,
           "\\endhead", "\\bottomrule", "\\endlastfoot"]
    md = ["| Text (generated title) | Complete text |", "|---|---|"]
    for r in dump["rows"]:
        source = f"{r['atlas']} {REF_SHORT.get(r['map'], r['map'])}"
        tex += ["\\midrule", f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{{tex_escape(r['title'])}}} ({tex_escape(source)})}} \\\\*"]
        md.append(f"| **{r['title']}** ({source}) | |")
        rows = [("Input description", None, r["input_text"])]  # (label, generated title, text)
        for b in BASES:
            g = r["generated"][b]
            rows.append((BASIS_LABEL[b], g["title"] or "(no title)", " ".join(str(g["body"]).split())))
        for k, (label, title, text) in enumerate(rows):
            what = f"\\textit{{{label}}}" + (f"\\newline\\hspace{{0pt}}\\textbf{{{tex_escape(title)}}}" if title else "")
            tex.append(f"{what} & {tex_escape(text)} \\\\" + ("[3pt]" if k < len(rows) - 1 else ""))
            md.append(f"| *{label}*" + (f": **{title}**" if title else "") + f" | {text} |")
    tex += ["\\end{longtable}}", ""]
    out = Path(out)
    paths = [out / "fig2_texts_table.tex", out / "fig2_texts_table.md"]
    paths[0].write_text("\n".join(tex))
    paths[1].write_text("\n".join(md) + "\n")
    return paths


def main():
    args = parse_args()
    banner(log, args, STEM)
    device = select_device(args.device)
    payload = compute(args, device)
    with timed("render", log, args.cache_dir, stage=STEM):
        figs, dump = render(args, payload)
        for page, fig in enumerate(figs[1:], start=1):  # continuation pages first; page 1 also writes the JSON
            finish(fig, f"{STEM}_p{page + 1}", args, log, components=COMPONENTS)
        finish(figs[0], STEM, args, log, text_dump=dump, components=COMPONENTS)
        for path in write_text_table(dump, args.out):
            log.info("wrote %s", path)


if __name__ == "__main__":
    main()
