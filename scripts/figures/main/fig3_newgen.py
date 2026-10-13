#!/usr/bin/env python
"""Fig. 3: NeuroVault text-to-brain and brain-to-text examples (docs/figures/12_neurovault_decoding.ipynb), six rows in
the layout of the paper's original figure. Quality = the two cosines of a row; a row is High / Medium / Low when both
rank in the top / middle / bottom third of all NeuroVault papers (the pool), whose densities are drawn on the left with
the rows marked. select_rows picks the rows by a fixed rule: every row has a clean generated text (text_check), High and
Medium rows also a title that names one domain and a clean true map; low_check says why a Low row scores low (caption).

    .venv/bin/python scripts/figures/main/fig3_newgen.py
"""
from __future__ import annotations

import re

import numpy as np
from _shared import (
    FIG_FULL_W, MAP_DISPLAY_RULE, NULL_COLOR, STATUS_BAD, STATUS_GOOD, STATUS_WARN, banner, brain_views_mm, cache_file,
    column_header, display_vector, display_vmin, draw_brain_views, encode_texts, fig_parser, finalize, finish, free_gpu,
    generate_brain_maps, get_logger, load_models, load_pt, make_nvlm, map_cal_tag, map_factor, neurovault_keep, save_pt,
    select_device, split_generated, timed, wrap_measured,
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
# a generated text is clean when it has no code-like character, no invented term (a word used fewer than twice in the
# PubMed titles and abstracts and in no bank title, or an anatomical name with a word pair PubMed never uses), a title of
# the stage-14 bank of its level and no content word used more than twice (title, stop and anatomical words aside)
CODE_RE = re.compile(r"[\[\]{}<>_#\\|=*~^`@$]|[^\x00-\x7F‘’“”–—]")
ANAT_HEADS = {"cortex", "cortices", "gyrus", "gyri", "sulcus", "lobule", "lobe", "nucleus", "nuclei", "area", "pole",
              "operculum", "network", "stream"}
VERBS = set("involves involve involving includes include including spans spanning links linking alongside engages engaging "
            "connects connecting supports supporting extends extending encompasses encompassing comprises comprising "
            "recruits recruiting covers covering engaged involved linked associated tied related commonly often typically "
            "generally broadly mainly primarily particularly especially notably".split())
MARKER_PT = 4.5  # density markers, diameter in pt
LINE_SPACING = 1.0
# column widths, inches: the shared y label with the tick labels, each density strip (gap between them), group bar, true
# title, each map column (sagittal above axial); the generated text takes the rest of FIG_FULL_W
YAXIS_W, STRIP_W, STRIP_GAP, BAR_W, TITLE_W, MAP_W = 0.37, 0.28, 0.06, 0.15, 1.15, 0.4
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


def text_vocabulary(args) -> dict:
    """What text_check compares a generated text with: the words used at least twice in the PubMed titles and abstracts
    and the words of the bank titles, the word pairs PubMed uses, the bank titles per level and the anatomical words
    (those of the region and network titles)."""
    from collections import Counter

    from _shared import load_dataset, title_vocabulary
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    pm = load_dataset("pubmed_text")
    counts, pairs = Counter(), set()
    for t in (pm["name"].astype(str) + " " + pm["description"].astype(str)).str.lower():
        w = re.findall(r"[a-z]+", t)
        counts.update(w)
        pairs.update(zip(w, w[1:]))
    bank = title_vocabulary(args.model_dir, logger=log)["titles"]
    bank_words = {w for names in bank.values() for t in names for w in re.findall(r"[a-z]+", t.lower())}
    anat = {w for lv in ("region", "network") for t in bank[lv] for w in re.findall(r"[a-z]+", t.lower())} | ANAT_HEADS
    return {"words": {w for w, c in counts.items() if c >= 2} | bank_words, "pairs": pairs, "n_papers": len(pm),
            "bank": {lv: {t.lower() for t in v} for lv, v in bank.items()}, "anatomy": anat,
            "stop": set(ENGLISH_STOP_WORDS) | VERBS}


def text_check(text: str, voc: dict) -> dict:
    """Junk in a generated text: a title outside the bank of its level, code-like characters, invented words, invented
    anatomical names (the up to three words before an anatomical head, inside a clause, with an adjacent pair PubMed
    never uses) and content words used more than twice. issues = the number of these findings, 0 = clean."""
    m = re.match(r"\s*\[(NETWORK|REGION|FUNCTION)\]", text)
    title, body = split_generated(text)
    title = (title or "").strip()
    words = re.findall(r"[a-z]+", body.lower())
    out = {"title_canonical": bool(m) and title.lower() in voc["bank"].get(m.group(1).lower(), ()),
           "code": sorted(set(CODE_RE.findall(title + " " + body))),
           "invented_words": sorted({w for w in words if w not in voc["words"]}), "invented_names": []}
    for clause in re.split(r"[.,;:()]|\band\b|\bor\b", body.lower()):
        w = re.findall(r"[a-z]+", clause)
        for j in range(len(w)):
            if w[j] not in ANAT_HEADS:
                continue
            k = j
            while k > 0 and j - k < 3 and w[k - 1] not in voc["stop"]:
                k -= 1
            if any(p not in voc["pairs"] for p in zip(w[k:j], w[k + 1:j])):
                out["invented_names"].append(" ".join(w[k:j + 1]))
    own = set(re.findall(r"[a-z]+", title.lower()))
    n = {}
    for x in words:
        if len(x) > 3 and x not in voc["stop"] and x not in voc["anatomy"] and x not in own:
            n[x] = n.get(x, 0) + 1
    out["repeated_words"] = sorted(f"{x} x{c}" for x, c in n.items() if c > 2)
    out["issues"] = int(not out["title_canonical"]) + sum(len(out[k]) for k in ("code", "invented_words", "invented_names",
                                                                                 "repeated_words"))
    out["clean"] = out["issues"] == 0
    return out


def pool_checks(payload, voc: dict) -> list[dict]:
    """Per pool paper: the domains its title names, the clusters of its true map and the check of its generated text."""
    mask3d = mask_3d()
    return [{"domains": title_domains(t), **map_clusters(payload["maps"]["true"][i].astype(np.float32), mask3d),
             "text": text_check(payload["generated_text"][i], voc)} for i, t in enumerate(payload["title"])]


def low_check(c: dict, true_pair_pct: float) -> str:
    """Why a Low row scores low: a poor input map (not clean), else a surprising pair (the paper's text and its own map
    in the bottom third of the pool by their cosine), else a model miss."""
    if not is_clean(c):
        return "Poor input map"
    return "Surprising pair" if true_pair_pct < 100 / 3 else "Model miss"


def passes(c: dict, group: str) -> bool:
    """The row rule: a clean generated text; High and Medium also a title naming one domain and a clean true map."""
    return c["text"]["clean"] and (group == "Low" or (len(c["domains"]) == 1 and is_clean(c)))


def select_rows(pool: dict, checks: list[dict], quality: str):
    """Two rows per third, nearest the third's centre (percentile space) first, among the papers that pass the row rule;
    High and Medium: the second row of another domain. Returns [(group, pool position)], the percentiles [n, 2] and
    the papers of each third (nearest first)."""
    pct = np.stack([percentiles(pool[k]) for k in QUALITY[quality]], axis=1)
    rows, cands = [], {}
    for group, _ in GROUPS:
        lo, hi = TERCILE[group]
        inside = np.flatnonzero(((pct >= lo) & (pct <= hi)).all(axis=1))
        order = sorted(inside.tolist(), key=lambda i: float(np.hypot(*(pct[i] - (lo + hi) / 2))))
        ok = [i for i in order if passes(checks[i], group)]
        cands[group] = order
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
    voc = text_vocabulary(args)
    checks = pool_checks(payload, voc)
    sel, pct, cands = select_rows(pool, checks, args.quality)
    tp = percentiles(pool["true_pair_cos"])
    mask3d = mask_3d()
    rows = []
    for group, i in sel:
        gen, true = (payload["maps"][k][i].astype(np.float32) for k in ("generated", "true"))
        c = {k: v for k, v in checks[i].items() if k not in ("domains", "text")}
        rows.append({"text_idx": int(pool["text_idx"][i]), "image_idx": int(pool["image_idx"][i]), "doi": payload["doi"][i],
                     "title": payload["title"][i], "abstract": payload["abstract"][i],
                     "generated_text": payload["generated_text"][i], **{k: float(pool[k][i]) for k in COS_KEYS},
                     "group": group, "image_percentile": float(pct[i, 0]), "text_percentile": float(pct[i, 1]),
                     "checks": {"title_domains": checks[i]["domains"], "true_map": c, "true_map_clean": is_clean(c),
                                "generated_text": checks[i]["text"], "generated_map": map_clusters(gen, mask3d),
                                "true_pair_percentile": round(float(tp[i]), 2),
                                "low_check": low_check(c, tp[i]) if group == "Low" else None},
                     "true_map": true, "generated_map": gen})
    picked = {i for _, i in sel}
    # every paper of each third, nearest its centre first, with the checks of the row rule
    log_ = {g: [{"text_idx": int(pool["text_idx"][i]), "title": payload["title"][i], "picked": i in picked,
                 "passes_rule": passes(checks[i], g), "image_percentile": round(float(pct[i, 0]), 2),
                 "text_percentile": round(float(pct[i, 1]), 2), "title_domains": checks[i]["domains"],
                 "true_map_clean": is_clean(checks[i]),
                 **{k: v for k, v in checks[i].items() if k not in ("domains", "text")},
                 "true_pair_percentile": round(float(tp[i]), 2),
                 "generated_title": split_generated(payload["generated_text"][i])[0],
                 "text_issues": checks[i]["text"]["issues"], "text_check": checks[i]["text"]} for i in ids]
            for g, ids in cands.items()}
    counts = {}
    for group, _ in GROUPS:
        ids = cands[group]
        clear = {i for i in ids if len(checks[i]["domains"]) == 1}
        clean = {i for i in ids if is_clean(checks[i])}
        text = {i for i in ids if checks[i]["text"]["clean"]}
        counts[group] = {"in_third": len(ids), "clear_title": len(clear), "clean_true_map": len(clean),
                         "clean_text": len(text), "pass_rule": sum(passes(checks[i], group) for i in ids)}
        log.info("%s: %s", group, counts[group])
    for r in rows:
        t = r["checks"]["generated_text"]
        log.info("row %s %d: text issues %d, low_check %s, %s", r["group"], r["text_idx"], t["issues"],
                 r["checks"]["low_check"], r["title"][:60])
    return rows, {"counts": counts, "nearest_first": log_,
                  "text_vocabulary": {"pubmed_papers": voc["n_papers"], "words": len(voc["words"]),
                                      "word_pairs": len(voc["pairs"]),
                                      "bank_titles": {k: len(v) for k, v in voc["bank"].items()}}}


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


def draw_density(axes, pool: dict, rows: list[dict], keys, fs):
    """Two strips side by side on one cosine axis (tick labels and the shared y label left of the first): the density
    of each quality cosine over the pool (grey, rising to the right), the pool's thirds (dashed) and one marker per row
    in its group colour, all on one vertical line next to the strip's axis."""
    from scipy.stats import gaussian_kde

    vals = [np.asarray(pool[k], dtype=float) for k in keys]
    lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
    lo, hi = lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)
    colour = dict(GROUPS)
    ticks = np.arange(np.ceil(lo / 0.2), np.floor(hi / 0.2) + 1) * 0.2
    for k, (ax, v, key) in enumerate(zip(axes, vals, keys)):
        grid = np.linspace(v.min(), v.max(), 300)  # the curve stops at the pool's extremes
        d = gaussian_kde(v)(grid)
        ax.fill_betweenx(grid, 0, 0.95 * d / d.max(), color=NULL_COLOR, lw=0)
        for q in np.percentile(v, [100 / 3, 200 / 3]):
            ax.plot([0, 1], [q, q], color="0.2", lw=0.6, ls=(0, (2.5, 1.5)))
        ax.set_xlim(0, 1)
        ax.set_ylim(lo, hi)
        x = MARKER_PT / 72.0 * 0.7 / (ax.get_position().width * ax.figure.get_figwidth())  # just off the axis line
        for r in sorted(rows, key=lambda r: -r[key]):
            ax.plot([x], [r[key]], marker="o", ms=MARKER_PT, color=colour[r["group"]], mec="white", mew=0.5,
                    clip_on=False, zorder=3)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.set_yticks(ticks, [f"{t + 0.0:.1f}" for t in ticks])
        ax.tick_params(axis="y", direction="out", length=2.5, width=0.6, pad=1.5, labelsize=fs, labelleft=k == 0)
    axes[0].set_ylabel("Cosine similarity", fontsize=fs, labelpad=2.0)


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
    """The figure: header, density strips, then per row the group bar, true title, generated and true map (one cut per
    view, shared) and the complete generated text. Returns the figure, the rows for the JSON and the layout."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from _shared import load_masker

    plt.rcParams["savefig.bbox"] = "standard"  # the saved figure is exactly FIG_FULL_W wide
    n = len(rows)
    masker = load_masker()
    fs = args.font_size
    keys = QUALITY[args.quality]
    fig_w = FIG_FULL_W

    # (name, width, gap after) in inches, after the y label and tick labels; the generated text takes the rest and sets
    # the row heights
    widths = [("density_image", STRIP_W, STRIP_GAP), ("density_text", STRIP_W, 0.03), ("group_bar", BAR_W, 0.02),
              ("true_title", TITLE_W, 0.05), ("generated_maps", MAP_W, 0.04), ("true_maps", MAP_W, 0.06)]
    cols, x = {}, YAXIS_W
    for name, w, gap in widths:
        cols[name] = (x, w)
        x += w + gap
    cols["generated_text"] = (x, fig_w - 0.02 - x)
    lh_in = fs / 72.0 * LINE_SPACING
    pad, group_gap, header_h = 0.04, 0.04, 0.36
    gen = []  # (bold title lines, body lines) of the generated text, complete
    for r in rows:
        title, body = split_generated(r["generated_text"])
        gen.append((wrap_measured(title, cols["generated_text"][1], fs, weight="bold", hyphens=True) if title else [],
                    wrap_measured(body, cols["generated_text"][1], fs, hyphens=True) if body else []))
    titles = [wrap_measured(r["title"], cols["true_title"][1], fs, weight="bold", hyphens=True) for r in rows]
    n_lines = [max(len(gt) + len(gb), len(tl)) for (gt, gb), tl in zip(gen, titles)]
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

    def right(name):
        return sum(cols[name])

    y_top, y_sub = fy(fig_h - 0.15), fy(fig_h - 0.32)
    column_header(fig, fx(0.02), fx(right("group_bar")), y_top, "Quality", size_delta=1)
    column_header(fig, fx(cols["true_title"][0]), fx(right("generated_maps")), y_top, "Text-to-Brain", size_delta=1)
    column_header(fig, fx(cols["true_maps"][0]), fx(right("generated_text")), y_top, "Brain-to-Text", size_delta=1)
    for name, lab in (("density_image", "Image"), ("density_text", "Text"), ("true_title", "True"),
                      ("generated_maps", "Generation"), ("true_maps", "True"), ("generated_text", "Generation")):
        column_header(fig, fx(cols[name][0]), fx(right(name)), y_sub, lab, size_delta=-1.0)

    axes = [fig.add_axes([fx(cols[c][0]), fy(rows_bot + 0.04), fx(cols[c][1]), fy(row_top[0] - rows_bot - 0.08)])
            for c in ("density_image", "density_text")]
    draw_density(axes, payload["pool"], rows, keys, fs)

    bar_x = cols["group_bar"][0]
    for group, col in GROUPS:  # one bar and name per group, over its rows
        idx = [i for i, r in enumerate(rows) if r["group"] == group]
        if not idx:
            continue
        top, bot = row_top[idx[0]] - pad / 2, row_top[idx[-1]] - row_h[idx[-1]] + pad / 2
        fig.add_artist(Rectangle((fx(bar_x + 0.09), fy(bot)), fx(0.05), fy(top - bot), transform=fig.transFigure,
                                 facecolor=col, edgecolor="none"))
        fig.text(fx(bar_x + 0.05), fy((top + bot) / 2), group, rotation=90, ha="center", va="center", fontsize=fs + 1)
    out = []
    for i, r in enumerate(rows):
        top, h = row_top[i], row_h[i]
        text_block(fig, cols["true_title"][0], top, [(titles[i], {"fontweight": "bold"})], fs, lh_in)
        shown = {k: display_vector(r[f"{k}_map"]) for k in ("generated", "true")}  # (drawn map, threshold)
        imgs = {k: masker.inverse_transform(v) for k, (v, _) in shown.items()}
        # the generated (text-to-brain) and the true (brain-to-text) map share the row's cuts
        coords = shared_cuts((imgs["generated"], imgs["true"]), (shown["generated"][1], shown["true"][1]))
        for name, k in (("generated_maps", "generated"), ("true_maps", "true")):
            draw_brain_views(fig, imgs[k], (cols[name][0], top - h + pad / 2, cols[name][1], h - pad), views=MAP_VIEWS,
                             cut_coords=coords, vertical=True, threshold=shown[k][1],
                             vmin=display_vmin(shown[k][1], float(shown[k][0].max())), vmax=float(shown[k][0].max()))
        n_gen = text_block(fig, cols["generated_text"][0], top, [(gen[i][0], {"fontweight": "bold"}), (gen[i][1], {})],
                           fs, lh_in)
        out.append({k: v for k, v in r.items() if not k.endswith("_map")} |
                   {"true_text_in_figure": " ".join(titles[i]), "generated_lines_in_figure": n_gen, "cut_coords": coords,
                    "map_display": {k: {"threshold": round(thr, 4), "voxels_drawn": int((v > 0).sum())}
                                    for k, (v, thr) in shown.items()}})
    layout = {"width_in": fig_w, "height_in": round(fig_h, 2), "font_pt": fs,
              "column_widths_in": {"y_axis": YAXIS_W, **{k: round(w, 3) for k, (_, w) in cols.items()}},
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
            "text_in_figure": "generated text complete; true text = the paper's complete title (the abstract is in "
                              "rows[].abstract); each Low row's check is in rows[].checks.low_check (caption only)",
            "density": "left strips on one cosine axis: Gaussian KDE (Scott bandwidth) of each quality cosine over the "
                       "pool, cut at its min / max; dashed = pool thirds; one marker per row in its group colour, all on "
                       "one vertical line (the rows' cosines are in rows[])",
            "selection": {
                "rule": "per group, the papers whose two percentiles both lie in the group's third, nearest the third's "
                        "centre (percentile space) first, among those with a clean generated text (text_clean). High "
                        "and Medium: also a title that names exactly one domain (title_domains) and a clean true map; "
                        "the second row must name another domain",
                "text_clean": "text_check of the generated text: issues = 1 if its title is not a name of the stage-14 "
                              "bank of its level (title_vocabulary.json) + the distinct code-like characters "
                              f"({CODE_RE.pattern}) + the invented words (used fewer than twice in the "
                              f"{selection['text_vocabulary']['pubmed_papers']:,} PubMed titles and abstracts and in no "
                              "bank title) + the invented anatomical names (the up to "
                              "three words before an anatomical head, inside a clause, with an adjacent word pair "
                              "PubMed never uses) + the content words used more than twice (stop, title and anatomical "
                              "words aside); clean = 0 issues",
                "domains": DOMAINS,
                "clusters": f"the true map as drawn, cut into 26-connected pieces; clusters = pieces of >= {MIN_CLUSTER} "
                            "voxels (2 mm), k50 = pieces (largest mass first) holding half of the drawn mass, top3 = "
                            f"mass share of the three largest, speckle = mass share of pieces < {MIN_CLUSTER} voxels",
                "clean": f"clusters >= {CLEAN['clusters']}, k50 <= {CLEAN['k50']}, speckle <= {CLEAN['speckle']}",
                "low_check": "why a Low row scores low (caption only): Poor input map = the true map is not clean; "
                             "else Surprising pair = the paper's text and its own true map in the bottom third of the "
                             "pool by their cosine (true_pair_cos); else Model miss",
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
