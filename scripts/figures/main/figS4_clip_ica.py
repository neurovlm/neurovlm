#!/usr/bin/env python
"""Fig. S4: contrastive and generative labels of UK Biobank and HCP ICA components (docs/figures/19_ica_networks.ipynb),
both cohorts on one page. The input maps show one axial slice, or with --views xyz sagittal, coronal and axial slices
(figS4_clip_ica_3slices).

    .venv/bin/python scripts/figures/main/figS4_clip_ica.py
    .venv/bin/python scripts/figures/main/figS4_clip_ica.py --views xyz
"""
from __future__ import annotations

import numpy as np
from _shared import (
    FIG_FULL_W, banner, cache_file, column_header, draw_brain_views, encode_texts,
    fig_parser, finalize, finish, free_gpu, get_logger, load_models, load_network_list, load_pt, make_nvlm, mask_vector,
    save_pt, select_device, split_generated, text_width_in, timed, to_device_batches, wrap_measured,
)

STEM = "figS4_clip_ica"
log = get_logger(STEM)
COHORTS = (("UKBICA", "UK Biobank"), ("HCPICA", "Human Connectome Project"))
NETS_REF = ("visual", "language", "sensorimotor", "default mode", "frontoparietal")  # one row each
MAP_W = {"z": 0.44, "xyz": 0.74}  # inches: input map column per --views
COMPONENTS = ("specter", "autoencoder", "proj_head_text_infonce", "proj_head_image_infonce", "qformer_canonical",
              "canonical_semantic_banks", "neuroqwen")

# the 17 label descriptions ("name [SEP] description" is encoded)
NETS = {
    "frontoparietal": (
        "A domain-general control network supporting flexible, goal-directed behavior (task set maintenance, rule switching, "
        "working-memory control, adaptive decision-making). Neuroanatomically centered on dorsolateral prefrontal cortex "
        "(middle frontal gyrus), inferior/superior parietal lobule and intraparietal sulcus, with contributions from dorsal "
        "premotor areas and midline control regions."
    ),
    "language": (
        "A left-lateralized network for speech perception/production and sentence-level comprehension (phonology, syntax, "
        "semantics, speech planning). Typically includes inferior frontal gyrus (Broca’s area: pars opercularis/triangularis), "
        "posterior superior temporal gyrus/sulcus, middle temporal gyrus, anterior temporal lobe, and angular gyrus, with white-matter "
        "support via arcuate/superior longitudinal fasciculus."
    ),
    "memory": (
        "A network supporting episodic memory encoding and retrieval (scene/context construction, recollection, consolidation). "
        "Anchored in the medial temporal lobe (hippocampus, parahippocampal/entorhinal cortex) and strongly coupled to posterior "
        "cingulate/precuneus, retrosplenial cortex, medial prefrontal cortex, and lateral parietal cortex."
    ),
    "motor": (
        "A network for planning, initiation, and execution of voluntary movements (motor sequencing, force control). Core nodes "
        "include primary motor cortex (precentral gyrus), supplementary motor area/pre-SMA, dorsal premotor cortex, with subcortical "
        "loops through basal ganglia and cerebellum."
    ),
    "action-observation": (
        "Often overlapping with the ‘mirror’ system, this network supports action understanding, imitation, and mapping observed actions "
        "to motor representations. Involves inferior frontal gyrus/ventral premotor cortex, inferior parietal lobule (supramarginal/angular), "
        "and posterior superior temporal sulcus as a key visual–biological motion hub."
    ),
    "reward": (
        "A network supporting valuation, reinforcement learning, and motivated behavior (prediction error, incentive salience, reward-guided choice). "
        "Includes ventral striatum/nucleus accumbens, ventromedial/orbitofrontal cortex, anterior cingulate cortex, amygdala, hippocampus, and midbrain "
        "dopaminergic regions (VTA/substantia nigra) with thalamic interactions."
    ),
    "semantic": (
        "A network for conceptual knowledge and meaning integration across modalities (word meaning, category knowledge, semantic control). "
        "Often includes anterior temporal lobe (semantic hub), middle/inferior temporal gyrus, angular gyrus, posterior superior temporal cortex, "
        "and inferior frontal gyrus for controlled semantic retrieval/selection."
    ),
    "sensorimotor": (
        "A network integrating somatosensory input with motor output (tactile perception, proprioception, sensorimotor coordination). "
        "Centered on precentral and postcentral gyri, paracentral lobule, SMA, and secondary somatosensory cortex (parietal operculum), "
        "with contributions from insula and cerebellum for integration and timing."
    ),
    "visual": (
        "A set of occipital and occipito-temporal systems supporting visual perception (low-level feature processing through higher-level object/scene processing). "
        "Includes primary visual cortex (calcarine sulcus/V1), extrastriate areas (V2–V5/MT), lateral occipital cortex, fusiform gyrus, and parahippocampal "
        "place area for scenes, often extending into intraparietal regions for visuospatial processing."
    ),
    "central executive": (
        "A classic control network for working memory, sustained attention, and top-down regulation of thought/action. Neuroanatomically overlaps strongly "
        "with frontoparietal control systems, emphasizing dorsolateral prefrontal cortex and posterior parietal cortex (IPS/SPL), with coordination via dorsal "
        "ACC/pre-SMA in demanding tasks."
    ),
    "cognitive control": (
        "A network implementing top-down control over perception and action (conflict monitoring, inhibition, rule implementation, error processing). "
        "Typically spans dorsolateral and ventrolateral prefrontal cortex, dorsal anterior cingulate cortex/pre-SMA, and posterior parietal cortex, "
        "with basal ganglia contributions for gating and action selection."
    ),
    "default mode": (
        "The default mode network supports internally oriented cognition (mind-wandering, autobiographical memory, self-referential thought, future simulation, "
        "social cognition). Core hubs include posterior cingulate cortex/precuneus and medial prefrontal cortex, with lateral parietal cortex (angular gyrus) "
        "and medial temporal lobe coupling (hippocampus/parahippocampal) depending on memory demands."
    ),
    "executive control": (
        "A network for executive functions like planning, set shifting, maintaining goals, and suppressing distractions. Neuroanatomically overlaps with "
        "frontoparietal/central executive systems (dlPFC and posterior parietal), often coordinating with dorsal ACC/pre-SMA for performance monitoring "
        "and with basal ganglia for action/response selection."
    ),
    "salience": (
        "A network that detects behaviorally relevant stimuli and coordinates switching between internal (DMN) and externally oriented control/attention systems "
        "(rapid orienting, interoception, threat detection). Anchored in anterior insula (especially right), dorsal anterior cingulate cortex, and subcortical "
        "nodes including amygdala, thalamus, and periaqueductal/midbrain regions."
    ),
    "attention": (
        "A broad term for systems supporting selective attention and attentional control (orienting, prioritizing sensory inputs, maintaining vigilance). "
        "Typically spans frontoparietal regions—frontal eye fields, intraparietal sulcus/superior parietal lobule—and interacts with ventral fronto-insular "
        "and temporoparietal regions depending on goal-driven vs. stimulus-driven demands."
    ),
    "ventral attention": (
        "A right-lateralized network for stimulus-driven reorienting to salient or unexpected events (circuit breaker for attention, target detection). "
        "Key nodes include temporoparietal junction, ventral frontal cortex (inferior/middle frontal gyrus), and anterior insula, interacting with sensory "
        "cortex to shift attention rapidly."
    ),
    "dorsal attention": (
        "A bilateral network for goal-directed, top-down allocation of attention (spatial orienting, sustained visuospatial attention, saccade planning). "
        "Centered on intraparietal sulcus/superior parietal lobule and frontal eye fields, with contributions from dorsal premotor regions and visual cortex "
        "for prioritized processing."
    ),
}


def parse_args():
    p = fig_parser("Fig. S4: HCP / UKB ICA component labelling.")
    p.add_argument("--num-beams", type=int, default=10)
    p.add_argument("--views", choices=sorted(MAP_W), default="xyz",
                   help="Slices of the input maps: z = one axial slice, xyz = sagittal, coronal and axial (_3slices).")
    return finalize(p.parse_args())


def apply_thresh(x: np.ndarray) -> np.ndarray:
    """|x| / 99.9th percentile, zero below the 95th percentile, clipped to 1 (per row)."""
    x = np.abs(x.copy())
    x = x / np.percentile(x, 99.9, axis=1)[:, None]
    x[x < np.percentile(x, 95.0, axis=1)[:, None]] = 0
    x[x > 1.0] = 1.0
    return x.astype("float32")


def compute(args, device):
    cache = cache_file(args, "figS4_panels")
    if cache.exists() and not args.force:
        log.info("cache hit %s", cache)
        return load_pt(cache)
    import torch
    import torch.nn.functional as F
    from nilearn.image import resample_to_img, smooth_img
    from _shared import load_masker

    refs = NETS_REF[: max(1, args.max_samples)] if args.max_samples else NETS_REF
    with timed("components + labels", log, args.cache_dir, stage=STEM):
        masker = load_masker()
        comps = load_network_list(include_only=[c for c, _ in COHORTS])
        mods = load_models(["specter", "autoencoder", "proj_head_text_infonce", "proj_head_image_infonce"], device,
                           args.model_dir, log)
        label_names = list(NETS.keys())
        e = encode_texts(mods["specter"], [f"{k} [SEP] {v}" for k, v in NETS.items()], batch_size=8)
        with torch.no_grad():
            labels_emb = F.normalize(mods["proj_head_text_infonce"](e.to(device)), dim=1).cpu()
        cohorts = {}
        for cohort, _ in COHORTS:  # resample -> smooth (12 mm) -> mask -> apply_thresh = the model input
            items = [(name, img) for atlas, name, img in comps if atlas == cohort]
            raw = np.stack([mask_vector(masker, smooth_img(resample_to_img(img, masker.mask_img, force_resample=True,
                                                                             copy_header=True), fwhm=12))
                            for _, img in items])
            thr = apply_thresh(raw)
            lat = to_device_batches(mods["autoencoder"].encoder, thr, device, batch_size=32)
            with torch.no_grad():
                emb = F.normalize(mods["proj_head_image_infonce"](lat.to(device)), dim=1).cpu()
            cos = (emb @ labels_emb.T).numpy()
            top = cos.argmax(1)
            rows = []
            for ref in refs:  # the component with the highest similarity among those whose top label is ref
                r = label_names.index(ref)
                cand = np.flatnonzero(top == r)
                if len(cand) == 0:
                    log.warning("%s: no component has '%s' as its top label; row skipped", cohort, ref)
                    continue
                j = int(cand[np.argmax(cos[cand, r])])
                order = np.argsort(-cos[j])[:5]
                rows.append({"ref": ref, "component": items[j][0], "index": j,
                             "top_labels": [label_names[k] for k in order], "top_sims": cos[j, order].tolist(),
                             "input_map": thr[j]})
            cohorts[cohort] = rows
        del mods
        free_gpu()
    with timed("generation", log, args.cache_dir, stage=STEM, num_beams=args.num_beams):
        nvlm = make_nvlm(device, args.model_dir, log)
        for cohort, rows in cohorts.items():
            if not rows:
                continue
            X = torch.from_numpy(np.stack([r["input_map"] for r in rows]))
            out = nvlm.brain(X).generate_text(seed=123, basis="network", canonical_basis="all", do_sample=False,
                                              num_beams=args.num_beams, max_new_tokens=128, projection_temp=0.05,
                                              repetition_penalty=1.1)
            out = [out] if isinstance(out, str) else list(out)
            for r, text in zip(rows, out):
                r["generated"] = text
        del nvlm
        free_gpu()
    payload = {"cohorts": cohorts, "label_names": label_names, "refs": list(refs)}
    save_pt(cache, payload)
    return payload


def peak_cut(img, axis: int = 2) -> float:
    """The slice along axis (0 x, 1 y, 2 z; mm) holding the most drawn activation of img."""
    v = np.clip(np.nan_to_num(np.asarray(img.get_fdata(), dtype=float)), 0, None)
    k = int(np.argmax(v.sum(axis=tuple(a for a in range(3) if a != axis))))
    return float(img.affine[axis, axis] * k + img.affine[axis, 3])  # diagonal affine (the masker's MNI grid)


def run_in_lines(title: str | None, body: str, *, font_pt: float, width_in: float) -> tuple[str | None, float, list[str]]:
    """Bold run-in title ('Title.') followed by the wrapped body: (head, indent of the first body line in inches,
    body lines). Never truncates."""
    if not title:
        return None, 0.0, wrap_measured(body, width_in, font_pt)
    head = f"{title}."
    indent = text_width_in(head, font_pt, "bold") + text_width_in("x x", font_pt) - text_width_in("xx", font_pt)
    words, first = body.split(), ""
    while words and text_width_in(f"{first} {words[0]}".strip(), font_pt) <= width_in - indent:
        first = f"{first} {words.pop(0)}".strip()
    return head, indent, [first] + wrap_measured(" ".join(words), width_in, font_pt)


def render(args, payload):
    """Both cohorts on one page: per network row the input map (args.views, each the slice with the most activation),
    the top-5 contrastive labels and the complete generated text with its title run in."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from _shared import load_masker

    plt.rcParams["savefig.bbox"] = "standard"  # the saved figure is exactly FIG_FULL_W wide
    masker = load_masker()
    fs = args.font_size
    # inches: cohort label | input map | bars | text (the rest)
    fig_w = FIG_FULL_W
    x0, label_w = 0.02, 0.14
    widths, col_gaps = [MAP_W[args.views], 0.93], [0.03, 0.07]
    widths.append(fig_w - x0 - label_w - 0.02 - sum(widths) - sum(col_gaps))
    line_spacing = 1.05  # the page also holds the section heading and the caption
    lh_in = fs / 72.0 * line_spacing
    pad, header_h, group_gap, axis_h, min_h = 0.02, 0.3, 0.08, 0.35, 0.5  # min_h: the five bars
    cohorts = [(c, t, payload["cohorts"].get(c, [])) for c, t in COHORTS]
    texts = {c: [run_in_lines(*split_generated(r["generated"]), font_pt=fs, width_in=widths[2]) for r in rows]
             for c, _, rows in cohorts}
    row_h = {c: [max(len(t[2]) * lh_in, min_h) + pad for t in texts[c]] for c, _, _ in cohorts}
    fig_h = header_h + sum(sum(h) for h in row_h.values()) + group_gap * (len(cohorts) - 1) + axis_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    # one bar axis for both cohorts, wide enough for the largest similarity
    all_sims = [s for _, _, rs in cohorts for r in rs for s in r["top_sims"]]
    x_max = max(0.5, float(np.ceil(max(all_sims, default=0.5) / 0.1 - 1e-9)) * 0.1)
    xs, x = [], x0 + label_w
    for w, g in zip(widths, col_gaps + [0.0]):
        xs.append((x / fig_w, w / fig_w))
        x += w + g
    for ci, lab in enumerate(("Input\nimage", "Contrastive\nprediction", "Generative prediction")):
        x_lo = x0 / fig_w if ci == 0 else xs[ci][0]
        column_header(fig, x_lo, xs[ci][0] + xs[ci][1], 1 - (header_h - 0.05) / fig_h, lab, size_delta=-1)
    dump, y_top_in = {}, fig_h - header_h
    last = max(i for i, (_, _, rs) in enumerate(cohorts) if rs)
    for gi, (cohort, title, rows) in enumerate(cohorts):
        if not rows:
            continue
        group_h = sum(row_h[cohort])
        fig.text((x0 + 0.05) / fig_w, (y_top_in - group_h / 2) / fig_h, title, rotation=90, ha="center", va="center",
                 fontsize=fs + 1)
        fig.add_artist(Line2D([(x0 + 0.12) / fig_w] * 2, [(y_top_in - group_h + pad) / fig_h, (y_top_in - pad) / fig_h],
                              transform=fig.transFigure, color="black", lw=0.8))
        dump[cohort] = []
        for ri, r in enumerate(rows):
            h_in = row_h[cohort][ri]
            y_mid = y_top_in - h_in / 2
            img = masker.inverse_transform(r["input_map"])
            coords = [peak_cut(img, a) if v in args.views else 0.0 for a, v in enumerate("xyz")]
            # map and bars the same size in every row, centred in it
            draw_brain_views(fig, img, (x0 + label_w, y_mid - min_h / 2, widths[0], min_h), views=args.views,
                             cut_coords=coords, threshold=1e-3, vmin=0.0, vmax=1.0)
            bar_h = min_h
            ax = fig.add_axes([xs[1][0], (y_mid - bar_h / 2) / fig_h, xs[1][1], bar_h / fig_h])
            sims = np.clip(np.asarray(r["top_sims"]), 0, None)
            ypos = np.arange(len(sims))[::-1]
            ax.barh(ypos, sims, height=0.85, color="C0", alpha=0.35)
            for yv, lab in zip(ypos, r["top_labels"]):  # centred on the bar's midline (descenders hang below)
                ax.text(0.012, yv, lab, ha="left", va="center_baseline", fontsize=fs, color="black")
            ax.set_xlim(0, x_max)
            ax.set_xticks(np.arange(0.0, x_max + 1e-9, 0.2))
            ax.set_ylim(-0.6, len(sims) - 0.4)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", labelsize=fs)
            if gi == last and ri == len(rows) - 1:
                ax.set_xlabel("Cosine similarity")
            else:
                ax.set_xticklabels([])
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", length=0)
            # complete generated text, top-aligned, one baseline per line
            head, indent, lines = texts[cohort][ri]
            ax = fig.add_axes([xs[2][0], (y_top_in - h_in) / fig_h, xs[2][1], h_in / fig_h])
            ax.set_axis_off()
            ax.set_xlim(0, widths[2])
            ax.set_ylim(-h_in, 0)
            base = -0.8 * fs / 72.0
            if head:
                ax.text(0, base, head, ha="left", va="baseline", fontsize=fs, fontweight="bold")
            for li, line in enumerate(lines):
                ax.text(indent if li == 0 else 0, base - li * lh_in, line, ha="left", va="baseline", fontsize=fs)
            dump[cohort].append({k: v for k, v in r.items() if k != "input_map"} |
                                {"generated_lines_in_figure": len(lines), "view": args.views,
                                 "cut_coords": [float(c) for c in coords]})
            y_top_in -= h_in
        y_top_in -= group_gap
    return fig, dump


def main():
    args = parse_args()
    banner(log, args, STEM)
    device = select_device(args.device)
    payload = compute(args, device)
    with timed("render", log, args.cache_dir, stage=STEM):
        fig, rows = render(args, payload)
        finish(fig, STEM if args.views == "z" else f"{STEM}_3slices", args, log, components=COMPONENTS,
               text_dump={"text_in_figure": "complete (no truncation)", "cohorts": rows})


if __name__ == "__main__":
    main()
