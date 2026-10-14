#!/usr/bin/env python
"""Fig. 5: contrastive recall (a PubMed 5-fold CV, b NeuroVault bounds from the a3 analysis), autoencoder ROC and
bits-per-pixel (c) and network labelling (d).  Panels a, c and d are cached separately; b reads the a3 CSVs.

    .venv/bin/python scripts/figures/main/fig5_perf_quant.py [--panels abcd]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from _shared import (
    BLUE, FIG_FULL_W, FOLD_PARTS, ORANGE, banner, cache_file, data_dir, encode_texts, excl_on, fig_parser, finalize,
    finish, fold_files, free_gpu, get_logger, load_models, load_network_list, load_pt, network_binary_vector,
    neurovault_keep, note_source, panel_label, pubmed_keep, pubmed_test_mask, read_json, recall_axes, rel_path,
    resolve_sources, run_file, save_pt, select_device, pubmed_split, text_width_in, timed, to_device_batches, trapz_auc,
)

STEM = "fig5_perf_quant"
log = get_logger(STEM)
COMPONENTS = ("cv_folds", "specter", "autoencoder", "proj_head_text_infonce", "proj_head_image_infonce", "pmids_split")
# panel b: the a3 analysis (scripts/figures/review/core/a3_neurovault_bounds) on the published NeuroVault map tensor
# (a3 variant 'released', the maps Fig. 5b always used; a3's own figure shows its re-read NIfTI variant 'pos')
A3_OUT = Path(__file__).resolve().parents[1] / "review" / "core" / "a3_neurovault_bounds" / "outputs"
B_VARIANT = "released"
B_RULES = {  # rule -> a3 definition per direction (as a3's figure)
    "upper": {"text_to_brain": "paper_upper", "brain_to_text": "paper_upper"},
    "multi": {"text_to_brain": "oracle_any_true", "brain_to_text": "per_image"},
    "lower": {"text_to_brain": "worst_image", "brain_to_text": "worst_image"},
}
B_TINT = {"upper": 0.0, "multi": 0.4, "lower": 0.6}  # mixed with white: one hue per direction, lighter = weaker target
B_ASYMMETRY = (
    "upper (paper recipe): both directions rank the same 298 x 298 similarity matrix (one map per paper, same "
    "exclusions), "
    "text->image along its rows and image->text along its columns, so the pools are matched and the gap is not an "
    "artefact: a few generic abstracts are close to most maps (text hubs; one abstract is the nearest text for 38 of "
    "the 298 maps, 8 abstracts are in the top 5 of >= 10 % of the maps vs 1 map in the top 5 of >= 10 % of the "
    "abstracts), which pushes the true abstract down when a map ranks the texts but is a constant within a text's own "
    "ranking of the maps; subtracting each abstract's mean similarity to all 3,074 maps lifts image->text from 0.787 "
    "to 0.842 and leaves text->image at 0.847 (check on the a3 latents, variant released). multi: the pools differ by definition "
    "(text->image: any of the paper's maps among all 3,074 is a hit; image->text: each map ranks the 298 abstracts "
    "with one hit), hence 0.87 vs 0.63 (chance 0.76 vs 0.51, 0.50 in expectation).")
B_CHANCE = (
    "dashed, one per curve in its colour: a3's null of the same rule (recall_curves.csv 'chance', bounds_metrics.csv "
    "chance_recall_curve_auc): every text is given to another paper (a derangement), then the same best / worst / "
    "any-map selection and ranking are applied; mean of 20 derangements. Not k/N: upper and lower pick the target with "
    "the similarities they rank and multi text->image counts any of the paper's maps, so only multi image->text (one "
    "text per map among 298) is the diagonal.")

# docs/figures/13 cell 3: class of every non-ICA atlas map in HF order
LABEL_MAP = [
    ('VIS-P', 'unknown'), ('CG-OP', 'cingulo_opercular'), ('DN-B', 'default_mode'), ('SMOT-B', 'motor'),
    ('AUD', 'auditory'), ('PM-PPr', 'motor'), ('dATN-B', 'attention'), ('SMOT-A', 'motor'), ('LANG', 'language'),
    ('FPN-B', 'frontoparietal_control'), ('FPN-A', 'frontoparietal_control'), ('dATN-A', 'attention'),
    ('VIS-C', 'visual'), ('SAL/PMN', 'cingulo_opercular'), ('DN-A', 'default_mode'), ('NONE', 'unknown'),
    ('Visual1', 'visual'), ('Visual2', 'visual'), ('Somatomotor', 'motor'), ('CingOperc', 'cingulo_opercular'),
    ('DorsAttn', 'attention'), ('Language', 'language'), ('FrontPar', 'frontoparietal_control'),
    ('Auditory', 'auditory'), ('Default', 'default_mode'), ('PostMulti', 'unknown'), ('VentMulti', 'unknown'),
    ('OrbitAffective', 'unknown'), ('Emo/Interoception1', 'unknown'), ('Emo/Interoception2', 'unknown'),
    ('Emo/Interoception3', 'unknown'), ('Emo/Interoception4', 'unknown'), ('Mot/Visspatial1', 'motor'),
    ('Mot/Visspatial2', 'motor'), ('Mot/Visspatial3', 'motor'), ('Mot/Visspatial4', 'motor'), ('Visual1', 'visual'),
    ('Visual2', 'visual'), ('Visual3', 'visual'), ('DivergentCog1', 'unknown'), ('DivergentCog3', 'unknown'),
    ('DivergentCog4', 'unknown'), ('DivergentCog5', 'unknown'), ('DivergentCog6', 'unknown'),
    ('medial frontal', 'frontoparietal_control'), ('frontoparietal', 'frontoparietal_control'),
    ('default mode', 'default_mode'), ('motor cortex', 'motor'), ('visual A', 'visual'), ('visual B', 'visual'),
    ('visual association', 'visual'), ('subcortical cerebellum', 'unknown'), ('AntSal', 'cingulo_opercular'),
    ('Auditory', 'auditory'), ('DorsalDMN', 'default_mode'), ('HighVisual', 'visual'), ('Language', 'language'),
    ('LECN', 'frontoparietal_control'), ('PostSal', 'cingulo_opercular'), ('Precuneus', 'default_mode'),
    ('PrimVisual', 'visual'), ('RECN', 'frontoparietal_control'), ('Sensorimotor', 'motor'),
    ('VentralDMN', 'default_mode'), ('Visuospatial', 'unknown'), ('Default', 'default_mode'), ('LatVis', 'visual'),
    ('FrontPar', 'frontoparietal_control'), ('MedVis', 'visual'), ('DorsAttn', 'attention'), ('Premotor', 'motor'),
    ('Language', 'language'), ('Salience', 'cingulo_opercular'), ('CingOperc', 'cingulo_opercular'),
    ('HandSM', 'motor'), ('FaceSM', 'motor'), ('Auditory', 'auditory'), ('AntMTL', 'unknown'), ('PostMTL', 'unknown'),
    ('ParMemory', 'unknown'), ('Context', 'unknown'), ('FootSM', 'motor'), ('Visual', 'visual'),
    ('VentAttn', 'unknown'), ('DorsalSM', 'motor'), ('VentralSM', 'motor'), ('MedPar', 'default_mode'),
    ('ParOcc', 'visual'), ('SCAN', 'frontoparietal_control'), ('Cingulo-Opercular', 'cingulo_opercular'),
    ('Effector-hand', 'motor'), ('Effector-mouth', 'motor'), ('Effector-foot', 'motor'), ('SM', 'motor'),
    ('LateralSM', 'motor'), ('ResponseOneHanded(1RESP)', 'motor'), ('ResponseTwoHanded(2RESP)', 'motor'),
    ('AuditoryAttentionResponse(AAR)', 'auditory'), ('AuditoryPrimarySensory(AUD)', 'auditory'),
    ('DMNNovel(DMNA)', 'default_mode'), ('DMNTraditional(DMNB)', 'default_mode'),
    ('FocusOnVisualFeatures(FoVF)', 'visual'), ('Initiation(INIT)', 'unknown'), ('Language(LN)', 'language'),
    ('MAIN', 'unknown'), ('MultipleDemand(MDN)', 'frontoparietal_control'),
    ('Re-evaluation(RE-EV)', 'frontoparietal_control'), ('DefaultA', 'default_mode'), ('DefaultB', 'default_mode'),
    ('DefaultC', 'default_mode'), ('Language', 'language'), ('ContA', 'frontoparietal_control'),
    ('ContB', 'frontoparietal_control'), ('ContC', 'frontoparietal_control'), ('SalVenAttnA', 'cingulo_opercular'),
    ('SalVenAttnB', 'cingulo_opercular'), ('DorsAttnA', 'attention'), ('DorsAttnB', 'attention'),
    ('Aud', 'auditory'), ('SomMotA', 'motor'), ('SomMotB', 'motor'), ('VisualA', 'visual'), ('VisualB', 'visual'),
    ('VisualC', 'visual'), ('TempPar', 'frontoparietal_control'), ('LimbicA', 'unknown'), ('LimbicB', 'unknown'),
    ('SalVentAttnB', 'cingulo_opercular'), ('SalVentAttnA', 'cingulo_opercular'), ('VisPeri', 'visual'),
    ('VisCent', 'visual'), ('SomatomotorA', 'motor'), ('SomatomotorB', 'motor'), ('Sal/VenAttnA', 'cingulo_opercular'),
    ('Sal/VenAttnB', 'cingulo_opercular'), ('ControlC', 'frontoparietal_control'), ('ControlA', 'frontoparietal_control'),
    ('ControlB', 'frontoparietal_control'), ('Visual', 'visual'), ('Somatomotor', 'motor'), ('DorsAttn', 'attention'),
    ('Sal/VenAttn', 'cingulo_opercular'), ('Limbic', 'unknown'), ('Control', 'frontoparietal_control'),
    ('Default', 'default_mode'),
]

# docs/figures/13 cell 4: the eight class descriptions (order = class index)
TEXT_LABELS = {
    "Language": (
        "Language network (LAN; perisylvian language network; frontotemporal language system) [SEP] "
        "Primary regions: left inferior frontal gyrus (Broca’s complex), posterior superior/middle temporal gyrus, "
        "temporoparietal junction/angular gyrus. "
        "Function: speech/text comprehension and production—semantics, syntax, phonology, and sentence-level integration."
    ),
    "Auditory": (
        "Auditory network (AUD; auditory cortex network) [SEP] "
        "Primary regions: Heschl’s gyrus (primary auditory cortex), planum temporale, superior temporal gyrus. "
        "Function: acoustic feature analysis (pitch/timbre/timing), auditory scene analysis, early speech-sound encoding, "
        "and detection of salient sounds."
    ),
    "Default Mode": (
        "Default mode network (DMN; default network; default state network) [SEP] "
        "Primary regions: medial prefrontal cortex, posterior cingulate/precuneus, angular gyrus (lateral parietal). "
        "Function: internally oriented cognition—autobiographical/episodic memory, self-referential thought, future simulation, "
        "social inference, and narrative/semantic integration."
    ),
    "Frontoparietal Control": (
        "Frontoparietal control network (FPCN; frontoparietal network/FPN; central executive network/CEN) [SEP] "
        "Primary regions: dorsolateral/rostrolateral prefrontal cortex (middle frontal), posterior parietal cortex "
        "(inferior parietal/around intraparietal sulcus). "
        "Function: goal maintenance and flexible executive control—working memory, rule implementation, planning, and rapid task switching."
    ),
    "Attention": (
        "Dorsal attention network [SEP] "
        "Primary regions: intraparietal sulcus/superior parietal lobule and frontal eye fields. "
        "Function: goal-directed, voluntary control of visuospatial attention."
    ),
    "Visual": (
        "Visual network (VIS; occipital visual network) [SEP] "
        "Primary regions: calcarine cortex/V1 (cuneus/lingual), extrastriate occipital cortex, ventral occipitotemporal visual areas. "
        "Function: visual feature processing (form/color/motion), object/scene representations, and visuospatial analysis."
    ),
    "Motor": (
        "Motor network (motor/sensorimotor network; SMN) [SEP] "
        "Primary regions: primary motor cortex (precentral gyrus), supplementary motor area, premotor cortex; "
        "with primary somatosensory cortex (postcentral gyrus) for sensorimotor integration. "
        "Function: movement planning, initiation, and execution, plus proprioceptive/tactile feedback used to control actions "
        "and maintain body-state representations."
    ),
    "Cingulo-Opercular": (
        "Salience network (SN; cingulo-opercular network/CON; midcingulo-insular network) [SEP] "
        "Primary regions: anterior insula/frontal operculum, dorsal anterior cingulate/medial frontal cortex. "
        "Function: detects behaviorally relevant internal/external events (including interoceptive/affective signals), "
        "prioritizes attention and arousal, and drives performance monitoring and control adjustments (error/conflict/uncertainty)."
    ),
}
CLASS_KEYS = ("language", "auditory", "default_mode", "frontoparietal_control", "attention", "visual", "motor",
              "cingulo_opercular")
CLASS_NAMES = ["Language", "Auditory", "Default mode", "Frontoparietal", "Attention", "Visual", "Motor",
               "Cingulo-opercular"]


def parse_args():
    p = fig_parser("Fig. 5: contrastive / autoencoder / network-labelling performance.")
    p.add_argument("--step", type=int, default=10, help="Recall-curve k step for PubMed.")
    p.add_argument("--panels", default="abcd", help="Panels to draw, e.g. cd.")
    p.add_argument("--bounds-dir", type=Path, default=A3_OUT,
                   help="a3_neurovault_bounds outputs for panel b (recall_curves.csv, bounds_metrics.csv, config.json).")
    return finalize(p.parse_args())


def _pubmed_local(args):
    """(neuro_vectors memory-mapped, pmids, split, split source) when neuro_vectors.pt exists, else None. The split is the
    run's pmids_split.pt, else the pinned HF split flags (pubmed_split)."""
    import torch

    nv = data_dir() / "neuro_vectors.pt"
    if not nv.exists():
        return None
    payload = torch.load(nv, map_location="cpu", weights_only=False, mmap=True)
    split, source = pubmed_split(args)
    return payload["neuro_vectors"], np.asarray(payload["pmid"]), split, source


def _rows(mm_tensor, rows):
    import torch

    return torch.as_tensor(np.asarray(mm_tensor[torch.as_tensor(np.asarray(rows, dtype=np.int64))]))


def panel_a(args, device):
    """PubMed recall of the 5 CV fold checkpoints (docs/figures/20 cells 9-10; evaluated, not retrained). Without the
    fold files: the single-split model on the test split (docs/figures/23)."""
    cache = cache_file(args, "fig5a_pubmed_cv", excl=True)
    if cache.exists() and not args.force:
        return load_pt(cache)
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from _shared import load_dataset
    from neurovlm.metrics.retrieval import recall_curve

    dd = data_dir()
    fold_dir = args.model_dir if fold_files(args.model_dir) else dd  # the run's CV folds, else data_dir's
    if args.model_dir is not None and fold_dir == dd:
        log.warning("--model-dir %s has no complete *_kfold-{0..4}.pt set; data_dir folds are used", args.model_dir)
    fold_paths = {i: {k: fold_dir / f"{k}_kfold-{i}.pt" for k in FOLD_PARTS} for i in range(5)}
    have_folds = all(p.exists() for d in fold_paths.values() for p in d.values())
    nv_path, sp_path = dd / "neuro_vectors.pt", dd / "latent_specter2_adhoc.pt"
    out = {"step": args.step, "folds": []}

    if have_folds and nv_path.exists() and sp_path.exists():
        from sklearn.model_selection import KFold

        with timed("a) 5-fold evaluation", log, args.cache_dir, stage=STEM):
            # test folds as in the notebook: neuro_vectors rows with <= 100 coordinates, KFold(5, shuffle, seed 0);
            # excluded pmids leave the held-out parts, the folds stay the same
            payload = torch.load(nv_path, map_location="cpu", weights_only=False, mmap=True)
            neuro_vectors, pmids = payload["neuro_vectors"], np.asarray(payload["pmid"])
            df_coords = load_dataset("pubmed_coordinates")
            counts = df_coords[df_coords["pmid"].isin(pmids)]["pmid"].value_counts()
            keep = pd.Series(pmids).isin(counts.index[counts <= 100].values).to_numpy()
            rows_keep, pmids_k = np.flatnonzero(keep), pmids[keep]
            sp = torch.load(sp_path, map_location="cpu", weights_only=False)
            lat, sp_pmids = sp["latent"].float(), np.asarray(sp["pmid"])
            m = pd.Series(sp_pmids).isin(pmids_k).to_numpy()
            lat, sp_pmids = lat[m], sp_pmids[m]
            if not (sp_pmids == pmids_k).all():
                raise RuntimeError("latent_specter2_adhoc.pt pmids do not align with neuro_vectors.pt")
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            ok_cv = pubmed_keep(args, pmids_k, key="PubMed CV pool (all folds)", log=log)
            for i, (_, te) in enumerate(kf.split(np.arange(len(pmids_k)))):
                n_excl = int((~ok_cv[te]).sum())
                te = te[ok_cv[te]]
                if args.max_samples:
                    te = te[: args.max_samples]
                mods = {k: torch.load(fold_paths[i][k], map_location="cpu", weights_only=False).to(device).eval()
                        for k in FOLD_PARTS}
                X = _rows(neuro_vectors, rows_keep[te])
                with torch.no_grad():
                    latent_img = to_device_batches(mods["autoencoder"].encoder, X, device, batch_size=1024)
                    proj_img = F.normalize(mods["proj_head_image_infonce"](latent_img.to(device)), dim=1).cpu()
                    proj_txt = F.normalize(mods["proj_head_text_infonce"](F.normalize(lat[te].to(device), dim=1)), dim=1).cpu()
                t2i, i2t = recall_curve(proj_txt, proj_img, step=args.step)
                n = len(te)
                x = np.arange(0, n, args.step) / n
                out["folds"].append({"n": n, "n_excluded": n_excl, "x": x, "t2i": t2i.numpy(), "i2t": i2t.numpy(),
                                     "auc_t2i": trapz_auc(x, t2i.numpy()), "auc_i2t": trapz_auc(x, i2t.numpy())})
                log.info("fold %d n=%d (excluded %d) AUC t2i=%.3f i2t=%.3f", i, n, n_excl, out["folds"][-1]["auc_t2i"], out["folds"][-1]["auc_i2t"])
                del mods, X, latent_img
                free_gpu()
            out["source"] = f"5-fold checkpoints from {rel_path(fold_dir)}"
            note_source("cv_folds", f"{rel_path(fold_dir)}/{{{','.join(FOLD_PARTS)}}}_kfold-{{0..4}}.pt")
    else:
        missing = [str(p) for d in fold_paths.values() for p in d.values() if not p.exists()]
        log.warning("fold checkpoints or local tensors missing (%d files, e.g. %s); evaluating the single-split model "
                    "on the PubMed test split", len(missing), missing[:1])
        with timed("a) single-split evaluation", log, args.cache_dir, stage=STEM):
            from _shared import load_latent

            df_pubs = load_dataset("pubmed_text")
            test_pmids = df_pubs.loc[pubmed_test_mask(args, df_pubs, log), "pmid"].to_numpy()
            test_pmids = test_pmids[pubmed_keep(args, test_pmids, key="PubMed test", log=log)]
            lt, pt = load_latent("pubmed_text")
            mt = pd.Series(np.asarray(pt)).isin(test_pmids).to_numpy()
            lt = lt[mt].float()
            heads = load_models(["proj_head_text_infonce", "proj_head_image_infonce"], device, args.model_dir, log)
            note_source("cv_folds", "none: single-split fallback")
            if run_file(args.model_dir, "autoencoder") is not None:
                ae = load_models(["autoencoder"], device, args.model_dir, log)["autoencoder"]
                images, pi = load_dataset("pubmed_images")
                mi = pd.Series(np.asarray(pi)).isin(test_pmids).to_numpy()
                li = to_device_batches(ae.encoder, images[mi], device, batch_size=1024)
            else:
                li, pi = load_latent("pubmed_images")
                note_source("autoencoder", "released:neurovlm/NeuroAutoEncoder (HF pubmed_images latents)")
                mi = pd.Series(np.asarray(pi)).isin(test_pmids).to_numpy()
                li = li[mi].float()
            if not np.array_equal(np.asarray(pt)[mt], np.asarray(pi)[mi]):
                raise RuntimeError("pubmed_text / pubmed_images latents are not aligned on the test split")
            if args.max_samples:
                lt, li = lt[: args.max_samples], li[: args.max_samples]
            with torch.no_grad():
                proj_txt = F.normalize(heads["proj_head_text_infonce"](F.normalize(lt.to(device), dim=1)), dim=1).cpu()
                proj_img = F.normalize(heads["proj_head_image_infonce"](li.to(device)), dim=1).cpu()
            t2i, i2t = recall_curve(proj_txt, proj_img, step=args.step)
            n = len(lt)
            x = np.arange(0, n, args.step) / n
            out["folds"].append({"n": n, "x": x, "t2i": t2i.numpy(), "i2t": i2t.numpy(),
                                 "auc_t2i": trapz_auc(x, t2i.numpy()), "auc_i2t": trapz_auc(x, i2t.numpy())})
            out["source"] = "released single-split model (fold checkpoints unavailable)"
            free_gpu()
    out["auc_t2i"] = float(np.mean([f["auc_t2i"] for f in out["folds"]]))
    out["auc_i2t"] = float(np.mean([f["auc_i2t"] for f in out["folds"]]))
    save_pt(cache, out)
    return out


def panel_b(args):
    """NeuroVault recall with three target maps per paper, read from the a3 analysis (--bounds-dir; computed with this
    figure's autoencoder, InfoNCE heads and exclusions, checked here): upper bound = the paper's map the model scores
    highest for the paper's text (the paper's recipe), lower bound = the one it scores lowest, multi-target = every map
    of the paper is a target. Chance per curve: the same rule with the texts deranged across papers (a3)."""
    import pandas as pd

    root = Path(args.bounds_dir)
    cfg = read_json(root / "config.json")
    want = resolve_sources(args.model_dir)
    other = {k: v for k, v in cfg["model_sources"].items() if want.get(k) != v}
    if other or cfg["exclusions"]["enabled"] != excl_on(args):
        raise SystemExit(f"{rel_path(root)} was computed with other components {other} or exclusions; rerun "
                         "scripts/figures/review/core/a3_neurovault_bounds/run_analysis.py for this model "
                         "(--out <dir>) and pass --bounds-dir <dir>, or leave b out with --panels acd")
    if args.max_samples:
        log.warning("b) the a3 curves cover all %d papers (--max-samples does not apply)", cfg["n_publications"])
    curves = pd.read_csv(root / "recall_curves.csv")
    metrics = pd.read_csv(root / "bounds_metrics.csv")
    curves, metrics = curves[curves["variant"] == B_VARIANT], metrics[metrics["variant"] == B_VARIANT]
    out = {"n": int(cfg["n_publications"]), "n_images": int(cfg["n_images"]), "variant": B_VARIANT,
           "source": rel_path(root / "recall_curves.csv"), "a3_model_sources": cfg["model_sources"], "curves": {}}
    for rule, defs in B_RULES.items():
        for direction, definition in defs.items():
            c = curves[(curves["direction"] == direction) & (curves["definition"] == definition)]
            c = c.sort_values("normalized_k")
            m = metrics[(metrics["direction"] == direction) & (metrics["definition"] == definition)].iloc[0]
            out["curves"][f"{rule}:{direction}"] = {
                "definition": definition, "x": c["normalized_k"].to_numpy(), "recall": c["recall"].to_numpy(),
                "chance": c["chance"].to_numpy(),
                "auc": float(m["recall_curve_auc"]), "chance_auc": float(m["chance_recall_curve_auc"]),
                "n_queries": int(m["n_queries"]), "mean_pool_size": float(m["mean_pool_size"])}
    log.info("b) %s", ", ".join(f"{k} {v['auc']:.3f}" for k, v in out["curves"].items()))
    return out


def panel_c(args, device):
    """Autoencoder voxel ROC and bits-per-pixel gain on PubMed test and NeuroVault (docs/figures/11)."""
    cache = cache_file(args, "fig5c_autoencoder", excl=True)
    if cache.exists() and not args.force:
        return load_pt(cache)
    import pandas as pd
    import torch
    from _shared import load_dataset
    from neurovlm.metrics.text_to_brain import compute_ae_performance

    dd = data_dir()
    ae = load_models(["autoencoder"], device, args.model_dir, log)["autoencoder"]
    out = {"datasets": {}}
    with timed("c) autoencoder test sets", log, args.cache_dir, stage=STEM):
        local = _pubmed_local(args)
        if local is not None:
            mm, pmids, split, split_src = local
            rows = np.flatnonzero(pd.Series(pmids).isin(split["test"]).to_numpy())
            rows = rows[pubmed_keep(args, pmids[rows], key="PubMed test", log=log)]
            if args.max_samples:
                rows = rows[: args.max_samples]
            X_pm = _rows(mm, rows).float()
            src_pm = f"neuro_vectors.pt + {split_src} (test)"
        else:
            log.warning("neuro_vectors.pt missing: using HF pubmed_images with the pubmed_text test flag")
            images, pi = load_dataset("pubmed_images")
            df = load_dataset("pubmed_text")
            m = pd.Series(np.asarray(pi)).isin(df.loc[df["test"].fillna(False).astype(bool), "pmid"]).to_numpy().copy()
            rows_t = np.flatnonzero(m)
            m[rows_t[~pubmed_keep(args, np.asarray(pi)[rows_t], key="PubMed test", log=log)]] = False
            X_pm = images[m].float()
            if args.max_samples:
                X_pm = X_pm[: args.max_samples]
            src_pm = "HF pubmed_images (test flag)"
        nv_path = dd / "neurovault.pt"
        if nv_path.exists():
            nv = torch.load(nv_path, map_location="cpu", weights_only=False)
            i_ok = neurovault_keep(args, nv["df_neuro"]["doi"].tolist(), key="NeuroVault images (neurovault.pt)", log=log,
                                   image_ids=nv["df_neuro"]["id"].to_numpy())
            X_nv = (torch.from_numpy(np.asarray(nv["neuro_clustered"])[i_ok]) > 0).float()
            src_nv = "neurovault.pt neuro_clustered > 0"
        else:
            log.warning("neurovault.pt missing: using HF neurovault_images > 0 (approximation of the clustered maps)")
            img = load_dataset("neurovault_images").float()
            meta = load_dataset("neurovault_images_meta").reset_index(drop=True)
            finite = ~img.isnan().any(dim=1)
            img, meta = img[finite], meta[finite.numpy()].reset_index(drop=True)
            i_ok = neurovault_keep(args, meta["doi"].tolist(), key="NeuroVault images (HF)", log=log, image_ids=meta["id"].to_numpy())
            X_nv = (img[torch.from_numpy(i_ok)] > 0).float()
            src_nv = "HF neurovault_images > 0"
        if args.max_samples:
            X_nv = X_nv[: args.max_samples]
        for name, X, src in (("PubMed test", X_pm, src_pm), ("NeuroVault", X_nv, src_nv)):
            logits = to_device_batches(ae, X, device, batch_size=256)
            fpr, tpr, pct, roc_auc = compute_ae_performance(X.cpu(), logits.cpu())
            keep = np.unique(np.linspace(0, len(fpr) - 1, 2000).astype(int))
            out["datasets"][name] = {"fpr": fpr[keep], "tpr": tpr[keep], "auc": float(roc_auc),
                                     "pct": np.asarray(pct, dtype="float32"), "n": int(len(X)), "source": src}
            log.info("c) %s n=%d AUROC=%.3f median bpp gain=%.1f%%", name, len(X), roc_auc, float(np.median(pct)))
            del logits
            free_gpu()
    save_pt(cache, out)
    return out


def panel_d(args, device):
    """Network-label confusion matrix (docs/figures/13 cell 8)."""
    cache = cache_file(args, "fig5d_confusion")
    if cache.exists() and not args.force:
        return load_pt(cache)
    import torch
    import torch.nn.functional as F
    from _shared import load_masker
    from sklearn.metrics import confusion_matrix

    with timed("d) network labelling", log, args.cache_dir, stage=STEM):
        nets = load_network_list()
        if len(nets) != len(LABEL_MAP):
            raise RuntimeError(f"expected {len(LABEL_MAP)} non-ICA network maps, found {len(nets)}")
        for (atlas, name, _), (lab_name, _) in zip(nets, LABEL_MAP):
            if name != lab_name:
                raise RuntimeError(f"network order changed: {atlas}/{name} vs LABEL_MAP {lab_name}")
        masker = load_masker()
        mods = load_models(["specter", "autoencoder", "proj_head_text_infonce", "proj_head_image_infonce"], device,
                           args.model_dir, log)
        emb_cache = cache_file(args, "fig5d_networks_emb")
        if emb_cache.exists() and not args.force:
            net_emb = load_pt(emb_cache)
        else:
            vecs = np.stack([network_binary_vector(img, masker) for _, _, img in nets])
            net_emb = to_device_batches(mods["autoencoder"].encoder, vecs, device, batch_size=64)
            save_pt(emb_cache, net_emb)
        e = encode_texts(mods["specter"], list(TEXT_LABELS.values()), batch_size=8)
        with torch.no_grad():
            text_c = F.normalize(mods["proj_head_text_infonce"](e.to(device)), dim=1).cpu()
            img_c = F.normalize(mods["proj_head_image_infonce"](net_emb.to(device)), dim=1).cpu()
        sim = img_c @ text_c.T
        pred = sim.argmax(dim=1).numpy()
        key_to_idx = {k: i for i, k in enumerate(CLASS_KEYS)}
        y_true = np.array([key_to_idx.get(lab, -1) for _, lab in LABEL_MAP])
        mask = y_true >= 0
        cm = confusion_matrix(y_true[mask], pred[mask], labels=np.arange(len(CLASS_KEYS)))
        out = {"cm_counts": cm, "class_names": CLASS_NAMES, "accuracy": float((pred[mask] == y_true[mask]).mean()),
               "n": int(mask.sum()), "sim": sim.numpy(), "y_true": y_true, "pred": pred}
        log.info("d) n=%d accuracy=%.3f", out["n"], out["accuracy"])
        del mods
        free_gpu()
    save_pt(cache, out)
    return out


def render(args, a, b, c, d):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fs = args.font_size
    plt.rcParams["savefig.bbox"] = "standard"  # exactly FIG_FULL_W wide
    # layout in inches: top row a | b + its legend column (square recall panels); bottom row c1 | c2 (square ROC,
    # c's legend inside it) | d, the matrix with the class names left of it and, at 60 degrees, below it.
    # The tops of c and of the matrix are level.
    fig_w = FIG_FULL_W
    side = 1.45                                 # a and b
    x_a, x_b = 0.45, 0.45 + side + 0.47
    names_w = max(text_width_in(n, fs) for n in CLASS_NAMES)
    names_h = names_w * np.sin(np.pi / 3) + 0.06  # the same names at 60 degrees
    m = 1.07                                    # matrix d: cells wide enough for two two-digit numbers side by side
    x_m = fig_w - 0.04 - m
    y_m = 0.03 + 0.14 + names_h + 0.07          # matrix bottom: "Predicted", class names, ticks
    x_c1, w_c1 = 0.48, 0.7
    x_c2 = x_c1 + w_c1 + 0.42
    s_c = x_m - 0.04 - names_w - 0.2 - 0.12 - x_c2  # c2 square up to d's "True" label
    y_c = y_m + m - s_c                         # c axes bottom, the top level with the matrix
    y_head = y_m + m + 0.39                     # top of the row-2 headers (c Autoencoder, d Network classification)
    y_top = y_head + 0.43                       # bottom of a and b (x labels below)
    fig_h = y_top + side + 0.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    def ax_in(x0, y0, w, h):
        return fig.add_axes([x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h])

    def header(letter, x_in, y_in, text):
        panel_label(fig, letter, x_in / fig_w, y_in / fig_h, size_delta=1.0)  # bold 9 pt
        fig.text((x_in + 0.18) / fig_w, y_in / fig_h, text, fontsize=fs + 2, va="top")

    def tint(colour, f):
        return tuple(v + (1.0 - v) * f for v in to_rgb(colour))

    ax_a, ax_b = ax_in(x_a, y_top, side, side), ax_in(x_b, y_top, side, side)
    ax_c1 = ax_in(x_c1, y_c, w_c1, s_c)
    ax_c2 = ax_in(x_c2, y_c, s_c, s_c)
    ax_d = ax_in(x_m, y_m, m, m)
    y_label = fig_h - 0.03                      # top edge of the bold panel letters a, b
    leg_kw = dict(fontsize=fs, title_fontsize=fs, handlelength=1.4, handletextpad=0.4, labelspacing=0.25,
                  borderpad=0.0, borderaxespad=0.0, alignment="left", frameon=False)

    # a) PubMed CV
    if a is not None:
        recall_axes(ax_a)
        for k, f in enumerate(a["folds"]):
            ax_a.plot(f["x"], f["t2i"], color=BLUE, alpha=0.6, lw=1.0,
                      label=f"Text-to-image {a['auc_t2i']:.2f}" if k == 0 else None)
            ax_a.plot(f["x"], f["i2t"], color=ORANGE, alpha=0.6, lw=1.0,
                      label=f"Image-to-text {a['auc_i2t']:.2f}" if k == 0 else None)
        h, lab = ax_a.get_legend_handles_labels()  # chance diagonal first
        ax_a.legend(h[1:] + h[:1], lab[1:] + [f"{lab[0]} 0.50"], title="AUC", loc="lower right",
                    **leg_kw | dict(alignment="right", handlelength=1.0, borderaxespad=0.2))  # below the diagonal
        note = f"{len(a['folds'])}-fold CV" if len(a["folds"]) > 1 else "single split"
        ax_a.text(0.03, 0.97, note, transform=ax_a.transAxes, va="top", fontsize=fs, color="0.3")
        header("a", 0.02, y_label, "PubMed contrastive recall")
    # b) NeuroVault: upper bound, multi-target and lower bound per direction (a3), one hue per direction; dashed = the
    # same rule with texts shuffled across papers (a3 chance)
    if b is not None:
        ax_b.set_xlabel(r"$k/n$ (fraction of all candidates)")
        ax_b.set_ylabel(r"Recall@$k$")
        ax_b.set_box_aspect(1)
        ax_b.grid(alpha=0.25)
        legends = []
        for direction, colour, name in (("text_to_brain", BLUE, "Text-to-image"),
                                        ("brain_to_text", ORANGE, "Image-to-text")):
            handles = []
            for rule in B_RULES:
                cv = b["curves"][f"{rule}:{direction}"]
                col = tint(colour, B_TINT[rule])
                ax_b.plot(cv["x"], cv["chance"], color=col, lw=0.9, ls=(0, (3, 1.5)))
                handles += ax_b.plot(cv["x"], cv["recall"], color=col, lw=1.1,
                                     label=f"{rule} {cv['auc']:.2f} ({cv['chance_auc']:.2f})")
            legends.append(dict(handles=handles, title=name))
        legends.append(dict(handles=[Line2D([], [], color="0.35", lw=0.9, ls=(0, (3, 1.5)), label="chance")]))
        # legend column right of b, stacked from the top under the "AUC (chance)" header
        x_leg = (side + 0.06) / side
        head = ax_b.text(x_leg, 1.0, "AUC (chance)", transform=ax_b.transAxes, va="top", fontsize=fs)
        renderer = fig.canvas.get_renderer()
        y_next = ax_b.transAxes.inverted().transform(head.get_window_extent(renderer))[0, 1] - 0.04
        for i, kw in enumerate(legends):
            leg = ax_b.legend(loc="upper left", bbox_to_anchor=(x_leg, y_next), **kw, **leg_kw)
            if i < len(legends) - 1:
                ax_b.add_artist(leg)  # kept when the next legend() call replaces the axes legend
            y_next = ax_b.transAxes.inverted().transform(leg.get_window_extent(renderer))[0, 1] - 0.06
        ax_b.text(0.03, 0.97, f"n={b['n']}", transform=ax_b.transAxes, va="top", fontsize=fs, color="0.3")
        header("b", x_b - 0.43, y_label, "NeuroVault contrastive recall")
    # c) autoencoder: one legend for both sub-panels, inside the voxel ROC
    if c is not None:
        bins = np.linspace(-20, 90, 45)
        for (name, ds), col in zip(c["datasets"].items(), (BLUE, ORANGE)):
            ax_c1.hist(ds["pct"], bins=bins, alpha=0.6, color=col)
        ax_c1.axvline(0.0, ls="--", color="black", lw=0.9)
        ax_c1.set_xlabel("Gain vs baseline (%)")
        ax_c1.set_ylabel("Count")
        ax_c1.set_title("Bits per pixel")
        for (name, ds), col in zip(c["datasets"].items(), (BLUE, ORANGE)):
            ax_c2.plot(ds["fpr"], ds["tpr"], color=col, lw=1.2)
        ax_c2.plot([0, 1], [0, 1], ls="--", color="black", lw=0.9)
        ax_c2.set_xlabel("False positive rate")
        ax_c2.set_ylabel("True positive rate")
        ax_c2.set_xticks([0, 0.5, 1])
        ax_c2.set_yticks([0, 0.5, 1])
        ax_c2.set_box_aspect(1)  # default data margins (no xlim/ylim at the origin)
        ax_c2.set_title("Voxel ROC")
        ax_c2.grid(alpha=0.25)
        keys = [Patch(color=col, alpha=0.6, label=name) for name, col in zip(c["datasets"], (BLUE, ORANGE))]
        keys.append(Line2D([], [], ls="--", color="black", lw=0.9, label="Chance"))
        # one key for both sub-panels, in the ROC's empty lower right (below the diagonal); AUROCs in the caption
        ax_c2.legend(handles=keys, loc="lower right", **leg_kw | dict(handlelength=1.0, borderaxespad=0.15,
                                                                      alignment="right"))
        header("c", 0.02, y_head, "Autoencoder")
    # d) confusion matrix: per cell the percentage of its true class (empty = 0); the same class names on both axes
    if d is not None:
        cm = np.asarray(d["cm_counts"])
        row = cm.sum(1, keepdims=True)
        props = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row != 0)
        ax_d.imshow(props, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        C = len(d["class_names"])
        for i in range(C):
            for j in range(C):
                if cm[i, j]:
                    ax_d.text(j, i, f"{100 * props[i, j]:.0f}", ha="center", va="center", fontsize=fs,
                              color="white" if props[i, j] > 0.5 else "black")
        ax_d.set_xticks(range(C), d["class_names"], rotation=60, ha="right", va="center", rotation_mode="anchor")
        ax_d.set_yticks(range(C), d["class_names"])
        ax_d.tick_params(length=2, pad=2)
        ax_d.set_xlabel("Predicted", labelpad=2)
        ax_d.set_ylabel("True", labelpad=2)
        ax_d.set_title("% of true class", fontsize=fs + 1)  # n and accuracy in the caption
        for sp in ("top", "right"):
            ax_d.spines[sp].set_visible(True)
        header("d", x_m - 0.04 - names_w - 0.3, y_head, "Network classification")
    else:
        ax_d.set_axis_off()
    return fig


def main():
    args = parse_args()
    banner(log, args, STEM)
    device = select_device(args.device)
    a = panel_a(args, device) if "a" in args.panels else None
    b = panel_b(args) if "b" in args.panels else None
    c = panel_c(args, device) if "c" in args.panels else None
    d = panel_d(args, device) if "d" in args.panels else None
    with timed("render", log, args.cache_dir, stage=STEM):
        fig = render(args, a, b, c, d)
        finish(fig, STEM, args, log, components=COMPONENTS, text_dump={
            "a": None if a is None else {"source": a["source"], "auc_t2i": a["auc_t2i"], "auc_i2t": a["auc_i2t"],
                                         "fold_n": [f["n"] for f in a["folds"]]},
            "b": None if b is None else {
                "source": b["source"], "a3_variant": b["variant"], "a3_model_sources": b["a3_model_sources"],
                "n_papers": b["n"], "n_maps": b["n_images"],
                "plotted": "solid lines, one hue per direction (blue text->image, orange image->text), lighter = "
                           "weaker target: upper = the paper's map the model scores highest for its text (paper "
                           "recipe; a3 paper_upper), multi = every map of the paper is a target (text->image: hit when "
                           "any of its maps is in the top k of all maps, a3 oracle_any_true; image->text: every map "
                           "ranks the 298 texts, a3 per_image), lower = the map it scores lowest (a3 worst_image); "
                           "dashed lines in the same colour = chance of that curve; legend: AUC (chance AUC)",
                "chance": B_CHANCE,
                "auc": {k: round(v["auc"], 4) for k, v in b["curves"].items()},
                "chance_auc": {k: round(v["chance_auc"], 4) for k, v in b["curves"].items()},
                "n_queries": {k: v["n_queries"] for k, v in b["curves"].items()},
                "mean_pool_size": {k: round(v["mean_pool_size"], 1) for k, v in b["curves"].items()},
                "asymmetry": B_ASYMMETRY},
            "c": None if c is None else {k: {"auc": v["auc"], "n": v["n"], "source": v["source"],
                                             "median_pct_gain": float(np.median(v["pct"]))} for k, v in c["datasets"].items()},
            "d": None if d is None else {"accuracy": d["accuracy"], "n": d["n"], "classes": d["class_names"],
                                         "labels": "full class names on both axes (columns at 60 degrees); cells: % "
                                                   "of the true class (rows), empty = 0",
                                         "cm_counts": np.asarray(d["cm_counts"]).tolist()},
        })


if __name__ == "__main__":
    main()
