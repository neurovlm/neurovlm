#!/usr/bin/env python
"""Reviewer 3.5: is the title of each [NETWORK] / [REGION] / [FUNCTION] caption of the final model correct for its map?

Captions network-atlas, ICA and PubMed test maps once per token (GPU, ~2 min; cached after that), scores the titles
against atlas regions, Yeo-7 networks and Neurosynth-decoded terms, and compares with titles shuffled across maps.
  .venv/bin/python scripts/figures/review/core/a6_steering_tokens/run_correctness.py   # then plot_correctness.py
  ... run_correctness.py --strict-containment   # previous [REGION] rule (no containment clause), e.g. with --out
  ... run_correctness.py --title-constraint none   # free [NETWORK] / [REGION] titles (default: only the bank titles)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _a6_common as C  # noqa: E402
import _final_model as M  # noqa: E402

# canonical names of the final run's stage-14 banks, next to the model (scripts/models/_common_models.title_vocabulary;
# --model-dir sets it in main)
VOCAB_FILE = M.DEFAULT_MODEL_DIR / "title_vocabulary.json"

log = C.get_logger("a6_correctness")
TOKENS = ("network", "region", "function")
# caption cleanup as scripts/models/_common_models.py (validate_models.py)
STEERING_TAG_RE = re.compile(r"(?i)[ \t]*\[(?:network|region|function)\][ \t]*")


def strip_steering_tags(text: str) -> str:
    """Drop the [NETWORK] / [REGION] / [FUNCTION] tags the LM sometimes writes into a caption, with the spaces around
    them (a tag between two words leaves one space); lines without a tag are left as they are."""
    return "\n".join(STEERING_TAG_RE.sub(" ", line).strip() if STEERING_TAG_RE.search(line) else line
                     for line in str(text).split("\n"))


ICA_ATLASES = ("UKBICA", "HCPICA")
OLD_CACHE = HERE / "outputs" / "cache" / "generations_9774092757ec_excl.parquet"  # captions of the removed released-model analysis

# [REGION]: a named region holds >= 5 % of the map's top voxels or its peak, or (unless --strict-containment) at least
# half of the region lies in the top voxels: a small structure (amygdala ~90, hypothalamus ~26 voxels) inside the map
# cannot reach 5 % of a 2,854-voxel top 10 %.
MIN_INSIDE = 0.5
# Harvard-Oxford cortical labels (bilateral) by short code
HO_CORT = {
    "FP": "Frontal Pole", "INS": "Insular Cortex", "SFG": "Superior Frontal Gyrus", "MFG": "Middle Frontal Gyrus",
    "IFGt": "Inferior Frontal Gyrus, pars triangularis", "IFGo": "Inferior Frontal Gyrus, pars opercularis",
    "PreCG": "Precentral Gyrus", "TP": "Temporal Pole", "STGa": "Superior Temporal Gyrus, anterior division",
    "STGp": "Superior Temporal Gyrus, posterior division", "MTGa": "Middle Temporal Gyrus, anterior division",
    "MTGp": "Middle Temporal Gyrus, posterior division", "MTGto": "Middle Temporal Gyrus, temporooccipital part",
    "ITGa": "Inferior Temporal Gyrus, anterior division", "ITGp": "Inferior Temporal Gyrus, posterior division",
    "ITGto": "Inferior Temporal Gyrus, temporooccipital part", "PostCG": "Postcentral Gyrus", "SPL": "Superior Parietal Lobule",
    "SMGa": "Supramarginal Gyrus, anterior division", "SMGp": "Supramarginal Gyrus, posterior division", "AG": "Angular Gyrus",
    "LOCs": "Lateral Occipital Cortex, superior division", "LOCi": "Lateral Occipital Cortex, inferior division",
    "ICC": "Intracalcarine Cortex", "FMC": "Frontal Medial Cortex",
    "SMA": "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "SubC": "Subcallosal Cortex",
    "PaCG": "Paracingulate Gyrus", "ACCg": "Cingulate Gyrus, anterior division", "PCCg": "Cingulate Gyrus, posterior division",
    "PCun": "Precuneous Cortex", "Cun": "Cuneal Cortex", "FOrb": "Frontal Orbital Cortex",
    "PHGa": "Parahippocampal Gyrus, anterior division", "PHGp": "Parahippocampal Gyrus, posterior division", "LG": "Lingual Gyrus",
    "TFa": "Temporal Fusiform Cortex, anterior division", "TFp": "Temporal Fusiform Cortex, posterior division",
    "TOF": "Temporal Occipital Fusiform Cortex", "OF": "Occipital Fusiform Gyrus", "FO": "Frontal Opercular Cortex",
    "CO": "Central Opercular Cortex", "PO": "Parietal Opercular Cortex", "PP": "Planum Polare",
    "HG": "Heschl's Gyrus (includes H1 and H2)", "PT": "Planum Temporale", "SCC": "Supracalcarine Cortex", "OP": "Occipital Pole",
}
HO_SUB = {"THA": "Thalamus", "CAU": "Caudate", "PUT": "Putamen", "PAL": "Pallidum", "HIP": "Hippocampus", "AMY": "Amygdala",
          "NAc": "Accumbens", "BS": "Brain-Stem"}  # left + right merged
# (name, regex on the lower-cased title, atlas parts), specific before general: a match is blanked before the next
# pattern is tried, so "primary visual cortex" is not also counted as "visual cortex".
REGIONS = [
    ("dorsal anterior cingulate cortex", r"dorsal anterior cingulate|\bdacc\b", "ACCg PaCG"),
    ("subgenual cingulate cortex", r"subgenual( anterior)?( cingulate)?|\bsgacc\b", "SubC ACCg"),
    ("anterior cingulate cortex", r"anterior cingulate|\bacc\b", "ACCg PaCG"),
    ("posterior cingulate cortex", r"posterior cingulate|\bpcc\b", "PCCg"),
    ("retrosplenial cortex", r"retrosplenial", "PCCg PCun ICC"),
    ("cingulate cortex", r"cingulate", "ACCg PCCg PaCG"),
    ("pre-supplementary motor area", r"pre-?\s?supplementary motor|\bpre-?sma\b", "SMA SFG PaCG"),
    ("supplementary motor area", r"supplementary motor|\bsma\b|juxtapositional", "SMA"),
    ("frontal eye fields", r"frontal eye fields?|\bfef\b", "PreCG MFG SFG"),
    ("dorsolateral prefrontal cortex", r"dorsolateral (pre)?frontal|\bdlpfc\b", "MFG SFG"),
    ("ventrolateral prefrontal cortex", r"ventrolateral prefrontal|\bvlpfc\b", "IFGt IFGo FO FOrb"),
    ("ventromedial prefrontal cortex", r"ventromedial prefrontal|\bvmpfc\b", "FMC SubC PaCG FP"),
    ("medial prefrontal cortex", r"medial prefrontal|\bmpfc\b|frontal medial", "FMC PaCG SFG FP SubC"),
    ("orbitofrontal cortex", r"orbito-?frontal|orbital frontal|frontal orbital|\bofc\b", "FOrb FMC"),
    ("frontal pole", r"(lateral )?frontopolar|frontal pole", "FP"),
    ("prefrontal cortex", r"prefrontal|\bpfc\b", "FP SFG MFG IFGt IFGo FMC FOrb PaCG SubC"),
    ("inferior frontal gyrus", r"inferior frontal|\bifg\b|broca'?s?|pars opercularis|pars triangularis", "IFGt IFGo"),
    ("frontal operculum", r"frontal opercul\w*", "FO"),
    ("middle frontal gyrus", r"middle frontal|\bmfg\b", "MFG"),
    ("superior frontal gyrus", r"superior frontal|\bsfg\b", "SFG"),
    ("frontal lobe", r"frontal (lobes?|cortex)", "FP SFG MFG IFGt IFGo PreCG FMC FOrb FO SMA PaCG SubC"),
    ("premotor cortex", r"premotor|\bpmc\b|\bpmd\b|\bpmv\b", "PreCG SFG MFG"),
    ("foot motor cortex", r"foot motor", "PreCG SMA"),
    ("paracentral lobule", r"paracentral", "PreCG PostCG SMA"),
    ("primary motor cortex", r"((primary|hand|face) )?motor cortex|precentral|\bm1\b", "PreCG"),
    ("secondary somatosensory cortex", r"secondary somatosensory|\bs2\b|\bsii\b", "PO CO"),
    ("primary somatosensory cortex", r"(primary )?somatosensory cortex|postcentral|\bs1\b", "PostCG"),
    ("primary gustatory cortex", r"(primary )?gustatory cortex", "INS FO"),
    ("vestibular cortex", r"vestibular cortex", "PO INS"),
    ("insula", r"(anterior |posterior )?insula\w*|interoceptive cortex", "INS"),
    ("olfactory cortex", r"olfactory cortex|piriform", "FOrb PHGa AMY"),
    ("parietal operculum", r"parietal opercul\w*", "PO"),
    ("central operculum", r"central opercul\w*|rolandic opercul\w*", "CO"),
    ("temporoparietal junction", r"temporo-?parietal junction|\btpj\b", "SMGp AG STGp"),
    ("wernicke's area", r"wernicke'?s?", "STGp PT SMGp"),
    ("intraparietal sulcus", r"intraparietal|\bips\b", "SPL SMGa SMGp AG"),
    ("superior parietal lobule", r"superior parietal|\bspl\b", "SPL"),
    ("inferior parietal lobule", r"inferior parietal|\bipl\b", "SMGa SMGp AG"),
    ("supramarginal gyrus", r"supramarginal|\bsmg\b", "SMGa SMGp"),
    ("angular gyrus", r"angular gyrus", "AG"),
    ("posterior parietal cortex", r"posterior parietal|\bppc\b", "SPL SMGa SMGp AG PCun"),
    ("precuneus", r"precune\w*", "PCun"),
    ("parietal cortex", r"parietal (cortex|lobes?)", "SPL SMGa SMGp AG PostCG PCun PO"),
    ("primary auditory cortex", r"primary auditory( cortex)?|heschl'?s?|\ba1\b", "HG"),
    ("planum temporale", r"planum temporale", "PT"),
    ("auditory cortex", r"(voice-selective |secondary )?auditory (association )?cortex", "HG PT PP STGa STGp"),
    ("superior temporal sulcus", r"superior temporal sulcus|\bsts\b", "STGa STGp MTGa MTGp"),
    ("superior temporal gyrus", r"superior temporal|\bstg\b", "STGa STGp PT PP HG"),
    ("middle temporal visual area (MT)", r"middle temporal visual|\bmt\+?(?![a-z])|\bv5\b|motion-selective visual( cortex)?", "MTGto LOCi ITGto"),
    ("middle temporal gyrus", r"middle temporal|\bmtg\b", "MTGa MTGp MTGto"),
    ("inferior temporal gyrus", r"inferior temporal|\bitg\b", "ITGa ITGp ITGto"),
    ("temporal pole", r"temporal pole|anterior temporal lobe|\batl\b", "TP"),
    ("visual word form area", r"visual word form area|\bvwfa\b", "TOF TFp ITGto"),
    ("fusiform gyrus", r"fusiform|\bffa\b", "TFa TFp TOF OF"),
    ("parahippocampal gyrus", r"parahippocampal( place)?|\bppa\b|entorhinal|perirhinal", "PHGa PHGp"),
    ("medial temporal lobe", r"medial temporal lobe|\bmtl\b", "HIP AMY PHGa PHGp TFa"),
    ("hippocampus", r"\bhippocamp\w*", "HIP"),
    ("amygdala", r"amygdal\w*", "AMY"),
    ("temporal cortex", r"temporal (cortex|lobes?)", "TP STGa STGp MTGa MTGp MTGto ITGa ITGp ITGto TFa TFp PP PT HG"),
    ("primary visual cortex", r"primary visual( cortex)?|striate cortex|calcarine( cortex| sulcus)?|\bv1\b", "ICC SCC OP"),
    ("lateral occipital cortex", r"lateral occipital|\bloc\b", "LOCs LOCi"),
    ("extrastriate visual cortex", r"extrastriate( visual)?|color-selective visual|\bv[234]\b", "LOCs LOCi OF LG Cun"),
    ("cuneus", r"\bcuneus|\bcuneal", "Cun"),
    ("lingual gyrus", r"lingual", "LG"),
    ("occipital cortex", r"occipital|visual cortex", "LOCs LOCi ICC SCC Cun LG OF OP"),
    ("hypothalamus", r"hypothalam\w*", "HTH"),
    ("subthalamic nucleus", r"subthalamic", "STH"),
    ("thalamus", r"\bthalam\w*|pulvinar", "THA"),
    ("ventral striatum", r"ventral striatum|(nucleus )?accumbens", "NAc CAU PUT"),
    ("caudate", r"caudate", "CAU"),
    ("putamen", r"putamen", "PUT"),
    ("globus pallidus", r"(globus )?pallid\w*", "PAL"),
    ("striatum", r"striatum|striatal", "CAU PUT NAc"),
    ("basal ganglia", r"basal ganglia", "CAU PUT PAL NAc"),
    ("brainstem", r"brain-?\s?stem|midbrain|\bpons\b|medulla|periaqueductal|\bpag\b", "BS"),
    ("cerebellar vermis", r"(cerebellar )?vermis", "VER"),
    ("cerebellum", r"cerebell\w*|dentate nucleus", "CB"),
]
REGION_TRAIL = r"(\s+(cortex|gyrus|gyri|area|areas|lobule|sulcus|nucleus|nuclei|region|lobe|complex|junction|formation|hemispheres?|place))*"
REGION_RE = [(n, re.compile(r"(" + p + r")" + REGION_TRAIL), parts.split()) for n, p, parts in REGIONS]
REGION_NAMES = [n for n, _, _ in REGIONS]


def regions_in(text: str) -> list[int]:
    """Indices into REGIONS of the regions named in ``text`` (each phrase counted once)."""
    t, hits = " " + str(text).lower() + " ", []
    for k, (_, rx, _) in enumerate(REGION_RE):
        if rx.search(t):
            hits.append(k)
            t = rx.sub(lambda m: " " * len(m.group(0)), t)
    return hits


# [NETWORK]: title regex -> (curated network_test_set_labels keys, Yeo-7 names) it counts as naming
NETWORKS = [
    (r"default|posterior medial|theory of mind", {"default_mode"}, {"default mode network"}),
    (r"dorsal attention|oculomotor|visuospatial", {"attention"}, {"dorsal attention network"}),
    (r"ventral attention|salience|cingulo-?\s?opercular", {"cingulo_opercular"}, {"ventral attention network"}),
    (r"fronto-?\s?parietal|multiple[- ]demand|executive|\bcontrol network", {"frontoparietal_control"}, {"frontoparietal control network"}),
    (r"somato-?\s?motor|sensori-?\s?motor|\bmotor\b|somatosensory", {"motor"}, {"somatomotor network"}),
    (r"visual|object perception|face perception|scene perception", {"visual"}, {"visual network"}),
    (r"auditory|speech perception", {"auditory"}, {"somatomotor network"}),  # Yeo-7 puts auditory cortex in somatomotor
    (r"language|linguistic|lexic|semantic|reading|speech", {"language"}, set()),  # no Yeo-7 counterpart
    (r"limbic|orbitofrontal", set(), {"limbic network"}),
]
NETWORK_RE = [(re.compile(p), cur, yeo) for p, cur, yeo in NETWORKS]


def strip_regions(text: str) -> str:
    """``text`` with region phrases blanked (a [FUNCTION] title that only names a region names no function)."""
    t = " " + str(text).lower() + " "
    for _, rx, _ in REGION_RE:
        t = rx.sub(lambda m: " " * len(m.group(0)), t)
    return t


def networks_in(title: str) -> set[str]:
    """Networks a title names; a title without "network" or "stream" names none."""
    t, out = str(title).lower(), set()
    if not re.search(r"network|stream", t):
        return out
    for rx, cur, yeo in NETWORK_RE:
        if rx.search(t):
            out |= cur | yeo
    return out


# [FUNCTION]: decoding vocabulary = Cognitive Atlas concepts with >= 50 Neurosynth studies + core perceptual / motor /
# affective / social terms that Cognitive Atlas lacks (also >= 50 studies).
CORE_TERMS = ("visual", "auditory", "motor", "somatosensory", "sensorimotor", "reward", "faces", "semantic", "speech", "music",
              "olfactory", "taste", "eye movements", "saccades", "mentalizing", "theory mind", "self referential", "arithmetic",
              "numerical", "scenes", "objects", "motion", "spatial", "interoceptive", "autonomic", "emotional", "affective",
              "conflict", "error", "syntactic", "phonological", "lexical", "action observation", "imitation", "social", "tactile",
              "painful", "words", "sentences", "sounds", "voice", "food", "body", "color", "visual motion", "visual word",
              "word form", "emotional faces", "facial expressions", "monetary reward", "finger tapping", "hand movements",
              "motor imagery", "recollection")
# a title and a decoded term match when they share a synonym group
FUNCTION_GROUPS = {
    "vision": r"visual|vision|visuo|colou?r|\bmotion\b",
    "face": r"\bfac(e|es|ial)\b",
    "object": r"\bobjects?\b|\bshape",
    "scene": r"\bscenes?\b|\bplaces?\b|navigation",
    "auditory": r"auditory|\bsounds?\b|listening|hearing|acoustic|music|\bvoices?\b|\bpitch\b",
    "speech": r"\bspeech\b|phonolog|spoken|\bvoices?\b",
    "language": r"language|linguistic|sentence|syntac|syntax|lexical|semantic|verbal|\bwords?\b|naming|reading|\bspeech\b|phonolog",
    "reading": r"reading|word form|orthograph|dyslexi|visual word",
    "motor": r"\bmotor\b|movements?\b|\baction\b|execution|finger|tapping|\bhand\b|\bfoot\b|imitation|sensorimotor|oromotor|coordination|bimanual",
    "somatosensory": r"somatosens|tactile|\btouch\b|propriocep|\bbody\b|bodily|sensorimotor",
    "oculomotor": r"\beyes?\b|saccad|oculomotor|\bgaze\b|fixation",
    "memory": r"memory|memories|encoding|retrieval|\brecall\b|recollection|episodic|autobiographical|familiarity|consolidation|retention|mnemonic",
    "working memory": r"working memory|maintenance|updating|rehearsal",
    "attention": r"attention|orienting|\bsearch\b|salience|alertness|vigilance|\bfocus\b|distraction|detection",
    "control": r"cognitive control|executive|inhibit|conflict|\berrors?\b|monitoring|switching|flexibility|interference|\brules?\b|planning|response selection|task difficulty|demand|\bgoal",
    "reward": r"reward|\bvalue\b|valuation|monetary|incentive|decision|\brisk|utility|reinforcement|motivation",
    "emotion": r"emotion|affective|\baffect\b|\bfear\b|anxiety|\bmood\b|valence|\bstress\b",
    "social": r"social|mentaliz|theory (of )?mind|empathy|self[- ]referential|\bself\b|belief|intention|attribution|moral",
    "pain": r"\bpain|nocicept|noxious",
    "interoception": r"interocept|autonomic|homeosta|visceral|arousal",
    "chemosensory": r"\btaste|gustat|olfact|\bodou?rs?\b|\bsmell|\bfood\b|eating",
    "spatial": r"spatial|navigation|localization|mental rotation",
    "number": r"numer|\bnumbers?\b|arithmetic|calculation|\bmath",
    "imagery": r"imagery",
    "learning": r"learning|\bhabit|procedural|\bskills?\b|\bsequenc",
    "timing": r"timing|rhythm",
    "multisensory": r"multisensory|crossmodal|audiovisual",
    "vestibular": r"vestibular|balance|self-motion",
    "awareness": r"awareness|conscious",
    "internal": r"mind[- ]wandering|self[- ]generated|internally|prospective|future thinking|mentation",
}
FUNCTION_RE = {g: re.compile(p) for g, p in FUNCTION_GROUPS.items()}
DOMAIN_GENERAL = {"attention", "control", "working memory", "multisensory", "awareness", "internal", "learning", "timing"}


def groups_in(text: str, *, strip: bool = True) -> set[str]:
    t = strip_regions(text) if strip else str(text).lower()
    return {g for g, rx in FUNCTION_RE.items() if rx.search(t)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=M.DEFAULT_MODEL_DIR, help="Training run with the final model.")
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--n-pubmed", type=int, default=150, help="PubMed test maps sampled.")
    p.add_argument("--num-beams", type=int, default=3, help="Beams for caption generation.")
    p.add_argument("--max-new-tokens", type=int, default=160, help="Max caption length (tokens).")
    p.add_argument("--batch-size", type=int, default=64, help="Maps per generate call.")
    p.add_argument("--n-perm", type=int, default=5000, help="Permutations for chance.")
    p.add_argument("--neurosynth-dir", type=Path, default=C.default_neurosynth_dir(), help="Neurosynth v7 data (nimare layout).")
    p.add_argument("--diagnose", action="store_true", help="Also run the prefix / bank check on the ICA maps (GPU).")
    p.add_argument("--force", action="store_true", help="Regenerate cached captions.")
    p.add_argument("--strict-containment", action="store_true",
                   help="[REGION] rule without the containment clause (a region counts only by its 5 %% share or the peak).")
    p.add_argument("--title-constraint", choices=("none", "vocab", "shortlist"), default="vocab",
                   help="Decoding of the [NETWORK] / [REGION] titles (shared implementation in _hf_pins.py, as "
                        "scripts/models/_common_models.py): vocab = only the stage-14 bank titles of that level (the "
                        "default of every caption); none = free titles; shortlist = [REGION] titles only from the "
                        "--shortlist-k region bank rows whose NeuroAdapter maps overlap the map most (needs "
                        "build_title_vocabulary.py). The body is never constrained. Default: vocab (outputs/; the other "
                        "modes write outputs/title_<mode>/).")
    p.add_argument("--shortlist-k", type=int, default=50, help="Region bank rows kept by --title-constraint shortlist.")
    p.add_argument("--cache-dir", type=Path, default=HERE / "outputs" / "cache", help="Atlas, Neurosynth and caption caches.")
    args = p.parse_args()
    if args.title_constraint != "vocab" and args.out == HERE / "outputs":
        args.out = HERE / "outputs" / f"title_{args.title_constraint}"
    return args


def atlas_maps(ica: bool, cache_dir: Path) -> dict:
    """Network or ICA atlas maps on the mask grid (binary maps as-is, continuous maps binarised at the 95th percentile)."""
    cache = cache_dir / f"{'ica' if ica else 'networks'}_maps.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import nibabel as nib
    from nilearn.image import resample_img
    from _hf_pins import load_dataset

    masker = C.get_masker()
    nets = load_dataset("networks")
    ids, raws, X = [], [], []
    for atlas in nets:
        if (atlas in ICA_ATLASES) != ica:
            continue
        for raw, payload in nets[atlas].items():
            arr = np.asarray(payload["array"], np.float32)
            if arr.ndim == 4:
                arr = arr[..., 0]
            img = nib.Nifti1Image(arr, affine=payload["affine"])
            if len(np.unique(arr)) == 2:
                res = resample_img(img, masker.affine_, interpolation="nearest", force_resample=True, copy_header=True)
            else:
                res = resample_img(img, masker.affine_, force_resample=True, copy_header=True)
                a = res.get_fdata()
                a[a < 0] = 0.0
                thr = np.percentile(a.flatten(), 95)
                res = nib.Nifti1Image((a >= thr).astype(np.float32), affine=masker.affine_)
            X.append(np.asarray(masker.transform(res), np.float32).reshape(-1))
            ids.append(f"{atlas}:{raw}")
            raws.append(raw)
    out = {"ids": np.array(ids), "raw": np.array(raws), "X": np.stack(X)}
    np.savez_compressed(cache, **out)
    return out


def load_sets(args) -> dict[str, dict]:
    """{set: {ids, labels, X, binary, curated}}; PubMed maps are a seeded sample of the run's test split."""
    import torch
    from _hf_pins import load_dataset

    cache = args.cache_dir
    out = {}
    net = atlas_maps(False, cache)
    lab = load_dataset("network_test_set_labels").drop_duplicates("raw_network_label").set_index("raw_network_label")
    keys = [str(lab.loc[r, "network_key"]) if r in lab.index else "unknown" for r in net["raw"]]
    names = [str(lab.loc[r, "network_name"]) if r in lab.index else "Unknown" for r in net["raw"]]
    out["networks"] = {"ids": [str(i) for i in net["ids"]], "X": net["X"].astype(np.float32), "binary": True, "curated": keys,
                       "labels": [f"{i} ({n})" for i, n in zip(net["ids"], names)]}
    ica = atlas_maps(True, cache)
    out["ica"] = {"ids": [str(i) for i in ica["ids"]], "X": ica["X"].astype(np.float32), "binary": True,
                  "curated": ["unknown"] * len(ica["ids"]), "labels": [str(i) for i in ica["ids"]]}
    split_path = M.run_file("pmids_split")
    if split_path is not None:
        test = np.asarray(torch.load(split_path, map_location="cpu", weights_only=False)["test"])
    else:  # split flags of the pinned HF publications.parquet, never the data-dir copy
        from _hf_pins import hf_split_pmids

        test, split_path = hf_split_pmids()["test"], "hf:neurovlm/neuro_image_papers/publications.parquet (split flags)"
    test = np.sort(test.astype(np.int64))
    keep = C.pubmed_keep_mask(test, enabled=True, label="PubMed test", log=log.info)
    pool = test[keep]
    pm = np.sort(np.random.default_rng(args.seed).choice(pool, size=min(args.n_pubmed, len(pool)), replace=False))
    titles = C.load_pubmed_text_rows(pm)["name"].tolist()
    out["pubmed"] = {"ids": [str(p) for p in pm], "X": C.load_pubmed_maps(pm).numpy().astype(np.float32), "binary": False,
                     "curated": ["unknown"] * len(pm), "labels": [f"PMID {p}: {t[:70]}" for p, t in zip(pm, titles)],
                     "split": str(split_path), "pool": int(len(pool)), "excluded": int((~keep).sum())}
    return out


def supra_and_peak(X: np.ndarray, binary: bool) -> tuple[np.ndarray, np.ndarray]:
    S = X > 0 if binary else np.stack([C.top_pct(x, 90.0) for x in X])
    peak = np.full(len(X), -1) if binary else X.argmax(1)
    return S, peak


def region_parts(cache_dir: Path) -> dict[str, np.ndarray]:
    """{code: bool mask [V]} for HO cortical/subcortical, AAL cerebellum / vermis, Pauli hypothalamus / STN (4 mm grid)."""
    path = cache_dir / "atlas_region_parts.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from nilearn import datasets

    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", verbose=0)
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0)
    clab = [str(x) for x in cort.labels]
    out = {k: C._label_vectors(cort.maps, [clab.index(v)])[0] for k, v in HO_CORT.items()}
    slab = [str(x) for x in sub.labels]
    for k, v in HO_SUB.items():
        idx = [i for i, n in enumerate(slab) if n.endswith(v)]
        out[k] = C._label_vectors(sub.maps, idx).any(0)
    aal = datasets.fetch_atlas_aal("SPM12", verbose=0)
    val = dict(zip([str(x) for x in aal.labels], [int(i) for i in aal.indices]))
    out["CB"] = C._label_vectors(aal.maps, [v for n, v in val.items() if n.startswith(("Cerebelum", "Vermis"))]).any(0)
    out["VER"] = C._label_vectors(aal.maps, [v for n, v in val.items() if n.startswith("Vermis")]).any(0)
    pauli = datasets.fetch_atlas_pauli_2017("deterministic", verbose=0)
    for code in ("HTH", "STH"):
        out[code] = C._label_vectors(pauli.maps, [[str(x) for x in pauli.labels].index(code)])[0]
    np.savez_compressed(path, **out)
    return out


class Decoder:
    """Top-5 Neurosynth v7 terms of a map (Pearson r with each term's association z-map)."""

    def __init__(self, args):
        ns = C.Neurosynth(args.neurosynth_dir, args.cache_dir)
        cog = pd.read_parquet(C.data_dir() / "cogatlas.parquet")
        cog_terms = sorted({str(t).strip().lower() for t in cog["term"].dropna()})
        terms = [t for t in cog_terms if ns.has(t) and ns.count(t) >= 50]
        terms += [t for t in CORE_TERMS if ns.has(t) and ns.count(t) >= 50 and t not in terms]
        self.terms, self.n_cogatlas = terms, sum(t in cog_terms for t in terms)
        self.Z = np.stack([ns.get(t) for t in terms]).astype(np.float32)
        self.source = ns.source

    def top5(self, X: np.ndarray) -> list[list[str]]:
        r = C.pearson_matrix(X, self.Z)
        return [[self.terms[j] for j in np.argsort(-row)[:5]] for row in r]


def model_tag(args) -> str:
    files = [args.model_dir / f for f in ("neuro_qformer_canonical/model.safetensors", "neuro_qformer_canonical/config.json",
                                          "canonical_semantic_banks.pt", "autoencoder.pt")]
    return C.sha_key(*[(f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in files if f.exists()])


def title_tries(args, d: dict, tok: str, tokenizer) -> list:
    """Per map: the trie of names its [NETWORK] / [REGION] title may take (None = free title)."""
    import _hf_pins as T

    if args.title_constraint == "none" or tok == "function":
        return [None] * len(d["ids"])
    vocab = T.read_title_vocabulary(VOCAB_FILE)
    if args.title_constraint == "vocab" or tok == "network":
        return [T.title_trie(tokenizer, vocab["titles"][tok])] * len(d["ids"])
    # shortlist: titles of the k region bank rows whose NeuroAdapter map lies most inside the map's top voxels
    z = np.load(args.cache_dir / "bank_maps.npz")
    P = np.unpackbits(z["region"], axis=1)[:, : int(z["region_n_voxels"])].astype(np.float32)
    S, _ = supra_and_peak(d["X"], d["binary"])
    share = (S.astype(np.float32) @ P.T) / P.sum(1)
    rows = np.array(vocab["bank_rows"]["region"])
    return [T.title_trie(tokenizer, sorted(set(rows[np.argsort(-r, kind="stable")[: args.shortlist_k]]))) for r in share]


def generate(args, sets: dict, device, timings) -> tuple[pd.DataFrame, dict]:
    mode = [] if args.title_constraint == "none" else [args.title_constraint, args.shortlist_k]  # free titles keep their first cache key
    tag = C.sha_key(model_tag(args), args.num_beams, args.max_new_tokens, *[",".join(d["ids"]) for d in sets.values()], *mode)
    path = args.cache_dir / f"final_generations_{tag}.parquet"
    if path.exists() and not args.force:
        log.info("cached final-run captions %s", path.name)
        meta = C.read_json(path.with_suffix(".json")) if path.with_suffix(".json").exists() else {}
        return pd.read_parquet(path), meta
    import torch
    import _hf_pins as T

    nvlm = M.make_nvlm(device, M.component("autoencoder", device), M.load_projector(device))
    lm, tokenizer = nvlm._neuro_qwen_model, nvlm._neuro_qwen_tokenizer
    rows, kw = [], dict(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, do_sample=False, seed=123)
    for name, d in sets.items():
        X = torch.from_numpy(d["X"])
        for tok in TOKENS:
            tries = title_tries(args, d, tok, tokenizer)
            with timings.time(f"final_generate_{name}_{tok}", n=len(X), batch=args.batch_size, title_constraint=args.title_constraint, **kw):
                outs = []
                for s in range(0, len(X), args.batch_size):
                    batch = tries[s:s + args.batch_size]
                    if all(t is None for t in batch):  # free titles (pin_neuroqwen's default would constrain them)
                        with T.free_titles(lm):
                            outs += M.nvlm_generate(nvlm, X[s:s + args.batch_size], basis=tok, **kw)
                        continue
                    with T.constrained_titles(lm, T.TitleConstraint(tokenizer, batch, args.num_beams)):
                        outs += M.nvlm_generate(nvlm, X[s:s + args.batch_size], basis=tok, **kw)
                log.info("%s/%s: %d captions", name, tok, len(outs))
            rows += [{"set": name, "id": d["ids"][i], "token": tok, "raw_text": t} for i, t in enumerate(outs)]
    meta = {"model_sources": M.model_sources(), "model_dir": M._rel(args.model_dir), "generation": kw,
            "title_constraint": args.title_constraint, "shortlist_k": args.shortlist_k if args.title_constraint == "shortlist" else None}
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    C.write_json(path.with_suffix(".json"), meta)
    return df, meta


def split_caption(raw: str) -> tuple[str, str, str]:
    """(title, first sentence of the body, body) of a ``[TOKEN]Title\\nbody`` caption."""
    text = strip_steering_tags(str(raw)).strip()
    title, _, body = text.partition("\n")
    if not body and len(title) > 80:  # no line break: title = first sentence
        title, _, body = title.partition(". ")
    first = re.split(r"(?<=[.!?])\s+", body.strip(), maxsplit=1)[0] if body.strip() else ""
    return title.strip(), first.strip(), body.strip()


def n_regions_caption(raw: str) -> int:
    """Number of regions named anywhere in a caption (title + body)."""
    title, _, body = split_caption(raw)
    return len(regions_in(title + " " + body))


def hit_matrices(d: dict, caps: pd.DataFrame, refs: dict, parts: dict, min_inside: float = MIN_INSIDE) -> dict[str, dict]:
    """Per token: H[i, j] = caption j judged against map i (diagonal = real pairs), plus per-caption details.
    ``min_inside``: a named region also counts when this share of its own voxels lies in the top voxels (0 = off)."""
    n = len(d["ids"])
    S, peak = refs["supra"], refs["peak"]
    nsup = np.maximum(S.sum(1), 1)
    region_masks = np.stack([np.any([parts[p] for p in pp], axis=0) for _, _, pp in REGION_RE])  # [R, V]
    overlap = S.astype(np.float32) @ region_masks.T.astype(np.float32)  # [n, R] top voxels in each region
    frac, inside = overlap / nsup[:, None], overlap / region_masks.sum(1)  # share of the top voxels / of the region
    at_peak = np.stack([region_masks[:, p] if p >= 0 else np.zeros(len(REGIONS), bool) for p in peak])  # [n, R]
    ok_region = (frac >= 0.05) | at_peak
    if min_inside > 0:
        ok_region |= inside >= min_inside
    out = {}
    for tok in TOKENS:
        g = caps[caps["token"] == tok].set_index("id").loc[d["ids"]]
        parsed = [split_caption(t) for t in g["raw_text"]]
        titles = [p[0] for p in parsed]
        H = np.zeros((n, n), bool)
        info = {"title": titles, "raw_text": g["raw_text"].tolist()}
        if tok == "region":
            named, src = [], []
            for title, first, _ in parsed:
                r = regions_in(title)
                named.append(r if r else regions_in(first))
                src.append("title" if r else ("first sentence" if named[-1] else "none"))
            for j, r in enumerate(named):
                if r:
                    H[:, j] = ok_region[:, r].any(1)
            info |= {"named": [[REGION_NAMES[k] for k in r] for r in named], "named_from": src,
                     "n_regions_title": [len(regions_in(t)) for t in titles],
                     "n_regions_caption": [n_regions_caption(t) for t in g["raw_text"]],
                     "best_frac": [float(frac[i, r].max()) if r else 0.0 for i, r in enumerate(named)],
                     "best_inside": [float(inside[i, r].max()) if r else 0.0 for i, r in enumerate(named)]}
        elif tok == "network":
            named = [networks_in(t) for t in titles]
            for j, nm in enumerate(named):
                H[:, j] = [ref in nm for ref in refs["network_ref"]]
            info["named"] = [sorted(x) for x in named]
        else:
            named = [groups_in(t) for t in titles]
            for j, gr in enumerate(named):
                H[:, j] = [bool(gr & mg) for mg in refs["function_groups"]]
            info["named"] = [sorted(x) for x in named]
        info["hit"] = np.diag(H).copy()
        info["H"] = H
        out[tok] = info
    return out


def map_references(d: dict, yeo: dict, decoder: Decoder, parts: dict) -> dict:
    S, peak = supra_and_peak(d["X"], d["binary"])
    nsup = np.maximum(S.sum(1), 1)
    yeo_frac = (S.astype(np.float32) @ yeo["masks"].T.astype(np.float32)) / nsup[:, None]
    yeo_ref = [str(yeo["names"][k]) for k in yeo_frac.argmax(1)]
    net_ref = [c if c != "unknown" else y for c, y in zip(d["curated"], yeo_ref)]
    top5 = decoder.top5(d["X"])
    ho_codes = list(HO_CORT) + list(HO_SUB)
    ho_masks = np.stack([parts[c] for c in ho_codes])
    ho_frac = (S.astype(np.float32) @ ho_masks.T.astype(np.float32)) / nsup[:, None]
    top_ho = [[{**HO_CORT, **HO_SUB}[ho_codes[k]] for k in np.argsort(-row)[:2]] for row in ho_frac]
    return {"supra": S, "peak": peak, "network_ref": net_ref, "yeo_ref": yeo_ref, "top5": top5,
            "function_groups": [set().union(*[groups_in(t, strip=False) for t in t5]) for t5 in top5], "top_ho": top_ho}


def chance_and_p(H: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    n = len(H)
    off = ~np.eye(n, dtype=bool)
    chance = float(H[off].mean())
    obs = float(np.diag(H).mean())
    perm = np.array([H[np.arange(n), rng.permutation(n)].mean() for _ in range(n_perm)])
    return chance, float((1 + np.sum(perm >= obs)) / (1 + n_perm))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def verdict(tok: str, info: dict, refs: dict, i: int) -> str:
    hit = "hit" if info["hit"][i] else "miss"
    if tok == "region":
        nm = ", ".join(info["named"][i]) or "no region named"
        src = " (named in the first sentence)" if info["named_from"][i] == "first sentence" else ""
        return (f"{hit}: {nm}{src} holds {100 * info['best_frac'][i]:.1f} % of the map's top voxels, "
                f"{100 * info['best_inside'][i]:.0f} % of it inside them (largest HO: {', '.join(refs['top_ho'][i])})")
    if tok == "network":
        return f"{hit}: reference = {refs['network_ref'][i].replace('_', ' ')}"
    return f"{hit}: top-5 Neurosynth = {', '.join(refs['top5'][i])}"


def diagnose(args, sets, device) -> dict:
    """ICA maps: does the canonical bank or the text prefix decide whether titles name a region or a network?"""
    import torch

    ae = M.component("autoencoder", device)
    proj = M.load_projector(device)
    nvlm = M.make_nvlm(device, ae, proj)
    X = torch.from_numpy(sets["ica"]["X"])
    kw = dict(num_beams=args.num_beams, max_new_tokens=48, do_sample=False, seed=123)
    out = {"bank_sizes": {b: int(proj.canonical_projection.get_bank(b).shape[0]) for b in ("all", "network", "region", "function")}}
    with torch.no_grad():
        lat = ae.encoder(X.to(device)).float()
        for b in ("region", "network", "function"):
            _, w = proj.project_semantic(lat, basis=b, return_weights=True)
            out[f"effective_bank_entries_{b}"] = float(np.median(1.0 / (w.float() ** 2).sum(1).cpu().numpy()))
    import _hf_pins as T

    rows = []
    for bank in ("region", "network"):
        for prefix in ("[REGION]", "[NETWORK]", ""):
            with T.free_titles(nvlm._neuro_qwen_model):  # the model's own choice of title
                outs = M.nvlm_generate(nvlm, X, basis=bank, prefix_text=prefix, **kw)
            titles = [split_caption(t[len(prefix):])[0] for t in outs]
            rows.append({"bank": bank, "prefix": prefix or "(none)", "n": len(titles),
                         "titles_naming_a_region": float(np.mean([bool(regions_in(t)) and "network" not in t.lower() for t in titles])),
                         "titles_naming_a_network": float(np.mean(["network" in t.lower() for t in titles])),
                         "example_titles": titles[:3]})
    tokzr = nvlm._neuro_qwen_tokenizer
    out["prefix_token_ids"] = {p: tokzr(p, add_special_tokens=False)["input_ids"] for p in ("[REGION]", "[NETWORK]", "[FUNCTION]")}
    out["grid"] = rows
    return out


def canonical_share(titles: list[str], tok: str) -> float | None:
    """Share of titles that are a stage-14 bank title of the token's level (case and spacing ignored)."""
    if tok == "function" or not VOCAB_FILE.exists():
        return None
    key = lambda t: " ".join(str(t).lower().split())
    names = {key(t) for t in C.read_json(VOCAB_FILE)["titles"][tok]}
    return float(np.mean([key(t) in names for t in titles]))


def score_sets(sets, caps, yeo, decoder, parts, args, rng):
    rows, items, details = [], [], {}
    for name, d in sets.items():
        refs = map_references(d, yeo, decoder, parts)
        info = hit_matrices(d, caps[caps["set"] == name], refs, parts, args.min_inside)
        details[name] = (refs, info)
        for tok in TOKENS:
            I = info[tok]
            n, k = len(I["hit"]), int(I["hit"].sum())
            chance, p = chance_and_p(I["H"], args.n_perm, rng)
            lo, hi = wilson(k, n)
            row = {"token": f"[{tok.upper()}]", "set": name, "n": n, "hit_rate": k / n, "ci_lo": lo, "ci_hi": hi,
                   "chance": chance, "p_perm": p, "distinct_titles": len(set(I["title"])), "canonical_titles": canonical_share(I["title"], tok)}
            if tok == "region":
                row |= {"regions_in_title": float(np.mean(I["n_regions_title"])),
                        "titles_one_region": float(np.mean(np.array(I["n_regions_title"]) == 1)),
                        "regions_in_caption": float(np.mean(I["n_regions_caption"]))}
            rows.append(row)
            for i in range(n):
                items.append({"set": name, "id": d["ids"][i], "label": d["labels"][i], "token": tok, "title": I["title"][i],
                              "named": "; ".join(I["named"][i]), "hit": bool(I["hit"][i]), "verdict": verdict(tok, I, refs, i),
                              "raw_text": I["raw_text"][i]})
    return pd.DataFrame(rows), pd.DataFrame(items), details


def main() -> None:
    global VOCAB_FILE
    args = parse_args()
    VOCAB_FILE = Path(args.model_dir) / VOCAB_FILE.name
    args.min_inside = 0.0 if args.strict_containment else MIN_INSIDE
    C.set_offline_env()
    C.seed_all(args.seed)
    t0 = time.time()
    device = C.select_device(args.device)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    M.set_model_dir(args.model_dir)
    timings = C.Timings(args.out / "correctness_timings.json")
    sets = load_sets(args)
    log.info("sets: %s", {k: len(v["ids"]) for k, v in sets.items()})
    caps, meta = generate(args, sets, device, timings)
    with timings.time("final_score"):
        parts = region_parts(args.cache_dir)
        yeo = C.yeo7(args.cache_dir)
        decoder = Decoder(args)
        rng = np.random.default_rng(args.seed)
        table, items, details = score_sets(sets, caps, yeo, decoder, parts, args, rng)
    items.to_csv(args.out / "correctness_per_caption.csv", index=False)
    set_names = {"networks": "Network-atlas maps", "ica": "ICA components", "pubmed": "PubMed test maps"}
    tex = pd.DataFrame({"Token": table["token"], "Map set": table["set"].map(set_names), "n": table["n"].astype(str),
                        "Correct (%)": [f"{100 * v:.0f}" for v in table["hit_rate"]],
                        "95 % CI": [f"{100 * a:.0f}-{100 * b:.0f}" for a, b in zip(table["ci_lo"], table["ci_hi"])],
                        "Chance (%)": [f"{100 * v:.0f}" for v in table["chance"]],
                        "p": ["$<$ 0.001" if v < 0.001 else f"{v:.3f}" for v in table["p_perm"]]})
    C.write_tex(tex, args.out / "correctness_table.tex",
                caption="Steering-token correctness (final model): share of captions whose title names a region / network / "
                        "function that matches the map; chance = captions shuffled across maps."
                        + ("" if args.strict_containment else " A region counts when it holds 5 \\% of the map's top voxels "
                           "or its peak, or when at least half of it lies in the top voxels."), label="tab:steering_correctness")
    table.to_csv(args.out / "correctness_table.csv", index=False, float_format="%.3f")

    # the same test on the released model's captions (removed released-model analysis)
    old = {}
    if OLD_CACHE.exists():
        oc = pd.read_parquet(OLD_CACHE).rename(columns={"dataset": "set"})
        for name, d in sets.items():
            if name == "pubmed":  # that analysis used the first 150 test maps
                ids = oc.loc[oc["set"] == "pubmed", "id"].drop_duplicates().tolist()
                dd = {"ids": ids, "X": C.load_pubmed_maps(np.array([int(i) for i in ids])).numpy(), "binary": False,
                      "curated": ["unknown"] * len(ids)}
            else:
                ids = [i for i in d["ids"] if i in set(oc.loc[oc["set"] == name, "id"])]
                pos = {i: k for k, i in enumerate(d["ids"])}
                dd = {"ids": ids, "X": d["X"][[pos[i] for i in ids]], "binary": True,
                      "curated": [d["curated"][pos[i]] for i in ids]}
            refs = map_references(dd, yeo, decoder, parts)
            info = hit_matrices(dd, oc[oc["set"] == name], refs, parts, args.min_inside)
            old[name] = {tok: {"n": len(ids), "hit_rate": float(info[tok]["hit"].mean()),
                               "chance": float(info[tok]["H"][~np.eye(len(ids), dtype=bool)].mean())} for tok in TOKENS}
            old[name]["regions_in_caption"] = {tok: float(np.mean([n_regions_caption(t)
                                                                    for t in oc[(oc["set"] == name) & (oc["token"] == tok)]["raw_text"]]))
                                               for tok in TOKENS}
    # regions named per caption (why similarity to a region list favours [NETWORK])
    reg_caption = {name: {tok: float(np.mean([n_regions_caption(t) for t in caps[(caps["set"] == name) & (caps["token"] == tok)]["raw_text"]]))
                          for tok in TOKENS} for name in sets}

    # [FUNCTION] misses: title names a region / a network / nothing, only domain-general functions, or the wrong domain
    fmode = {}
    for name in sets:
        I = details[name][1]["function"]
        miss = ~I["hit"]
        named = [set(x) for x in I["named"]]
        region_t = np.array([not g and bool(regions_in(t)) for g, t in zip(named, I["title"])])
        network_t = np.array([not g and "network" in t.lower() for g, t in zip(named, I["title"])])
        other_t = np.array([not g for g in named]) & ~region_t & ~network_t
        general = np.array([bool(g) and g <= DOMAIN_GENERAL for g in named])
        vc = pd.Series(I["title"]).value_counts()
        fmode[name] = {"n": int(len(miss)), "misses": int(miss.sum()), "miss_title_is_a_region": int((miss & region_t).sum()),
                       "miss_title_is_a_network": int((miss & network_t).sum()), "miss_title_names_no_function": int((miss & other_t).sum()),
                       "miss_domain_general_only": int((miss & general).sum()),
                       "miss_specific_wrong_domain": int((miss & ~region_t & ~network_t & ~other_t & ~general).sum()),
                       "distinct_titles": int(len(vc)), "top_titles": {str(k): int(v) for k, v in vc.head(5).items()}}
    # [REGION] misses: adjacent (the region mask dilated by one 4 mm voxel holds >= 5 %) or farther away
    from scipy.ndimage import binary_dilation

    mimg = np.asarray(C.get_masker().mask_img_.dataobj) > 0

    def dilate(v):
        vol = np.zeros(mimg.shape, bool)
        vol[mimg] = v
        return binary_dilation(vol, iterations=1)[mimg]

    rdil = np.stack([dilate(np.any([parts[p] for p in pp], axis=0)) for _, _, pp in REGION_RE])
    rmode, miss_rows = {}, []
    for name, d in sets.items():
        refs, info = details[name]
        I = info["region"]
        S = refs["supra"]
        near = (S.astype(np.float32) @ rdil.T.astype(np.float32)) / np.maximum(S.sum(1), 1)[:, None]
        cls = []
        for i, nm in enumerate(I["named"]):
            ks = [REGION_NAMES.index(x) for x in nm]
            cls.append("hit" if I["hit"][i] else "no region named" if not ks else
                       "adjacent (<= 4 mm)" if near[i, ks].max() >= 0.05 else "farther than 4 mm")
            if not I["hit"][i]:
                miss_rows.append({"set": name, "label": d["labels"][i], "title": I["title"][i], "class": cls[-1],
                                  "verdict": verdict("region", I, refs, i)})
        c = pd.Series(cls)
        rmode[name] = {k: int((c == k).sum()) for k in ("hit", "adjacent (<= 4 mm)", "farther than 4 mm", "no region named")}
    diag = diagnose(args, sets, device) if args.diagnose else None

    # examples: 10 per token (4 network, 3 ICA, 3 PubMed), seeded
    ex_rng = np.random.default_rng(args.seed)
    lines = ["# Example captions (final model): 10 per steering token", "",
             "Seeded random draw per token (4 network-atlas maps, 3 ICA components, 3 PubMed test maps). Verdict as in "
             "`run_correctness.py`.", ""]
    for tok in TOKENS:
        lines += [f"## [{tok.upper()}]", "", "| Map | Title | Verdict |", "|---|---|---|"]
        for name, k in (("networks", 4), ("ica", 3), ("pubmed", 3)):
            g = items[(items["set"] == name) & (items["token"] == tok)]
            for _, r in g.iloc[np.sort(ex_rng.choice(len(g), k, replace=False))].iterrows():
                lines.append(f"| {r['label'].replace('|', '/')} | {r['title'].replace('|', '/')} | {r['verdict'].replace('|', '/')} |")
        lines.append("")
    mr = pd.DataFrame(miss_rows)
    pick = mr.iloc[np.sort(np.random.default_rng(args.seed + 1).choice(len(mr), min(20, len(mr)), replace=False))]
    lines += ["## 20 [REGION] misses (seeded draw, inspected for the README)", "",
              "Adjacent = the named region's atlas mask, dilated by one 4 mm voxel, holds >= 5 % of the map's top voxels.", "",
              "| Map | Title | Class | Verdict |", "|---|---|---|---|"]
    lines += [f"| {r['label'].replace('|', '/')} | {r['title'].replace('|', '/')} | {r['class']} | {r['verdict'].replace('|', '/')} |"
              for _, r in pick.iterrows()]
    (args.out / "examples.md").write_text("\n".join(lines) + "\n")

    res = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"device": str(device)},
           "model": meta, "n_maps": {k: len(v["ids"]) for k, v in sets.items()},
           "pubmed_sample": {k: sets["pubmed"][k] for k in ("split", "pool", "excluded")},
           "decoding_vocabulary": {"n_terms": len(decoder.terms), "n_cognitive_atlas": decoder.n_cogatlas, "source": decoder.source},
           "table": table.to_dict(orient="records"), "regions_named_per_caption": reg_caption,
           "old_released_model_captions": old, "function_failure_modes": fmode, "region_miss_modes": rmode, "diagnostics": diag,
           "runtime_seconds": round(time.time() - t0, 1), "versions": C.installed_versions()}
    C.write_json(args.out / "correctness_results.json", res)
    log.info("\n%s", table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log.info("done in %.0f s -> %s", time.time() - t0, args.out)


if __name__ == "__main__":
    main()
