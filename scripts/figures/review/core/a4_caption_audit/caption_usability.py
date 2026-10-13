#!/usr/bin/env python
"""Are coordinate-table captions usable training text? (comment 3.7)

Labels each table's own coordinates with Harvard-Oxford, scores every caption with rule-based flags and tests which text
(caption or title + abstract) predicts the table's dominant region.  Reads run_analysis.py's outputs; CPU, ~2 min:
    .venv/bin/python scripts/figures/review/core/a4_caption_audit/caption_usability.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # single-threaded: BLAS spin-waiting slows sklearn here
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE))
from run_analysis import STAT_EXPR, SW, TABLE_GENERIC, strip_label  # noqa: E402

SEED = 0
N_REGIONS = 20      # decodability: the most frequent dominant regions kept
MIN_LABELLED = 5    # decodability: labelled coordinates a table needs for a dominant region
MIN_IN_REGION = 5   # sensitivity target: coordinates required inside the dominant region itself
N_BOOT = 2000
SPLITS = ["all", "train", "val", "test", "unassigned"]
CATS = ["contrast/condition", "region list", "boilerplate", "other"]

# atlas vocabulary
# Harvard-Oxford label -> (case-insensitive terms, case-sensitive abbreviations) = the "fine" match
SHORT = {"Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "Supplementary Motor Cortex",
         "Heschl's Gyrus (includes H1 and H2)": "Heschl's Gyrus"}
IFG = ["inferior frontal", "broca'?s?", "ventrolateral prefrontal", "inferior frontal junction"]
FUSI = ["fusiform", "visual word form", "occipito-?temporal", "temporo-?occipital"]
V1 = ["calcarine", "primary visual", "striate", "visual cortex"]
STR = ["striatum", "striatal"]
FINE: dict[str, tuple[list[str], list[str]]] = {
    "Frontal Pole": (["frontal pole", "fronto-?polar", "rostrolateral prefrontal", "anterior prefrontal", "rostral prefrontal"], ["RLPFC", "rlPFC", "aPFC", "FPC"]),
    "Insular Cortex": (["insula[er]?", "insulae"], []),
    "Superior Frontal Gyrus": (["superior frontal", "dorsomedial prefrontal", "frontal eye fields?"], ["SFG", "dmPFC", "DMPFC", "FEF"]),
    "Middle Frontal Gyrus": (["middle frontal", "dorso-?lateral prefrontal", "frontal eye fields?"], ["MFG", "DLPFC", "dlPFC", "FEF"]),
    "Inferior Frontal Gyrus, pars triangularis": (IFG + ["pars triangularis", "triangular"], ["IFG", "IFJ", "VLPFC", "vlPFC"]),
    "Inferior Frontal Gyrus, pars opercularis": (IFG + ["pars opercularis"], ["IFG", "IFJ", "VLPFC", "vlPFC"]),
    "Precentral Gyrus": (["pre-?central", "primary motor", "motor cortex", "pre-?motor", "frontal eye fields?"], ["M1", "PMC", "PMd", "PMv", "dPMC", "vPMC", "FEF"]),
    "Temporal Pole": (["temporal pole", "temporopolar", "anterior temporal lobe"], ["ATL"]),
    "Superior Temporal Gyrus, anterior division": (["superior temporal"], ["STG", "STS", "aSTG", "aSTS"]),
    "Superior Temporal Gyrus, posterior division": (["superior temporal", "wernicke'?s?"], ["STG", "STS", "pSTG", "pSTS"]),
    "Middle Temporal Gyrus, anterior division": (["middle temporal"], ["MTG", "aMTG", "STS"]),
    "Middle Temporal Gyrus, posterior division": (["middle temporal"], ["MTG", "pMTG", "STS"]),
    "Middle Temporal Gyrus, temporooccipital part": (["middle temporal", "occipito-?temporal", "temporo-?occipital"], ["MTG", "MT", "hMT", "V5", "EBA"]),
    "Inferior Temporal Gyrus, anterior division": (["inferior temporal"], ["ITG", "ITS"]),
    "Inferior Temporal Gyrus, posterior division": (["inferior temporal"], ["ITG", "ITS"]),
    "Inferior Temporal Gyrus, temporooccipital part": (["inferior temporal", "occipito-?temporal", "temporo-?occipital"], ["ITG", "ITS"]),
    "Postcentral Gyrus": (["post-?central", "primary somatosensory", "somatosensory cortex"], ["S1", "SI"]),
    "Superior Parietal Lobule": (["superior parietal", "intraparietal"], ["SPL", "IPS"]),
    "Supramarginal Gyrus, anterior division": (["supramarginal", "inferior parietal", "intraparietal"], ["SMG", "IPL", "IPS"]),
    "Supramarginal Gyrus, posterior division": (["supramarginal", "inferior parietal", "temporo-?parietal"], ["SMG", "IPL", "TPJ"]),
    "Angular Gyrus": (["angular", "inferior parietal", "temporo-?parietal"], ["AG", "IPL", "TPJ"]),
    "Lateral Occipital Cortex, superior division": (["lateral occipital", "superior occipital", "middle occipital", "extrastriate"], ["LOC", "SOG", "MOG", "IPS"]),
    "Lateral Occipital Cortex, inferior division": (["lateral occipital", "inferior occipital", "middle occipital", "extrastriate", "occipito-?temporal"], ["LOC", "LO", "IOG", "MOG", "MT", "hMT", "V5", "EBA"]),
    "Intracalcarine Cortex": (V1 + ["intracalcarine"], ["V1", "V2"]),
    "Frontal Medial Cortex": (["medial frontal", "medial prefrontal", "ventromedial prefrontal", "medial orbitofrontal", "gyrus rectus", "rectal gyrus"], ["MPFC", "mPFC", "vmPFC", "VMPFC", "mOFC"]),
    "Supplementary Motor Cortex": (["supplementary motor", "medial frontal gyrus"], ["SMA", "preSMA"]),
    "Subcallosal Cortex": (["subcallosal", "subgenual", "ventromedial prefrontal"], ["sgACC", "vmPFC", "VMPFC"]),
    "Paracingulate Gyrus": (["paracingulate", "anterior cingulate", "medial prefrontal", "medial frontal", "dorsomedial prefrontal"], ["ACC", "dACC", "mPFC", "MPFC", "dmPFC", "DMPFC"]),
    "Cingulate Gyrus, anterior division": (["anterior cingulate", "cingulate", "mid-?cingulate"], ["ACC", "dACC", "rACC", "pgACC", "sgACC", "MCC", "aMCC"]),
    "Cingulate Gyrus, posterior division": (["posterior cingulate", "cingulate", "mid-?cingulate", "retrosplenial"], ["PCC", "RSC", "MCC", "pMCC"]),
    "Precuneous Cortex": (["precuneus", "precuneous", "precunei"], ["PCun", "PCu"]),
    "Cuneal Cortex": (["cuneus", "cuneal", "cunei"], []),
    "Frontal Orbital Cortex": (["orbito-?frontal", "orbital", "orbitalis"], ["OFC", "lOFC", "mOFC"]),
    "Parahippocampal Gyrus, anterior division": (["parahippocamp\\w*", "entorhinal", "perirhinal", "uncus", "medial temporal lobe"], ["PHG", "PHC", "MTL"]),
    "Parahippocampal Gyrus, posterior division": (["parahippocamp\\w*"], ["PHG", "PHC", "PPA", "MTL"]),
    "Lingual Gyrus": (["lingual"], []),
    "Temporal Fusiform Cortex, anterior division": (["fusiform"], ["FG", "FFA"]),
    "Temporal Fusiform Cortex, posterior division": (["fusiform", "visual word form"], ["FG", "FFA", "VWFA"]),
    "Temporal Occipital Fusiform Cortex": (FUSI, ["FG", "FFA", "VWFA", "OFA", "vOT"]),
    "Occipital Fusiform Gyrus": (["fusiform"], ["FG", "OFA"]),
    "Frontal Opercular Cortex": (["frontal operculum", "operculum", "opercular", "fronto-?insular"], ["FO"]),
    "Central Opercular Cortex": (["central operculum", "rolandic operculum", "operculum", "opercular", "secondary somatosensory"], ["S2", "SII"]),
    "Parietal Opercular Cortex": (["parietal operculum", "operculum", "opercular", "secondary somatosensory"], ["S2", "SII"]),
    "Planum Polare": (["planum polare", "superior temporal"], ["STG"]),
    "Heschl's Gyrus": (["heschl'?s?", "transverse temporal", "primary auditory", "auditory cortex"], ["A1", "HG"]),
    "Planum Temporale": (["planum temporale", "auditory cortex", "superior temporal"], ["PT", "STG"]),
    "Supracalcarine Cortex": (V1 + ["supracalcarine"], ["V1", "V2"]),
    "Occipital Pole": (V1 + ["occipital pole"], ["V1", "V2"]),
    "Thalamus": (["thalam\\w*", "pulvinar", "geniculate", "mediodorsal"], ["LGN", "MGN"]),
    "Caudate": (["caudate", "dorsal striatum"] + STR, []),
    "Putamen": (["putamen", "lentiform", "dorsal striatum"] + STR, []),
    "Pallidum": (["pallidum", "pallidus", "pallidal", "lentiform"], ["GP", "GPi", "GPe"]),
    "Brain-Stem": (["brain-?stem", "brain stem", "midbrain", "mesencephal\\w*", "pons", "pontine", "medulla", "periaqueductal",
                    "substantia nigra", "ventral tegmental", "tegment\\w*", "collicul\\w*", "raphe", "locus coeruleus", "red nucleus"],
                   ["PAG", "VTA", "SN", "LC"]),
    "Hippocampus": (["hippocamp\\w*", "dentate", "subiculum", "medial temporal lobe"], ["MTL", "CA1", "CA3"]),
    "Amygdala": (["amygdal\\w*"], ["BLA"]),
    "Accumbens": (["accumbens", "ventral striatum"] + STR, ["NAcc", "NAc"]),
}
# lobe / structure level ("coarse" match): group -> (terms, abbreviations); each label belongs to one or more groups
GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "frontal": (["frontal", "prefrontal", "pre-?central", "pre-?motor", "motor cortex", "supplementary motor", "orbito-?frontal", "orbital", "broca'?s?"],
                ["PFC", "DLPFC", "dlPFC", "VLPFC", "vlPFC", "DMPFC", "dmPFC", "VMPFC", "vmPFC", "MPFC", "mPFC", "OFC", "SFG", "MFG", "IFG", "IFJ", "FEF", "SMA", "preSMA", "M1", "PMC", "PMd", "PMv"]),
    "temporal": (["temporal", "temporo\\w*", "heschl'?s?", "planum", "auditory cortex", "fusiform", "wernicke'?s?"],
                 ["STG", "STS", "MTG", "ITG", "ITS", "ATL", "A1", "MT", "hMT", "V5", "FFA", "VWFA"]),
    "parietal": (["parietal", "supramarginal", "angular", "precune\\w*", "post-?central", "somatosensory"],
                 ["IPL", "SPL", "IPS", "SMG", "TPJ", "S1", "S2", "SII", "PCun"]),
    "occipital": (["occipital", "occipito\\w*", "calcarine", "visual cortex", "striate", "extrastriate", "cuneus", "cuneal", "lingual", "fusiform"],
                  ["V1", "V2", "V3", "V4", "V5", "MT", "hMT", "LOC", "LO", "MOG", "IOG", "SOG", "OFA"]),
    "cingulate": (["cingulate", "paracingulate", "subcallosal", "subgenual", "retrosplenial", "mid-?cingulate"],
                  ["ACC", "dACC", "rACC", "pgACC", "sgACC", "PCC", "MCC", "aMCC", "RSC"]),
    "insula": (["insula[er]?", "insulae"], []),
    "medial temporal": (["hippocamp\\w*", "parahippocamp\\w*", "amygdal\\w*", "entorhinal", "perirhinal", "medial temporal"], ["MTL", "PHG", "PPA"]),
    "basal ganglia": (["striatum", "striatal", "caudate", "putamen", "pallidum", "pallidus", "accumbens", "lentiform", "basal ganglia"], ["NAcc", "NAc", "GP"]),
    "thalamus": (["thalam\\w*", "pulvinar", "geniculate"], ["LGN", "MGN"]),
    "brainstem": (FINE["Brain-Stem"][0], FINE["Brain-Stem"][1]),
    "cerebellum": (["cerebell\\w*", "vermis", "crus"], []),  # not in Harvard-Oxford: counts for names_region only
}
LABEL_GROUPS = {
    "frontal": ["Frontal Pole", "Superior Frontal Gyrus", "Middle Frontal Gyrus", "Inferior Frontal Gyrus, pars triangularis",
                "Inferior Frontal Gyrus, pars opercularis", "Precentral Gyrus", "Frontal Medial Cortex", "Supplementary Motor Cortex",
                "Frontal Orbital Cortex", "Frontal Opercular Cortex", "Subcallosal Cortex", "Paracingulate Gyrus", "Central Opercular Cortex"],
    "temporal": ["Temporal Pole", "Superior Temporal Gyrus, anterior division", "Superior Temporal Gyrus, posterior division",
                 "Middle Temporal Gyrus, anterior division", "Middle Temporal Gyrus, posterior division", "Middle Temporal Gyrus, temporooccipital part",
                 "Inferior Temporal Gyrus, anterior division", "Inferior Temporal Gyrus, posterior division",
                 "Inferior Temporal Gyrus, temporooccipital part", "Planum Polare", "Heschl's Gyrus", "Planum Temporale",
                 "Temporal Fusiform Cortex, anterior division", "Temporal Fusiform Cortex, posterior division", "Temporal Occipital Fusiform Cortex"],
    "parietal": ["Postcentral Gyrus", "Superior Parietal Lobule", "Supramarginal Gyrus, anterior division", "Supramarginal Gyrus, posterior division",
                 "Angular Gyrus", "Parietal Opercular Cortex", "Precuneous Cortex", "Central Opercular Cortex"],
    "occipital": ["Lateral Occipital Cortex, superior division", "Lateral Occipital Cortex, inferior division", "Intracalcarine Cortex",
                  "Cuneal Cortex", "Lingual Gyrus", "Occipital Fusiform Gyrus", "Supracalcarine Cortex", "Occipital Pole",
                  "Temporal Occipital Fusiform Cortex", "Middle Temporal Gyrus, temporooccipital part", "Inferior Temporal Gyrus, temporooccipital part"],
    "cingulate": ["Cingulate Gyrus, anterior division", "Cingulate Gyrus, posterior division", "Paracingulate Gyrus", "Subcallosal Cortex"],
    "insula": ["Insular Cortex"],
    "medial temporal": ["Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division", "Hippocampus", "Amygdala"],
    "basal ganglia": ["Caudate", "Putamen", "Pallidum", "Accumbens"],
    "thalamus": ["Thalamus"],
    "brainstem": ["Brain-Stem"],
    "cerebellum": [],
}
GROUPS_OF: dict[str, set[str]] = {}
for _g, _labs in LABEL_GROUPS.items():
    for _l in _labs:
        GROUPS_OF.setdefault(_l, set()).add(_g)


def _rx(terms: list[str], abbrevs: list[str]) -> tuple[re.Pattern | None, re.Pattern | None]:
    """Whole-word regexes (no letter on either side): terms case-insensitive, abbreviations case-sensitive."""
    t = re.compile(r"(?<![A-Za-z])(?:" + "|".join(terms) + r")(?![A-Za-z])", re.I) if terms else None
    a = re.compile(r"(?<![A-Za-z])(?:" + "|".join(abbrevs) + r")(?![A-Za-z])") if abbrevs else None
    return t, a


FINE_RX = {k: _rx(*v) for k, v in FINE.items()}
GROUP_RX = {k: _rx(*v) for k, v in GROUPS.items()}


def _hits(text: str, rxs: dict[str, tuple]) -> set[str]:
    return {k for k, (t, a) in rxs.items() if (t is not None and t.search(text)) or (a is not None and a.search(text))}


# caption rules (label "Table N" and statistics stripped first):
#   has_contrast    a comparison / condition pattern or a curated task term, and >= 1 specific non-anatomical word
#   is_boilerplate  no specific word, or <= 3 content words and no contrast
#   region_match    names one of the top-3 atlas labels of its own coordinates ("fine"; lobe / structure level: "coarse")
#   task_match      shares >= 2 stemmed specific words with its own title + abstract
#   category        contrast/condition > region list > boilerplate > other (first match)
COMPARE_RE = re.compile(
    r"\bvs\.?(?=[\s,;:)]|$)|\bversus\b|\bv\.(?=\s)|[A-Za-z0-9)\]]\s*[<>]\s*[A-Za-z0-9(\[]|\bminus\b|\bcompar\w*|\bgreater\b|\bless\b"
    r"|\bhigher\b|\blower\b|\bstronger\b|\bweaker\b|\bmore\b|\brelative to\b|\bthan\b|\bdifferen\w*|\bdiffer\b|\bcontrasts?\b"
    r"|\binteractions?\b|\bmain effects?\b|\beffects? of\b|\bcorrelat\w*|\bassociat\w*|\brelated to\b|\b\w+-related\b|\bpredict\w*"
    r"|\bmodulat\w*|\bincreas\w*|\bdecreas\w*|\breduc\w*|\benhanc\w*|\bdeactivat\w*|\bconjunction\b|\bparametric\b|\bconnectivity\b"
    r"|\bPPI\b|\bduring\b|\bin response to\b|\belicited by\b|\bevoked by\b|\bactivations? (?:for|by|to)\b|\bresponses? (?:for|to)\b"
    r"|\balter\w*|\babnormal\w*|\batroph\w*|\bchang\w*|\bdegenerat\w*|\bimpair\w*|\bdeficit\w*|\brelationship\b|\bbetween\b",
    re.I)
# curated task / stimulus / process / population terms (specific: they say WHAT was measured)
TASK_TERMS = (
    "faces? facial words? pseudowords? letters? sentences? pictures? images? scenes? objects? tools? bodies body houses? places? "
    "tones? sounds? music\\w* speech voices? vocal\\w* odou?rs? olfactory gustatory tastes? food\\w* touch\\w* tactile vibrotactile "
    "pain\\w* nocicept\\w* heat thermal itch rewards? reward\\w* monetary money gains? loss(?:es)? punish\\w* incentive\\w* "
    "anticipat\\w* outcomes? feedback decisions? choices? risk\\w* gambl\\w* ambiguity uncertainty delay discount\\w* "
    "memory memories encoding retrieval recognition recall familiarity recollection n-back \\d-back working stroop "
    "go/no-?go no-?go nogo stop-signal stop flanker oddball inhibit\\w* conflict errors? attention\\w* attend\\w* cue\\w* "
    "targets? distract\\w* saccad\\w* fixation antisaccade motor movements? finger\\w* tapping grasp\\w* hand foot "
    "imagery imagin\\w* emotion\\w* affective fear\\w* anger angry happ\\w* sad\\w* disgust\\w* threat\\w* anxi\\w* "
    "arousal valence social theory-of-mind mentali[sz]\\w* empathy empathic self self-referential moral\\w* trust "
    "language linguistic semantic\\w* phonolog\\w* syntactic syntax reading naming verbs? nouns? lexical calculation arithmetic "
    "numbers? numerical magnitude spatial navigation rotation learning conditioning extinction prediction placebo drugs? "
    "alcohol cocaine nicotine smok\\w* cannabis craving methylphenidate ketamine oxytocin dopamine\\w* serotonin\\w* "
    "sleep deprivation hypoxia exercise meditation hypnosis rest resting-state resting "
    "schizophreni\\w* psychosis psychotic depress\\w* bipolar autis\\w* asd adhd alzheimer\\w* dementia mci parkinson\\w* "
    "ptsd ocd obsessive anorexia bulimia obesity addict\\w* dependence epilep\\w* stroke migraine tinnitus dyslexi\\w* "
    "schizotyp\\w* tbi injury carriers? genotype\\w* allele\\w* "
    "visual\\w* auditory acoustic somatosensory vision hearing audiovisual multisensory crossmodal "
    "gray-matter grey-matter vbm thickness fractional anisotropy white-matter "
    "deception lying humor rhythm\\w* timing interval video\\w* movies? film game\\w* "
    "eye gaze biological action\\w* observation imitation mirror"
).split()
TASK_RE = re.compile(r"(?<![A-Za-z])(?:" + "|".join(TASK_TERMS) + r")(?![A-Za-z])", re.I)
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")
# non-specific words: table / statistics / anatomy-generic / direction / design words that never say WHAT was compared
GENERIC = set(TABLE_GENERIC) | set(
    "brain regions region areas area showing show shows shown observed found revealed identified obtained detected using used based "
    "respectively including increased decreased increase decrease increases decreases greater higher lower stronger weaker less more "
    "difference differences different differential differentially comparison comparisons compared versus minus effect effects interaction "
    "interactions contrast contrasts conjunction correlation correlations correlated association associations associated related during "
    "condition conditions task tasks trial trials stimuli stimulus block blocks event events group groups participants subjects individuals "
    "sample samples study studies experiment experiments experimental session sessions run runs fmri mri pet functional imaging bold signal "
    "change changes response responses activity activities activated activation activations deactivation deactivations deactivated "
    "significant significantly analysis analyses whole roi rois model models parametric modulation modulated peak peaks cluster clusters "
    "maxima maximal local coordinates space table tables supplementary all both each two three four first second third one main overall "
    "additional separate separately positive negative positively negatively relative between within across versus neural cerebral "
    "networks network connectivity seed seeds brodmann hemisphere hemispheres bilateral left right lobe lobes gyrus gyri sulcus sulci "
    "cortex cortices cortical subcortical nucleus nuclei anterior posterior superior inferior middle medial lateral dorsal ventral rostral "
    "caudal regional structures structural anatomical locations location list listed foci focus effect-related demonstrating demonstrated "
    "showed exhibited exhibiting display displaying displayed elicited evoked reflecting related predicted predicting predict modulating "
    "identified identify identifying associated linked significantly-activated detail details including information reported "
    "results result data map maps image cluster-wise peak-level uncorrected corrected fwe fdr svc small volume threshold thresholded "
    "obtained derived revealed independent component components ica coordinates mni talairach altered alteration alterations "
    "abnormal abnormalities abnormality changed relationship relationships impaired impairment deficits deficit atrophy "
    "degeneration".split())
# anatomical tokens (specific, but not a contrast/condition): words used by the region vocabulary
ANAT = set()
for _terms, _abbr in list(FINE.values()) + list(GROUPS.values()):
    for _t in _terms + _abbr:
        ANAT |= set(re.findall(r"[a-z]{3,}", _t.lower().replace("\\w*", "")))
ANAT -= {"motor", "visual", "auditory", "word", "form", "place", "primary", "secondary", "somatosensory", "eye", "fields", "junction"}
ANAT |= {"cingulate", "insula", "insular", "amygdala", "hippocampus", "hippocampal", "parahippocampal", "thalamus", "thalamic",
         "striatum", "striatal", "caudate", "putamen", "cerebellum", "cerebellar", "vermis", "precuneus", "cuneus", "fusiform",
         "temporal", "frontal", "parietal", "occipital", "prefrontal", "orbitofrontal", "operculum", "opercular", "lingual", "angular",
         "supramarginal", "calcarine", "postcentral", "precentral", "pole", "planum", "gyral", "sulcal", "lobule", "pars", "division",
         "vermis", "crus"}


def content_words(s: str) -> list[str]:
    return [t.strip("-") for t in TOKEN_RE.findall(s.lower()) if t.strip("-") not in SW and len(t.strip("-")) > 2]


def clean_caption(caption: str) -> str:
    return STAT_EXPR.sub(" ", strip_label(caption or ""))


def score_caption(caption: str) -> dict:
    s = clean_caption(caption)
    words = content_words(s)
    specific = [w for w in words if w not in GENERIC]
    nonanat = [w for w in specific if w not in ANAT]
    pattern = bool(COMPARE_RE.search(s) or TASK_RE.search(s))
    has_contrast = pattern and len(nonanat) > 0
    regions_named = _hits(s, GROUP_RX) | _hits(s, FINE_RX)
    is_boiler = (len(specific) == 0) or (len(words) <= 3 and not has_contrast)
    if has_contrast:
        cat = "contrast/condition"
    elif regions_named:
        cat = "region list"
    elif is_boiler:
        cat = "boilerplate"
    else:
        cat = "other"
    return {"caption_clean": s.strip(), "n_words": len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-/]*", strip_label(caption or ""))),
            "n_content_words": len(words), "n_specific_words": len(specific), "specific_words": " ".join(dict.fromkeys(specific)),
            "has_contrast": has_contrast, "is_boilerplate": is_boiler, "names_region": bool(regions_named), "category": cat}


# data
def load_tables() -> pd.DataFrame:
    t = pd.read_parquet(OUT / "tables_catalog.parquet")
    t = t[t["is_coord_table"]].copy()
    return t[["pmid", "pmcid", "split", "table_id", "table_label", "caption_text", "n_coords_pubget"]].reset_index(drop=True)


def load_coords() -> pd.DataFrame:
    chunks = sorted((OUT / "pubget_data").glob("neurovlm_corpus_c2500_*/subset_allArticles_extractedData"))
    c = pd.concat([pd.read_csv(d / "coordinates.csv", dtype={"table_id": str, "table_label": str}) for d in chunks], ignore_index=True)
    sp = pd.concat([pd.read_csv(d / "coordinate_space.csv") for d in chunks], ignore_index=True).drop_duplicates("pmcid")
    return c.merge(sp, on="pmcid", how="left")


def load_papers() -> pd.DataFrame:
    from _hf_pins import hf_file

    p = pd.read_parquet(hf_file("neurovlm/neuro_image_papers", "publications.parquet"),  # pinned revision, sha256-checked
                        columns=["pmid", "name", "description"])
    p["pmid"] = p["pmid"].astype("int64")
    return p.rename(columns={"name": "title", "description": "abstract"}).drop_duplicates("pmid")


def atlas_labels(c: pd.DataFrame) -> np.ndarray:
    """Harvard-Oxford label of every coordinate (subcortical structure first, else cortical, else '')."""
    import nibabel as nib
    from nilearn import datasets
    from nimare.utils import tal2mni

    xyz = c[["x", "y", "z"]].to_numpy(float)
    tal = (c["coordinate_space"] == "TAL").to_numpy()
    xyz[tal] = tal2mni(xyz[tal])
    out = np.full(len(c), "", dtype=object)
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0)
    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", verbose=0)
    for atlas, kind in ((cort, "cort"), (sub, "sub")):
        img = atlas.maps if hasattr(atlas.maps, "affine") else nib.load(atlas.maps)
        data = np.asarray(img.dataobj).astype(int)
        ijk = np.rint(np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(img.affine).T)[:, :3].astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(data.shape)), axis=1)
        idx = np.zeros(len(xyz), dtype=int)
        idx[ok] = data[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        names = np.array([SHORT.get(n, re.sub(r"^(Left|Right) ", "", n)) for n in atlas.labels], dtype=object)
        lab = names[idx]
        if kind == "cort":
            out[idx > 0] = lab[idx > 0]
        else:  # subcortical structures override; cortex / white matter / ventricle labels of the sub atlas are ignored
            keep = (idx > 0) & ~pd.Series(lab).str.contains("Cerebral|Ventricle").to_numpy()
            out[keep] = lab[keep]
    return out


def table_regions(c: pd.DataFrame) -> pd.DataFrame:
    c = c.assign(region=atlas_labels(c))
    lab = c[c["region"] != ""]
    cnt = lab.groupby(["pmcid", "table_id", "region"]).size().rename("n").reset_index()
    cnt = cnt.sort_values(["pmcid", "table_id", "n", "region"], ascending=[True, True, False, True])
    top = cnt.groupby(["pmcid", "table_id"]).head(3)
    agg = top.groupby(["pmcid", "table_id"]).agg(top3=("region", list), top3_n=("n", list)).reset_index()
    n = c.groupby(["pmcid", "table_id"]).agg(n_coords=("x", "size"), n_labelled=("region", lambda s: int((s != "").sum()))).reset_index()
    out = n.merge(agg, on=["pmcid", "table_id"], how="left")
    out["top3"] = out["top3"].apply(lambda v: v if isinstance(v, list) else [])
    out["top3_n"] = out["top3_n"].apply(lambda v: v if isinstance(v, list) else [])
    out["top3_str"] = [", ".join(f"{r} ({k})" for r, k in zip(rs, ks)) for rs, ks in zip(out["top3"], out["top3_n"])]
    # dominant region: top-1 label of a table with >= MIN_LABELLED labelled coordinates, strictly ahead of the runner-up
    out["dominant"] = [rs[0] if nl >= MIN_LABELLED and (len(ks) == 1 or ks[0] > ks[1]) else None
                       for rs, ks, nl in zip(out["top3"], out["top3_n"], out["n_labelled"])]
    # stricter sensitivity target: >= MIN_IN_REGION coordinates inside the top-1 label itself
    out["dominant_strict"] = [rs[0] if rs and ks[0] >= MIN_IN_REGION and (len(ks) == 1 or ks[0] > ks[1]) else None
                              for rs, ks in zip(out["top3"], out["top3_n"])]
    return out


# flags that need the coordinates or the abstract
def region_match(caption: str, top3: list[str]) -> tuple[bool, bool]:
    """(fine, coarse): caption names one of the top-3 labels (label synonyms) / one of their lobe-structure groups."""
    if not top3:
        return False, False
    fine = any(_hits(caption, {r: FINE_RX[r]}) for r in top3 if r in FINE_RX)
    groups = set().union(*(GROUPS_OF.get(r, set()) for r in top3))
    coarse = bool(_hits(caption, {g: GROUP_RX[g] for g in groups}))
    return fine, coarse


def derange(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation without fixed points (each item is paired with another one)."""
    p = rng.permutation(n)
    for i in np.where(p == np.arange(n))[0]:
        if p[i] != i:
            continue
        j = (i + 1 + rng.integers(0, n - 1)) % n
        p[i], p[j] = p[j], p[i]
    return p


# decodability
def _f1_acc(y: np.ndarray, p: np.ndarray, k: int) -> tuple[float, float]:
    """Macro-F1 over k integer classes (0 for a class never true nor predicted, as sklearn zero_division=0) and accuracy."""
    tp = np.bincount(y[y == p], minlength=k)
    fp = np.bincount(p, minlength=k) - tp
    fn = np.bincount(y, minlength=k) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return float(f1.mean()), float((y == p).mean())


def decodability(df: pd.DataFrame, target: str = "dominant") -> tuple[pd.DataFrame, dict]:
    """Caption vs title+abstract (vs both) -> the table's dominant atlas region; train/val -> test split papers."""
    from scipy.sparse import hstack, vstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    d = df[df[target].notna() & df["split"].isin(["train", "val", "test"])].copy()
    labels = sorted(d.loc[d["split"] == "train", target].value_counts().head(N_REGIONS).index)
    d = d[d[target].isin(labels)].copy()
    d["y"] = d[target].map({r: i for i, r in enumerate(labels)}).astype(int)
    d["cap_txt"] = d["caption_clean"].fillna("")
    d["abs_txt"] = d["title"].fillna("") + ". " + d["abstract"].fillna("")
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    K = len(labels)

    def vec():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True, stop_words="english")

    def clf(C):
        return OneVsRestClassifier(LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=2000))

    feats = {}
    for name in ("cap", "abs"):
        v = vec().fit(tr[f"{name}_txt"])
        feats[name] = tuple(v.transform(x[f"{name}_txt"]) for x in (tr, va, te))
    feats["both"] = tuple(hstack([a, b]).tocsr() for a, b in zip(feats["cap"], feats["abs"]))
    names = {"cap": "caption", "abs": "title + abstract", "both": "caption + title + abstract"}
    preds, probs, chosen = {}, {}, {}
    for k, (Xtr, Xva, Xte) in feats.items():
        scores = {C: _f1_acc(va["y"].to_numpy(), clf(C).fit(Xtr, tr["y"]).predict(Xva), K)[0] for C in (0.3, 1, 3, 10, 30)}
        chosen[k] = max(scores, key=scores.get)
        m = clf(chosen[k]).fit(vstack([Xtr, Xva]).tocsr(), np.r_[tr["y"], va["y"]])
        probs[k] = m.predict_proba(Xte)
        preds[k] = m.classes_[probs[k].argmax(1)]
    y = te["y"].to_numpy()
    rng = np.random.default_rng(SEED)
    upm, inv = np.unique(te["pmid"].to_numpy(), return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(upm))]
    boot = [rng.integers(0, len(upm), len(upm)) for _ in range(N_BOOT)]  # resampled papers
    boot_idx = [np.concatenate([members[j] for j in b]) for b in boot]
    point = {k: _f1_acc(y, p, K) for k, p in preds.items()}
    boots = {k: np.array([_f1_acc(y[b], p[b], K) for b in boot_idx]) for k, p in preds.items()}
    rows = []
    for k in preds:
        lo, hi = np.percentile(boots[k], [2.5, 97.5], axis=0)
        rows.append({"input": names[k], "macro_f1": point[k][0], "macro_f1_lo": lo[0], "macro_f1_hi": hi[0],
                     "top1_acc": point[k][1], "top1_acc_lo": lo[1], "top1_acc_hi": hi[1], "C": chosen[k]})
    # chance: labels drawn from the training prior, independently of any text (2,000 draws)
    prior = np.bincount(np.r_[tr["y"], va["y"]], minlength=K) / (len(tr) + len(va))
    ch = np.array([_f1_acc(y, rng.choice(K, size=len(y), p=prior), K) for _ in range(N_BOOT)])
    lo, hi = np.percentile(ch, [2.5, 97.5], axis=0)
    rows.append({"input": "chance (training prior)", "macro_f1": ch[:, 0].mean(), "macro_f1_lo": lo[0], "macro_f1_hi": hi[0],
                 "top1_acc": ch[:, 1].mean(), "top1_acc_lo": lo[1], "top1_acc_hi": hi[1], "C": np.nan})
    maj = int(prior.argmax())
    mf, ma = _f1_acc(y, np.full(len(y), maj), K)
    rows.append({"input": f"majority class ({labels[maj]})", "macro_f1": mf, "macro_f1_lo": np.nan, "macro_f1_hi": np.nan,
                 "top1_acc": ma, "top1_acc_lo": np.nan, "top1_acc_hi": np.nan, "C": np.nan})
    # within-paper test of table-specific information: for two tables of the same test paper with different dominant
    # regions, does the caption model assign each table its own region? (chance and the abstract model = 0.5 by design)
    pairs = []
    for pi, mem in enumerate(members):
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                i, j = mem[a], mem[b]
                if y[i] != y[j]:
                    P = probs["cap"]
                    own, swap = P[i, y[i]] + P[j, y[j]], P[i, y[j]] + P[j, y[i]]
                    pairs.append((pi, 1.0 if own > swap else 0.5 if own == swap else 0.0))
    pairs = np.array(pairs) if pairs else np.zeros((0, 2))
    if len(pairs):
        per_paper = pd.DataFrame(pairs, columns=["paper", "ok"]).groupby("paper")["ok"].agg(["sum", "size"])
        pb = []
        for b in boot:
            sel = per_paper.reindex(b).dropna()
            pb.append(sel["sum"].sum() / max(sel["size"].sum(), 1))
        within = [float(pairs[:, 1].mean()), *map(float, np.percentile(pb, [2.5, 97.5])), int(len(pairs)), int(len(per_paper))]
    else:
        within = [np.nan, np.nan, np.nan, 0, 0]
    res = pd.DataFrame(rows)

    def dci(a, b, col):
        dd = boots[a][:, col] - boots[b][:, col]
        return [float(point[a][col] - point[b][col]), *map(float, np.percentile(dd, [2.5, 97.5]))]

    info = {"target": target, "n_train_tables": int(len(tr)), "n_val_tables": int(len(va)), "n_test_tables": int(len(te)),
            "n_test_papers": int(len(upm)), "regions": labels, "majority_region": labels[maj],
            "test_class_counts": {labels[k]: int(v) for k, v in pd.Series(y).value_counts().items()},
            "caption_minus_abstract_macro_f1": dci("cap", "abs", 0), "caption_minus_abstract_top1": dci("cap", "abs", 1),
            "both_minus_abstract_macro_f1": dci("both", "abs", 0), "both_minus_abstract_top1": dci("both", "abs", 1),
            "within_paper_pair_accuracy_caption": within[:3], "within_paper_n_pairs": within[3], "within_paper_n_papers": within[4]}
    return res, info


# reporting
def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        return s.replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def informativeness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sp in SPLITS:
        d = df if sp == "all" else df[df["split"] == sp]
        lab = d[d["n_labelled"] > 0]
        nr = lab[lab["names_region"]]
        r = {"split": sp, "n_tables": len(d), "n_papers": d["pmid"].nunique(),
             "words_median": d["n_words"].median(), "words_p25": d["n_words"].quantile(.25), "words_p75": d["n_words"].quantile(.75),
             "words_p10": d["n_words"].quantile(.10), "words_p90": d["n_words"].quantile(.90),
             "pct_le3_words": 100 * (d["n_words"] <= 3).mean(), "pct_le7_words": 100 * (d["n_words"] <= 7).mean(),
             "pct_has_contrast": 100 * d["has_contrast"].mean(), "pct_is_boilerplate": 100 * d["is_boilerplate"].mean(),
             "pct_names_region": 100 * d["names_region"].mean()}
        for c in CATS:
            r[f"pct_cat_{c}"] = 100 * (d["category"] == c).mean()
        r.update({"n_with_atlas_label": len(lab),
                  "pct_region_match_coarse": 100 * lab["region_match_coarse"].mean(),
                  "pct_region_match_coarse_chance": 100 * lab["region_match_coarse_chance"].mean(),
                  "pct_region_match_fine": 100 * lab["region_match_fine"].mean(),
                  "pct_region_match_fine_chance": 100 * lab["region_match_fine_chance"].mean(),
                  "n_naming_region": len(nr),
                  "pct_region_match_fine_if_named": 100 * nr["region_match_fine"].mean(),
                  "pct_region_match_fine_if_named_chance": 100 * nr["region_match_fine_chance"].mean(),
                  "pct_task_match": 100 * d["task_match"].mean(), "pct_task_match_chance": 100 * d["task_match_chance"].mean(),
                  "pct_contrast_and_task_match": 100 * (d["has_contrast"] & d["task_match"]).mean()})
        rows.append(r)
    return pd.DataFrame(rows)


def fmt_info_md(info: pd.DataFrame) -> str:
    f = lambda v: f"{v:.1f}"  # noqa: E731
    rows = []
    for rr in info.to_dict(orient="records"):
        rows.append({"split": rr["split"], "tables": f"{rr['n_tables']:,}",
                     "words, median [IQR]": f"{rr['words_median']:.0f} [{rr['words_p25']:.0f}–{rr['words_p75']:.0f}]",
                     "has_contrast": f(rr["pct_has_contrast"]), "is_boilerplate": f(rr["pct_is_boilerplate"]),
                     "names a region": f(rr["pct_names_region"]),
                     "region_match fine (chance)": f"{f(rr['pct_region_match_fine'])} ({f(rr['pct_region_match_fine_chance'])})",
                     "region_match coarse (chance)": f"{f(rr['pct_region_match_coarse'])} ({f(rr['pct_region_match_coarse_chance'])})",
                     "task_match (chance)": f"{f(rr['pct_task_match'])} ({f(rr['pct_task_match_chance'])})",
                     "contrast / region / boilerplate / other": " / ".join(f(rr[f"pct_cat_{c}"]) for c in CATS)})
    return md_table(pd.DataFrame(rows))


def examples(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    picked, rows = set(), []
    strata = [("random", df.index.to_numpy()), *[(c, df.index[df["category"] == c].to_numpy()) for c in ("contrast/condition", "boilerplate", "region list")]]
    for name, pool in strata:
        pool = np.array([i for i in pool if i not in picked])
        for i in rng.choice(pool, size=10, replace=False):
            picked.add(i)
            rows.append((name, i))
    ex = df.loc[[i for _, i in rows]].copy()
    ex.insert(0, "stratum", [n for n, _ in rows])
    ex["caption_full"] = ex["caption_text"]
    return ex[["stratum", "pmid", "table_label", "caption_full", "n_coords", "n_labelled", "top3_str", "title", "category",
               "has_contrast", "is_boilerplate", "region_match_fine", "region_match_coarse", "task_match", "split"]]


def examples_md(ex: pd.DataFrame, info: pd.DataFrame) -> str:
    a = info.set_index("split").loc["all"]
    out = [
        "# Coordinate-table captions: 40 seeded examples (caption, coordinates, atlas lookup)",
        "",
        "Drawn with `numpy.random.default_rng(0)` from the 20,541 pubget coordinate tables: 10 at random, then 10 from each",
        "rule-based category (`caption_usability.py`). **HO top-3** = the three Harvard-Oxford labels (cortical maxprob-thr25-2mm;",
        "subcortical structures from sub-maxprob-thr25-2mm) holding most of the table's own coordinates (count in brackets;",
        "coordinates outside both atlases, e.g. cerebellum or white matter, are not labelled: `labelled / n`). Flags: **C** names a",
        "contrast/condition, **B** boilerplate, **R** caption names one of the HO top-3 (label level; `r` = lobe level only),",
        "**A** shares >= 2 specific words with the abstract.",
        "",
        f"Over all tables: contrast {a['pct_has_contrast']:.0f} %, boilerplate {a['pct_is_boilerplate']:.0f} %, names any region "
        f"{a['pct_names_region']:.0f} %, names one of its own top-3 regions {a['pct_region_match_fine']:.0f} % (chance "
        f"{a['pct_region_match_fine_chance']:.0f} %), abstract overlap {a['pct_task_match']:.0f} % (chance {a['pct_task_match_chance']:.0f} %).",
        "",
    ]
    for st, g in ex.groupby("stratum", sort=False):
        out += [f"## {st} (n = {len(g)})", "",
                md_table(pd.DataFrame({
                    "PMID": g["pmid"].astype(str), "Table": g["table_label"].fillna(""),
                    "Caption (full)": g["caption_full"].fillna("").str.strip(),
                    "Coords (labelled / n)": [f"{a_}/{b_}" for a_, b_ in zip(g["n_labelled"], g["n_coords"])],
                    "HO top-3 (count)": g["top3_str"], "Paper title": g["title"].fillna(""),
                    "Flags": ["".join(s for s, on in (("C", c), ("B", b), ("R" if rf else "r", rf or rc), ("A", t)) if on) or "–"
                              for c, b, rf, rc, t in zip(g["has_contrast"], g["is_boilerplate"], g["region_match_fine"],
                                                         g["region_match_coarse"], g["task_match"])],
                }))]
    return "\n".join(out)


def _ci(a, lo, hi) -> str:
    return f"{a:.3f}" + ("" if lo is None or np.isnan(lo) else f" [{lo:.3f}, {hi:.3f}]")


def decodability_md(decs: dict, dinfos: dict) -> str:
    out = ["# Which text predicts a table's dominant atlas region?", "",
           "One-vs-rest logistic regression (liblinear, class-balanced) on TF-IDF 1–2-grams of the caption (label and statistics "
           "stripped), of the paper's title + abstract, or of both; C chosen on the val split from {0.3, 1, 3, 10, 30}, refitted on "
           "train + val, scored on test-split papers. Target: the Harvard-Oxford label holding most of the table's coordinates, "
           f"strictly ahead of the runner-up, restricted to the {N_REGIONS} most frequent targets in train. CI = 2,000 paper-level "
           "bootstrap resamples; chance = labels drawn from the training prior (2,000 draws).", ""]
    titles = {"dominant": f"Main: tables with >= {MIN_LABELLED} labelled coordinates",
              "dominant_strict": f"Sensitivity: >= {MIN_IN_REGION} coordinates inside the dominant region"}
    for t, dec in decs.items():
        i = dinfos[t]
        out += [f"## {titles[t]} ({i['n_train_tables']:,} train + {i['n_val_tables']:,} val tables; test {i['n_test_tables']:,} tables "
                f"from {i['n_test_papers']:,} papers)", "",
                md_table(pd.DataFrame({"Input text": dec["input"],
                                       "Macro-F1 [95 % CI]": [_ci(*r) for r in zip(dec["macro_f1"], dec["macro_f1_lo"], dec["macro_f1_hi"])],
                                       "Top-1 accuracy [95 % CI]": [_ci(*r) for r in zip(dec["top1_acc"], dec["top1_acc_lo"], dec["top1_acc_hi"])]}))]
        c, b, w = i["caption_minus_abstract_macro_f1"], i["both_minus_abstract_macro_f1"], i["within_paper_pair_accuracy_caption"]
        c1, b1 = i["caption_minus_abstract_top1"], i["both_minus_abstract_top1"]
        out += [f"Caption − abstract: macro-F1 {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}], top-1 {c1[0]:+.3f} [{c1[1]:+.3f}, {c1[2]:+.3f}]. "
                f"Caption + abstract − abstract: macro-F1 {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}], top-1 {b1[0]:+.3f} [{b1[1]:+.3f}, {b1[2]:+.3f}].",
                "",
                f"Within-paper test (table-specific information): for {i['within_paper_n_pairs']} pairs of tables from the same test paper "
                f"({i['within_paper_n_papers']} papers) with different dominant regions, the caption model gives each table its own region "
                f"more probability than the other table's region in {w[0]:.3f} [{w[1]:.3f}, {w[2]:.3f}] of pairs (abstract model and chance: "
                "0.5 by construction, since both tables share one abstract).", ""]
    return "\n".join(out)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT, help="Directory with run_analysis.py's outputs; results are written there.")
    OUT = p.parse_args().out
    from nltk.stem import PorterStemmer

    warnings.filterwarnings("ignore")
    stem = PorterStemmer().stem
    log("load")
    tables = load_tables()
    coords = load_coords()
    reg = table_regions(coords)
    papers = load_papers()
    df = tables.merge(reg, on=["pmcid", "table_id"], how="left").merge(papers, on="pmid", how="left")
    df["n_coords"] = df["n_coords"].fillna(0).astype(int)
    df["n_labelled"] = df["n_labelled"].fillna(0).astype(int)
    df["top3"] = df["top3"].apply(lambda v: v if isinstance(v, list) else [])
    df["top3_str"] = df["top3_str"].fillna("")
    df = pd.concat([df, pd.DataFrame([score_caption(c) for c in df["caption_text"].fillna("")])], axis=1)

    rng = np.random.default_rng(SEED)
    rm = [region_match(c, t) for c, t in zip(df["caption_clean"], df["top3"])]
    df["region_match_fine"], df["region_match_coarse"] = [x[0] for x in rm], [x[1] for x in rm]
    lab = np.where(df["n_labelled"] > 0)[0]
    perm = lab[derange(len(lab), rng)]
    df["region_match_fine_chance"] = False
    df["region_match_coarse_chance"] = False
    rmc = [region_match(df.at[i, "caption_clean"], df.at[j, "top3"]) for i, j in zip(lab, perm)]
    df.loc[lab, "region_match_fine_chance"] = [x[0] for x in rmc]
    df.loc[lab, "region_match_coarse_chance"] = [x[1] for x in rmc]

    cap_st = [{stem(w) for w in s.split()} for s in df["specific_words"]]
    abs_words = [{stem(w) for w in content_words(f"{t} {a}")} for t, a in zip(df["title"].fillna(""), df["abstract"].fillna(""))]
    df["n_shared_abstract"] = [len(c & a) for c, a in zip(cap_st, abs_words)]
    df["task_match"] = df["n_shared_abstract"] >= 2
    # chance: a random other PAPER's abstract
    pm = df["pmid"].to_numpy()
    pperm = derange(len(df), rng)
    same = pm[pperm] == pm
    while same.any():  # avoid drawing a table of the same paper
        pperm[same] = rng.integers(0, len(df), same.sum())
        same = pm[pperm] == pm
    df["task_match_chance"] = [len(cap_st[i] & abs_words[j]) >= 2 for i, j in enumerate(pperm)]

    log("informativeness")
    info = informativeness(df)
    info.round(2).to_csv(OUT / "caption_informativeness.csv", index=False)
    (OUT / "caption_informativeness.md").write_text(
        "# Caption informativeness (rule-based, all 20,541 coordinate tables)\n\n"
        "Percent of coordinate tables; chance in brackets = the same caption scored against a random other table's top-3 regions /"
        " a random other paper's abstract. Fine = Harvard-Oxford label level, coarse = lobe / structure level. Rules in"
        " `caption_usability.py`.\n\n" + fmt_info_md(info))

    log("decodability")
    decs, dinfos = {}, {}
    for target in ("dominant", "dominant_strict"):
        decs[target], dinfos[target] = decodability(df, target)
        log(f"  {target}: done")
    dec, dinfo = decs["dominant"], dinfos["dominant"]
    pd.concat([d.assign(target=t) for t, d in decs.items()]).round(4).to_csv(OUT / "caption_decodability.csv", index=False)
    (OUT / "caption_decodability.md").write_text(decodability_md(decs, dinfos))
    ex = examples(df)
    ex.to_csv(OUT / "caption_examples.csv", index=False)
    (OUT / "caption_examples.md").write_text(examples_md(ex, info))

    keep = ["pmid", "pmcid", "split", "table_id", "table_label", "caption_text", "n_coords", "n_labelled", "top3_str", "dominant", "dominant_strict",
            "n_words", "n_content_words", "specific_words", "has_contrast", "is_boilerplate", "names_region", "category",
            "region_match_fine", "region_match_coarse", "region_match_fine_chance", "region_match_coarse_chance",
            "n_shared_abstract", "task_match", "task_match_chance"]
    df[keep].to_parquet(OUT / "caption_flags.parquet", index=False)
    a = info.set_index("split").loc["all"]
    summary = {"informativeness_all": {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else int(v)) for k, v in a.items()},
               "decodability": dec.round(4).to_dict(orient="records"), "decodability_info": dinfo,
               "decodability_strict": decs["dominant_strict"].round(4).to_dict(orient="records"),
               "decodability_strict_info": dinfos["dominant_strict"],
               "pct_coords_unlabelled": round(100 * (1 - df["n_labelled"].sum() / df["n_coords"].sum()), 2),
               "n_tables_with_dominant_region": int(df["dominant"].notna().sum())}
    (OUT / "caption_usability.json").write_text(json.dumps(summary, indent=1, default=str))
    print(fmt_info_md(info))
    print(decodability_md(decs, dinfos))


if __name__ == "__main__":
    main()
