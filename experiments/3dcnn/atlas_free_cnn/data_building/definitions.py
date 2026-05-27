"""Definitions and category policy for clean multi-positive text targets."""

from __future__ import annotations

import re


ALLOWED_PUBMED_MESH_CATEGORIES = {
    "anatomical_region",
    "cognitive_construct",
    "disorder",
}

EXCLUDED_POSITIVE_CATEGORIES = {
    "molecular",
    "biological_process",
    "method",
    "demographic",
    "organism",
    "broad_general",
    "broad/general",
    "raw_abstract",
    "llm_extracted_term",
}

POSITIVE_WEIGHTS = {
    "nilearn_atlas_label": 1.00,
    "nilearn_network_label": 1.00,
    "mesh_anatomical_region": 1.00,
    "mesh_cognitive_construct": 0.90,
    "cognitive_atlas_exact_match": 0.90,
    "mesh_disorder": 0.70,
    "paper_wiki_summary": 0.70,
    "paper_title": 0.50,
}

NETWORK_DEFINITIONS = {
    "default mode network": "A large-scale functional network commonly associated with internally directed cognition, autobiographical memory, self-referential processing, and mind wandering.",
    "visual network": "A functional network centered on occipital and extrastriate regions involved in visual perception and visual information processing.",
    "somatomotor network": "A functional network involving motor and somatosensory cortices that supports movement control and bodily sensation.",
    "dorsal attention network": "A functional network involved in voluntary spatial attention and goal-directed orienting.",
    "ventral attention network": "A functional network involved in stimulus-driven attention and detection of salient events.",
    "frontoparietal control network": "A functional network supporting cognitive control, task switching, working memory, and flexible goal-directed behavior.",
    "limbic network": "A functional network involving medial temporal and orbitofrontal regions associated with emotion, valuation, and memory-related processing.",
    "control network": "A functional network supporting cognitive control, task switching, working memory, and flexible goal-directed behavior.",
    "attention network": "A functional network involved in selection, orienting, and detection of behaviorally relevant information.",
    "salience network": "A functional network involving anterior insula and cingulate regions associated with salience detection, interoception, and task set maintenance.",
    "visual": "A functional network centered on occipital and extrastriate regions involved in visual perception and visual information processing.",
    "somatomotor": "A functional network involving motor and somatosensory cortices that supports movement control and bodily sensation.",
    "dorsal attention": "A functional network involved in voluntary spatial attention and goal-directed orienting.",
    "ventral attention": "A functional network involved in stimulus-driven attention and detection of salient events.",
    "frontoparietal": "A functional network supporting cognitive control, task switching, working memory, and flexible goal-directed behavior.",
    "control": "A functional network supporting cognitive control, task switching, working memory, and flexible goal-directed behavior.",
    "limbic": "A functional network involving medial temporal and orbitofrontal regions associated with emotion, valuation, and memory-related processing.",
}

YEO_7_NETWORK_LABELS = {
    "7Networks_1": "Visual network",
    "7Networks_2": "Somatomotor network",
    "7Networks_3": "Dorsal attention network",
    "7Networks_4": "Ventral attention network",
    "7Networks_5": "Limbic network",
    "7Networks_6": "Frontoparietal control network",
    "7Networks_7": "Default mode network",
}

SCHAEFER_NETWORK_ABBREVIATIONS = {
    "Vis": "Visual network",
    "SomMot": "Somatomotor network",
    "DorsAttn": "Dorsal attention network",
    "SalVentAttn": "Ventral attention network",
    "Limbic": "Limbic network",
    "Cont": "Frontoparietal control network",
    "Default": "Default mode network",
}


def normalize_key(text: str) -> str:
    """Normalize text for deterministic IDs and conservative exact matching."""

    text = str(text).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str, *, max_len: int = 80) -> str:
    slug = normalize_key(text).replace(" ", "_")
    return (slug[:max_len].strip("_") or "unknown")


def fallback_region_definition(label_name: str) -> str:
    return f"{label_name} is an anatomical brain region or atlas-defined parcel represented by this spatial mask."


def network_definition(label_name: str) -> str | None:
    key = normalize_key(label_name)
    if key in NETWORK_DEFINITIONS:
        return NETWORK_DEFINITIONS[key]
    for name, definition in NETWORK_DEFINITIONS.items():
        if normalize_key(name) in key or key in normalize_key(name):
            return definition
    return None


def display_atlas_label(label_name: str, atlas_name: str = "") -> str:
    """Convert atlas-specific shorthand into a readable display label."""

    atlas_key = normalize_key(atlas_name)
    raw = str(label_name).strip()
    if raw in YEO_7_NETWORK_LABELS:
        return YEO_7_NETWORK_LABELS[raw]
    if "schaefer" in atlas_key and "_" in raw:
        parts = raw.split("_")
        # Example: 7Networks_LH_Vis_1 -> Left visual network parcel 1.
        hemi = {"LH": "Left", "RH": "Right"}.get(parts[1], parts[1]) if len(parts) > 1 else ""
        net = SCHAEFER_NETWORK_ABBREVIATIONS.get(parts[2], parts[2]) if len(parts) > 2 else ""
        parcel = parts[-1] if parts[-1].isdigit() else ""
        if hemi and net:
            return f"{hemi} {net} parcel {parcel}".strip()
    if "smith" in atlas_key and raw.lower().startswith("smith_2009 component"):
        return raw.replace("_", " ")
    return raw


def atlas_label_definition(label_name: str, atlas_name: str = "") -> tuple[str, bool]:
    """Return a definition and whether a fallback template was used."""

    display = display_atlas_label(label_name, atlas_name)
    definition = network_definition(display) or network_definition(label_name)
    if definition is not None:
        return definition, False
    return fallback_region_definition(display), True


def text_pair(term: str, definition: str) -> str:
    definition = " ".join(str(definition or "").split())
    return f"{str(term).strip()} [SEP] {definition}"
