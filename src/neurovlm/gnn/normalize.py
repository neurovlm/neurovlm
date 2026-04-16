"""Entity normalization — Phase 2, Step 5.

Merges MeSH, CogAtlas, and NLP knowledge graphs into a single unified entity
table with stable canonical IDs.

Merge priority
--------------
1. **MeSH** — DescriptorUI is the canonical ID backbone.  Every concept that
   exists in MeSH keeps its MeSH ID (e.g. ``D009048``).

2. **CogAtlas** — Each concept is checked against the MeSH synonym table
   (name, normalized to lowercase + stripped punctuation).  Match → use MeSH
   DescriptorUI.  No match → assign ``cogat_{trm_id}`` as canonical ID using
   the CogAtlas internal identifier.

3. **NLP** — Each term is checked against the MeSH synonym table first, then
   against the CogAtlas concept name table.  Match → re-point to that
   canonical ID.  No match → *discard* (conservative default; set
   ``keep_unmatched_nlp=True`` to instead keep with ``nlp_{slug}`` IDs).

The key insight: NLP primarily contributes *edges* (re-pointed to canonical
node IDs), not new nodes.  A co_occurs_with edge between two NLP terms that
both map to MeSH nodes survives in the unified graph under the MeSH IDs.

All merge decisions are written to a JSON audit log.

Typical usage
-------------
>>> from neurovlm.gnn.normalize import run_normalization
>>> import pandas as pd
>>> results = run_normalization(
...     mesh_descriptors_df=pd.read_parquet("data/mesh_kg/mesh_descriptors.parquet"),
...     mesh_nodes_df=pd.read_parquet("data/mesh_kg/mesh_kg_nodes.parquet"),
...     mesh_edges_df=pd.read_parquet("data/mesh_kg/mesh_kg_edges_all.parquet"),
...     nlp_nodes_df=pd.read_parquet("data/nlp_kg/nlp_kg_nodes.parquet"),
...     nlp_edges_df=pd.read_parquet("data/nlp_kg/nlp_kg_edges.parquet"),
...     cogatlas_out_path="data/cogatlas/cogatlas_concepts.parquet",
...     out_dir="data/unified_kg",
... )
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    This is the canonical normalization applied to every string before lookup.
    Punctuation is replaced with spaces so that ``motor-cortex`` and
    ``motor cortex`` both normalize to ``motor cortex``.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# MeSH synonym lookup
# ---------------------------------------------------------------------------

def build_mesh_synonym_lookup(
    descriptor_df: pd.DataFrame,
) -> dict[str, str]:
    """Build ``{normalized_string → MeSH DescriptorUI}`` lookup.

    Indexes every descriptor name and every synonym from the MeSH descriptor
    table.  When the same normalized string maps to multiple UIs (rare in
    practice), the first one encountered wins.

    Parameters
    ----------
    descriptor_df:
        DataFrame with columns ``ui``, ``name``, ``synonyms`` (list of str).
        Produced by :func:`neurovlm.gnn.mesh.parse_mesh_descriptor_xml`.

    Returns
    -------
    dict mapping normalized strings → DescriptorUI.
    """
    lookup: dict[str, str] = {}
    collision_count = 0

    for _, row in descriptor_df.iterrows():
        ui = row["ui"]
        raw_syns = row["synonyms"]
        # Parquet may deserialise list columns as numpy arrays or other iterables
        if raw_syns is None or isinstance(raw_syns, float):
            synonyms = []
        elif isinstance(raw_syns, (list, tuple)):
            synonyms = list(raw_syns)
        else:
            try:
                synonyms = list(raw_syns)
            except TypeError:
                synonyms = []
        all_names = [row["name"]] + synonyms

        for name in all_names:
            if not name:
                continue
            key = _norm(name)
            if not key:
                continue
            if key in lookup:
                collision_count += 1
            else:
                lookup[key] = ui

    logger.info(
        "MeSH synonym lookup: %d entries for %d descriptors (%d collisions ignored)",
        len(lookup), len(descriptor_df), collision_count,
    )
    return lookup


# ---------------------------------------------------------------------------
# CogAtlas fetcher — HTML scraping (JSON endpoint unreliable as of 2026)
# ---------------------------------------------------------------------------

# Mapping from CogAtlas relationship labels to unified KG relation types.
# Only relationships with a clear mapping are kept; others are dropped.
_COGAT_REL_MAP: dict[str, str] = {
    "kind_of": "narrower_term_of",
}


def fetch_cogatlas_concepts(
    out_path: Optional[str | Path] = None,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Fetch all CogAtlas concept IDs and names from cognitiveatlas.org.

    Scrapes the alphabetical listing pages (``/concepts/a`` … ``/concepts/z``).
    The JSON API endpoint at ``/concept/json/{id}`` is unreliable; this
    function uses the stable HTML listing pages instead.

    Parameters
    ----------
    out_path:
        If provided, save the result to this parquet path and load from it on
        subsequent calls (avoids re-fetching the full site).
    delay:
        Inter-request delay in seconds (default 0.3 s → ~3 req/s).

    Returns
    -------
    pd.DataFrame with columns ``trm_id`` and ``name``.
    """
    if out_path is not None and Path(out_path).exists():
        logger.info("Loading CogAtlas concepts from %s", out_path)
        return pd.read_parquet(out_path)

    base = "https://www.cognitiveatlas.org"
    rows: list[dict] = []

    for alpha in tqdm(range(ord("a"), ord("z") + 1), total=26, desc="CogAtlas listing pages"):
        char = chr(alpha)
        url = f"{base}/concepts/{char}"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch /concepts/%s: %s — skipping", char, exc)
            time.sleep(delay * 5)
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for link in soup.find_all("a", class_="concept tooltip"):
            href = link.get("href", "")
            name = link.get_text(strip=True)
            # href pattern: /concept/id/trm_XXXXX
            if "/concept/id/" in href:
                trm_id = href.split("/")[-1]
                if trm_id and name:
                    rows.append({"trm_id": trm_id, "name": name})

        time.sleep(delay)

    df = pd.DataFrame(rows).drop_duplicates(subset=["trm_id"]).reset_index(drop=True)
    logger.info("Fetched %d CogAtlas concepts", len(df))

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Saved CogAtlas concepts to %s", out_path)

    return df


def fetch_cogatlas_relations(
    cogatlas_df: pd.DataFrame,
    out_path: Optional[str | Path] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Scrape 'is a kind of' relationships from individual CogAtlas concept pages.

    For each concept in *cogatlas_df*, fetches
    ``https://www.cognitiveatlas.org/concept/id/{trm_id}`` and parses the
    "Asserted relationships" section.  Only ``kind_of`` relationships are
    extracted (the others are almost universally blank on the site).

    Parameters
    ----------
    cogatlas_df:
        DataFrame with columns ``trm_id`` and ``name``, as returned by
        :func:`fetch_cogatlas_concepts`.
    out_path:
        If provided, save/load from this parquet path to avoid re-fetching.
    delay:
        Inter-request delay in seconds (default 0.5 s).

    Returns
    -------
    pd.DataFrame with columns ``child``, ``parent``, ``relationship``.
        ``child`` is a kind of ``parent``.
        ``relationship`` is always ``"kind_of"`` (the only populated relation).
    """
    if out_path is not None and Path(out_path).exists():
        logger.info("Loading CogAtlas relations from %s", out_path)
        return pd.read_parquet(out_path)

    base = "https://www.cognitiveatlas.org"
    # Build trm_id → name lookup for matching scraped concept names to known entries
    id_to_name = dict(zip(cogatlas_df["trm_id"], cogatlas_df["name"]))
    name_to_id = {v: k for k, v in id_to_name.items()}

    rows: list[dict] = []

    for _, row in tqdm(cogatlas_df.iterrows(), total=len(cogatlas_df), desc="CogAtlas relations"):
        trm_id = str(row["trm_id"])
        child_name = str(row["name"])
        url = f"{base}/concept/id/{trm_id}"

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s — skipping", url, exc)
            time.sleep(delay * 5)
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # The page contains a section like:
        # "<concept> is a kind of <link to parent concept>"
        # We look for anchor tags whose href matches the /concept/id/ pattern
        # inside a block that contains the text "is a kind of".
        for tag in soup.find_all(string=lambda t: t and "is a kind of" in t):
            parent_elem = tag.parent if tag.parent else None
            if parent_elem is None:
                continue
            # Look for concept links in the same container or siblings
            container = parent_elem.find_parent() or parent_elem
            for link in container.find_all("a", href=lambda h: h and "/concept/id/" in h):
                parent_name = link.get_text(strip=True)
                if parent_name and parent_name != child_name and parent_name in name_to_id:
                    rows.append({
                        "child": child_name,
                        "parent": parent_name,
                        "relationship": "kind_of",
                    })

        time.sleep(delay)

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    logger.info(
        "CogAtlas relations: %d kind_of triples across %d concepts",
        len(df), df["child"].nunique() if len(df) else 0,
    )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Saved CogAtlas relations to %s", out_path)

    return df


# ---------------------------------------------------------------------------
# CogAtlas KG builder
# ---------------------------------------------------------------------------

def build_cogatlas_kg(
    cogatlas_df: pd.DataFrame,
    mesh_lookup: dict[str, str],
    cogatlas_edges_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Map CogAtlas concepts to canonical IDs and build CogAtlas KG tables.

    For each concept:

    1. Normalize the concept name and check against the MeSH synonym table.
       Match → use that MeSH DescriptorUI as the canonical ID.
    2. No match → assign ``cogat_{trm_id}`` as the canonical ID.

    Parameters
    ----------
    cogatlas_df:
        DataFrame with columns ``trm_id``, ``name``
        (from :func:`fetch_cogatlas_concepts`).
    mesh_lookup:
        ``{normalized_string → MeSH UI}`` from :func:`build_mesh_synonym_lookup`.
    cogatlas_edges_df:
        Optional DataFrame with columns ``parent``, ``child``, ``relationship``
        (concept names, from ``parse_cogatlas`` in ``cogatlas.py``).
        When provided, edges are re-pointed to canonical IDs.

    Returns
    -------
    cogat_nodes : pd.DataFrame
        Node table.  Columns: ``canonical_id``, ``name``, ``node_type``,
        ``source``, ``trm_id``, ``match_type``.
    cogat_edges : pd.DataFrame
        Edge table (empty DataFrame if *cogatlas_edges_df* is None).
        Columns: ``subject_id``, ``relation_type``, ``object_id``,
        ``source``, ``weight``.
    cogat_to_canonical : dict[trm_id → canonical_id]
    """
    records: list[dict] = []
    cogat_to_canonical: dict[str, str] = {}

    for _, row in cogatlas_df.iterrows():
        trm_id = str(row["trm_id"])
        name = str(row["name"])
        key = _norm(name)

        if key and key in mesh_lookup:
            canonical_id = mesh_lookup[key]
            match_type = "mesh_synonym_match"
        else:
            canonical_id = f"cogat_{trm_id}"
            match_type = "new_cogat_node"

        cogat_to_canonical[trm_id] = canonical_id
        records.append({
            "canonical_id": canonical_id,
            "name": name,
            "node_type": "cognitive_construct",
            "source": "cogatlas",
            "trm_id": trm_id,
            "match_type": match_type,
        })

    cogat_nodes = pd.DataFrame(records)
    n_mesh = (cogat_nodes["match_type"] == "mesh_synonym_match").sum()
    n_new = (cogat_nodes["match_type"] == "new_cogat_node").sum()
    logger.info(
        "CogAtlas: %d concepts → %d mapped to MeSH, %d new cogat_ nodes",
        len(cogat_nodes), n_mesh, n_new,
    )

    # Build edge table if relationships were provided
    _empty_edges = pd.DataFrame(
        columns=["subject_id", "relation_type", "object_id", "source", "weight"]
    )

    if cogatlas_edges_df is None or len(cogatlas_edges_df) == 0:
        return cogat_nodes, _empty_edges, cogat_to_canonical

    # Build name → trm_id reverse map for edge endpoint lookup
    name_to_trm: dict[str, str] = {
        str(row["name"]): str(row["trm_id"])
        for _, row in cogatlas_df.iterrows()
    }

    edge_rows: list[dict] = []
    skipped = 0
    skipped_unmapped_rel = 0
    for _, erow in cogatlas_edges_df.iterrows():
        # Map CogAtlas relationship label to unified KG relation type; skip if unknown
        rel_type = _COGAT_REL_MAP.get(str(erow.get("relationship", "")))
        if rel_type is None:
            skipped_unmapped_rel += 1
            continue

        child_trm = name_to_trm.get(str(erow["child"]))
        parent_trm = name_to_trm.get(str(erow["parent"]))
        if child_trm is None or parent_trm is None:
            skipped += 1
            continue
        child_canon = cogat_to_canonical.get(child_trm)
        parent_canon = cogat_to_canonical.get(parent_trm)
        if child_canon is None or parent_canon is None:
            skipped += 1
            continue
        if child_canon == parent_canon:
            continue  # self-loop after merging
        edge_rows.append({
            "subject_id": child_canon,
            "relation_type": rel_type,
            "object_id": parent_canon,
            "source": "cogatlas",
            "weight": 1.0,
        })

    cogat_edges = pd.DataFrame(edge_rows).drop_duplicates(
        subset=["subject_id", "relation_type", "object_id"]
    )
    logger.info(
        "CogAtlas edges: %d triples kept (%d skipped — name not in concept list, "
        "%d skipped — unmapped relation type)",
        len(cogat_edges), skipped, skipped_unmapped_rel,
    )
    return cogat_nodes, cogat_edges, cogat_to_canonical


# ---------------------------------------------------------------------------
# Substring matching helpers for NLP normalization
# ---------------------------------------------------------------------------

#: Node types where exact matching is trusted — discard if no match.
DISCARD_TYPES: frozenset[str] = frozenset({"modality", "method", "metric"})

#: Node types where a richer matching strategy is used before giving up.
#: These contribute semantically important graph structure even as nlp_ nodes.
KEEP_TYPES: frozenset[str] = frozenset({
    "disorder", "anatomical_region", "cognitive_construct", "network", "intervention"
})

#: Per-type secondary discard patterns applied to KEEP_TYPE nodes that failed
#: all matching attempts and are about to become ``nlp_`` nodes.
#:
#: These catch systematic misclassifications by the upstream NLP node labeller
#: that are not caught by degree/community filters because the terms are
#: high-degree (they co-occur with everything) rather than low-degree noise.
#:
#: Applied ONLY to nodes that would otherwise become ``nlp_`` nodes; nodes that
#: matched MeSH or CogAtlas are never filtered here.
#:
#: Extend this dict in the notebook for any domain-specific false positives.
NLP_NODE_FILTERS: dict[str, re.Pattern] = {
    "anatomical_region": re.compile(
        # Functional states mislabelled as anatomy by the NLP classifier
        r"\b(?:activity|activation)\b"
        # Modality terms that slipped into anatomical_region
        r"|\bmri\b"
        # Standalone over-generic anatomical plurals / meta-descriptors.
        # Anchored to the full normalised string so "left hemisphere" is kept.
        r"|^(?:hemisphere|lobes|whole-brain|whole brain)$"
    ),
}


def _build_mesh_word_index(
    mesh_lookup: dict[str, str],
    primary_names: Optional[frozenset[str] | set[str]] = None,
) -> dict[str, list[tuple[str, str, bool]]]:
    """Build ``{first_word → [(normalized_syn, ui, is_primary_name), ...]}`` for substring lookups.

    Indexed only by the *first* word of each synonym so that ``_try_substring_match``
    can narrow candidates efficiently.  Building once and reusing across all NLP
    terms cuts the substring search from O(N·M) to O(N·k) where k ≪ M.

    Parameters
    ----------
    mesh_lookup:
        ``{normalized_string → ui}`` from :func:`build_mesh_synonym_lookup`.
    primary_names:
        Set of normalized *primary* descriptor names (``_norm(row["name"])``).
        When provided, entries that are primary names are tagged ``is_primary=True``
        so that :func:`_try_substring_match` can prefer them over synonyms.
    """
    pn = primary_names or set()
    idx: dict[str, list[tuple[str, str, bool]]] = {}
    for syn, ui in mesh_lookup.items():
        words = syn.split()
        if words:
            is_primary = syn in pn
            idx.setdefault(words[0], []).append((syn, ui, is_primary))
    return idx


def _try_substring_match(
    term: str,
    word_index: dict[str, list[tuple[str, str]]],
    min_len: int = 6,
) -> Optional[str]:
    """Return a MeSH UI if *term* matches a MeSH synonym via word-boundary substring.

    Two directions are checked, both with whole-word boundary enforcement:

    * **NLP ⊆ MeSH** (Direction 1) — the NLP term appears as a contiguous,
      word-aligned sequence inside a MeSH synonym.
      e.g. ``"alzheimer"`` inside ``"alzheimer disease"`` → D000544.
      Among all candidates, the one with the *shortest* MeSH synonym is
      returned — a tighter match is more specific than a longer one.
      e.g. ``"occipital"`` matches both ``"occipital lobe"`` (14 chars) and
      ``"occipital encephalocele"`` (23 chars); the shorter wins → D006657.

    * **MeSH ⊆ NLP** (Direction 2) — a multi-word MeSH synonym appears inside
      the NLP term.  e.g. ``"alzheimer disease"`` inside ``"alzheimer disease
      patients"``.  **Single-word MeSH synonyms are excluded** from this
      direction to prevent generic words (``"affect"``, ``"nucleus"``) from
      matching longer NLP terms that happen to contain them.

    Whole-word boundary is enforced by padding both strings with spaces before
    checking containment, so ``"rest"`` does *not* match ``"resting"`` and
    ``"brain"`` does *not* match ``"brainstem"``.

    Parameters
    ----------
    term:
        Normalized NLP term (output of :func:`_norm`).
    word_index:
        Built by :func:`_build_mesh_word_index`.
    min_len:
        Minimum character length for the *shorter* string in both directions.
        Default 6.

    Returns
    -------
    MeSH DescriptorUI string, or ``None`` if no match found.
    """
    if not term or len(term) < min_len:
        return None

    words = term.split()
    if not words:
        return None

    p_term = f" {term} "

    # Direction-1 candidates: NLP term ⊆ MeSH syn.
    # Preference order: primary-name matches > synonym matches; shorter wins within each tier.
    dir1_primary: Optional[tuple[int, str]] = None   # (syn_len, ui)
    dir1_synonym: Optional[tuple[int, str]] = None

    # Direction-2 candidates: multi-word MeSH syn ⊆ NLP term.
    dir2_primary: Optional[tuple[int, str]] = None
    dir2_synonym: Optional[tuple[int, str]] = None

    for syn, ui, is_primary in word_index.get(words[0], []):
        p_syn = f" {syn} "

        # Direction 1: NLP term is a word-aligned substring of the MeSH synonym
        if p_term in p_syn:
            bucket = dir1_primary if is_primary else dir1_synonym
            if bucket is None or len(syn) < bucket[0]:
                if is_primary:
                    dir1_primary = (len(syn), ui)
                else:
                    dir1_synonym = (len(syn), ui)

        # Direction 2: multi-word MeSH synonym is a word-aligned substring of the NLP term.
        # Require ≥ 2 words to exclude generic single-word synonyms (e.g. "affect", "nucleus")
        # from matching as substrings of longer NLP terms.
        elif (
            len(syn.split()) >= 2
            and len(syn) >= min_len
            and p_syn in p_term
        ):
            bucket = dir2_primary if is_primary else dir2_synonym
            if bucket is None or len(syn) < bucket[0]:
                if is_primary:
                    dir2_primary = (len(syn), ui)
                else:
                    dir2_synonym = (len(syn), ui)

    # Return best match: Direction-1 primary > Direction-1 synonym > Direction-2 primary > Direction-2 synonym
    for candidate in (dir1_primary, dir1_synonym, dir2_primary, dir2_synonym):
        if candidate is not None:
            return candidate[1]
    return None


# ---------------------------------------------------------------------------
# NLP entity normalization
# ---------------------------------------------------------------------------

def normalize_nlp_entities(
    nlp_nodes_df: pd.DataFrame,
    mesh_lookup: dict[str, str],
    cogat_name_lookup: dict[str, str],
    keep_unmatched: bool = False,
    keep_types: Optional[frozenset[str] | set[str]] = KEEP_TYPES,
    min_substring_len: int = 6,
    mesh_primary_names: Optional[frozenset[str] | set[str]] = None,
) -> tuple[dict[str, str], list[dict]]:
    """Map each NLP term to a canonical ID using a type-conditional strategy.

    Matching is applied in this order for every NLP term:

    1. **Exact MeSH synonym match** (all types).
    2. **Exact CogAtlas name match** (all types).
    3. **Word-boundary substring match against MeSH** (*keep_types* only) —
       catches truncated surface forms like ``"alzheimer"`` → ``"Alzheimer Disease"``
       or ``"occipital"`` → ``"Occipital Lobe"``.
    4. **Keep as** ``nlp_{slug}`` (*keep_types* only, regardless of *keep_unmatched*) —
       semantically important terms that survive all matching attempts are retained
       rather than discarded; their edges still carry graph structure.
    5. **Discard** — terms in *DISCARD_TYPES* or outside *keep_types* with no match.
       Generic English words mis-labelled by the NLP classifier fall here.

    The type-conditional design means that ``"modality"``/``"method"``/``"metric"``
    terms use exact matching only (the right call — their unmatched residuals are
    generic clinical vocabulary), while ``"disorder"``/``"anatomical_region"``/
    ``"cognitive_construct"``/``"network"``/``"intervention"`` get the full
    treatment including substring matching and nlp_ fallback.

    Parameters
    ----------
    nlp_nodes_df:
        NLP KG nodes with ``node_id`` (term string) and ``node_type`` columns.
    mesh_lookup:
        ``{normalized_string → MeSH UI}`` from :func:`build_mesh_synonym_lookup`.
    cogat_name_lookup:
        ``{normalized_name → canonical_id}`` from the CogAtlas node table.
    keep_unmatched:
        Fallback for types *not* in *keep_types*: if True, keep as ``nlp_`` node
        instead of discarding.  Default False (conservative for noisy types).
    keep_types:
        Set of ``node_type`` values that receive substring matching and nlp_
        fallback.  Defaults to :data:`KEEP_TYPES`.  Pass ``None`` to disable
        type-conditional logic entirely (equivalent to old ``keep_unmatched``
        behaviour).
    min_substring_len:
        Minimum character length for substring matching (default 6).
    mesh_primary_names:
        Set of normalized *primary* descriptor names (i.e.
        ``{_norm(row["name"]) for _, row in mesh_descriptors_df.iterrows()}``).
        When provided, substring matching prefers hits on primary names over
        hits on synonyms — avoids e.g. ``"alzheimer"`` resolving to
        ``Amyloid beta-Peptides`` (via a synonym) instead of
        ``Alzheimer Disease`` (its primary name).

    Returns
    -------
    nlp_to_canonical : dict[nlp_term → canonical_id]
    merge_log : list[dict]
        One record per NLP term: ``entity``, ``from_source``, ``canonical_id``,
        ``match_type``, ``node_type``.
    """
    nlp_to_canonical: dict[str, str] = {}
    merge_log: list[dict] = []

    counters: dict[str, int] = {
        "mesh_synonym_match": 0,
        "cogat_name_match": 0,
        "mesh_substring_match": 0,
        "new_nlp_node": 0,
        "discarded": 0,
    }

    # Build word index once, only when substring matching is needed
    word_index = (
        _build_mesh_word_index(mesh_lookup, primary_names=mesh_primary_names)
        if keep_types else {}
    )

    for _, row in nlp_nodes_df.iterrows():
        term = str(row["node_id"])
        node_type = str(row.get("node_type", "")) if hasattr(row, "get") else str(row["node_type"])
        key = _norm(term)

        canonical_id: Optional[str] = None
        match_type: str = "discarded"

        # Step 1: exact MeSH match
        if key and key in mesh_lookup:
            canonical_id = mesh_lookup[key]
            match_type = "mesh_synonym_match"

        # Step 2: exact CogAtlas match
        elif key and key in cogat_name_lookup:
            canonical_id = cogat_name_lookup[key]
            match_type = "cogat_name_match"

        # Step 3 & 4: type-conditional — for semantically important types,
        # try substring matching then fall back to keeping as nlp_ node
        elif keep_types and node_type in keep_types:
            ui = _try_substring_match(key, word_index, min_len=min_substring_len)
            if ui is not None:
                canonical_id = ui
                match_type = "mesh_substring_match"
            else:
                # Still no match — apply secondary per-type filter before keeping.
                # Catches functional states / modality terms the NLP labeller
                # misclassified as a semantically important type (e.g. "brain
                # activity" labelled as anatomical_region).
                filter_pat = NLP_NODE_FILTERS.get(node_type)
                if filter_pat is not None and filter_pat.search(key):
                    match_type = "discarded"
                else:
                    # Genuine unmapped concept — keep as nlp_ node
                    slug = re.sub(r"\s+", "_", key)[:64]
                    canonical_id = f"nlp_{slug}"
                    match_type = "new_nlp_node"

        # Step 5: for noisy types, fall back to keep_unmatched flag
        elif keep_unmatched:
            slug = re.sub(r"\s+", "_", key)[:64]
            canonical_id = f"nlp_{slug}"
            match_type = "new_nlp_node"

        counters[match_type] += 1

        if match_type == "discarded":
            merge_log.append({
                "entity": term, "node_type": node_type,
                "from_source": "nlp", "canonical_id": None, "match_type": "discarded",
            })
            continue

        nlp_to_canonical[term] = canonical_id  # type: ignore[assignment]
        merge_log.append({
            "entity": term, "node_type": node_type,
            "from_source": "nlp", "canonical_id": canonical_id, "match_type": match_type,
        })

    total = sum(counters.values())
    logger.info(
        "NLP normalization (%d terms): %d exact MeSH, %d substring MeSH, "
        "%d CogAtlas, %d new nlp_ nodes, %d discarded",
        total,
        counters["mesh_synonym_match"],
        counters["mesh_substring_match"],
        counters["cogat_name_match"],
        counters["new_nlp_node"],
        counters["discarded"],
    )
    return nlp_to_canonical, merge_log


# ---------------------------------------------------------------------------
# Master entity table
# ---------------------------------------------------------------------------

def build_master_entity_table(
    mesh_nodes_df: pd.DataFrame,
    cogat_nodes_df: pd.DataFrame,
    nlp_nodes_df: pd.DataFrame,
    nlp_to_canonical: dict[str, str],
) -> pd.DataFrame:
    """Assemble the master canonical entity table.

    MeSH nodes form the backbone.  CogAtlas adds new ``cogat_`` nodes only
    (concepts that mapped to an existing MeSH node annotate it as an
    additional source but don't add a new row).  NLP only annotates existing
    nodes as an additional source — it never adds new rows in conservative mode.

    Parameters
    ----------
    mesh_nodes_df:
        MeSH KG nodes with ``node_id``, ``name``, ``node_type``.
    cogat_nodes_df:
        Output of :func:`build_cogatlas_kg`: ``canonical_id``, ``name``,
        ``node_type``, ``source``, ``trm_id``, ``match_type``.
    nlp_nodes_df:
        NLP KG nodes (used only to look up node_type for nlp_ nodes when
        *keep_unmatched_nlp* was True).
    nlp_to_canonical:
        ``{nlp_term → canonical_id}`` from :func:`normalize_nlp_entities`.

    Returns
    -------
    pd.DataFrame with columns:
        ``canonical_id``, ``name``, ``node_type``, ``primary_source``,
        ``sources``  (comma-separated list of contributing KGs).
    """
    master: dict[str, dict] = {}

    # 1 — MeSH backbone
    for _, row in mesh_nodes_df.iterrows():
        cid = str(row["node_id"])
        master[cid] = {
            "canonical_id": cid,
            "name": str(row["name"]),
            "node_type": str(row["node_type"]),
            "primary_source": "mesh",
            "sources": {"mesh"},
        }

    # 2 — CogAtlas: new nodes only; mapped ones annotate an existing MeSH node
    for _, row in cogat_nodes_df.iterrows():
        cid = str(row["canonical_id"])
        if cid not in master:
            master[cid] = {
                "canonical_id": cid,
                "name": str(row["name"]),
                "node_type": str(row["node_type"]),
                "primary_source": "cogatlas",
                "sources": {"cogatlas"},
            }
        else:
            master[cid]["sources"].add("cogatlas")

    # 3 — NLP: annotate existing nodes; add nlp_ nodes if keep_unmatched was True
    nlp_type_map: dict[str, str] = {}
    for _, row in nlp_nodes_df.iterrows():
        nlp_type_map[str(row["node_id"])] = str(row.get("node_type", "unknown"))

    for nlp_term, cid in nlp_to_canonical.items():
        if cid in master:
            master[cid]["sources"].add("nlp")
        else:
            # nlp_ node (keep_unmatched mode)
            master[cid] = {
                "canonical_id": cid,
                "name": nlp_term,
                "node_type": nlp_type_map.get(nlp_term, "unknown"),
                "primary_source": "nlp",
                "sources": {"nlp"},
            }

    # Serialise the sources set to a sorted comma-separated string for parquet
    for entry in master.values():
        srcs = entry["sources"]
        entry["sources"] = ",".join(sorted(srcs))

    df = pd.DataFrame(list(master.values()))
    logger.info("Master entity table: %d canonical entities", len(df))
    return df


# ---------------------------------------------------------------------------
# Edge remapping
# ---------------------------------------------------------------------------

def remap_edges(
    edges_df: pd.DataFrame,
    id_mapping: dict[str, str],
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Re-point ``subject_id`` / ``object_id`` to canonical IDs.

    Parameters
    ----------
    edges_df:
        Edge DataFrame with ``subject_id`` and ``object_id`` columns.
    id_mapping:
        ``{old_id → canonical_id}``.  For NLP edges, this is
        ``nlp_term → canonical_id``.  For MeSH/CogAtlas edges that already
        use canonical IDs, pass an identity map.
    drop_missing:
        If True (default), drop edges where either endpoint is absent from
        *id_mapping*.  If False, retain the original ID for unmapped endpoints.

    Returns
    -------
    pd.DataFrame with remapped IDs and self-loops removed.
    """
    df = edges_df.copy()
    mapped_subject = df["subject_id"].map(id_mapping)
    mapped_object = df["object_id"].map(id_mapping)

    if drop_missing:
        df["subject_id"] = mapped_subject
        df["object_id"] = mapped_object
        before = len(df)
        df = df.dropna(subset=["subject_id", "object_id"])
        dropped = before - len(df)
        if dropped:
            logger.debug("remap_edges: dropped %d edges with unmapped endpoints", dropped)
    else:
        df["subject_id"] = mapped_subject.fillna(df["subject_id"])
        df["object_id"] = mapped_object.fillna(df["object_id"])

    # Remove self-loops introduced by merging two concepts into one canonical ID
    df = df[df["subject_id"] != df["object_id"]]
    df = df.drop_duplicates(subset=["subject_id", "relation_type", "object_id"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_normalization(
    mesh_descriptors_df: pd.DataFrame,
    mesh_nodes_df: pd.DataFrame,
    mesh_edges_df: pd.DataFrame,
    nlp_nodes_df: pd.DataFrame,
    nlp_edges_df: pd.DataFrame,
    cogatlas_out_path: Optional[str | Path] = None,
    cogatlas_edges_df: Optional[pd.DataFrame] = None,
    keep_unmatched_nlp: bool = False,
    keep_types: Optional[frozenset[str] | set[str]] = KEEP_TYPES,
    min_substring_len: int = 6,
    out_dir: Optional[str | Path] = None,
) -> dict:
    """Run the full entity normalization pipeline (Phase 2, Step 5).

    Parameters
    ----------
    mesh_descriptors_df:
        Full MeSH descriptor table with ``ui``, ``name``, ``synonyms``
        (from :func:`neurovlm.gnn.mesh.parse_mesh_descriptor_xml`).
    mesh_nodes_df:
        MeSH KG nodes (``node_id`` = DescriptorUI, ``name``, ``node_type``).
    mesh_edges_df:
        MeSH KG edges (already use MeSH UIs — returned unchanged).
    nlp_nodes_df, nlp_edges_df:
        NLP KG tables from ``experiments/data/nlp_kg/``.
    cogatlas_out_path:
        Path to save/load the CogAtlas concept list parquet.  Fetches from
        ``cognitiveatlas.org`` when the file does not exist.
    cogatlas_edges_df:
        Optional CogAtlas relationship edges from ``parse_cogatlas`` in
        ``cogatlas.py``.  Columns: ``parent``, ``child``, ``relationship``.
    keep_unmatched_nlp:
        Fallback for types not in *keep_types*: keep as ``nlp_`` instead of
        discarding.  Default False (noisy types are discarded on no match).
    keep_types:
        Node types that receive substring matching and nlp_ fallback.
        Defaults to :data:`KEEP_TYPES`.
    min_substring_len:
        Minimum character length for substring matching (default 6).
    out_dir:
        Directory to write output parquet files and the merge log JSON.

    Returns
    -------
    dict with keys:

    ``master_entities``
        Canonical entity table (pd.DataFrame).
    ``unified_edges``
        All edges from all three KGs merged, endpoints re-pointed to canonical
        IDs (pd.DataFrame).
    ``cogat_nodes``
        CogAtlas node table including match metadata (pd.DataFrame).
    ``merge_log``
        Full audit log: one dict per mapped/discarded entity (list[dict]).
    """
    # ------------------------------------------------------------------
    # Step 1: MeSH synonym lookup
    # ------------------------------------------------------------------
    logger.info("Step 1 — Building MeSH synonym lookup...")
    mesh_lookup = build_mesh_synonym_lookup(mesh_descriptors_df)

    # ------------------------------------------------------------------
    # Step 2: CogAtlas KG
    # ------------------------------------------------------------------
    logger.info("Step 2 — Fetching/loading CogAtlas concepts...")
    cogatlas_df = fetch_cogatlas_concepts(out_path=cogatlas_out_path)

    cogat_nodes, cogat_edges, cogat_to_canonical = build_cogatlas_kg(
        cogatlas_df, mesh_lookup, cogatlas_edges_df
    )

    # CogAtlas name → canonical_id lookup for NLP matching
    cogat_name_lookup: dict[str, str] = {
        _norm(str(row["name"])): str(row["canonical_id"])
        for _, row in cogat_nodes.iterrows()
        if _norm(str(row["name"]))
    }

    # ------------------------------------------------------------------
    # Step 3: NLP normalization
    # ------------------------------------------------------------------
    logger.info("Step 3 — Normalizing NLP entities...")
    primary_names = frozenset(
        _norm(str(row["name"])) for _, row in mesh_descriptors_df.iterrows()
    )
    nlp_to_canonical, nlp_merge_log = normalize_nlp_entities(
        nlp_nodes_df, mesh_lookup, cogat_name_lookup,
        keep_unmatched=keep_unmatched_nlp,
        keep_types=keep_types,
        min_substring_len=min_substring_len,
        mesh_primary_names=primary_names,
    )

    # ------------------------------------------------------------------
    # Step 4: Master entity table
    # ------------------------------------------------------------------
    logger.info("Step 4 — Building master entity table...")
    master_entities = build_master_entity_table(
        mesh_nodes_df, cogat_nodes, nlp_nodes_df, nlp_to_canonical
    )

    # ------------------------------------------------------------------
    # Step 5: Remap edges to canonical IDs
    # ------------------------------------------------------------------
    logger.info("Step 5 — Remapping edges...")

    # MeSH edges: already use canonical MeSH UIs — identity map
    mesh_id_map = {str(row["node_id"]): str(row["node_id"]) for _, row in mesh_nodes_df.iterrows()}
    unified_mesh_edges = remap_edges(mesh_edges_df, mesh_id_map, drop_missing=False)
    unified_mesh_edges = unified_mesh_edges.copy()
    unified_mesh_edges["source_kg"] = "mesh"

    # CogAtlas edges: already use canonical IDs after build_cogatlas_kg
    edge_parts = [unified_mesh_edges]
    if len(cogat_edges) > 0:
        cogat_id_map = {str(row["canonical_id"]): str(row["canonical_id"]) for _, row in cogat_nodes.iterrows()}
        unified_cogat_edges = remap_edges(cogat_edges, cogat_id_map, drop_missing=False)
        unified_cogat_edges = unified_cogat_edges.copy()
        unified_cogat_edges["source_kg"] = "cogatlas"
        edge_parts.append(unified_cogat_edges)

    # NLP edges: re-point NLP term → canonical ID; drop if either end unmatched
    if len(nlp_to_canonical) > 0:
        unified_nlp_edges = remap_edges(nlp_edges_df, nlp_to_canonical, drop_missing=True)
        unified_nlp_edges = unified_nlp_edges.copy()
        unified_nlp_edges["source_kg"] = "nlp"
        edge_parts.append(unified_nlp_edges)

    unified_edges = pd.concat(edge_parts, ignore_index=True)
    # Final de-duplication across KGs (same semantic edge may appear in >1 source)
    unified_edges = unified_edges.drop_duplicates(
        subset=["subject_id", "relation_type", "object_id", "source"]
    )
    logger.info(
        "Unified graph: %d entities, %d edges (%d relation types)",
        len(master_entities),
        len(unified_edges),
        unified_edges["relation_type"].nunique(),
    )

    # ------------------------------------------------------------------
    # Merge log
    # ------------------------------------------------------------------
    cogatlas_merge_log: list[dict] = [
        {
            "entity": str(row["name"]),
            "trm_id": str(row["trm_id"]),
            "from_source": "cogatlas",
            "canonical_id": str(row["canonical_id"]),
            "match_type": str(row["match_type"]),
        }
        for _, row in cogat_nodes.iterrows()
    ]
    full_merge_log = cogatlas_merge_log + nlp_merge_log

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        master_entities.to_parquet(out_dir / "unified_kg_nodes.parquet", index=False)
        unified_edges.to_parquet(out_dir / "unified_kg_edges.parquet", index=False)
        cogat_nodes.to_parquet(out_dir / "cogat_kg_nodes.parquet", index=False)

        with open(out_dir / "merge_log.json", "w") as fh:
            json.dump(full_merge_log, fh, indent=2)

        logger.info("All outputs saved to %s", out_dir)

    return {
        "master_entities": master_entities,
        "unified_edges": unified_edges,
        "cogat_nodes": cogat_nodes,
        "merge_log": full_merge_log,
    }
