"""LLM-based relation extraction from abstracts — NLP graph rebuild, Step 2.

Extracts semantically-typed edges by prompting a local Ollama model with
paper abstracts and their MeSH term lists.  Only edges whose endpoints are
already present in unified_kg_nodes are emitted (no new nodes).

Typical usage
-------------
>>> from neurovlm.gnn.llm_re import (
...     RELATION_TYPES, SYSTEM_PROMPT,
...     build_user_prompt, call_ollama, validate_triple,
...     process_paper, aggregate_results,
... )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELATION_TYPES: frozenset[str] = frozenset(
    {
        "implicated_in",
        "associated_with_disorder",
        "treated_by",
        "used_in",
        "co_activates_with",
        "expressed_in",
    }
)

SYSTEM_PROMPT: str = """\
You are a biomedical knowledge graph expert. Given a scientific abstract \
and a list of MeSH terms that annotate that paper, identify explicit \
semantic relationships between pairs of those terms as stated or strongly \
implied by the abstract.

Output ONLY a JSON array of objects. Each object must have exactly these keys:
  "subject": the subject term (must be one of the provided MeSH terms)
  "relation": one of: implicated_in, associated_with_disorder, treated_by, \
used_in, co_activates_with, expressed_in
  "object": the object term (must be one of the provided MeSH terms)

Rules:
- Both subject and object must come from the provided MeSH term list.
- Only output relations that are clearly supported by the abstract text.
- If no relation fits, output an empty array [].
- Do not use any relation type not in the list above.
- Do not output co_occurs_with.
- A term can appear in multiple triples.

Relation definitions:
  implicated_in:            a brain region, process, or gene is functionally \
involved in a cognitive function or disorder
  associated_with_disorder: a concept is clinically associated with a disorder
  treated_by:               a disorder is treated by a drug or intervention
  used_in:                  a method or tool is used to study a concept
  co_activates_with:        two brain regions co-activate together
  expressed_in:             a gene or molecule is expressed in a brain region\
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_user_prompt(abstract: str, mesh_terms: list[str]) -> str:
    """Build the per-paper user prompt for the LLM.

    Parameters
    ----------
    abstract:
        Full abstract text; truncated to 1 500 characters internally.
    mesh_terms:
        List of base MeSH term names (no qualifier suffixes) that are
        already present in the unified KG.

    Returns
    -------
    str prompt ready to pass as the user message to Ollama.
    """
    truncated = abstract[:1500]
    terms_block = "\n".join(f"- {t}" for t in mesh_terms)
    return (
        f"Abstract:\n{truncated}\n\n"
        f"MeSH terms for this paper:\n{terms_block}\n\n"
        "Output the JSON array of relation triples:"
    )


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------


def call_ollama(
    user_prompt: str,
    model: str = "qwen2.5:7b-instruct",
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Call a local Ollama model and return parsed relation triples.

    Uses ``format="json"`` to request structured output.  Any parse error
    or unexpected response shape returns an empty list rather than raising.

    Parameters
    ----------
    user_prompt:
        The per-paper user message built by :func:`build_user_prompt`.
    model:
        Ollama model name.  Defaults to ``qwen2.5:7b-instruct``.
    system_prompt:
        System message; defaults to :data:`SYSTEM_PROMPT`.

    Returns
    -------
    list of dicts, each with keys ``"subject"``, ``"relation"``, ``"object"``.
    Returns ``[]`` on any error.
    """
    try:
        import ollama  # local import so the module is importable without ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        raw = response["message"]["content"]
    except Exception as exc:
        logger.warning("Ollama call failed: %s", exc)
        return []

    return _parse_triples(raw)


def _parse_triples(raw: str) -> list[dict]:
    """Parse the raw JSON string from the LLM into a list of triple dicts."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first JSON array substring
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.debug("No JSON array found in LLM output")
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            logger.debug("Failed to parse extracted JSON substring")
            return []

    if not isinstance(data, list):
        # Some models wrap in {"triples": [...]} — unwrap one level
        if isinstance(data, dict):
            for key in ("triples", "relations", "results", "edges"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                logger.debug("LLM returned a dict without a known list key")
                return []
        else:
            return []

    triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if {"subject", "relation", "object"} <= item.keys():
            triples.append(
                {
                    "subject": str(item["subject"]).strip(),
                    "relation": str(item["relation"]).strip(),
                    "object": str(item["object"]).strip(),
                }
            )
    return triples


# ---------------------------------------------------------------------------
# Triple validation
# ---------------------------------------------------------------------------


def validate_triple(
    triple: dict,
    valid_terms: set[str],
    valid_relations: frozenset[str] = RELATION_TYPES,
) -> bool:
    """Return True if the triple is well-formed and both endpoints are valid.

    Parameters
    ----------
    triple:
        Dict with keys ``"subject"``, ``"relation"``, ``"object"``.
    valid_terms:
        Set of MeSH term names (original case) for this specific paper.
    valid_relations:
        Allowed relation type strings; defaults to :data:`RELATION_TYPES`.

    Returns
    -------
    bool — True only if all three fields are present, non-empty, the
    relation is in *valid_relations*, both endpoints are in *valid_terms*,
    and subject ≠ object.
    """
    subject = triple.get("subject", "")
    relation = triple.get("relation", "")
    obj = triple.get("object", "")

    if not (subject and relation and obj):
        return False
    if relation not in valid_relations:
        return False
    if subject not in valid_terms or obj not in valid_terms:
        return False
    if subject == obj:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-paper pipeline
# ---------------------------------------------------------------------------


def process_paper(
    pmid: str,
    abstract: str,
    mesh_terms: list[str],
    name_to_ui: dict[str, str],
    kg_node_ids: set[str],
    checkpoint_dir: Path,
    model: str = "qwen2.5:7b-instruct",
) -> list[dict]:
    """Run the full extraction pipeline for one paper.

    Checks for an existing checkpoint file first.  If found, loads and
    returns those triples without calling the LLM.  Otherwise calls the
    LLM, validates the output, writes the checkpoint, and returns the
    canonical triples.

    Parameters
    ----------
    pmid:
        Paper identifier (string); used as the checkpoint filename.
    abstract:
        Full abstract text for the paper.
    mesh_terms:
        Base MeSH term names (no qualifiers) already present in the KG.
    name_to_ui:
        Lowercase term name → DescriptorUI lookup from mesh_descriptors.
    kg_node_ids:
        Set of canonical_ids in unified_kg_nodes.
    checkpoint_dir:
        Directory where per-paper JSON result files are stored.
    model:
        Ollama model name.

    Returns
    -------
    List of dicts with keys ``"subject_id"``, ``"relation_type"``,
    ``"object_id"`` (canonical IDs, not term names).
    """
    checkpoint_path = checkpoint_dir / f"{pmid}.json"

    # ---- resume from checkpoint ----
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open() as fh:
                raw_triples = json.load(fh)
            return _to_canonical(raw_triples, name_to_ui, kg_node_ids)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt checkpoint for %s — reprocessing", pmid)

    # ---- build and call LLM ----
    if not abstract or not mesh_terms:
        _write_checkpoint(checkpoint_path, [])
        return []

    valid_terms: set[str] = set(mesh_terms)
    user_prompt = build_user_prompt(abstract, mesh_terms)
    raw_triples = call_ollama(user_prompt, model=model)

    # ---- validate ----
    valid_raw = [
        t for t in raw_triples if validate_triple(t, valid_terms, RELATION_TYPES)
    ]

    # ---- write checkpoint ----
    _write_checkpoint(checkpoint_path, valid_raw)

    return _to_canonical(valid_raw, name_to_ui, kg_node_ids)


def _write_checkpoint(path: Path, triples: list[dict]) -> None:
    """Write validated raw triples (term names) to a checkpoint JSON file."""
    try:
        with path.open("w") as fh:
            json.dump(triples, fh)
    except OSError as exc:
        logger.warning("Could not write checkpoint %s: %s", path, exc)


def _to_canonical(
    raw_triples: list[dict],
    name_to_ui: dict[str, str],
    kg_node_ids: set[str],
) -> list[dict]:
    """Convert raw {subject, relation, object} name triples to canonical IDs.

    Drops triples where either endpoint cannot be mapped to a KG node ID.
    """
    canonical = []
    for t in raw_triples:
        subj_id = name_to_ui.get(t.get("subject", "").lower().strip())
        obj_id = name_to_ui.get(t.get("object", "").lower().strip())
        rel = t.get("relation", "")

        if not subj_id or not obj_id:
            continue
        if subj_id not in kg_node_ids or obj_id not in kg_node_ids:
            continue
        if subj_id == obj_id:
            continue
        if rel not in RELATION_TYPES:
            continue

        canonical.append(
            {"subject_id": subj_id, "relation_type": rel, "object_id": obj_id}
        )
    return canonical


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    checkpoint_dir: Path,
    name_to_ui: dict[str, str],
    existing_edges: frozenset,
    kg_node_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Load all checkpoint files and build the final edge DataFrame.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing per-paper ``{pmid}.json`` checkpoint files.
    name_to_ui:
        Lowercase term name → DescriptorUI lookup.
    existing_edges:
        frozenset of ``(subject_id, relation_type, object_id)`` tuples from
        Step 1; matched triples are excluded from the output.
    kg_node_ids:
        If provided, filters out any edges whose endpoints are not in this
        set (defensive check; should already be guaranteed by process_paper).

    Returns
    -------
    pandas.DataFrame with columns matching unified_kg_edges schema:
        ``subject_id``, ``relation_type``, ``object_id``,
        ``source``, ``weight``, ``source_kg``.
    Weight is the number of distinct papers supporting each edge.
    """
    rows: list[dict] = []
    checkpoints = sorted(checkpoint_dir.glob("*.json"))

    for cp in checkpoints:
        pmid = cp.stem
        try:
            with cp.open() as fh:
                raw_triples = json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping corrupt checkpoint: %s", cp)
            continue

        canonical = _to_canonical(raw_triples, name_to_ui, kg_node_ids or set())
        for t in canonical:
            t["pmid"] = pmid
            rows.append(t)

    if not rows:
        return pd.DataFrame(
            columns=["subject_id", "relation_type", "object_id", "source", "weight", "source_kg"]
        )

    df = pd.DataFrame(rows)

    # Aggregate: weight = number of distinct pmids per (subj, rel, obj)
    edges = (
        df.groupby(["subject_id", "relation_type", "object_id"], sort=False)["pmid"]
        .nunique()
        .reset_index()
        .rename(columns={"pmid": "weight"})
    )
    edges["weight"] = edges["weight"].astype(float)

    # Drop self-loops (defensive)
    edges = edges[edges["subject_id"] != edges["object_id"]]

    # Deduplicate against Step 1
    mask = edges.apply(
        lambda r: (r["subject_id"], r["relation_type"], r["object_id"])
        not in existing_edges,
        axis=1,
    )
    edges = edges[mask]

    # Add provenance
    edges["source"] = "llm_re"
    edges["source_kg"] = "llm_re"

    edges = edges[
        ["subject_id", "relation_type", "object_id", "source", "weight", "source_kg"]
    ].reset_index(drop=True)

    return edges
