"""Qualifier-based MeSH edge typing — NLP graph rebuild, Step 1.

Extracts semantically-typed edges from MeSH annotation qualifiers.
When two MeSH terms co-occur in the same paper and at least one carries a
qualifier in QUALIFIER_TO_RELATION, this module emits a typed relation
(implicated_in, associated_with_disorder, used_in, treated_by) instead of
the generic co_occurs_with.

Typical usage
-------------
>>> import pandas as pd
>>> from neurovlm.gnn.mesh_re import (
...     QUALIFIER_TO_RELATION,
...     build_qualifier_name_lookup,
...     extract_qualifier_edges,
... )
>>> descriptors = pd.read_parquet("data/mesh_kg/mesh_descriptors.parquet")
>>> name_to_ui = build_qualifier_name_lookup(descriptors)
>>> annotations = pd.read_parquet("data/mesh_kg/mesh_annotations_long.parquet")
>>> kg_nodes = pd.read_parquet("data/unified_kg/unified_kg_nodes.parquet")
>>> kg_node_ids = set(kg_nodes["canonical_id"])
>>> edges = extract_qualifier_edges(annotations, name_to_ui, kg_node_ids)
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Qualifier → relation type mapping
# ---------------------------------------------------------------------------

QUALIFIER_TO_RELATION: dict[str, str] = {
    # implicated_in: physiological/biological involvement
    "physiology": "implicated_in",
    "physiopathology": "implicated_in",
    "pathology": "implicated_in",
    "genetics": "implicated_in",
    "metabolism": "implicated_in",
    "psychology": "implicated_in",
    "drug effects": "implicated_in",
    "pharmacology": "implicated_in",
    "immunology": "implicated_in",
    # associated_with_disorder: clinical/diagnostic relationship
    "diagnosis": "associated_with_disorder",
    "diagnostic imaging": "associated_with_disorder",
    "complications": "associated_with_disorder",
    "etiology": "associated_with_disorder",
    "adverse effects": "associated_with_disorder",
    "epidemiology": "associated_with_disorder",
    # used_in: method/tool applied to a domain
    "methods": "used_in",
    "instrumentation": "used_in",
    # treated_by: intervention for a condition
    "drug therapy": "treated_by",
    "therapeutic use": "treated_by",
    "therapy": "treated_by",
    "surgery": "treated_by",
}


# ---------------------------------------------------------------------------
# Lookup builder
# ---------------------------------------------------------------------------

def build_qualifier_name_lookup(descriptors_df: pd.DataFrame) -> dict[str, str]:
    """Build a lowercase name → DescriptorUI lookup from mesh_descriptors.parquet.

    Parameters
    ----------
    descriptors_df:
        DataFrame with columns ``ui`` and ``name``.

    Returns
    -------
    dict mapping ``lower(name)`` → ``ui``.
    """
    return dict(
        zip(
            descriptors_df["name"].str.lower().str.strip(),
            descriptors_df["ui"],
        )
    )


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_qualifier_edges(
    annotations_df: pd.DataFrame,
    name_to_ui: dict[str, str],
    kg_node_ids: set[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract typed edges from MeSH qualifier annotations.

    For each paper (PMID), identifies terms that carry a qualifier mapping to
    a typed relation and pairs them bidirectionally with all other terms in the
    same paper.

    Parameters
    ----------
    annotations_df:
        Long-form MeSH annotations with columns ``pmid`` and ``mesh_term``.
        ``mesh_term`` may include a qualifier suffix, e.g. ``"Brain/physiology"``.
    name_to_ui:
        ``{lower_name: descriptor_ui}`` from :func:`build_qualifier_name_lookup`.
    kg_node_ids:
        Set of canonical IDs in the unified KG; edges where either endpoint is
        absent are dropped.
    verbose:
        Print progress and timing information.

    Returns
    -------
    pandas.DataFrame with columns matching unified_kg_edges schema:
        ``subject_id``, ``relation_type``, ``object_id``, ``source``,
        ``weight``, ``source_kg``.
    Weight is the number of distinct PMIDs supporting each edge.
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Parse base_term and qualifier (vectorised)
    # ------------------------------------------------------------------
    df = annotations_df[["pmid", "mesh_term"]].copy()
    split = df["mesh_term"].str.split("/", n=1, expand=True)
    df["base_term"] = split[0].str.strip()
    df["qualifier"] = split[1].str.strip().str.lower() if split.shape[1] > 1 else None
    # fill None where no '/' present
    if split.shape[1] < 2:
        df["qualifier"] = None
    else:
        df["qualifier"] = df["qualifier"].where(split[1].notna(), other=None)

    if verbose:
        n_with_qual = df["qualifier"].notna().sum()
        print(f"Total annotation rows : {len(df):>12,}")
        print(f"Rows with qualifier   : {n_with_qual:>12,}  ({100*n_with_qual/len(df):.1f}%)")

    # ------------------------------------------------------------------
    # Step 2: Map base_term → canonical_id (DescriptorUI)
    # ------------------------------------------------------------------
    lower_base = df["base_term"].str.lower()
    df["canonical_id"] = lower_base.map(name_to_ui)

    n_mapped = df["canonical_id"].notna().sum()
    if verbose:
        print(f"Terms mapped to UI    : {n_mapped:>12,}  ({100*n_mapped/len(df):.1f}%)")

    # Drop rows where base_term has no DescriptorUI
    df = df[df["canonical_id"].notna()].copy()

    # ------------------------------------------------------------------
    # Step 3: Map qualifier → relation_type (vectorised)
    # ------------------------------------------------------------------
    qual_series = pd.Series(QUALIFIER_TO_RELATION, name="relation_type")
    df["relation_type"] = df["qualifier"].map(qual_series)

    # Separate typed (has a mapped relation) from untyped rows
    typed_df = df[df["relation_type"].notna()][
        ["pmid", "canonical_id", "relation_type"]
    ].copy()
    all_terms_df = df[["pmid", "canonical_id"]].drop_duplicates()

    if verbose:
        print(f"Typed term rows       : {len(typed_df):>12,}")
        print(f"Unique (pmid, term)   : {len(all_terms_df):>12,}")

    # ------------------------------------------------------------------
    # Step 4: Merge typed terms with all terms in the same paper
    # ------------------------------------------------------------------
    # typed_df: (pmid, subject canonical_id, relation_type)
    # all_terms_df: (pmid, object canonical_id)
    merged = typed_df.merge(
        all_terms_df.rename(columns={"canonical_id": "object_id"}),
        on="pmid",
        how="inner",
    )
    merged = merged.rename(columns={"canonical_id": "subject_id"})

    # Drop self-loops
    merged = merged[merged["subject_id"] != merged["object_id"]]

    if verbose:
        print(f"Raw edge pairs        : {len(merged):>12,}")

    # ------------------------------------------------------------------
    # Step 5: Add reverse direction (bidirectional edges)
    # ------------------------------------------------------------------
    forward = merged[["pmid", "subject_id", "relation_type", "object_id"]].copy()
    reverse = merged[["pmid", "object_id", "relation_type", "subject_id"]].copy()
    reverse.columns = ["pmid", "subject_id", "relation_type", "object_id"]

    all_edges = pd.concat([forward, reverse], ignore_index=True)
    # Drop self-loops that may appear in reverse (shouldn't, but defensive)
    all_edges = all_edges[all_edges["subject_id"] != all_edges["object_id"]]

    # ------------------------------------------------------------------
    # Step 6: Filter to KG nodes
    # ------------------------------------------------------------------
    mask = (
        all_edges["subject_id"].isin(kg_node_ids)
        & all_edges["object_id"].isin(kg_node_ids)
    )
    all_edges = all_edges[mask]

    if verbose:
        print(f"After KG node filter  : {len(all_edges):>12,}")

    # ------------------------------------------------------------------
    # Step 7: Aggregate — count distinct PMIDs per (subject, rel, object)
    # ------------------------------------------------------------------
    edges = (
        all_edges.groupby(["subject_id", "relation_type", "object_id"], sort=False)["pmid"]
        .nunique()
        .reset_index()
        .rename(columns={"pmid": "weight"})
    )
    edges["weight"] = edges["weight"].astype(float)

    # ------------------------------------------------------------------
    # Step 8: Add provenance columns
    # ------------------------------------------------------------------
    edges["source"] = "mesh_qualifier"
    edges["source_kg"] = "mesh_qualifier"

    # Reorder to match unified_kg_edges schema
    edges = edges[
        ["subject_id", "relation_type", "object_id", "source", "weight", "source_kg"]
    ].reset_index(drop=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nFinal edge count      : {len(edges):>12,}")
        print(f"Elapsed               : {elapsed:.1f}s")

    return edges
