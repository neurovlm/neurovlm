"""Build PubMed ALE-map JSONL examples with clean multi-positive text targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from atlas_free_cnn.data_building.definitions import (
    ALLOWED_PUBMED_MESH_CATEGORIES,
    POSITIVE_WEIGHTS,
    normalize_key,
    text_pair,
)
from atlas_free_cnn.data_building.preprocessing import nifti_metadata
from atlas_free_cnn.data_building.text_registry import write_jsonl


def _load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _read_table(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_pubmed_metadata(paths: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from neurovlm.retrieval_resources import _load_pubmed_dataframe, _load_pubmed_summaries_dataframe

        pubmed = _load_pubmed_dataframe().copy()
        summaries = _load_pubmed_summaries_dataframe().copy()
    except Exception:
        pubmed = pd.DataFrame()
        summaries = _read_table(paths.get("pubmed_summaries"))
    if pubmed.empty:
        try:
            from neurovlm.retrieval_resources import _load_pubmed_coordinates

            coords = _load_pubmed_coordinates()
            pubmed = pd.DataFrame({"pmid": sorted(coords["pmid"].astype(str).unique())})
        except Exception:
            pubmed = pd.DataFrame()
    for df in (pubmed, summaries):
        if not df.empty and "pmid" in df.columns:
            df["pmid"] = df["pmid"].astype(str)
    return pubmed, summaries


def load_mesh_resources(paths: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    annotations_path = Path(paths.get("pubmed_mesh_annotations", ""))
    pmid_mesh: dict[str, list[str]] = {}
    if annotations_path.exists():
        with annotations_path.open() as f:
            pmid_mesh = {str(k): list(v or []) for k, v in json.load(f).items()}

    definition_lookup: dict[str, str] = {}
    nodes = _read_table(paths.get("pubmed_mesh_definitions"))
    if not nodes.empty:
        name_cols = [c for c in ("name", "term", "mesh_term", "label") if c in nodes.columns]
        def_cols = [c for c in ("definition", "description", "scope_note", "text") if c in nodes.columns]
        if name_cols and def_cols:
            for _, row in nodes.iterrows():
                name = str(row[name_cols[0]]).strip()
                definition = str(row[def_cols[0]]).strip()
                if name and definition and definition.lower() != "nan":
                    definition_lookup[name] = definition
                    definition_lookup[name.lower()] = definition

    node_type_lookup: dict[str, str] = {}
    node_df = _read_table(paths.get("pubmed_mesh_nodes"))
    if not node_df.empty:
        name_cols = [c for c in ("name", "term", "mesh_term", "label") if c in node_df.columns]
        type_cols = [c for c in ("node_type", "semantic_type", "category", "type") if c in node_df.columns]
        if name_cols and type_cols:
            for _, row in node_df.iterrows():
                name = str(row[name_cols[0]]).strip()
                node_type = str(row[type_cols[0]]).strip()
                if name and node_type:
                    node_type_lookup[name] = node_type
                    node_type_lookup[name.lower()] = node_type
    return pmid_mesh, definition_lookup, node_type_lookup


def _title_for_pmid(pubmed: pd.DataFrame, pmid: str) -> str:
    if pubmed.empty or "pmid" not in pubmed.columns:
        return ""
    row = pubmed[pubmed["pmid"].astype(str) == str(pmid)]
    if row.empty:
        return ""
    for col in ("title", "article_title", "name"):
        if col in row.columns:
            val = row.iloc[0][col]
            if pd.notna(val):
                return " ".join(str(val).split())
    return ""


def _summary_for_pmid(summaries: pd.DataFrame, pmid: str, *, max_words: int = 220) -> str:
    if summaries.empty or "pmid" not in summaries.columns:
        return ""
    row = summaries[summaries["pmid"].astype(str) == str(pmid)]
    if row.empty:
        return ""
    for col in ("summary", "wiki_style_summary", "neuro_summary"):
        if col in row.columns:
            val = row.iloc[0][col]
            if pd.notna(val):
                words = str(val).split()
                return " ".join(words[:max_words])
    return ""


def _mesh_positive(term: str, category: str, definition_lookup: dict[str, str]) -> dict | None:
    if category not in ALLOWED_PUBMED_MESH_CATEGORIES:
        return None
    display = term.split("/")[0].strip()
    definition = definition_lookup.get(term) or definition_lookup.get(term.lower()) or definition_lookup.get(display) or definition_lookup.get(display.lower())
    if not definition:
        definition = f"{display} is a MeSH term used as a neuroscience-relevant label for this coordinate-derived brain map."
    return {
        "text": text_pair(display, definition),
        "term": display,
        "category": category,
        "source": "mesh",
        "weight": POSITIVE_WEIGHTS[f"mesh_{category}"],
        "reliability": "strong",
    }


def _base_row(pmid: str, *, map_id: str, tensor_path: str | None, nifti_path: str | None, tensor_index: int | None, metadata: dict[str, Any]) -> dict:
    return {
        "map_id": map_id,
        "source": "pubmed",
        "map_type": "coordinate_ale",
        "nifti_path": nifti_path,
        "tensor_path": tensor_path,
        "tensor_index": tensor_index,
        "space": "MNI",
        "affine": metadata.get("affine"),
        "resolution": metadata.get("resolution"),
        "shape": metadata.get("shape"),
        "preprocessing_config": metadata.get("preprocessing_config", {}),
        "positive_texts": [],
        "positive_terms": [],
        "positive_categories": [],
        "negative_sampling_groups": {"source": "pubmed", "map_type": "coordinate_ale"},
        "pmid": str(pmid),
        "mesh_terms": [],
        "quality_flags": {
            "is_coordinate_derived": True,
            "has_mesh_terms": False,
            "has_summary": False,
            "has_title": False,
            "no_positive_texts": False,
        },
    }


def build_pubmed_rows(paths: dict[str, Any], dataset_cfg: dict[str, Any]) -> list[dict]:
    pubmed, summaries = load_pubmed_metadata(paths)
    pmid_mesh, definitions, node_types = load_mesh_resources(paths)
    max_words = int(dataset_cfg.get("summary_max_words", 220))

    cache_value = paths.get("existing_ale_cache")
    cache_path = Path(cache_value) if cache_value else None
    if cache_path is not None and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        pmids = np.asarray(payload["pmids"]).astype(str)
        tensor_path = str(cache_path)
        meta = payload.get("metadata", {})
        preprocess = payload.get("config", {})
        resolution = float(preprocess.get("resolution_mm", dataset_cfg.get("target_resolution_mm", 2.0)))
        common_meta = {
            "affine": meta.get("affine"),
            "shape": meta.get("shape"),
            "resolution": [resolution] * 3,
            "preprocessing_config": preprocess,
        }
        entries = [(pmid, tensor_path, None, i, common_meta) for i, pmid in enumerate(pmids)]
    else:
        entries = []
        for path in sorted(Path(paths.get("map_cache_dir", "experiments/3dcnn/atlas_free_cnn/cache/maps")).glob("*.nii*")):
            pmid = path.name.split("_")[-1].split(".")[0]
            entries.append((pmid, None, str(path), None, {**nifti_metadata(path), "preprocessing_config": {}}))

    rows: list[dict] = []
    for pmid, tensor_path, nifti_path, tensor_index, meta in entries:
        row = _base_row(
            pmid,
            map_id=f"pubmed_ale_{pmid}",
            tensor_path=tensor_path,
            nifti_path=nifti_path,
            tensor_index=tensor_index,
            metadata=meta,
        )
        terms = pmid_mesh.get(str(pmid), [])
        seen = set()
        for term in terms:
            category = node_types.get(term, "") or node_types.get(term.lower(), "")
            pos = _mesh_positive(term, category, definitions)
            row["mesh_terms"].append({"term": term, "category": category})
            if pos and normalize_key(pos["term"]) not in seen:
                row["positive_texts"].append(pos)
                seen.add(normalize_key(pos["term"]))
        title = _title_for_pmid(pubmed, pmid)
        summary = _summary_for_pmid(summaries, pmid, max_words=max_words)
        if title:
            row["quality_flags"]["has_title"] = True
        if summary and dataset_cfg.get("include_pubmed_summary", True):
            row["quality_flags"]["has_summary"] = True
            row["positive_texts"].append(
                {
                    "text": text_pair(title or f"PubMed {pmid}", summary),
                    "term": title or f"PubMed {pmid}",
                    "category": "paper_summary",
                    "source": "wiki_style_summary",
                    "weight": POSITIVE_WEIGHTS["paper_wiki_summary"],
                    "reliability": "medium",
                }
            )
        if title and dataset_cfg.get("include_pubmed_title", True):
            row["positive_texts"].append(
                {
                    "text": title,
                    "term": title,
                    "category": "paper_title",
                    "source": "pubmed_title",
                    "weight": POSITIVE_WEIGHTS["paper_title"],
                    "reliability": "medium",
                }
            )
        row["quality_flags"]["has_mesh_terms"] = bool([p for p in row["positive_texts"] if p["source"] == "mesh"])
        row["quality_flags"]["no_positive_texts"] = not bool(row["positive_texts"])
        row["positive_terms"] = sorted({p["term"] for p in row["positive_texts"]})
        row["positive_categories"] = sorted({p["category"] for p in row["positive_texts"]})
        if row["positive_texts"] or not dataset_cfg.get("skip_examples_with_no_positive_texts", True):
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", default="experiments/3dcnn/atlas_free_cnn/configs/paths.yaml")
    p.add_argument("--config", default="experiments/3dcnn/atlas_free_cnn/configs/dataset_config.yaml")
    p.add_argument("--output", default="experiments/3dcnn/atlas_free_cnn/cache/unified_jsonl/pubmed_ale.jsonl")
    args = p.parse_args()
    paths = _load_yaml(args.paths)
    cfg = _load_yaml(args.config)
    rows = build_pubmed_rows(paths, cfg)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} PubMed rows to {args.output}")


if __name__ == "__main__":
    main()
