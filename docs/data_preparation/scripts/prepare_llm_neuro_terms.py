"""Prepare LLM-extracted neuroscience terms for retrieval.

This script parses the LLM document-cache JSON produced by the extraction
notebooks, writes CSV summaries, removes terms already present in CogAtlas or
MeSH/KG, and embeds the remaining novel terms with SPECTER.

Outputs
-------
``llm_neuro_terms_all.csv``
    All recovered LLM-extracted terms.

``llm_neuro_terms_new.csv`` / ``llm_neuro_terms.parquet``
    Novel terms after filtering against CogAtlas concepts/tasks/disorders and
    KG-MeSH terms. Upload ``llm_neuro_terms.parquet`` to
    ``neurovlm/embedded_text``.

``latent_llm_neuro_terms.pt``
    SPECTER embeddings for the novel terms, saved as
    ``{"latent": tensor, "term": array}``. Upload to ``neurovlm/embedded_text``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_term(term: Any) -> str:
    term = normalize_text(term).lower()
    term = term.replace("_", " ")
    # Strip truncated parentheticals: "(xyz" at end with no closing ")"
    term = re.sub(r"\s*\([^)]*$", "", term)
    term = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", term)
    return re.sub(r"\s+", " ", term)


def normalize_category(category: Any) -> str:
    category = normalize_term(category).replace(" ", "_").replace("-", "_")
    aliases = {
        "anatomy": "anatomical_region",
        "brain_region": "anatomical_region",
        "region": "anatomical_region",
        "network": "brain_network",
        "cognition": "cognitive_function",
        "cognitive_process": "cognitive_function",
        "task": "cognitive_task",
        "experimental_paradigm": "experimental_task",
        "disease": "disease_or_disorder",
        "disorder": "disease_or_disorder",
        "modality": "imaging_modality",
        "statistic": "measurement_or_statistic",
        "measure": "measurement_or_statistic",
    }
    return aliases.get(category, category or "other_neuroscience_term")


def extract_json_array(text: Any) -> list[Any]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "terms" in obj:
            return obj["terms"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def clean_term_record(record: Any) -> dict[str, str] | None:
    if not isinstance(record, dict):
        return None
    term = normalize_term(record.get("term", ""))
    if len(term) < 3 or term in {"brain", "human", "humans", "study", "result", "results"}:
        return None
    definition = normalize_text(record.get("definition", "")).lower()[:600]
    evidence = normalize_text(record.get("evidence", ""))[:400]
    return {
        "term": term,
        "category": normalize_category(record.get("category", "other_neuroscience_term")),
        "definition": definition or f"{term} is a neuroscience term identified from source papers.",
        "evidence": evidence,
    }


def load_cache_terms(cache_path: Path) -> pd.DataFrame:
    cache = json.loads(cache_path.read_text())
    rows: list[dict[str, Any]] = []
    for key, item in cache.items():
        dataset = item.get("dataset", "")
        doc_id = str(item.get("doc_id", ""))
        title = item.get("title", "")

        records = item.get("terms") or []
        if not records and item.get("raw_response"):
            records = extract_json_array(item.get("raw_response"))

        for record in records:
            cleaned = clean_term_record(record)
            if cleaned is None:
                continue
            rows.append({
                "dataset": dataset,
                "doc_id": doc_id,
                "title": title,
                **cleaned,
            })

    if not rows:
        return pd.DataFrame(columns=["dataset", "doc_id", "title", "term", "category", "definition", "evidence"])
    return pd.DataFrame(rows).drop_duplicates(["dataset", "doc_id", "term", "category"]).reset_index(drop=True)


def aggregate_terms(raw_terms: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (term, category), sub in raw_terms.groupby(["term", "category"], dropna=False):
        definition_counts = Counter(sub["definition"].dropna().astype(str))
        best_definition = definition_counts.most_common(1)[0][0] if definition_counts else ""
        rows.append({
            "term": term,
            "category": category,
            "definition": best_definition,
            "document_count": int(sub[["dataset", "doc_id"]].drop_duplicates().shape[0]),
            "datasets": "; ".join(sorted(sub["dataset"].astype(str).unique())),
            "example_docs": "; ".join((sub["dataset"].astype(str) + ":" + sub["doc_id"].astype(str)).drop_duplicates().head(5)),
            "evidence_examples": " | ".join(sub["evidence"].dropna().astype(str).drop_duplicates().head(3)),
        })
    if not rows:
        return pd.DataFrame(columns=["term", "category", "definition", "document_count", "datasets", "example_docs", "evidence_examples"])
    return pd.DataFrame(rows).sort_values(["document_count", "term"], ascending=[False, True]).reset_index(drop=True)


def iter_alias_terms(value: Any) -> Iterable[str]:
    if pd.isna(value):
        return []
    return [part for part in re.split(r"[;,|]", str(value)) if part.strip()]


def load_existing_terms() -> set[str]:
    from neurovlm.data import load_dataset

    existing: set[str] = set()
    for dataset_name in ["cogatlas", "cogatlas_task", "cogatlas_disorder", "kg_mesh"]:
        df = load_dataset(dataset_name)
        for col in ["term", "title", "name"]:
            if col in df.columns:
                existing.update(normalize_term(x) for x in df[col].dropna().astype(str))
                break
        if "alias" in df.columns:
            for value in df["alias"].dropna():
                existing.update(normalize_term(x) for x in iter_alias_terms(value))
    return {term for term in existing if term}


def _batched_indices(n_rows: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def embed_terms(terms_df: pd.DataFrame, batch_size: int, device: str, normalize: bool = True):
    import torch
    from neurovlm.models import Specter

    specter = Specter(
        "allenai/specter2_aug2023refresh",
        adapter="adhoc_query",
        device=device,
    )
    encode_df = terms_df[["term", "definition"]].rename(columns={"term": "title", "definition": "summary"}).copy()
    embeddings: list[torch.Tensor] = []
    for start, end in _batched_indices(len(encode_df), batch_size):
        emb = specter(encode_df.iloc[start:end]).detach().cpu()
        if normalize:
            emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        embeddings.append(emb)
        print(f"Embedded {end:,}/{len(encode_df):,}", end="\r", flush=True)
    print()
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 768))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-json",
        default="artifacts/llm_extracted_neuro_terms/llm_neuroscience_terms_doc_cache_hf.json",
        help="LLM extraction document-cache JSON.",
    )
    parser.add_argument("--output-dir", default="artifacts/llm_extracted_neuro_terms")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Embedding device. Defaults to cuda when available, else cpu.")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--overwrite-latent", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_json)

    raw_terms = load_cache_terms(cache_path)
    all_terms = aggregate_terms(raw_terms)
    all_path = output_dir / "llm_neuro_terms_all.csv"
    raw_path = output_dir / "llm_neuro_terms_by_doc.csv"
    raw_terms.to_csv(raw_path, index=False)
    all_terms.to_csv(all_path, index=False)
    print(f"Wrote document terms: {raw_path} ({len(raw_terms):,} rows)")
    print(f"Wrote all aggregated terms: {all_path} ({len(all_terms):,} terms)")

    existing = load_existing_terms()
    new_terms = all_terms[~all_terms["term"].map(lambda term: normalize_term(term) in existing)].copy()
    new_terms["already_in_cogatlas_or_mesh"] = False
    all_terms["already_in_cogatlas_or_mesh"] = all_terms["term"].map(lambda term: normalize_term(term) in existing)
    all_terms.to_csv(all_path, index=False)

    new_csv_path = output_dir / "llm_neuro_terms_new.csv"
    new_parquet_path = output_dir / "llm_neuro_terms.parquet"
    new_terms.to_csv(new_csv_path, index=False)
    new_terms.to_parquet(new_parquet_path, index=False)
    print(f"Existing CogAtlas/MESH terms filtered out: {int(all_terms['already_in_cogatlas_or_mesh'].sum()):,}")
    print(f"New terms: {len(new_terms):,}")
    print(f"Wrote new-term CSV: {new_csv_path}")
    print(f"Wrote new-term parquet for upload: {new_parquet_path}")

    latent_path = output_dir / "latent_llm_neuro_terms.pt"
    if args.skip_embedding:
        print("Skipping embedding because --skip-embedding was set.")
        return
    if latent_path.exists() and not args.overwrite_latent:
        print(f"Embedding file already exists; leaving it untouched: {latent_path}")
        print("Pass --overwrite-latent to regenerate it.")
        return

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    latent = embed_terms(new_terms, batch_size=args.batch_size, device=device)
    torch.save({"latent": latent, "term": new_terms["term"].astype(str).to_numpy()}, latent_path)
    print(f"Wrote embeddings for upload: {latent_path}")
    print(f"Latent shape: {tuple(latent.shape)}")


if __name__ == "__main__":
    main()
