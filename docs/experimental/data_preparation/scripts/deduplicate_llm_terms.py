"""Deduplicate llm_neuro_terms.parquet by normalizing terms in-place.

Normalizations applied:
  - underscores → spaces
  - truncated parentheticals stripped: "term (abbr" with no closing ")" → "term"
  - excess whitespace collapsed

When multiple rows collapse to the same term key, they are merged:
  - document_count: summed
  - category: taken from the highest-document-count row
  - datasets / example_docs / evidence_examples: union
  - definition: taken from the highest-document-count row
  - already_in_cogatlas_or_mesh: True if any row was True

Bare abbreviations (fmr, fmri) are dropped since the full form is present.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TERMS_TO_DROP = {"fmr", "fmri"}


def _norm(term: str) -> str:
    term = term.replace("_", " ")
    term = re.sub(r"\s*\([^)]*$", "", term)
    return re.sub(r"\s+", " ", term).strip()


def merge_group(group: pd.DataFrame) -> pd.Series:
    best = group.loc[group["document_count"].idxmax()]
    return pd.Series({
        "term": _norm(best["term"]),
        "category": best["category"],
        "definition": best["definition"],
        "document_count": int(group["document_count"].sum()),
        "datasets": "; ".join(sorted(set(
            ds.strip()
            for val in group["datasets"].dropna().astype(str)
            for ds in val.split(";")
            if ds.strip()
        ))),
        "example_docs": "; ".join(list(dict.fromkeys(
            doc.strip()
            for val in group["example_docs"].dropna().astype(str)
            for doc in val.split(";")
            if doc.strip()
        ))[:5]),
        "evidence_examples": " | ".join(list(dict.fromkeys(
            ex.strip()
            for val in group["evidence_examples"].dropna().astype(str)
            for ex in val.split("|")
            if ex.strip()
        ))[:3]),
        "already_in_cogatlas_or_mesh": bool(group["already_in_cogatlas_or_mesh"].any()),
    })


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_key"] = df["term"].map(_norm)

    # Drop bare abbreviations that are noise
    df = df[~df["_key"].isin(TERMS_TO_DROP)].copy()

    merged = (
        df.groupby("_key", sort=False)
        .apply(merge_group, include_groups=False)
        .reset_index(drop=True)
    )
    return (
        merged
        .sort_values(["document_count", "term"], ascending=[False, True])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="artifacts/llm_extracted_neuro_terms/llm_neuro_terms.parquet",
    )
    parser.add_argument("--output", default=None, help="Defaults to overwriting input.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} rows from {in_path}")

    clean = deduplicate(df)
    print(f"After dedup: {len(clean):,} rows ({len(df) - len(clean):,} removed)")

    clean.to_parquet(out_path, index=False)
    clean.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
