"""Prepare PubMed neuro summaries for Hugging Face upload.

This script creates two artifacts:

1. ``neuro_summaries.parquet``
   ``neuro_summaries.parquet`` with the publication boolean split columns
   (``train``, ``val``, ``test``) copied from ``publications.parquet`` by
   PMID. If summaries do not contain a title column, titles are also copied
   from publications for SPECTER encoding.

2. ``latent_neuro_summaries.pt``
   SPECTER embeddings for ``title [SEP] summary`` rows, saved as
   ``{"latent": tensor, "pmid": array}`` so it matches existing NeuroVLM
   latent text artifacts.

Example
-------
python docs/data_preparation/scripts/prepare_neuro_summaries.py \
    --output-dir /tmp/neuro_summaries_artifacts \
    --device cuda \
    --batch-size 32

Upload the generated ``neuro_summaries.parquet`` to
``neurovlm/neuro_image_papers`` and upload ``latent_neuro_summaries.pt`` to
``neurovlm/embedded_text``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_parquet_or_dataset(path: str | None, dataset_name: str) -> pd.DataFrame:
    if path:
        return pd.read_parquet(path)
    from neurovlm.data import load_dataset

    return load_dataset(dataset_name)


def _resolve_title_column(df: pd.DataFrame) -> str | None:
    for col in ("title", "name"):
        if col in df.columns:
            return col
    return None


def _batched_indices(n_rows: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def add_publication_metadata(
    summaries: pd.DataFrame,
    publications: pd.DataFrame,
) -> pd.DataFrame:
    """Add publication split booleans and title metadata to neuro summaries by PMID."""
    if "pmid" not in summaries.columns:
        raise ValueError("Expected neuro summaries to contain a 'pmid' column.")
    if "summary" not in summaries.columns:
        raise ValueError("Expected neuro summaries to contain a 'summary' column.")
    if "pmid" not in publications.columns:
        raise ValueError("Expected publications to contain a 'pmid' column.")

    summaries = summaries.copy()
    publications = publications.copy()
    summaries["pmid"] = summaries["pmid"].astype(str)
    publications["pmid"] = publications["pmid"].astype(str)

    split_bool_cols = [col for col in ("train", "val", "test") if col in publications.columns]
    if not split_bool_cols:
        print(
            "Warning: publications has no train/val/test boolean columns. "
            "Creating them as False."
        )
        for col in ("train", "val", "test"):
            publications[col] = False
        split_bool_cols = ["train", "val", "test"]

    pub_title_col = _resolve_title_column(publications)
    merge_cols = ["pmid"] + split_bool_cols
    if pub_title_col is not None:
        merge_cols.append(pub_title_col)

    pub_meta = publications[merge_cols].drop_duplicates("pmid")
    out = summaries.merge(pub_meta, on="pmid", how="left", suffixes=("", "_publication"))

    if "title" not in out.columns:
        if pub_title_col is None:
            out["title"] = ""
        elif pub_title_col == "title":
            # Covered by merge, but kept explicit for readability.
            out["title"] = out["title"].fillna("")
        else:
            out["title"] = out[pub_title_col].fillna("")
    else:
        if pub_title_col is not None:
            pub_title_name = pub_title_col if pub_title_col != "title" else "title_publication"
            if pub_title_name in out.columns:
                out["title"] = out["title"].fillna(out[pub_title_name])
        out["title"] = out["title"].fillna("")

    for col in ("train", "val", "test"):
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)

    missing_mask = ~out[["train", "val", "test"]].any(axis=1)
    missing_split_meta = int(missing_mask.sum())
    if missing_split_meta:
        print(
            f"Warning: {missing_split_meta:,} summary rows did not match any "
            "publication split by PMID; assigning them to test=True."
        )
        out.loc[missing_mask, "test"] = True

    return out


def embed_summaries(
    summaries: pd.DataFrame,
    batch_size: int,
    device: str,
    normalize: bool = True,
):
    """Encode title + summary rows with SPECTER."""
    import torch
    from neurovlm.models import Specter

    specter = Specter(
        "allenai/specter2_aug2023refresh",
        adapter="adhoc_query",
        device=device,
    )
    embeddings: list[torch.Tensor] = []

    encode_df = summaries[["title", "summary"]].copy()
    encode_df["title"] = encode_df["title"].fillna("").astype(str)
    encode_df["summary"] = encode_df["summary"].fillna("").astype(str)

    for start, end in _batched_indices(len(encode_df), batch_size):
        batch = encode_df.iloc[start:end]
        emb = specter(batch).detach().cpu()
        if normalize:
            emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        embeddings.append(emb)
        print(f"Embedded {end:,}/{len(encode_df):,}", end="\r", flush=True)

    print()
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 768))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries-parquet", default=None, help="Optional local neuro_summaries.parquet path.")
    parser.add_argument("--publications-parquet", default=None, help="Optional local publications.parquet path.")
    parser.add_argument("--output-dir", default="artifacts/neuro_summaries", help="Directory for generated artifacts.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Encoding device. Defaults to cuda when available, else cpu.")
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize SPECTER embeddings.")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Only write neuro_summaries.parquet with publication split booleans; do not regenerate embeddings.",
    )
    parser.add_argument(
        "--overwrite-latent",
        action="store_true",
        help="Regenerate latent_neuro_summaries.pt even if it already exists.",
    )
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = _read_parquet_or_dataset(args.summaries_parquet, "pubmed_summaries")
    publications = _read_parquet_or_dataset(args.publications_parquet, "pubmed_text")
    prepared = add_publication_metadata(summaries, publications)

    prepared_path = output_dir / "neuro_summaries.parquet"
    prepared.to_parquet(prepared_path, index=False)
    print(f"Wrote prepared summaries: {prepared_path}")
    print("Split boolean counts:")
    for col in ("train", "val", "test"):
        if col in prepared.columns:
            print(f"{col}: {int(prepared[col].sum()):,}")

    latent_path = output_dir / "latent_neuro_summaries.pt"
    if args.skip_embedding:
        print("Skipping embedding because --skip-embedding was set.")
        if latent_path.exists():
            print(f"Existing embeddings left untouched: {latent_path}")
        return
    if latent_path.exists() and not args.overwrite_latent:
        print(f"Embedding file already exists; leaving it untouched: {latent_path}")
        print("Pass --overwrite-latent to regenerate it.")
        return

    latent = embed_summaries(
        prepared,
        batch_size=args.batch_size,
        device=device,
        normalize=not args.no_normalize,
    )
    torch.save(
        {
            "latent": latent,
            "pmid": prepared["pmid"].astype(str).to_numpy(),
        },
        latent_path,
    )
    print(f"Wrote summary embeddings: {latent_path}")
    print(f"Latent shape: {tuple(latent.shape)}")


if __name__ == "__main__":
    main()
