"""Reusable data setup helpers for the 21/22 evaluation notebooks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from neurovlm.data import load_dataset, load_latent


NETWORK_TEST_SET_SOURCE = "huggingface:neurovlm/embedded_text/network_test_set_labels.csv"


def resolve_evaluation_output_dir(path: str = "docs/03_evaluation/outputs") -> Path:
    """Resolve evaluation outputs from repo root or from inside docs/03_evaluation."""

    output_dir = Path(path)
    if output_dir.parent.exists():
        return output_dir
    local_output_dir = Path("outputs")
    if Path.cwd().name == "03_evaluation":
        return local_output_dir
    return output_dir


def normalize_expected_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_network_test_set_labels_table() -> tuple[pd.DataFrame, str]:
    """Load network labels from Hugging Face through the public NeuroVLM API."""

    return load_dataset("network_test_set_labels"), NETWORK_TEST_SET_SOURCE


def build_network_info(network_labels_df: pd.DataFrame) -> pd.DataFrame:
    """Build one canonical row per network from the label table."""

    labels = network_labels_df[network_labels_df["network_key"] != "unknown"].copy()
    return (
        labels.sort_values(["network_key", "raw_network_label"])
        .groupby("network_key", as_index=False)
        .agg(
            display=("network_name", "first"),
            short_definition=("short_definition", "first"),
            long_definition=("long_definition", "first"),
            mapped_terms=("mapped_terms", "first"),
            raw_aliases=("raw_network_label", lambda s: "; ".join(sorted(set(map(str, s))))),
        )
    )


def load_network_label_resources() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    labels_df, source = load_network_test_set_labels_table()
    labels_df = labels_df[labels_df["network_key"] != "unknown"].copy()
    return labels_df, build_network_info(labels_df), source


def build_labeled_network_data(network_labels_df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Align precomputed network latents to canonical labels."""

    all_net_latents = load_latent("networks_neuro")
    network_label_rows = network_labels_df.drop_duplicates("raw_network_label").set_index("raw_network_label")

    networks_data: dict[str, dict[str, Any]] = {}
    for atlas_name, atlas_latents in all_net_latents.items():
        if not hasattr(atlas_latents, "items"):
            continue
        for raw_label, latent in atlas_latents.items():
            if raw_label not in network_label_rows.index:
                continue
            row = network_label_rows.loc[raw_label]
            sample_name = f"{atlas_name}:{raw_label}"
            networks_data[sample_name] = {
                "latent": latent,
                "short_gt": normalize_expected_text(row["short_definition"]),
                "long_gt": normalize_expected_text(row["long_definition"]),
                "network_key": row["network_key"],
                "display": row["network_name"],
                "atlas": atlas_name,
                "raw_network_label": raw_label,
            }
    return networks_data, all_net_latents


def build_pubmed_b2t_eval(max_samples: int | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build PubMed B2T records from summary text ground truth."""

    df_summaries = load_dataset("pubmed_summaries")
    df_pubs = load_dataset("pubmed_text")
    if "test" not in df_summaries.columns:
        raise ValueError("pubmed_summaries must include a boolean 'test' column for PubMed evaluation.")

    df_test = df_summaries[df_summaries["test"].fillna(False).astype(bool)].reset_index(drop=True)
    pmid_col = "pmid" if "pmid" in df_test.columns else df_test.columns[0]
    summary_col = "summary" if "summary" in df_test.columns else "description"
    title_col = "title" if "title" in df_test.columns else ("name" if "name" in df_test.columns else None)

    pub_pmid_col = "pmid" if "pmid" in df_pubs.columns else df_pubs.columns[0]
    pub_title_col = "name" if "name" in df_pubs.columns else ("title" if "title" in df_pubs.columns else None)
    pub_title_lookup = {}
    if pub_title_col is not None:
        pub_title_lookup = (
            df_pubs.assign(_pmid_key=lambda df: df[pub_pmid_col].astype(str))
            .drop_duplicates("_pmid_key")
            .set_index("_pmid_key")[pub_title_col]
            .astype(str)
            .to_dict()
        )

    pubmed_latents, pubmed_pmids = load_latent("pubmed_images")
    pubmed_pmids = np.asarray(pubmed_pmids).astype(str)
    summary_text_latents, summary_latent_pmids = load_latent("pubmed_summaries")
    summary_latent_pmids = np.asarray(summary_latent_pmids).astype(str)
    summary_latent_by_pmid = {pmid: summary_text_latents[i] for i, pmid in enumerate(summary_latent_pmids)}

    mask = np.isin(pubmed_pmids, df_test[pmid_col].astype(str).values)
    aligned_latents = pubmed_latents[mask]
    aligned_pmids = pubmed_pmids[mask]
    pmid_to_row = df_test.assign(_pmid_key=lambda df: df[pmid_col].astype(str)).drop_duplicates("_pmid_key").set_index("_pmid_key")

    records = []
    missing_summary_latents = []
    for i, pmid in enumerate(aligned_pmids):
        if pmid not in pmid_to_row.index:
            continue
        if pmid not in summary_latent_by_pmid:
            missing_summary_latents.append(pmid)
            continue
        row = pmid_to_row.loc[pmid]
        title = str(row[title_col]) if title_col is not None and title_col in row.index else pub_title_lookup.get(pmid, "")
        summary = str(row[summary_col]) if summary_col in row.index else ""
        records.append(
            {
                "pmid": pmid,
                "latent": aligned_latents[i],
                "text_latent": summary_latent_by_pmid[pmid],
                "short_gt": title,
                "long_gt": summary,
            }
        )

    eval_records = records[:max_samples] if max_samples else records
    stats = {
        "records": len(records),
        "eval_records": len(eval_records),
        "summary_rows": len(df_summaries),
        "test_rows": len(df_test),
        "missing_summary_latents": len(missing_summary_latents),
    }
    return eval_records, stats


@torch.no_grad()
def project_precomputed_text_latents(nvlm, text_latents, batch_size: int = 512) -> torch.Tensor:
    nvlm._ensure_projection_heads()
    x = torch.as_tensor(text_latents).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    chunks = []
    for start in range(0, len(x), batch_size):
        text_batch = F.normalize(x[start : start + batch_size].to(nvlm.device), dim=1, eps=1e-8)
        z = nvlm._proj_head_text_infonce(text_batch)
        chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def project_precomputed_brain_latents(nvlm, brain_latents, batch_size: int = 512) -> torch.Tensor:
    nvlm._ensure_projection_heads()
    x = torch.as_tensor(brain_latents).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    chunks = []
    image_head = nvlm._proj_head_image_infonce
    for start in range(0, len(x), batch_size):
        z = image_head(x[start : start + batch_size].to(nvlm.device))
        chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


def build_neurovault_b2t_eval(nvlm, output_dir, max_samples: int | None = None) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Build NeuroVault B2T records, selecting the best image per DOI."""

    df_nv = load_dataset("neurovault_text")
    df_nv_meta = load_dataset("neurovault_images_meta")
    nv_latents = load_latent("neurovault_images")
    nv_text_latents = load_latent("neurovault_text")

    doi_pub = "doi" if "doi" in df_nv.columns else df_nv.columns[0]
    doi_meta = "doi" if "doi" in df_nv_meta.columns else df_nv_meta.columns[0]
    title_nv = "title" if "title" in df_nv.columns else df_nv.columns[1]
    abs_nv = "abstract" if "abstract" in df_nv.columns else df_nv.columns[2]

    nv_brain_shared = project_precomputed_brain_latents(nvlm, nv_latents)
    nv_text_shared = project_precomputed_text_latents(nvlm, nv_text_latents)

    records = []
    for text_idx, pub_row in df_nv.reset_index(drop=True).iterrows():
        if text_idx >= len(nv_text_latents):
            continue
        doi = pub_row[doi_pub]
        img_indices = np.where((df_nv_meta[doi_meta] == doi).values)[0]
        img_indices = np.asarray([idx for idx in img_indices if idx < len(nv_latents)], dtype=int)
        if len(img_indices) == 0:
            continue
        sims = nv_brain_shared[img_indices] @ nv_text_shared[text_idx : text_idx + 1].T
        best_img_idx = int(img_indices[int(torch.argmax(sims.squeeze()).item())])
        records.append(
            {
                "doi": doi,
                "latent": nv_latents[best_img_idx],
                "text_latent": nv_text_latents[text_idx],
                "short_gt": str(pub_row[title_nv]),
                "long_gt": str(pub_row[abs_nv]),
                "selected_image_index": best_img_idx,
                "n_candidate_images": int(len(img_indices)),
                "selected_image_text_similarity": float(sims.max().item()),
            }
        )

    selection_df = pd.DataFrame(
        [
            {
                "doi": d["doi"],
                "selected_image_index": d["selected_image_index"],
                "n_candidate_images": d["n_candidate_images"],
                "selected_image_text_similarity": d["selected_image_text_similarity"],
            }
            for d in records
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_df.to_csv(output_dir / "b2t_neurovault_argmax_image_selection.csv", index=False)
    selection_df.to_json(output_dir / "b2t_neurovault_argmax_image_selection.json", orient="records", indent=2)
    return (records[:max_samples] if max_samples else records), selection_df


def build_pubmed_t2b_eval(max_samples: int | None = None) -> list[dict[str, Any]]:
    df_pubs = load_dataset("pubmed_text")
    if "test" not in df_pubs.columns:
        raise ValueError("pubmed_text must include a boolean 'test' column for PubMed evaluation.")
    df_test = df_pubs[df_pubs["test"].fillna(False).astype(bool)].reset_index(drop=True)

    pmid_col = "pmid" if "pmid" in df_test.columns else df_test.columns[0]
    title_col = "name" if "name" in df_test.columns else "title"
    abstract_col = "description" if "description" in df_test.columns else "abstract"
    pubmed_latents, pubmed_pmids = load_latent("pubmed_images")
    pubmed_pmids = np.asarray(pubmed_pmids)
    mask = np.isin(pubmed_pmids, df_test[pmid_col].values)
    aligned_latents = pubmed_latents[mask]
    aligned_pmids = pubmed_pmids[mask]
    pmid_to_row = df_test.set_index(pmid_col)

    records = []
    for i, pmid in enumerate(aligned_pmids):
        if pmid not in pmid_to_row.index:
            continue
        row = pmid_to_row.loc[pmid]
        records.append(
            {
                "pmid": pmid,
                "latent": aligned_latents[i],
                "short_gt": str(row[title_col]) if title_col in row.index else "",
                "long_gt": str(row[abstract_col]) if abstract_col in row.index else "",
            }
        )
    return records[:max_samples] if max_samples else records


def build_neurovault_t2b_eval(max_samples: int | None = None) -> list[dict[str, Any]]:
    df_nv = load_dataset("neurovault_text")
    df_nv_meta = load_dataset("neurovault_images_meta")
    nv_latents = load_latent("neurovault_images")

    doi_pub = "doi" if "doi" in df_nv.columns else df_nv.columns[0]
    doi_meta = "doi" if "doi" in df_nv_meta.columns else df_nv_meta.columns[0]
    title_nv = "title" if "title" in df_nv.columns else df_nv.columns[1]
    abs_nv = "abstract" if "abstract" in df_nv.columns else df_nv.columns[2]

    records = []
    for _, pub_row in df_nv.iterrows():
        doi = pub_row[doi_pub]
        img_indices = np.where((df_nv_meta[doi_meta] == doi).values)[0]
        img_indices = np.asarray([idx for idx in img_indices if idx < len(nv_latents)], dtype=int)
        if len(img_indices) == 0:
            continue
        candidate_latents = nv_latents[img_indices]
        records.append(
            {
                "doi": doi,
                "latent": candidate_latents[0],
                "candidate_latents": candidate_latents,
                "candidate_image_indices": img_indices,
                "n_candidate_images": int(len(img_indices)),
                "short_gt": str(pub_row[title_nv]),
                "long_gt": str(pub_row[abs_nv]),
            }
        )
    return records[:max_samples] if max_samples else records
