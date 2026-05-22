"""Reusable helpers for the 21/22 evaluation notebooks."""

from __future__ import annotations

import re
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from tqdm.notebook import tqdm

from neurovlm.data import load_dataset, load_latent
from neurovlm.metrics import (
    dice_percentile,
    mni152_to_fsaverage_arrays,
    nct_dice_spin_test_surface,
    normalized_k_values,
    normalized_recall_curve_auc,
    pearson_correlation,
)
from neurovlm.semantic_evaluation import multi_positive_ranking_metrics


NETWORK_TEST_SET_SOURCE = "huggingface:neurovlm/embedded_text/network_test_set_labels.csv"


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
        z = nvlm._proj_head_text_infonce(x[start : start + batch_size].to(nvlm.device))
        chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def project_precomputed_brain_latents(nvlm, brain_latents, batch_size: int = 512) -> torch.Tensor:
    nvlm._ensure_projection_heads()
    x = torch.as_tensor(brain_latents).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    chunks = []
    for start in range(0, len(x), batch_size):
        z = nvlm._proj_head_image(x[start : start + batch_size].to(nvlm.device))
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


def as_latent_batch(latents) -> torch.Tensor:
    if isinstance(latents, torch.Tensor):
        batch = latents.detach().cpu()
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        return batch
    batch = [x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in latents]
    batch = [x.unsqueeze(0) if x.dim() == 1 else x for x in batch]
    return torch.cat(batch, dim=0)


def decode_latents_to_brain(latents, decoder, batch_size: int = 64) -> np.ndarray:
    batch = as_latent_batch(latents)
    dec_device = next(decoder.parameters()).device
    chunks = []
    with torch.no_grad():
        for start in range(0, int(batch.shape[0]), batch_size):
            lat = batch[start : start + batch_size].to(dec_device)
            chunks.append(torch.sigmoid(decoder(lat)).detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype("float32")


def build_surface_eligibility_mask(masker):
    """Return a cortical/surface-oriented mask vector aligned to masker output."""

    try:
        import nibabel as nib
        from nilearn.datasets import fetch_atlas_harvard_oxford
        from nilearn.image import resample_to_img

        mask_img = getattr(masker, "mask_img_", None) or getattr(masker, "mask_img", None)
        atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_img = resample_to_img(atlas.maps, mask_img, interpolation="nearest")
        atlas_data = np.asarray(atlas_img.get_fdata())
        cortical_img = nib.Nifti1Image((atlas_data > 0).astype("int8"), mask_img.affine, mask_img.header)
        cortical_vec = masker.transform(cortical_img).ravel() > 0.5
        if cortical_vec.sum() == 0:
            raise RuntimeError("Harvard-Oxford cortical mask was empty after resampling.")
        return cortical_vec, "harvard_oxford_cortical_thr25"
    except Exception as exc:
        import nibabel as nib

        mask_img = getattr(masker, "mask_img_", None) or getattr(masker, "mask_img", None)
        mask_data = np.asarray(mask_img.get_fdata()).astype(bool)
        ijk = np.column_stack(np.where(mask_data))
        xyz = nib.affines.apply_affine(mask_img.affine, ijk)
        central_subcortical_proxy = (np.abs(xyz[:, 0]) < 35) & (np.abs(xyz[:, 1]) < 45) & (np.abs(xyz[:, 2]) < 35)
        inferior_proxy = xyz[:, 2] < -35
        cortical_vec = (~central_subcortical_proxy) & (~inferior_proxy)
        print(f"Surface eligibility atlas unavailable; using MNI cortical-shell fallback. Reason: {type(exc).__name__}: {exc}")
        return cortical_vec, "mni_cortical_shell_fallback"


def cortical_top_mass_fraction(brain_true: np.ndarray, surface_mask: np.ndarray, pct: float) -> tuple[float, str]:
    values = np.asarray(brain_true, dtype=float).ravel()
    positive = np.clip(values, 0, None)
    if positive.size != surface_mask.size:
        return np.nan, "surface_mask_shape_mismatch"
    threshold = np.percentile(positive, pct)
    top = positive >= threshold
    if not np.any(top):
        return np.nan, "no_top_activation_voxels"
    top_mass = float(positive[top].sum())
    if top_mass <= 0:
        return np.nan, "zero_top_activation_mass"
    cortical_mass = float(positive[top & surface_mask].sum())
    return cortical_mass / top_mass, "ok"


def surface_metric_eligibility(
    dataset_name: str,
    brain_true: np.ndarray,
    *,
    surface_mask: np.ndarray,
    dice_pct: float,
    pubmed_surface_filter_enabled: bool,
    pubmed_min_cortical_top_mass_fraction: float,
) -> tuple[bool, float, str]:
    frac, reason = cortical_top_mass_fraction(brain_true, surface_mask, pct=dice_pct)
    if dataset_name != "pubmed" or not pubmed_surface_filter_enabled:
        return True, frac, "not_pubmed_or_filter_disabled"
    if not np.isfinite(frac):
        return False, frac, reason
    if frac < pubmed_min_cortical_top_mass_fraction:
        return False, frac, f"pubmed_low_cortical_top_mass_fraction<{pubmed_min_cortical_top_mass_fraction}"
    return True, frac, "ok"


def select_true_brain_for_eval(true_latent, decoder, brain_pred, candidate_latents=None, candidate_indices=None, selection_mode="first") -> dict[str, Any]:
    latents = candidate_latents if candidate_latents is not None else true_latent
    true_brains = decode_latents_to_brain(latents, decoder)
    n_candidates = int(true_brains.shape[0])
    if candidate_indices is None:
        candidate_indices = np.arange(n_candidates)
    candidate_indices = np.asarray(candidate_indices)

    scores = np.asarray([pearson_correlation(brain, brain_pred) for brain in true_brains], dtype=float)
    selected_pos = int(np.nanargmax(scores)) if selection_mode == "argmax_pearson" and n_candidates > 1 else 0
    return {
        "brain_true": true_brains[selected_pos],
        "selected_candidate_position": selected_pos,
        "selected_candidate_index": int(candidate_indices[selected_pos]) if selected_pos < len(candidate_indices) else selected_pos,
        "selected_candidate_pearson": float(scores[selected_pos]) if len(scores) else np.nan,
        "n_candidate_images": n_candidates,
        "candidate_pearson_max": float(np.nanmax(scores)) if len(scores) else np.nan,
        "candidate_pearson_mean": float(np.nanmean(scores)) if len(scores) else np.nan,
    }


def evaluate_t2b_sample(
    *,
    nvlm,
    masker,
    name,
    text_input,
    true_latent,
    dataset_name="unknown",
    candidate_latents=None,
    candidate_indices=None,
    neurovault_selection_mode="first",
    dice_pct: float = 90,
    surface_mask: np.ndarray,
    surface_eligibility_method: str,
    pubmed_surface_filter_enabled: bool = True,
    pubmed_min_cortical_top_mass_fraction: float = 0.5,
    spin_use_neuromaps: bool = True,
    spin_require_neuromaps: bool = False,
    spin_test_n_perm: int = 1000,
    spin_test_random_state: int = 13,
    spin_fsaverage_density: str = "41k",
    spin_transform_method: str = "linear",
) -> dict[str, Any] | None:
    try:
        gen_result = nvlm.text(text_input).to_brain(head="mse")
        nifti_pred = gen_result.to_nifti(index=0)
        brain_pred = masker.transform(nifti_pred).ravel().astype("float32")

        selection_mode = neurovault_selection_mode if dataset_name == "neurovault" else "first"
        selected = select_true_brain_for_eval(
            true_latent=true_latent,
            decoder=gen_result.decoder,
            brain_pred=brain_pred,
            candidate_latents=candidate_latents,
            candidate_indices=candidate_indices,
            selection_mode=selection_mode,
        )
        brain_true = selected["brain_true"]
        nifti_true = masker.inverse_transform(brain_true.reshape(1, -1))

        pearson_r = float(pearson_correlation(brain_true, brain_pred))
        spearman_rho, _ = spearmanr(brain_true.ravel(), brain_pred.ravel())
        surface_eligible, cortical_frac, surface_reason = surface_metric_eligibility(
            dataset_name,
            brain_true,
            surface_mask=surface_mask,
            dice_pct=dice_pct,
            pubmed_surface_filter_enabled=pubmed_surface_filter_enabled,
            pubmed_min_cortical_top_mass_fraction=pubmed_min_cortical_top_mass_fraction,
        )

        dice_val = np.nan
        dice_method = "not_run_surface_ineligible" if not surface_eligible else "volume_masker_percentile"
        pred_lh = pred_rh = true_lh = true_rh = None
        spin_p_value = np.nan
        spin_method = "not_run_surface_ineligible" if not surface_eligible else "not_run"
        spin_significant = False

        if surface_eligible:
            dice_val = dice_percentile(brain_pred, brain_true, pct=dice_pct)
            if spin_use_neuromaps:
                try:
                    pred_lh, pred_rh = mni152_to_fsaverage_arrays(nifti_pred, density=spin_fsaverage_density, method=spin_transform_method)
                    true_lh, true_rh = mni152_to_fsaverage_arrays(nifti_true, density=spin_fsaverage_density, method=spin_transform_method)
                    nct = nct_dice_spin_test_surface(
                        pred_lh,
                        pred_rh,
                        true_lh,
                        true_rh,
                        pct=dice_pct,
                        n_perm=spin_test_n_perm,
                        random_state=spin_test_random_state,
                        density=spin_fsaverage_density,
                    )
                    dice_val = nct.dice_pct
                    dice_method = "surface_fsaverage_percentile_nct"
                    spin_p_value = nct.spin_p_value
                    spin_method = nct.spin_method
                    spin_significant = nct.spin_significant
                except ImportError as exc:
                    spin_method = f"not_run_missing_dependency:{exc.name}: {exc}"
                    if spin_require_neuromaps:
                        raise RuntimeError(f"Spin dependency failed for {name}: {spin_method}") from exc
                except Exception as exc:
                    spin_method = f"not_run_nct_error:{type(exc).__name__}: {exc}"
                    if spin_require_neuromaps:
                        raise

        return {
            "name": name,
            "text_input": text_input[:120],
            "pearson_r": pearson_r,
            "spearman_rho": float(spearman_rho),
            f"dice_pct{dice_pct}": float(dice_val) if np.isfinite(dice_val) else np.nan,
            "dice_method": dice_method,
            "spin_p_value": float(spin_p_value) if np.isfinite(spin_p_value) else np.nan,
            "spin_significant": bool(spin_significant),
            "spin_method": spin_method,
            "surface_metric_eligible": bool(surface_eligible),
            "surface_metric_skip_reason": surface_reason,
            "cortical_top_mass_fraction": float(cortical_frac) if np.isfinite(cortical_frac) else np.nan,
            "surface_eligibility_method": surface_eligibility_method,
            "neurovault_selection_mode": selection_mode if dataset_name == "neurovault" else "not_applicable",
            "n_candidate_images": selected["n_candidate_images"],
            "selected_candidate_position": selected["selected_candidate_position"],
            "selected_candidate_index": selected["selected_candidate_index"],
            "selected_candidate_pearson": selected["selected_candidate_pearson"],
            "candidate_pearson_max": selected["candidate_pearson_max"],
            "candidate_pearson_mean": selected["candidate_pearson_mean"],
            "_brain_pred": brain_pred,
            "_brain_true": brain_true,
            "_pred_lh": pred_lh,
            "_pred_rh": pred_rh,
            "_true_lh": true_lh,
            "_true_rh": true_rh,
        }
    except Exception as exc:
        print(f"[T2B error] {name}: {type(exc).__name__}: {exc}")
        if spin_use_neuromaps and spin_require_neuromaps:
            raise
        return None


def make_t2b_runner(**config):
    """Return a notebook-friendly positional T2B runner bound to local config."""

    def runner(
        name,
        text_input,
        true_latent,
        dataset_name="unknown",
        candidate_latents=None,
        candidate_indices=None,
        neurovault_selection_mode="first",
    ):
        return evaluate_t2b_sample(
            **config,
            name=name,
            text_input=text_input,
            true_latent=true_latent,
            dataset_name=dataset_name,
            candidate_latents=candidate_latents,
            candidate_indices=candidate_indices,
            neurovault_selection_mode=neurovault_selection_mode,
        )

    return runner


def random_baseline_voxel_index(n_voxels: int, *, max_voxels: int, seed: int) -> np.ndarray:
    n = min(int(max_voxels), int(n_voxels))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_voxels, size=n, replace=False)) if n < n_voxels else np.arange(n_voxels)


def _zscore_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype="float32")
    arr = arr - arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return arr / std


def _rank_zscore_rows(arr: np.ndarray) -> np.ndarray:
    ranks = np.vstack([rankdata(row, method="average") for row in np.asarray(arr, dtype="float32")]).astype("float32")
    return _zscore_rows(ranks)


def add_random_correlation_baseline(
    df: pd.DataFrame,
    *,
    n_random: int = 25,
    seed: int = 13,
    max_voxels: int = 10000,
) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "pearson_baseline_actual",
        "pearson_random_mean",
        "pearson_random_std",
        "pearson_minus_random",
        "pearson_random_percentile",
        "spearman_baseline_actual",
        "spearman_random_mean",
        "spearman_random_std",
        "spearman_minus_random",
        "spearman_random_percentile",
    ]:
        out[col] = np.nan
    if out.empty:
        return out

    voxel_idx = random_baseline_voxel_index(len(out.iloc[0]["_brain_pred"]), max_voxels=max_voxels, seed=seed)
    rng = np.random.default_rng(seed)

    for dataset, sub in out.groupby("dataset", sort=False):
        idxs = sub.index.to_numpy()
        if len(idxs) < 2:
            continue
        pred_subset = np.vstack([out.at[idx, "_brain_pred"][voxel_idx] for idx in idxs])
        true_subset = np.vstack([out.at[idx, "_brain_true"][voxel_idx] for idx in idxs])
        pred_z = _zscore_rows(pred_subset)
        true_z = _zscore_rows(true_subset)
        pred_rank_z = _rank_zscore_rows(pred_subset)
        true_rank_z = _rank_zscore_rows(true_subset)
        denom = float(len(voxel_idx))

        for pos, idx in enumerate(tqdm(idxs, desc=f"{dataset} random correlation baseline")):
            pool = np.array([p for p in range(len(idxs)) if p != pos], dtype=int)
            sampled_pos = rng.choice(pool, size=n_random, replace=len(pool) < n_random)
            pearson_vals = (true_z[sampled_pos] @ pred_z[pos]) / denom
            spearman_vals = (true_rank_z[sampled_pos] @ pred_rank_z[pos]) / denom
            actual_pearson_subset = float((true_z[pos] @ pred_z[pos]) / denom)
            actual_spearman_subset = float((true_rank_z[pos] @ pred_rank_z[pos]) / denom)

            out.at[idx, "pearson_baseline_actual"] = actual_pearson_subset
            out.at[idx, "pearson_random_mean"] = float(np.nanmean(pearson_vals))
            out.at[idx, "pearson_random_std"] = float(np.nanstd(pearson_vals))
            out.at[idx, "pearson_minus_random"] = float(actual_pearson_subset - np.nanmean(pearson_vals))
            out.at[idx, "pearson_random_percentile"] = float((np.sum(pearson_vals < actual_pearson_subset) + 1) / (np.sum(np.isfinite(pearson_vals)) + 1))
            out.at[idx, "spearman_baseline_actual"] = actual_spearman_subset
            out.at[idx, "spearman_random_mean"] = float(np.nanmean(spearman_vals))
            out.at[idx, "spearman_random_std"] = float(np.nanstd(spearman_vals))
            out.at[idx, "spearman_minus_random"] = float(actual_spearman_subset - np.nanmean(spearman_vals))
            out.at[idx, "spearman_random_percentile"] = float((np.sum(spearman_vals < actual_spearman_subset) + 1) / (np.sum(np.isfinite(spearman_vals)) + 1))
    out["random_baseline_n"] = n_random
    out["random_baseline_n_voxels"] = len(voxel_idx)
    return out


def finite_box_values(values: Iterable[float]) -> np.ndarray:
    vals = pd.Series(values).dropna().to_numpy()
    return vals if len(vals) else np.asarray([np.nan])


def finite_sem(values: Iterable[float]) -> float:
    series = pd.Series(values).dropna()
    return float(series.std() / np.sqrt(max(series.count(), 1)))


def semantic_similarity(st_model, st_util, generated: str, reference: str) -> float:
    emb1 = st_model.encode(generated, convert_to_tensor=True)
    emb2 = st_model.encode(reference, convert_to_tensor=True)
    return float(st_util.cos_sim(emb1, emb2))


def bertscore_single(bert_score_fn, generated: str, reference: str, model_type: str) -> tuple[float, float, float]:
    p, r, f1 = bert_score_fn(
        cands=[generated],
        refs=[reference],
        lang="en",
        model_type=model_type,
        verbose=False,
    )
    return float(p[0]), float(r[0]), float(f1[0])


def nvlm_latent_similarity(nvlm, brain_query_emb: torch.Tensor, generated: str) -> float:
    nvlm._ensure_projection_heads()
    with torch.no_grad():
        raw_emb = nvlm._encode_text(generated)
        z_text = nvlm._proj_head_text_infonce(raw_emb.to(nvlm.device))
        z_text = F.normalize(z_text, dim=-1).cpu()
    z_brain = brain_query_emb.cpu()
    if z_brain.dim() == 1:
        z_brain = z_brain.unsqueeze(0)
    return float(F.cosine_similarity(z_brain, z_text))


def format_context_summary(table: pd.DataFrame) -> str:
    lines = []
    for _, row in table.iterrows():
        lines.append(f"[{row.get('dataset', '?')}] sim={row.get('cosine_similarity', float('nan')):.3f} | {row.get('title', '')}")
    return "\n".join(lines)


def run_b2t_sample(
    *,
    nvlm,
    st_model,
    st_util,
    bert_score_fn,
    bertscore_model: str,
    llm_backend: str,
    llm_model: str,
    b2t_datasets: list[str],
    b2t_top_k: int,
    b2t_sim_threshold: float,
    name,
    latent,
    short_gt,
    long_gt,
    short_prompt,
    long_prompt="",
    short_tokens=64,
    long_tokens=512,
    datasets=None,
) -> list[dict[str, Any]]:
    try:
        result = nvlm.brain(latent).to_text(datasets=datasets or b2t_datasets)
        all_table = result.top_k(b2t_top_k)
        table = all_table[all_table["cosine_similarity"] > b2t_sim_threshold]
        if table.empty:
            table = all_table
        if len(table) > b2t_top_k:
            table = table.nlargest(b2t_top_k, "cosine_similarity").reset_index(drop=True)

        records = []
        for mode, prompt, gt, tokens in [
            ("short", short_prompt, short_gt, short_tokens),
            ("long", long_prompt, long_gt, long_tokens),
        ]:
            generated = nvlm.generate_llm_response(
                backend=llm_backend,
                model_name=llm_model,
                table=table,
                user_prompt=prompt,
                max_new_tokens=tokens,
                verbose=False,
            )
            bert_p, bert_r, bert_f1 = bertscore_single(bert_score_fn, generated, gt, bertscore_model)
            records.append(
                {
                    "generated": generated,
                    "gt_text": gt,
                    "bert_p": bert_p,
                    "bert_r": bert_r,
                    "bert_f1": bert_f1,
                    "sem_sim": semantic_similarity(st_model, st_util, generated, gt),
                    "nvlm_sim": nvlm_latent_similarity(nvlm, result.query_embeddings, generated),
                    "name": name,
                    "mode": mode,
                    "context_summary": format_context_summary(table),
                }
            )
        return records
    except Exception as exc:
        print(f"[B2T error] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return []


def make_b2t_runner(**config):
    """Return a notebook-friendly positional B2T runner bound to local config."""

    def runner(name, latent, short_gt, long_gt, short_prompt, long_prompt="", short_tokens=64, long_tokens=512, datasets=None):
        return run_b2t_sample(
            **config,
            name=name,
            latent=latent,
            short_gt=short_gt,
            long_gt=long_gt,
            short_prompt=short_prompt,
            long_prompt=long_prompt,
            short_tokens=short_tokens,
            long_tokens=long_tokens,
            datasets=datasets,
        )

    return runner


def normalize_label_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def network_aliases(row: dict[str, Any]) -> list[str]:
    aliases = []
    for value in [row["display"], row.get("mapped_terms", ""), row.get("raw_aliases", "")]:
        aliases.extend([normalize_label_text(x) for x in str(value).split(";")])
    aliases.append(normalize_label_text(row["network_key"].replace("_", " ")))
    return [x for x in dict.fromkeys(aliases) if x]


def predict_network_label(text: str, network_info: pd.DataFrame, st_model, st_util, min_semantic_margin: float = 0.02):
    network_rows = network_info.to_dict("records")
    alias_map = {row["network_key"]: network_aliases(row) for row in network_rows}
    text_norm = normalize_label_text(text)
    alias_hits = []
    for key, aliases in alias_map.items():
        for alias in aliases:
            if alias and re.search(rf"\b{re.escape(alias)}\b", text_norm):
                alias_hits.append((key, alias))
                break
    if len(alias_hits) == 1:
        return alias_hits[0][0], "alias", alias_hits[0][1], 1.0

    label_texts = [f"{row['display']}. {row['long_definition']}" for row in network_rows]
    generated_emb = st_model.encode(text, convert_to_tensor=True)
    label_emb = st_model.encode(label_texts, convert_to_tensor=True)
    sims = st_util.cos_sim(generated_emb, label_emb).cpu().numpy().ravel()
    order = sims.argsort()[::-1]
    keys = [row["network_key"] for row in network_rows]
    margin = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else float("nan")
    method = "semantic" if margin >= min_semantic_margin else "semantic_low_margin"
    best_row = network_rows[order[0]]
    return keys[order[0]], method, best_row["display"], float(sims[order[0]])


def add_network_label_accuracy(df: pd.DataFrame, *, networks_data: dict[str, dict[str, Any]], network_info: pd.DataFrame, st_model, st_util) -> pd.DataFrame:
    out = df.copy()
    preds = out["generated"].apply(lambda text: predict_network_label(text, network_info, st_model, st_util))
    out["pred_network_key"] = [p[0] for p in preds]
    out["label_match_method"] = [p[1] for p in preds]
    out["label_match_evidence"] = [p[2] for p in preds]
    out["label_match_score"] = [p[3] for p in preds]
    out["true_network_key"] = out["name"].map(lambda name: networks_data[name]["network_key"])
    out["network_label_correct"] = out["pred_network_key"] == out["true_network_key"]
    return out


def make_network_label_accuracy_adder(**config):
    """Return a dataframe annotator bound to network labels and embedding model."""

    def adder(df: pd.DataFrame) -> pd.DataFrame:
        return add_network_label_accuracy(df, **config)

    return adder


def normalize_term_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.split("/")[0]
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def split_gold_terms(value) -> list[str]:
    if pd.isna(value):
        return []
    terms = []
    for chunk in re.split(r";|\n|\|", str(value)):
        term = chunk.strip()
        if term:
            terms.append(term)
    return terms


def terms_for_dataset(dataset_name: str) -> set[str]:
    df = load_dataset(dataset_name)
    if not isinstance(df, pd.DataFrame):
        return set()
    for col in ["term", "title", "name", "label"]:
        if col in df.columns:
            return {normalize_term_text(x) for x in df[col].dropna().astype(str)}
    return set()


def network_gold_terms(sample_name: str, networks_data: dict[str, dict[str, Any]], network_labels_df: pd.DataFrame) -> list[str]:
    d = networks_data[sample_name]
    label_rows = network_labels_df[network_labels_df["raw_network_label"].astype(str) == str(d["raw_network_label"])]
    if label_rows.empty:
        label_rows = network_labels_df[network_labels_df["network_key"] == d["network_key"]]
    terms = []
    for col in ["mapped_terms", "region_terms", "cognitive_terms"]:
        if col in label_rows.columns:
            for value in label_rows[col].dropna().tolist():
                terms.extend(split_gold_terms(value))
    return list(dict.fromkeys(terms))


def table_terms(table: pd.DataFrame) -> list[str]:
    if table is None or table.empty:
        return []
    return [str(x) for x in table["title"].fillna("").tolist() if str(x).strip()]


def retrieval_table_for_sample(
    *,
    nvlm,
    cache: dict[tuple[str, str, str], pd.DataFrame],
    term_datasets_by_eval_dataset: dict[str, list[str]],
    b2t_term_top_k: int,
    b2t_evidence_top_k: int,
    b2t_sim_threshold: float,
    latent,
    dataset_name: str,
    sample: str,
) -> pd.DataFrame:
    cache_key = (dataset_name, str(sample), "evidence")
    if cache_key in cache:
        return cache[cache_key]
    datasets = term_datasets_by_eval_dataset[dataset_name]
    result = nvlm.brain(latent).to_text(datasets=datasets)
    all_table = result.top_k(max(b2t_term_top_k, b2t_evidence_top_k))
    table = all_table[all_table["cosine_similarity"] > b2t_sim_threshold]
    if table.empty:
        table = all_table
    table = table.nlargest(max(b2t_term_top_k, b2t_evidence_top_k), "cosine_similarity").reset_index(drop=True)
    cache[cache_key] = table
    return table


def full_retrieval_table_for_sample(
    *,
    nvlm,
    cache: dict[tuple[str, str, str], pd.DataFrame],
    term_datasets_by_eval_dataset: dict[str, list[str]],
    latent,
    dataset_name: str,
    sample: str | None = None,
) -> pd.DataFrame:
    cache_key = (dataset_name, str(sample), "full") if sample is not None else None
    if cache_key is not None and cache_key in cache:
        return cache[cache_key]
    datasets = term_datasets_by_eval_dataset[dataset_name]
    result = nvlm.brain(latent).to_text(datasets=datasets)
    n_candidates = sum(int(scores.shape[0]) for scores in result.scores_by_dataset.values())
    table = result.top_k(n_candidates)
    if cache_key is not None:
        cache[cache_key] = table
    return table


def unique_ranked_terms_from_table(table: pd.DataFrame) -> list[str]:
    if table is None or table.empty:
        return []
    ranked = table.sort_values("cosine_similarity", ascending=False, kind="mergesort").copy()
    ranked["_normalized_term"] = ranked["title"].map(normalize_term_text)
    ranked = ranked[ranked["_normalized_term"] != ""]
    ranked = ranked.drop_duplicates("_normalized_term", keep="first")
    return ranked["title"].astype(str).tolist()


def pubmed_abstract_lookup() -> dict[str, str]:
    df_pubs = load_dataset("pubmed_text")
    pmid_col = "pmid" if "pmid" in df_pubs.columns else df_pubs.columns[0]
    abs_col = "description" if "description" in df_pubs.columns else ("abstract" if "abstract" in df_pubs.columns else None)
    if abs_col is None:
        return {}
    return (
        df_pubs.assign(_pmid_key=lambda df: df[pmid_col].astype(str))
        .drop_duplicates("_pmid_key")
        .set_index("_pmid_key")[abs_col]
        .astype(str)
        .to_dict()
    )


def dataset_records_for_retrieval_eval(
    *,
    run_networks: bool,
    run_pubmed: bool,
    run_neurovault: bool,
    networks_data: dict[str, dict[str, Any]],
    network_labels_df: pd.DataFrame,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    pubmed_abs_lookup: dict[str, str],
):
    if run_networks:
        for name, d in networks_data.items():
            yield "networks", name, d["latent"], network_gold_terms(name, networks_data, network_labels_df), {"long_description": d["long_gt"]}
    if run_pubmed:
        for d in pubmed_eval:
            pmid = str(d["pmid"])
            references = {"summary": d["long_gt"]}
            abstract = pubmed_abs_lookup.get(pmid, "")
            if abstract:
                references["abstract"] = abstract
            yield "pubmed", pmid, d["latent"], [], references
    if run_neurovault:
        for d in neurovault_eval:
            yield "neurovault", str(d["doi"]), d["latent"], [], {"abstract": d["long_gt"]}


def k_from_normalized_k(normalized_k: float, n_candidates: int) -> int:
    if n_candidates <= 0 or normalized_k <= 0:
        return 0
    return min(n_candidates, max(1, int(np.ceil(float(normalized_k) * n_candidates))))


def auc_trapezoid(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def exact_term_ranking_outputs(
    *,
    dataset: str,
    sample: str,
    gold_terms: list[str],
    retrieved_terms: list[str],
    term_eval_normalized_ks: Iterable[float],
    term_recall_curve_normalized_ks: Iterable[float],
    reachable_terms: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    gold_norm_all = {normalize_term_text(t) for t in gold_terms if normalize_term_text(t)}
    if reachable_terms is None:
        gold_norm = gold_norm_all
        excluded = set()
    else:
        gold_norm = gold_norm_all & reachable_terms
        excluded = gold_norm_all - reachable_terms

    retrieved_norm = []
    seen = set()
    for term in retrieved_terms:
        norm = normalize_term_text(term)
        if norm and norm not in seen:
            retrieved_norm.append(norm)
            seen.add(norm)

    if not gold_norm or not retrieved_norm:
        return [], [], None

    n_candidates = len(retrieved_norm)
    first_hit_rank = next((i + 1 for i, term in enumerate(retrieved_norm) if term in gold_norm), np.nan)
    normalized_first_hit_rank = float(first_hit_rank / n_candidates) if not pd.isna(first_hit_rank) else np.nan

    metric_rows = []
    for normalized_k_target in term_eval_normalized_ks:
        k = k_from_normalized_k(normalized_k_target, n_candidates)
        topk = retrieved_norm[:k]
        hits = set(topk) & gold_norm
        normalized_k = k / n_candidates if n_candidates else np.nan
        metric_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "normalized_k_target": float(normalized_k_target),
                "normalized_k": float(normalized_k),
                "k": int(k),
                "n_candidate_terms": int(n_candidates),
                "n_gold_terms": len(gold_norm),
                "n_unreachable_gold_terms": len(excluded),
                "n_retrieved_terms": len(topk),
                "n_hits": len(hits),
                "precision_at_normalized_k": len(hits) / max(len(topk), 1),
                "recall_at_normalized_k": len(hits) / len(gold_norm),
                "hit_at_normalized_k": bool(hits),
                "mrr_at_normalized_k": 0.0 if pd.isna(first_hit_rank) or first_hit_rank > k else 1.0 / float(first_hit_rank),
                "normalized_first_hit_rank": normalized_first_hit_rank,
                "expected_random_recall_at_normalized_k": float(normalized_k_target),
                "matched_terms": "; ".join(sorted(hits)),
            }
        )

    curve_rows = []
    recall_values = []
    normalized_ks = list(term_recall_curve_normalized_ks)
    for normalized_k_target in normalized_ks:
        k = k_from_normalized_k(normalized_k_target, n_candidates)
        topk = retrieved_norm[:k]
        hits = set(topk) & gold_norm
        recall = len(hits) / len(gold_norm)
        recall_values.append(recall)
        curve_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "normalized_k_target": float(normalized_k_target),
                "normalized_k": float(k / n_candidates) if n_candidates else np.nan,
                "k": int(k),
                "n_candidate_terms": int(n_candidates),
                "n_gold_terms": len(gold_norm),
                "recall_at_normalized_k": float(recall),
                "expected_random_recall_at_normalized_k": float(normalized_k_target),
            }
        )

    auc = auc_trapezoid(normalized_ks, recall_values)
    return metric_rows, curve_rows, {
        "dataset": dataset,
        "sample": sample,
        "n_candidate_terms": int(n_candidates),
        "n_gold_terms": len(gold_norm),
        "n_unreachable_gold_terms": len(excluded),
        "recall_auc": float(auc),
        "expected_random_recall_auc": 0.5,
        "recall_auc_minus_random": float(auc - 0.5),
        "normalized_first_hit_rank": normalized_first_hit_rank,
    }


def run_network_gold_term_ranking(
    *,
    nvlm,
    networks_data: dict[str, dict[str, Any]],
    network_labels_df: pd.DataFrame,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    pubmed_abs_lookup: dict[str, str],
    network_candidate_terms: set[str],
    term_datasets_by_eval_dataset: dict[str, list[str]],
    retrieval_table_cache: dict[tuple[str, str, str], pd.DataFrame],
    term_eval_normalized_ks: Iterable[float],
    term_recall_curve_normalized_ks: Iterable[float],
    b2t_term_example_top_k: int,
    output_dir,
    run_networks: bool = True,
    run_pubmed: bool = True,
    run_neurovault: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    term_metric_rows = []
    term_curve_rows = []
    term_auc_rows = []
    term_examples = []
    retrieval_records = list(
        dataset_records_for_retrieval_eval(
            run_networks=run_networks,
            run_pubmed=run_pubmed,
            run_neurovault=run_neurovault,
            networks_data=networks_data,
            network_labels_df=network_labels_df,
            pubmed_eval=pubmed_eval,
            neurovault_eval=neurovault_eval,
            pubmed_abs_lookup=pubmed_abs_lookup,
        )
    )

    for dataset, sample, latent, gold_terms, _references in tqdm(retrieval_records, desc="Network normalized gold-term ranking"):
        if dataset != "networks":
            continue
        table = full_retrieval_table_for_sample(
            nvlm=nvlm,
            cache=retrieval_table_cache,
            term_datasets_by_eval_dataset=term_datasets_by_eval_dataset,
            latent=latent,
            dataset_name=dataset,
            sample=sample,
        )
        retrieved_terms = unique_ranked_terms_from_table(table)
        metric_rows, curve_rows, auc_row = exact_term_ranking_outputs(
            dataset=dataset,
            sample=sample,
            gold_terms=gold_terms,
            retrieved_terms=retrieved_terms,
            term_eval_normalized_ks=term_eval_normalized_ks,
            term_recall_curve_normalized_ks=term_recall_curve_normalized_ks,
            reachable_terms=network_candidate_terms,
        )
        term_metric_rows.extend(metric_rows)
        term_curve_rows.extend(curve_rows)
        if auc_row is not None:
            term_auc_rows.append(auc_row)
        term_examples.append(
            {
                "dataset": dataset,
                "sample": sample,
                "gold_terms": "; ".join(gold_terms[:50]),
                "n_ranked_candidate_terms": len(retrieved_terms),
                "top_terms": "; ".join(retrieved_terms[:b2t_term_example_top_k]),
            }
        )

    metrics_df = pd.DataFrame(term_metric_rows)
    curve_df = pd.DataFrame(term_curve_rows)
    auc_df = pd.DataFrame(term_auc_rows)
    examples_df = pd.DataFrame(term_examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "b2t_approach1_gold_term_ranking_metrics.csv", index=False)
    curve_df.to_csv(output_dir / "b2t_approach1_gold_term_recall_curve.csv", index=False)
    auc_df.to_csv(output_dir / "b2t_approach1_gold_term_recall_auc.csv", index=False)
    examples_df.to_csv(output_dir / "b2t_approach1_gold_term_examples.csv", index=False)
    return metrics_df, curve_df, auc_df, examples_df


def project_text_latents_to_shared(nvlm, text_latents, batch_size: int = 4096) -> torch.Tensor:
    nvlm._ensure_projection_heads()
    x = torch.as_tensor(text_latents).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            z = nvlm._proj_head_text_infonce(x[start : start + batch_size].to(nvlm.device))
            chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


def project_brain_latents_to_shared(nvlm, brain_latents, batch_size: int = 4096) -> torch.Tensor:
    nvlm._ensure_projection_heads()
    batch = as_latent_batch(brain_latents).float()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            z = nvlm._proj_head_image(batch[start : start + batch_size].to(nvlm.device))
            chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


def mesh_descriptor_name(term: str) -> str:
    return str(term).split("/")[0].strip()


def load_pubmed_mesh_gold_annotations_or_none():
    try:
        annotations = load_dataset("pubmed_mesh_annotations")
        print(f"Loaded PubMed MeSH annotations for {len(annotations):,} PMIDs")
        return annotations
    except Exception as exc:
        print("PubMed MeSH gold annotations are unavailable; skipping MeSH term-ranking diagnostics.")
        print(f"Loader error: {type(exc).__name__}: {exc}")
        return None


def run_pubmed_mesh_gold_ranking(
    *,
    nvlm,
    pubmed_eval: list[dict[str, Any]],
    pubmed_b2t_dataset: str,
    mesh_brain_rankable_node_types: Iterable[str],
    b2t_term_example_top_k: int,
    output_dir,
    run_pubmed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotations = load_pubmed_mesh_gold_annotations_or_none()
    node_type_by_term: dict[str, str] = {}
    if annotations is not None:
        try:
            mesh_nodes_for_gold = load_dataset("mesh_kg_nodes")
            if "node_type" in mesh_nodes_for_gold.columns:
                name_col = "name" if "name" in mesh_nodes_for_gold.columns else "term"
                node_type_by_term = {
                    normalize_term_text(row[name_col]): row["node_type"]
                    for _, row in mesh_nodes_for_gold.iterrows()
                    if pd.notna(row.get(name_col)) and pd.notna(row.get("node_type"))
                }
        except Exception as exc:
            print(f"Could not load mesh_kg_nodes for node-type filtering: {type(exc).__name__}: {exc}")

    allowed_mesh_types = set(mesh_brain_rankable_node_types)
    mesh_candidate_df = load_dataset(pubmed_b2t_dataset).copy()
    mesh_candidate_latents, _mesh_candidate_terms = load_latent(pubmed_b2t_dataset)

    if len(mesh_candidate_df) != len(mesh_candidate_latents):
        raise ValueError(
            f"MeSH candidate metadata/latent length mismatch: metadata={len(mesh_candidate_df)} latents={len(mesh_candidate_latents)}"
        )

    mesh_term_col = next((col for col in ["term", "title", "name", "label"] if col in mesh_candidate_df.columns), None)
    if mesh_term_col is None:
        raise KeyError(f"{pubmed_b2t_dataset} must contain one of term/title/name/label columns.")

    mesh_candidate_df["term"] = mesh_candidate_df[mesh_term_col].astype(str).map(mesh_descriptor_name)
    mesh_candidate_df["normalized_term"] = mesh_candidate_df["term"].map(normalize_term_text)
    mesh_candidate_df["node_type"] = mesh_candidate_df["normalized_term"].map(node_type_by_term).fillna("")
    keep_mask = mesh_candidate_df["node_type"].isin(allowed_mesh_types).to_numpy() if node_type_by_term else np.ones(len(mesh_candidate_df), dtype=bool)
    mesh_candidate_df = mesh_candidate_df.loc[keep_mask].copy()
    mesh_candidate_latents = mesh_candidate_latents[keep_mask]
    unique_mask = ~mesh_candidate_df["normalized_term"].duplicated(keep="first").to_numpy()
    mesh_candidate_df = mesh_candidate_df.loc[unique_mask].reset_index(drop=True)
    mesh_candidate_latents = mesh_candidate_latents[unique_mask]

    mesh_candidate_embeddings = project_text_latents_to_shared(nvlm, mesh_candidate_latents)
    mesh_candidate_terms = mesh_candidate_df["term"].astype(str).tolist()
    mesh_norm_to_idx = {norm: i for i, norm in enumerate(mesh_candidate_df["normalized_term"].astype(str))}
    mesh_candidate_norms = set(mesh_norm_to_idx)

    def pubmed_mesh_gold_terms(pmid) -> list[str]:
        if annotations is None:
            return []
        out = []
        for term in annotations.get(str(pmid), []):
            base = mesh_descriptor_name(term)
            norm = normalize_term_text(base)
            if not norm or norm not in mesh_candidate_norms:
                continue
            if node_type_by_term and node_type_by_term.get(norm) not in allowed_mesh_types:
                continue
            out.append(base)
        return list(dict.fromkeys(out))

    mesh_records = []
    mesh_true_indices = []
    mesh_true_terms = []
    if run_pubmed and annotations is not None:
        for d in pubmed_eval:
            positives = []
            terms = []
            for term in pubmed_mesh_gold_terms(str(d["pmid"])):
                idx = mesh_norm_to_idx.get(normalize_term_text(term))
                if idx is not None:
                    positives.append(idx)
                    terms.append(mesh_candidate_terms[idx])
            if positives:
                mesh_records.append(d)
                mesh_true_indices.append(set(positives))
                mesh_true_terms.append(sorted(set(terms)))

    if mesh_records:
        mesh_brain_embeddings = project_brain_latents_to_shared(nvlm, [d["latent"] for d in mesh_records])
        mesh_scores = (mesh_brain_embeddings @ mesh_candidate_embeddings.T).detach().cpu().numpy()
        mesh_metrics = multi_positive_ranking_metrics(mesh_scores, mesh_true_indices, ks=(1, 5, 10, 50), ndcg_k=10)
        mesh_order = np.argsort(-mesh_scores, axis=1)
        metrics_df = pd.DataFrame(
            [
                {
                    "dataset": "pubmed_mesh",
                    "n_queries": mesh_metrics["n_queries"],
                    "n_candidates": mesh_metrics["n_candidates"],
                    "mesh_paper_recall_curve_auc": mesh_metrics["paper_recall_curve_auc"],
                    "mesh_normalized_k_recall_curve_auc": mesh_metrics["normalized_k_recall_curve_auc"],
                    "mesh_recall@1": mesh_metrics["recall@1"],
                    "mesh_recall@5": mesh_metrics["recall@5"],
                    "mesh_recall@10": mesh_metrics["recall@10"],
                    "mesh_recall@50": mesh_metrics["recall@50"],
                    "mesh_map": mesh_metrics["map"],
                    "mesh_mrr": mesh_metrics["mrr"],
                    "mesh_ndcg@10": mesh_metrics["ndcg@10"],
                    "mesh_median_best_true_term_rank": mesh_metrics["median_best_true_term_rank"],
                    "allowed_node_types": ";".join(mesh_brain_rankable_node_types),
                }
            ]
        )

        counts = np.zeros(len(mesh_candidate_terms), dtype=np.int64)
        for i, positives in enumerate(mesh_true_indices):
            ranks_by_candidate = np.empty(len(mesh_candidate_terms), dtype=np.int64)
            ranks_by_candidate[mesh_order[i]] = np.arange(1, len(mesh_candidate_terms) + 1)
            counts[min(int(ranks_by_candidate[j]) for j in positives) - 1] += 1
        recall_curve = np.cumsum(counts) / float(len(mesh_true_indices))
        norm_k = normalized_k_values(len(mesh_candidate_terms)).cpu().numpy()
        curve_df = pd.DataFrame(
            {
                "dataset": "pubmed_mesh",
                "k": np.arange(1, len(mesh_candidate_terms) + 1),
                "normalized_k": norm_k,
                "recall_at_normalized_k": recall_curve,
                "expected_random_recall_at_normalized_k": norm_k,
            }
        )

        example_rows = []
        for i, d in enumerate(mesh_records):
            positives = mesh_true_indices[i]
            ranks_by_candidate = np.empty(len(mesh_candidate_terms), dtype=np.int64)
            ranks_by_candidate[mesh_order[i]] = np.arange(1, len(mesh_candidate_terms) + 1)
            top = mesh_order[i, :b2t_term_example_top_k]
            example_rows.append(
                {
                    "dataset": "pubmed_mesh",
                    "sample": str(d["pmid"]),
                    "true_mesh_terms": "; ".join(mesh_true_terms[i]),
                    "best_true_term_rank": min(int(ranks_by_candidate[j]) for j in positives),
                    "top_terms": "; ".join(mesh_candidate_terms[j] for j in top),
                    "top_scores": "; ".join(f"{mesh_scores[i, j]:.6f}" for j in top),
                    "hit@5": bool(any(j in positives for j in mesh_order[i, :5])),
                    "hit@10": bool(any(j in positives for j in mesh_order[i, :10])),
                }
            )
        examples_df = pd.DataFrame(example_rows)
    else:
        metrics_df = pd.DataFrame()
        curve_df = pd.DataFrame()
        examples_df = pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "b2t_pubmed_mesh_gold_term_ranking_metrics.csv", index=False)
    metrics_df.to_json(output_dir / "b2t_pubmed_mesh_gold_term_ranking_metrics.json", orient="records", indent=2)
    curve_df.to_csv(output_dir / "b2t_pubmed_mesh_gold_term_recall_curve.csv", index=False)
    curve_df.to_json(output_dir / "b2t_pubmed_mesh_gold_term_recall_curve.json", orient="records", indent=2)
    examples_df.to_csv(output_dir / "b2t_pubmed_mesh_gold_term_examples.csv", index=False)
    examples_df.to_json(output_dir / "b2t_pubmed_mesh_gold_term_examples.json", orient="records", indent=2)
    return metrics_df, curve_df, examples_df


def brain_latents_for_generated_group(
    df: pd.DataFrame,
    *,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
) -> list[Any]:
    pubmed_by_pmid = {str(d["pmid"]): d["latent"] for d in pubmed_eval}
    neurovault_by_doi = {str(d["doi"]): d["latent"] for d in neurovault_eval}
    brain_embs = []
    for _, row in df.iterrows():
        source = row["dataset"]
        name = str(row["name"])
        if source == "networks":
            brain_embs.append(networks_data[name]["latent"])
        elif source == "pubmed":
            brain_embs.append(pubmed_by_pmid[name])
        elif source == "neurovault":
            brain_embs.append(neurovault_by_doi[name])
    return brain_embs


def generated_text_retrieval_curve(
    nvlm,
    df: pd.DataFrame,
    *,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
) -> tuple[float, pd.DataFrame]:
    if len(df) < 2:
        return np.nan, pd.DataFrame()
    generated = df["generated"].astype(str).tolist()
    z_text = project_text_latents_to_shared(nvlm, nvlm._encode_text(generated))
    z_brain = project_brain_latents_to_shared(
        nvlm,
        brain_latents_for_generated_group(df, networks_data=networks_data, pubmed_eval=pubmed_eval, neurovault_eval=neurovault_eval),
    )
    scores = z_text @ z_brain.T
    order = scores.argsort(dim=1, descending=True)
    target = torch.arange(len(df)).view(-1, 1)
    first_hits = order.eq(target).int().argmax(dim=1)
    hit_counts = torch.bincount(first_hits.cpu(), minlength=len(df)).float()
    recall_curve = torch.cumsum(hit_counts, dim=0) / float(len(df))
    normalized_k = normalized_k_values(len(df)).cpu().numpy()
    auc = normalized_recall_curve_auc(recall_curve)
    curve_df = pd.DataFrame(
        {
            "k": np.arange(1, len(df) + 1),
            "normalized_k": normalized_k,
            "recall_at_normalized_k": recall_curve.cpu().numpy(),
            "expected_random_recall_at_normalized_k": normalized_k,
        }
    )
    return auc, curve_df


def generated_text_metric_summary(
    *,
    nvlm,
    b2t_all: pd.DataFrame,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    output_dir,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generated_text_recall_rows = []
    generated_text_curve_rows = []
    for (dataset, mode), sub in b2t_all.groupby(["dataset", "mode"]):
        sub = sub.reset_index(drop=True)
        auc, curve_df = generated_text_retrieval_curve(
            nvlm,
            sub,
            networks_data=networks_data,
            pubmed_eval=pubmed_eval,
            neurovault_eval=neurovault_eval,
        )
        generated_text_recall_rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "generated_text_normalized_k_recall_curve_auc": auc,
                "n": len(sub),
            }
        )
        if len(curve_df):
            curve_df.insert(0, "mode", mode)
            curve_df.insert(0, "dataset", dataset)
            generated_text_curve_rows.append(curve_df)

    recall_auc_df = pd.DataFrame(generated_text_recall_rows)
    recall_curve_df = pd.concat(generated_text_curve_rows, ignore_index=True) if generated_text_curve_rows else pd.DataFrame()
    summary = b2t_all.groupby(["dataset", "mode"])[["nvlm_sim", "bert_f1", "sem_sim"]].agg(["mean", "std", "count"]).round(3)
    label_summary = pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "b2t_generated_text_metric_summary.csv")
    summary.reset_index().to_json(output_dir / "b2t_generated_text_metric_summary.json", orient="records", indent=2)
    recall_auc_df.round(3).to_csv(output_dir / "b2t_generated_text_recall_auc.csv", index=False)
    recall_auc_df.round(3).to_json(output_dir / "b2t_generated_text_recall_auc.json", orient="records", indent=2)
    if len(recall_curve_df):
        recall_curve_df.to_csv(output_dir / "b2t_generated_text_recall_curve.csv", index=False)
        recall_curve_df.to_json(output_dir / "b2t_generated_text_recall_curve.json", orient="records", indent=2)
    if "network_label_correct" in b2t_all.columns:
        label_summary = b2t_all[b2t_all["dataset"] == "networks"].groupby("mode")["network_label_correct"].agg(["mean", "sum", "count"]).round(3)
        label_summary.to_csv(output_dir / "b2t_generated_text_network_label_accuracy_summary.csv")
        label_summary.reset_index().to_json(output_dir / "b2t_generated_text_network_label_accuracy_summary.json", orient="records", indent=2)
    return summary, recall_auc_df, recall_curve_df, label_summary


def generated_text_pair_baseline(
    *,
    nvlm,
    b2t_all: pd.DataFrame,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for (dataset, mode), sub in b2t_all.groupby(["dataset", "mode"]):
        if len(sub) < 2:
            continue
        sub = sub.reset_index(drop=True)
        z_text = project_text_latents_to_shared(nvlm, nvlm._encode_text(sub["generated"].astype(str).tolist()))
        z_brain = project_brain_latents_to_shared(
            nvlm,
            brain_latents_for_generated_group(sub, networks_data=networks_data, pubmed_eval=pubmed_eval, neurovault_eval=neurovault_eval),
        )
        scores = z_text @ z_brain.T
        eye = torch.eye(len(sub), dtype=torch.bool)
        for val in scores[eye].numpy():
            rows.append({"dataset": dataset, "mode": mode, "pair": "matched", "score": float(val)})
        for val in scores[~eye].numpy():
            rows.append({"dataset": dataset, "mode": mode, "pair": "random/off-diagonal", "score": float(val)})
    return pd.DataFrame(rows)


def paper_record_text(record: dict) -> str:
    title = str(record.get("short_gt", "")).strip()
    body = str(record.get("long_gt", "")).strip()
    if title and body:
        return f"{title}. {body}"
    return title or body


def paper_records_for_dataset(dataset_name: str, pubmed_eval: list[dict[str, Any]], neurovault_eval: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if dataset_name == "pubmed":
        return [
            {
                "dataset": "pubmed",
                "sample": str(d["pmid"]),
                "latent": d["latent"],
                "text_latent": d["text_latent"],
                "text": paper_record_text(d),
            }
            for d in pubmed_eval
            if "text_latent" in d
        ]
    if dataset_name == "neurovault":
        return [
            {
                "dataset": "neurovault",
                "sample": str(d["doi"]),
                "latent": d["latent"],
                "text_latent": d["text_latent"],
                "text": paper_record_text(d),
            }
            for d in neurovault_eval
            if "text_latent" in d
        ]
    raise ValueError(f"Unknown paper retrieval dataset: {dataset_name}")


def semantic_recall_curve(scores: np.ndarray, positives: list[set[int]]) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    n_queries, n_candidates = scores.shape
    best_ranks = []
    for i, pos in enumerate(positives):
        ranks_by_candidate = np.empty(n_candidates, dtype=np.int64)
        ranks_by_candidate[order[i]] = np.arange(1, n_candidates + 1)
        best_ranks.append(min(int(ranks_by_candidate[j]) for j in pos))
    counts = torch.bincount(torch.as_tensor(best_ranks, dtype=torch.long) - 1, minlength=n_candidates)
    return (counts.cumsum(0).float() / float(n_queries)).numpy()


def run_paper_retrieval_eval(
    *,
    nvlm,
    dataset_name: str,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    semantic_neighbors: int = 10,
    brain_batch_size: int = 512,
    text_batch_size: int = 512,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    from neurovlm.semantic_evaluation import exact_pmid_retrieval_metrics, exact_recall_curve, semantic_neighbor_positive_sets

    records = paper_records_for_dataset(dataset_name, pubmed_eval, neurovault_eval)
    if len(records) < 2:
        return {}, pd.DataFrame(), pd.DataFrame()

    ids = [r["sample"] for r in records]
    brain_embeddings = project_brain_latents_to_shared(nvlm, [r["latent"] for r in records], batch_size=brain_batch_size)
    text_embeddings = project_text_latents_to_shared(nvlm, [r["text_latent"] for r in records], batch_size=text_batch_size)
    sim = brain_embeddings @ text_embeddings.T

    brain_to_paper = exact_pmid_retrieval_metrics(sim)
    paper_to_brain = exact_pmid_retrieval_metrics(sim.T)
    semantic_positives = semantic_neighbor_positive_sets(
        text_embeddings,
        n_neighbors=min(semantic_neighbors, max(len(records) - 1, 1)),
    )
    semantic_curve = semantic_recall_curve(sim.detach().cpu().numpy(), semantic_positives)
    semantic_auc = normalized_recall_curve_auc(torch.as_tensor(semantic_curve))

    metrics = {
        "dataset": dataset_name,
        "n_papers": len(records),
        "brain_to_paper_normalized_k_recall_curve_auc": brain_to_paper["normalized_k_recall_curve_auc"],
        "paper_to_brain_normalized_k_recall_curve_auc": paper_to_brain["normalized_k_recall_curve_auc"],
        "semantic_normalized_k_recall_curve_auc": semantic_auc,
    }

    normalized_k = normalized_k_values(len(records)).cpu().numpy()
    curve_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "k": np.arange(1, len(records) + 1),
            "normalized_k": normalized_k,
            "brain_to_paper_recall_curve": exact_recall_curve(sim).cpu().numpy(),
            "paper_to_brain_recall_curve": exact_recall_curve(sim.T).cpu().numpy(),
            "semantic_recall_curve": semantic_curve,
            "random_recall_curve": normalized_k,
        }
    )

    order = torch.argsort(sim, dim=1, descending=True).cpu().numpy()
    example_rows = []
    for i, sample_id in enumerate(ids):
        top = order[i, : min(10, len(ids))].tolist()
        correct_rank = int(np.where(order[i] == i)[0][0] + 1)
        example_rows.append(
            {
                "dataset": dataset_name,
                "sample": sample_id,
                "correct_rank": correct_rank,
                "top10_samples": "|".join(ids[j] for j in top),
                "top10_scores": "|".join(f"{float(sim[i, j]):.6f}" for j in top),
                "reference_text": records[i]["text"][:500],
            }
        )
    return metrics, curve_df, pd.DataFrame(example_rows)


def run_paper_retrieval_evaluations(
    *,
    nvlm,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    output_dir,
    run_pubmed: bool = True,
    run_neurovault: bool = True,
    semantic_neighbors: int = 10,
    brain_batch_size: int = 512,
    text_batch_size: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paper_metric_rows = []
    paper_curve_frames = []
    paper_example_frames = []
    for dataset_name, should_run in [("pubmed", run_pubmed), ("neurovault", run_neurovault)]:
        if not should_run:
            continue
        metrics, curves, examples = run_paper_retrieval_eval(
            nvlm=nvlm,
            dataset_name=dataset_name,
            pubmed_eval=pubmed_eval,
            neurovault_eval=neurovault_eval,
            semantic_neighbors=semantic_neighbors,
            brain_batch_size=brain_batch_size,
            text_batch_size=text_batch_size,
        )
        if metrics:
            paper_metric_rows.append(metrics)
            paper_curve_frames.append(curves)
            paper_example_frames.append(examples)

    metrics_df = pd.DataFrame(paper_metric_rows)
    curves_df = pd.concat(paper_curve_frames, ignore_index=True) if paper_curve_frames else pd.DataFrame()
    examples_df = pd.concat(paper_example_frames, ignore_index=True) if paper_example_frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "b2t_paper_retrieval_metrics.csv", index=False)
    metrics_df.to_json(output_dir / "b2t_paper_retrieval_metrics.json", orient="records", indent=2)
    curves_df.to_csv(output_dir / "b2t_paper_retrieval_curves.csv", index=False)
    curves_df.to_json(output_dir / "b2t_paper_retrieval_curves.json", orient="records", indent=2)
    examples_df.to_csv(output_dir / "b2t_paper_retrieval_examples.csv", index=False)
    examples_df.to_json(output_dir / "b2t_paper_retrieval_examples.json", orient="records", indent=2)
    return metrics_df, curves_df, examples_df


def predownload_hf_model(model_name: str, tokenizer_cls, model_cls) -> None:
    print(f"Pre-downloading Hugging Face model for BERTScore: {model_name}")
    tokenizer_cls.from_pretrained(model_name, use_fast=False)
    model_cls.from_pretrained(model_name)
    print("BERTScore model is cached.")


def sensitivity_rows_for_df(
    df: pd.DataFrame,
    dataset: str,
    *,
    pcts: Iterable[float],
    spin_test_n_perm: int,
    spin_test_random_state: int,
    spin_fsaverage_density: str,
    spin_require_neuromaps: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset} Dice sensitivity"):
        for pct in pcts:
            if not bool(row.get("surface_metric_eligible", True)):
                rows.append(
                    {
                        "dataset": dataset,
                        "sample": row["name"],
                        "pct": pct,
                        "top_fraction": (100 - pct) / 100,
                        "dice": np.nan,
                        "spin_p_value": np.nan,
                        "spin_significant": False,
                        "method": "not_run_surface_ineligible",
                    }
                )
                continue
            has_surface = all(row.get(k) is not None for k in ["_pred_lh", "_pred_rh", "_true_lh", "_true_rh"])
            spin_p = np.nan
            spin_sig = False
            method = "volume_masker_percentile"
            if has_surface:
                try:
                    nct = nct_dice_spin_test_surface(
                        row["_pred_lh"],
                        row["_pred_rh"],
                        row["_true_lh"],
                        row["_true_rh"],
                        pct=pct,
                        n_perm=spin_test_n_perm,
                        random_state=spin_test_random_state,
                        density=spin_fsaverage_density,
                    )
                    dice_val = nct.dice_pct
                    spin_p = nct.spin_p_value
                    spin_sig = nct.spin_significant
                    method = "surface_fsaverage_percentile_nct"
                except Exception:
                    if spin_require_neuromaps:
                        raise
                    dice_val = dice_percentile(row["_brain_pred"], row["_brain_true"], pct=pct)
            else:
                dice_val = dice_percentile(row["_brain_pred"], row["_brain_true"], pct=pct)
            rows.append(
                {
                    "dataset": dataset,
                    "sample": row["name"],
                    "pct": pct,
                    "top_fraction": (100 - pct) / 100,
                    "dice": float(dice_val),
                    "spin_p_value": float(spin_p) if np.isfinite(spin_p) else np.nan,
                    "spin_significant": bool(spin_sig),
                    "method": method,
                }
            )
    return rows
