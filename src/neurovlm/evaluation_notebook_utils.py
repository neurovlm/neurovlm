"""Reusable helpers for the 21/22 evaluation notebooks."""

from __future__ import annotations

import re
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
    pearson_correlation,
)


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
