"""Text-to-brain evaluation metrics and notebook workflow helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from tqdm.notebook import tqdm

from neurovlm.metrics import (
    as_latent_batch,
    dice_percentile,
    mni152_to_fsaverage_arrays,
    nct_dice_spin_test_nifti,
    nct_dice_spin_test_surface,
    pearson_correlation,
)


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

__all__ = [
    "add_random_correlation_baseline",
    "build_surface_eligibility_mask",
    "cortical_top_mass_fraction",
    "decode_latents_to_brain",
    "dice_percentile",
    "evaluate_t2b_sample",
    "finite_box_values",
    "finite_sem",
    "make_t2b_runner",
    "mni152_to_fsaverage_arrays",
    "nct_dice_spin_test_nifti",
    "nct_dice_spin_test_surface",
    "pearson_correlation",
    "select_true_brain_for_eval",
    "sensitivity_rows_for_df",
    "surface_metric_eligibility",
]
