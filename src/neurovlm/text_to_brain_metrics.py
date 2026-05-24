"""Text-to-brain evaluation metrics and notebook workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from os import PathLike
import tempfile
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from tqdm.notebook import tqdm

from neurovlm.metric_utils import as_latent_batch


def pearson_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Pearson correlation between a true and predicted brain map.

    Captures linear correspondence independently of scale, making it a
    natural companion to MSE and Dice for evaluating text-to-brain predictions.

    Parameters
    ----------
    y_true : array-like
        True brain activation map (any shape; flattened internally).
    y_pred : array-like
        Predicted brain activation map (same shape).

    Returns
    -------
    r : float
        Pearson *r* in [−1, 1].  Returns 0.0 when either array has zero
        variance.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def psnr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio between a true and predicted brain map.

    Provides a decibel-scale quality measure that is intuitive for reporting
    reconstruction fidelity.  Higher values indicate better reconstruction.

    Parameters
    ----------
    y_true : array-like
        True brain activation map.
    y_pred : array-like
        Predicted brain activation map (same shape).
    data_range : float, optional
        Value range of the data (``max − min``).  Default is ``1.0``,
        appropriate for maps normalised to [0, 1].

    Returns
    -------
    psnr_db : float
        PSNR in decibels.

    Notes
    -----
    Requires ``scikit-image`` (included in the ``metrics`` optional dependency
    group).
    """
    from skimage.metrics import peak_signal_noise_ratio

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(peak_signal_noise_ratio(y_true, y_pred, data_range=data_range))


def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    thresholds: Optional[tuple]=(0.001, 0.01, 0.1),
    percentile: Optional[bool]=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MSE, SSIM, and Dice between original and reconstructed brain maps.

    Parameters
    ----------
    original : 1d tensor
        Torch tensor target.
    reconstructed : 1d tensor
        Prediction of target.
    thresholds : tuple of float
        Thresholds to compute metrics for. Used to binarize tensors.
    percentile : bool, optional, False
        Thresholds should be interpreted as a percentile threshold.

    Returns
    -------
    mse_scores_t : 1d tensor
        MSE scores at each threshold.
    ssim_scores_t : 1d tensor
        Stuctural similarity at each threshold.
    dice_score_t : 1d tensor
        Dice score at each threshold.
    """

    from skimage.metrics import structural_similarity as ssim

    if hasattr(original, "detach"):  # torch tensor
        original = original.detach().cpu().numpy()
        reconstructed = reconstructed.detach().cpu().numpy()

    mse_scores_t = np.zeros(len(thresholds))
    ssim_scores_t = np.zeros(len(thresholds))
    dice_score_t = np.zeros(len(thresholds))

    for it, t in enumerate(thresholds):

        # Threshold
        if percentile:
            t_recon = np.percentile(reconstructed, t)
            t_orig = np.percentile(original, t)
        else:
            t_recon = t
            t_orig = t

        orig_bin = (original > t_orig).astype(np.uint8)
        recon_bin = (reconstructed > t_recon).astype(np.uint8)

        # MSE
        mse_scores_t[it] = ((orig_bin - recon_bin) ** 2).mean()

        # SSIM
        ssim_scores_t[it] = ssim(orig_bin, recon_bin, data_range=1)

        # Dice
        dice_score_t[it] = dice(orig_bin, recon_bin)

    return mse_scores_t, ssim_scores_t, dice_score_t


def dice(img_a, img_b):
    """Compute dice score.

    Parameters
    ----------
    img_a : ndarray
        Binary image.
    img_b : ndarray
        Binary image.

    Returns
    -------
    dice : float
        Dice score.
    """
    intersection = np.logical_and(img_a, img_b).sum()
    denom = img_a.sum() + img_b.sum()
    dice = 1.0  # default if denom == 0
    if denom > 0:
        dice = 2.0 * intersection / denom
    return dice


def dice_percentile(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pct: float = 90,
) -> float:
    """Dice coefficient after percentile thresholding both maps.

    This keeps the effect-size part of the Network Correspondence Toolbox
    workflow while avoiding absolute intensity thresholds.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    pred_bin = y_pred > np.percentile(y_pred, pct)
    true_bin = y_true > np.percentile(y_true, pct)
    return float(dice(pred_bin, true_bin))


def permutation_pvalue(overlap_val: float, overlap_val_spin: np.ndarray) -> float:
    """CBIG/NCT-style permutation p-value for overlap statistics.

    Mirrors ``cbig_network_correspondence.compute_overlap_with_atlases``:
    ``(count(null > observed) + 1) / (n_perm + 1)``.
    """
    null = np.asarray(overlap_val_spin, dtype=float).ravel()
    return float((np.sum((overlap_val - null) < 0) + 1) / (null.size + 1))


@dataclass
class NCTDiceResult:
    """Dice effect size and spin-test significance for two brain maps."""

    dice_pct: float
    spin_p_value: float
    spin_method: str
    spin_significant: bool


def _gifti_to_array(gifti) -> np.ndarray:
    """Return one flat vector from a neuromaps/niBabel GIFTI object or filename."""
    import nibabel as nib

    img = nib.load(str(gifti)) if isinstance(gifti, (str, PathLike)) else gifti
    data = img.agg_data()
    if isinstance(data, tuple):
        data = data[0]
    return np.asarray(data).ravel()


def _load_fsaverage_spheres(density: str = "41k") -> tuple[np.ndarray, np.ndarray]:
    """Load left/right fsaverage sphere coordinates for BrainSpace spins."""
    import nibabel as nib
    from neuromaps.datasets import fetch_fsaverage

    fsavg = fetch_fsaverage(density=density)
    sphere_files = getattr(fsavg, "sphere", None) or fsavg["sphere"]
    if len(sphere_files) != 2:
        raise RuntimeError(f"Expected left/right fsaverage sphere files, got: {sphere_files}")

    points_lh = np.asarray(nib.load(str(sphere_files[0])).agg_data("pointset"), dtype=float)
    points_rh = np.asarray(nib.load(str(sphere_files[1])).agg_data("pointset"), dtype=float)
    return points_lh, points_rh


def _fit_spin_permutations(density: str, n_perm: int, random_state):
    """Fit BrainSpace spin permutations for one fsaverage configuration."""
    from brainspace.null_models import SpinPermutations

    points_lh, points_rh = _load_fsaverage_spheres(density=density)
    spinner = SpinPermutations(n_rep=n_perm, random_state=random_state)
    spinner.fit(points_lh, points_rh=points_rh)
    return spinner


@lru_cache(maxsize=8)
def _cached_spin_permutations(density: str, n_perm: int, random_state: int):
    """Return cached spin permutations for deterministic spin-test settings."""
    return _fit_spin_permutations(density=density, n_perm=n_perm, random_state=random_state)


def _get_spin_permutations(density: str, n_perm: int, random_state):
    """Return fitted spin permutations, caching deterministic configurations."""
    if random_state is None:
        return _fit_spin_permutations(density=density, n_perm=n_perm, random_state=random_state)
    if isinstance(random_state, np.integer):
        random_state = int(random_state)
    if not isinstance(random_state, int):
        return _fit_spin_permutations(density=density, n_perm=n_perm, random_state=random_state)
    return _cached_spin_permutations(density=density, n_perm=int(n_perm), random_state=random_state)


def precompute_spin_permutations(
    density: str = "41k",
    n_perm: int = 1000,
    random_state: int = 0,
):
    """Precompute and cache deterministic BrainSpace spin permutations.

    This is an optional performance helper. It does not change metric values;
    it moves the expensive spin-index fitting step outside per-sample loops.
    """
    return _get_spin_permutations(density=density, n_perm=n_perm, random_state=random_state)


def mni152_to_fsaverage_arrays(
    nifti_img,
    density: str = "41k",
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Project an MNI152 NIfTI map to fsaverage surface arrays.

    ``neuromaps.transforms.mni152_to_fsaverage`` wraps the Wu et al. (2018)
    nonlinear MNI-to-fsaverage registrations used by NCT-style correspondence
    workflows.
    """
    import nibabel as nib
    from neuromaps import transforms

    if not hasattr(transforms, "mni152_to_fsaverage"):
        raise RuntimeError("neuromaps.transforms.mni152_to_fsaverage is unavailable")

    # Some neuromaps/nitransforms versions are stricter about input types and
    # expect an on-disk NIfTI path instead of an in-memory Nifti1Image.
    if isinstance(nifti_img, (str, PathLike)):
        img_arg = str(nifti_img)
        surf_lh, surf_rh = transforms.mni152_to_fsaverage(
            img_arg,
            fsavg_density=density,
            method=method,
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
            nib.save(nifti_img, tmp.name)
            surf_lh, surf_rh = transforms.mni152_to_fsaverage(
                tmp.name,
                fsavg_density=density,
                method=method,
            )
    return _gifti_to_array(surf_lh), _gifti_to_array(surf_rh)


def nct_dice_spin_test_surface(
    pred_lh: np.ndarray,
    pred_rh: np.ndarray,
    true_lh: np.ndarray,
    true_rh: np.ndarray,
    pct: float = 90,
    n_perm: int = 1000,
    random_state: int = 0,
    density: str = "41k",
) -> NCTDiceResult:
    """Compute NCT-style percentile Dice plus BrainSpace spin-test p-value.

    The implementation follows the core CBIG network correspondence pattern:
    compute observed Dice, rotate the reference map with
    ``brainspace.null_models.SpinPermutations``, recompute Dice for each
    rotation, then calculate the permutation p-value.
    """
    pred_lh = np.asarray(pred_lh).ravel()
    pred_rh = np.asarray(pred_rh).ravel()
    true_lh = np.asarray(true_lh).ravel()
    true_rh = np.asarray(true_rh).ravel()

    pred = np.concatenate([pred_lh, pred_rh])
    true = np.concatenate([true_lh, true_rh])
    observed = dice_percentile(pred, true, pct=pct)

    spinner = _get_spin_permutations(density=density, n_perm=n_perm, random_state=random_state)
    rand_lh, rand_rh = spinner.randomize(pred_lh, pred_rh)

    null = np.asarray([
        dice_percentile(np.concatenate([rand_lh[i], rand_rh[i]]), true, pct=pct)
        for i in range(n_perm)
    ])
    p_value = permutation_pvalue(observed, null)
    return NCTDiceResult(
        dice_pct=float(observed),
        spin_p_value=p_value,
        spin_method=f"neuromaps_mni152_to_fsaverage_{density}+brainspace_spin",
        spin_significant=bool(p_value < 0.05),
    )


def nct_dice_spin_test_nifti(
    pred_img,
    true_img,
    pct: float = 90,
    n_perm: int = 1000,
    random_state: int = 0,
    density: str = "41k",
    method: str = "linear",
) -> NCTDiceResult:
    """Project MNI152 NIfTI maps to fsaverage and run NCT-style Dice spins."""
    pred_lh, pred_rh = mni152_to_fsaverage_arrays(pred_img, density=density, method=method)
    true_lh, true_rh = mni152_to_fsaverage_arrays(true_img, density=density, method=method)
    return nct_dice_spin_test_surface(
        pred_lh,
        pred_rh,
        true_lh,
        true_rh,
        pct=pct,
        n_perm=n_perm,
        random_state=random_state,
        density=density,
    )


def dice_top_k(y_true: np.ndarray, y_prob: np.ndarray, k=None):
    """Compute dice score of top k.

    Parameters
    ----------
    y_true : 1d array

    """
    if k is None:
        k = int(y_true.sum())
    idx = np.argpartition(-y_prob.ravel(), k-1)[:k]
    y_hat = np.zeros_like(y_true.ravel()); y_hat[idx] = 1
    y_hat = y_hat.reshape(y_true.shape)
    return (2*(y_hat & y_true).sum()) / (y_hat.sum() + y_true.sum() + 1e-8)


def bernoulli_bce(y, p, eps=1e-7):
    """Elementwise Bernoulli negative log-likelihood (cross-entropy), in nats."""
    p = np.clip(p, eps, 1 - eps)
    y = np.clip(y, 0.0, 1.0)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bits_per_pixel(y_true, logits, *, baseline="per_pixel", eps=1e-7):
    """
    y_true: (N, D) floats in [0,1]
    logits: (N, D) raw logits from decoder (before sigmoid)

    Returns dict with:
      - bpp_model_per_image
      - bpp_base_per_image
      - delta_bpp_per_image (base - model)
      - delta_bpp_per_pixel (mean over N, in bits)  # for a skill map
    """
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    assert y_true.shape == logits.shape, f"shape mismatch: {y_true.shape} vs {logits.shape}"
    N, D = y_true.shape

    p = 1 / (1 + np.exp(-logits)) # sigmoid(logits)

    # Model BCE per image (mean over pixels), then to bits
    bce_model = bernoulli_bce(y_true, p, eps=eps).mean(axis=1)  # (N,)
    bpp_model = bce_model / np.log(2)

    # Baseline probabilities
    if baseline == "global":
        p0 = float(y_true.mean())
        p_base = np.full((N, D), p0, dtype=np.float64)
    elif baseline == "per_pixel":
        p0 = y_true.mean(axis=0, keepdims=True)  # (1, D)
        p_base = np.repeat(p0, repeats=N, axis=0)  # (N, D)
    else:
        raise ValueError("baseline must be 'global' or 'per_pixel'")

    bce_base = bernoulli_bce(y_true, p_base, eps=eps).mean(axis=1)  # (N,)
    bpp_base = bce_base / np.log(2)

    delta_bpp = bpp_base - bpp_model  # (N,) improvement over baseline, bits/pixel

    # Per-pixel skill map (average improvement over images)
    bce_model_px = bernoulli_bce(y_true, p, eps=eps)          # (N, D)
    bce_base_px  = bernoulli_bce(y_true, p_base, eps=eps)     # (N, D)
    delta_bpp_px = (bce_base_px - bce_model_px).mean(axis=0) / np.log(2)  # (D,)

    return dict(
        bpp_model_per_image=bpp_model,
        bpp_base_per_image=bpp_base,
        delta_bpp_per_image=delta_bpp,
        delta_bpp_per_pixel=delta_bpp_px,
        baseline=baseline,
    )


def compute_ae_performance(X: torch.Tensor, X_re: torch.Tensor):
    """Autoencoder performance metrics."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(
        X.numpy().reshape(-1) > 0.5,
        torch.sigmoid(X_re).numpy().reshape(-1)
    )
    roc_auc = auc(fpr, tpr)

    res = bits_per_pixel(X, X_re, baseline="per_pixel")

    delta = res["delta_bpp_per_image"]
    bpp_base = res["bpp_base_per_image"]

    pct = 100.0 * (delta / np.clip(bpp_base, 1e-12, None))

    return fpr, tpr, pct, roc_auc


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
    "psnr",
    "precompute_spin_permutations",
    "permutation_pvalue",
    "NCTDiceResult",
    "dice_top_k",
    "dice",
    "compute_metrics",
    "compute_ae_performance",
    "bits_per_pixel",
    "bernoulli_bce",
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
