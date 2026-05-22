"""Performance metrics"""
from dataclasses import dataclass
from functools import lru_cache
from os import PathLike
import tempfile
from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Text generation metrics (BLEU, ROUGE)
# ---------------------------------------------------------------------------

def bleu(references: list[str], hypothesis: str, n: int = 4) -> float:
    """Compute BLEU score for text generation evaluation.

    Useful for evaluating text produced from brain activations against a set
    of reference descriptions.

    Parameters
    ----------
    references : list of str
        One or more reference texts to compare against.
    hypothesis : str
        The generated/predicted text.
    n : int, optional
        Maximum n-gram order (1–4). Default is 4.

    Returns
    -------
    score : float
        Sentence-level BLEU score in [0, 1].

    Notes
    -----
    Requires ``nltk`` (included in the ``metrics`` optional dependency group).
    Uses ``SmoothingFunction.method1`` to avoid zero scores on short texts.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tokens = [ref.lower().split() for ref in references]
    hyp_tokens = hypothesis.lower().split()
    weights = tuple(1.0 / n for _ in range(n))
    smoother = SmoothingFunction().method1
    return float(sentence_bleu(ref_tokens, hyp_tokens, weights=weights,
                               smoothing_function=smoother))


def rouge(reference: str, hypothesis: str) -> dict[str, dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores for text generation.

    Useful for evaluating text produced from brain activations against a
    reference description.

    Parameters
    ----------
    reference : str
        The ground-truth reference text.
    hypothesis : str
        The generated/predicted text.

    Returns
    -------
    scores : dict
        Keys are ``'rouge1'``, ``'rouge2'``, ``'rougeL'``.  Each value is a
        dict with ``'precision'``, ``'recall'``, and ``'fmeasure'`` floats.

    Notes
    -----
    Requires ``rouge-score`` (included in the ``metrics`` optional dependency
    group).  Stemming is enabled for robustness to inflectional variation.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    raw = scorer.score(reference, hypothesis)
    return {
        key: {
            'precision': val.precision,
            'recall': val.recall,
            'fmeasure': val.fmeasure,
        }
        for key, val in raw.items()
    }


def bertscore_single(
    bert_score_fn,
    generated: str,
    reference: str,
    model_type: str,
) -> tuple[float, float, float]:
    """Compute BERTScore precision, recall, and F1 for one generated string."""

    p, r, f1 = bert_score_fn(
        cands=[generated],
        refs=[reference],
        lang="en",
        model_type=model_type,
        verbose=False,
    )
    return float(p[0]), float(r[0]), float(f1[0])


def semantic_similarity(st_model, st_util, generated: str, reference: str) -> float:
    """Sentence-level cosine similarity for generated/reference text."""

    emb1 = st_model.encode(generated, convert_to_tensor=True)
    emb2 = st_model.encode(reference, convert_to_tensor=True)
    return float(st_util.cos_sim(emb1, emb2))


def nvlm_latent_similarity(nvlm, brain_query_emb: torch.Tensor, generated: str) -> float:
    """Cosine similarity between a brain query and generated text in NeuroVLM space."""

    nvlm._ensure_projection_heads()
    with torch.no_grad():
        raw_emb = nvlm._encode_text(generated)
        z_text = nvlm._proj_head_text_infonce(raw_emb.to(nvlm.device))
        z_text = F.normalize(z_text, dim=-1).cpu()
    z_brain = brain_query_emb.cpu()
    if z_brain.dim() == 1:
        z_brain = z_brain.unsqueeze(0)
    return float(F.cosine_similarity(z_brain, z_text))


# ---------------------------------------------------------------------------
# Brain image similarity / quality metrics
# ---------------------------------------------------------------------------

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

def token_f1(reference: str, hypothesis: str) -> float:
    """Compute token-level F1 between a reference and hypothesis string.

    Standard SQuAD-style metric: multi-set token overlap over lowercased
    whitespace-split tokens.

    Parameters
    ----------
    reference : str
        Ground-truth text.
    hypothesis : str
        Generated/predicted text.

    Returns
    -------
    f1 : float
        Token F1 in [0, 1].  Returns 0.0 when either string is empty.
    """
    from collections import Counter

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    common = sum((ref_counts & hyp_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(hyp_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


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

# Recall metrics
@torch.no_grad()
def normalized_k_values(n: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return normalized full-curve k values ``[1/n, ..., n/n]``."""

    if n < 1:
        raise ValueError("n must be >= 1")
    return torch.arange(1, n + 1, device=device).float() / float(n)


@torch.no_grad()
def normalized_recall_curve_auc(curve: torch.Tensor) -> float:
    """Area under recall(k) vs normalized k = k/n.

    Recall curves are saved at every integer k, so the normalized-k samples are
    uniformly spaced by 1/n. The right-endpoint Riemann area is therefore the
    mean of the full recall curve. This is the paper-style AUC, not recall@K.
    """

    if curve.numel() == 0:
        raise ValueError("curve must contain at least one point")
    return float(curve.float().mean().item())


def recall_at_k(cos_sim: torch.Tensor, k: int) -> float:
    """
    Parameters
    ----------
    cos_sim : 2d tensor
        Cosine similarity matrix.
    k : int
        Top k most similar items to consider.

    Returns
    -------
    recall : float
        Recall @ k.
    """
    ranks = cos_sim.argsort(dim=1, descending=True)
    correct = torch.arange(ranks.size(0), device=ranks.device)
    hit = (ranks[:, :k] == correct[:, None]).any(dim=1)
    return hit.float().mean().item()


@torch.no_grad()
def retrieval_ranks(cos_sim: torch.Tensor) -> torch.Tensor:
    """Return 1-indexed retrieval ranks for diagonal matches.

    ``cos_sim[i, j]`` is the similarity between query ``i`` and target ``j``;
    the correct target is assumed to be on the diagonal.
    """
    ranks = cos_sim.argsort(dim=1, descending=True)
    correct = torch.arange(ranks.size(0), device=ranks.device)
    pos = ranks.eq(correct[:, None]).to(torch.int32).argmax(dim=1)
    return pos + 1


@torch.no_grad()
def retrieval_metrics(
    latent_query: torch.Tensor,
    latent_target: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """Compute retrieval metrics for aligned query/target embeddings."""
    q = F.normalize(latent_query.float(), dim=1, eps=1e-8)
    t = F.normalize(latent_target.float(), dim=1, eps=1e-8)
    sim = q @ t.T
    ranks = retrieval_ranks(sim).float()
    n = float(sim.size(0))

    out: dict[str, float] = {
        "median_rank": float(ranks.median().item()),
        "mean_rank": float(ranks.mean().item()),
        "mrr": float((1.0 / ranks).mean().item()),
    }
    for k in ks:
        out[f"recall@{k}"] = float((ranks <= k).float().mean().item())
        out[f"random_recall@{k}"] = min(float(k) / n, 1.0)

    curve, _ = recall_curve(q, t)
    auc = normalized_recall_curve_auc(curve)
    out["auc"] = auc
    out["paper_recall_curve_auc"] = auc
    out["normalized_k_recall_curve_auc"] = auc
    return out


@torch.no_grad()
def bidirectional_retrieval_metrics(
    latent_text: torch.Tensor,
    latent_image: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """Compute text-to-image, image-to-text, and averaged retrieval metrics."""
    t2i = retrieval_metrics(latent_text, latent_image, ks=ks)
    i2t = retrieval_metrics(latent_image, latent_text, ks=ks)
    out = {f"t2i_{k}": v for k, v in t2i.items()}
    out.update({f"i2t_{k}": v for k, v in i2t.items()})
    for key in t2i:
        out[f"mean_{key}"] = (t2i[key] + i2t[key]) / 2.0
    return out


@torch.no_grad()
def recall_curve(latent_text: torch.Tensor,
                 latent_image: torch.Tensor,
                 step: int = 1) -> tuple[torch.Tensor, torch.Tensor]:

    t = F.normalize(latent_text, dim=1, eps=1e-8)
    v = F.normalize(latent_image, dim=1, eps=1e-8)

    n = t.shape[0]
    device = t.device
    ks = torch.arange(0, n, step, device=device) + 1

    def _curve_from_sim(sim: torch.Tensor) -> torch.Tensor:
        ranks = sim.argsort(dim=1, descending=True)
        correct = torch.arange(n, device=sim.device)
        pos = ranks.eq(correct[:, None]).to(torch.int32).argmax(dim=1)  # FIX

        counts = torch.bincount(pos, minlength=n)
        recall_all = counts.cumsum(0).float() / float(n)
        return recall_all.index_select(0, ks - 1)

    sim = t @ v.T
    t_to_i = _curve_from_sim(sim)
    i_to_t = _curve_from_sim(sim.T)
    return t_to_i, i_to_t


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
