"""Performance metrics"""
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
