"""Performance metrics"""
from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, f1_score

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

def recall_curve(latent_text: torch.Tensor, latent_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute recall@k across all k, in both directions.

    Parameters
    ----------
    latent_text : 2d tensor
        Latent text embeddings.
    latent_image : 2d tensor
        Latent image embeddings.

    Returns
    -------
    t_to_i : 1d tensor
        Recall@k for text-to-image direction, stepping across all k.
    i_to_t : 1d tensor
        Recall@k for image-to-text direction, stepping across all k.
    """
    # Unit vectors
    if not (latent_text.norm(dim=1) == 1).all():
        latent_text_norm = F.normalize(latent_text, dim=1, eps=1e-8)
    else:
        latent_text_norm = latent_text

    if not (latent_image.norm(dim=1) == 1).all():
        latent_image_norm = F.normalize(latent_image, dim=1, eps=1e-8)
    else:
        latent_image_norm = latent_image
        
    # Text-to-image
    cos_t_to_i = latent_text_norm @ latent_image_norm.T
    t_to_i = torch.zeros(latent_text_norm.shape[0])
    for k in range(len(latent_text_norm)):
        t_to_i[k] = recall_at_k(cos_t_to_i, k + 1)

    # Image-to-text
    cos_i_to_t = latent_image_norm @ latent_text_norm.T
    i_to_t = torch.zeros(latent_image_norm.shape[0])
    for k in range(len(latent_text_norm)):
        i_to_t[k] = recall_at_k(cos_i_to_t, k + 1)

    return t_to_i, i_to_t
