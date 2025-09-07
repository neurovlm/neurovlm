"""Non-standard torch loss functions."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, f1_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class InfoNCELoss(torch.nn.Module):
    """Compute InfoNCE loss between image and text embeddings."""
    def __init__(self,  temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image, text):
        # Normalize embeddings
        image = F.normalize(image, dim=1)
        text = F.normalize(text, dim=1)

        # Compute similarity matrix: (batch_size, batch_size)
        logits = torch.matmul(image, text.T) / self.temperature

        # Labels are indices of the correct pairs
        batch_size = image.size(0)
        labels = torch.arange(batch_size, device=image.device)

        # Cross-entropy loss in both directions (symmetrized)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

class TruncatedLoss(nn.Module):
    def __init__(self, percentile=0.8, base_loss="l1"):
        super().__init__()
        self.percentile = percentile
        self.base_loss = base_loss

    def forward(self, predicted, target):
        # either smooth l1, here it's huber, or mse loss
        if self.base_loss == "l1":
            loss_per_sample = F.smooth_l1_loss(predicted, target, reduction='none', beta=1.).mean(dim=1)
        else:
            loss_per_sample = F.mse_loss(predicted, target, reduction='none').mean(dim=1)

        # keep the easiest x% of examples
        threshold = torch.quantile(loss_per_sample, self.percentile)
        easy_mask = loss_per_sample <= threshold

        if easy_mask.sum() > 0:
            return loss_per_sample[easy_mask].mean()
        else:
            return loss_per_sample.mean()

def compute_metrics(original, reconstructed, thresholds=(0.001, 0.01, 0.1), percentile=False):
    """Compute MSE, SSIM, and Dice between original and reconstructed brain maps."""

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

        # Dice for 1D arrays
        intersection = np.logical_and(orig_bin, recon_bin).sum()
        denom = orig_bin.sum() + recon_bin.sum()
        dice = 1.0  # default if denom == 0
        if denom > 0:
            dice = 2.0 * intersection / denom
        dice_score_t[it] = dice

    return mse_scores_t, ssim_scores_t, dice_score_t


def recall_n(y_pred, y_truth, n_first=10, thresh=0.95, reduce_mean=False):
    assert (y_pred.ndim in (1, 2)) and (
        y_truth.ndim in (1, 2)
    ), "arrays should be of dimension 1 or 2"
    assert y_pred.shape == y_truth.shape, "both arrays should have the same shape"

    if y_pred.ndim == 1:
        # recall@n for a single sample
        targets = np.where(y_truth >= thresh)[0]
        pred_n_first = np.argsort(y_pred)[::-1][:n_first]

        if len(targets) > 0:
            ratio_in_n = len(np.intersect1d(targets, pred_n_first)) / len(targets)
        else:
            ratio_in_n = np.nan

        return ratio_in_n
    else:
        # recall@n for a dataset (mean of recall@n for all samples)
        result = np.zeros(len(y_pred))
        for i, (sample_y_pred, sample_y_truth) in enumerate(zip(y_pred, y_truth)):
            result[i] = recall_n(sample_y_pred, sample_y_truth, n_first, thresh)
        if reduce_mean:
            return np.nanmean(result)

        return result

def mix_match(similarity):
    accuracies = []
    for row_index in range(len(similarity)):
        current_row_accumulator = 0
        for col_index in range(len(similarity[row_index])):
            if col_index == row_index:
                continue
            else:
                if similarity[row_index][row_index] > similarity[row_index][col_index]:
                    current_row_accumulator += 1

        accuracies.append(current_row_accumulator / (len(similarity[row_index])-1))

    return np.mean(accuracies)
