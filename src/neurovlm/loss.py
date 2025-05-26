"""Non-standard torch loss functions."""
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)  # predicted prob of the true label
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
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
