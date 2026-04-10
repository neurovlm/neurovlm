"""GATTrainer — training loop for the DiFuMo Soft Atlas GAT.

Phase 4: InfoNCE contrastive training.  The GAT encodes brain graphs to
384-dim vectors; a TextProjHead maps frozen SPECTER embeddings (768-dim)
to the same 384-dim space.  The InfoNCE loss aligns paired (brain, text)
embeddings while repelling non-pairs.

Key differences from NeuroVLM's existing Trainer:
- Uses PyG DataLoader (batched graphs) instead of raw tensor batching.
- Separate learning rates for GAT and TextProjHead (AdamW).
- Cosine LR schedule with a linear warmup phase.
- Validation recall@k tracked every N epochs.
- Optional attention weight snapshots for interpretability.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from neurovlm.loss import InfoNCELoss
from neurovlm.metrics import recall_at_k, recall_curve
from .dataset import BrainGraphDataset
from .model import BrainGAT, TextProjHead


# ---------------------------------------------------------------------------
# Learning-rate schedule helpers
# ---------------------------------------------------------------------------

def _cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class GATTrainer:
    """Contrastive trainer for BrainGAT + TextProjHead.

    Parameters
    ----------
    brain_encoder:
        :class:`~neurovlm.gnn.model.BrainGAT` instance.
    text_proj:
        :class:`~neurovlm.gnn.model.TextProjHead` instance.
    lr_gat:
        Learning rate for the GAT parameters.  Default 1e-4.
    lr_proj:
        Learning rate for the TextProjHead parameters.  Default 1e-5.
    batch_size:
        Number of graphs per mini-batch.  Default 512.
    n_epochs:
        Total training epochs.  Default 150.
    warmup_epochs:
        Linear LR warmup duration.  Default 10.
    temperature:
        InfoNCE temperature τ.  Default 0.07.
    device:
        ``"auto"`` picks GPU/MPS/CPU automatically.
    val_interval:
        Compute validation recall@k every *val_interval* epochs.  Default 5.
    checkpoint_dir:
        If given, save the best model to ``<checkpoint_dir>/best_gat.pt``.
    verbose:
        Print training progress.
    """

    def __init__(
        self,
        brain_encoder: BrainGAT,
        text_proj: TextProjHead,
        lr_gat: float = 1e-4,
        lr_proj: float = 1e-5,
        batch_size: int = 512,
        n_epochs: int = 150,
        warmup_epochs: int = 10,
        temperature: float = 0.07,
        device: str = "auto",
        val_interval: int = 5,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        if device == "auto":
            from neurovlm.train import which_device
            device = which_device()

        self.device = torch.device(device)
        self.brain_encoder = brain_encoder.to(self.device)
        self.text_proj = text_proj.to(self.device)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.val_interval = val_interval
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self.loss_fn = InfoNCELoss(temperature=temperature)

        # Separate param groups for different LRs
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.brain_encoder.parameters(), "lr": lr_gat},
                {"params": self.text_proj.parameters(), "lr": lr_proj},
            ],
            weight_decay=1e-4,
        )

        # LR scheduler is stepped per epoch; set up after knowing total steps
        self._n_train_batches: Optional[int] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None

        self._best_state: Optional[dict] = None
        self._best_val_recall: float = -1.0
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_recall_t2i": [],
            "val_recall_i2t": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        total_steps = self.n_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            return _cosine_with_warmup(step, warmup_steps, total_steps)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

    def _make_dataloader(self, dataset: BrainGraphDataset):
        from torch_geometric.loader import DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _forward_batch(self, batch) -> tuple[Tensor, Tensor]:
        """Return (brain_emb, text_emb) for one mini-batch."""
        batch = batch.to(self.device)
        brain_emb = self.brain_encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        text_emb = self.text_proj(batch.y)
        return brain_emb, text_emb

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, val_dataset: BrainGraphDataset) -> tuple[float, float]:
        """Return (recall_t2i_auc, recall_i2t_auc) on the validation set."""
        self.brain_encoder.eval()
        self.text_proj.eval()

        loader = self._make_dataloader(val_dataset)
        all_brain, all_text = [], []
        for batch in loader:
            b, t = self._forward_batch(batch)
            all_brain.append(b)
            all_text.append(t)

        brain_emb = torch.cat(all_brain, dim=0)
        text_emb = torch.cat(all_text, dim=0)

        t2i, i2t = recall_curve(text_emb, brain_emb)
        auc_t2i = float(t2i.mean())
        auc_i2t = float(i2t.mean())

        self.brain_encoder.train()
        self.text_proj.train()
        return auc_t2i, auc_i2t

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: BrainGraphDataset,
        val_dataset: Optional[BrainGraphDataset] = None,
    ) -> None:
        """Run the training loop.

        Parameters
        ----------
        train_dataset:
            Training split.
        val_dataset:
            Validation split.  Required for recall@k monitoring and best-model
            checkpointing.  If *None*, only training loss is tracked.
        """
        from torch_geometric.loader import DataLoader

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        steps_per_epoch = len(train_loader)
        self._build_scheduler(steps_per_epoch)

        if self.verbose:
            print(
                f"Training BrainGAT: {self.n_epochs} epochs, "
                f"batch={self.batch_size}, device={self.device}"
            )
            if val_dataset is not None:
                print(f"  train={len(train_dataset)}  val={len(val_dataset)}")

        for epoch in range(self.n_epochs):
            self.brain_encoder.train()
            self.text_proj.train()

            epoch_losses = []
            for batch in train_loader:
                brain_emb, text_emb = self._forward_batch(batch)
                loss = self.loss_fn(brain_emb, text_emb)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.brain_encoder.parameters())
                    + list(self.text_proj.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.scheduler.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(mean_loss)

            # Validation
            if val_dataset is not None and epoch % self.val_interval == 0:
                auc_t2i, auc_i2t = self._validate(val_dataset)
                self.history["val_recall_t2i"].append(auc_t2i)
                self.history["val_recall_i2t"].append(auc_i2t)

                if self.verbose:
                    lr_gat = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch:4d}/{self.n_epochs} | "
                        f"loss={mean_loss:.4f} | "
                        f"recall_t2i={auc_t2i:.4f} | "
                        f"recall_i2t={auc_i2t:.4f} | "
                        f"lr={lr_gat:.2e}"
                    )

                mean_recall = (auc_t2i + auc_i2t) / 2
                if mean_recall > self._best_val_recall:
                    self._best_val_recall = mean_recall
                    self._best_state = {
                        "brain_encoder": deepcopy(self.brain_encoder.state_dict()),
                        "text_proj": deepcopy(self.text_proj.state_dict()),
                        "epoch": epoch,
                        "val_recall": mean_recall,
                    }
                    if self.checkpoint_dir is not None:
                        self._save_checkpoint()

            elif self.verbose and epoch % max(1, self.n_epochs // 20) == 0:
                lr_gat = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d}/{self.n_epochs} | "
                    f"loss={mean_loss:.4f} | lr={lr_gat:.2e}"
                )

        # Early-epoch sanity check: if loss hasn't dropped below 5.0 by
        # epoch 50, the node features are likely not normalized.
        if len(self.history["train_loss"]) >= 50:
            loss_ep50 = self.history["train_loss"][49]
            if loss_ep50 > 5.0:
                import warnings
                warnings.warn(
                    f"Training loss at epoch 50 is {loss_ep50:.3f} (>5.0). "
                    "Check that DiFuMo coefficients are z-scored (normalize=True "
                    "in compute_difumo_coefficients)."
                )

    # ------------------------------------------------------------------
    # Checkpointing and restoration
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "best_gat.pt"
        torch.save(self._best_state, path)
        if self.verbose:
            print(f"  → checkpoint saved to {path}")

    def restore_best(self) -> None:
        """Restore encoder weights from the best-validation checkpoint."""
        if self._best_state is None:
            raise RuntimeError("No checkpoint available; call fit() first.")
        self.brain_encoder.load_state_dict(self._best_state["brain_encoder"])
        self.text_proj.load_state_dict(self._best_state["text_proj"])

    def save(self, path: str) -> None:
        """Save current model state dicts."""
        torch.save(
            {
                "brain_encoder": self.brain_encoder.state_dict(),
                "text_proj": self.text_proj.state_dict(),
            },
            path,
        )

    # ------------------------------------------------------------------
    # Attention-weight inspection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_attention_snapshot(
        self, dataset: BrainGraphDataset, n_samples: int = 64
    ) -> dict:
        """Collect GAT attention weights for interpretability analysis.

        Returns a dict with:
        - ``edge_index``: (2, E) global edge indices
        - ``attn_mean``: mean attention weight per edge across *n_samples* graphs
          (aggregated over all heads from layer 3)

        Usage example::

            snap = trainer.get_attention_snapshot(val_dataset)
            top_edges = snap["attn_mean"].topk(20).indices
        """
        from torch_geometric.loader import DataLoader

        loader = DataLoader(dataset, batch_size=min(n_samples, len(dataset)), shuffle=False)
        batch = next(iter(loader)).to(self.device)

        self.brain_encoder.eval()
        x, ei = batch.x, batch.edge_index

        # Layer 1
        _, (_, a1) = self.brain_encoder.conv1(x, ei, return_attention_weights=True)
        x = F.elu(self.brain_encoder.conv1(x, ei))
        # Layer 2
        _, (_, a2) = self.brain_encoder.conv2(x, ei, return_attention_weights=True)
        x = F.elu(self.brain_encoder.conv2(x, ei))
        # Layer 3
        _, (ei3, a3) = self.brain_encoder.conv3(x, ei, return_attention_weights=True)

        # Average heads in layer 3 and aggregate over graphs in the batch
        attn_mean = a3.mean(dim=1)  # (E_batch,)

        return {
            "edge_index": ei3.cpu(),
            "attn_mean": attn_mean.cpu(),
        }
