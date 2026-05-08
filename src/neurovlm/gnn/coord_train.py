"""CoordTrainer — training loop for the Atlas-Free Coordinate GNN (Track 2).

InfoNCE contrastive training: CoordGNN encodes variable-size coordinate graphs
to 384-dim brain embeddings; TextProjHead maps frozen SPECTER embeddings (768)
to the same space.

Additions beyond GATTrainer
----------------------------
- Embedding collapse monitor: after each val run, logs the mean pairwise cosine
  similarity of 256 random val embeddings.  Above 0.95 → model is collapsing.
- Dual checkpoints: best_coord_gnn.pt (best val AUC) + last_coord_gnn.pt (final epoch).
- Per-batch node-count guard: warns if any batch has >8,000 total nodes (MPS limit).
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
from neurovlm.metrics import bidirectional_retrieval_metrics, recall_curve
from .coord_dataset import CoordGraphDataset, _SubsetCoordDataset
from .coord_model import CoordGNN
from .model import TextProjHead

_AnyDataset = CoordGraphDataset | _SubsetCoordDataset


def _cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class CoordTrainer:
    """Contrastive trainer for CoordGNN + TextProjHead.

    Parameters
    ----------
    brain_encoder : CoordGNN
    text_proj : TextProjHead
    lr_gnn : float
        Learning rate for CoordGNN. Default 1e-4.
    lr_proj : float
        Learning rate for TextProjHead. Default 1e-5.
    batch_size : int
        Graphs per mini-batch. Default 64.
    n_epochs : int
        Total training epochs. Default 200.
    warmup_epochs : int
        Linear LR warmup duration. Default 15.
    temperature : float
        InfoNCE temperature τ. Default 0.07.
    device : str
        ``"auto"`` picks CUDA/MPS/CPU automatically.
    val_interval : int
        Compute val recall@k every N epochs. Default 5.
    checkpoint_dir : str or None
        Saves best_coord_gnn.pt and last_coord_gnn.pt here.
    collapse_sample_n : int
        Number of val embeddings to sample for the collapse monitor. Default 256.
    num_workers : int
        DataLoader worker processes. 0 = main process (safe for MPS/Windows).
        Set to 4 on Colab A100 for parallel data loading.
    pin_memory : bool
        Pin DataLoader output to page-locked memory for faster GPU transfers.
        Set True on CUDA, False on MPS/CPU.
    use_amp : bool
        Use BF16 automatic mixed precision on CUDA. A100 Tensor Cores run BF16
        at ~2x the speed of FP32. No GradScaler needed (BF16 doesn't overflow).
        Ignored silently on MPS/CPU.
    prefetch_factor : int or None
        Number of batches loaded in advance by each worker. Only used when
        num_workers > 0.
    verbose : bool
    """

    def __init__(
        self,
        brain_encoder: CoordGNN,
        text_proj: TextProjHead,
        lr_gnn: float = 1e-4,
        lr_proj: float = 1e-5,
        batch_size: int = 64,
        n_epochs: int = 200,
        warmup_epochs: int = 15,
        temperature: float = 0.07,
        device: str = "auto",
        val_interval: int = 5,
        checkpoint_dir: Optional[str] = None,
        collapse_sample_n: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_amp: bool = False,
        prefetch_factor: Optional[int] = None,
        monitor_metric: str = "mean_auc",
        early_stopping_patience: Optional[int] = None,
        freeze_text_proj: bool = False,
        config: Optional[dict] = None,
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
        self.collapse_sample_n = collapse_sample_n
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.monitor_metric = monitor_metric
        self.early_stopping_patience = early_stopping_patience
        self.freeze_text_proj = freeze_text_proj
        self.config = config or {}
        # BF16 AMP only makes sense on CUDA; silently disable elsewhere
        self.use_amp = use_amp and self.device.type == "cuda"
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self.loss_fn = InfoNCELoss(temperature=temperature)

        param_groups = [{"params": self.brain_encoder.parameters(), "lr": lr_gnn}]
        if freeze_text_proj:
            for p in self.text_proj.parameters():
                p.requires_grad_(False)
        else:
            param_groups.append({"params": self.text_proj.parameters(), "lr": lr_proj})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None

        self._best_state: Optional[dict] = None
        self._best_val_recall: float = -1.0
        self._last_epoch: int = -1
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_recall_t2i": [],  # legacy aliases for val_t2i_auc / val_i2t_auc
            "val_recall_i2t": [],
            "embed_sim": [],       # collapse monitor
            "epoch": [],
            "lr_gnn": [],
            "lr_proj": [],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        total_steps = self.n_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            return _cosine_with_warmup(step, warmup_steps, total_steps)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

    def _make_dataloader(self, dataset: _AnyDataset, shuffle: bool = True):
        from torch_geometric.loader import DataLoader
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def _forward_batch(self, batch) -> tuple[Tensor, Tensor]:
        batch = batch.to(self.device, non_blocking=self.pin_memory)
        if self.use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                brain_emb = self.brain_encoder(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                text_emb = self.text_proj(batch.y)
        else:
            brain_emb = self.brain_encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            text_emb = self.text_proj(batch.y)
        return brain_emb, text_emb

    # ------------------------------------------------------------------
    # Collapse monitor
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _embedding_collapse_sim(self, val_dataset: _AnyDataset) -> float:
        """Return mean pairwise cosine similarity for a random sample of val embs.

        Values near 1.0 indicate embedding collapse.
        """
        self.brain_encoder.eval()
        self.text_proj.eval()
        loader = self._make_dataloader(val_dataset, shuffle=True)
        brain_embs = []
        for batch in loader:
            b, _ = self._forward_batch(batch)
            brain_embs.append(b.float().clone())
            if sum(e.shape[0] for e in brain_embs) >= self.collapse_sample_n:
                break
        embs = F.normalize(torch.cat(brain_embs, dim=0)[:self.collapse_sample_n], dim=1)
        sim_matrix = embs @ embs.T
        # Exclude diagonal (self-similarity = 1.0)
        mask = ~torch.eye(len(embs), dtype=torch.bool, device=embs.device)
        self.brain_encoder.train()
        self.text_proj.train()
        return float(sim_matrix[mask].mean())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, val_dataset: _AnyDataset) -> tuple[float, float]:
        self.brain_encoder.eval()
        self.text_proj.eval()

        loader = self._make_dataloader(val_dataset, shuffle=False)
        all_brain, all_text = [], []
        for batch in loader:
            b, t = self._forward_batch(batch)
            all_brain.append(b.float().clone())
            all_text.append(t.float().clone())

        brain_emb = torch.cat(all_brain, dim=0)
        text_emb = torch.cat(all_text, dim=0)

        t2i, i2t = recall_curve(text_emb, brain_emb)
        auc_t2i = float(t2i.mean())
        auc_i2t = float(i2t.mean())

        self.brain_encoder.train()
        self.text_proj.train()
        return auc_t2i, auc_i2t

    @torch.no_grad()
    def _evaluate(self, dataset: _AnyDataset) -> dict[str, float]:
        """Return validation loss and bidirectional retrieval metrics."""
        self.brain_encoder.eval()
        self.text_proj.eval()

        brain_emb, text_emb = self.collect_embeddings(dataset)
        loss = float(self.loss_fn(brain_emb, text_emb).item())
        metrics = bidirectional_retrieval_metrics(text_emb, brain_emb)
        metrics["loss"] = loss

        self.brain_encoder.train()
        self.text_proj.train()
        return metrics

    def _record_val_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        self.history["epoch"].append(epoch)
        self.history["val_loss"].append(metrics["loss"])
        self.history["val_recall_t2i"].append(metrics["t2i_auc"])
        self.history["val_recall_i2t"].append(metrics["i2t_auc"])
        for key, value in metrics.items():
            self.history.setdefault(f"val_{key}", []).append(float(value))

    def _monitor_value(self, metrics: dict[str, float]) -> float:
        if self.monitor_metric not in metrics:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"Unknown monitor_metric={self.monitor_metric!r}; "
                f"available metrics: {available}"
            )
        return float(metrics[self.monitor_metric])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: _AnyDataset,
        val_dataset: Optional[_AnyDataset] = None,
    ) -> None:
        train_loader = self._make_dataloader(train_dataset, shuffle=True)
        steps_per_epoch = len(train_loader)
        self._build_scheduler(steps_per_epoch)

        if self.verbose:
            n_params = self.brain_encoder.count_parameters()
            print(
                f"\nTraining CoordGNN ({n_params:,} params): "
                f"{self.n_epochs} epochs, "
                f"batch={self.batch_size}, device={self.device}"
            )
            if self.device.type == "cuda":
                print(
                    f"  dataloader: workers={self.num_workers} "
                    f"pin_memory={self.pin_memory} "
                    f"prefetch_factor={self.prefetch_factor}"
                )
            if self.freeze_text_proj:
                print("  text projection: frozen")
            if val_dataset is not None:
                print(f"  train={len(train_dataset)}  val={len(val_dataset)}")

        bad_val_checks = 0
        for epoch in range(self.n_epochs):
            self._last_epoch = epoch
            self.brain_encoder.train()
            self.text_proj.train()

            epoch_losses = []
            max_nodes_seen = 0

            for batch in train_loader:
                # Monitor peak node count per batch (MPS guard: warn if >8000)
                n_nodes = batch.x.shape[0]
                max_nodes_seen = max(max_nodes_seen, n_nodes)

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
            self.history["lr_gnn"].append(float(self.optimizer.param_groups[0]["lr"]))
            lr_proj = 0.0 if self.freeze_text_proj else self.optimizer.param_groups[1]["lr"]
            self.history["lr_proj"].append(float(lr_proj))

            if epoch == 0 and max_nodes_seen > 8000:
                import warnings
                warnings.warn(
                    f"Batch had {max_nodes_seen} total nodes (>8000). "
                    "Consider reducing batch_size to 32 for MPS stability."
                )

            if val_dataset is not None and epoch % self.val_interval == 0:
                val_metrics = self._evaluate(val_dataset)
                embed_sim = self._embedding_collapse_sim(val_dataset)
                self._record_val_metrics(epoch, val_metrics)
                self.history["embed_sim"].append(embed_sim)

                if embed_sim > 0.95:
                    import warnings
                    warnings.warn(
                        f"Embedding collapse detected at epoch {epoch}: "
                        f"embed_sim={embed_sim:.3f} > 0.95. "
                        "Add nn.LayerNorm after self.proj and restart training."
                    )

                if self.verbose:
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch:4d}/{self.n_epochs} | "
                        f"loss={mean_loss:.4f} | "
                        f"val_loss={val_metrics['loss']:.4f} | "
                        f"auc={val_metrics['mean_auc']:.4f} "
                        f"r@10={val_metrics['mean_recall@10']:.4f} "
                        f"mrr={val_metrics['mean_mrr']:.4f} | "
                        f"embed_sim={embed_sim:.3f} | "
                        f"lr={lr_now:.2e}"
                    )

                score = self._monitor_value(val_metrics)
                if score > self._best_val_recall:
                    self._best_val_recall = score
                    self._best_state = {
                        "brain_encoder": deepcopy(self.brain_encoder.state_dict()),
                        "text_proj": deepcopy(self.text_proj.state_dict()),
                        "epoch": epoch,
                        "val_recall": score,
                        "monitor_metric": self.monitor_metric,
                        "metrics": val_metrics,
                        "config": self.config,
                    }
                    self._save_checkpoint("best_coord_gnn.pt")
                    bad_val_checks = 0
                else:
                    bad_val_checks += 1
                    if (
                        self.early_stopping_patience is not None
                        and bad_val_checks >= self.early_stopping_patience
                    ):
                        if self.verbose:
                            print(
                                f"Early stopping after {bad_val_checks} "
                                f"validation checks without improvement."
                            )
                        break

            elif self.verbose and epoch % max(1, self.n_epochs // 20) == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d}/{self.n_epochs} | "
                    f"loss={mean_loss:.4f} | lr={lr_now:.2e}"
                )

        # Always save the final epoch state
        self._save_last_checkpoint()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        if self.checkpoint_dir is None or self._best_state is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        torch.save(self._best_state, path)
        if self.verbose:
            print(f"  → checkpoint saved to {path}")

    def _save_last_checkpoint(self) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "last_coord_gnn.pt"
        torch.save(
            {
                "brain_encoder": self.brain_encoder.state_dict(),
                "text_proj": self.text_proj.state_dict(),
                "epoch": self._last_epoch,
                "history": self.history,
                "config": self.config,
                "monitor_metric": self.monitor_metric,
            },
            path,
        )
        if self.verbose:
            print(f"  → last checkpoint saved to {path}")

    def restore_best(self) -> None:
        if self._best_state is None:
            raise RuntimeError("No checkpoint available; call fit() first.")
        self.brain_encoder.load_state_dict(self._best_state["brain_encoder"])
        self.text_proj.load_state_dict(self._best_state["text_proj"])

    def save(self, path: str) -> None:
        torch.save(
            {
                "brain_encoder": self.brain_encoder.state_dict(),
                "text_proj": self.text_proj.state_dict(),
            },
            path,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_embeddings(self, dataset: _AnyDataset) -> tuple[Tensor, Tensor]:
        """Return (brain_emb, text_emb) for the full dataset."""
        self.brain_encoder.eval()
        self.text_proj.eval()
        loader = self._make_dataloader(dataset, shuffle=False)
        all_b, all_t = [], []
        for batch in loader:
            b, t = self._forward_batch(batch)
            all_b.append(b.float().clone())
            all_t.append(t.float().clone())
        self.brain_encoder.train()
        self.text_proj.train()
        return torch.cat(all_b, dim=0), torch.cat(all_t, dim=0)

    @torch.no_grad()
    def get_attention_snapshot(
        self,
        dataset: _AnyDataset,
        n_samples: int = 10,
        top_k: int = 5,
        exclude_self_loops: bool = True,
    ) -> list[dict]:
        """Extract per-paper top-attention edges with MNI coordinates.

        Returns a list of dicts, one per paper:
            paper_idx, node_coords_mni (N,3), top_edges [(src_mni, dst_mni, weight)]
        """
        from torch_geometric.loader import DataLoader
        from .coord_graph import MNI_HALF

        self.brain_encoder.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        results = []

        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            batch = batch.to(self.device)
            _, attn = self.brain_encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                return_attention=True,
            )
            ei, attn_weights = attn           # (2,E), (E,1)
            attn_weights = attn_weights.squeeze(-1)
            is_self_loop = ei[0].eq(ei[1])

            candidate = ~is_self_loop if exclude_self_loops else torch.ones_like(
                is_self_loop, dtype=torch.bool
            )
            if not candidate.any():
                candidate = torch.ones_like(is_self_loop, dtype=torch.bool)

            candidate_idx = candidate.nonzero(as_tuple=True)[0]
            weights = attn_weights[candidate_idx]
            top_idx = weights.topk(min(top_k, len(weights))).indices
            top_edge_idx = candidate_idx[top_idx]
            coords_norm = batch.x[:, :3].float().cpu()  # (N, 3)
            coords_mni = (coords_norm * MNI_HALF).numpy()

            top_edges = []
            for eidx in top_edge_idx.tolist():
                s, d = ei[0, eidx].item(), ei[1, eidx].item()
                top_edges.append({
                    "src_mni": coords_mni[s].tolist(),
                    "dst_mni": coords_mni[d].tolist(),
                    "weight": float(attn_weights[eidx]),
                    "is_self_loop": bool(is_self_loop[eidx]),
                })

            results.append({
                "paper_idx": int(batch.paper_idx[0]),
                "node_coords_mni": coords_mni,
                "top_edges": top_edges,
                "attention_self_loop_frac": float(is_self_loop.float().mean()),
                "attention_self_loop_mass": float(
                    attn_weights[is_self_loop].sum() / attn_weights.sum().clamp_min(1e-8)
                ),
            })

        self.brain_encoder.train()
        return results
