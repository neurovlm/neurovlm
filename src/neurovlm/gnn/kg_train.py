"""R-GCN training loop for KG link prediction.

Phase 3, Steps 9–10.

Training objective
------------------
Binary cross-entropy on positive vs. filtered-negative triples.

Evaluation metrics (filtered setting)
--------------------------------------
- MRR   — Mean Reciprocal Rank
- Hits@1, Hits@3, Hits@10

Early stopping
--------------
Stops when validation MRR has not improved for ``patience`` epochs.

Typical usage
-------------
>>> from neurovlm.gnn.kg_data import load_kg, KGSplits
>>> from neurovlm.gnn.rgcn import RGCNLinkPredictor
>>> from neurovlm.gnn.kg_train import RGCNTrainer
>>>
>>> kg = load_kg("experiments/data/unified_kg/unified_kg_nodes.parquet",
...              "experiments/data/unified_kg/unified_kg_edges.parquet")
>>> splits = KGSplits.from_kg(kg)
>>> model = RGCNLinkPredictor(num_entities=kg.num_entities,
...                           num_relations=kg.num_relations)
>>> trainer = RGCNTrainer(model, splits, checkpoint_dir="checkpoints/rgcn")
>>> trainer.fit()
>>> emb = trainer.extract_embeddings()
>>> trainer.save_embeddings("experiments/data/unified_kg/entity_embeddings.pt")
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .kg_data import KGSplits, kg_collate_fn
from .rgcn import RGCNLinkPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _which_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bce_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """Binary cross-entropy: positives → 1, negatives → 0."""
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    return F.binary_cross_entropy_with_logits(scores, labels)


# ---------------------------------------------------------------------------
# Rank-based evaluation (filtered)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_link_prediction(
    model: RGCNLinkPredictor,
    entity_emb: Tensor,
    eval_triples: Tensor,
    all_triple_set: frozenset,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compute MRR and Hits@k using the filtered evaluation protocol.

    For each test triple ``(s, r, o)``, all entities are scored as candidate
    objects.  Known true triples other than ``(s, r, o)`` itself are masked out
    before computing the rank of the true object.

    Parameters
    ----------
    model:
        Trained :class:`RGCNLinkPredictor`.
    entity_emb:
        Contextual embeddings from ``model.encode()``, shape ``(N_e, d)``.
    eval_triples:
        Triples to evaluate, shape ``(N, 3)``.
    all_triple_set:
        Full set of KG triples used for filtering.
    batch_size:
        Number of triples evaluated per GPU batch.
    device:
        Evaluation device.

    Returns
    -------
    dict with keys ``mrr``, ``hits@1``, ``hits@3``, ``hits@10``.
    """
    model.eval()
    entity_emb = entity_emb.to(device)
    eval_triples = eval_triples.to(device)

    reciprocal_ranks: list[float] = []
    hits: dict[int, list[float]] = {1: [], 3: [], 10: []}

    # Build a lookup: (s, r) → set of known true objects (for filtering)
    # Only build over the eval set's (s, r) pairs for efficiency
    sr_to_objects: dict[tuple[int, int], set[int]] = {}
    for triple in all_triple_set:
        s, r, o = triple
        key = (s, r)
        if key not in sr_to_objects:
            sr_to_objects[key] = set()
        sr_to_objects[key].add(o)

    for start in range(0, len(eval_triples), batch_size):
        batch = eval_triples[start : start + batch_size]
        s_batch = batch[:, 0]
        r_batch = batch[:, 1]
        o_batch = batch[:, 2]

        # Score all entities as objects: (B, N_e)
        all_scores = model.score_all_objects(entity_emb, s_batch, r_batch)

        for i in range(len(batch)):
            s = int(s_batch[i])
            r = int(r_batch[i])
            o_true = int(o_batch[i])

            scores = all_scores[i].clone()  # (N_e,)

            # Filtered: mask known true objects (except the query triple itself)
            true_objects = sr_to_objects.get((s, r), set())
            for o_known in true_objects:
                if o_known != o_true:
                    scores[o_known] = float("-inf")

            # Rank of the true object (1-indexed, lower is better)
            rank = int((scores > scores[o_true]).sum().item()) + 1

            reciprocal_ranks.append(1.0 / rank)
            for k in hits:
                hits[k].append(1.0 if rank <= k else 0.0)

    return {
        "mrr":     float(np.mean(reciprocal_ranks)),
        "hits@1":  float(np.mean(hits[1])),
        "hits@3":  float(np.mean(hits[3])),
        "hits@10": float(np.mean(hits[10])),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RGCNTrainer:
    """Training harness for :class:`RGCNLinkPredictor`.

    Parameters
    ----------
    model:
        :class:`RGCNLinkPredictor` instance (not yet moved to device).
    splits:
        :class:`~neurovlm.gnn.kg_data.KGSplits` with train/val/test triples
        and the pre-built training edge_index / edge_type.
    lr:
        Initial learning rate.  Default 1e-3.
    weight_decay:
        AdamW weight decay.  Default 1e-4.
    n_epochs:
        Maximum training epochs.  Default 500.
    batch_size:
        Triples per mini-batch.  Default 1024.
    neg_ratio:
        Negatives per positive triple during training.  Default 10.
    eval_batch_size:
        Batch size for ranking evaluation (entity-object scoring).  Default 512.
    val_interval:
        Evaluate on validation set every N epochs.  Default 10.
    patience:
        Early-stopping patience (epochs without val MRR improvement).  Default 30.
    lr_patience:
        Plateau LR scheduler patience (epochs).  Default 10.
    lr_factor:
        Multiplicative LR reduction factor on plateau.  Default 0.5.
    device:
        ``"auto"`` selects GPU → MPS → CPU.
    checkpoint_dir:
        If provided, best model state is saved to ``<dir>/best_rgcn.pt``.
    verbose:
        Print training progress.  Default True.
    """

    def __init__(
        self,
        model: RGCNLinkPredictor,
        splits: KGSplits,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 500,
        batch_size: int = 1024,
        neg_ratio: int = 10,
        eval_batch_size: int = 512,
        val_interval: int = 10,
        patience: int = 30,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        device: str = "auto",
        checkpoint_dir: Optional[str | Path] = None,
        verbose: bool = True,
    ):
        if device == "auto":
            device = _which_device()

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.splits = splits
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.eval_batch_size = eval_batch_size
        self.val_interval = val_interval
        self.patience = patience
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Pre-move static graph tensors to device
        self.edge_index = splits.train_edge_index.to(self.device)
        self.edge_type  = splits.train_edge_type.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",    # maximize MRR
            factor=lr_factor,
            patience=lr_patience,
        )

        self._best_mrr: float = -1.0
        self._best_state: Optional[dict] = None
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_mrr": [],
            "val_hits@1": [],
            "val_hits@3": [],
            "val_hits@10": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_train_loader(self) -> DataLoader:
        ds = self.splits.train_dataset(neg_ratio=self.neg_ratio)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=kg_collate_fn,
            num_workers=0,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        epoch_losses: list[float] = []

        for batch in loader:
            pos = batch["positives"].to(self.device)  # (B, 3)
            neg = batch["negatives"].to(self.device)  # (B*neg_ratio, 3)

            if len(neg) == 0:
                continue

            pos_scores, neg_scores = self.model(
                self.edge_index, self.edge_type,
                pos[:, 0], pos[:, 1], pos[:, 2],
                neg[:, 0], neg[:, 1], neg[:, 2],
            )

            loss = _bce_loss(pos_scores, neg_scores)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_losses.append(loss.item())

        return float(np.mean(epoch_losses)) if epoch_losses else float("nan")

    @torch.no_grad()
    def _get_entity_emb(self) -> Tensor:
        self.model.eval()
        return self.model.encode(self.edge_index, self.edge_type)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the training loop with early stopping on validation MRR."""
        loader = self._make_train_loader()
        no_improve = 0

        if self.verbose:
            print(
                f"R-GCN training: {self.n_epochs} epochs, "
                f"batch={self.batch_size}, neg_ratio={self.neg_ratio}, "
                f"device={self.device}"
            )
            print(
                f"  entities={self.splits.kg.num_entities:,}  "
                f"relations={self.splits.kg.num_relations}  "
                f"train_triples={len(self.splits.train_triples):,}"
            )

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            mean_loss = self._train_epoch(loader)
            self.history["train_loss"].append(mean_loss)

            # Validation
            if epoch % self.val_interval == 0:
                entity_emb = self._get_entity_emb()
                metrics = evaluate_link_prediction(
                    self.model,
                    entity_emb,
                    self.splits.val_triples,
                    self.splits.kg.triple_set,
                    batch_size=self.eval_batch_size,
                    device=self.device,
                )
                val_mrr = metrics["mrr"]
                self.history["val_mrr"].append(val_mrr)
                self.history["val_hits@1"].append(metrics["hits@1"])
                self.history["val_hits@3"].append(metrics["hits@3"])
                self.history["val_hits@10"].append(metrics["hits@10"])

                self.scheduler.step(val_mrr)
                lr_now = self.optimizer.param_groups[0]["lr"]

                if self.verbose:
                    elapsed = time.time() - t0
                    print(
                        f"Epoch {epoch:4d}/{self.n_epochs} | "
                        f"loss={mean_loss:.4f} | "
                        f"MRR={val_mrr:.4f} | "
                        f"H@1={metrics['hits@1']:.4f} | "
                        f"H@3={metrics['hits@3']:.4f} | "
                        f"H@10={metrics['hits@10']:.4f} | "
                        f"lr={lr_now:.2e} | "
                        f"{elapsed:.1f}s"
                    )

                if val_mrr > self._best_mrr:
                    self._best_mrr = val_mrr
                    self._best_state = deepcopy(self.model.state_dict())
                    no_improve = 0
                    if self.checkpoint_dir is not None:
                        self._save_checkpoint(epoch, metrics)
                else:
                    no_improve += self.val_interval
                    if no_improve >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs).")
                        break

            elif self.verbose and epoch % max(1, self.n_epochs // 20) == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:4d}/{self.n_epochs} | loss={mean_loss:.4f} | lr={lr_now:.2e}")

        if self.verbose:
            print(f"Training complete. Best val MRR={self._best_mrr:.4f}")

    # ------------------------------------------------------------------
    # Evaluation on test set
    # ------------------------------------------------------------------

    def evaluate_test(self) -> dict[str, float]:
        """Evaluate on the test split using the best-checkpoint weights.

        Returns
        -------
        dict with ``mrr``, ``hits@1``, ``hits@3``, ``hits@10``.
        """
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        entity_emb = self._get_entity_emb()
        metrics = evaluate_link_prediction(
            self.model,
            entity_emb,
            self.splits.test_triples,
            self.splits.kg.triple_set,
            batch_size=self.eval_batch_size,
            device=self.device,
        )
        if self.verbose:
            print(
                f"Test | MRR={metrics['mrr']:.4f} | "
                f"H@1={metrics['hits@1']:.4f} | "
                f"H@3={metrics['hits@3']:.4f} | "
                f"H@10={metrics['hits@10']:.4f}"
            )
        return metrics

    # ------------------------------------------------------------------
    # Embedding extraction (Step 10)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embeddings(self) -> dict[str, Tensor]:
        """Extract final entity and relation embeddings after training.

        Restores best-checkpoint weights, then returns:

        Returns
        -------
        dict with:
        - ``"entity_embeddings"``: FloatTensor ``(num_entities, emb_dim)``
        - ``"relation_embeddings"``: FloatTensor ``(num_relations, emb_dim)``
        """
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        entity_emb = self._get_entity_emb().cpu()
        relation_emb = self.model.relation_emb.weight.detach().cpu()
        return {
            "entity_embeddings": entity_emb,
            "relation_embeddings": relation_emb,
        }

    def save_embeddings(self, out_path: str | Path) -> None:
        """Save entity/relation embeddings alongside the entity ID map.

        The saved dict contains:
        - ``entity_embeddings``: ``(num_entities, emb_dim)``
        - ``relation_embeddings``: ``(num_relations, emb_dim)``
        - ``entity_to_idx``: ``{canonical_id: int}``
        - ``idx_to_entity``: ``{int: canonical_id}``
        - ``relation_to_idx``: ``{relation_type: int}``
        - ``idx_to_relation``: ``{int: relation_type}``

        Parameters
        ----------
        out_path:
            Output path for the ``.pt`` checkpoint file.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        emb_dict = self.extract_embeddings()
        kg = self.splits.kg
        payload = {
            **emb_dict,
            "entity_to_idx":   kg.entity_to_idx,
            "idx_to_entity":   kg.idx_to_entity,
            "relation_to_idx": kg.relation_to_idx,
            "idx_to_relation": kg.idx_to_relation,
        }
        torch.save(payload, out_path)
        if self.verbose:
            print(f"Embeddings saved to {out_path}")
            print(
                f"  entity_embeddings : {emb_dict['entity_embeddings'].shape}\n"
                f"  relation_embeddings: {emb_dict['relation_embeddings'].shape}"
            )

    # ------------------------------------------------------------------
    # Nearest-neighbour spot-check (Step 10)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def nearest_neighbours(
        self,
        query: str,
        k: int = 10,
        entity_emb: Optional[Tensor] = None,
    ) -> list[tuple[float, str]]:
        """Return the k nearest entities to *query* by cosine similarity.

        Parameters
        ----------
        query:
            Canonical entity ID or entity name.  Checked against
            ``entity_to_idx`` first, then against ``idx_to_entity`` values.
        k:
            Number of neighbours to return (excluding the query itself).
        entity_emb:
            Pre-computed embedding matrix.  If *None*, extracted on the fly.

        Returns
        -------
        List of ``(cosine_similarity, entity_name)`` tuples, descending order.
        """
        kg = self.splits.kg

        # Resolve query string → index
        if query in kg.entity_to_idx:
            q_idx = kg.entity_to_idx[query]
        else:
            # Try name lookup (case-insensitive)
            q_lower = query.lower()
            q_idx = None
            for idx, eid in kg.idx_to_entity.items():
                if eid.lower() == q_lower:
                    q_idx = idx
                    break
            if q_idx is None:
                raise KeyError(f"Entity '{query}' not found in KG.")

        if entity_emb is None:
            entity_emb = self.extract_embeddings()["entity_embeddings"]

        emb_norm = F.normalize(entity_emb, dim=-1)  # (N_e, d)
        q_emb = emb_norm[q_idx].unsqueeze(0)         # (1, d)
        sims = (q_emb @ emb_norm.t()).squeeze(0)     # (N_e,)
        sims[q_idx] = -2.0                           # exclude self

        top_vals, top_idx = sims.topk(k)
        return [
            (float(top_vals[i]), kg.idx_to_entity[int(top_idx[i])])
            for i in range(k)
        ]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "best_rgcn.pt"
        torch.save(
            {
                "model_state": self._best_state,
                "epoch": epoch,
                "metrics": metrics,
                "num_entities": self.model.num_entities,
                "num_relations": self.model.num_relations,
                "emb_dim": self.model.emb_dim,
            },
            path,
        )
        if self.verbose:
            print(f"  → checkpoint saved to {path}")

    def restore_best(self) -> None:
        """Load the best-validation checkpoint into the model."""
        if self._best_state is None:
            raise RuntimeError("No checkpoint available. Call fit() first.")
        self.model.load_state_dict(self._best_state)
