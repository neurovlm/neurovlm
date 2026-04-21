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
from .kg_data import KGSplits
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


def _bce_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    pos_rel_idx: Optional[Tensor] = None,
    neg_rel_idx: Optional[Tensor] = None,
    relation_weights: Optional[Tensor] = None,
) -> Tensor:
    """Binary cross-entropy: positives → 1, negatives → 0.

    When *relation_weights* is provided, each sample is scaled by the weight
    of its relation type so that rare relation types contribute proportionally
    to the gradient.  Negatives inherit the weight of their positive's relation.

    Parameters
    ----------
    pos_scores, neg_scores:
        Raw logit scores for positive and negative triples.
    pos_rel_idx, neg_rel_idx:
        Relation-type integer indices for positives / negatives (LongTensor).
        Required when *relation_weights* is not None.
    relation_weights:
        Per-relation weight tensor of shape ``(num_relations,)``.  Typically
        computed as inverse-frequency weights, normalised to mean 1 and capped
        at some maximum to avoid extreme gradients for very sparse types.
    """
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    sample_weights: Optional[Tensor] = None
    if relation_weights is not None and pos_rel_idx is not None and neg_rel_idx is not None:
        pos_w = relation_weights[pos_rel_idx]
        neg_w = relation_weights[neg_rel_idx]
        sample_weights = torch.cat([pos_w, neg_w])

    return F.binary_cross_entropy_with_logits(scores, labels, weight=sample_weights)


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
    graph_sample_size:
        Maximum number of edges passed to the R-GCN each training batch.
        Full-graph message passing over millions of edges stores activations
        proportional to ``n_edges × hidden_dim`` per layer, which OOMs on
        MPS/GPU with large KGs.  A random edge subgraph of this size is used
        instead; the model sees the full graph over many batches/epochs.
        Default 200_000.  Set to ``sys.maxsize`` to disable subsampling.
    checkpoint_dir:
        If provided, best model state is saved to ``<dir>/best_rgcn.pt``.
    verbose:
        Print training progress.  Default True.
    relation_weights:
        Optional 1-D FloatTensor of shape ``(num_relations,)`` used to
        upweight rare relation types in the BCE loss.  Typically computed
        as inverse-frequency weights normalised to mean 1.  When *None*,
        all relation types are treated equally (original behaviour).
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
        graph_sample_size: int = 200_000,
        checkpoint_dir: Optional[str | Path] = None,
        verbose: bool = True,
        relation_weights: Optional[Tensor] = None,
        print_interval: int = 1,
        resume_checkpoint_interval: int = 0,
        resume_from: Optional[str | Path] = None,
        max_steps_per_epoch: int = 0,
    ):
        if device == "auto":
            device = _which_device()

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.splits = splits
        self.n_epochs = n_epochs
        self.graph_sample_size = graph_sample_size
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.eval_batch_size = eval_batch_size
        self.val_interval = val_interval
        self.patience = patience
        self.verbose = verbose
        self.max_steps_per_epoch = max_steps_per_epoch
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Pre-move static tensors to device once
        self.edge_index          = splits.train_edge_index.to(self.device)
        self.edge_type           = splits.train_edge_type.to(self.device)
        self.train_triples_dev   = splits.train_triples.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",    # maximize MRR
            factor=lr_factor,
            patience=lr_patience,
        )

        # Pre-move relation weights to device if provided
        if relation_weights is not None:
            self.relation_weights: Optional[Tensor] = relation_weights.to(self.device)
        else:
            self.relation_weights = None

        self.print_interval = print_interval
        self.resume_checkpoint_interval = resume_checkpoint_interval
        self._resume_from = Path(resume_from) if resume_from else None

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

    def _train_epoch(self) -> float:
        """GPU-native training epoch: all sampling done on-device, no DataLoader."""
        self.model.train()
        epoch_losses: list[float] = []

        n_train   = self.train_triples_dev.size(0)
        n_edges   = self.edge_index.size(1)
        n_ent     = self.model.num_entities
        n_neg     = self.batch_size * self.neg_ratio
        sample_size = min(n_edges, self.graph_sample_size)
        steps = self.max_steps_per_epoch if self.max_steps_per_epoch > 0 else (n_train // self.batch_size)

        for _ in range(steps):
            # ── Sample positive triples on GPU ─────────────────────────────
            pos_idx = torch.randint(n_train, (self.batch_size,), device=self.device)
            pos = self.train_triples_dev[pos_idx]                     # (B, 3)

            # ── Generate negatives on GPU (unfiltered) ─────────────────────
            # False-negative rate ≈ 15M / (33k² × 8) < 0.2% — negligible.
            neg_ent      = torch.randint(n_ent, (n_neg,), device=self.device)
            corrupt_subj = torch.rand(n_neg, device=self.device) < 0.5
            neg = pos.repeat_interleave(self.neg_ratio, dim=0).clone()  # (B*K, 3)
            neg[:, 0] = torch.where(corrupt_subj,  neg_ent, neg[:, 0])
            neg[:, 2] = torch.where(~corrupt_subj, neg_ent, neg[:, 2])

            # ── Sample graph edges with randint (much faster than randperm) ─
            edge_perm = torch.randint(n_edges, (sample_size,), device=self.device)
            edge_idx  = self.edge_index[:, edge_perm]
            edge_typ  = self.edge_type[edge_perm]

            pos_scores, neg_scores = self.model(
                edge_idx, edge_typ,
                pos[:, 0], pos[:, 1], pos[:, 2],
                neg[:, 0], neg[:, 1], neg[:, 2],
            )

            loss = _bce_loss(
                pos_scores, neg_scores,
                pos_rel_idx=pos[:, 1],
                neg_rel_idx=neg[:, 1],
                relation_weights=self.relation_weights,
            )
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
        import sys

        no_improve = 0
        start_epoch = 1

        # ── Resume from a previous session ────────────────────────────────────
        if self._resume_from is not None and self._resume_from.exists():
            ckpt = torch.load(self._resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.history    = ckpt["history"]
            self._best_mrr  = ckpt["best_mrr"]
            self._best_state = ckpt.get("best_state")
            no_improve      = ckpt["no_improve"]
            start_epoch     = ckpt["epoch"] + 1
            if self.verbose:
                print(
                    f"Resumed from {self._resume_from} at epoch {ckpt['epoch']} "
                    f"(best MRR={self._best_mrr:.4f})",
                    flush=True,
                )
        elif self._resume_from is not None:
            if self.verbose:
                print(f"Resume checkpoint not found at {self._resume_from} — starting fresh.", flush=True)

        if self.verbose:
            total_batches = len(self.splits.train_triples) // self.batch_size
            steps = self.max_steps_per_epoch if self.max_steps_per_epoch > 0 else total_batches
            print(
                f"R-GCN training: epochs {start_epoch}–{self.n_epochs}, "
                f"batch={self.batch_size}, neg_ratio={self.neg_ratio}, "
                f"steps/epoch={steps:,} (of {total_batches:,} possible), "
                f"device={self.device}",
                flush=True,
            )
            print(
                f"  entities={self.splits.kg.num_entities:,}  "
                f"relations={self.splits.kg.num_relations}  "
                f"train_triples={len(self.splits.train_triples):,}",
                flush=True,
            )

        for epoch in range(start_epoch, self.n_epochs + 1):
            t0 = time.time()
            mean_loss = self._train_epoch()
            self.history["train_loss"].append(mean_loss)
            elapsed = time.time() - t0

            # ── Per-epoch loss print (no val eval) ────────────────────────────
            if self.verbose and self.print_interval > 0 and epoch % self.print_interval == 0:
                if epoch % self.val_interval != 0:
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch:4d}/{self.n_epochs} | "
                        f"loss={mean_loss:.4f} | "
                        f"lr={lr_now:.2e} | "
                        f"{elapsed:.1f}s",
                        flush=True,
                    )
                    sys.stdout.flush()

            # ── Validation ────────────────────────────────────────────────────
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
                    print(
                        f"Epoch {epoch:4d}/{self.n_epochs} | "
                        f"loss={mean_loss:.4f} | "
                        f"MRR={val_mrr:.4f} | "
                        f"H@1={metrics['hits@1']:.4f} | "
                        f"H@3={metrics['hits@3']:.4f} | "
                        f"H@10={metrics['hits@10']:.4f} | "
                        f"lr={lr_now:.2e} | "
                        f"{elapsed:.1f}s",
                        flush=True,
                    )
                    sys.stdout.flush()

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
                            print(
                                f"Early stopping at epoch {epoch} "
                                f"(no improvement for {self.patience} epochs).",
                                flush=True,
                            )
                        break

            # ── Periodic resume checkpoint ────────────────────────────────────
            if (
                self.resume_checkpoint_interval > 0
                and self.checkpoint_dir is not None
                and epoch % self.resume_checkpoint_interval == 0
            ):
                self._save_resume_checkpoint(epoch, no_improve)

        if self.verbose:
            print(f"Training complete. Best val MRR={self._best_mrr:.4f}", flush=True)

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
        """Save entity/relation embeddings and ID maps.

        Writes two files:

        ``<out_path>`` (e.g. ``entity_embeddings.pt``)
            Tensors only — safe to load with ``weights_only=True``:

            - ``entity_embeddings``: ``(num_entities, emb_dim)``
            - ``relation_embeddings``: ``(num_relations, emb_dim)``

        ``<out_path>.meta.json``
            ID maps (not serialisable as tensors):

            - ``entity_to_idx``: ``{canonical_id: int}``
            - ``idx_to_entity``: ``{str(int): canonical_id}``
            - ``relation_to_idx``: ``{relation_type: int}``
            - ``idx_to_relation``: ``{str(int): relation_type}``

        Parameters
        ----------
        out_path:
            Output path for the ``.pt`` file.  The ``.meta.json`` sidecar
            is written to the same directory with the same stem.
        """
        import json as _json

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        emb_dict = self.extract_embeddings()
        kg = self.splits.kg

        # Tensors only → weights_only=True compatible
        torch.save(emb_dict, out_path)

        # ID maps → JSON sidecar (keys must be strings in JSON)
        meta_path = out_path.with_suffix("").with_suffix(".pt.meta.json")
        meta = {
            "entity_to_idx":   kg.entity_to_idx,
            "idx_to_entity":   {str(k): v for k, v in kg.idx_to_entity.items()},
            "relation_to_idx": kg.relation_to_idx,
            "idx_to_relation": {str(k): v for k, v in kg.idx_to_relation.items()},
        }
        with open(meta_path, "w") as fh:
            _json.dump(meta, fh)

        if self.verbose:
            print(f"Embeddings saved to {out_path}")
            print(f"ID maps saved to    {meta_path}")
            print(
                f"  entity_embeddings  : {emb_dict['entity_embeddings'].shape}\n"
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
            print(f"  → best checkpoint saved to {path}", flush=True)

    def _save_resume_checkpoint(self, epoch: int, no_improve: int) -> None:
        """Save full trainer state so training can be resumed after a crash."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "resume_rgcn.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "best_state": self._best_state,
                "best_mrr": self._best_mrr,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "history": self.history,
                "no_improve": no_improve,
                "num_entities": self.model.num_entities,
                "num_relations": self.model.num_relations,
                "emb_dim": self.model.emb_dim,
            },
            path,
        )
        if self.verbose:
            print(f"  → resume checkpoint saved to {path} (epoch {epoch})", flush=True)

    def restore_best(self) -> None:
        """Load the best-validation checkpoint into the model."""
        if self._best_state is None:
            raise RuntimeError("No checkpoint available. Call fit() first.")
        self.model.load_state_dict(self._best_state)
