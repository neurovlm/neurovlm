"""Knowledge graph data preparation for R-GCN link prediction.

Phase 3, Step 7: loads the unified KG (nodes + edges parquets), assigns
contiguous entity and relation-type integers, and provides a PyTorch Dataset
that yields positive/negative triple batches for link prediction training.

Relation types (8 total)
-------------------------
co_occurs_with, narrower_term_of, associated_with_disorder,
implicated_in, co_activates_with, expressed_in,
treated_by, used_in

Typical usage
-------------
>>> from neurovlm.gnn.kg_data import load_kg, KGSplits
>>> kg = load_kg("experiments/data/unified_kg/unified_kg_nodes.parquet",
...              "experiments/data/unified_kg/unified_kg_edges.parquet")
>>> splits = KGSplits.from_kg(kg, train_frac=0.85, val_frac=0.075, seed=42)
>>> train_ds = splits.train_dataset(neg_ratio=10)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# KG container
# ---------------------------------------------------------------------------

@dataclass
class KGData:
    """Immutable container for the unified knowledge graph.

    Attributes
    ----------
    num_entities:
        Total number of canonical entities.
    num_relations:
        Number of distinct relation types (contiguous 0 … num_relations-1).
    entity_to_idx:
        Mapping from canonical_id string → integer index.
    idx_to_entity:
        Inverse mapping: integer index → canonical_id string.
    relation_to_idx:
        Mapping from relation_type string → integer index.
    idx_to_relation:
        Inverse mapping: integer index → relation_type string.
    triples:
        All triples as a LongTensor of shape ``(N_triples, 3)`` with columns
        ``[subject_idx, relation_idx, object_idx]``.
    triple_set:
        Frozenset of ``(subject_idx, relation_idx, object_idx)`` tuples used
        for filtered negative sampling.
    edge_index:
        PyG-compatible edge index ``(2, N_triples)`` over the training graph.
        Built from training triples only; set after ``KGSplits`` is created.
    edge_type:
        Relation-type integer per edge, shape ``(N_triples,)``.
    """

    num_entities: int
    num_relations: int
    entity_to_idx: dict[str, int]
    idx_to_entity: dict[int, str]
    relation_to_idx: dict[str, int]
    idx_to_relation: dict[int, str]
    triples: Tensor          # (N, 3)  LongTensor
    triple_set: FrozenSet[Tuple[int, int, int]]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_kg(nodes_path: str | Path, edges_path: str | Path) -> KGData:
    """Load unified KG from parquet files and build integer ID maps.

    Parameters
    ----------
    nodes_path:
        Path to ``unified_kg_nodes.parquet``.  Must contain a ``canonical_id``
        column.
    edges_path:
        Path to ``unified_kg_edges.parquet``.  Must contain ``subject_id``,
        ``relation_type``, and ``object_id`` columns.

    Returns
    -------
    KGData
        Fully-indexed knowledge graph ready for splitting and training.
    """
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)

    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)

    # --- entity index (sorted for reproducibility) ---
    all_ids = sorted(nodes_df["canonical_id"].unique().tolist())
    entity_to_idx = {eid: i for i, eid in enumerate(all_ids)}
    idx_to_entity = {i: eid for eid, i in entity_to_idx.items()}

    # --- relation index (sorted for contiguous 0-based integers) ---
    all_rels = sorted(edges_df["relation_type"].unique().tolist())
    relation_to_idx = {r: i for i, r in enumerate(all_rels)}
    idx_to_relation = {i: r for r, i in relation_to_idx.items()}

    # --- filter edges with unknown endpoints (defensive) ---
    known = set(entity_to_idx)
    mask = edges_df["subject_id"].isin(known) & edges_df["object_id"].isin(known)
    edges_df = edges_df[mask].reset_index(drop=True)

    # --- build triple tensor ---
    subj = edges_df["subject_id"].map(entity_to_idx).to_numpy(dtype=np.int64)
    rel  = edges_df["relation_type"].map(relation_to_idx).to_numpy(dtype=np.int64)
    obj  = edges_df["object_id"].map(entity_to_idx).to_numpy(dtype=np.int64)

    triples = torch.tensor(np.stack([subj, rel, obj], axis=1), dtype=torch.long)
    triple_set = frozenset(map(tuple, triples.tolist()))

    print(
        f"KG loaded: {len(entity_to_idx):,} entities, "
        f"{len(relation_to_idx)} relation types, "
        f"{len(triples):,} triples."
    )
    print("  Relation types:", list(relation_to_idx.keys()))

    return KGData(
        num_entities=len(entity_to_idx),
        num_relations=len(relation_to_idx),
        entity_to_idx=entity_to_idx,
        idx_to_entity=idx_to_entity,
        relation_to_idx=relation_to_idx,
        idx_to_relation=idx_to_relation,
        triples=triples,
        triple_set=triple_set,
    )


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

@dataclass
class KGSplits:
    """Train / validation / test splits of a KG, stratified by relation type.

    Attributes
    ----------
    kg:
        Parent :class:`KGData` (contains entity/relation maps and full triple set).
    train_triples:
        Training triples, shape ``(N_train, 3)``.
    val_triples:
        Validation triples, shape ``(N_val, 3)``.
    test_triples:
        Test triples, shape ``(N_test, 3)``.
    train_edge_index:
        ``(2, N_train)`` edge index built from training triples (for R-GCN).
    train_edge_type:
        Relation-type integers for training edges, shape ``(N_train,)``.
    """

    kg: KGData
    train_triples: Tensor
    val_triples: Tensor
    test_triples: Tensor
    train_edge_index: Tensor
    train_edge_type: Tensor

    @classmethod
    def from_kg(
        cls,
        kg: KGData,
        train_frac: float = 0.85,
        val_frac: float = 0.075,
        seed: int = 42,
    ) -> "KGSplits":
        """Stratified split by relation type.

        Parameters
        ----------
        kg:
            Loaded :class:`KGData`.
        train_frac:
            Fraction of triples for training (default 0.85).
        val_frac:
            Fraction for validation (default 0.075).
            Test fraction is ``1 - train_frac - val_frac``.
        seed:
            Random seed for reproducibility.
        """
        test_frac = 1.0 - train_frac - val_frac
        assert test_frac > 0, "train_frac + val_frac must be < 1.0"

        rng = np.random.default_rng(seed)
        triples_np = kg.triples.numpy()

        train_idx, val_idx, test_idx = [], [], []

        # Stratify by relation type
        for rel_id in range(kg.num_relations):
            rel_mask = triples_np[:, 1] == rel_id
            indices = np.where(rel_mask)[0]
            rng.shuffle(indices)

            n = len(indices)
            n_val  = max(1, int(n * val_frac))
            n_test = max(1, int(n * test_frac))
            n_train = n - n_val - n_test

            train_idx.append(indices[:n_train])
            val_idx.append(indices[n_train : n_train + n_val])
            test_idx.append(indices[n_train + n_val :])

        train_idx = np.concatenate(train_idx)
        val_idx   = np.concatenate(val_idx)
        test_idx  = np.concatenate(test_idx)

        train_triples = kg.triples[train_idx]
        val_triples   = kg.triples[val_idx]
        test_triples  = kg.triples[test_idx]

        # Build PyG edge_index / edge_type from training triples only
        train_edge_index = train_triples[:, [0, 2]].t().contiguous()  # (2, N_train)
        train_edge_type  = train_triples[:, 1]                         # (N_train,)

        print(
            f"Split: train={len(train_triples):,}  "
            f"val={len(val_triples):,}  "
            f"test={len(test_triples):,}"
        )

        return cls(
            kg=kg,
            train_triples=train_triples,
            val_triples=val_triples,
            test_triples=test_triples,
            train_edge_index=train_edge_index,
            train_edge_type=train_edge_type,
        )

    # ------------------------------------------------------------------
    # Dataset factories
    # ------------------------------------------------------------------

    def train_dataset(self, neg_ratio: int = 10) -> "KGTripleDataset":
        """Return a :class:`KGTripleDataset` for the training split."""
        return KGTripleDataset(
            positives=self.train_triples,
            num_entities=self.kg.num_entities,
            triple_set=self.kg.triple_set,
            neg_ratio=neg_ratio,
        )

    def val_dataset(self) -> "KGTripleDataset":
        """Return a :class:`KGTripleDataset` for the validation split (no negatives)."""
        return KGTripleDataset(
            positives=self.val_triples,
            num_entities=self.kg.num_entities,
            triple_set=self.kg.triple_set,
            neg_ratio=0,
        )

    def test_dataset(self) -> "KGTripleDataset":
        """Return a :class:`KGTripleDataset` for the test split (no negatives)."""
        return KGTripleDataset(
            positives=self.test_triples,
            num_entities=self.kg.num_entities,
            triple_set=self.kg.triple_set,
            neg_ratio=0,
        )


# ---------------------------------------------------------------------------
# Triple dataset with on-the-fly filtered negative sampling
# ---------------------------------------------------------------------------

class KGTripleDataset(Dataset):
    """PyTorch Dataset of KG triples with on-the-fly filtered negative sampling.

    Each item yields a dict:
    ``{"positive": (s, r, o), "negatives": Tensor(neg_ratio, 3)}``

    When ``neg_ratio=0`` (validation / test), ``"negatives"`` is an empty
    tensor and only the positive triple is returned for ranking evaluation.

    Parameters
    ----------
    positives:
        Positive triples, shape ``(N, 3)``.
    num_entities:
        Total number of entities in the KG.
    triple_set:
        Frozenset of all known triples (train + val + test) used to filter
        corrupted negatives.
    neg_ratio:
        Number of corrupted negatives generated per positive triple.
        Use 10 for training, 0 for evaluation.
    max_attempts:
        Maximum rejection-sampling attempts per negative before giving up.
        In practice, for a large KG the rejection rate is < 1 %.
    """

    def __init__(
        self,
        positives: Tensor,
        num_entities: int,
        triple_set: FrozenSet[Tuple[int, int, int]],
        neg_ratio: int = 10,
        max_attempts: int = 50,
    ):
        self.positives = positives
        self.num_entities = num_entities
        self.triple_set = triple_set
        self.neg_ratio = neg_ratio
        self.max_attempts = max_attempts

    def __len__(self) -> int:
        return len(self.positives)

    def __getitem__(self, idx: int) -> dict:
        pos = self.positives[idx]  # (3,) LongTensor
        s, r, o = int(pos[0]), int(pos[1]), int(pos[2])

        if self.neg_ratio == 0:
            return {
                "positive": pos,
                "negatives": torch.zeros(0, 3, dtype=torch.long),
            }

        negs = self._sample_negatives(s, r, o)
        return {"positive": pos, "negatives": negs}

    def _sample_negatives(self, s: int, r: int, o: int) -> Tensor:
        """Generate ``neg_ratio`` filtered negative triples."""
        negatives: list[list[int]] = []
        attempts = 0

        while len(negatives) < self.neg_ratio and attempts < self.max_attempts * self.neg_ratio:
            # Randomly corrupt subject or object
            corrupt_subject = random.random() < 0.5
            corrupt_entity = random.randrange(self.num_entities)

            if corrupt_subject:
                candidate = (corrupt_entity, r, o)
            else:
                candidate = (s, r, corrupt_entity)

            attempts += 1
            if candidate not in self.triple_set:
                negatives.append(list(candidate))

        # Pad with unfiltered samples if rejection rate is unusually high
        while len(negatives) < self.neg_ratio:
            if random.random() < 0.5:
                negatives.append([random.randrange(self.num_entities), r, o])
            else:
                negatives.append([s, r, random.randrange(self.num_entities)])

        return torch.tensor(negatives[:self.neg_ratio], dtype=torch.long)


# ---------------------------------------------------------------------------
# Collate helper for DataLoader
# ---------------------------------------------------------------------------

def kg_collate_fn(batch: list[dict]) -> dict[str, Tensor]:
    """Collate a list of KGTripleDataset items into a batch dict.

    Returns
    -------
    dict with:
    - ``"positives"``: shape ``(B, 3)``
    - ``"negatives"``: shape ``(B * neg_ratio, 3)`` or ``(0, 3)``
    """
    positives = torch.stack([item["positive"] for item in batch], dim=0)
    neg_list = [item["negatives"] for item in batch if len(item["negatives"]) > 0]
    if neg_list:
        negatives = torch.cat(neg_list, dim=0)
    else:
        negatives = torch.zeros(0, 3, dtype=torch.long)
    return {"positives": positives, "negatives": negatives}
