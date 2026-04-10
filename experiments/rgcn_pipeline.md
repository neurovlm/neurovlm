# R-GCN Training Pipeline

Phase 3 of the NeuroVLM GNN experiment: train a Relational Graph Convolutional
Network (R-GCN) on the unified neuroscience knowledge graph for link prediction,
then extract entity and relation embeddings for downstream use.

---

## Input

The unified KG produced by Phase 2 (`experiments/kg_unification.ipynb`):

| file | description |
|---|---|
| `experiments/data/unified_kg/unified_kg_nodes.parquet` | 33,784 canonical entities |
| `experiments/data/unified_kg/unified_kg_edges.parquet` | 329,566 typed edges |

**Relation types (6 total, contiguous 0-based integers at runtime):**

| relation_type | count |
|---|---|
| `co_occurs_with` | 281,629 |
| `narrower_term_of` | 42,519 |
| `associated_with_disorder` | 2,107 |
| `implicated_in` | 1,803 |
| `co_activates_with` | 931 |
| `expressed_in` | 577 |

---

## Architecture

```
Entity indices  (33,784)
      ↓
nn.Embedding(33784, 256)            ← learned entity embeddings
      ↓  (33784, 256)
RGCNConv(256→256, num_rels=6,
         num_bases=4)  + ReLU       ← basis decomposition keeps param count low
      ↓  (33784, 256)
RGCNConv(256→256, num_rels=6,
         num_bases=4)               ← no activation on final layer
      ↓  (33784, 256)               ← contextual entity embeddings

DistMult decoder:
  score(s, r, o) = Σ_d  e_s[d] · W_r[d] · e_o[d]
  where W_r = nn.Embedding(6, 256) — one diagonal matrix per relation
```

**Why basis decomposition?**  With 6 relation types and `emb_dim=256`, using
`num_bases=4` means each relation matrix is a linear combination of 4 shared
basis matrices (`4 × 256² / 6` ≈ 10× fewer parameters than full decomposition).
This regularises rare relation types (`expressed_in`: 577 edges) against
overfitting.

---

## Step 7 — Data Preparation

**Code:** `src/neurovlm/gnn/kg_data.py`

### Entity and relation indexing

```python
from neurovlm.gnn.kg_data import load_kg
kg = load_kg(
    "experiments/data/unified_kg/unified_kg_nodes.parquet",
    "experiments/data/unified_kg/unified_kg_edges.parquet",
)
# kg.num_entities = 33784
# kg.num_relations = 6  (contiguous 0-based)
```

### Train / val / test split — stratified by relation type

```python
from neurovlm.gnn.kg_data import KGSplits
splits = KGSplits.from_kg(kg, train_frac=0.85, val_frac=0.075, seed=42)
# Split: train=280,131  val=24,717  test=24,718
```

Stratification ensures every relation type, including the rarest
(`expressed_in`: 577 edges → ~49 val, 49 test), appears in all three splits.

### Negative sampling

For each positive triple `(s, r, o)`, generate 10 corrupted negatives by
randomly replacing either the subject or object with a random entity.

**Filtered negatives:** before accepting a corrupted triple, check it against
the full triple set (all 329,566 known triples).  If the corrupted triple is
actually true, resample.  This prevents training the model to suppress real
facts.

```python
train_ds = splits.train_dataset(neg_ratio=10)
# Each __getitem__ returns:
# {
#   "positive": LongTensor([s, r, o]),          # shape (3,)
#   "negatives": LongTensor([..., 3])            # shape (10, 3)
# }
```

---

## Step 8 — Model Setup

**Code:** `src/neurovlm/gnn/rgcn.py`

```python
from neurovlm.gnn.rgcn import RGCNLinkPredictor

model = RGCNLinkPredictor(
    num_entities=kg.num_entities,   # 33,784
    num_relations=kg.num_relations, # 6
    emb_dim=256,
    num_bases=4,
    num_layers=2,
    dropout=0.1,
)
```

**Parameter count (emb_dim=256):**
- Entity embeddings: 33,784 × 256 ≈ 8.6 M
- RGCN layer 1 + 2: 4 bases × 256 × 256 × 2 layers ≈ 524 K
- Relation (DistMult): 6 × 256 ≈ 1.5 K
- **Total: ~9.1 M parameters**

The graph comfortably fits in CPU RAM; with a GPU the full forward pass
runs in seconds per epoch.

---

## Step 9 — Training Loop

**Code:** `src/neurovlm/gnn/kg_train.py`

```python
from neurovlm.gnn.kg_train import RGCNTrainer

trainer = RGCNTrainer(
    model=model,
    splits=splits,
    lr=1e-3,
    n_epochs=500,
    batch_size=1024,
    neg_ratio=10,
    val_interval=10,   # evaluate val MRR every 10 epochs
    patience=30,       # early stopping: 30 epochs without MRR improvement
    checkpoint_dir="checkpoints/rgcn",
)
trainer.fit()
```

### Loss

Binary cross-entropy on positive (label=1) vs. negative (label=0) triples:

```
L = -Σ [ log σ(score_pos) + log(1 − σ(score_neg)) ]
```

### Evaluation (filtered MRR, Hits@k)

For each validation triple `(s, r, o)`:
1. Score all 33,784 entities as candidate objects using DistMult.
2. Mask out other known true objects for `(s, r, ?)` (filtered setting).
3. Compute rank of the true object `o`.

**Metrics:**
- **MRR** — Mean Reciprocal Rank (primary early-stopping signal)
- **Hits@1** — fraction of queries where true object ranks first
- **Hits@3** — fraction where true object ranks in top 3
- **Hits@10** — fraction where true object ranks in top 10

```
Epoch  100/500 | loss=0.3821 | MRR=0.1842 | H@1=0.1103 | H@3=0.2047 | H@10=0.3518 | lr=1.00e-03
Epoch  200/500 | loss=0.2914 | MRR=0.2371 | H@1=0.1512 | H@3=0.2804 | H@10=0.4213 | lr=5.00e-04
```

### LR schedule

`ReduceLROnPlateau(mode="max", factor=0.5, patience=10)` on validation MRR.
Initial LR 1e-3 decays by 0.5× whenever MRR plateaus for 10 epochs.

---

## Step 10 — Extract and Save Embeddings

**Code:** `src/neurovlm/gnn/kg_train.py` — `save_embeddings()`

```python
test_metrics = trainer.evaluate_test()
# Test | MRR=0.2341 | H@1=0.1489 | H@3=0.2768 | H@10=0.4198

trainer.save_embeddings(
    "experiments/data/unified_kg/entity_embeddings.pt"
)
```

The saved `.pt` file contains:

| key | shape | description |
|---|---|---|
| `entity_embeddings` | `(33784, 256)` | R-GCN contextual entity embeddings |
| `relation_embeddings` | `(6, 256)` | DistMult diagonal relation matrices |
| `entity_to_idx` | `dict[str, int]` | canonical_id → row index |
| `idx_to_entity` | `dict[int, str]` | row index → canonical_id |
| `relation_to_idx` | `dict[str, int]` | relation_type → column index |
| `idx_to_relation` | `dict[int, str]` | column index → relation_type |

### Nearest-neighbour spot-check

```python
# Load saved embeddings for offline checking
payload = torch.load("experiments/data/unified_kg/entity_embeddings.pt")
emb = payload["entity_embeddings"]

# Check hippocampus neighbourhood
neighbours = trainer.nearest_neighbours("D006624", k=10, entity_emb=emb)
# Expected: entorhinal cortex, spatial memory, Alzheimer disease, ...
for sim, eid in neighbours:
    print(f"  {sim:.4f}  {eid}")
```

If hippocampus is near retina or cardiac muscle, entity normalisation likely
failed — check the MeSH DescriptorUI mapping in `unified_kg_nodes.parquet`.

---

## Output Files

All outputs written to `experiments/data/unified_kg/`:

| file | description |
|---|---|
| `entity_embeddings.pt` | Entity + relation embeddings with ID maps |
| `checkpoints/rgcn/best_rgcn.pt` | Best model checkpoint (state dict + metrics) |

---

## Quick-start (end-to-end)

```python
from pathlib import Path
from neurovlm.gnn import load_kg, KGSplits, RGCNLinkPredictor, RGCNTrainer

DATA = Path("experiments/data/unified_kg")

kg     = load_kg(DATA / "unified_kg_nodes.parquet", DATA / "unified_kg_edges.parquet")
splits = KGSplits.from_kg(kg, train_frac=0.85, val_frac=0.075, seed=42)
model  = RGCNLinkPredictor(num_entities=kg.num_entities, num_relations=kg.num_relations)

trainer = RGCNTrainer(
    model, splits,
    n_epochs=500,
    checkpoint_dir="checkpoints/rgcn",
)
trainer.fit()
trainer.evaluate_test()
trainer.save_embeddings(DATA / "entity_embeddings.pt")
```

---

## Design decisions

| decision | rationale |
|---|---|
| `emb_dim=256` | Balances capacity vs. memory; 512 is viable on GPU |
| `num_bases=4` | Regularises the 6 relations, especially the 3 with <2k edges |
| `num_layers=2` | 3 layers showed marginal gain but increased over-smoothing risk on the dense `co_occurs_with` subgraph |
| DistMult decoder | Simpler than TransE; works well on symmetric relations (`co_occurs_with` is 85% of edges) |
| BCE loss | More stable than margin ranking loss at this scale |
| Filtered negatives | Prevents suppressing true facts that happen to appear as corruptions |
| Stratified split | Ensures rare relation types (`expressed_in`: 577) appear in val/test |
