# R-GCN Training Pipeline

Phase 3 of the NeuroVLM GNN experiment: train a Relational Graph Convolutional
Network (R-GCN) on the unified neuroscience knowledge graph for link prediction,
then extract entity and relation embeddings for downstream use.

---

## KG Build Pipeline (Phase 2)

The unified KG is built across four notebooks before training begins:

| step | notebook | output |
|---|---|---|
| 1 | `kg_unification.ipynb` | Initial unified KG: MeSH + CogAtlas + NLP graph merged into canonical entity table (33,784 nodes, 329,566 edges) |
| 2 | `mesh_qualifier_re.ipynb` | Qualifier-based edge typing: extracts 14.79M semantically-typed edges from MeSH annotation qualifiers → `nlp_kg/nlp_kg_edges_qualified.parquet` |
| 3 | `kg_edge_merge.ipynb` | Merges original edges + qualifier edges + 52 LLM-RE edges, deduplicates → overwrites `unified_kg_edges.parquet` with 15.1M edges |
| 4 | `nlp_graph.ipynb` | NLP KG preprocessing: filter 133K raw nodes → 12,841 typed nodes, 155K edges → `nlp_kg/nlp_kg_nodes.parquet` + `nlp_kg_edges.parquet` (input to step 1) |

> **Note on notebook output staleness:** `kg_unification.ipynb` outputs show the old 329,566-edge counts because it was last run before the qualifier step. The files on disk reflect the merged 15.1M-edge state.

---

## Input

| file | rows | description |
|---|---|---|
| `experiments/data/unified_kg/unified_kg_nodes.parquet` | 33,784 | Canonical entities (MeSH + CogAtlas + NLP-new nodes) |
| `experiments/data/unified_kg/unified_kg_edges.parquet` | 15,113,176 | Typed edges from all sources |

**Node source breakdown:**

| primary_source | count |
|---|---|
| `mesh` | 31,110 |
| `nlp` | 1,922 |
| `cogatlas` | 752 |

> CogAtlas nodes are present but CogAtlas edges are absent — the `kind_of` relation scrape produced zero edges that survived endpoint remapping.

**Relation types (8 total):**

| relation_type | count | % | source |
|---|---|---|---|
| `implicated_in` | 6,619,101 | 43.8% | mesh_qualifier |
| `associated_with_disorder` | 4,185,364 | 27.7% | mesh_qualifier |
| `treated_by` | 2,695,381 | 17.8% | mesh_qualifier |
| `used_in` | 1,290,297 | 8.5% | mesh_qualifier |
| `co_occurs_with` | 278,985 | 1.8% | mesh / nlp |
| `narrower_term_of` | 42,519 | 0.3% | mesh |
| `co_activates_with` | 933 | 0.0% | mesh |
| `expressed_in` | 596 | 0.0% | mesh |

**Edge source breakdown:**

| source_kg | count |
|---|---|
| `mesh_qualifier` | 14,786,202 |
| `mesh` | 303,310 |
| `nlp` | 23,612 |
| `llm_re` | 52 |

---

## Architecture

```
Entity indices  (33,784)
      ↓
nn.Embedding(33784, 512)            ← learned entity embeddings
      ↓  (33784, 512)
RGCNConv(512→512, num_rels=8,
         num_bases=None) + ReLU     ← full per-relation weight matrices
      ↓  (33784, 512)
RGCNConv(512→512, num_rels=8,
         num_bases=None)            ← no activation on final layer
      ↓  (33784, 512)               ← contextual entity embeddings

RotatE decoder (asymmetric):
  entity embeddings split into real/imag halves
  score(s, r, o) = -||e_s ∘ r - e_o||
  where r = unit-norm complex rotation per relation
```

**Why RotatE over DistMult?**  The new relations are directed and asymmetric
(`treated_by`, `expressed_in`, `implicated_in` ≠ their inverses). DistMult is
symmetric by construction and cannot distinguish `A treated_by B` from
`B treated_by A`. RotatE handles this correctly. The decoder mismatch is caught
automatically at resume time — old DistMult checkpoints start fresh.

**Relation-frequency weighting:**  With `implicated_in` at 44% and `expressed_in`
at <0.01%, plain inverse-frequency gives a ~11,000× weight ratio. The trainer
uses log-frequency weighting (`1 / log(count+1)`, mean-normalised) for ~2.5× spread.

---

## Step 7 — Data Preparation

**Code:** `src/neurovlm/gnn/kg_data.py`

```python
from neurovlm.gnn.kg_data import load_kg, KGSplits

kg = load_kg(
    nodes_path="experiments/data/unified_kg/unified_kg_nodes.parquet",
    edges_path="experiments/data/unified_kg/unified_kg_edges.parquet",
)
# kg.num_entities  = 33,784
# kg.num_relations = 8
# kg.triples       = 15,113,176

splits = KGSplits.from_kg(kg, train_frac=0.85, val_frac=0.075, seed=42)
# train ≈ 12.8M   val ≈ 1.1M   test ≈ 1.1M
```

Stratification ensures every relation type appears in all three splits.

**Negative sampling:** 10 corrupted negatives per positive, filtered against all
known triples to avoid suppressing true facts.

---

## Step 8 — Model Setup

**Code:** `src/neurovlm/gnn/rgcn.py`

```python
from neurovlm.gnn.rgcn import RGCNLinkPredictor

model = RGCNLinkPredictor(
    num_entities=num_entities,   # 33,784
    num_relations=num_relations, # 8
    emb_dim=512,
    num_bases=None,              # full decomposition (8 relations — no need for basis trick)
    num_layers=2,
    dropout=0.1,
    decoder='rotate',
)
```

---

## Step 9 — Training

**Code:** `src/neurovlm/gnn/kg_train.py`

Training runs on Google Colab (A100 GPU) via `rgcn_kg_colab.ipynb`.
Data files are read from Google Drive at `MyDrive/neurovlm/data/`.

```python
from neurovlm.gnn.kg_train import RGCNTrainer

trainer = RGCNTrainer(
    model=model,
    splits=splits,
    lr=1e-3,
    weight_decay=1e-4,
    n_epochs=1000,
    batch_size=16384,
    neg_ratio=10,
    eval_batch_size=512,
    val_interval=5,
    patience=75,
    lr_patience=10,
    lr_factor=0.5,
    graph_sample_size=6_000_000,  # subsample 6M edges for R-GCN message passing per epoch
    max_steps_per_epoch=250,      # 250 × 16384 = 4M triples/epoch ≈ 33s on A100
    device='auto',
    checkpoint_dir=CHECKPOINT_DIR,
    relation_weights=rel_weights, # log-frequency, mean-normalised
    resume_from=RESUME_PATH,      # auto-resumes from resume_rgcn.pt if present
    resume_checkpoint_interval=5,
)
trainer.fit()
```

**Throughput:** ~40–60s/epoch on A100. Full 1000-epoch run ≈ 11–17 hours.

**Checkpoints (saved to `MyDrive/neurovlm/checkpoints/rgcn/`):**
- `best_rgcn.pt` — saved on every validation MRR improvement
- `resume_rgcn.pt` — full state (weights + optimizer + scheduler + history) saved every 5 epochs

To resume after a Colab session crash: re-run all cells from the top.

### Evaluation metrics

Filtered ranking (masks known true triples):
- **MRR** — primary early-stopping signal
- **Hits@1**, **Hits@3**, **Hits@10**

---

## Step 10 — Extract and Save Embeddings

**Code:** `src/neurovlm/gnn/kg_train.py` — `save_embeddings()`

```python
test_metrics = trainer.evaluate_test()

trainer.save_embeddings(EMB_PATH)
# EMB_PATH = MyDrive/neurovlm/embeddings/entity_embeddings_v2.pt
```

The saved `.pt` file contains:

| key | shape | description |
|---|---|---|
| `entity_embeddings` | `(33784, 512)` | R-GCN contextual entity embeddings |
| `relation_embeddings` | `(8, 512)` | RotatE complex rotation per relation |
| `entity_to_idx` | `dict[str, int]` | canonical_id → row index |
| `idx_to_entity` | `dict[int, str]` | row index → canonical_id |
| `relation_to_idx` | `dict[str, int]` | relation_type → column index |
| `idx_to_relation` | `dict[int, str]` | column index → relation_type |

A companion `.pt.meta.json` file is saved alongside with the same ID maps in JSON.

### Nearest-neighbour spot-check

```python
import torch, json
from pathlib import Path

EMB_PATH = Path("MyDrive/neurovlm/embeddings/entity_embeddings_v2.pt")
emb_payload  = torch.load(EMB_PATH, weights_only=True)
entity_emb   = emb_payload["entity_embeddings"]

trainer.nearest_neighbours("D006624", k=10, entity_emb=entity_emb)
# D006624 = Hippocampus — expected: entorhinal cortex, spatial memory, Alzheimer disease, ...
```

---

## Output Files

| file | location | description |
|---|---|---|
| `unified_kg_nodes.parquet` | `experiments/data/unified_kg/` | 33,784 canonical entities |
| `unified_kg_edges.parquet` | `experiments/data/unified_kg/` | 15,113,176 typed edges |
| `unified_kg_edges_backup.parquet` | `experiments/data/unified_kg/` | Pre-merge backup (329,566 edges) |
| `best_rgcn.pt` | `MyDrive/neurovlm/checkpoints/rgcn/` | Best model checkpoint |
| `resume_rgcn.pt` | `MyDrive/neurovlm/checkpoints/rgcn/` | Full resume state |
| `entity_embeddings_v2.pt` | `MyDrive/neurovlm/embeddings/` | Entity + relation embeddings |
| `rgcn_training_curves.png` | `MyDrive/neurovlm/` | Loss + MRR training curves |
| `rgcn_umap.png` | `MyDrive/neurovlm/` | UMAP of entity embedding space |

---

## Design decisions

| decision | rationale |
|---|---|
| `emb_dim=512` | Larger capacity for 15M-edge graph; 256 was undersized for 8 relation types |
| `num_bases=None` | With 8 relations basis decomposition is unnecessary; full matrices per relation |
| `num_layers=2` | 3 layers risk over-smoothing on dense `implicated_in` subgraph (44% of edges) |
| RotatE decoder | Handles asymmetric directed relations (`treated_by`, `expressed_in`); DistMult is symmetric by construction |
| log-frequency relation weights | 11,000× count ratio between rarest/most-frequent relation makes inverse-frequency unstable; log compression keeps spread at ~2.5× |
| `graph_sample_size=6M` | Samples 6M edges for R-GCN message passing — covers 40% of graph per epoch, balancing neighbourhood quality vs. memory/speed |
| `max_steps_per_epoch=250` | 4M triples/epoch at bs=16384 gives ~33s/epoch on A100 without exhausting the full 12.8M train set every time |
| BCE loss | More stable than margin ranking loss at this scale |
| Filtered negatives | Prevents suppressing true facts that appear as corruptions |
| Stratified split | Ensures rare relation types (`expressed_in`: 596) appear in val/test |
