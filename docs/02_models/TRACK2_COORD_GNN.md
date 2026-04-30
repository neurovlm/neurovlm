# Track 2: Atlas-Free Coordinate GNN

## What This Is

Track 2 is a brain encoder that learns directly from raw MNI peak coordinates — the (x, y, z) locations where a neuroimaging study found significant activation — without ever consulting a brain atlas.

Every neuroscience paper that reports fMRI or PET results includes a table of peak coordinates in MNI space. These coordinates are the most fundamental, atlas-independent representation of what a study found. Track 2 treats them as a spatial point cloud, builds a K-nearest-neighbor graph over them, and trains a Graph Attention Network to compress that spatial structure into a 384-dimensional embedding.

The output embedding is contrastively trained against the paper's SPECTER text embedding using InfoNCE loss, so the resulting vector lives in the same shared latent space as the text side of NeuroVLM.

---

## Why We Built This

Track 1 (DiFuMo GAT) is atlas-dependent. It projects raw brain images onto 512 DiFuMo components and builds a graph over those components. This is powerful, but it has two costs:

1. **Atlas assumption**: the DiFuMo parcellation decides which brain regions exist and where their boundaries are. A paper reporting activation in a region the atlas splits awkwardly will be misrepresented.
2. **Fixed topology**: every paper's graph has exactly 512 nodes with the same edges. Only the node *values* change. The graph structure carries no information about the specific spatial pattern of that paper's activations.

Track 2 removes both constraints. The number of nodes equals the number of reported peaks. The edges are determined by the actual spatial relationships between those peaks. Two papers with activations in completely different brain systems produce completely different graph structures — the model must learn from the geometry itself.

This matters because:
- About 30,000 papers in NeuroVLM have coordinate tables but no full activation maps
- Coordinate-based encoding scales to any new paper immediately (no atlas download, no image preprocessing)
- The spatial graph is interpretable: high-attention edges correspond to anatomically meaningful coordinate pairs

---

## What It Does

### Input
A set of MNI peak coordinates from one paper — for example, a paper on working memory might report 23 peaks scattered across prefrontal, parietal, and thalamic regions.

### Processing pipeline

```
Raw MNI coords (N × 3, mm space)
        ↓  normalize: x÷90, y÷126, z÷108
Normalized coords (N × 3, values in [-1, 1])
        ↓  deduplicate within paper
        ↓  KNN graph (k=7, max edge distance 30mm)
        ↓  add hemisphere flag + depth proxy
Node features (N × 5): [x, y, z, hemisphere, depth_proxy]
Edge features (E × 4): [dist_normalized, dx, dy, dz]
        ↓
CoordGNN encoder
  • 2-layer MLP input projection (LayerNorm + GELU)
  • GATConv layer 1: hidden=128, heads=8  → (N, 1024)
  • GATConv layer 2: hidden=128, heads=8  → (N, 1024)
  • GATConv layer 3: hidden=128, heads=1  → (N, 128)
  • global_mean_pool                      → (1, 128)
  • Linear projection                     → (1, 384)
        ↓
Brain embedding (384-dim)
```

The text side uses the same frozen SPECTER encoder + TextProjHead as Track 1, projecting to the same 384-dim space.

### Output
A 384-dimensional vector that represents the paper's spatial activation pattern. This vector is directly comparable to embeddings from Track 1 and the original NeuroVLM MLP — same dimensionality, same contrastive objective, same evaluation protocol.

---

## Architecture Decisions

### edge_dim=4 in every GATConv
Each edge carries four features: normalized distance, and the dx/dy/dz direction vector. These are passed into the attention computation via `edge_dim=4`. Without this, PyG silently ignores the edge features and the model never uses spatial distance or direction — it would learn no better than a node-feature-only model.

### LayerNorm, not BatchNorm
Each paper has a different number of peaks, so batches contain graphs of wildly different sizes. BatchNorm computes statistics over the batch dimension, which is unstable when batch sizes (in terms of total nodes) swing from 200 to 5,000. LayerNorm normalizes per sample and is correct for variable-size graphs.

### GELU in the input projection
The input features are continuous normalized coordinates. GELU's smooth gradient around zero is better suited to this than ReLU's hard cutoff, especially in the early training epochs when the model is learning what coordinate ranges mean.

### Pre-computed disk cache
Building KNN graphs inside the DataLoader at training time would make each epoch ~50× slower. All graphs are built once and saved as `.pt` files in `data/coord_graphs/` before training starts. The cache is keyed by paper index and is reused across runs.

---

## Files

| File | Role |
|------|------|
| `src/neurovlm/gnn/coord_graph.py` | `coords_to_graph()` — converts a (N,3) normalized array into a PyG `Data` object with 5-dim node and 4-dim edge features |
| `src/neurovlm/gnn/coord_dataset.py` | `CoordGraphDataset` — aligns coordinates with SPECTER embeddings by PMID, builds the disk cache, exposes `split()` and `split_by_index()` |
| `src/neurovlm/gnn/coord_model.py` | `CoordGNN` — the 3-layer GAT encoder described above |
| `src/neurovlm/gnn/coord_train.py` | `CoordTrainer` — InfoNCE training loop with embedding collapse monitor, cosine LR schedule, and dual checkpoints |
| `experiments/train_coord_gnn.py` | End-to-end CLI training script (runs locally on MPS) |
| `experiments/coord_gnn_colab.ipynb` | Google Colab A100 notebook — same pipeline, saves to Google Drive |

---

## Training

**Local (MPS, MacBook M-series):**
```bash
python experiments/train_coord_gnn.py
```

**Google Colab (A100):**
Open `experiments/coord_gnn_colab.ipynb`. Set your repo URL in cell 1. Switch runtime to A100. Run all cells. Checkpoints and the graph cache are saved to Google Drive.

Key hyperparameters:

| Hyperparameter | Local (MPS) | Colab (A100) |
|----------------|-------------|--------------|
| Epochs | 200 | 200 |
| Batch size | 64 | 128 |
| LR (GNN) | 1e-4 | 1e-4 |
| LR (TextProj) | 1e-5 | 1e-5 |
| Warmup epochs | 15 | 15 |
| Temperature | 0.07 | 0.07 |
| KNN k | 7 | 7 |
| Max edge dist | 30mm | 30mm |

---

## Monitoring During Training

Each validation step (every 5 epochs) prints:

```
Epoch  50/200 | loss=4.1230 | t2i=0.6412 i2t=0.6389 | embed_sim=0.312 | lr=8.23e-05
```

**`embed_sim`** is the collapse monitor: it measures the mean pairwise cosine similarity of 256 random validation embeddings. If it rises above 0.95, all embeddings are collapsing toward the same vector — training has failed and should be restarted after adding a `LayerNorm` after the final projection layer.

Two checkpoints are saved:
- `checkpoints/coord_gnn/best_coord_gnn.pt` — best validation AUC (load this for evaluation)
- `checkpoints/coord_gnn/last_coord_gnn.pt` — final epoch (useful after Colab disconnects)

---

## Evaluation

The evaluation protocol is identical to Track 1 and the NeuroVLM baseline: recall@1/5/10 and AUC in both text→brain and brain→text directions.

Target comparison table:

| Model | AUC | Atlas-free |
|-------|-----|------------|
| NeuroVLM MLP (baseline) | ~0.81 | No |
| Track 1 DiFuMo GAT | TBD | No |
| Track 2 Coord GNN | TBD | **Yes** |

An attention analysis is also run automatically: for 10 test papers, the top-5 highest-attention edges are printed with their MNI coordinates, so you can check that high-attention connections correspond to known anatomical relationships rather than noise.

---

## What "Atlas-Free" Actually Means

At no point in the Track 2 pipeline is any atlas loaded, downloaded, or consulted. The model sees only:

1. The list of (x, y, z) coordinates reported in the paper
2. The paper's abstract (via SPECTER, on the text side)

There is no DiFuMo, no AAL, no Schaefer, no Harvard-Oxford. The parcellation is emergent — the model learns which spatial configurations co-occur with which types of text, building its own implicit parcellation from the data.
