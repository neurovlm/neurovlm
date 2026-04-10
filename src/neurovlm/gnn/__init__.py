"""NeuroVLM GNN module.

Two tracks:

Track 1 — DiFuMo Soft Atlas GAT
    Encodes brain activation maps as node features on a functional-connectivity
    graph over 512 DiFuMo components and trains a GAT with an InfoNCE objective
    against SPECTER text embeddings.

Track 2 — Unified KG R-GCN
    Trains a relational GCN on the unified neuroscience knowledge graph
    (33,784 entities, 6 relation types, 329,566 edges) using a link prediction
    objective with DistMult scoring.  Produces entity/relation embeddings that
    can be used for KG completion and downstream retrieval.

Typical usage — Track 1
-----------------------
>>> from neurovlm.gnn.atlas import load_difumo_components, compute_difumo_coefficients
>>> from neurovlm.gnn.graph import build_brain_graph
>>> from neurovlm.gnn.dataset import BrainGraphDataset
>>> from neurovlm.gnn.model import BrainGAT
>>> from neurovlm.gnn.train import GATTrainer

Typical usage — Track 2
-----------------------
>>> from neurovlm.gnn.kg_data import load_kg, KGSplits
>>> from neurovlm.gnn.rgcn import RGCNLinkPredictor
>>> from neurovlm.gnn.kg_train import RGCNTrainer
"""

# Track 1 — GAT
from .atlas import load_difumo_components, compute_difumo_coefficients
from .graph import build_brain_graph, load_fc_matrix
from .dataset import BrainGraphDataset
from .model import BrainGAT, TextProjHead
from .train import GATTrainer

# Track 2 — R-GCN
from .kg_data import load_kg, KGData, KGSplits, KGTripleDataset, kg_collate_fn
from .rgcn import RGCNLinkPredictor
from .kg_train import RGCNTrainer, evaluate_link_prediction

__all__ = [
    # Track 1
    "load_difumo_components",
    "compute_difumo_coefficients",
    "build_brain_graph",
    "load_fc_matrix",
    "BrainGraphDataset",
    "BrainGAT",
    "TextProjHead",
    "GATTrainer",
    # Track 2
    "load_kg",
    "KGData",
    "KGSplits",
    "KGTripleDataset",
    "kg_collate_fn",
    "RGCNLinkPredictor",
    "RGCNTrainer",
    "evaluate_link_prediction",
]
