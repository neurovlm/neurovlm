"""NeuroVLM GNN module.

Unified KG R-GCN
    Trains a relational GCN on the unified neuroscience knowledge graph
    (33,784 entities, 6 relation types, 329,566 edges) using a link prediction
    objective with DistMult scoring.  Produces entity/relation embeddings that
    can be used for KG completion and downstream retrieval.

ALE dense CNN
    Atlas-free 3D CNN encoder/decoder over dense ALE volumes, used by the
    contrastive and text-to-brain generation pipelines in
    `experiments/3dcnn/`.

Typical usage — KG
------------------
>>> from neurovlm.gnn.kg_data import load_kg, KGSplits
>>> from neurovlm.gnn.rgcn import RGCNLinkPredictor
>>> from neurovlm.gnn.kg_train import RGCNTrainer

Typical usage — ALE CNN
------------------------
>>> from neurovlm.gnn.ale_cnn import ALE3DCNNEncoder
>>> from neurovlm.gnn.ale_dataset import ALEVolumeDataset
>>> from neurovlm.gnn.model import TextProjHead
"""

# atlas utilities (DiFuMo components, used by semantic evaluation)
from .atlas import load_difumo_components, compute_difumo_coefficients

# Unified KG — R-GCN
from .kg_data import load_kg, KGData, KGSplits, KGTripleDataset, kg_collate_fn
from .rgcn import RGCNLinkPredictor
from .kg_train import RGCNTrainer, evaluate_link_prediction

# ALE dense CNN
from .model import TextProjHead
from .ale_cnn import ALE3DCNNEncoder, ALEFlatMLPEncoder
from .ale_dataset import ALEPreprocessConfig, ALEVolumeDataset, build_or_load_ale_cache

__all__ = [
    # atlas utilities
    "load_difumo_components",
    "compute_difumo_coefficients",
    # Unified KG — R-GCN
    "load_kg",
    "KGData",
    "KGSplits",
    "KGTripleDataset",
    "kg_collate_fn",
    "RGCNLinkPredictor",
    "RGCNTrainer",
    "evaluate_link_prediction",
    # ALE dense CNN
    "TextProjHead",
    "ALE3DCNNEncoder",
    "ALEFlatMLPEncoder",
    "ALEPreprocessConfig",
    "ALEVolumeDataset",
    "build_or_load_ale_cache",
]
