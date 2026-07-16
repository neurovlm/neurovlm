"""Compatibility imports for the packaged atlas-free CNN architecture.

The runtime model definitions live in :mod:`neurovlm.ale_cnn` so they are
available when NeuroVLM is installed from a wheel.  Experiment scripts keep
importing this module for backwards compatibility.
"""

from neurovlm.ale_cnn import (
    ALE3DCNNAutoEncoder,
    ALE3DCNNDecoder,
    ALE3DCNNEncoder,
    ALEResNet3DEncoder,
    GlobalContextType,
    ModelSummary,
    NormType,
    PoolType,
    count_parameters,
    embedding_covariate_correlations,
    summarize_encoder,
    validate_retained_resnet_architecture,
)

__all__ = [
    "ALE3DCNNAutoEncoder",
    "ALE3DCNNDecoder",
    "ALE3DCNNEncoder",
    "ALEResNet3DEncoder",
    "GlobalContextType",
    "ModelSummary",
    "NormType",
    "PoolType",
    "count_parameters",
    "embedding_covariate_correlations",
    "summarize_encoder",
    "validate_retained_resnet_architecture",
]
