"""Backward-compatible metric imports.

Domain-specific implementations live in:
- :mod:`neurovlm.brain_to_text_metrics` for generated-text/B2T metrics.
- :mod:`neurovlm.text_to_brain_metrics` for brain-map/T2B metrics.
- :mod:`neurovlm.retrieval_metrics` for modality-agnostic ranking metrics.

This module remains as a stable import index for older notebooks and user code.
"""

from neurovlm.brain_to_text_metrics import (
    bertscore_single,
    bleu,
    nvlm_latent_similarity,
    rouge,
    semantic_similarity,
    token_f1,
)
from neurovlm.metric_utils import as_latent_batch
from neurovlm.retrieval_metrics import (
    bidirectional_retrieval_metrics,
    normalized_k_values,
    normalized_recall_curve_auc,
    recall_at_k,
    recall_curve,
    retrieval_metrics,
    retrieval_ranks,
)
from neurovlm.text_to_brain_metrics import (
    NCTDiceResult,
    bernoulli_bce,
    bits_per_pixel,
    compute_ae_performance,
    compute_metrics,
    dice,
    dice_percentile,
    dice_top_k,
    mni152_to_fsaverage_arrays,
    nct_dice_spin_test_nifti,
    nct_dice_spin_test_surface,
    pearson_correlation,
    permutation_pvalue,
    precompute_spin_permutations,
    psnr,
)

__all__ = [
    "NCTDiceResult",
    "as_latent_batch",
    "bernoulli_bce",
    "bertscore_single",
    "bidirectional_retrieval_metrics",
    "bits_per_pixel",
    "bleu",
    "compute_ae_performance",
    "compute_metrics",
    "dice",
    "dice_percentile",
    "dice_top_k",
    "mni152_to_fsaverage_arrays",
    "nct_dice_spin_test_nifti",
    "nct_dice_spin_test_surface",
    "normalized_k_values",
    "normalized_recall_curve_auc",
    "nvlm_latent_similarity",
    "pearson_correlation",
    "permutation_pvalue",
    "precompute_spin_permutations",
    "psnr",
    "recall_at_k",
    "recall_curve",
    "retrieval_metrics",
    "retrieval_ranks",
    "rouge",
    "semantic_similarity",
    "token_f1",
]
