"""Reusable evaluation primitives for NeuroVLM training and inference."""

from .contrastive import ContrastiveEvaluation, evaluate_contrastive
from .brain_to_text import (
    BrainToTextBatch,
    BrainToTextGenerationEvaluation,
    BrainToTextLMOutput,
    brain_to_text_lm_forward,
    evaluate_brain_to_text_generation,
    parse_brain_to_text_batch,
)
from .mlp import (
    MLPContrastiveEvaluation,
    MLPEvaluation,
    evaluate_mlp_autoencoder,
    evaluate_mlp_contrastive,
    evaluate_mlp_text_to_brain,
)
from .spatial import reconstruction_metrics, voxel_auroc
from .text_to_brain import TextToBrainEvaluation, evaluate_text_to_brain
from .comparison import (
    ComparisonResult,
    ComparisonSelection,
    default_comparison_matrix,
    evaluate_contrastive_comparison,
    evaluate_reconstruction_comparison,
    evaluate_text_to_brain_comparison,
    resolve_comparison_manifest,
    write_comparison_manifest,
)

__all__ = [
    "ContrastiveEvaluation",
    "BrainToTextBatch",
    "BrainToTextGenerationEvaluation",
    "BrainToTextLMOutput",
    "brain_to_text_lm_forward",
    "evaluate_brain_to_text_generation",
    "parse_brain_to_text_batch",
    "evaluate_contrastive",
    "MLPContrastiveEvaluation",
    "MLPEvaluation",
    "evaluate_mlp_autoencoder",
    "evaluate_mlp_contrastive",
    "evaluate_mlp_text_to_brain",
    "reconstruction_metrics",
    "TextToBrainEvaluation",
    "evaluate_text_to_brain",
    "voxel_auroc",
    "ComparisonResult",
    "ComparisonSelection",
    "default_comparison_matrix",
    "evaluate_contrastive_comparison",
    "evaluate_reconstruction_comparison",
    "evaluate_text_to_brain_comparison",
    "resolve_comparison_manifest",
    "write_comparison_manifest",
]
