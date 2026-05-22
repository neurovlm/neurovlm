"""Brain-to-text evaluation metrics and notebook workflow helpers."""

from neurovlm.metrics import bertscore_single, nvlm_latent_similarity, semantic_similarity
from neurovlm.evaluation_notebook_utils import (
    add_network_label_accuracy,
    exact_term_ranking_outputs,
    generated_text_metric_summary,
    generated_text_pair_baseline,
    generated_text_retrieval_curve,
    make_b2t_runner,
    make_network_label_accuracy_adder,
    predownload_hf_model,
    run_b2t_sample,
    run_network_gold_term_ranking,
    run_paper_retrieval_eval,
    run_paper_retrieval_evaluations,
    run_pubmed_mesh_gold_ranking,
)

__all__ = [
    "add_network_label_accuracy",
    "bertscore_single",
    "exact_term_ranking_outputs",
    "generated_text_metric_summary",
    "generated_text_pair_baseline",
    "generated_text_retrieval_curve",
    "make_b2t_runner",
    "make_network_label_accuracy_adder",
    "nvlm_latent_similarity",
    "predownload_hf_model",
    "run_b2t_sample",
    "run_network_gold_term_ranking",
    "run_paper_retrieval_eval",
    "run_paper_retrieval_evaluations",
    "run_pubmed_mesh_gold_ranking",
    "semantic_similarity",
]
