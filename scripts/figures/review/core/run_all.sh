#!/usr/bin/env bash
# Rerun every analysis in core/ one after the other (one GPU job at a time), re-plot its figure, then rebuild ../reviewer_0N/.
#
#   cd neurovlm && nohup bash scripts/figures/review/core/run_all.sh > /dev/null 2>&1 &
#   bash scripts/figures/review/core/run_all.sh a3_neurovault_bounds a7_ood_confidence   # only these directories
#   FIGURES_ONLY=1 bash scripts/figures/review/core/run_all.sh                           # only re-plot (CPU, ~2 min)
#
# Needs the final model (scripts/train_pipeline.py), its validation (scripts/validate_models.py, read by a2_granularity_tradeoff and a5_mlp_projector)
# and the paper figures (scripts/figures/main/make_all.py, read by a7_failure_cases_percentiles).
# Logs: core/logs/<dir>.log and core/logs/status.tsv (dir, script, exit code, seconds); a failed step does not stop the queue.
# ~2.5 h on one RTX 5090; A5_FULL=1 adds the 8-h A5.1 projector study.  DEVICE (default cuda) is passed as --device.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."   # repository root
PY=".venv/bin/python"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"  # fixed set order (bert_score batches texts via a set)
CORE="scripts/figures/review/core"
RUN="scripts/outputs/runs/final"
DEV="${DEVICE:-cuda}"
LOGS="$CORE/logs"; mkdir -p "$LOGS"; STATUS="$LOGS/status.tsv"; touch "$STATUS"
ONLY=" $* "

# step <dir> <script> [flags...]: run one script, append its output to the log, record exit code and seconds
step () {
  local sub="$1" script="$2" t0 rc; shift 2
  if [ "${FIGURES_ONLY:-0}" = "1" ] && [[ "$script" != make_figure.py && "$script" != plot_correctness.py ]]; then return 0; fi
  echo "===== $(date '+%F %T') $script $*" >> "$LOGS/$sub.log"
  echo "[$(date +%H:%M)] $sub/$script $*"
  t0=$(date +%s)
  $PY "$CORE/$sub/$script" "$@" >> "$LOGS/$sub.log" 2>&1; rc=$?
  printf "%s\t%s\t%s\t%s\n" "$sub" "$script" "$rc" "$(( $(date +%s) - t0 ))" >> "$STATUS"
  [ $rc -eq 0 ] || echo "[$(date +%H:%M)] $sub/$script FAILED (exit $rc, see $LOGS/$sub.log)"
}
want () { [ "$ONLY" = "  " ] || [[ "$ONLY" == *" $1 "* ]]; }

# 2.4 / 1.2 leakage audit (CPU; --write-exclusions scripts/eval_exclusions would also rewrite the exclusion lists)
want a4_leakage_audit && { step a4_leakage_audit run_analysis.py --force; step a4_leakage_audit make_figure.py; }
# 1.2 / 2.2 / 1.5 counts (CPU; per-stage sizes from the final run's summaries)
want a4_counts_table && { step a4_counts_table run_analysis.py --model-dir "$RUN" --force; step a4_counts_table make_figure.py; }
# 1.2 / 2.2 data-flow diagram (CPU)
want a4_dataflow_diagram && { step a4_dataflow_diagram run_analysis.py --model-dir "$RUN" --force; step a4_dataflow_diagram make_figure.py; }
# 3.8 table captions (CPU; uses the cached PMC-OA download in outputs/pubget_data)
want a4_caption_audit && { step a4_caption_audit run_analysis.py --skip-download --n-jobs 12; step a4_caption_audit caption_usability.py
                           step a4_caption_audit more_bad_examples.py; step a4_caption_audit make_figure.py; }
# 2.m5 memorization: do test captions copy their abstracts more than training captions? (GPU ~5 min, then cached;
# DAPT_CORPUS = NeuroQwen3's PubMed pretraining parquet, optional: counts the papers it holds)
want a4_memorization_check && { DAPT_CORPUS="${DAPT_CORPUS:-$HOME/efficient_ai/project/pubmed/pubmed_neuroimaging.parquet}"; DAPT=()
                                 [ -f "$DAPT_CORPUS" ] && DAPT=(--dapt-corpus "$DAPT_CORPUS")
                                 step a4_memorization_check run_analysis.py --model-dir "$RUN" --device "$DEV" "${DAPT[@]}"; }
# 1.1 retrieval metrics and hard negatives
want a1_retrieval_metrics_table && { step a1_retrieval_metrics_table run_analysis.py --model-dir "$RUN" --device "$DEV" --force
                                     step a1_retrieval_metrics_table make_figure.py; }
# 1.1 / 2.5 / 3.2 text-to-brain baselines (also writes the Neurosynth cache the a6_* steps reuse)
want a1_text_to_brain_baselines && { step a1_text_to_brain_baselines run_text_to_brain.py --model-dir "$RUN" --device "$DEV"
                                     step a1_text_to_brain_baselines make_figure.py; }
# 1.1 / 3.3 brain-to-text generation vs retrieval
want a1_brain_to_text_baselines && { step a1_brain_to_text_baselines run_analysis.py --model-dir "$RUN" --device "$DEV" --force
                                     step a1_brain_to_text_baselines make_figure.py; }
# 3.3 granularity / canonical projection (final model)
want a2_granularity_tradeoff && { step a2_granularity_tradeoff run_analysis.py --model-dir "$RUN" --device "$DEV" --force; step a2_granularity_tradeoff make_figure.py; }
# 1.3 / 2.3 / 2.8 NeuroVault bounds
want a3_neurovault_bounds && { step a3_neurovault_bounds run_analysis.py --model-dir "$RUN" --device "$DEV" --force --n-jobs 12 --qc-count-all
                               step a3_neurovault_bounds make_figure.py; }
# 3.4 / 3.7 context-free encoder (heads retrained on the final run's latents)
want a5_encoder_swap && { step a5_encoder_swap run_analysis.py --model-dir "$RUN" --device "$DEV" --force; step a5_encoder_swap make_figure.py; }
# 3.5 loss ablation
want a5_loss_ablation && { step a5_loss_ablation run_analysis.py --model-dir "$RUN" --device "$DEV" --force; step a5_loss_ablation make_figure.py; }
# 2.1 MLP projector: the figure reads scripts/outputs/validation/; A5_FULL=1 also reruns the A5.1 variant study
# (8 h, a5_mlp_projector/run_variants.sh, which ends with make_figure.py --table)
if want a5_mlp_projector; then
  if [ "${A5_FULL:-0}" = "1" ] && [ "${FIGURES_ONLY:-0}" != "1" ]; then
    echo "===== $(date '+%F %T') run_variants.sh" >> "$LOGS/a5_mlp_projector.log"
    bash "$CORE/a5_mlp_projector/run_variants.sh" >> "$LOGS/a5_mlp_projector.log" 2>&1
  fi
  step a5_mlp_projector make_figure.py
fi
# 3.5 latent dimensions
want a6_latent_interpretation && { step a6_latent_interpretation run_analysis.py --model-dir "$RUN" --device "$DEV" --force
                                   step a6_latent_interpretation select_dimensions.py
                                   step a6_latent_interpretation make_figure.py; }
# 3.6 network-, anatomy- and cognition-only text to brain
want a6_modality_restricted_t2b && { step a6_modality_restricted_t2b run_analysis.py --model-dir "$RUN" --device "$DEV" --force
                                     step a6_modality_restricted_t2b make_figure.py; }
# 3.5 perturbation probes
want a6_perturbation_probes && { step a6_perturbation_probes run_analysis.py --model-dir "$RUN" --device "$DEV" --force
                                 step a6_perturbation_probes make_figure.py; }
# 3.6 [NETWORK] / [REGION] / [FUNCTION] correctness (final model)
want a6_steering_tokens && { step a6_steering_tokens run_correctness.py --model-dir "$RUN" --device "$DEV"; step a6_steering_tokens plot_correctness.py; }
# 2.12 / 3.9 failure cases and example percentiles (reads the paper figures' *_text.json in scripts/figures/main/)
want a7_failure_cases_percentiles && { step a7_failure_cases_percentiles run_analysis.py --model-dir "$RUN" --device "$DEV" --gen-batch 64
                                       step a7_failure_cases_percentiles make_figure.py; }
# 1.4 OOD detection (CPU)
want a7_ood_confidence && { (export CUDA_VISIBLE_DEVICES=""; step a7_ood_confidence run_analysis.py --model-dir "$RUN"); step a7_ood_confidence make_figure.py; }
# 1.8 memory and latency (needs an idle GPU)
want a8_edge_benchmark && { step a8_edge_benchmark run_analysis.py --device auto --require-idle-gpu --force; step a8_edge_benchmark make_figure.py; }

$PY scripts/figures/review/build_reviewer_dirs.py | tee -a "$LOGS/build_reviewer_dirs.log"
echo "[$(date +%H:%M)] done; status: $STATUS"
