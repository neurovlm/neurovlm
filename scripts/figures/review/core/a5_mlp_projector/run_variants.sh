#!/usr/bin/env bash
# A5.1 variant study: one GPU command per projector variant, run one after the other (~72 min each, ~7.5 h on one RTX 5090).
# Each command is resumable: finished variants are skipped and an interrupted training resumes from its last epoch.
# Needs >= ~24 GB free GPU memory (below that LM gradient checkpointing turns on, ~30 % slower).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.."   # repository root
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
PY=.venv/bin/python
A5=scripts/figures/review/core/a5_mlp_projector
mkdir -p $A5/outputs/logs

for v in mlp_m mlp_l mlp_xl mlp_s mlp_m_lowlr mlp_m_notoken; do
  $PY $A5/run_analysis.py --variants $v --device cuda 2>&1 | tee -a $A5/outputs/logs/$v.log
done
$PY $A5/run_analysis.py --aggregate-only 2>&1 | tee -a $A5/outputs/logs/aggregate.log
$PY $A5/make_figure.py --table
