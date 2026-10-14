#!/usr/bin/env bash
# Smoke test of the training pipeline: every stage with tiny settings into scripts/outputs/runs/smoke/, validated on 64 samples.
# Stage 00 (NeuroQwen DAPT, opt-in, not in train_pipeline.py) runs first: one step on 256 documents, on the CPU (~1 min);
# without the pubmed_neuroimaging.parquet corpus (not on the Hub) it trains on the pinned publications.parquet instead.
# Inputs: the train/ files of scripts/hf_manifest.json (scripts/data/fetch_hf.py) and the AAL atlas in ~/nilearn_data.
# Usage: bash scripts/smoke_pipeline.sh [extra train_pipeline.py args, e.g. --device cpu --no-validate --skip-data]
# SMOKE_COVERAGE=1 runs both commands under coverage (.coveragerc, subprocesses included, parallel data files: run
# `coverage combine` afterwards), as .github/workflows/smoke.yml does.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
RUN=("$PY")
if [ "${SMOKE_COVERAGE:-0}" = "1" ]; then
  RUN=("$PY" -m coverage run --parallel-mode)
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
"${RUN[@]}" scripts/models/00_neuroqwen_dapt.py --smoke --force --device cpu
exec "${RUN[@]}" scripts/train_pipeline.py --smoke --force "$@"
