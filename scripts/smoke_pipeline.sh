#!/usr/bin/env bash
# Smoke test of the training pipeline: every stage with tiny settings into scripts/outputs/runs/smoke/, validated on 64 samples.
# Stage 00 (NeuroQwen DAPT, opt-in, not in train_pipeline.py) runs first: one step on 256 documents, on the CPU (~1 min).
# Usage: bash scripts/smoke_pipeline.sh [extra train_pipeline.py args, e.g. --device cpu --no-validate]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
"$PY" scripts/models/00_neuroqwen_dapt.py --smoke --force --device cpu
exec "$PY" scripts/train_pipeline.py --smoke --force "$@"
