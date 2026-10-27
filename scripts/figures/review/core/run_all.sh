#!/usr/bin/env bash
# Rerun the response analyses. The queue lives in scripts/figures/review/core/run_all.py (its header documents the
# arguments and the FIGURES_ONLY, A5_FULL, DEVICE and DAPT_CORPUS variables); this wrapper runs it with .venv.
#
#   cd neurovlm && nohup bash scripts/figures/review/core/run_all.sh > /dev/null 2>&1 &
#   FIGURES_ONLY=1 bash scripts/figures/review/core/run_all.sh    # only re-plot (CPU, ~2 min)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."   # repository root
exec .venv/bin/python scripts/figures/review/core/run_all.py "$@"
