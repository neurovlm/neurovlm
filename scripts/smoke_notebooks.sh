#!/usr/bin/env bash
# Export and smoke-execute notebooks after a refactor.
#
# Usage:
#   scripts/smoke_notebooks.sh                 # everything (public docs + experimental + figures)
#   scripts/smoke_notebooks.sh public           # 01_tutorials, 02_data, 03_models only
#   scripts/smoke_notebooks.sh experimental     # docs/experimental only
#   scripts/smoke_notebooks.sh figures          # docs/figures only
#   scripts/smoke_notebooks.sh -k coordinate    # pytest -k filter, any scope
#
# Notebooks are exported to temporary Python scripts with training
# capped to a single epoch over a couple of batches (NEUROVLM_SMOKE=1); this
# proves the code path still runs after a refactor, not that metrics match a
# full run. Requires network access and the cached datasets/models the
# notebooks themselves would download (`neurovlm.data.fetch_data`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "${1:-}" in
  public)
    shift
    exec python -m pytest tests/notebooks -v -k "01_tutorials or 02_data or 03_models" "$@"
    ;;
  experimental)
    shift
    exec python -m pytest tests/notebooks -v -k "experimental" "$@"
    ;;
  figures)
    shift
    exec python -m pytest tests/notebooks -v -k "figures" "$@"
    ;;
  *)
    exec python -m pytest tests/notebooks -v "$@"
    ;;
esac
