#!/usr/bin/env bash
# One-command reproduction of the NeuroVLM revision. The steps live in scripts/pipeline/reproduce.py (its --help lists
# the options); this wrapper only builds the locked environment (CPython 3.12.12 + scripts/config/requirements.lock)
# when .venv is missing, then runs reproduce.py with it. --skip-env also skips the build.
#
#   bash scripts/pipeline/reproduce.sh                    # all five CV folds, ~2.5 h on one RTX 5090 (plus the download)
#   bash scripts/pipeline/reproduce.sh --smoke            # every stage tiny (~5 min), code path only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root
PY=.venv/bin/python
if [ ! -x "$PY" ] && [[ " $* " != *" --skip-env "* ]]; then
  command -v uv >/dev/null || { echo "uv not found; install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
  uv venv .venv --python 3.12.12
  uv pip install --python "$PY" -r scripts/config/requirements.lock -e ".[full,test]"
fi
[ -x "$PY" ] || { echo "$PY missing (drop --skip-env)" >&2; exit 1; }
exec "$PY" scripts/pipeline/reproduce.py "$@"
