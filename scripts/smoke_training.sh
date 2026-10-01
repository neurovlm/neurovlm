#!/usr/bin/env bash
# Verify every training pipeline still works after a refactor.
#
# Runs the integration suite: one real (CPU, tiny synthetic data) epoch of
# MLP autoencoder/contrastive/text-to-brain, CNN autoencoder/contrastive/
# text-to-brain, and brain-to-text generation, including checkpoint
# save/resume/reload round-trips. Deterministic and offline (~15s).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -m pytest tests/integration/training -v "$@"
