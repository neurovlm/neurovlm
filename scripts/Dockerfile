# NeuroVLM revision: reproduction image (locked environment; entrypoint scripts/reproduce.sh).
#
# Build from the repository root (the build context is the repository; only the files below are copied):
#   docker build -f scripts/Dockerfile -t neurovlm-repro .
# Run (NVIDIA driver >= 580 and the NVIDIA Container Toolkit on the host). Mounts: the Hugging Face cache (models and
# datasets, filled on first use), the inputs without a producer in this repository (scripts/README.md, Reproducibility),
# the package data dir (data stages 01-07 write ~25 GB), the outputs and the figures:
#   docker run --gpus all --shm-size 16g \
#     -v $HOME/.cache/huggingface:/hf -v /path/to/external:/external:ro -v $PWD/repro_data:/work/.cache/neurovlm \
#     -v $PWD/repro_outputs:/opt/neurovlm/scripts/outputs -v $PWD/repro_figures:/opt/neurovlm/scripts/figures/main/repro \
#     neurovlm-repro                  # arguments go to reproduce.sh: --smoke, --full, --offline, ... (--help)
#
# The torch wheel pinned in requirements.lock (2.14.1 from PyPI) is the CUDA 13.0 build and bundles its CUDA libraries,
# so the base image only has to match the driver requirement: CUDA 13.0 (driver >= 580). A CUDA 12.x base image would
# start on older drivers and then fail inside torch, so it is not used here.
ARG BASE_IMAGE=nvidia/cuda:13.0.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    HOME=/work \
    HF_HOME=/hf \
    NEUROVLM_EXTERNAL_DIR=/external \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /work/.cache/neurovlm /hf /external

# uv of the reference environment
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/

WORKDIR /opt/neurovlm

# 1. the locked third-party packages (own layer: rebuilt only when the lock changes)
COPY scripts/requirements.lock scripts/requirements.lock
RUN uv venv .venv --python 3.12.12 \
 && uv pip install --python .venv/bin/python -r scripts/requirements.lock

# 2. the package and the scripts
COPY pyproject.toml README.md LICENSE ./
COPY src src
COPY scripts/*.py scripts/*.sh scripts/reference_metrics.json scripts/README.md scripts/
COPY scripts/data scripts/data
COPY scripts/models scripts/models
COPY scripts/eval_exclusions scripts/eval_exclusions
COPY scripts/figures/main/*.py scripts/figures/main/README.md scripts/figures/main/
RUN uv pip install --python .venv/bin/python --no-deps -e . \
 && diff <(grep -v '^#' scripts/requirements.lock) <(uv pip freeze --python .venv/bin/python | grep -v '^-e ') \
 && .venv/bin/python -c "import neurovlm, torch; print('neurovlm', neurovlm.__file__, 'torch', torch.__version__)"

# reproduce.sh finds the image's .venv, checks it against the lock and runs the pipeline
ENTRYPOINT ["bash", "scripts/reproduce.sh"]
CMD []
