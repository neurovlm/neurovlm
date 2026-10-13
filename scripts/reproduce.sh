#!/usr/bin/env bash
# One-command reproduction of the NeuroVLM revision: locked environment -> data inputs from Hugging Face -> training ->
# validation -> paper figures -> comparison with the revision's numbers (scripts/reference_metrics.json). Run from anywhere.
#
#   bash scripts/reproduce.sh                              # all five CV folds, ~3 h on one RTX 5090 (plus the first download)
#   bash scripts/reproduce.sh --cv-folds 0                 # one CV fold, ~2.5 h
#   bash scripts/reproduce.sh --smoke                      # every stage tiny (~5 min), code path only
#   bash scripts/reproduce.sh --home /scratch/nvlm_home    # isolated data dir (keep an existing ~/.cache/neurovlm untouched)
#
# Data: scripts/data/fetch_hf.py downloads every input verbatim from the Hugging Face org neurovlm (and the third-party
# models) at the commits pinned in scripts/hf_manifest.json and checks each sha256 (~23 GB on first use). A file already
# in the data dir that differs from the pinned one stops the run; replace it with fetch_hf.py --overwrite.
#
# Options
#   --external-dir DIR  optional local copies, linked into the data dir when missing there before the fetch (they must
#                       match the manifest's sha256): synth.parquet, text_synth_less.parquet, the neurovault_* tables and
#                       neurovault_images/; and nilearn_data/aal_SPM12/ (stage 11; nilearn's AAL download server
#                       currently fails TLS verification). Default: $NEUROVLM_EXTERNAL_DIR.
#   --home DIR          run with HOME=DIR, so the package data dir (~/.cache/neurovlm) is DIR/.cache/neurovlm; the Hugging Face
#                       cache stays where it was (HF_HOME is pinned before HOME changes).
#   --data MODE         released (default, the only training input): the pinned HF files. scratch: also rerun data stages
#                       01-07 into scripts/outputs/data_regenerated/ for provenance (run_all.py --regenerate); training
#                       still uses the released inputs.
#   --run-name NAME     run directory scripts/outputs/runs/NAME, validation NAME, figures scripts/figures/main/NAME (default: repro)
#   --cv-folds LIST     CV folds that stage 20 trains and validate_models.py evaluates (default 0,1,2,3,4; ~9 min per fold
#                       on one RTX 5090). The comparison takes fold 0 alone (vs fold 0 of runs/final) or all five (vs the
#                       5-fold mean); Fig. 5a needs all five, else it draws the single-split model.
#   --full              same as --cv-folds 0,1,2,3,4 (the default)
#   --smoke             scripts/smoke_pipeline.sh + figure previews on 64 samples, fold 0 only; FAILs of the comparison are
#                       expected
#   --offline           HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 (the HF cache must already hold every model and dataset)
#   --device DEV        cuda (default), cpu
#   --seeds-14 LIST     stage-14 seeds; the lowest validation loss is packaged (default 1,2,3,4,5, the paper model's recipe)
#   --skip-env | --skip-data | --skip-train | --skip-validate | --skip-figures
# Steps are resumable: the fetch skips verified files, model stages skip outputs that exist (delete the run directory to retrain).
# Logs and per-step wall times: scripts/outputs/reproduce/<run>/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$(pwd)

RUN=repro CV_FOLDS=0,1,2,3,4 SMOKE=0 OFFLINE=0 DEVICE=cuda SEEDS14=1,2,3,4,5 NEW_HOME="" DATA_MODE=released
EXTERNAL="${NEUROVLM_EXTERNAL_DIR:-}"
SKIP_ENV=0 SKIP_DATA=0 SKIP_TRAIN=0 SKIP_VALIDATE=0 SKIP_FIGURES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --external-dir) EXTERNAL="$2"; shift 2 ;;
    --home) NEW_HOME="$2"; shift 2 ;;
    --run-name) RUN="$2"; shift 2 ;;
    --data) DATA_MODE="$2"; shift 2 ;;
    --cv-folds) CV_FOLDS="$2"; shift 2 ;;
    --full) CV_FOLDS=0,1,2,3,4; shift ;;
    --smoke) SMOKE=1; shift ;;
    --offline) OFFLINE=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seeds-14) SEEDS14="$2"; shift 2 ;;
    --skip-env) SKIP_ENV=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-validate) SKIP_VALIDATE=1; shift ;;
    --skip-figures) SKIP_FIGURES=1; shift ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$DATA_MODE" in released|scratch) ;; *) echo "--data released|scratch" >&2; exit 2 ;; esac
[[ "$CV_FOLDS" =~ ^[0-4](,[0-4])*$ ]] || { echo "--cv-folds: comma-separated folds 0-4, e.g. 0 or 0,1,2,3,4" >&2; exit 2; }
[ "$SMOKE" = 1 ] && { RUN=smoke; CV_FOLDS=0; }
LOGDIR="$ROOT/scripts/outputs/reproduce/$RUN"
mkdir -p "$LOGDIR"
TIMINGS="$LOGDIR/timings.tsv"
T_START=$(date +%s)
log() { echo "[reproduce $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/reproduce.log"; }
step() {  # step <name> <command...>: run, tee the output to $LOGDIR/<name>.log, record the wall time; returns the exit code
  local name=$1; shift
  log "step $name: $*"
  local t0=$(date +%s) rc=0
  "$@" 2>&1 | tee "$LOGDIR/$name.log" || rc=${PIPESTATUS[0]}
  local dt=$(( $(date +%s) - t0 ))
  printf "%s\t%s\t%s\t%s\n" "$name" "$dt" "$rc" "$(date -Iseconds)" >> "$TIMINGS"
  log "step $name: exit $rc after ${dt}s"
  return $rc
}

# 1. environment: CPython 3.12.12 + the exact package set of scripts/requirements.lock
PY="$ROOT/.venv/bin/python"
if [ "$SKIP_ENV" = 0 ]; then
  command -v uv >/dev/null || { echo "uv not found; install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
  if [ ! -x "$PY" ]; then
    step env_create uv venv "$ROOT/.venv" --python 3.12.12
    step env_install uv pip install --python "$PY" -r scripts/requirements.lock -e ".[full,test]"
  fi
  if diff <(grep -v '^#' scripts/requirements.lock) <(uv pip freeze --python "$PY" | grep -v '^-e ') > "$LOGDIR/freeze.diff"; then
    log "environment matches scripts/requirements.lock"
  else
    log "WARNING: .venv differs from scripts/requirements.lock (see $LOGDIR/freeze.diff); remove .venv to rebuild it"
  fi
fi
[ -x "$PY" ] || { echo "$PY missing (drop --skip-env)" >&2; exit 1; }

# 2. caches and data dir
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"  # deterministic cuBLAS in every step (common.py's default)
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"  # fixed set/dict order (bert_score sorts texts via a set; batch order moves BERTScore F1 by ~1e-7)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -n "$NEW_HOME" ]; then
  mkdir -p "$NEW_HOME/.cache/neurovlm"
  export NILEARN_DATA="${NILEARN_DATA:-$NEW_HOME/nilearn_data}"
  export HOME="$(cd "$NEW_HOME" && pwd)"
fi
[ "$OFFLINE" = 1 ] && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
DATA_DIR=$("$PY" -c "from neurovlm.data import data_dir; print(data_dir)")
NILEARN_DIR="${NILEARN_DATA:-$HOME/nilearn_data}"
log "data dir $DATA_DIR | HF cache $HF_HOME | nilearn data $NILEARN_DIR | run $RUN | data $DATA_MODE | CV folds $CV_FOLDS | device $DEVICE"
{
  echo "date: $(date -Iseconds)"; echo "host: $(uname -srm)"; echo "uv: $(uv --version 2>/dev/null || echo -)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: -"
  "$PY" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
} > "$LOGDIR/environment.txt"
cat "$LOGDIR/environment.txt"

if [ -n "$EXTERNAL" ]; then  # optional local copies; fetch_hf.py below checks them against the manifest
  for f in synth.parquet text_synth_less.parquet neurovault_collections.parquet neurovault_images.parquet \
           neurovault_abstracts.parquet neurovault_abstracts_missing.csv neurovault_images; do
    if [ ! -e "$DATA_DIR/$f" ] && [ -e "$EXTERNAL/$f" ]; then ln -s "$(cd "$EXTERNAL" && pwd)/$f" "$DATA_DIR/$f"; log "linked $f"; fi
  done
  if [ -d "$EXTERNAL/nilearn_data/aal_SPM12" ] && [ ! -d "$NILEARN_DIR/aal_SPM12/aal" ]; then
    mkdir -p "$NILEARN_DIR"; cp -r "$EXTERNAL/nilearn_data/aal_SPM12" "$NILEARN_DIR/"; log "copied nilearn aal_SPM12"
  fi
fi

# 3. data: every input verbatim from Hugging Face at the commits pinned in scripts/hf_manifest.json, sha256 checked
# (neuro_vectors.pt = embedded_text/pubmed_images.pt, latent_specter2_adhoc.pt, synth.parquet, the NeuroVault scrape, ...).
# A file in the data dir that differs from the pinned one stops the run: fetch_hf.py --overwrite replaces it.
# --data scratch also reruns data stages 01-07 into scripts/outputs/data_regenerated/ (provenance; not used for training).
if [ "$SKIP_DATA" = 0 ]; then
  step data "$PY" scripts/data/fetch_hf.py --data-dir "$DATA_DIR" \
    || { echo "fetch_hf.py failed: pinned inputs missing or different (see $LOGDIR/data.log)" >&2; exit 1; }
  if [ "$DATA_MODE" = scratch ]; then
    REGEN_ARGS=(--regenerate --no-fetch); [ "$SMOKE" = 1 ] && REGEN_ARGS+=(--smoke)
    step data_regenerate "$PY" scripts/data/run_all.py "${REGEN_ARGS[@]}" || log "WARNING: data stage regeneration failed (provenance only)"
  fi
fi
# from here on the Hugging Face libraries read only the HF cache, whose refs fetch_hf.py pinned to the manifest commits
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
[ -d "$NILEARN_DIR/aal_SPM12/aal" ] || log "WARNING: $NILEARN_DIR/aal_SPM12 missing; stage 11 will try nilearn's download (www.gin.cnrs.fr, TLS errors seen 2026-10)"

prefetch_atlases() {  # Fig. 4's parallel spin-test workers race on the first neuromaps download into an empty HOME: fetch once here
  "$PY" -c "from neuromaps.datasets import fetch_fsaverage, fetch_regfusion; fetch_fsaverage(density='3k'); fetch_regfusion('fsaverage')" \
    || log "WARNING: neuromaps atlas download failed; the Fig. 4 spin test needs network access or a populated ~/neuromaps-data"
}

# 3-6. smoke: every stage tiny, figure previews (no review step: Fig. 5b's a3 input is not computed for smoke models)
if [ "$SMOKE" = 1 ]; then
  step smoke_pipeline bash scripts/smoke_pipeline.sh --device "$DEVICE" || log "smoke pipeline exit != 0 (validation FAILs on 64 samples are expected; check for tracebacks)"
  step smoke_validate "$PY" scripts/validate_models.py --run scripts/outputs/runs/smoke --name smoke_all --max-samples 64 --folds "$CV_FOLDS" \
    --device "$DEVICE" --generation-batch 64 || true
  [ "$SKIP_FIGURES" = 1 ] || prefetch_atlases
  [ "$SKIP_FIGURES" = 1 ] || step smoke_figures "$PY" scripts/figures/main/make_all.py --model-dir scripts/outputs/runs/smoke \
    --max-samples 64 --continue-on-error --device "$DEVICE" || true
  "$PY" scripts/verify_reproduction.py --name smoke_all --figures-dir scripts/figures/main/smoke/preview_n64 --since "$T_START" \
    | tee "$LOGDIR/verify.log" || true
  log "smoke done in $(( $(date +%s) - T_START ))s (metrics of 1-epoch models FAIL by design; tracebacks would mean a broken path)"
  exit 0
fi

# 4. training: 08 09 11 12 13 14 (stage 14: one run per seed, best validation loss packaged), then the CV stage 20 on
# --cv-folds (folds already in the run directory are kept)
if [ "$SKIP_TRAIN" = 0 ]; then
  step train_models "$PY" scripts/train_pipeline.py --run-name "$RUN" --stages models --seeds-14 "$SEEDS14" --skip-data --no-validate --device "$DEVICE"
  step train_cv "$PY" scripts/train_pipeline.py --run-name "$RUN" --only 20 --stage-args "--folds $CV_FOLDS" --skip-data --no-validate --device "$DEVICE"
fi

# 5. validation (PubMed test with the leakage-audit exclusions) against the released numbers of reference_metrics.json
REF="scripts/outputs/validation/released_reference/metrics.json"
if [ "$SKIP_VALIDATE" = 0 ]; then
  [ -e "$REF" ] || "$PY" scripts/verify_reproduction.py --write-released-reference "$REF"
  step validate "$PY" scripts/validate_models.py --run "scripts/outputs/runs/$RUN" --name "$RUN" --reference "$REF" \
    --folds "$CV_FOLDS" --generation-batch 128 --device "$DEVICE" || log "validate_models.py reports a FAIL vs the released checkpoints"
fi

# 6. paper figures from this run. Fig. 5b reads the reviewer analysis a3_neurovault_bounds, which fig5_perf_quant.py
# requires to be computed with this run's autoencoder and InfoNCE heads (~30 s). core/run_all.sh a3_neurovault_bounds
# runs it for runs/final only and into the committed outputs, so it runs here with run_all.sh's flags into $LOGDIR.
# Without the NeuroVault inputs, or if it fails, Fig. 5 is drawn as a, c, d.
FIG_DIR="scripts/figures/main/$RUN"
A3_OUT="$LOGDIR/a3_neurovault_bounds"
if [ "$SKIP_FIGURES" = 0 ]; then
  prefetch_atlases
  A3_OK=0
  if [ -e "$DATA_DIR/neurovault_images.parquet" ] && [ -d "$DATA_DIR/neurovault_images" ]; then
    step a3_neurovault_bounds "$PY" scripts/figures/review/core/a3_neurovault_bounds/run_analysis.py \
      --model-dir "scripts/outputs/runs/$RUN" --out "$A3_OUT" --device "$DEVICE" --force --n-jobs 12 --qc-count-all && A3_OK=1 || true
  else
    log "WARNING: NeuroVault inputs missing in $DATA_DIR: no a3 bounds, Fig. 5 without panel b"
  fi
  FIGS_NO5=$("$PY" -c "import sys; sys.path.insert(0, 'scripts/figures/main'); from make_all import FIGURES; print(','.join(k for k, _, _ in FIGURES if k != 'fig5'))")
  step figures "$PY" scripts/figures/main/make_all.py --model-dir "scripts/outputs/runs/$RUN" --only "$FIGS_NO5" --continue-on-error --device "$DEVICE" || true
  F5_OK=0
  if [ "$A3_OK" = 1 ]; then
    step figure5 "$PY" scripts/figures/main/fig5_perf_quant.py --model-dir "scripts/outputs/runs/$RUN" --bounds-dir "$A3_OUT" --device "$DEVICE" && F5_OK=1 || true
  fi
  if [ "$F5_OK" = 0 ]; then
    step figure5_acd "$PY" scripts/figures/main/fig5_perf_quant.py --model-dir "scripts/outputs/runs/$RUN" --panels acd --device "$DEVICE" || true
  fi
fi

# 7. comparison with runs/final
rc=0
"$PY" scripts/verify_reproduction.py --name "$RUN" --figures-dir "$FIG_DIR" --since "$T_START" | tee "$LOGDIR/verify.log" || rc=${PIPESTATUS[0]}
log "done in $(( $(date +%s) - T_START ))s; step times in $TIMINGS"
exit $rc
