#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[augsplat-pipeline] %s\n' "$*" >&2
}

die() {
  printf '[augsplat-pipeline] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scene_pipeline.sh --data-dir /path/to/scene_sparse [options]

Required:
  --data-dir PATH                 Dataset directory. Must contain images/ at minimum.

Core options:
  --artifact-root PATH           Stable output root for NeRF / gsplat artifacts.
                                 Default: sibling directory named <scene>_artifacts
  --vggt-env NAME                Conda env for VGGT preprocessing. Default: vggt
  --nerfstudio-env NAME          Conda env for Nerfstudio. Default: nerfstudio
  --gsplat-env NAME              Conda env for gsplat. Default: gsplat
  --cluster-gsplat-env           Before gsplat training, run the ETH cluster module/CUDA setup.

NeRF options:
  --num-ensembles N              Number of NeRF ensemble models. Default: 5
  --nerf-method NAME             nerfacto | depth-nerfacto. Default: nerfacto
  --nerf-max-steps N             Nerfstudio max iterations. Default: 15000
  --nerf-save-every N            Save interval. Default: --nerf-max-steps

Augmentation options:
  --checkpoint-step N            Force a specific checkpoint step for all ensemble models.
  --camera-id N                  COLMAP camera model id for synthetic views. Default: 1
  --num-final-samples N          Number of synthetic views to keep. Default: 200
  --tau FLOAT                    Weight-map temperature. Default: 1.5
  --final-render-scale FLOAT     Final synthetic render scale. Default: 0.125
  --tmp-root PATH                Temporary render root. Default: \$TMPDIR or /tmp

gsplat options:
  --splat-mode MODE              staged | dual. Default: staged
  --splat-max-steps N            Main gsplat training steps. Default: 10000
  --staged-nerf-phase-steps N    Staged NeRF-only phase steps. Default: 300
  --staged-real-phase-steps N    Staged real-image phase steps. Default: --splat-max-steps
  --nerf-samples-data-factor N   Default: 8
  --batch-size N                 Default: 1
  --nerf-batch-factor N          Default: 20
  --dual-nerf-loss-weight FLOAT  Default: 2
  --dual-nerf-decay-steps-to-quarter N  Default: 300
  --dual-nerf-disable-threshold FLOAT   Default: 0.1
  --strategy-reset-every N       Default: 100000000
  --no-save-last-ckpt            Do not save last_ckpt_rank0.pt

Cleanup:
  --cleanup-intermediate         Remove derived datasets and NeRF artifacts after gsplat finishes.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR=""
ARTIFACT_ROOT=""
VGGT_ENV="vggt"
NERFSTUDIO_ENV="nerfstudio"
GSPLAT_ENV="gsplat"
CLUSTER_GSPLAT_ENV=0

NUM_ENSEMBLES=5
NERF_METHOD="nerfacto"
NERF_MAX_STEPS=15000
NERF_SAVE_EVERY=""

CHECKPOINT_STEP=""
CAMERA_ID="1"
NUM_FINAL_SAMPLES=200
TAU="1.5"
FINAL_RENDER_SCALE="0.125"
TMP_ROOT="${TMPDIR:-/tmp}"

SPLAT_MODE="staged"
SPLAT_MAX_STEPS=10000
STAGED_NERF_PHASE_STEPS=300
STAGED_REAL_PHASE_STEPS=""
NERF_SAMPLES_DATA_FACTOR=8
BATCH_SIZE=1
NERF_BATCH_FACTOR=20
DUAL_NERF_LOSS_WEIGHT=""
DUAL_NERF_DECAY_STEPS_TO_QUARTER=300
DUAL_NERF_DISABLE_THRESHOLD="0.1"
STRATEGY_RESET_EVERY=100000000
SAVE_LAST_CKPT=1

CLEANUP_INTERMEDIATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --vggt-env) VGGT_ENV="$2"; shift 2 ;;
    --nerfstudio-env) NERFSTUDIO_ENV="$2"; shift 2 ;;
    --gsplat-env) GSPLAT_ENV="$2"; shift 2 ;;
    --cluster-gsplat-env) CLUSTER_GSPLAT_ENV=1; shift ;;
    --num-ensembles) NUM_ENSEMBLES="$2"; shift 2 ;;
    --nerf-method) NERF_METHOD="$2"; shift 2 ;;
    --nerf-max-steps) NERF_MAX_STEPS="$2"; shift 2 ;;
    --nerf-save-every) NERF_SAVE_EVERY="$2"; shift 2 ;;
    --checkpoint-step) CHECKPOINT_STEP="$2"; shift 2 ;;
    --camera-id) CAMERA_ID="$2"; shift 2 ;;
    --num-final-samples) NUM_FINAL_SAMPLES="$2"; shift 2 ;;
    --tau) TAU="$2"; shift 2 ;;
    --final-render-scale) FINAL_RENDER_SCALE="$2"; shift 2 ;;
    --tmp-root) TMP_ROOT="$2"; shift 2 ;;
    --splat-mode) SPLAT_MODE="$2"; shift 2 ;;
    --splat-max-steps) SPLAT_MAX_STEPS="$2"; shift 2 ;;
    --staged-nerf-phase-steps) STAGED_NERF_PHASE_STEPS="$2"; shift 2 ;;
    --staged-real-phase-steps) STAGED_REAL_PHASE_STEPS="$2"; shift 2 ;;
    --nerf-samples-data-factor) NERF_SAMPLES_DATA_FACTOR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --nerf-batch-factor) NERF_BATCH_FACTOR="$2"; shift 2 ;;
    --dual-nerf-loss-weight) DUAL_NERF_LOSS_WEIGHT="$2"; shift 2 ;;
    --dual-nerf-decay-steps-to-quarter) DUAL_NERF_DECAY_STEPS_TO_QUARTER="$2"; shift 2 ;;
    --dual-nerf-disable-threshold) DUAL_NERF_DISABLE_THRESHOLD="$2"; shift 2 ;;
    --strategy-reset-every) STRATEGY_RESET_EVERY="$2"; shift 2 ;;
    --no-save-last-ckpt) SAVE_LAST_CKPT=0; shift ;;
    --cleanup-intermediate) CLEANUP_INTERMEDIATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$DATA_DIR" ]] || { usage; die "--data-dir is required"; }
[[ "$NERF_METHOD" == "nerfacto" || "$NERF_METHOD" == "depth-nerfacto" ]] || die "--nerf-method must be nerfacto or depth-nerfacto"
[[ "$SPLAT_MODE" == "staged" || "$SPLAT_MODE" == "dual" ]] || die "--splat-mode must be staged or dual"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
[[ -d "$DATA_DIR/images" ]] || die "Expected images/ under $DATA_DIR"

SCENE_BASE="$(basename "$DATA_DIR")"
if [[ "$SCENE_BASE" == *_vggt ]]; then
  SCENE_TAG="${SCENE_BASE%_vggt}"
elif [[ "$SCENE_BASE" == *_aug ]]; then
  die "Please start from the raw or _vggt dataset, not from an _aug dataset."
else
  SCENE_TAG="$SCENE_BASE"
fi
if [[ "$SCENE_TAG" == *_sparse ]]; then
  SCENE_NAME="${SCENE_TAG%_sparse}"
else
  SCENE_NAME="$SCENE_TAG"
fi
SCENE_PARENT="$(dirname "$DATA_DIR")"

if [[ -z "$ARTIFACT_ROOT" ]]; then
  ARTIFACT_ROOT="${SCENE_PARENT}/${SCENE_TAG}_artifacts"
fi
ARTIFACT_ROOT="$(mkdir -p "$ARTIFACT_ROOT" && cd "$ARTIFACT_ROOT" && pwd)"

if [[ -z "$DUAL_NERF_LOSS_WEIGHT" ]]; then
  DUAL_NERF_LOSS_WEIGHT="2"
fi
if [[ -z "$STAGED_REAL_PHASE_STEPS" ]]; then
  STAGED_REAL_PHASE_STEPS="$SPLAT_MAX_STEPS"
fi
if [[ -z "$NERF_SAVE_EVERY" ]]; then
  NERF_SAVE_EVERY="$NERF_MAX_STEPS"
fi

VGGT_DIR="${SCENE_PARENT}/${SCENE_TAG}_vggt"
AUG_DIR="${SCENE_PARENT}/${SCENE_TAG}_aug"
NERF_ROOT="${ARTIFACT_ROOT}/nerf_models"
DEBUG_PLOT_DIR="${ARTIFACT_ROOT}/image_supervision"
SPLIT_PAYLOAD="${ARTIFACT_ROOT}/ray_sample.pt"
GSPLAT_RESULT_DIR="${ARTIFACT_ROOT}/gsplat_${SPLAT_MODE}"

created_vggt=0
created_aug=0

source "$(conda info --base)/etc/profile.d/conda.sh"

setup_cluster_gsplat_env() {
  if [[ "$CLUSTER_GSPLAT_ENV" -ne 1 ]]; then
    return
  fi

  command -v module >/dev/null 2>&1 || die "--cluster-gsplat-env was requested, but 'module' is not available in this shell"
  command -v nvcc >/dev/null 2>&1 || true

  log "Applying cluster gsplat environment setup"
  module purge
  module load stack/2024-06 gcc/12.2.0
  module load cuda/12.1.1
  module load eth_proxy

  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  export CC=gcc
  export CXX=g++
  export CUDAHOSTCXX=g++
  export CUDA_HOME
  CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
}

colmap_model_exists() {
  local base="$1"
  if [[ -f "${base}/sparse/cameras.bin" && -f "${base}/sparse/images.bin" && -f "${base}/sparse/points3D.bin" ]]; then
    return 0
  fi
  if [[ -f "${base}/sparse/0/cameras.bin" && -f "${base}/sparse/0/images.bin" && -f "${base}/sparse/0/points3D.bin" ]]; then
    return 0
  fi
  return 1
}

prepared_dataset_exists() {
  local base="$1"
  [[ -d "${base}/images" && -d "${base}/images_2" && -d "${base}/images_4" && -d "${base}/images_8" ]] || return 1
  colmap_model_exists "$base" || return 1
  if [[ "$NERF_METHOD" == "depth-nerfacto" ]]; then
    [[ -d "${base}/depths_vggt" ]] || return 1
  fi
  return 0
}

find_run_config() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
from pathlib import Path
import sys
run_dir = Path(sys.argv[1])
matches = sorted(run_dir.glob("outputs/*/*/*/config.y*ml"))
if not matches:
    raise SystemExit(1)
print(matches[0])
PY
}

all_ensemble_configs_present() {
  local root="$1"
  local i
  for ((i=1; i<=NUM_ENSEMBLES; i++)); do
    local run_dir="${root}/nerf_ensemble_${i}"
    [[ -d "$run_dir" ]] || return 1
    find_run_config "$run_dir" >/dev/null 2>&1 || return 1
  done
  return 0
}

run_vggt_if_needed() {
  if prepared_dataset_exists "$DATA_DIR"; then
    log "Reusing existing prepared dataset: $DATA_DIR"
    VGGT_DIR="$DATA_DIR"
    return
  fi

  [[ ! -e "$VGGT_DIR" ]] || die "Destination already exists: $VGGT_DIR"
  created_vggt=1

  log "Preparing VGGT dataset: $VGGT_DIR"
  conda activate "$VGGT_ENV"
  pushd "$REPO_ROOT" >/dev/null
  python scripts/prepare_vggt_dataset.py \
    --src "$DATA_DIR" \
    --dst "$VGGT_DIR" \
    --overwrite \
    --conf-thres-value 0.0
  popd >/dev/null
  conda deactivate
}

train_nerf_ensemble() {
  mkdir -p "$NERF_ROOT"
  if all_ensemble_configs_present "$NERF_ROOT"; then
    log "Reusing existing NeRF ensemble under $NERF_ROOT"
    return
  fi

  local colmap_path="${VGGT_DIR}/sparse"
  local images_path="${VGGT_DIR}/images"
  local depths_path="${VGGT_DIR}/depths_vggt"

  if [[ "$NERF_METHOD" == "depth-nerfacto" && ! -d "$depths_path" ]]; then
    die "depth-nerfacto requested, but depths_vggt is missing under $VGGT_DIR"
  fi

  log "Training ${NUM_ENSEMBLES} ${NERF_METHOD} models under $NERF_ROOT"
  conda activate "$NERFSTUDIO_ENV"

  local i run_dir
  for ((i=1; i<=NUM_ENSEMBLES; i++)); do
    run_dir="${NERF_ROOT}/nerf_ensemble_${i}"
    mkdir -p "$run_dir"
    if find_run_config "$run_dir" >/dev/null 2>&1; then
      log "Skipping existing NeRF run: nerf_ensemble_${i}"
      continue
    fi

    pushd "$run_dir" >/dev/null
    if [[ "$NERF_METHOD" == "depth-nerfacto" ]]; then
      ns-train depth-nerfacto \
        --vis tensorboard \
        --experiment-name "ensemble_${i}" \
        --timestamp "${SCENE_NAME}" \
        --steps-per-save "${NERF_SAVE_EVERY}" \
        --steps-per-eval-image 0 \
        --steps-per-eval-all-images 0 \
        --max-num-iterations "${NERF_MAX_STEPS}" \
        --save-only-latest-checkpoint True \
        --logging.steps-per-log 100 \
        --logging.profiler pytorch \
        colmap \
        --downscale-factor 1 \
        --colmap-path "${colmap_path}" \
        --images-path "${images_path}" \
        --depths-path "${depths_path}"
    else
      ns-train nerfacto \
        --vis tensorboard \
        --experiment-name "ensemble_${i}" \
        --timestamp "${SCENE_NAME}" \
        --steps-per-save "${NERF_SAVE_EVERY}" \
        --steps-per-eval-image 0 \
        --steps-per-eval-all-images 0 \
        --max-num-iterations "${NERF_MAX_STEPS}" \
        --save-only-latest-checkpoint True \
        --logging.steps-per-log 100 \
        --logging.profiler pytorch \
        colmap \
        --downscale-factor 1 \
        --colmap-path "${colmap_path}" \
        --images-path "${images_path}"
    fi
    popd >/dev/null
  done

  conda deactivate
}

augment_dataset() {
  if [[ -d "$AUG_DIR" ]] && compgen -G "${AUG_DIR}/images/nerf_sample_*" > /dev/null; then
    log "Reusing existing augmented dataset: $AUG_DIR"
    return
  fi

  [[ ! -e "$AUG_DIR" ]] || die "Augmented dataset path already exists but does not look reusable: $AUG_DIR"

  conda activate "$NERFSTUDIO_ENV"
  pushd "$REPO_ROOT" >/dev/null

  local cmd=(
    python scripts/augment_dataset.py
    --model-roots "$NERF_ROOT"
    --num-ensembles "$NUM_ENSEMBLES"
    --checkpoint-selection latest
    --input-dataset "$VGGT_DIR"
    --output-dataset "$AUG_DIR"
    --tmp-root "$TMP_ROOT"
    --tau "$TAU"
    --debug-plot-dir "$DEBUG_PLOT_DIR"
    --num-final-samples "$NUM_FINAL_SAMPLES"
    --final-render-scale "$FINAL_RENDER_SCALE"
  )

  if [[ -n "$CHECKPOINT_STEP" ]]; then
    cmd+=(--checkpoint-step "$CHECKPOINT_STEP")
  fi

  if [[ -n "$CAMERA_ID" ]]; then
    cmd+=(--camera-id "$CAMERA_ID")
  fi

  "${cmd[@]}"

  popd >/dev/null
  conda deactivate
  created_aug=1
}

export_split_payload() {
  if [[ -f "$SPLIT_PAYLOAD" ]]; then
    log "Reusing existing split payload: $SPLIT_PAYLOAD"
    return
  fi

  local config_path
  config_path="$(find_run_config "${NERF_ROOT}/nerf_ensemble_1")" || die "Could not find config for nerf_ensemble_1"

  conda activate "$NERFSTUDIO_ENV"
  pushd "$REPO_ROOT" >/dev/null
  python scripts/export_nerf_rays.py \
    --nerf-folder "$config_path" \
    --output-name "$SPLIT_PAYLOAD" \
    --split-only
  popd >/dev/null
  conda deactivate
}

train_gsplat() {
  [[ ! -e "$GSPLAT_RESULT_DIR" ]] || die "gsplat result directory already exists: $GSPLAT_RESULT_DIR"
  mkdir -p "$GSPLAT_RESULT_DIR"

  setup_cluster_gsplat_env
  conda activate "$GSPLAT_ENV"
  pushd "$REPO_ROOT" >/dev/null

  local cmd=(
    python scripts/run_splatting.py default
    --no-nerf_init
    --data_dir "$AUG_DIR"
    --result_dir "$GSPLAT_RESULT_DIR"
    --pt_path "$SPLIT_PAYLOAD"
    --data_factor 1
    --nerf_samples_data_factor "$NERF_SAMPLES_DATA_FACTOR"
    --batch_size "$BATCH_SIZE"
    --nerf_batch_factor "$NERF_BATCH_FACTOR"
    --dual_nerf_loss_weight "$DUAL_NERF_LOSS_WEIGHT"
    --dual_nerf_decay_steps_to_quarter "$DUAL_NERF_DECAY_STEPS_TO_QUARTER"
    --dual_nerf_disable_threshold "$DUAL_NERF_DISABLE_THRESHOLD"
    --max_steps "$SPLAT_MAX_STEPS"
    --strategy.reset_every "$STRATEGY_RESET_EVERY"
    --deterministic
    --disable_viewer
  )

  if [[ "$SPLAT_MODE" == "staged" ]]; then
    cmd+=(
      --staged-runner
      --staged_nerf_phase_steps "$STAGED_NERF_PHASE_STEPS"
      --staged_real_phase_steps "$STAGED_REAL_PHASE_STEPS"
    )
  else
    cmd+=(--dual_runner)
  fi
  if [[ "$SAVE_LAST_CKPT" -eq 1 ]]; then
    cmd+=(--save-last-ckpt)
  fi

  "${cmd[@]}"

  popd >/dev/null
  conda deactivate
}

cleanup_intermediate() {
  [[ "$CLEANUP_INTERMEDIATE" -eq 1 ]] || return

  log "Cleaning intermediate artifacts"
  rm -rf "$NERF_ROOT" "$DEBUG_PLOT_DIR"
  rm -f "$SPLIT_PAYLOAD"
  if [[ "$created_aug" -eq 1 && -d "$AUG_DIR" ]]; then
    rm -rf "$AUG_DIR"
  fi
  if [[ "$created_vggt" -eq 1 && -d "$VGGT_DIR" ]]; then
    rm -rf "$VGGT_DIR"
  fi
}

log "Repo root: $REPO_ROOT"
log "Data dir: $DATA_DIR"
log "Artifact root: $ARTIFACT_ROOT"

run_vggt_if_needed
train_nerf_ensemble
augment_dataset
export_split_payload
train_gsplat
cleanup_intermediate

log "Done"
log "Prepared dataset: $VGGT_DIR"
log "Augmented dataset: $AUG_DIR"
log "NeRF artifacts: $NERF_ROOT"
log "Split payload: $SPLIT_PAYLOAD"
log "gsplat result: $GSPLAT_RESULT_DIR"
