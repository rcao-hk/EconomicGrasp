#!/usr/bin/env bash
# Controlled P0: fixed RGB support + dynamic geometry support. Fresh training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATASET_ROOT:?Set DATASET_ROOT to original GraspNet root}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to material-shift RGB directory}"
ROOT="${ROOT:-/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007}"
VARIANT=volume_hybrid
export VARIANT ROOT
export GPUS="${GPUS:-3,4,5}"
export BATCH_SIZE=3 MAX_EPOCH=21
export NUM_WORKERS="${NUM_WORKERS:-3}" EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export ACTION_CHUNK="${ACTION_CHUNK:-2048}"
export ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-0}" SEED=0
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/train/volume_hybrid}"
# Do not silently switch to a pre-existing checkpoint or incompatible recipe.
[[ -z "${RESUME_CKPT:-}" ]] || { echo 'This experiment must start fresh; RESUME_CKPT not allowed' >&2; exit 2; }
[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPUS must be a comma-separated list' >&2; exit 2; }
[[ "$(awk -F, '{print NF}' <<<"${GPUS}")" -eq 3 ]] || { echo 'Use exactly 3 GPUs (global batch=9)' >&2; exit 2; }
echo "[GVAR-HYBRID] fixed-RGB/dynamic-geometry, fresh seed0, 10%+10%, 3 GPUs x batch3"
bash scripts/run_gvar_train.sh "$@"
