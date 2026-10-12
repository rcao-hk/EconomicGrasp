#!/usr/bin/env bash
# P0-B: dynamic RGB + fixed geometry. Fresh train to complete crossed 2x2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATASET_ROOT:?Set DATASET_ROOT to original GraspNet root}"
: "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT to GN-Trans RGB directory}"
ROOT="${ROOT:-/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007}"
export VARIANT=volume_inverse_hybrid
export GPUS="${GPUS:-3,4,5}"
export BATCH_SIZE=3 MAX_EPOCH=21
export NUM_WORKERS="${NUM_WORKERS:-3}" EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export ACTION_CHUNK="${ACTION_CHUNK:-2048}" ACTIVATION_CHECKPOINT=0 SEED=0
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/train/volume_inverse_hybrid}"
[[ -z "${RESUME_CKPT:-}" ]] || { echo 'Fresh experiment forbids RESUME_CKPT' >&2; exit 2; }
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Malformed GPUS' >&2; exit 2; }
[[ "$(awk -F, '{print NF}' <<<"$GPUS")" == 3 ]] || { echo 'Exactly three GPUs required, per GPU batch 3' >&2; exit 2; }
[[ "${MAX_BATCHES:-0}" == 0 || "${DRY_RUN:-0}" == 1 ]] || { echo 'MAX_BATCHES>0 requires separate smoke output directory' >&2; exit 2; }
echo '[GVAR-INVERSE-HYBRID] dynamic RGB/fixed geometry, fresh seed=0, 10% Original+10% GN-Trans'
bash scripts/run_gvar_train.sh "$@"
