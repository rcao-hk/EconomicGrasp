#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
SELECTOR_CKPT=${SELECTOR_CKPT:-${WORK_ROOT}/train/checkpoint_best.tar}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/test_best_heldout}
GPUS=${GPUS:-0,1}

[[ -f "${SELECTOR_CKPT}" ]] || { echo "Best selector checkpoint not found: ${SELECTOR_CKPT}" >&2; exit 2; }

# Formal held-out protocol: test_seen was consumed as validation, therefore only
# Similar and Novel are evaluated here.  The checkpoint-selected threshold is
# used unless SELECTOR_THRESHOLD is explicitly supplied by the caller.
export WORK_ROOT SELECTOR_CKPT OUTPUT_ROOT GPUS
export SPLITS="test_similar,test_novel"
export SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
export QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-128}
export QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
export OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
export PROFILE_TIMING=${PROFILE_TIMING:-1}
export SAVE_CANDIDATE_ROWS=${SAVE_CANDIDATE_ROWS:-0}

printf '[BEST-HELDOUT] checkpoint=%s\n' "${SELECTOR_CKPT}"
printf '[BEST-HELDOUT] splits=%s gpus=%s output=%s\n' "${SPLITS}" "${GPUS}" "${OUTPUT_ROOT}"

exec bash "${ROOT_DIR}/run_ray_pairwise_selector_test.sh"
