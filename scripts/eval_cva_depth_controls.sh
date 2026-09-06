#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/depth_controls_eval_common.sh"
EVAL_ARGS=(--num_workers "${EVAL_NUM_WORKERS:-4}")
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then EVAL_ARGS+=(--check_only); fi
if [[ "${FORCE_EVAL:-0}" == 1 ]]; then EVAL_ARGS+=(--force); fi
run_depth_eval_command "$PYTHON" -u eval_cva_depth_controls.py "${DEPTH_EVAL_ARGS[@]}" "${EVAL_ARGS[@]}" "$@"
