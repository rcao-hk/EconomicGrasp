#!/usr/bin/env bash
# Shared inference/AP selection; evaluation reads the remaining protocol from manifests.
set -euo pipefail
DEPTH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$DEPTH_REPO_ROOT"
: "${DATASET_ROOT:?Set DATASET_ROOT to your GraspNet directory}"
: "${PREDICTION_ROOT:?Set PREDICTION_ROOT to a separate inference/AP output directory}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$DEPTH_REPO_ROOT:$DEPTH_REPO_ROOT/libs/graspnetAPI:${PYTHONPATH:-}"
DEPTH_EVAL_ARGS=(
  --dataset_root "$DATASET_ROOT"
  --prediction_root "$PREDICTION_ROOT"
  --variants "${VARIANTS:-base,none,foreground,anchor}"
  --splits "${SPLITS:-test_seen,test_similar,test_novel}"
)
run_depth_eval_command() {
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$@"
  fi
}
