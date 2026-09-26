#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# E1-3: action-vs-score decomposition.
# E1-1 still selects the physical center, but the emitted query ranking score
# is restored to the immutable Stage-1 native score.
E1_BASE_ROOT=${E1_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}
E1_3_ROOT=${E1_3_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1_3_action_only_10pct}

export WORK_ROOT=${WORK_ROOT:-$E1_3_ROOT}
export TRAIN_ROOT=${TRAIN_ROOT:-$E1_BASE_ROOT/train}
export VARIANTS=E1
export SCORE_SOURCE=stage1
export GPUS=${GPUS:-3,5,6}
export OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-12}
export SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
export INFER_QUERIES=${INFER_QUERIES:-0}
export TEST_CASES=${TEST_CASES:-nominal,bias:-20,bias:20}
export SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
export PHASES=${PHASES:-infer,eval,summary}
export RESUME=${RESUME:-1}

[[ -f "$TRAIN_ROOT/E1/checkpoint_best.pt" ]] || {
  echo "E1-3 requires E1-1 checkpoint: $TRAIN_ROOT/E1/checkpoint_best.pt" >&2
  exit 2
}

exec bash "$SCRIPT_DIR/run_e1e2_cva.sh"
