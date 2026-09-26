#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# E1-2: center-hypothesis architecture WITHOUT structured depth-error training.
# Reuses E1-1's exact-action cache so architecture/actions/labels/data are fixed.
E1_BASE_ROOT=${E1_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}
E1_2_ROOT=${E1_2_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1_2_no_error_10pct}

export WORK_ROOT=${WORK_ROOT:-$E1_2_ROOT}
export CACHE_ROOT=${CACHE_ROOT:-$E1_BASE_ROOT/action_cache}
export TRAIN_ROOT=${TRAIN_ROOT:-$WORK_ROOT/train}
export VARIANTS=E1
export ERROR_TRAINING=0
export SCORE_SOURCE=model
export GPUS=${GPUS:-0,1,2}
export OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-12}
export SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
export INFER_QUERIES=${INFER_QUERIES:-0}
export TEST_CASES=${TEST_CASES:-nominal,bias:-20,bias:20}
export SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
export PHASES=${PHASES:-train,infer,eval,summary}
export RESUME=${RESUME:-1}

[[ -f "$CACHE_ROOT/protocol.json" ]] || {
  echo "E1-2 requires the shared E1-1 action cache: $CACHE_ROOT/protocol.json" >&2
  exit 2
}

exec bash "$SCRIPT_DIR/run_e1e2_cva.sh"
