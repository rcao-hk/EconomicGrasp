#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# E1-1: full E1 reference.
# Center-hypothesis CVA + structured depth-error training + model CDF ranking.
E1_BASE_ROOT=${E1_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}

export WORK_ROOT=${WORK_ROOT:-$E1_BASE_ROOT}
export CACHE_ROOT=${CACHE_ROOT:-$WORK_ROOT/action_cache}
export TRAIN_ROOT=${TRAIN_ROOT:-$WORK_ROOT/train}
export VARIANTS=E1
export ERROR_TRAINING=1
export SCORE_SOURCE=model
export SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
export INFER_QUERIES=${INFER_QUERIES:-0}
export TEST_CASES=${TEST_CASES:-nominal,bias:-20,bias:20}
export SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
export PHASES=${PHASES:-prepare,train,infer,eval,summary}
export RESUME=${RESUME:-1}

exec bash "$SCRIPT_DIR/run_e1e2_cva.sh"
