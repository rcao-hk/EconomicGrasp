#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# E1-4: generalization beyond the exact +/-20-mm training/evaluation points.
# - +/-15 mm: off-grid global bias
# - +/-25 mm: outside the +/-20-mm bias-training support
# - +/-3%: multiplicative metric-scale error
# - smooth 5/10 mm RMS: spatially varying error
E1_BASE_ROOT=${E1_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}
E1_4_ROOT=${E1_4_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1_4_corruption_suite_10pct}
E1_4_CASES=${E1_4_CASES:-nominal,bias:-15,bias:15,bias:-25,bias:25,scale:-0.03,scale:0.03,smooth:5,smooth:10}

export WORK_ROOT=${WORK_ROOT:-$E1_4_ROOT}
export TRAIN_ROOT=${TRAIN_ROOT:-$E1_BASE_ROOT/train}
export VARIANTS=E1
export SCORE_SOURCE=model
export SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
export INFER_QUERIES=${INFER_QUERIES:-0}
export TEST_CASES=${TEST_CASES:-$E1_4_CASES}
export SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
export PHASES=${PHASES:-infer,eval,summary}
export RESUME=${RESUME:-1}

[[ -f "$TRAIN_ROOT/E1/checkpoint_best.pt" ]] || {
  echo "E1-4 requires E1-1 checkpoint: $TRAIN_ROOT/E1/checkpoint_best.pt" >&2
  exit 2
}

exec bash "$SCRIPT_DIR/run_e1e2_cva.sh"
