#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_e1_4_10pct}
DCR_ROOT=${DCR_ROOT:-$WORK_ROOT/test/dcr}
E1_ROOT=${E1_ROOT:-$WORK_ROOT/test/e1_ref}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK_ROOT/selection_audit}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
CASES=${CASES:-}
TOPKS=${TOPKS:-10,50}

args=(
  "$ROOT_DIR/audit_dcr_selection.py"
  --dcr-root "$DCR_ROOT"
  --e1-root "$E1_ROOT"
  --output-root "$OUTPUT_ROOT"
  --splits "$SPLITS"
  --topks "$TOPKS"
)
[[ -n "$CASES" ]] && args+=(--cases "$CASES")

"$PYTHON_BIN" "${args[@]}"
