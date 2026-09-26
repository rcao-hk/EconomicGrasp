#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PHASES=${PHASES:-cache,train,test}

IFS=',' read -r -a STEP_IDS <<< "$PHASES"
for raw_phase in "${STEP_IDS[@]}"; do
  phase="$(echo "$raw_phase" | xargs)"
  case "$phase" in
    cache)
      bash "$ROOT_DIR/scripts/run_rep_p1_cache.sh"
      ;;
    train)
      bash "$ROOT_DIR/scripts/run_rep_p1_train.sh"
      ;;
    test|summary)
      if [[ "$phase" == "test" ]]; then
        bash "$ROOT_DIR/scripts/run_rep_p1_test.sh"
      else
        "${PYTHON_BIN:-python}" "$ROOT_DIR/summarize_rep_p1.py"           --test_root "${TEST_ROOT:-${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p1_action_evidence}/test}"           --output_dir "${TEST_ROOT:-${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p1_action_evidence}/test}"           --variants "${VARIANTS:-geo_pred,img_point,img_region}"           --splits "${SPLITS:-test_similar,test_novel}"
      fi
      ;;
    *)
      echo "Unknown Rep-P1 phase: $phase" >&2
      exit 2
      ;;
  esac
done
