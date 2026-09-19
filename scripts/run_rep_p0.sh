#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PHASES=${PHASES:-mine,train,test}

IFS=',' read -r -a PHASE_ARRAY <<< "${PHASES}"
for raw_phase in "${PHASE_ARRAY[@]}"; do
  phase="$(echo "${raw_phase}" | xargs)"
  case "${phase}" in
    mine)
      bash "${ROOT_DIR}/scripts/run_rep_p0_mine.sh"
      ;;
    train)
      bash "${ROOT_DIR}/scripts/run_rep_p0_train.sh"
      ;;
    test)
      bash "${ROOT_DIR}/scripts/run_rep_p0_test.sh"
      ;;
    "")
      ;;
    *)
      echo "Unknown Rep-P0 phase: ${phase}. Use mine,train,test." >&2
      exit 2
      ;;
  esac
done
