#!/usr/bin/env bash
# Independent next-round worktree; no background execution or automatic training.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Use check|cross-old|trace|plan|fit|audit|infer|eval|report|p0-attribution|p1-support-mine|p1-support-summarize}"
: "${RUN_ROOT:?Set a NEW output directory}"
DEVICE="${DEVICE:-cuda:0}"
FACTORY="${FACTORY:-rare_grasp.runtime:Runtime}"
PLAN="$RUN_ROOT/plan.json"
mkdir -p "$RUN_ROOT"
case "$STAGE" in
check)
  OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_decision_supervision.py' -v
  OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_p0_p1_decision_diagnostics.py' -v
  python -m compileall -q diagnostics/decision_supervision tests/test_decision_supervision.py tests/test_p0_p1_decision_diagnostics.py
  bash -n scripts/decision_supervision/run.sh
  ;;
cross-old)
  : "${SOURCE_INDEX:?Accepted P1 merged index}" "${OLD_HEADS:?Four original checkpoint paths and SHA JSON}"
  python -m diagnostics.decision_supervision.train cross --source-index "$SOURCE_INDEX" \
    --heads "$OLD_HEADS" --output "$RUN_ROOT/cross_old" --device "$DEVICE"
  ;;
trace)
  : "${SOURCE_INDEX:?}" "${ACCEPTANCE:?Previous label acceptance report}" "${BOUND_CONFIG:?}" \
    "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  extra=(); [[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
  python -m diagnostics.decision_supervision.runtime trace --source-index "$SOURCE_INDEX" \
    --accepted-label-report "$ACCEPTANCE" --config "$BOUND_CONFIG" --factory "$FACTORY" \
    --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" --device "$DEVICE" \
    --output "$RUN_ROOT/traces" --shards "${SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}" "${extra[@]}"
  ;;
plan)
  : "${SOURCE_INDEX:?}" "${ACCEPTANCE:?}"
  args=(); for f in "$RUN_ROOT"/traces/trace_*.json; do
    [[ -f "$f" ]] || { echo "Missing trace shards" >&2; exit 2; }
    args+=(--trace-manifest "$f")
  done
  python -m diagnostics.decision_supervision.data --source-index "$SOURCE_INDEX" \
    --accepted-label-report "$ACCEPTANCE" "${args[@]}" --output "$PLAN"
  ;;
fit)
  for seed in ${SEEDS:-20261113 20261114 20261115}; do
    for arm in ${ARMS:-A B C D}; do
      python -m diagnostics.decision_supervision.train fit --plan "$PLAN" \
        --arm "$arm" --seed "$seed" --epochs 10 --batch-frames 4 \
        --output "$RUN_ROOT/heads/$seed/$arm" --device "$DEVICE"
    done
  done
  ;;
audit)
  for seed in ${SEEDS:-20261113 20261114 20261115}; do
    export SEED="$seed"
    python - <<'PY'
import os
from pathlib import Path
from diagnostics.decision_supervision.core import sha,write_json
r=Path(os.environ['RUN_ROOT']);s=os.environ['SEED']
write_json(r/f'heads_{s}.json',{a:{'file':str(r/'heads'/s/a/'last.pt'),'sha256':sha(r/'heads'/s/a/'last.pt')} for a in 'ABCD'})
PY
    python -m diagnostics.decision_supervision.train cross --plan "$PLAN" \
      --heads "$RUN_ROOT/heads_$seed.json" --output "$RUN_ROOT/audit/$seed" --device "$DEVICE"
  done
  ;;
infer|eval)
  : "${DATASET_ROOT:?}"; seed="${SEED:-20261113}"; arm="${ARM:-A}"; interval="${INTERVAL:-10}"
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    dest="$RUN_ROOT/test_i$interval/$seed/$arm/$split"
    if [[ "$STAGE" == infer ]]; then
      : "${BOUND_CONFIG:?}"; extra=(); policies="${POLICIES:-rank_executed,both}"
      if [[ "$arm" == native ]]; then policies=native; else
        weights="$RUN_ROOT/heads/$seed/$arm/last.pt"
        signature="$(python -c 'import sys;from diagnostics.decision_supervision.core import sha;print(sha(sys.argv[1]))' "$weights")"
        extra+=(--weights "$weights" --weights-sha256 "$signature")
      fi
      python -m diagnostics.decision_supervision.runtime infer --config "$BOUND_CONFIG" \
        --factory "$FACTORY" --dataset-root "$DATASET_ROOT" --gntrans-root "${GNTRANS_RGB_ROOT:-}" \
        --device "$DEVICE" --split "$split" --interval "$interval" --policies "$policies" \
        --snapshot-seed 20261113 --output "$dest" "${extra[@]}"
    else
      python -m diagnostics.decision_supervision.runtime eval --inference "$dest" \
        --dataset-root "$DATASET_ROOT" --workers "${EVAL_WORKERS:-16}"
    fi
  done
  ;;
report)
  : "${RUNS_JSON:?seed -> arm -> split -> inference directory mapping}"
  python -m diagnostics.decision_supervision.report --runs "$RUNS_JSON" \
    --policy "${POLICY:-rank_executed}" --output "$RUN_ROOT/reports/${POLICY:-rank_executed}"
  ;;
p0-attribution)
  : "${RUNS_JSON:?Accepted 3-seed A/B/C/D inference roots}" "${DATASET_ROOT:?GraspNet dataset root}"
  split="${P0_SPLIT:-test_novel}"
  left="${P0_LEFT:-A}"; right="${P0_RIGHT:-C}"
  out="${P0_OUTPUT:-$RUN_ROOT/p0_attribution/$split/${left}_vs_${right}}"
  extra=()
  if [[ "${P0_LIMIT:-0}" != 0 ]]; then
    [[ -n "${P0_OUTPUT:-}" ]] || { echo "P0_LIMIT smoke requires an independent P0_OUTPUT" >&2; exit 2; }
    extra+=(--limit "$P0_LIMIT" --allow-partial)
  fi
  [[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
  python -m diagnostics.decision_supervision.action_attribution \
    --runs "$RUNS_JSON" --split "$split" --left "$left" --right "$right" \
    --seeds "${P0_SEEDS:-20261113,20261114,20261115}" --policy rank_executed \
    --dataset-root "$DATASET_ROOT" --output "$out" \
    --max-compiled-pairs "${P0_COMPILED_PAIR_CHECKS:-64}" "${extra[@]}"
  ;;
p1-support-mine)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}" "${BOUND_CONFIG:?Verified MIX50 GVAR bound config}"
  out="${P1_SUPPORT_ROOT:-$RUN_ROOT/p1_support/cache}"
  extra=()
  if [[ "${P1_LIMIT:-0}" != 0 ]]; then
    [[ -n "${P1_SUPPORT_ROOT:-}" ]] || { echo "P1_LIMIT smoke requires an independent P1_SUPPORT_ROOT" >&2; exit 2; }
    extra+=(--limit "$P1_LIMIT")
  fi
  [[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
  python -m diagnostics.decision_supervision.support_audit mine \
    --plan "$PLAN" --config "$BOUND_CONFIG" --factory "$FACTORY" \
    --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
    --device "$DEVICE" --output "$out" --domains "${P1_DOMAINS:-original,gntrans}" \
    --frames-per-scene "${P1_FRAMES_PER_SCENE:-2}" --snapshot-seed 20261113 \
    --shards "${SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}" "${extra[@]}"
  ;;
p1-support-summarize)
  out="${P1_SUPPORT_ROOT:-$RUN_ROOT/p1_support/cache}"
  extra=(); [[ "${P1_PARTIAL:-0}" != 1 ]] || extra+=(--allow-partial)
  python -m diagnostics.decision_supervision.support_audit summarize \
    --root "$out" --output "$RUN_ROOT/p1_support/summary" "${extra[@]}"
  ;;
*) echo "Unknown stage $STAGE" >&2; exit 2 ;;
esac
