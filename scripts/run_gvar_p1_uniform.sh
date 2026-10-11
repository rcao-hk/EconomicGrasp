#!/usr/bin/env bash
# Reuse saved GraspNet e19 AP. Default stage is AP-blind selection, CPU only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${ROOT:-/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007}"
STAGE="${STAGE:-select}"
VARIANT="${VARIANT:-volume_rel}"
PYTHON="${PYTHON:-python}"
CASES="${CASES:-$ROOT/analysis/novel_uniform_20261011}"
SIDECARS="${SIDECARS:-$ROOT/analysis/novel_uniform_depth_sidecars_20261011}"
PROBE="${PROBE:-$ROOT/analysis/novel_uniform_probe_20261011}"
COND="${COND:-$ROOT/analysis/novel_uniform_conditional_20261011}"
COMP="${COMP:-$ROOT/analysis/novel_uniform_vs_selected_20261011}"
SELECTED_COND="${SELECTED_COND:-$ROOT/analysis/p1_conditional_20261011}"
case "$STAGE" in
  select)
    cmd=("$PYTHON" select_gvar_uniform_novel.py --root "$ROOT" --output-dir "$CASES") ;;
  export)
    : "${DATASET_ROOT:?Export requires DATASET_ROOT}"
    cmd=("$PYTHON" export_gvar_probe_sidecars.py --root "$ROOT" --dataset-root "$DATASET_ROOT"
       --selection-json "$CASES/selected_frames.json" --variant "$VARIANT"
       --checkpoint "$ROOT/train/$VARIANT/checkpoint_epoch_019.tar"
       --output-dir "$SIDECARS" --device "${DEVICE:-cuda:0}" --max-frames "${MAX_FRAMES:-0}") ;;
  analyze)
    cmd=("$PYTHON" analyze_gvar_probe_uncertainty.py --root "$ROOT"
       --selection-json "$CASES/selected_frames.json" --sidecar-root "$SIDECARS"
       --output-dir "$PROBE" --variants "$VARIANT" --topk "${TOPK:-20}" --bins "${BINS:-10}") ;;
  conditional)
    cmd=("$PYTHON" analyze_gvar_p1_conditional.py --cases-dir "$CASES"
       --probe-dir "$PROBE" --output-dir "$COND" --variants "$VARIANT"
       --min-probes 15 --min-frames 3) ;;
  compare)
    cmd=("$PYTHON" compare_gvar_p1_cohorts.py --selected-dir "$SELECTED_COND"
       --uniform-dir "$COND" --output-dir "$COMP" --variant "$VARIANT") ;;
  *) echo "STAGE must be select|export|analyze|conditional|compare" >&2; exit 2 ;;
esac
printf '[GVAR-P1 %s] ' "$STAGE"; printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
"${cmd[@]}"
