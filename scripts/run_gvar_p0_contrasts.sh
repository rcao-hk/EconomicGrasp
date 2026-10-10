#!/usr/bin/env bash
# P0: reuse existing official e19 AP; NO training / inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${ROOT:-/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007}"
OUT="${OUT:-$ROOT/analysis/p0_fixed_vs_rel_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python}"
BOOTSTRAP="${BOOTSTRAP:-50000}"
SEED="${SEED:-20261009}"
# Require all already-complete variants. A missing one is an error.
VARIANTS=(baseline slot volume_fixed volume volume_rel)
if [[ "${INCLUDE_FIXED_REL:-0}" == "1" ]]; then VARIANTS+=(volume_fixed_rel); fi
cmd=("$PYTHON" analyze_gvar_scene_paired.py --root "$ROOT" --output-dir "$OUT" --variants "${VARIANTS[@]}"
 --baseline baseline --epoch 19 --bootstrap "$BOOTSTRAP" --seed "$SEED" --ci 0.95
 --contrast volume_fixed:volume_rel)
if [[ "${INCLUDE_FIXED_REL:-0}" == "1" ]]; then
  cmd+=(--contrast volume_fixed_rel:volume_rel --contrast volume_fixed:volume_fixed_rel)
fi
printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
"${cmd[@]}"
