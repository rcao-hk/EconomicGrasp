#!/usr/bin/env bash
# Authorized 117 experiment: three fresh models, identical data/optimization.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/home/robotarm/miniconda3/envs/grasp/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export GC6D_ROOT=/data/robotarm/dataset/GraspClutter6D
export GPUS=3,4,5 BATCH_SIZE=4 NUM_WORKERS=4
export TRAIN_FRACTION=0.1 EVAL_FRACTION=0.1 SAMPLE_INTERVAL=0.1
export MAX_EPOCH=20 LR=0.0001 WEIGHT_DECAY=0 DEPTH_WEIGHT_DECAY=0 SEED=0
export TRAIN_VALIDATE=1 EVAL_START_EPOCH=10 EVAL_NUM_WORKERS=3
export CKPT_SAVE_INTERVAL=20 POSE_DEPTH_MODE=global_film USE_FUSE_DEPTH=1
export GC6D_GRASPNESS_MODE=instance GC6D_WORKSPACE_POSE_MODE=json
export COLLISION_THRESH=0 GNTRANS_COLLISION_SOURCE=none REMOVE_DUMP=0
export RAYROPE_GROUP_CHUNK=32 GVAR_ACTION_CHUNK=256
ROOT=/data/robotarm/result/grasp/rgbgrasp/triple_mix_20261024
mkdir -p "$ROOT"
exec 9>"$ROOT/pipeline.lock"
flock -n 9 || { echo 'Pipeline already running'; exit 2; }
stage=initializing
status() { stage="$1"; printf '%s %s\n' "$(date -Iseconds)" "$stage" | tee "$ROOT/status.txt"; }
trap 'rc=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date -Iseconds)" "$stage" "$rc" > "$ROOT/status.txt"; exit "$rc"' ERR
echo $$ > "$ROOT/pipeline.pid"
git rev-parse HEAD > "$ROOT/source_commit.txt"
env | sort | grep -E '^(GPUS=|BATCH_SIZE=|NUM_WORKERS=|TRAIN_|EVAL_|MAX_EPOCH=|LR=|WEIGHT_DECAY=|DEPTH_WEIGHT_DECAY=|SEED=|CKPT_SAVE_INTERVAL=|POSE_DEPTH_MODE=|USE_FUSE_DEPTH=|GC6D_|COLLISION_|GNTRANS_|DATASET_ROOT=|SAMPLE_INTERVAL=|RAYROPE_|GVAR_)' > "$ROOT/environment.txt"

if [[ ! -f "$ROOT/data_ready.txt" ]]; then
  status waiting_for_full_CDF_on_119
  SSH=(ssh -S /tmp/gc6d-cdf-transfer-%r@%h:%p -o BatchMode=yes -o ConnectTimeout=20 robotarm@10.30.7.119)
  while ! "${SSH[@]}" 'test -f /home/robotarm/gc6d_cdf_prep_20261023/complete.txt'; do sleep 60; done
  status copying_full_CDF
  rsync -a --checksum -e 'ssh -S /tmp/gc6d-cdf-transfer-%r@%h:%p -o BatchMode=yes' \
    robotarm@10.30.7.119:/home/robotarm/gc6d_cdf_prep_20261023/sample/ \
    "$GC6D_ROOT/economic_grasp_label_300views_extend_angle_cdf_depth/"
  status validating_all_232_CDF_caches
  OUTPUT_ROOT="$ROOT/preflight" DRY_RUN=1 CHECK_DATA=1 CHECK_ALL_CACHES=1 bash run_gntrans_gc6d_mix_cva_cdf_train.sh > "$ROOT/preflight.log" 2>&1
  date -Iseconds > "$ROOT/data_ready.txt"
fi

for design in cva_cdf point_rayrope point_rayrope_gvar; do
  export ENABLE_POINT_RAYROPE=0 ENABLE_GVAR=0
  [[ "$design" == cva_cdf ]] || export ENABLE_POINT_RAYROPE=1
  [[ "$design" != point_rayrope_gvar ]] || export ENABLE_GVAR=1
  run="$ROOT/$design"
  mkdir -p "$run/train"
  if [[ ! -f "$run/train_complete.json" ]]; then
    # A failed run is inspected before an explicit resume; never silently overwrite it.
    [[ ! -f "$run/train/checkpoint.tar" ]] || { echo "Existing checkpoint requires inspected resume: $run" >&2; exit 3; }
    python -c 'import shutil; assert shutil.disk_usage("/data").free > 5*2**30, "Less than 5 GiB free"'
    status "training_$design"
    OUTPUT_ROOT="$run/train" CHECK_DATA=0 EVAL_NUM_WORKERS=3 bash run_gntrans_gc6d_mix_cva_cdf_train.sh > "$run/train/console.log" 2>&1
    python - "$run" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]); f=p/'train/checkpoint.tar'
c=torch.load(f,map_location='cpu'); assert c['epoch']==20
protocol=c['gntrans_mix_protocol']; assert protocol['mixed_train_count']==8216
h=hashlib.sha256()
with f.open('rb') as r:
    for chunk in iter(lambda:r.read(8*1024*1024),b''): h.update(chunk)
(p/'train_complete.json').write_text(json.dumps(dict(checkpoint=str(f),sha256=h.hexdigest(),epoch=c['epoch'],design=c['grasp_design']),indent=2))
PY
  fi
  if [[ ! -f "$run/evaluation_complete.json" ]]; then
    status "inference_and_AP_$design"
    CKPT="$run/train/checkpoint.tar" OUTPUT_ROOT="$run/eval_e19" EVAL_NUM_WORKERS=12 \
      bash run_gntrans_mix_cva_cdf_eval.sh > "$run/evaluation_console.log" 2>&1
    python tools/summarize_triple_design_evaluation.py "$run"
  fi
done
status complete
date -Iseconds > "$ROOT/complete.txt"
