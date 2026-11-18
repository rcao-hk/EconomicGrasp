#!/usr/bin/env bash
# Full 90-scene / 23040-frame ORIGINAL GraspNet inference; eval optional.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET_ROOT:?Set original GraspNet root}"
: "${CKPT:?Set the checkpoint trained by height mix50}"
: "${OUTPUT_ROOT:?Set a NEW directory for 100% GraspNet results}"
[[ -f "$CKPT" ]] || { echo "Checkpoint is missing: $CKPT" >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT/original" ]] || { echo "Refusing to overwrite existing inference predictions" >&2; exit 2; }
export POSE_DEPTH_MODE=height_analytic
export ENABLE_POINT_RAYROPE=1 ENABLE_GVAR=1
export DATASETS=original
export SAMPLE_INTERVAL=1.0
export SPLITS=test_seen,test_similar,test_novel
export ENABLE_INFERENCE=1
export ENABLE_EVAL="${ENABLE_EVAL:-0}"
export GPUS="${GPUS:-0,1,2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export USE_TOP4_VIEW_INFER=0
export COLLISION_THRESH="${COLLISION_THRESH:-0}"
export HEIGHT_MIN_M="${HEIGHT_MIN_M:--0.05}"
export HEIGHT_MAX_M="${HEIGHT_MAX_M:-0.35}"
export HEIGHT_MIN_DOWNWARD_DOT="${HEIGHT_MIN_DOWNWARD_DOT:-0.1}"
python - "$CKPT" "$HEIGHT_MIN_M" "$HEIGHT_MAX_M" "$HEIGHT_MIN_DOWNWARD_DOT" <<'PY'
import sys,torch
ck=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
p=ck.get("height_analytic_protocol")
assert isinstance(p,dict) and p.get("mode")=="height_analytic" and p.get("no_pose_ray_plane_film") is True, "Wrong depth model"
for k,v in zip(("height_min_m","height_max_m","height_min_downward_dot"),map(float,sys.argv[2:])):
    assert abs(float(p[k])-v)<1e-6, f"{k} differs from checkpoint"
g=ck.get("grasp_design",{})
assert g.get("enable_point_rayrope") and g.get("enable_gvar"), "Wrong grasp design"
m=ck["model_state_dict"]
assert "depth_net.height_analytic_signature" in m, "Height signature missing"
assert not any(k.startswith("depth_net.pose_aware_adapter.") for k in m), "FiLM parameters present"
assert ck.get("gntrans_mix_protocol",{}).get("train_fraction_each_domain")==.5, "Not mix50"
assert ck["gntrans_mix_protocol"].get("original_train_count")==12800
assert ck["gntrans_mix_protocol"].get("gntrans_train_count")==12800
print("Checkpoint protocol valid: height-only, paired mix50, Point RayRoPE + GVAR")
PY
bash run_gntrans_mix_cva_cdf_eval.sh
python - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])/"original"
for split in ("test_seen","test_similar","test_novel"):
    files=list((root/split).glob("scene_*/realsense/*.npy"))
    if len(files)!=7680: raise RuntimeError(f"{split}: expected 7680 grasp npy files, got {len(files)}")
    print(split, "frames", len(files))
print("FULL_GRASPNET_INFERENCE_COMPLETE: 23040 frames")
PY
