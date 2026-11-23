# Point RayRoPE LAN inference demo

This demo exposes the **pose-aware DPT + Point RayRoPE + CVA-CDF** model as a
small TCP service on a private LAN. The GPU machine runs the model; the robot
PC sends one scene crop and receives scene-level Top-10 grasps.

The demo does not contain robot motion or gripper commands. The AUBO
pick-and-place script is used only as a reference for collision filtering,
score sorting and grasp NMS.

## Model and checkpoint

The intended model is the **two-source** 50% GraspNet-1Billion + 50% GN-Trans
Point RayRoPE run, with the pose-aware DAV2/DPT metric-depth path using
`global_film`. This is distinct from the GC6D-inclusive three-source
experiment; **do not use the GraspNet+GN-Trans+GC6D checkpoint** for this demo.

The exact checkpoint path and SHA256 for the requested two-source checkpoint
are not present in the retrieved experiment records. Therefore the server
requires an explicit `--checkpoint` path (or
`ECONOMICGRASP_RAYROPE_CKPT`) and does not guess a similarly named checkpoint.
If you have the checkpoint's SHA256, pass it with
`--expected-checkpoint-sha256`.

The server checks the checkpoint's `grasp_design` metadata for Point RayRoPE
and rejects GVAR-enabled checkpoints. It also loads the state dict strictly
and reads RayRoPE grouping chunk/checkpoint settings from the checkpoint
metadata. These checks validate architecture compatibility; they do not
independently prove the training-data mixture if the checkpoint does not store
that provenance.

The complete checkpoint should contain the DINOv2/DPT weights, so the server
uses the inference-only `load_pretrained_depth_backbone=False` path and does
not require a separate `checkpoints/depth_anything_v2_vitb.pth` file.

## 1. Start the GPU server

Run from the repository root on the machine that has the checkpoint and CUDA
environment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m demo.rayrope_lan.server \
  --host 0.0.0.0 \
  --port 8765 \
  --device cuda:0 \
  --checkpoint /path/to/two_source_rs50_gntrans50_point_rayrope/checkpoint.tar \
  --top-k 10 \
  --collision-thresh 0.05 \
  --collision-voxel-size 0.01 \
  --approach-dist 0.05
```

The default collision threshold (0.05), voxel size (0.01 m) and approach
distance (0.05 m) follow the post-processing convention used by
`demo_pick_place_aubo.py`. Set `--collision-thresh 0` to disable the
model-free collision filter deliberately. By default, the complete valid
depth crop is passed to Open3D voxel downsampling before collision checking;
set `--max-collision-points 120000` only if the input crop is unusually large.
The current `ModelFreeCollisionDetectorTorch` implementation allocates on
`cuda:0`; use `CUDA_VISIBLE_DEVICES` to select the physical GPU and keep
`--device cuda:0`. The server serializes GPU inference with a lock while
allowing multiple TCP clients to connect.

For a shared token on a trusted private network:

```bash
export ECONOMICGRASP_LAN_TOKEN='replace-with-a-private-token'
```

The client must send the same value using `--auth-token`. Do not expose this
demo port to the public internet.

## 2. Client request

The client sends two arrays:

- `rgb`: uint8 RGB image crop, shape `[Hcrop,Wcrop,3]`.
- `depth`: aligned single-channel depth crop, shape `[Hcrop,Wcrop]`.

It also sends a JSON metadata object. The image arrays must be the
**raw-resolution crop**, not resized to 448x448. The server applies the same
RGB transform as `GraspNetMultiDataset` and computes the crop-adjusted,
448x448 camera intrinsics.

Example `crop_meta.json`:

```json
{
  "request_id": "scene_0001_frame_0001",
  "crop_box_xyxy": [160, 80, 1120, 680],
  "original_hw": [720, 1280],
  "camera_intrinsics": [
    [910.0, 0.0, 640.0],
    [0.0, 910.0, 360.0],
    [0.0, 0.0, 1.0]
  ],
  "depth_scale_to_m": 0.001,
  "camera_pose_vec": [0.0, 0.0, 1.0],
  "camera_frame_id": "camera_color_optical_frame",
  "base_frame_id": "base_link",
  "T_base_from_camera": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

The numbers above are illustrative; replace them with the actual camera
intrinsics and robot/camera transforms.

### Required camera-pose input

The selected DPT checkpoint uses `pose_depth_mode=global_film`. Therefore,
intrinsics alone are **not sufficient**. Every request must provide one of:

1. `camera_pose_vec`: a 3D unit vector in the table frame, following the
   dataset convention `-R_table_from_camera[:,2]`; or
2. `T_table_from_camera`: a 4x4 transform, from which the server derives
   that vector.

This is separate from the intrinsics. For an eye-in-hand camera, compute the
transform from the current robot pose and the calibrated hand-eye transform.
If the table frame equals the robot base frame, the same transform may also
be supplied as `T_base_from_camera`.

`T_base_from_camera` is optional. When supplied, the response includes both
camera-frame and base-frame grasp poses. Without it, results are returned in
the camera optical frame.

### Crop and depth conventions

- `crop_box_xyxy` is `[x0,y0,x1,y1]` in the original full-image pixel
  coordinates, with the right/bottom edges exclusive.
- `original_hw` is `[Hfull,Wfull]`.
- `camera_intrinsics` is the original full-image 3x3 pinhole matrix. The
  server subtracts the crop offset from the principal point and scales
  `fx,fy,cx,cy` to 448x448, matching the dataset's
  `resize_intrinsics_with_crop`.
- RGB and depth crops must have shape `[y1-y0,x1-x0]`.
- `depth_scale_to_m` is a multiplier converting raw depth values to meters:
  uint16 millimeters use `0.001`; float depth already in meters uses `1.0`.
- Images must be rectified/undistorted pinhole images. Distortion coefficients
  are not part of this protocol.
- The crop should cover the **whole scene/workspace** if the requested result
  is scene-level Top-10. An object-only crop produces Top-10 for that object
  crop, not for the full scene.

The model predicts metric geometry from RGB through the pose-aware DPT head.
The transmitted sensor depth is used only to build the model-free collision
cloud; it is not passed into the network or fused into the grasp representation.
This preserves the RGB-only model path while allowing the AUBO-style
post-processing step.

## 3. Send a request

```bash
python -m demo.rayrope_lan.client \
  --server-ip 192.168.1.20 \
  --port 8765 \
  --rgb /path/to/crop_rgb.png \
  --depth /path/to/crop_depth.png \
  --metadata /path/to/crop_meta.json \
  --output /path/to/top10_grasps.npz
```

With authentication enabled, add `--auth-token`. The client also supports
NumPy `.npy` depth files. It opens one TCP connection per request.

For an eye-in-hand program that already has arrays in memory, use the client
function directly instead of writing temporary image files:

```python
from demo.rayrope_lan.client import (
    send_rgbd_request,
    decode_response_metadata,
)

response = send_rgbd_request(
    server_ip="192.168.1.20",
    port=8765,
    rgb=rgb_crop,          # uint8 [Hcrop,Wcrop,3], RGB
    depth=depth_crop,      # aligned [Hcrop,Wcrop]
    metadata=crop_metadata # same fields as crop_meta.json
)
response_meta = decode_response_metadata(response)
R_camera = response["rotation_matrices_camera"]  # [N,3,3]
t_camera = response["translations_camera_m"]     # [N,3], meters
width_m = response["widths_m"]                   # [N], meters
scores = response["scores"]                      # [N]
```

## 4. Wire protocol

The transport uses only Python's standard socket API plus NumPy:

1. An 8-byte unsigned big-endian payload length.
2. A compressed NPZ payload containing `rgb`, `depth` and
   `metadata_json` (UTF-8 JSON stored as uint8).

The response uses the same framing and contains `metadata_json` plus grasp
arrays. No pickle deserialization is used. The server enforces a maximum
request size (default 64 MiB).

## 5. Response format

The response includes:

| Field | Shape | Meaning |
|---|---|---|
| `scores` | `[N]` | Predicted grasp confidence |
| `widths_m` | `[N]` | Parallel-jaw opening width in meters |
| `rotation_matrices_camera` | `[N,3,3]` | Rotation matrix R in the camera frame |
| `translations_camera_m` | `[N,3]` | Translation t in the camera frame, in meters |
| `grasps_camera` | `[N,17]` | GraspNet-compatible row: score, width, height, depth, flattened rotation R, translation t, object id |
| `grasps_base` | `[N,17]` | Optional base-frame rows when `T_base_from_camera` is supplied |

The response metadata reports the coordinate frame, checkpoint SHA256,
number of raw candidates, candidates removed by collision filtering, final
Top-K count and timings. N is at most 10; fewer than 10 are returned if
collision filtering and NMS leave fewer valid candidates.

The post-processing order is finite/width validity filtering, optional
model-free collision filtering using the received depth crop, score sorting,
grasp NMS, then Top-10 truncation. Robot-specific reachability, approach-angle
constraints and trajectory planning remain on the robot client.

## 6. Smoke test

The transport unit test can be run without loading the grasp model:

```bash
python -m unittest discover -s tests -p test_rayrope_lan_protocol.py
```

A full model smoke test requires the checkpoint, the repository's CUDA/PyTorch
environment, MinkowskiEngine and `graspnetAPI`.
