# Rep-P0: Fixed-Action Geometry-Source Diagnosis

Branch: `exp/rep-p0-geometry-source-diagnostic` created directly from `main`.

## Scientific question

For an explicit physical grasp action (g), how much of its exact-action
quality can be decoded from different geometry information sources?

Rep-P0 does **not** change the action when the geometry source changes.

The RGB-only Stage-1 model first predicts a native grasp. Around each native
translation, Rep-P0 creates the fixed same-camera-ray offsets

`[-40,-20,-10,0,+10,+20,+40] mm`.

For every offset:
- translation changes along the same camera ray;
- rotation is fixed;
- width is fixed;
- gripper height is fixed;
- insertion depth is fixed.

The resulting physical action is evaluated once by the exact CAD/DexNet
evaluator. All geometry sources share that exact label.

## Geometry sources

- `pred`: RGB-only Stage-1 predicted metric depth.
- `sensor`: captured RealSense depth.
- `rendered`: clean virtual/rendered visible depth from `virtual_scenes`.
- `cad_full`: complete posed CAD objects plus table in camera coordinates.

Only `pred` is RGB-deployable. `sensor`, `rendered`, and especially
`cad_full` are information-source diagnostics. `cad_full` is a privileged
upper bound and must never be reported as an RGB-only test input.

## Common action-conditioned descriptor

Each source is converted to a point cloud and voxelized at the same resolution
(default 8 mm). For the same grasp action, points are transformed into the
gripper frame using the convention already used by the model-free collision
detector.

The descriptor contains:
- normalized local 3D occupancy histogram;
- left/right finger, palm, approach, and closing-region occupancies;
- local point-distribution statistics;
- the complete action parameters (R,t,w,h,d).

The same descriptor and the same MLP are used for every source.

## Supervision

Exact evaluator friction (mu^*(g)) is converted to the six threshold labels

[
y_	au(g)=1[0<mu^*(g)le	au],
quad
	auin{0.2,0.4,0.6,0.8,1.0,1.2}.
]

Each source-specific probe is trained with the same unbalanced six-threshold
BCE. Predicted action utility is the mean of the six sigmoid outputs.

The validation split (test_seen scenes 100--129) selects a conservative margin:
an alternative center replaces native only if its predicted utility exceeds the
predicted native utility by more than the selected margin.

Similar/Novel never tune this margin.

## Files

Root Python entry points:

- `mine_rep_p0_geometry_sources.py`
- `train_rep_p0_geometry_probe.py`
- `test_rep_p0_geometry_probe.py`
- `summarize_rep_p0_geometry_sources.py`
- `rep_p0_geometry_common.py`

Bash launchers:

- `scripts/run_rep_p0_mine.sh`
- `scripts/run_rep_p0_train.sh`
- `scripts/run_rep_p0_test.sh`
- `scripts/run_rep_p0.sh`

## Unit tests

```bash
pytest -q tests/test_rep_p0_geometry.py
```

## Smoke mining

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources \
SPLITS=train \
MINE_GPUS=0 \
MAX_SAMPLES_PER_SHARD=2 \
QUERY_EVAL_NUM=16 \
bash scripts/run_rep_p0_mine.sh
```

Use a separate `WORK_ROOT` for smoke tests so partial smoke caches are never
mistaken for the formal run.

## Formal pipeline

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources \
MINE_GPUS=0,1,2,3,4,5 \
TRAIN_GPUS=0,1,2,3 \
TEST_GPUS=0,1,2,3 \
SAMPLE_INTERVAL=0.1 \
QUERY_EVAL_NUM=64 \
PHASES=mine,train,test \
bash scripts/run_rep_p0.sh
```

The cache layout is

```text
WORK_ROOT/cache/
  train/
  test_seen/
  test_similar/
  test_novel/
```

The training layout is

```text
WORK_ROOT/train/
  pred/
  sensor/
  rendered/
  cad_full/
```

Final summaries are written to

```text
WORK_ROOT/test/comparison.csv
WORK_ROOT/test/comparison.json
```

## Interpretation

The primary comparison is source-wise, on exactly paired physical actions.

- `sensor >> pred`: predicted geometry accuracy is a major bottleneck.
- `rendered >> sensor`: captured-depth missingness/noise or visible-surface
  quality materially limits action judgment.
- `cad_full >> rendered`: hidden/full geometry contributes information that
  visible depth does not provide.
- `pred ~= sensor ~= rendered << cad_full`: visible geometry may be
  insufficient for the exact target; shape prior/uncertainty is more relevant
  than further local selector engineering.
- `pred` already approaches `rendered/cad_full`: the failure is less likely
  to be geometry information and more likely to be how the grasp network forms
  or trains its action-conditioned representation.
