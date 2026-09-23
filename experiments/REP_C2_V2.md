# Rep-C2-v2: Full-path Native-vs-Correction Verification

Branch: `exp/rep-a-depth-robust-reader`.

Rep-C2-v1 was intentionally diagnostic, but its physical actions were fixed while
only reader depth was perturbed. Under that regime A1 moves are often harmful.
Rep-C1 showed the opposite regime is the one that matters for deployment: when
joint depth error changes the physical Stage-1 anchor, A1 fixed-0 proposals
contain substantial recoverable grasp utility, and ideal accept/reject is a
large remaining upper bound.

Rep-C2-v2 therefore changes the problem definition rather than enlarging the
v1 gate.

## 1. Problem definition

Proposal is always frozen A1 fixed-0:

[
g_c = g_{A1,;fixed0}.
]

The verifier decides whether to execute (g_c) or keep the regenerated native
grasp (g_0).

Training labels come from the exact CAD/DexNet evaluator after the *full Stage-1
path* has been rerun under a joint depth perturbation. The target is explicitly
three-way:

[
	ext{harmful},quad 	ext{equivalent},quad 	ext{beneficial},
]

with an auxiliary regression target

[
Delta U = U(g_c)-U(g_0).
]

Equivalent proposals are no longer silently merged into the harmful class.

## 2. Evidence variants

All variants instantiate the same modules and parameter count.

| Variant | Evidence |
|---|---|
| `score` | A1 native/proposal six-threshold CDF profiles, relative score, offset and Stage-1 native score |
| `rgb` | score + paired candidate-conditioned pre-enhancer RGB feature evidence |
| `rgb_only` | paired pre-enhancer RGB evidence only |

The RGB branch reuses `IndependentImageReader` and projects native/corrected
gripper regions into the pre-enhancer image feature map. It receives camera
intrinsics and the physical grasp hypothesis but **no predicted depth map**.
This makes the new evidence structurally independent from the metric-depth
reader that generated the error.

The intended comparison is:

[
	ext{score} ightarrow 	ext{score+RGB}.
]

`rgb_only` is an attribution control, not expected to be the strongest model.

## 3. Training-cache mining

The mining phase uses only training scenes 0--99.

For each frame in the existing Rep-A train schedule:

1. run the nominal RGB-only Stage-1 once;
2. deterministically sample one joint error using the A1 augmentation family
   (nominal / bias / scale / smooth);
3. rerun anchor, backprojection, ViewNet, CVA/CDF/width and decode;
4. expand the usual K translations;
5. score them with frozen A1;
6. take the A1 fixed-0 proposed move;
7. keep a small deterministic mixture of high-margin and uniformly sampled
   moved queries;
8. exact-evaluate native + proposed physical actions only.

The cache does not duplicate RGB feature maps. The training model loads the
matching pre-enhancer feature map from the existing Rep-A cache. Mining verifies
that this map matches the live Stage-1 capture after the recorded float16
quantization.

Default cost controls:

```text
QUERY_LIMIT = 64 Stage-1 queries scored per frame
MOVE_LIMIT  = 16 moved proposals exact-labelled per frame
one sampled joint-error case per frame
```

Thus the exact evaluator never labels all K candidates.

## 4. Validation and test

Checkpoint/threshold selection uses only `test_seen` as the existing
validation split. It consumes the existing full-query/full-path source root and
its fresh exact labels from the Rep-C1 label phase.

Default validation cases:

```text
nominal
bias:-20
bias:20
```

Validation now scores **every A1 fixed-0 moved proposal** from the full-query
Seen source (`query_limit=0`); proposal subsampling is forbidden for formal
calibration. Queries on which A1 keeps native remain in the denominator.

For each case, the verifier is evaluated on the same full-query utility scale
used by C1/test:

[
G_V(\tau)=\frac{1}{Q}\sum_q
\left[U(g_V(q;\tau))-U(g_0(q))\right].
]

The A1 fixed-0 baseline and oracle acceptance upper bound are

[
G_{A1}=\frac{1}{Q}\sum_q
\left[U(g_{A1}(q))-U(g_0(q))\right],
]

[
G_O=\frac{1}{Q}\sum_q
\max\left(U(g_{A1}(q))-U(g_0(q)),0\right).
]

Threshold selection directly maximizes the verifier increment

[
\Delta_V(\tau)=G_V(\tau)-G_{A1},
]

macro-averaged over Seen nominal / -20 / +20. Accept-all therefore has
`verifier_increment=0` rather than being rewarded by the already-positive A1
corruption gain.

Checkpoint selection uses C1 oracle-gap recovery

[
R(\tau)=\frac{G_V(\tau)-G_{A1}}{G_O-G_{A1}},
]

macro-averaged across validation cases. The primary checkpoint key is
`macro_oracle_gap_recovery`; verifier increment, beneficial retention and
harmful rejection are tie-breakers.

Similar and Novel never tune the threshold or checkpoint.

Testing runs on the same full-path source root. If that source used
`QUERY_LIMIT=0`, C2-v2 also emits complete GraspNet dumps and can run official
AP without new CAD/DexNet evaluation.

Official dump scores are matched A1 scores:
- accepted correction -> selected-hypothesis A1 score;
- rejected correction -> A1 zero/native-hypothesis score.

## 5. Update and tests

```bash
cd /home/robotarm/EconomicGrasp
git switch exp/rep-a-depth-robust-reader
git pull --ff-only

python -m pytest -q tests/test_rep_c2v2.py
```

## 6. Smoke run

Use one GPU and a small number of training frames first. Mining includes the
exact evaluator and is the highest-risk runtime phase.

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_smoke \
GPUS=0 \
MINE_MAX_FRAMES=8 \
QUERY_LIMIT=16 \
MOVE_LIMIT=4 \
MAX_TRAIN_FRAMES=8 \
MAX_VAL_FILES=12 \
MAX_TEST_FILES=12 \
EPOCHS=2 \
PHASES=mine,train,test \
bash scripts/run_rep_c2v2.sh
```

Inspect:

```text
WORK_ROOT/logs/mine_0.log
WORK_ROOT/logs/train.log
WORK_ROOT/test_*.log
WORK_ROOT/train/protocol.json
WORK_ROOT/eval/test/<split>/comparison.csv
```

## 7. Formal training/evaluation

A cautious first formal run:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier_objective_v2 \
TRAIN_CACHE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier/train_cache \
SOURCE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap \
GPUS=0 \
QUERY_LIMIT=64 \
MOVE_LIMIT=16 \
VAL_MOVE_LIMIT=0 \
EPOCHS=12 \
SPLITS=test_seen,test_similar,test_novel \
CASES=nominal,bias:-20,bias:20 \
PHASES=mine,train,test \
bash scripts/run_rep_c2v2.sh
```

Mining shards by scene across the listed GPUs. Each process also owns an exact
CAD/DexNet evaluator, so increase GPU/shard count only if host RAM is sufficient.

If test metrics are positive, official AP needs no new inference or exact action
labels:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier_objective_v2 \
SPLITS=test_seen,test_similar,test_novel \
OFFICIAL_WORKERS=4 \
PHASES=official \
RESUME=1 \
bash scripts/run_rep_c2v2.sh
```

## 8. Decision criteria

Do not judge C2-v2 only by classification accuracy.

The primary held-out quantities are:

```text
utility_gain
a1_fixed0_gain
oracle_accept_gain
beneficial_retention
harmful_rejection
accept_precision
```

Interpretation:

- `utility_gain > a1_fixed0_gain`: verification improves the aggressive A1
  proposal policy.
- `rgb > score` on Similar/Novel: independent image evidence adds information
  beyond A1 score calibration.
- `rgb ~= score`: full-path regime alignment helped, but current RGB evidence
  did not add useful verification information.
- all variants collapse to near-zero acceptance: the available observable
  evidence is still insufficient, and more gate capacity is not justified.
- a substantial gap to `oracle_accept_gain` is expected; Rep-C1 uses privileged
  exact geometry and is an upper bound, not a recoverable target.


## 9. Re-running after the full-query objective change

The mined training cache is unchanged and should be reused. The training
signature now contains an explicit objective version, so old checkpoints cannot
silently resume under the new calibration rule.

Recommended rerun:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier_objective_v2 \
TRAIN_CACHE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2v2_fullpath_verifier/train_cache \
SOURCE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap \
GPUS=0 \
VAL_MOVE_LIMIT=0 \
VAL_QUERY_CHUNK=128 \
VAL_EVERY=1 \
EPOCHS=12 \
SPLITS=test_seen,test_similar,test_novel \
CASES=nominal,bias:-20,bias:20 \
PHASES=train,test \
bash scripts/run_rep_c2v2.sh
```

No new CAD/DexNet mining is required. Validation uses the exact labels already
produced in the formal full-path source. Full-query RGB verification is chunked
by `VAL_QUERY_CHUNK` for GPU-memory control.
