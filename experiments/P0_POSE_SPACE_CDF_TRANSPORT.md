# P0 — Pose-Space CDF Transport Diagnostic

## Question

The current Uniform Exact-Query experiment aligns the student and teacher in
**image seed** and **selected view**, but not in physical 3D center:

```text
student center = backproject(pixel, predicted depth)
teacher center = backproject(pixel, clean GT depth)
```

P0 asks a narrower mechanism question before changing training:

> How much of this physical-center mismatch lies along the selected grasp
> approach and can therefore be represented by the existing EconomicGrasp
> grasp-depth axis, without calling the GraspNet/Dex-Net evaluator?

This is a diagnostic only. It never updates model weights and never creates
force-closure/collision labels with Dex-Net.

## Why this implementation matches `main`

`main` already contains the same geometry in the legacy helper
`process_grasp_labels_depth_cls_compensated`:

```text
e = matched_grasp_point - predicted_seed
parallel = dot(e, approach)
lateral = ||e - parallel * approach||
valid_comp = lateral < 5 mm and |parallel| < tolerated_depth
new_depth = original_depth + parallel
```

P0 extends this idea to the current CDF setup **only as a counterfactual
measurement**. The normal CDF matcher and model are unchanged.

The current CDF matcher marks a query valid only if its predicted center is
within 5 mm of the nearest CAD-derived grasp point. Therefore P0 reports both:

1. teacher/student center transportability; and
2. how much CDF support rejected by the current 5-mm center criterion would be
   recovered by approach-line compensation.

## Protocol

For each RGB student query:

1. Student runs normally in `eval()` and owns its image-FPS seed and Top-1 view.
2. Stage-0 clean-depth teacher reuses the **exact ordered seed indices**.
3. Teacher also reuses the **exact student Top-1 view index**.
4. Let `a` be the grasp approach axis, i.e. the first column of
   `grasp_top_view_rot` (`a == -grasp_top_view_xyz`).
5. Decompose center error:

   ```text
   e_center = c_teacher - c_student
   delta_parallel = dot(e_center, a)
   e_lateral = e_center - delta_parallel * a
   ```

6. The continuous transport lower bound removes `delta_parallel`; its remaining
   action-space mismatch is `||e_lateral||`.
7. The nearest-bin transport uses the same rounding convention as the existing
   legacy compensation helper:

   ```text
   shift_bin = floor(delta_parallel / 10 mm + 0.5)
   ```

   Source depth `d` maps to `d + shift_bin`; out-of-range candidates are dropped,
   not clipped.

A second decomposition uses the **student's nearest CAD-derived grasp point**
rather than the teacher center. This directly diagnoses recovery of the current
CDF label-matching support.

## Primary outputs

`summary.json` contains globally reduced metrics. The important ones are:

| Metric | Meaning |
|---|---|
| `common_valid_query_ratio` | Current student/teacher common-valid support |
| `support_iou_before` | Current query-support IoU |
| `support_iou_after_continuous` | IoU if approach-parallel error were absorbed continuously |
| `support_iou_after_nearest` | IoU after 1-cm depth-bin quantization and finite 4-bin range |
| `p0_recovered_*_fraction_of_missing_teacher_support` | Fraction of teacher-supported/student-invalid queries recovered |
| `center_lateral_mean_m` | Irreducible teacher/student center error for depth transport |
| `center_nearest_action_residual_mean_m` | Lateral + depth-bin quantization residual |
| `center_depth_retained_fraction` | Fraction of source depth candidates that remain inside D=4 after transport |
| `common_gt_cdf_element_exact_before` | Compatibility check against historical common-valid label exactness |
| `teacher_cdf_bce_before_transport` | Teacher CDF BCE on paired current student targets |
| `teacher_cdf_bce_after_nearest_transport` | Same BCE after depth-axis teacher transport |
| `teacher_cdf_transport_improved_query_ratio` | Fraction of paired queries whose BCE improves |

The runner also writes:

```text
protocol.json
batch_summary_rankXX.csv
query_sample_rankXX.csv.gz
```

The sampled query rows preserve center/lateral/parallel error, discrete shift,
support status, and CDF BCE before/after transport for failure analysis without
creating millions of CSV rows.

## Interpretation

P0 is positive if the following pattern appears, especially on Similar/Novel:

```text
support_iou_after > support_iou_before
teacher-support recall increases materially
many missing teacher-supported queries are recovered
center lateral residual is much smaller than full center error
nearest-bin residual remains mostly within ~5 mm
finite D=4 range does not discard most transported candidates
```

CDF BCE is a secondary check. The primary P0 question is geometric/support
recoverability. A recovered query has no valid target under the current hard
matcher, so P0 deliberately does **not** manufacture a Dex-Net label for it.

If continuous recovery is high but nearest-bin recovery is low, the bottleneck
is the coarse 1-cm / four-bin grasp-depth representation; this argues for soft
or continuous depth transport. If both are low because lateral residual remains
large, depth-axis transport cannot solve the mismatch and the next experiment
should move to query-center recentering or a continuous privileged grasp field.

## Run

Smoke test on one split:

```bash
DATASET_ROOT=/path/to/graspnet \
STUDENT_CKPT=/path/to/stage1_checkpoint.tar \
TEACHER_CKPT=/path/to/stage0_teacher.tar \
GPUS=0 \
SPLITS=test_seen \
MAX_BATCHES=10 \
bash run_p0_pose_space_cdf_transport.sh
```

Full three-split run on six GPUs:

```bash
DATASET_ROOT=/path/to/graspnet \
STUDENT_CKPT=/path/to/stage1_checkpoint_20.tar \
TEACHER_CKPT=/path/to/stage0_teacher.tar \
GPUS=0,1,2,3,4,5 \
SPLITS=test_seen,test_similar,test_novel \
OUTPUT_ROOT=/path/to/results/p0_pose_space_cdf_transport \
bash run_p0_pose_space_cdf_transport.sh
```

The launcher defaults to `POSE_DEPTH_MODE=global_film` and
`USE_FUSE_DEPTH=1`, matching the recent controlled RGB student protocol. Change
these environment variables if the selected checkpoint metadata says otherwise;
the diagnostic rejects mismatched checkpoint/config combinations rather than
silently loading them.

After all three splits finish:

```bash
python tools/summarize_p0_pose_space_cdf_transport.py \
  /path/to/results/p0_pose_space_cdf_transport
```

This writes `p0_split_summary.csv` and `p0_split_summary.md` at the output root.
