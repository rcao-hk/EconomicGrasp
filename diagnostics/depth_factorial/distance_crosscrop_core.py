"""Pure, CPU-testable preregistered Distance(near/wide) × Crop(ROI/multi3).

The cross factors MUST not be confounded by using step%3 for both.
For multi3, distance rotates every step and crop every THREE steps:
all 9 distance×crop combinations occur approximately equally often.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

CROPS=("fixed_full","fixed_roi","adaptive_workspace")
ARMS=("near_only","near_mid_far")
CROP_DESIGNS=("roi","multi3")
MAIN_STEPS=200
MAIN_PIXELS=256
MAIN_SEED=20261117


def schedule(steps=MAIN_STEPS,seed=MAIN_SEED,objects=2,arm="near_mid_far",
             crop_design="multi3"):
    if (type(steps) is not int or steps<9 or type(seed) is not int or seed<0 or
        type(objects) is not int or objects<2 or arm not in ARMS or
        crop_design not in CROP_DESIGNS):
        raise ValueError("Invalid fixed-budget distance×crop protocol")
    rng=np.random.default_rng(seed+701)
    chosen=rng.integers(0,objects,size=steps)
    rows=[]
    for step,obj in enumerate(chosen):
        distance=(0 if arm=="near_only" else (0,2,4)[step%3])
        crop=("fixed_roi" if crop_design=="roi"
              else CROPS[(step//3)%3])
        rows.append(dict(step=step,object_index=int(obj),
                         distance_id=distance,crop_policy=crop))
    if crop_design=="multi3":
        counts=Counter((r["distance_id"],r["crop_policy"]) for r in rows)
        keys=((0,) if arm=="near_only" else (0,2,4))
        values=[counts[(d,c)] for d in keys for c in CROPS]
        if min(values)<1 or max(values)-min(values)>1:
            raise RuntimeError("A distance×crop factor is missing or confounded")
    return rows


def verify_four_arm_protocols(protocols,registration_sha):
    """Strict compare historical ROI vs new multi3 arm training metadata.

    Historical ROI runs do not encode crop schedule: exactly fixed_roi.
    The new multi3 protocols include explicit per-step crop identities.
    """
    if set(protocols)!={
        "roi_near","roi_wide","multi_near","multi_wide"}:
        raise ValueError("Need four 2x2 distance×crop training arms")
    keys=("base_checkpoint_sha256","initial_depth_head_sha256",
          "registration_identity_sha256","steps","pixels_per_step",
          "total_supervised_pixels","learning_rate","weight_decay",
          "smooth_l1_beta","gradient_clip","seed","optimizer",
          "trainable_module","trainable_params","train_object_sequence")
    master=protocols["roi_near"]
    for name,p in protocols.items():
        if p.get("train_registration_sha256")!=registration_sha:
            raise ValueError(f"Different CAD registration source: {name}")
        for key in keys:
            if p.get(key)!=master.get(key):
                raise ValueError(f"Confounded training setting {key}: {name}")
        target_arm=("near_only" if name.endswith("near") else "near_mid_far")
        if p.get("arm")!=target_arm:
            raise ValueError(f"Incorrect distance arm: {name}")
        ctype="multi3" if name.startswith("multi") else "roi"
        expected=schedule(int(p["steps"]),int(p["seed"]),
                          objects=len(set(master["train_object_sequence"])),
                          arm=target_arm,crop_design=ctype)
        # The object index sequence is random from the actual original list,
        # so only compare object SHAs directly; not assumed integers.
        distances=[r["distance_id"] for r in expected]
        if list(p.get("train_distance_indices",[]))!=distances:
            raise ValueError(f"Distance schedule changed: {name}")
        if ctype=="roi":
            if p.get("crop_policy")!="fixed_roi":
                raise ValueError("Historical arms must train on ROI only")
        else:
            if p.get("crop_policy")!="multi3":
                raise ValueError("New arms must use matched 3-crop schedule")
            if p.get("train_crop_indices")!=[r["crop_policy"] for r in expected]:
                raise ValueError("Crop and distance schedule do not cross correctly")
    return dict(verified=True,steps=int(master["steps"]),
                pixels=int(master["pixels_per_step"]),
                frozen_base=master["base_checkpoint_sha256"],
                registration_sha256=registration_sha,
                seed=int(master["seed"]))
