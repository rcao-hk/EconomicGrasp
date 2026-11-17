"""Pure 200-vs-600 compute and per-crop supervision exposure accounting.

200-step originals are ACCEPTED results, not re-trained. New ROI600 and
Multi600 start from original base, same train objects/seed, same 600-step
object sequence, optimizer and 256 foreground pixels per step.

600-step Multi: each crop exactly 200 steps, each physical distance 200,
nine distance×crop cells 66 or 67 exposures. ROI600: ROI all 600 steps.
"""
from __future__ import annotations

from collections import Counter

from .distance_crosscrop_core import MAIN_PIXELS,MAIN_SEED,schedule,CROPS


DESIGNS=("roi","multi3")
DISTANCES=(0,2,4)
STEPS=(200,600)


def budget_schedule(design,steps=600,seed=MAIN_SEED,objects=2):
    if design not in DESIGNS or steps not in STEPS:
        raise ValueError("Only preregistered ROI/Multi×200/600 designs allowed")
    return schedule(steps,seed,objects,"near_mid_far",design)


def exposure_summary(rows,pixels_per_step=MAIN_PIXELS):
    if pixels_per_step!=MAIN_PIXELS:
        raise ValueError("200/600 study keeps exactly 256 foreground pixels per step")
    crop=Counter(r["crop_policy"] for r in rows)
    dist=Counter(r["distance_id"] for r in rows)
    cells=Counter((r["distance_id"],r["crop_policy"]) for r in rows)
    return dict(steps=len(rows),supervised_pixels=len(rows)*pixels_per_step,
                per_crop=dict(crop),per_distance={str(k):v for k,v in dist.items()},
                nine_cells={f"{k[0]}:{k[1]}":v for k,v in sorted(cells.items())})


def assert_600_pair(roi,multi):
    if len(roi)!=600 or len(multi)!=600:
        raise ValueError("600-step ROI and mixed arms are mandatory")
    names=("step","object_index","distance_id")
    if any(any(a[k]!=b[k] for k in names) for a,b in zip(roi,multi)):
        raise ValueError("600 budget arms differ in object or distance sequence")
    if {r["crop_policy"] for r in roi}!={"fixed_roi"}:
        raise ValueError("ROI600 must see only the original fixed ROI")
    expo=exposure_summary(multi)
    if set(expo["per_crop"])!=set(CROPS) or set(expo["per_crop"].values())!={200}:
        raise ValueError("Multi600 per-crop exposure not equal to ROI200")
    if set(expo["per_distance"].values())!={200}:
        raise ValueError("Multi600 physical distance supervision not balanced")
    nine=expo["nine_cells"]
    if len(nine)!=9 or min(nine.values())!=66 or max(nine.values())!=67:
        raise ValueError("Nine-cell interaction under-represented at 600 steps")
    return dict(total_step_matched=True,per_crop_roi200_vs_multi600=True,
                roi_exposure_600=600,multi_per_crop_exposure=200,
                multi_nine_cell_counts=nine,
                pixels_per_step=MAIN_PIXELS)
