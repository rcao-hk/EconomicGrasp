"""Strict, immutable per-action evidence/label bundle for P0/P1 diagnostics.

A bundle is a collection of *matched physical actions*, NOT a new grasp model.
Every action receives native/score64/all scores on the same N×17 geometry.
Offline exact CAD labels are permitted only for analysis, never probe fitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA, physical_ids

SCHEMA = "gvar-evidence-transfer-v1"
METHODS = ("native", "score64", "all")
SPLITS = ("probe_validation", "probe_audit", "test_seen",
          "test_similar", "test_novel", "external_holdout")
SCORES = ("score_native", "score_score64", "score_all")
MASKS = ("top10_native", "top10_score64", "top10_all",
         "top50_native", "top50_score64", "top50_all")
LABELS = ("quality_bits", "assigned_obj", "pure_collision", "empty")
MU = np.array([.2, .4, .6, .8, 1.0, 1.2], dtype=np.float64)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for buf in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True,
              separators=(",", ":"), ensure_ascii=False,
              allow_nan=False).encode("utf-8")).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, ensure_ascii=False,
                              indent=2, allow_nan=False), encoding="utf-8")


def save_npz(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **fields)


def validate_frame(x, *, require_quality=True):
    """One record is a labeled SAME-action union, normally NMS-survivor union.

    Actual source labels and scores must be read; neither missing object ID
    nor missing geometry is substituted with a plausible synthetic value.
    """
    required=("grasps", "physical_id", "native_percentile", *SCORES,
              *MASKS) + (LABELS if require_quality else ())
    absent=[k for k in required if k not in x]
    if absent:
        raise ValueError(f"Missing essential per-action fields: {absent}")
    g=np.asarray(x["grasps"])
    if g.ndim != 2 or g.shape[1] != 17 or not len(g) or not np.isfinite(g).all():
        raise ValueError("Expected finite nonempty [N,17] grasps")
    n=len(g)
    if np.asarray(x["physical_id"]).shape != (n,) or not np.array_equal(
           physical_ids(g), np.asarray(x["physical_id"]).astype(str)):
        raise ValueError("Physical-action SHA256 IDs do not match 17D grasps")
    if len(np.unique(x["physical_id"])) != n:
        raise ValueError("Repeated physical action in one matched frame")
    for key in SCORES:
        p=np.asarray(x[key], dtype=np.float64)
        if p.shape != (n,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError(f"Expected [N] finite 0..1 quality scores: {key}")
    pct=np.asarray(x["native_percentile"],dtype=np.float64)
    if pct.shape!=(n,) or not np.isfinite(pct).all() or np.any((pct<0)|(pct>1)):
        raise ValueError("Native score percentile must be [N] in [0,1]")
    for name in MASKS:
        val=np.asarray(x[name])
        if val.shape!=(n,) or not np.isin(val,[0,1]).all():
            raise ValueError(f"Invalid final-prefix membership mask: {name}")
        k=10 if "top10" in name else 50
        if int(val.sum())>k:
            raise ValueError(f"Too many selected members in {name}")
    for method in METHODS:
        if np.any((np.asarray(x[f"top10_{method}"])>0)&
                  (np.asarray(x[f"top50_{method}"])==0)):
            raise ValueError(f"Top10 not contained in Top50: {method}")
    if require_quality:
        bits=np.asarray(x["quality_bits"])
        if bits.shape!=(n,6) or not np.isin(bits,[0,1]).all():
            raise ValueError("Quality bits must be binary [N,6]")
        if np.any(np.diff(bits.astype(int),axis=1)<0):
            raise ValueError("Six-friction success bits are not monotone")
        for name in ("assigned_obj","pure_collision","empty"):
            if np.asarray(x[name]).shape!=(n,):
                raise ValueError(f"Exact CAD label {name} does not align")
        for name in ("pure_collision","empty"):
            if not np.isin(x[name],[0,1]).all():
                raise ValueError(f"Invalid failure flag: {name}")
        if not np.issubdtype(np.asarray(x["assigned_obj"]).dtype, np.integer):
            raise ValueError("CAD object assignment must be integer IDs")
        if np.any((bits.sum(axis=1)>0)&(
               (np.asarray(x["pure_collision"])>0)|
               (np.asarray(x["empty"])>0))):
            raise ValueError("Positive quality contradicts collision/empty")
    if "pred_geometry" in x:
        geom=np.asarray(x["pred_geometry"])
        if geom.shape!=(n,6,6) or not np.isfinite(geom).all():
            raise ValueError("Expected action-aligned [N,6,6] predicted geometry")
    if "reference_geometry" in x:
        if "pred_geometry" not in x:
            raise ValueError("Reference geometry requires paired predicted geometry")
        geom=np.asarray(x["reference_geometry"])
        if geom.shape!=(n,6,6) or not np.isfinite(geom).all():
            raise ValueError("Reference geometry is not action aligned")
    return n


def failure_classes(x):
    """Mutually exclusive classes: 0 success, 1 collision, 2 empty, 3 other-zero."""
    q=np.asarray(x["quality_bits"]).sum(axis=1)
    collision=np.asarray(x["pure_collision"],dtype=bool)
    empty=np.asarray(x["empty"],dtype=bool)
    result=np.full(len(q),3,dtype=np.int64)
    result[q>0]=0
    result[(q==0)&empty]=2
    result[(q==0)&collision]=1
    return result


def load_manifest(path, *, complete=True):
    path=Path(path)
    data=json.loads(path.read_text(encoding="utf-8"))
    sig=data.pop("manifest_sha256",None)
    if sig!=digest(data):
        raise ValueError(f"Evidence manifest signature invalid: {path}")
    data["manifest_sha256"]=sig
    if data.get("schema") != SCHEMA or data.get("base_checkpoint_sha256") != BASE_SHA:
        raise ValueError("Evidence manifest/baseline identity mismatch")
    if complete and not data.get("complete"):
        raise ValueError("Partial/smoke evidence bundle is not formally comparable")
    frames=data.get("frames",[])
    if not frames:
        raise ValueError("Empty evidence bundle")
    keys=set()
    for record in frames:
        for key in ("dataset","split","scene","anno","file","sha256"):
            if key not in record:
                raise ValueError(f"Malformed frame manifest missing {key}")
        if record["split"] not in SPLITS:
            raise ValueError(f"Unknown split category: {record['split']}")
        ident=(str(record["dataset"]),str(record["split"]),
               str(record["scene"]),int(record["anno"]))
        if ident in keys:
            raise ValueError(f"Duplicate source frame: {ident}")
        keys.add(ident)
        if len(str(record["sha256"]))!=64:
            raise ValueError("Expected SHA256 per frame")
    return data


def load_frame(record):
    path=Path(record["file"])
    if not path.is_file() or sha256(path)!=record["sha256"]:
        raise ValueError(f"Missing or modified action-label bundle: {path}")
    with np.load(path,allow_pickle=False) as z:
        data={key:z[key] for key in z.files}
    validate_frame(data)
    return data


def pose_attributes(g):
    """Physical pose only, no unverified geometry-error claims."""
    g=np.asarray(g,dtype=np.float64)
    rotation=g[:,4:13].reshape(-1,3,3)
    approach=rotation[:,:,0]
    polar=np.rad2deg(np.arccos(np.clip(np.abs(approach[:,2]),0,1)))
    width=g[:,1]
    center_z=g[:,15]
    insertion=g[:,3]
    return dict(
        polar_bin=np.digitize(polar,[30,60]).astype(np.int64),
        width_bin=np.digitize(width,[.03,.06]).astype(np.int64),
        center_depth_bin=np.digitize(center_z,[.45,.75]).astype(np.int64),
        insertion_bin=np.digitize(insertion,[.015,.025,.035]).astype(np.int64),
    )


def geometry_error(x):
    """Only explicit aligned reference depth supports a geometric-error claim.

    Soft known flags from predicted and reference geometry must both exceed .5.
    Unknown/occluded probes are never considered free or occupied.
    """
    if "pred_geometry" not in x or "reference_geometry" not in x:
        return None
    a=np.asarray(x["pred_geometry"],dtype=np.float64)
    b=np.asarray(x["reference_geometry"],dtype=np.float64)
    both=(a[...,4]>.5)&(b[...,4]>.5)
    diff=np.abs(a[...,3]-b[...,3])
    error=np.full(len(a),np.nan,dtype=np.float64)
    for i in range(len(a)):
        if both[i].any():
            error[i]=float(diff[i][both[i]].mean())
    return error
