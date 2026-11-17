"""Source-checked physical camera distance x crop experiment contracts.

The distance factor must move a perspective camera in 3D while fixing its K,
orientation, object mesh/world pose, material, lights, and render engine.
No image zoom or post-hoc depth multiplication substitutes for camera motion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PLAN_SCHEMA="physical-distance-crop-render-plan-v1"
SOURCE_SCHEMA="physical-distance-render-source-v1"
INFER_SCHEMA="physical-distance-by-crop-frozen-pred-v1"
CROP_POLICIES=("fixed_full","fixed_roi","adaptive_workspace")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False,
                ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def write_json(path,data,*,overwrite=False):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not overwrite:raise FileExistsError(p)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,
                           allow_nan=False),encoding="utf-8")


def _R(a):
    r=np.asarray(a,dtype=np.float64)
    if (r.shape!=(3,3) or not np.isfinite(r).all()
            or not np.allclose(r.T@r,np.eye(3),atol=1e-5)
            or abs(np.linalg.det(r)-1)>1e-5):
        raise ValueError("Camera-to-world rotation must be right-handed SO(3)")
    return r


def validate_plan(plan,*,allow_procedural=True):
    if plan.get("schema")!=PLAN_SCHEMA:raise ValueError("Wrong physical render plan schema")
    cam=plan.get("camera",{})
    w=int(cam.get("width",0));h=int(cam.get("height",0))
    fx=float(cam.get("fx",0));fy=float(cam.get("fy",0))
    cx=float(cam.get("cx",-100));cy=float(cam.get("cy",-100))
    if w<128 or h<128 or w%2 or h%2 or min(fx,fy)<=0:
        raise ValueError("Need even, positive perspective camera resolution/focal")
    # The Blender physically controlled renderer below only implements a
    # verified centered square-pixel camera. Do not silently approximate K.
    if not(np.isclose(fx,fy,rtol=0,atol=1e-6) and
           np.isclose(cx,w/2.,atol=1e-6) and
           np.isclose(cy,h/2.,atol=1e-6)):
        raise ValueError("Blender renderer currently supports centered fx=fy camera only")
    R=_R(cam["R_c2w"])
    target=np.asarray(plan.get("target_world"),dtype=float)
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError("Invalid fixed world target")
    values=plan.get("distances_m",[])
    if (len(values)<3 or len(values)!=len(set(values)) or
            sorted(values)!=values or
            not all(isinstance(v,(int,float)) and .15<float(v)<2. for v in values)):
        raise ValueError("Need >=3 pre-registered unique ascending physical distances in metres")
    obj=plan.get("object",{})
    if obj.get("kind")=="mesh":
        mesh=Path(obj.get("path",""))
        if not mesh.is_file() or sha256(mesh)!=obj.get("sha256"):
            raise ValueError("CAD mesh SHA missing/mismatched")
        if mesh.suffix.lower() not in (".obj",".ply",".stl",".glb"):
            raise ValueError("Unsupported Blender CAD mesh extension")
        if not 0<float(obj.get("metres_per_mesh_unit",0))<=1:
            raise ValueError("Require explicit mesh-unit-to-metre scale")
    elif obj.get("kind")=="procedural_box" and allow_procedural:
        dims=np.asarray(obj.get("dimensions_m",[]),dtype=float)
        if dims.shape!=(3,) or not np.isfinite(dims).all() or np.any(dims<=0):
            raise ValueError("Procedural box must specify valid metre dimensions")
    else:
        raise ValueError("Declare verified mesh or explicit procedural box")
    rendering=plan.get("render",{})
    if not 8<=int(rendering.get("samples",0))<=1024 or not isinstance(rendering.get("seed"),int):
        raise ValueError("Render samples and seed must be explicit")
    if rendering.get("engine")!="CYCLES":
        raise ValueError("Only fixed Cycles render engine is supported")
    return plan


def expected_c2w(plan,distance):
    validate_plan(plan)
    R=np.asarray(plan["camera"]["R_c2w"],dtype=np.float64)
    center=np.asarray(plan["target_world"],dtype=np.float64)-float(distance)*R[:,2]
    T=np.eye(4,dtype=np.float64)
    T[:3,:3]=R;T[:3,3]=center
    return T


def verify_fixed_physics(manifest,*,strict_files=True):
    if manifest.get("schema")!=SOURCE_SCHEMA or not manifest.get("complete"):
        raise ValueError("Incomplete physical distance source")
    plan=manifest.get("plan")
    validate_plan(plan)
    if len(manifest.get("views",[]))!=len(plan["distances_m"]):
        raise ValueError("Physical distance views missing")
    if manifest.get("plan_sha256")!=digest(plan):
        raise ValueError("Physical render plan modified")
    common=np.array([[plan["camera"]["fx"],0.,plan["camera"]["cx"]],
                     [0.,plan["camera"]["fy"],plan["camera"]["cy"]],
                     [0.,0.,1.]])
    distances=[]
    hashes=set()
    for view,dist in zip(manifest["views"],plan["distances_m"]):
        if not np.isclose(view["distance_m"],dist,atol=1e-9):
            raise ValueError("Unexpected/fitted camera distance")
        if not np.allclose(view["K"],common,rtol=0,atol=1e-6):
            raise ValueError("Camera K changed across physical distance experiment")
        T=expected_c2w(plan,dist)
        if not np.allclose(view["T_c2w"],T,atol=1e-6,rtol=0):
            raise ValueError("Unexpected camera rotation/trajectory in physical plan")
        paths=view.get("files",{})
        for name in ("rgb","depth","seg"):
            v=paths.get(name,{})
            path=Path(v.get("path",""))
            if not v.get("sha256") or (strict_files and (not path.is_file() or sha256(path)!=v["sha256"])):
                raise ValueError(f"Source frame {name} missing/tampered")
        if strict_files:
            for name in ("rgb","depth","seg"):
                import numpy as _np
                # Distinct per-distance file is mandatory.
                pair=str(Path(paths[name]["path"]).resolve())
                if pair in hashes:raise ValueError("Repeated output path for different physical distance")
                hashes.add(pair)
        distances.append(float(dist))
    return dict(n_views=len(distances),distances_m=distances,
                fixed_intrinsics=True,fixed_orientation=True,
                physical_translation_only=True,object_pose_fixed_by_plan=True)


def source_image(frame):
    """Load source RGB + optical-axis depth + instance mask in top-left origin."""
    from PIL import Image
    p=frame["files"]
    rgb=np.asarray(Image.open(p["rgb"]["path"]).convert("RGB"),dtype=np.uint8)
    depth=np.load(p["depth"]["path"],allow_pickle=False)
    mask=np.asarray(Image.open(p["seg"]["path"]).convert("L"))>127
    if depth.shape!=rgb.shape[:2] or mask.shape!=depth.shape:
        raise ValueError("RGB/reference depth/instance seg frame alignment differs")
    if not np.isfinite(depth).all() or np.any(depth<0):
        raise ValueError("Rendered depth nonfinite/negative")
    if not np.any(mask & (depth>0)):
        raise ValueError("Rendered object not visible")
    return rgb,depth.astype(np.float32),mask
