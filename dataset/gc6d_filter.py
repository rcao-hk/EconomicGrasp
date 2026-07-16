#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter the official GraspClutter6D training split by removing scenes whose
object library overlaps with GraspNet-1Billion test_similar/test_novel objects.

Default behavior:
  1) Read GraspNet test_similar and test_novel scene object ids:
       test_similar: scene_0130 ... scene_0159
       test_novel:   scene_0160 ... scene_0189
     i.e. scene ids 130-189. GraspNet has 190 scenes indexed 0-189.

  2) Read GraspClutter6D official train split:
       GC6D_ROOT/split_info/grasp_train_scene_ids.json

  3) For each GC6D train scene, read:
       GC6D_ROOT/scenes/<scene_id:06d>/scene_gt.json
     and collect unique obj_id values.

  4) Use a GC6D-object -> GraspNet-object overlap mapping, then exclude any
     GC6D train scene containing a GC6D object mapped to an object appearing in
     GraspNet test_similar/test_novel.

Outputs:
  OUTPUT_DIR/graspnet_test_object_library.csv
  OUTPUT_DIR/gc6d_train_scene_filter_report.csv
  OUTPUT_DIR/gc6d_conflict_object_mapping.csv
  OUTPUT_DIR/grasp_train_scene_ids_no_graspnet_test_overlap.json
  OUTPUT_DIR/grasp_train_scene_ids_excluded_by_graspnet_test_overlap.json
  OUTPUT_DIR/filter_summary.json

Example:
  python gc6d_filter_train_by_graspnet_test_objects.py \
    --graspnet_root /data/robotarm/dataset/graspnet \
    --gc6d_root /data/robotarm/dataset/GraspClutter6D \
    --graspnet_camera realsense \
    --output_dir /data/robotarm/dataset/GraspClutter6D/split_info/no_gn_test_overlap

If your GraspNet meta files store object ids as 1-based ids, use:
  --graspnet_obj_id_offset -1

If you use a custom mapping CSV, pass:
  --mapping_csv /path/to/gc6d_graspnet_overlap_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


GRASPNET_DEFAULT_SPLITS = {
    "test_seen": list(range(100, 130)),
    "test_similar": list(range(130, 160)),
    "test_novel": list(range(160, 190)),
}

# Built from instance-level overlap analysis between GraspClutter6D and
# GraspNet-1Billion object ids. GraspNet ids follow the 0-based object-list ids.
# Confidence:
#   high       = exact or high-confidence same instance/source
#   medium     = likely but should be verified with CAD/image if strict leakage audit is needed
#   collective = GC6D rows are collectively mapped to a GraspNet object set; row order is ambiguous
BUILTIN_GC6D_TO_GRASPNET = {
    # YCB exact/normalized names
    1:  (5,  "banana", "011 banana", "high"),
    2:  (12, "apple", "013 apple", "high"),
    3:  (15, "pear", "016 pear", "high"),
    4:  (11, "strawberry", "012 strawberry", "high"),
    5:  (16, "orange", "017 orange", "high"),
    6:  (14, "peach", "015 peach", "high"),
    7:  (17, "plum", "018 plum", "high"),
    8:  (13, "lemon", "014 lemon", "high"),
    10: (0,  "cracker_box", "003 cracker box", "high"),
    11: (1,  "sugar_box", "004 sugar box", "high"),
    12: (3,  "mustard_bottle", "006 mustard bottle", "high"),
    13: (2,  "tomato_soup_can", "005 tomato soup can", "high"),
    17: (4,  "potted_meat_can", "010 potted meat can", "high"),
    21: (18, "knife", "032 knife", "high"),
    25: (7,  "mug", "025 mug", "high"),
    27: (6,  "bowl", "024 bowl", "high"),
    30: (9,  "scissors", "037 scissors", "high"),
    33: (20, "flat_screwdriver", "044 flat screwdriver", "high"),
    34: (19, "phillips_screwdriver", "043 phillips screwdriver", "high"),
    39: (8,  "power_drill", "035 power drill", "high"),
    40: (32, "padlock", "038 padlock", "high"),

    # GraspNet-collected objects
    80: (47, "logitech_M330_mouse", "white mouse", "high"),
    81: (60, "Logitech M170 mouse", "black mouse", "high"),
    82: (42, "Nivea Men Oil Control Face Wash", "nivea men oil control", "high"),
    83: (62, "Pantene", "pantene", "high"),
    84: (63, "Headshoulders Supreme", "head shoulders supreme", "high"),
    85: (64, "Thera-Med toothpaste", "thera med", "high"),
    86: (65, "Dove body wash", "dove", "high"),
    87: (45, "pitcher_cap", "pitcher cap", "high"),
    88: (70, "3M double sided tape", "tape", "high"),
    89: (74, "Mastrad icecube tray", "ice cube mould", "high"),
    90: (37, "Mouth Rinse", "nzskincare mouth rinse", "high"),
    91: (59, "SafeGuard Soap box", "soap", "medium"),
    92: (50, "Schleich Wild Life Starter", "zebra", "collective"),
    93: (53, "Schleich Wild Life Starter", "small elephant", "collective"),
    94: (54, "Schleich Wild Life Starter", "monkey", "collective"),
    95: (67, "Schleich Wild Life Starter", "lion", "collective"),
    96: (33, "dragon", "dragon", "high"),
    97: (39, "SafeGuard Soap box", "soap box", "high"),
    98: (58, "Toothpaste", "darlie box", "medium"),

    # DexNet adversarial / 3D-print objects
    99:  (75, "Bar clamp", "bar clamp", "high"),
    100: (76, "Climbing hold", "climbing hold", "high"),
    101: (87, "Vase", "vase", "high"),
    102: (81, "Nozzel", "nozzle", "high"),
    103: (84, "Pawn", "pawn", "high"),
    104: (77, "end tip holder", "endstop holder", "high"),
    105: (86, "Turbin housing", "turbine housing", "high"),
    106: (85, "Pipe connector", "pipe connector", "high"),
    107: (78, "Gearbox", "gearbox", "high"),
    108: (82, "Connect part 1", "part1", "high"),
    109: (83, "Connect part 2", "part3", "medium"),
    110: (79, "Mount 1", "mount1", "high"),
    111: (80, "Mount 2", "mount2", "high"),
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out))


def read_gc6d_split_scene_ids(gc6d_root: Path, split: str) -> List[int]:
    split = str(split).lower()
    split_dir = gc6d_root / "split_info"
    if split == "train":
        path = split_dir / "grasp_train_scene_ids.json"
    elif split == "test":
        path = split_dir / "grasp_test_scene_ids.json"
    else:
        return parse_int_list(split)

    data = load_json(path)
    if isinstance(data, dict):
        vals = list(data.keys()) if all(str(k).isdigit() for k in data.keys()) else list(data.values())
    else:
        vals = list(data)
    return sorted({int(x) for x in vals})


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def format_scene_ids_6d(scene_ids):
    """Return scene ids in the official GC6D split JSON format, e.g. "000005"."""
    return [f"{int(sid):06d}" for sid in scene_ids]

@dataclass(frozen=True)
class MappingEntry:
    gc6d_obj_id: int
    graspnet_obj_id: int
    gc6d_name: str = ""
    graspnet_name: str = ""
    confidence: str = "unknown"


def norm_header(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def pick_col(headers: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {norm_header(h): h for h in headers}
    for c in candidates:
        c_norm = norm_header(c)
        if c_norm in normalized:
            return normalized[c_norm]
    for h in headers:
        hn = norm_header(h)
        if any(norm_header(c) in hn for c in candidates):
            return h
    return None


def load_mapping(mapping_csv: Optional[Path], allowed_confidences: Set[str]) -> Dict[int, MappingEntry]:
    if mapping_csv is None:
        entries = {}
        for gc6d_id, (gn_id, gc6d_name, gn_name, conf) in BUILTIN_GC6D_TO_GRASPNET.items():
            if conf.lower() in allowed_confidences:
                entries[int(gc6d_id)] = MappingEntry(int(gc6d_id), int(gn_id), gc6d_name, gn_name, conf)
        return entries

    rows: List[Dict[str, str]] = []
    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty mapping CSV: {mapping_csv}")

    headers = list(rows[0].keys())
    gc_col = pick_col(headers, ["gc6d_id", "gc6d obj id", "gc6d object id", "gc6d id", "gc6d_obj_id"])
    gn_col = pick_col(headers, ["graspnet_id", "graspnet obj id", "graspnet object id", "gn_id", "graspnet id"])
    gc_name_col = pick_col(headers, ["gc6d_name", "gc6d object name", "gc6d name"])
    gn_name_col = pick_col(headers, ["graspnet_name", "graspnet object name", "graspnet name"])
    conf_col = pick_col(headers, ["confidence", "match_confidence", "match confidence"])
    include_col = pick_col(headers, ["include", "include in object level overlap", "include in object-level overlap"])

    if gc_col is None or gn_col is None:
        raise ValueError(
            f"Cannot find GC6D/GraspNet id columns in {mapping_csv}. Headers={headers}"
        )

    entries: Dict[int, MappingEntry] = {}
    for row in rows:
        if include_col is not None:
            include_val = str(row.get(include_col, "")).strip().lower()
            if include_val in {"0", "false", "no", "n", "exclude"}:
                continue
        conf = str(row.get(conf_col, "unknown") if conf_col else "unknown").strip().lower() or "unknown"
        if allowed_confidences and conf not in allowed_confidences:
            continue
        try:
            gc6d_id = int(float(str(row[gc_col]).strip()))
            gn_id = int(float(str(row[gn_col]).strip()))
        except Exception:
            continue
        entries[gc6d_id] = MappingEntry(
            gc6d_obj_id=gc6d_id,
            graspnet_obj_id=gn_id,
            gc6d_name=str(row.get(gc_name_col, "") if gc_name_col else ""),
            graspnet_name=str(row.get(gn_name_col, "") if gn_name_col else ""),
            confidence=conf,
        )
    return entries


def find_graspnet_scene_dir(graspnet_root: Path, scene_id: int) -> Optional[Path]:
    candidates = [
        graspnet_root / "scenes" / f"scene_{scene_id:04d}",
        graspnet_root / "scenes" / f"scene_{scene_id:06d}",
        graspnet_root / f"scene_{scene_id:04d}",
        graspnet_root / f"scene_{scene_id:06d}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_camera_dirs(scene_dir: Path, camera: str) -> List[Path]:
    camera = str(camera).lower()
    if camera != "all":
        aliases = {
            "realsense": ["realsense", "realsense-d435", "realsense-d415"],
            "kinect": ["kinect", "azure-kinect"],
        }
        names = aliases.get(camera, [camera])
        out = [scene_dir / n for n in names if (scene_dir / n).exists()]
        if not out and any((scene_dir / d).exists() for d in ["meta", "scene_gt.json", "annotations"]):
            out = [scene_dir]
        return out

    out: List[Path] = []
    for child in scene_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "meta").exists() or (child / "scene_gt.json").exists() or (child / "annotations").exists():
            out.append(child)
    if (scene_dir / "meta").exists() or (scene_dir / "scene_gt.json").exists() or (scene_dir / "annotations").exists():
        out.append(scene_dir)
    return sorted(out)


def ints_from_array(arr) -> Set[int]:
    import numpy as np

    a = np.asarray(arr)
    if a.size == 0:
        return set()
    a = np.squeeze(a)
    vals: Set[int] = set()
    for x in a.reshape(-1):
        try:
            if np.isfinite(x):
                vals.add(int(x))
        except Exception:
            pass
    return vals


def load_mat_obj_ids(path: Path) -> Set[int]:
    candidate_keys = [
        "cls_indexes", "cls_index", "class_ids", "class_id", "classes",
        "obj_ids", "obj_id", "object_ids", "object_id",
    ]
    try:
        import scipy.io as sio
        data = sio.loadmat(str(path))
        ids: Set[int] = set()
        for k in candidate_keys:
            if k in data:
                ids.update(ints_from_array(data[k]))
        return ids
    except NotImplementedError:
        pass
    except Exception:
        # Fall through to h5py for v7.3 mat or return empty if h5py also fails.
        pass

    try:
        import h5py
        ids: Set[int] = set()
        with h5py.File(str(path), "r") as f:
            for k in candidate_keys:
                if k in f:
                    ids.update(ints_from_array(f[k][()]))
        return ids
    except Exception:
        return set()


def load_scene_gt_obj_ids(path: Path, max_items: Optional[int] = None) -> Set[int]:
    data = load_json(path)
    ids: Set[int] = set()
    n_items = 0
    if isinstance(data, dict):
        iterable = data.values()
    elif isinstance(data, list):
        iterable = data
    else:
        return ids

    for item in iterable:
        # BOP scene_gt: dict image_id -> list[dict]
        if isinstance(item, list):
            for obj in item:
                if isinstance(obj, dict) and "obj_id" in obj:
                    ids.add(int(obj["obj_id"]))
                    n_items += 1
        elif isinstance(item, dict):
            if "obj_id" in item:
                ids.add(int(item["obj_id"]))
                n_items += 1
            else:
                # Some JSONs nest objects under a key like "objects".
                for v in item.values():
                    if isinstance(v, list):
                        for obj in v:
                            if isinstance(obj, dict) and "obj_id" in obj:
                                ids.add(int(obj["obj_id"]))
                                n_items += 1
        if max_items is not None and n_items >= max_items:
            break
    return ids


def load_xml_obj_ids(path: Path) -> Set[int]:
    ids: Set[int] = set()
    try:
        root = ET.parse(str(path)).getroot()
    except Exception:
        return ids

    interesting = {"obj_id", "object_id", "class_id", "cls_id", "cls_index", "cls_indexes"}
    for elem in root.iter():
        tag = str(elem.tag).strip().lower()
        text = (elem.text or "").strip()
        if tag in interesting and text:
            for m in re.finditer(r"-?\d+", text):
                ids.add(int(m.group(0)))
        for k, v in elem.attrib.items():
            kk = str(k).strip().lower()
            if kk in interesting:
                for m in re.finditer(r"-?\d+", str(v)):
                    ids.add(int(m.group(0)))
    return ids


def collect_graspnet_scene_raw_obj_ids(
    graspnet_root: Path,
    scene_id: int,
    camera: str,
    scan_policy: str = "first",
) -> Tuple[Set[int], List[str]]:
    scene_dir = find_graspnet_scene_dir(graspnet_root, scene_id)
    warnings: List[str] = []
    if scene_dir is None:
        return set(), [f"missing_scene_dir:{scene_id}"]

    cam_dirs = find_camera_dirs(scene_dir, camera)
    if not cam_dirs:
        return set(), [f"missing_camera_dir:{scene_dir}:{camera}"]

    ids: Set[int] = set()
    for cam_dir in cam_dirs:
        # BOP-style scene_gt.json.
        for gt_path in [cam_dir / "scene_gt.json", scene_dir / "scene_gt.json"]:
            if gt_path.exists():
                ids.update(load_scene_gt_obj_ids(gt_path, max_items=None if scan_policy == "all" else 200))
                if ids and scan_policy == "first":
                    return ids, warnings

        # GraspNet-style meta/*.mat.
        meta_dir = cam_dir / "meta"
        if meta_dir.exists():
            mats = sorted(meta_dir.glob("*.mat"))
            if scan_policy == "first":
                mats = mats[:1]
            for mat_path in mats:
                ids.update(load_mat_obj_ids(mat_path))
            if ids and scan_policy == "first":
                return ids, warnings

        # XML fallback.
        ann_dir = cam_dir / "annotations"
        if ann_dir.exists():
            xmls = sorted(ann_dir.glob("*.xml"))
            if scan_policy == "first":
                xmls = xmls[:1]
            for xml_path in xmls:
                ids.update(load_xml_obj_ids(xml_path))
            if ids and scan_policy == "first":
                return ids, warnings

    if not ids:
        warnings.append(f"no_obj_ids_found:{scene_dir}")
    return ids, warnings


def infer_graspnet_obj_id_offset(graspnet_root: Path, raw_ids: Set[int], setting: str) -> Tuple[int, str]:
    s = str(setting).strip().lower()
    if s in {"0", "+0", "none"}:
        return 0, "explicit"
    if s in {"-1", "minus1", "one_based_to_zero_based"}:
        return -1, "explicit"
    if s != "auto":
        raise ValueError(f"Unknown --graspnet_obj_id_offset={setting}")

    if not raw_ids:
        return 0, "auto_empty_raw_ids_default_0"
    if 0 in raw_ids:
        return 0, "auto_raw_ids_contain_0"
    if max(raw_ids) > 87 and max(raw_ids) <= 88 and min(raw_ids) >= 1:
        return -1, "auto_raw_ids_contain_88_assume_1_based"

    model_roots = [graspnet_root / "models", graspnet_root / "models_m"]
    for mr in model_roots:
        if not mr.exists():
            continue
        zero_markers = [mr / "000", mr / "000" / "nontextured.ply", mr / "obj_000000.ply"]
        one_markers = [mr / "001", mr / "001" / "nontextured.ply", mr / "obj_000001.ply"]
        has_zero = any(p.exists() for p in zero_markers)
        has_one = any(p.exists() for p in one_markers)
        if has_zero:
            return 0, f"auto_model_root_has_zero_based_marker:{mr}"
        if has_one:
            return -1, f"auto_model_root_has_one_based_marker:{mr}"

    return 0, "auto_ambiguous_default_0"


def normalize_ids(raw_ids: Iterable[int], offset: int) -> Set[int]:
    out = {int(x) + int(offset) for x in raw_ids}
    return {x for x in out if x >= 0}


def collect_graspnet_test_objects(
    graspnet_root: Path,
    split_to_scene_ids: Dict[str, List[int]],
    camera: str,
    scan_policy: str,
    obj_id_offset_setting: str,
) -> Tuple[Dict[int, Set[str]], List[Dict[str, object]], Dict[str, object]]:
    raw_by_scene: List[Dict[str, object]] = []
    all_raw_ids: Set[int] = set()

    for split_name, scene_ids in split_to_scene_ids.items():
        for scene_id in scene_ids:
            raw_ids, warnings = collect_graspnet_scene_raw_obj_ids(
                graspnet_root, scene_id, camera=camera, scan_policy=scan_policy
            )
            all_raw_ids.update(raw_ids)
            raw_by_scene.append({
                "split": split_name,
                "scene_id": int(scene_id),
                "raw_obj_ids": sorted(raw_ids),
                "warnings": ";".join(warnings),
            })

    offset, offset_reason = infer_graspnet_obj_id_offset(graspnet_root, all_raw_ids, obj_id_offset_setting)

    split_by_obj: Dict[int, Set[str]] = defaultdict(set)
    library_rows: List[Dict[str, object]] = []
    for row in raw_by_scene:
        norm_ids = sorted(normalize_ids(row["raw_obj_ids"], offset))
        row["norm_obj_ids"] = norm_ids
        for oid in norm_ids:
            split_by_obj[int(oid)].add(str(row["split"]))

    for oid in sorted(split_by_obj.keys()):
        library_rows.append({
            "graspnet_obj_id": int(oid),
            "test_splits": ";".join(sorted(split_by_obj[oid])),
        })

    meta = {
        "raw_obj_id_min": min(all_raw_ids) if all_raw_ids else None,
        "raw_obj_id_max": max(all_raw_ids) if all_raw_ids else None,
        "raw_obj_id_count": len(all_raw_ids),
        "normalized_obj_id_count": len(split_by_obj),
        "graspnet_obj_id_offset": offset,
        "graspnet_obj_id_offset_reason": offset_reason,
        "scene_scan_policy": scan_policy,
        "camera": camera,
        "scene_rows": raw_by_scene,
    }
    return split_by_obj, library_rows, meta


def collect_gc6d_scene_obj_ids(gc6d_root: Path, scene_id: int) -> Set[int]:
    scene_dir = gc6d_root / "scenes" / f"{int(scene_id):06d}"
    gt_path = scene_dir / "scene_gt.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GC6D scene_gt.json: {gt_path}")
    return load_scene_gt_obj_ids(gt_path, max_items=None)


def split_arg_to_scene_ids(graspnet_splits: str, scene_ids_override: Optional[str]) -> Dict[str, List[int]]:
    if scene_ids_override:
        return {"custom": parse_int_list(scene_ids_override)}
    out: Dict[str, List[int]] = {}
    for s in str(graspnet_splits).split(","):
        s = s.strip()
        if not s:
            continue
        if s not in GRASPNET_DEFAULT_SPLITS:
            raise ValueError(f"Unknown GraspNet split={s}. Expected {sorted(GRASPNET_DEFAULT_SPLITS)} or use --graspnet_scene_ids.")
        out[s] = list(GRASPNET_DEFAULT_SPLITS[s])
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        "Filter GC6D train scenes by object overlap with GraspNet test_similar/test_novel."
    )
    p.add_argument("--graspnet_root", default='/data/robotarm/dataset/graspnet', help="Root of GraspNet-1Billion dataset.")
    p.add_argument("--gc6d_root", default='/data2/robotarm/dataset/GraspClutter6D', help="Root of GraspClutter6D dataset.")
    p.add_argument("--gc6d_split", default="train", help="Usually 'train'. Can also be explicit ids like '1,2,5-10'.")
    p.add_argument("--graspnet_splits", default="test_similar,test_novel", help="Comma-separated: test_seen,test_similar,test_novel.")
    p.add_argument("--graspnet_scene_ids", default=None, help="Override --graspnet_splits, e.g. '130-189'.")
    p.add_argument("--graspnet_camera", default="realsense", help="realsense, kinect, all, or an exact camera folder name.")
    p.add_argument("--graspnet_scan_policy", default="first", choices=["first", "all"],
                   help="Read the first metadata file per scene/camera or all metadata files. 'all' is safer but slower.")
    p.add_argument("--graspnet_obj_id_offset", default="-1", choices=["auto", "0", "-1"],
                   help="Normalize GraspNet raw object ids to 0-based object-list ids. Use -1 if meta cls_indexes are 1-based.")
    p.add_argument("--mapping_csv", default=None,
                   help="Optional GC6D->GraspNet mapping CSV. If omitted, use built-in mapping from prior overlap audit.")
    p.add_argument("--mapping_confidence", default="high,medium,collective",
                   help="Comma-separated confidence labels to include from the mapping.")
    p.add_argument("--output_dir", default=None,
                   help="Default: GC6D_ROOT/split_info/no_graspnet_test_overlap")
    p.add_argument("--dry_run", action="store_true", help="Only print summary; still writes CSV/JSON reports unless --no_write_reports.")
    p.add_argument("--no_write_reports", action="store_true")
    args = p.parse_args()

    graspnet_root = Path(args.graspnet_root).expanduser().resolve()
    gc6d_root = Path(args.gc6d_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else gc6d_root / "split_info" / "no_graspnet_test_overlap"

    allowed_conf = {x.strip().lower() for x in str(args.mapping_confidence).split(",") if x.strip()}
    mapping = load_mapping(Path(args.mapping_csv).expanduser().resolve() if args.mapping_csv else None, allowed_conf)
    if not mapping:
        raise RuntimeError("No GC6D->GraspNet mapping entries after confidence/include filtering.")

    split_to_scene_ids = split_arg_to_scene_ids(args.graspnet_splits, args.graspnet_scene_ids)
    gn_split_by_obj, gn_library_rows, gn_meta = collect_graspnet_test_objects(
        graspnet_root=graspnet_root,
        split_to_scene_ids=split_to_scene_ids,
        camera=args.graspnet_camera,
        scan_policy=args.graspnet_scan_policy,
        obj_id_offset_setting=args.graspnet_obj_id_offset,
    )
    gn_test_obj_ids = set(gn_split_by_obj.keys())

    train_scene_ids = read_gc6d_split_scene_ids(gc6d_root, args.gc6d_split)
    kept_scene_ids: List[int] = []
    excluded_scene_ids: List[int] = []
    scene_rows: List[Dict[str, object]] = []
    conflict_rows: List[Dict[str, object]] = []

    for scene_id in train_scene_ids:
        try:
            gc6d_obj_ids = collect_gc6d_scene_obj_ids(gc6d_root, scene_id)
            warning = ""
        except Exception as e:
            gc6d_obj_ids = set()
            warning = repr(e)

        conflicts: List[Tuple[int, MappingEntry]] = []
        for gc6d_oid in sorted(gc6d_obj_ids):
            ent = mapping.get(int(gc6d_oid))
            if ent is None:
                continue
            if ent.graspnet_obj_id in gn_test_obj_ids:
                conflicts.append((int(gc6d_oid), ent))

        if conflicts:
            excluded_scene_ids.append(int(scene_id))
        else:
            kept_scene_ids.append(int(scene_id))

        scene_rows.append({
            "scene_id": int(scene_id),
            "exclude": bool(conflicts),
            "num_gc6d_objs": len(gc6d_obj_ids),
            "gc6d_obj_ids": ";".join(map(str, sorted(gc6d_obj_ids))),
            "conflict_gc6d_obj_ids": ";".join(str(x[0]) for x in conflicts),
            "conflict_graspnet_obj_ids": ";".join(str(x[1].graspnet_obj_id) for x in conflicts),
            "conflict_graspnet_splits": ";".join(sorted({s for _, ent in conflicts for s in gn_split_by_obj[ent.graspnet_obj_id]})),
            "warning": warning,
        })

        for gc6d_oid, ent in conflicts:
            conflict_rows.append({
                "scene_id": int(scene_id),
                "gc6d_obj_id": int(gc6d_oid),
                "gc6d_name": ent.gc6d_name,
                "graspnet_obj_id": ent.graspnet_obj_id,
                "graspnet_name": ent.graspnet_name,
                "mapping_confidence": ent.confidence,
                "graspnet_test_splits": ";".join(sorted(gn_split_by_obj[ent.graspnet_obj_id])),
            })

    target_mapping_rows: List[Dict[str, object]] = []
    for ent in sorted(mapping.values(), key=lambda x: (x.graspnet_obj_id, x.gc6d_obj_id)):
        if ent.graspnet_obj_id in gn_test_obj_ids:
            target_mapping_rows.append({
                "gc6d_obj_id": ent.gc6d_obj_id,
                "gc6d_name": ent.gc6d_name,
                "graspnet_obj_id": ent.graspnet_obj_id,
                "graspnet_name": ent.graspnet_name,
                "mapping_confidence": ent.confidence,
                "graspnet_test_splits": ";".join(sorted(gn_split_by_obj[ent.graspnet_obj_id])),
            })

    summary = {
        "graspnet_root": str(graspnet_root),
        "gc6d_root": str(gc6d_root),
        "gc6d_split": args.gc6d_split,
        "graspnet_splits": split_to_scene_ids,
        "graspnet_camera": args.graspnet_camera,
        "graspnet_obj_id_offset": gn_meta["graspnet_obj_id_offset"],
        "graspnet_obj_id_offset_reason": gn_meta["graspnet_obj_id_offset_reason"],
        "graspnet_test_obj_count": len(gn_test_obj_ids),
        "mapping_entry_count": len(mapping),
        "mapped_gc6d_objects_conflicting_with_gn_test_count": len(target_mapping_rows),
        "official_gc6d_train_scene_count": len(train_scene_ids),
        "kept_scene_count": len(kept_scene_ids),
        "excluded_scene_count": len(excluded_scene_ids),
        "kept_scene_ids": kept_scene_ids,
        "excluded_scene_ids": excluded_scene_ids,
        "output_dir": str(output_dir),
    }

    print("[Summary]")
    print(f"  GraspNet scene splits: { {k: [min(v), max(v), len(v)] for k, v in split_to_scene_ids.items()} }")
    print(f"  GraspNet raw obj id min/max/count: {gn_meta['raw_obj_id_min']} / {gn_meta['raw_obj_id_max']} / {gn_meta['raw_obj_id_count']}")
    print(f"  GraspNet id offset: {gn_meta['graspnet_obj_id_offset']} ({gn_meta['graspnet_obj_id_offset_reason']})")
    print(f"  GraspNet normalized test object count: {len(gn_test_obj_ids)}")
    print(f"  Mapping entries used: {len(mapping)}; target mapped GC6D objects: {len(target_mapping_rows)}")
    print(f"  GC6D train scenes: {len(train_scene_ids)}")
    print(f"  Kept: {len(kept_scene_ids)}")
    print(f"  Excluded: {len(excluded_scene_ids)}")
    if excluded_scene_ids:
        print(f"  First excluded scenes: {excluded_scene_ids[:20]}{' ...' if len(excluded_scene_ids) > 20 else ''}")

    if args.no_write_reports:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "graspnet_test_object_library.csv", gn_library_rows)
    write_csv(output_dir / "graspnet_test_scene_object_library.csv", gn_meta["scene_rows"])
    write_csv(output_dir / "gc6d_train_scene_filter_report.csv", scene_rows)
    write_csv(output_dir / "gc6d_conflict_object_mapping.csv", conflict_rows)
    write_csv(output_dir / "gc6d_mapped_objects_present_in_graspnet_test.csv", target_mapping_rows)
    save_json(output_dir / "filter_summary.json", summary)
    save_json(output_dir / "grasp_train_scene_ids_no_graspnet_test_overlap.json",  format_scene_ids_6d(kept_scene_ids))
    save_json(output_dir / "grasp_train_scene_ids_excluded_by_graspnet_test_overlap.json", format_scene_ids_6d(excluded_scene_ids))

    # Convenience files for scripts that accept --scene_ids as a comma-separated list
    # but do not directly read a JSON split path.
    (output_dir / "grasp_train_scene_ids_no_graspnet_test_overlap_arg.txt").write_text(
        ",".join(map(str, kept_scene_ids)) + "\n", encoding="utf-8"
    )
    (output_dir / "grasp_train_scene_ids_excluded_by_graspnet_test_overlap_arg.txt").write_text(
        ",".join(map(str, excluded_scene_ids)) + "\n", encoding="utf-8"
    )

    print(f"[Wrote] {output_dir}")
    if args.dry_run:
        print("[Dry-run note] Reports were written, but this script never modifies official GC6D split files in place.")


if __name__ == "__main__":
    main()
