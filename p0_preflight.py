#!/usr/bin/env python3
"""Preflight current repository/checkpoints before running PKD P0 experiments."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from pkd_p0.common import atomic_json_dump, load_current_checkpoint


P0_RUNTIME_SYMBOLS: Set[str] = {
    "economicgrasp_dpt_p0_student",
    "economicgrasp_dpt_p0_teacher",
    "economicgrasp_dpt_student",
    "economicgrasp_dpt_teacher",
    "forward_with_p0_geometry_override",
    "enable_p0_exact_query_runtime",
    "extract_p0_query_contract",
    "build_p0_exact_query_input",
    "assert_p0_exact_query_output",
    "runtime_contract",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student_checkpoint", required=True)
    p.add_argument("--teacher_checkpoint", required=True)
    p.add_argument("--uniform_checkpoint", default="")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--output_json", default="pkd_p0_preflight.json")
    return p.parse_args()


def python_symbols(path: Path) -> Dict[str, List[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.append(target.id)
    return {
        "functions": [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ],
        "classes": [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
        "assignments": assignments,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).expanduser().resolve()
    runtime_file = root / "models" / "economicgrasp_dpt_p0.py"
    required_files = [
        root / "models" / "economicgrasp_dpt_distill.py",
        root / "models" / "economicgrasp_bip3d.py",
        root / "models" / "kview_query_transformer.py",
        runtime_file,
        root / "dataset" / "graspnet_dataset.py",
        root / "inference_cva_distill.py",
        root / "eval.py",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing current repository files: {missing}")

    core_sources = [
        root / "models" / "economicgrasp_bip3d.py",
        root / "models" / "kview_query_transformer.py",
    ]
    legacy_patch_markers = {
        str(path): [
            marker
            for marker in (
                "PKD_P0_GEOMETRY_OVERRIDE_V1",
                "PKD_P0_QUERY_OVERRIDE_V1",
            )
            if marker in path.read_text(encoding="utf-8")
        ]
        for path in core_sources
    }
    legacy_patch_markers = {
        path: markers
        for path, markers in legacy_patch_markers.items()
        if markers
    }

    _, student = load_current_checkpoint(args.student_checkpoint, expected_stage=1)
    _, teacher = load_current_checkpoint(args.teacher_checkpoint, expected_stage=0)
    uniform = None
    if args.uniform_checkpoint:
        _, uniform = load_current_checkpoint(args.uniform_checkpoint)

    runtime_symbols = python_symbols(runtime_file)
    runtime_found = (
        set(runtime_symbols["functions"])
        | set(runtime_symbols["classes"])
        | set(runtime_symbols["assignments"])
    )
    missing_runtime_symbols = sorted(P0_RUNTIME_SYMBOLS - runtime_found)

    diagnostic_scripts = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if any(
            token in path.name.lower()
            for token in ("diagnos", "oracle", "privileged", "distill", "p0")
        )
    )
    payload: Dict[str, Any] = {
        "repo_root": str(root),
        "required_files": [str(path) for path in required_files],
        "student": student.to_dict(),
        "teacher": teacher.to_dict(),
        "uniform": None if uniform is None else uniform.to_dict(),
        "p0_runtime": {
            "path": str(runtime_file),
            "source_patch_required": False,
            "source_files_modified": [],
            "legacy_installed_markers": legacy_patch_markers,
            "required_symbols": sorted(P0_RUNTIME_SYMBOLS),
            "missing_symbols": missing_runtime_symbols,
            "query_semantics": (
                "same ordered image-FPS pixels and selected view indices; "
                "teacher center XYZ is recomputed from teacher geometry"
            ),
        },
        "python_symbols": {
            str(path.relative_to(root)): python_symbols(path)
            for path in required_files
            if path.suffix == ".py"
        },
        "related_repository_scripts": diagnostic_scripts,
        "checks": {
            "dedicated_runtime_complete": not missing_runtime_symbols,
            "core_sources_unmodified_by_legacy_installer": not legacy_patch_markers,
            "student_teacher_same_seed_mode": (
                student.seed_selection_mode == teacher.seed_selection_mode == "image_fps"
            ),
            "student_current_pred_geometry": student.geometry_depth_source == "pred",
            "teacher_current_gt_geometry": teacher.geometry_depth_source == "gt",
            "legacy_use_gt_depth_disabled": (
                not student.legacy_dataset_use_gt_depth
                and not teacher.legacy_dataset_use_gt_depth
            ),
            "contract_version_current": (
                student.contract_version == teacher.contract_version == 2
            ),
            "cdf_shape_compatible": (
                student.feature_dim == teacher.feature_dim
                and student.num_depths == teacher.num_depths
                and student.num_thresholds == teacher.num_thresholds
            ),
        },
    }
    failed = [key for key, value in payload["checks"].items() if not value]
    payload["status"] = "pass" if not failed else "fail"
    payload["failed_checks"] = failed
    atomic_json_dump(payload, args.output_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
