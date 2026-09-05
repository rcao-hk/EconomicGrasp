"""Compatibility entry point for the first P0 overlay revision.

The first revision edited model source files to add two runtime hooks.  The
current revision no longer patches any repository file.  P0 controls live in
``models/economicgrasp_dpt_p0.py`` and are attached only to model instances in
memory.

Keeping this module means an old command such as

    python -m pkd_p0.install_runtime_hooks --dry_run 1

now performs a static availability check and exits successfully without writing
anything.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Set


REQUIRED_SYMBOLS = {
    "economicgrasp_dpt_p0_student",
    "economicgrasp_dpt_p0_teacher",
    "runtime_contract",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Retain old flags so existing shell commands do not fail.
    parser.add_argument("--model_file", default="models/economicgrasp_dpt_p0.py")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--dry_run", type=int, choices=(0, 1), default=1)
    parser.add_argument("--install_query_override", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def top_level_symbols(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def main() -> None:
    args = parse_args()
    path = Path(args.model_file).expanduser()
    if path.name != "economicgrasp_dpt_p0.py":
        # Old callers usually pass economicgrasp_dpt_distill.py. Resolve the
        # dedicated module beside it rather than editing the requested file.
        path = Path(args.models_dir).expanduser() / "economicgrasp_dpt_p0.py"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dedicated P0 runtime module: {path}. Copy "
            "models/economicgrasp_dpt_p0.py into the repository root first."
        )
    symbols = top_level_symbols(path)
    missing = sorted(REQUIRED_SYMBOLS - symbols)
    if missing:
        raise RuntimeError(
            f"{path} is incomplete; missing P0 runtime symbols: {missing}"
        )
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"[OK] dedicated P0 runtime module: {path}")
    print("[OK] source_patch_required=False")
    print("[OK] economicgrasp_bip3d.py and kview_query_transformer.py were not modified")
    print("[CHECK-ONLY] no installation was performed")


if __name__ == "__main__":
    main()
