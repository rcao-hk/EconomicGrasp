#!/usr/bin/env python3
"""Single-process P0-C2 entry point for module-wise PKD gradient audit.

Launch with the same Stage-2 Uniform Exact-Query arguments as
``train_cva_distill_ddp.py``, but use one visible GPU / WORLD_SIZE=1.  The
wrapper compiles an in-memory copy of ``Trainer.train_one_epoch`` with one audit
call immediately before its ordinary ``loss.backward()``.  No repository source
file is edited.
"""
from __future__ import annotations

import inspect
import os
import textwrap
from typing import List

from pkd_p0.gradient_audit import (
    maybe_audit_from_training_locals as _pkd_p0_grad_audit,
)

# This import intentionally receives the original training CLI arguments.
import train_cva_distill_ddp as _base


_INSERTED_MARKER = "# PKD_P0_IN_MEMORY_GRADIENT_AUDIT_V1_1"


def _assert_single_process() -> None:
    if not os.environ.get("PKD_P0_GRAD_AUDIT_DIR", "").strip():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "P0-C2 parameter-gradient audit is single-process by design. "
            "Set GPU_IDS=0 so torchrun uses --nproc_per_node=1.  DDP does not "
            "support torch.autograd.grad() for model parameters."
        )


def _instrument_train_one_epoch() -> None:
    original = _base.Trainer.train_one_epoch
    source = textwrap.dedent(inspect.getsource(original))
    lines = source.splitlines(keepends=True)
    candidates: List[int] = []
    for index, line in enumerate(lines):
        if line.strip() == "loss.backward()":
            candidates.append(index)
    if len(candidates) != 1:
        raise RuntimeError(
            "The current Trainer.train_one_epoch must contain exactly one "
            f"active 'loss.backward()' line; found {len(candidates)}."
        )

    index = candidates[0]
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines.insert(
        index,
        f"{indent}{_INSERTED_MARKER}\n"
        f"{indent}_pkd_p0_grad_audit(locals())\n",
    )
    instrumented_source = "".join(lines)

    compile_globals = dict(_base.__dict__)
    compile_globals["_pkd_p0_grad_audit"] = _pkd_p0_grad_audit
    compile_locals = {}
    code = compile(
        instrumented_source,
        f"{inspect.getsourcefile(original)}::<P0-in-memory-audit-v1.1>",
        "exec",
    )
    exec(code, compile_globals, compile_locals)
    replacement = compile_locals.get("train_one_epoch")
    if replacement is None:
        replacement = compile_globals.get("train_one_epoch")
    if not callable(replacement):
        raise RuntimeError("Failed to compile instrumented train_one_epoch.")

    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__
    replacement.__doc__ = original.__doc__
    _base.Trainer.train_one_epoch = replacement
    print(
        "[PKD-P0-GRAD] Installed in-memory P0-C2 audit v1.1; "
        "model source will be self.unwrap_model(); no source file modified.",
        flush=True,
    )


def main() -> None:
    _assert_single_process()
    _instrument_train_one_epoch()
    _base.main()


if __name__ == "__main__":
    main()
