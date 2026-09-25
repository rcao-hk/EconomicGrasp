#!/usr/bin/env python3
"""CPU/CUDA checks of the actual deterministic CDF view inverse helper.

AST loading avoids importing the dataset argument parser and CUDA KNN extension;
the function under test is compiled directly from utils/label_generation.py.
"""
import ast
import json
from pathlib import Path

import torch


def load_helper():
    path = Path(__file__).resolve().parents[1] / "utils" / "label_generation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_deterministic_top_view_scene")
    namespace = {"torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function.name]


def cpu_reference(view_inds, top_view_index):
    """Explicit row/slot/ascending-scene loop, with last write winning."""
    result = torch.full_like(top_view_index.cpu(), -1)
    for row in range(top_view_index.shape[0]):
        for slot in range(top_view_index.shape[1]):
            object_view = int(top_view_index[row, slot])
            for scene_view, matched in enumerate(view_inds.cpu().tolist()):
                if matched == object_view:
                    result[row, slot] = scene_view
    return result


def old_inverse(view_inds, top_view_index):
    result = -torch.ones_like(top_view_index)
    row, slot, scene = torch.where(view_inds == top_view_index.unsqueeze(-1))
    result[row, slot] = scene
    return result


def main():
    inverse = load_helper()
    torch.manual_seed(1703)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    one_to_one = torch.tensor([2, 0, 3, 1])
    unambiguous_slots = torch.tensor([[0, 1, 2, 3, -1, 4], [3, 2, 1, 0, -1, 5]])
    many_to_one = torch.tensor([0, 1, 1, 1, 4, 4, 6, 6])
    ambiguous_slots = torch.randint(-1, 10, (128, 16))
    ambiguous_slots[0, :8] = torch.arange(8)  # includes missing views 2,3,5,7
    expected = cpu_reference(many_to_one, ambiguous_slots)
    expected_unambiguous = old_inverse(one_to_one, unambiguous_slots)
    rows = []
    prior_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for device in devices:
            v1, t1 = one_to_one.to(device), unambiguous_slots.to(device)
            assert torch.equal(inverse(v1, t1).cpu(), expected_unambiguous)
            vm, tm = many_to_one.to(device), ambiguous_slots.to(device)
            vm_before, tm_before = vm.clone(), tm.clone()
            for _ in range(32):
                assert torch.equal(inverse(vm, tm).cpu(), expected)
            assert torch.equal(vm, vm_before) and torch.equal(tm, tm_before)
            assert inverse(vm, tm[:0]).shape == tm[:0].shape
            assert inverse(vm, torch.tensor([[-1, 8, 99]], device=device)).tolist() == [[-1, -1, -1]]
            rows.append({"device": device, "one_to_one_old_parity": True,
                         "many_to_one_cpu_loop_parity": True, "repeat_count": 32})
        try:
            inverse(torch.empty(0, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long))
        except ValueError:
            pass
        else:
            raise AssertionError("Empty inverse view map accepted")
    finally:
        torch.use_deterministic_algorithms(prior_determinism)
    print(json.dumps({"status": "passed", "tests": rows,
                      "cuda_available": torch.cuda.is_available()}, indent=2))


if __name__ == "__main__":
    main()
