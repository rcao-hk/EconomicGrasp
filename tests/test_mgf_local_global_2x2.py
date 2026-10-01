"""CPU contracts for the local-selection x global-scoring 2x2 diagnostic."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mgf_local_global_2x2 import CONDITIONS, _decode_local_global


def _fake_query_module(monkeypatch):
    fake = types.ModuleType("models.economicgrasp_bip3d")

    def query_idx(end_points, batch_i, total_q, use_top4_view):
        assert use_top4_view is False
        return torch.arange(total_q)

    fake._cva_decode_query_indices = query_idx
    monkeypatch.setitem(sys.modules, "models.economicgrasp_bip3d", fake)


def _rotation(approach, angle):
    n = approach.shape[0]
    eye = torch.eye(3, dtype=approach.dtype, device=approach.device)
    return eye[None].expand(n, -1, -1).clone()


def _fixture():
    B, T, Q, A, D = 1, 6, 2, 2, 2
    ep = {
        "xyz_graspable": torch.tensor(
            [[[0.0, 0.0, 0.5], [0.1, 0.0, 0.6]]], dtype=torch.float32
        ),
        "grasp_top_view_xyz": torch.tensor(
            [[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]], dtype=torch.float32
        ),
        # [B,D,Q,A]
        "grasp_width_pred_angle_depth": torch.tensor(
            [[
                [[0.20, 0.30], [0.40, 0.50]],
                [[0.60, 0.70], [0.80, 0.90]],
            ]],
            dtype=torch.float32,
        ),
    }

    def logits(selected, values):
        x = torch.full((B, T, Q, A, D), -8.0)
        for q, (joint, value) in enumerate(zip(selected, values)):
            a, d = divmod(joint, D)
            x[:, :, q, a, d] = value
        return x

    # q0/q1: Base picks joints 0/1, Full picks joints 3/2.
    base = logits((0, 1), (3.0, 2.0))
    full = logits((3, 2), (4.0, 1.0))
    # Give alternate candidates finite, distinct global scores so cross-scoring
    # at the locally selected action can be checked.
    base[:, :, 0, 1, 1] = -1.0
    base[:, :, 1, 1, 0] = 0.5
    full[:, :, 0, 0, 0] = -0.5
    full[:, :, 1, 0, 1] = -2.0
    return ep, base, full


def test_condition_names_are_complete_2x2():
    assert set(CONDITIONS) == {
        "lbase_gbase",
        "lfull_gbase",
        "lbase_gfull",
        "lfull_gfull",
    }


def test_local_selector_controls_physical_action_global_controls_score(monkeypatch):
    _fake_query_module(monkeypatch)
    ep, base, full = _fixture()
    decoded = {
        "bb": _decode_local_global(ep, base, base, _rotation, 0.1)[0],
        "fb": _decode_local_global(ep, full, base, _rotation, 0.1)[0],
        "bf": _decode_local_global(ep, base, full, _rotation, 0.1)[0],
        "ff": _decode_local_global(ep, full, full, _rotation, 0.1)[0],
    }

    # Same local selector -> byte-identical physical action columns.
    torch.testing.assert_close(decoded["bb"][:, 1:], decoded["bf"][:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(decoded["fb"][:, 1:], decoded["ff"][:, 1:], atol=0, rtol=0)

    # Base-local and Full-local deliberately choose different A,D candidates.
    assert not torch.equal(decoded["bb"][:, 1:], decoded["fb"][:, 1:])

    # Cross global scorer changes only score for the same action.
    assert not torch.equal(decoded["bb"][:, 0], decoded["bf"][:, 0])
    assert not torch.equal(decoded["fb"][:, 0], decoded["ff"][:, 0])

    # Scores are mean-sigmoid of the GLOBAL scorer at the LOCAL-selected action.
    expected_bb_q0 = torch.sigmoid(torch.tensor(3.0))
    expected_bf_q0 = torch.sigmoid(torch.tensor(-0.5))
    torch.testing.assert_close(decoded["bb"][0, 0], expected_bb_q0)
    torch.testing.assert_close(decoded["bf"][0, 0], expected_bf_q0)

    # Depth is determined by local joint index: Base q0 -> d=0 => 1 cm;
    # Full q0 -> joint=3 => d=1 => 2 cm.
    torch.testing.assert_close(decoded["bb"][0, 3], torch.tensor(0.01))
    torch.testing.assert_close(decoded["fb"][0, 3], torch.tensor(0.02))


def test_diagnostic_python_names_do_not_enter_p0_code_digest_glob():
    names = {p.name for p in ROOT.glob("mgf_p0_*.py")}
    assert "mgf_local_global_2x2.py" not in names
    assert "mgf_local_global_2x2_compare.py" not in names


def test_cli_help_without_cuda():
    for script in ("mgf_local_global_2x2.py", "mgf_local_global_2x2_compare.py"):
        p = subprocess.run(
            [sys.executable, str(ROOT / script), "--help"],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr


def test_bash_syntax():
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "run_mgf_p0_local_global_2x2.sh")],
        check=True,
    )
