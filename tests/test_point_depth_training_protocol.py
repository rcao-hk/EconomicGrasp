"""CPU-only tests for original/mixed point-head training manifest semantics."""
import copy

import pytest

from utils.point_depth_training_protocol import (
    checkpoint_training_protocol,
    expected_frames_per_domain,
    validate_training_protocol,
)

SIG = "0123456789abcdef" * 4


def _full_manifest(head, domain="original", fraction=0.5):
    if domain == "original":
        return {
            "head": head,
            "pose_depth_mode": "global_film",
            "train_domain": "original",
            "train_frame_count": 25600,
            "gntrans_train_count": 0,
            "max_train_batches": 0,
            "max_val_batches": 0,
        }
    count = expected_frames_per_domain(fraction)
    return {
        "head": head,
        "pose_depth_mode": "global_film",
        "train_domain": "mixed",
        "mix_train_fraction": fraction,
        "train_frame_count": 2 * count,
        "original_train_count": count,
        "gntrans_train_count": count,
        "paired_scene_frame_sampling": True,
        "original_train_scene_frame_sha256": SIG,
        "gntrans_train_scene_frame_sha256": SIG,
        "gntrans_rgb_root": "/dataset/GN-Trans",
        "max_train_batches": 0,
        "max_val_batches": 0,
    }


def _checkpoint(head, protocol, key="full100_protocol"):
    return {
        "architecture": "economicgrasp_depth_baseline",
        "pose_depth_mode": "global_film",
        "point_head": head,
        "completed_epoch": 19,
        key: protocol,
    }


@pytest.mark.parametrize("fraction,count", [
    (0.1, 2600), (0.5, 12800), (1.0, 25600),
])
def test_selection_contract(fraction, count):
    assert expected_frames_per_domain(fraction) == count


@pytest.mark.parametrize("bad", [0, -1, 1.01, float("nan"), float("inf")])
def test_invalid_selection_fraction(bad):
    with pytest.raises(ValueError):
        expected_frames_per_domain(bad)


@pytest.mark.parametrize("head", ["legacy", "cdf"])
def test_full_original_and_paired_half_are_distinct(head):
    original = _checkpoint(head, _full_manifest(head))
    mixed = _checkpoint(head, _full_manifest(head, "mixed", 0.5))
    domain_a, info_a, key_a = checkpoint_training_protocol(original, expected_head=head)
    domain_b, info_b, key_b = checkpoint_training_protocol(mixed, expected_head=head)
    assert domain_a == "original" and domain_b == "mixed"
    assert key_a == key_b == "full100_protocol"
    # They contain the same 25,600 observations but different input domains.
    assert info_a["train_frame_count"] == info_b["train_frame_count"] == 25600
    assert (info_b["original_train_count"], info_b["gntrans_train_count"]) == (
        12800, 12800
    )


@pytest.mark.parametrize("head", ["legacy", "cdf"])
@pytest.mark.parametrize("fraction", [0.1, 0.5])
def test_new_mixed_contract_both_heads(head, fraction):
    ckpt = _checkpoint(head, _full_manifest(head, "mixed", fraction))
    assert checkpoint_training_protocol(ckpt, expected_head=head)[0] == "mixed"


@pytest.mark.parametrize("defect", [
    "wrong_count", "nonpaired", "different_frame_ids", "no_rgb_root", "partial",
])
def test_mixed_manifest_rejects_corruption(defect):
    protocol = _full_manifest("cdf", "mixed", 0.5)
    if defect == "wrong_count":
        protocol["gntrans_train_count"] = 2600
    if defect == "nonpaired":
        protocol["paired_scene_frame_sampling"] = False
    if defect == "different_frame_ids":
        protocol["gntrans_train_scene_frame_sha256"] = "abcdef0123456789" * 4
    if defect == "no_rgb_root":
        protocol["gntrans_rgb_root"] = None
    if defect == "partial":
        protocol["max_train_batches"] = 2
    with pytest.raises(ValueError):
        checkpoint_training_protocol(
            _checkpoint("cdf", protocol), expected_head="cdf"
        )


def test_head_and_epoch_mismatch_fail_closed():
    ckpt = _checkpoint("cdf", _full_manifest("cdf", "mixed"))
    with pytest.raises(ValueError, match="head"):
        checkpoint_training_protocol(ckpt, expected_head="legacy")
    ckpt["completed_epoch"] = 18
    with pytest.raises(ValueError, match="epoch"):
        checkpoint_training_protocol(ckpt, expected_head="cdf")


def test_backwards_compatible_original_full100():
    protocol = _full_manifest("legacy")
    del protocol["train_domain"]
    ckpt = _checkpoint("legacy", protocol)
    assert checkpoint_training_protocol(ckpt, expected_head="legacy")[0] == "original"


@pytest.mark.parametrize("fraction", [0.1, 0.5])
def test_historical_mixed_legacy_supports_fraction_without_5200_hardcode(fraction):
    n = expected_frames_per_domain(fraction)
    old = {
        "pose_depth_mode": "global_film",
        "train_fraction": fraction,
        "original_train_count": n,
        "gntrans_train_count": n,
        "train_count": 2 * n,
        "paired_scene_frame_sampling": True,
        "original_train_sha256": SIG,
        "gntrans_train_sha256": SIG,
    }
    ckpt = _checkpoint("legacy", old, "mixed_protocol")
    assert checkpoint_training_protocol(ckpt, expected_head="legacy")[0] == "mixed"
    with pytest.raises(ValueError, match="head"):
        checkpoint_training_protocol(ckpt, expected_head="cdf")


def test_wrong_mixed_split_even_if_total_still_25600():
    p = _full_manifest("legacy", "mixed", 0.5)
    p["original_train_count"] = 15000
    p["gntrans_train_count"] = 10600
    with pytest.raises(ValueError):
        validate_training_protocol(p, head="legacy", source="full100_protocol")
