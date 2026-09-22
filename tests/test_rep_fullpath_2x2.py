import numpy as np

from make_rep_fullpath_2x2_decomposition import compose_2x2


def _payload():
    k, q, c = 3, 4, 6
    actions = np.zeros((k, q, 17), np.float32)
    for kk in range(k):
        actions[kk, :, 1:16] = kk + np.arange(q)[:, None] * 0.01
        actions[kk, :, 15] = 0.5 + 0.01 * kk
    valid = np.ones((k, q), bool)

    # A1 probability mean grows with hypothesis index and query.
    probs = np.zeros((1, k, q, c), np.float32)
    for kk in range(k):
        for qq in range(q):
            probs[0, kk, qq] = 0.1 * kk + 0.01 * qq

    selected_fixed = np.array([0, 1, 2, 1])
    selected_val = np.array([0, 0, 2, 2])
    qq = np.arange(q)
    fixed_score = probs[0, selected_fixed, qq].mean(-1)
    val_score = probs[0, selected_val, qq].mean(-1)

    return {
        "actions": actions,
        "valid": valid,
        "zero_index": np.array(1),
        "scorer_names": np.array(["A1"]),
        "probabilities": probs,
        "original_native_score": np.array([0.9, 0.8, 0.7, 0.6], np.float32),
        "output_methods": np.array(["stage1_native", "A1", "A1"]),
        "output_policies": np.array(["native", "fixed_0", "val_selected"]),
        "selected": np.stack([
            np.ones(q, np.int64),
            selected_fixed,
            selected_val,
        ]),
        "rank_scores": np.stack([
            np.array([0.9, 0.8, 0.7, 0.6], np.float32),
            fixed_score,
            val_score,
        ]),
    }


def test_compose_2x2_changes_only_requested_factor():
    d = _payload()
    out = compose_2x2(d)
    q = np.arange(4)
    native = d["actions"][1]

    # Baseline keeps native action and Stage-1 score.
    np.testing.assert_allclose(out["baseline"][:, 1:16], native[:, 1:16])
    np.testing.assert_allclose(out["baseline"][:, 0], d["original_native_score"])

    # Score-only changes score but never physical action.
    np.testing.assert_allclose(out["score_only"][:, 1:16], native[:, 1:16])
    expected_zero = d["probabilities"][0, 1].mean(-1)
    np.testing.assert_allclose(out["score_only"][:, 0], expected_zero)

    for policy, row in (("fixed_0", 1), ("val_selected", 2)):
        sel = d["selected"][row]
        action_only = out["per_policy"][policy]["action_only"]
        full = out["per_policy"][policy]["full"]

        # Both use exactly the A1-selected physical action.
        np.testing.assert_allclose(
            action_only[:, 1:16], d["actions"][sel, q, 1:16]
        )
        np.testing.assert_allclose(full[:, 1:16], d["actions"][sel, q, 1:16])

        # Action-only retains Stage-1 ranking; full retains A1 ranking.
        np.testing.assert_allclose(action_only[:, 0], d["original_native_score"])
        np.testing.assert_allclose(full[:, 0], d["rank_scores"][row])


def test_compose_2x2_rejects_score_replay_mismatch():
    d = _payload()
    d["rank_scores"] = d["rank_scores"].copy()
    d["rank_scores"][1, 0] += 0.1
    try:
        compose_2x2(d, score_atol=1e-7)
    except RuntimeError as exc:
        assert "score replay mismatch" in str(exc)
    else:
        raise AssertionError("Expected score replay mismatch")
