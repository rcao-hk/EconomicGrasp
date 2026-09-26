import numpy as np

from audit_rep_p1_full_k_curves import (
    exact_oracle_k,
    policy_selection,
    raw_argmax_k,
    sign_accuracy,
)


def test_exact_oracle_falls_back_to_native_when_no_better_candidate():
    exact = np.asarray([
        [0.4, 0.5, 0.4],
        [0.6, 0.5, 0.4],
    ], dtype=np.float32)
    valid = np.ones_like(exact, dtype=bool)
    zero = 1
    oracle = exact_oracle_k(exact, valid, zero)
    # Row 0 has a tie/no improvement over native -> stay.
    # Row 1 has a genuinely better -offset candidate -> move.
    np.testing.assert_array_equal(oracle, np.asarray([1, 0], dtype=np.int16))


def test_policy_selection_uses_best_alt_plus_margin():
    pred = np.asarray([
        [0.2, 0.5, 0.9],
        [0.8, 0.5, 0.7],
    ], dtype=np.float32)
    valid = np.ones_like(pred, dtype=bool)
    zero = 1
    selected, best_alt, advantage = policy_selection(
        pred, valid, zero, margin=0.3
    )
    np.testing.assert_array_equal(best_alt, np.asarray([2, 0], dtype=np.int16))
    # Floating-point 0.9-0.5 is strictly > 0.3; 0.8-0.5 is also > 0.3.
    np.testing.assert_array_equal(selected, np.asarray([2, 0], dtype=np.int16))
    np.testing.assert_allclose(advantage, np.asarray([0.4, 0.3]), atol=1e-6)


def test_raw_argmax_respects_validity():
    pred = np.asarray([
        [0.9, 0.1, 0.8],
        [0.2, 0.7, 0.6],
    ], dtype=np.float32)
    valid = np.asarray([
        [False, True, True],
        [True, True, True],
    ])
    got = raw_argmax_k(pred, valid)
    np.testing.assert_array_equal(got, np.asarray([2, 1], dtype=np.int16))


def test_sign_accuracy_ignores_exact_ties():
    pred_delta = np.asarray([0.3, -0.2, 0.9, -0.7])
    exact_delta = np.asarray([0.1, -0.1, 0.0, 0.2])
    acc, count = sign_accuracy(pred_delta, exact_delta)
    assert count == 3
    assert np.isclose(acc, 2.0 / 3.0)
