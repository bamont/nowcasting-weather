import numpy as np
import pytest

from nowcasting.ensemble import EnsembleScores, fair_crps


def _brute_force(members, obs):
    m = members.shape[1]
    term1 = np.abs(members - obs[:, None]).mean(axis=1)
    term2 = np.zeros_like(term1)
    for i in range(m):
        for j in range(m):
            if i != j:
                term2 += np.abs(members[:, i] - members[:, j])
    return term1 - term2 / (2 * m * (m - 1))


def test_fair_crps_matches_definition():
    rng = np.random.default_rng(0)
    members = rng.random((2, 5, 3, 4, 4)).astype(np.float32)
    obs = rng.random((2, 3, 4, 4)).astype(np.float32)
    np.testing.assert_allclose(fair_crps(members, obs), _brute_force(members, obs), atol=1e-5)


def test_fair_crps_two_member_example():
    members = np.array([0.0, 1.0], dtype=np.float32).reshape(1, 2, 1, 1, 1)
    obs = np.zeros((1, 1, 1, 1), dtype=np.float32)
    assert abs(float(fair_crps(members, obs)[0, 0, 0, 0])) < 1e-6  # 0.5 - 0.5


def test_identical_members_reduce_to_absolute_error():
    x = np.random.default_rng(1).random((2, 1, 3, 4, 4)).astype(np.float32)
    members = np.repeat(x, 4, axis=1)
    obs = np.random.default_rng(2).random((2, 3, 4, 4)).astype(np.float32)
    np.testing.assert_allclose(fair_crps(members, obs), np.abs(x[:, 0] - obs), atol=1e-5)


def test_single_member_is_rejected():
    with pytest.raises(ValueError):
        fair_crps(np.zeros((1, 1, 1, 2, 2), dtype=np.float32), np.zeros((1, 1, 2, 2), dtype=np.float32))


def test_spread_skill_is_one_for_a_reliable_ensemble():
    rng = np.random.default_rng(3)
    m = 8
    scores = EnsembleScores(out_len=2)
    for _ in range(4):
        truth_mean = rng.normal(size=(4, 1, 2, 16, 16))
        members = (truth_mean + rng.normal(size=(4, m, 2, 16, 16))).astype(np.float32)
        obs = (truth_mean[:, 0] + rng.normal(size=(4, 2, 16, 16))).astype(np.float32)
        scores.update(members, obs)
    assert np.all(np.abs(scores.compute()["spread_skill"] - 1.0) < 0.05)


def test_brier_and_reference():
    obs = np.full((1, 1, 4, 4), 200 / 255, dtype=np.float32)  # (B, T, H, W), always above threshold 74
    members = np.stack([obs, np.zeros_like(obs)], axis=1)  # (B, M=2, T, H, W): one member right, one wrong
    scores = EnsembleScores(out_len=1, thresholds=(74,))
    scores.update(members, obs, reference=obs)
    res = scores.compute()
    assert abs(res["brier"][0, 0] - 0.25) < 1e-6          # probability 0.5 vs event: (0.5 - 1)^2
    assert res["brier_reference"][0, 0] == 0.0            # perfect deterministic forecast
    assert res["mae_reference"][0] == 0.0
