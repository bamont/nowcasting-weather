"""Ensemble / probabilistic scores (numpy only): fair CRPS, spread-skill, Brier score.

Shapes: members (B, M, T, H, W) and observation (B, T, H, W), values in [0, 1] (VIL / 255).
A deterministic forecast can be passed as `reference`: its CRPS equals its MAE, and its Brier
score is computed from the 0/1 exceedance field.
"""

from __future__ import annotations

import numpy as np

MAX_VALUE = 255.0
_EPS = 1e-3


def fair_crps(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS (Ferro 2014), pixel-wise: (B, T, H, W). Needs at least 2 members.

    mean_m |x_m - y| - 1 / (2 M (M - 1)) * sum_{m != m'} |x_m - x_m'|,
    with the pairwise term computed from sorted members: sum_{m != m'} |.| = 2 sum_k (2k - M + 1) x_(k).
    """
    m = members.shape[1]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    mean_abs = np.abs(members - obs[:, None]).mean(axis=1)
    xs = np.sort(members, axis=1)
    coef = (2 * np.arange(m) - m + 1).astype(np.float32).reshape(1, m, 1, 1, 1)
    pairwise = (coef * xs).sum(axis=1) / (m * (m - 1))
    return mean_abs - pairwise


class EnsembleScores:
    def __init__(self, out_len: int, thresholds: tuple[int, ...] = (74, 133, 181)) -> None:
        self.out_len, self.thresholds = out_len, tuple(thresholds)
        k = len(self.thresholds)
        self.crps = np.zeros(out_len)
        self.abs_ref = np.zeros(out_len)
        self.sq_mean = np.zeros(out_len)
        self.var = np.zeros(out_len)
        self.brier = np.zeros((out_len, k))
        self.brier_ref = np.zeros((out_len, k))
        self.n = 0
        self.n_members = 0

    def update(self, members: np.ndarray, obs: np.ndarray, reference: np.ndarray | None = None) -> None:
        b, m, t, h, w = members.shape
        if t != self.out_len:
            raise ValueError(f"expected {self.out_len} lead times, got {t}")
        axes = (0, 2, 3)
        self.crps += fair_crps(members, obs).sum(axis=axes, dtype=np.float64)
        self.sq_mean += ((members.mean(axis=1) - obs) ** 2).sum(axis=axes, dtype=np.float64)
        self.var += members.var(axis=1, ddof=1).sum(axis=axes, dtype=np.float64)
        for k, thr in enumerate(self.thresholds):
            occurred = obs * MAX_VALUE >= thr - _EPS
            prob = (members * MAX_VALUE >= thr - _EPS).mean(axis=1)
            self.brier[:, k] += ((prob - occurred) ** 2).sum(axis=axes, dtype=np.float64)
            if reference is not None:
                ref_event = (reference * MAX_VALUE >= thr - _EPS).astype(np.float32)
                self.brier_ref[:, k] += ((ref_event - occurred) ** 2).sum(axis=axes, dtype=np.float64)
        if reference is not None:
            self.abs_ref += np.abs(reference - obs).sum(axis=axes, dtype=np.float64)
        self.n += b * h * w
        self.n_members = m

    def compute(self) -> dict[str, np.ndarray]:
        n, m = max(self.n, 1), max(self.n_members, 2)
        rmse = np.sqrt(self.sq_mean / n)
        spread = np.sqrt(self.var / n)
        return {
            "crps": self.crps / n,
            "mae_reference": self.abs_ref / n,
            "rmse_ensemble_mean": rmse,
            "spread": spread,
            # for a reliable ensemble: E[RMSE^2] = (1 + 1/M) * spread^2, so this ratio is ~1
            "spread_skill": spread * np.sqrt((m + 1) / m) / np.maximum(rmse, 1e-12),
            "brier": self.brier / n,
            "brier_reference": self.brier_ref / n,
        }
