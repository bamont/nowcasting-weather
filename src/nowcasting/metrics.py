"""Categorical nowcasting metrics (CSI, POD, FAR) plus MSE/MAE, per lead time.

Predictions and targets are expected in [0, 1] (VIL / 255), with shape (B, T, C, H, W).

Contingency counts (hits, misses, false alarms) are ACCUMULATED over the whole dataset and
the scores are computed once at the end. Averaging per-sample scores would be wrong:
samples without any event at a given threshold would make the score undefined or biased.
"""

from __future__ import annotations

import torch

# Standard SEVIR VIL thresholds (on the 0-255 scale).
THRESHOLDS: tuple[int, ...] = (16, 74, 133, 160, 181, 219)
MAX_VALUE = 255.0
_EPS = 1e-3  # tolerance on the 0-255 scale, to be robust to float rounding at the threshold


def scores_from_counts(counts: torch.Tensor) -> dict[str, torch.Tensor]:
    """counts[..., 0:3] = (hits, misses, false alarms). Undefined scores are NaN."""
    hits, misses, fas = counts[..., 0], counts[..., 1], counts[..., 2]
    nan = torch.full_like(hits, float("nan"))

    def ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        return torch.where(den > 0, num / den.clamp(min=1), nan)

    return {
        "csi": ratio(hits, hits + misses + fas),
        "pod": ratio(hits, hits + misses),
        "far": ratio(fas, hits + fas),
    }


class NowcastingMetrics:
    def __init__(self, out_len: int, thresholds: tuple[int, ...] = THRESHOLDS) -> None:
        self.out_len = out_len
        self.thresholds = thresholds
        # counts[t, k] = (hits, misses, false_alarms) at lead time t and threshold k
        self.counts = torch.zeros(out_len, len(thresholds), 3, dtype=torch.float64)
        self.sq_err = torch.zeros(out_len, dtype=torch.float64)
        self.abs_err = torch.zeros(out_len, dtype=torch.float64)
        self.n_pixels = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.shape[1] != self.out_len:
            raise ValueError(f"expected {self.out_len} lead times, got {pred.shape[1]}")
        b, t = pred.shape[:2]
        p = pred.detach().float().cpu().reshape(b, t, -1) * MAX_VALUE
        y = target.detach().float().cpu().reshape(b, t, -1) * MAX_VALUE

        diff = (p - y) / MAX_VALUE
        self.sq_err += diff.pow(2).sum(dim=(0, 2)).double()
        self.abs_err += diff.abs().sum(dim=(0, 2)).double()
        self.n_pixels += b * p.shape[2]

        for k, thr in enumerate(self.thresholds):
            pb, yb = p >= thr - _EPS, y >= thr - _EPS
            self.counts[:, k, 0] += (pb & yb).sum(dim=(0, 2)).double()
            self.counts[:, k, 1] += (~pb & yb).sum(dim=(0, 2)).double()
            self.counts[:, k, 2] += (pb & ~yb).sum(dim=(0, 2)).double()

    def frequency_bias(self) -> torch.Tensor:
        """(K,) pooled over lead times: (hits + false alarms) / (hits + misses).

        1 = the forecast covers as much area above the threshold as the observation.
        """
        c = self.counts.sum(dim=0)
        den = c[:, 0] + c[:, 1]
        return torch.where(den > 0, (c[:, 0] + c[:, 2]) / den.clamp(min=1), torch.full_like(den, float("nan")))

    def compute(self) -> dict[str, torch.Tensor]:
        """Per lead time: csi/pod/far with shape (T, K); mse/mae with shape (T,)."""
        per_horizon = scores_from_counts(self.counts)
        pooled = scores_from_counts(self.counts.sum(dim=0))  # pooled over lead times, (K,)
        return {
            **per_horizon,
            **{f"{k}_pooled": v for k, v in pooled.items()},
            "mse": self.sq_err / max(self.n_pixels, 1),
            "mae": self.abs_err / max(self.n_pixels, 1),
        }
