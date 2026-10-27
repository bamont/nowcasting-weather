"""Sharpness / spatial-scale diagnostics (numpy only): radial power spectrum and FSS.

Why: a model trained with MSE predicts a blurry mean. Pixel-wise scores such as CSI do not
measure this directly, so we add
  * the radially averaged power spectrum (how much small-scale detail is kept), and
  * the Fractions Skill Score (does the forecast get the right amount of rain
    at the right place, up to a tolerance of `scale` pixels?).

All fields are (..., H, W) arrays with values in [0, 1] (VIL / 255).
"""

from __future__ import annotations

import numpy as np

MAX_VALUE = 255.0
_EPS = 1e-3  # tolerance on the 0-255 scale when thresholding (same as nowcasting.metrics)


# ----------------------------------------------------------------------------- spectrum
class SpectrumAccumulator:
    """Mean radially averaged power spectrum per lead time.

    Frames are multiplied by a 2D Hann window (the radar domain is not periodic), then the
    2D FFT power is averaged over rings of integer wavenumber k = 0..H/2 (cycles per image).
    """

    def __init__(self, out_len: int, size: int = 128) -> None:
        self.out_len, self.size = out_len, size
        self.n_bins = size // 2 + 1
        window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
        self.window = window / np.sqrt((window**2).mean())  # unit mean-square: power stays comparable
        k = np.fft.fftfreq(size) * size
        radius = np.rint(np.sqrt(k[None, :] ** 2 + k[:, None] ** 2)).astype(int).ravel()
        onehot = np.zeros((size * size, self.n_bins), dtype=np.float32)
        valid = radius < self.n_bins  # drop the corners beyond Nyquist
        onehot[np.flatnonzero(valid), radius[valid]] = 1.0
        self.ring_mean = onehot / np.maximum(onehot.sum(axis=0, keepdims=True), 1.0)
        self.total = np.zeros((out_len, self.n_bins), dtype=np.float64)
        self.n = 0

    def update(self, frames: np.ndarray) -> None:
        """frames: (B, T, H, W)."""
        b, t, h, w = frames.shape
        if (h, w) != (self.size, self.size) or t != self.out_len:
            raise ValueError(f"expected (B, {self.out_len}, {self.size}, {self.size}), got {frames.shape}")
        power = np.abs(np.fft.fft2(frames.astype(np.float32) * self.window)) ** 2  # (B, T, H, W)
        rings = power.reshape(b * t, -1).astype(np.float32) @ self.ring_mean  # (B*T, n_bins)
        self.total += rings.reshape(b, t, self.n_bins).sum(axis=0)
        self.n += b

    def compute(self) -> np.ndarray:
        """(T, n_bins) mean power. Index k is the wavenumber; k=0 (mean field) is kept but not used."""
        return self.total / max(self.n, 1)


# ----------------------------------------------------------------------------------- FSS
def box_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean over an n x n window (n odd) with zero padding outside the domain, via integral image."""
    if n % 2 == 0:
        raise ValueError("window size must be odd")
    if n == 1:
        return x.astype(np.float32, copy=False)
    r = n // 2
    lead = [(0, 0)] * (x.ndim - 2)
    xp = np.pad(x.astype(np.float32, copy=False), lead + [(r, r), (r, r)])
    ii = xp.cumsum(axis=-1, dtype=np.float32).cumsum(axis=-2, dtype=np.float32)
    ii = np.pad(ii, lead + [(1, 0), (1, 0)])
    s = ii[..., n:, n:] - ii[..., :-n, n:] - ii[..., n:, :-n] + ii[..., :-n, :-n]
    return s / float(n * n)


class FSSAccumulator:
    """Fractions Skill Score per lead time, threshold and neighbourhood size.

    FSS = 1 - sum((Pf - Of)^2) / (sum(Pf^2) + sum(Of^2)), where Pf / Of are the fractions of
    pixels above the threshold in an n x n window. Sums are accumulated over the whole dataset
    and the score is computed once at the end (1 = perfect, 0 = no skill at that scale).
    """

    def __init__(self, out_len: int, thresholds: tuple[int, ...], scales: tuple[int, ...]) -> None:
        self.thresholds, self.scales = tuple(thresholds), tuple(scales)
        shape = (out_len, len(self.thresholds), len(self.scales))
        self.num = np.zeros(shape, dtype=np.float64)
        self.den = np.zeros(shape, dtype=np.float64)

    def observed_fractions(self, obs: np.ndarray) -> list[list[np.ndarray]]:
        """Precompute the observation fractions once per batch, to share them between models."""
        o255 = obs * MAX_VALUE
        out = []
        for thr in self.thresholds:
            ob = (o255 >= thr - _EPS).astype(np.float32)
            out.append([box_mean(ob, n) for n in self.scales])
        return out

    def update(self, pred: np.ndarray, obs: np.ndarray, obs_fractions=None) -> None:
        """pred, obs: (B, T, H, W) in [0, 1]. `obs_fractions`: result of observed_fractions(obs)."""
        if obs_fractions is None:
            obs_fractions = self.observed_fractions(obs)
        p255 = pred * MAX_VALUE
        for k, thr in enumerate(self.thresholds):
            pb = (p255 >= thr - _EPS).astype(np.float32)
            for s, n in enumerate(self.scales):
                pf, of = box_mean(pb, n), obs_fractions[k][s]
                self.num[:, k, s] += ((pf - of) ** 2).sum(axis=(0, 2, 3))
                self.den[:, k, s] += (pf**2).sum(axis=(0, 2, 3)) + (of**2).sum(axis=(0, 2, 3))

    def compute(self) -> np.ndarray:
        """(T, n_thresholds, n_scales); NaN where neither forecast nor observation has an event."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.den > 0, 1.0 - self.num / self.den, np.nan)
