"""Non-learned baselines. Same interface as learned models: (B, T_in, C, H, W) -> (B, T_out, C, H, W)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


class Persistence(nn.Module):
    """Repeat the last observed frame for every lead time."""

    def __init__(self, out_len: int = 12) -> None:
        super().__init__()
        self.out_len = out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:].expand(-1, self.out_len, -1, -1, -1)


class OpticalFlowAdvection(nn.Module):
    """Extrapolate the last frame along a constant motion field (Farneback optical flow).

    The flow is estimated between frame[-1 - lookback] and frame[-1], divided by `lookback`
    to get a per-step displacement. The prediction at lead time k is the last frame
    warped backwards by k * flow (semi-Lagrangian, constant-flow assumption).
    Intensities are conserved: no growth or decay of cells is modelled.
    """

    def __init__(self, out_len: int = 12, lookback: int = 3, winsize: int = 15) -> None:
        super().__init__()
        self.out_len = out_len
        self.lookback = lookback
        self.winsize = winsize  # Farneback averaging window: larger = smoother, more robust flow

    @staticmethod
    def _to_u8(frame: np.ndarray) -> np.ndarray:
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t_in, c, h, w = x.shape
        if c != 1:
            raise ValueError("OpticalFlowAdvection expects single-channel frames")
        if t_in <= self.lookback:
            raise ValueError(f"need more than lookback={self.lookback} input frames, got {t_in}")

        frames = x[:, :, 0].detach().cpu().numpy().astype(np.float32)  # (B, T, H, W)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        out = np.empty((b, self.out_len, h, w), dtype=np.float32)

        for i in range(b):
            last = np.ascontiguousarray(frames[i, -1])
            prev = frames[i, -1 - self.lookback]
            flow = cv2.calcOpticalFlowFarneback(
                self._to_u8(prev), self._to_u8(last), None, 0.5, 3, self.winsize, 3, 5, 1.2, 0
            )
            flow /= self.lookback  # displacement per 5-minute step, (H, W, 2) = (dx, dy)
            for k in range(1, self.out_len + 1):
                out[i, k - 1] = cv2.remap(
                    last,
                    grid_x - k * flow[..., 0],
                    grid_y - k * flow[..., 1],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        return torch.from_numpy(out).unsqueeze(2).to(x.device)
