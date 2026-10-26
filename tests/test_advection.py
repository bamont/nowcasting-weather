import numpy as np
import torch

from nowcasting.models.baselines import OpticalFlowAdvection, Persistence


def moving_blobs(n_frames: int, size: int = 64, speed: float = 2.0) -> torch.Tensor:
    """Two Gaussian blobs translating along +x. Returns (1, T, 1, H, W) in [0, 1]."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    frames = []
    for i in range(n_frames):
        f = np.zeros((size, size), np.float32)
        for cy, cx, s in ((20, 6, 5), (40, 8, 4)):
            f += np.exp(-(((yy - cy) ** 2 + (xx - (cx + speed * i)) ** 2) / (2 * s**2)))
        frames.append(np.clip(f, 0, 1))
    return torch.from_numpy(np.stack(frames))[None, :, None]


def test_output_shape_and_range():
    x = torch.rand(2, 13, 1, 32, 32)
    out = OpticalFlowAdvection(out_len=12)(x)
    assert out.shape == (2, 12, 1, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_advection_beats_persistence_on_pure_translation():
    seq = moving_blobs(13 + 12)
    x, y = seq[:, :13], seq[:, 13:]
    adv = OpticalFlowAdvection(out_len=12)(x)
    per = Persistence(out_len=12)(x)
    mse_adv = (adv[:, -1] - y[:, -1]).pow(2).mean()
    mse_per = (per[:, -1] - y[:, -1]).pow(2).mean()
    assert mse_adv < 0.75 * mse_per
