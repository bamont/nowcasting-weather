import numpy as np
import pytest
import torch

from nowcasting.data.cache import CachedSEVIRDataset, build_advection_cache, cache_path
from nowcasting.models.baselines import OpticalFlowAdvection
from nowcasting.models.unet import UNet

SIZE = 64


@pytest.fixture()
def fake_cache(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, size=(2, 49, SIZE, SIZE), dtype=np.uint8)
    np.save(cache_path(tmp_path, "test", SIZE), data)
    return tmp_path


def test_advection_channels_match_direct_computation(fake_cache):
    build_advection_cache(fake_cache, "test", image_size=SIZE, lookback=4, winsize=15, batch_size=3)
    plain = CachedSEVIRDataset(fake_cache, "test", image_size=SIZE)
    ext = CachedSEVIRDataset(fake_cache, "test", image_size=SIZE, advection=True)
    assert len(plain) == len(ext) == 6  # 2 events x 3 windows

    x, y = ext[4]  # event 1, window starting at frame 12
    x_plain, y_plain = plain[4]
    assert x.shape == (25, 1, SIZE, SIZE) and y.shape == (12, 1, SIZE, SIZE)
    torch.testing.assert_close(x[:13], x_plain)
    torch.testing.assert_close(y, y_plain)
    direct = OpticalFlowAdvection(12, lookback=4, winsize=15)(x_plain[None])[0]
    assert (x[13:] - direct).abs().max() <= 0.5 / 255 + 1e-5  # the cache stores rounded uint8


def test_advection_input_needs_cache_and_default_stride(fake_cache):
    with pytest.raises(FileNotFoundError):
        CachedSEVIRDataset(fake_cache, "test", image_size=SIZE, advection=True)
    build_advection_cache(fake_cache, "test", image_size=SIZE, lookback=4, winsize=15)
    with pytest.raises(ValueError):
        CachedSEVIRDataset(fake_cache, "test", image_size=SIZE, stride=4, advection=True)


def test_unet_accepts_25_input_frames():
    out = UNet(in_len=25, out_len=12, base=8)(torch.rand(2, 25, 1, 32, 32))
    assert out.shape == (2, 12, 1, 32, 32)
