import pytest
import torch

from nowcasting.models.unet import UNet


def test_unet_shape_and_range():
    out = UNet(in_len=13, out_len=12, base=8)(torch.rand(2, 13, 1, 32, 32))
    assert out.shape == (2, 12, 1, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_unet_rejects_bad_size():
    with pytest.raises(ValueError):
        UNet(base=8)(torch.rand(1, 13, 1, 30, 30))
