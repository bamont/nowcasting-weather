import numpy as np
import pytest

from nowcasting.sharpness import FSSAccumulator, SpectrumAccumulator, box_mean


def test_box_mean_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.random((2, 9, 11)).astype(np.float32)
    n = 3
    padded = np.pad(x, [(0, 0), (1, 1), (1, 1)])
    expected = np.zeros_like(x)
    for i in range(9):
        for j in range(11):
            expected[:, i, j] = padded[:, i : i + n, j : j + n].mean(axis=(1, 2))
    np.testing.assert_allclose(box_mean(x, n), expected, atol=1e-5)


def test_box_mean_rejects_even_window():
    with pytest.raises(ValueError):
        box_mean(np.zeros((4, 4)), 4)


def test_spectrum_detects_blur():
    rng = np.random.default_rng(0)
    sharp = rng.random((8, 2, 64, 64)).astype(np.float32)
    blurred = box_mean(sharp, 5)  # same field, small scales removed
    acc_a, acc_b = SpectrumAccumulator(2, 64), SpectrumAccumulator(2, 64)
    acc_a.update(sharp)
    acc_b.update(blurred)
    ratio = acc_b.compute()[0] / acc_a.compute()[0]
    assert ratio[2:6].mean() > 0.5       # large scales kept
    assert ratio[24:].mean() < 0.1       # small scales removed


def test_spectrum_of_identical_fields_is_equal():
    x = np.random.default_rng(1).random((3, 2, 32, 32)).astype(np.float32)
    a, b = SpectrumAccumulator(2, 32), SpectrumAccumulator(2, 32)
    a.update(x)
    b.update(x[:1])
    b.update(x[1:])
    np.testing.assert_allclose(a.compute(), b.compute(), rtol=1e-4)


def _blob(size, top, left, side=5):
    f = np.zeros((1, 1, size, size), dtype=np.float32)
    f[..., top : top + side, left : left + side] = 1.0  # VIL = 255 > every threshold
    return f


def test_fss_perfect_forecast():
    obs = _blob(64, 10, 10)
    acc = FSSAccumulator(1, (74,), (1, 5, 9))
    acc.update(obs, obs)
    np.testing.assert_allclose(acc.compute(), 1.0)


def test_fss_increases_with_neighbourhood_size_for_displaced_forecast():
    obs, pred = _blob(64, 10, 10), _blob(64, 10, 18)  # disjoint, displaced by 8 px
    acc = FSSAccumulator(1, (74,), (1, 5, 9, 17, 33))
    acc.update(pred, obs)
    fss = acc.compute()[0, 0]
    assert fss[0] == 0.0                      # no overlap at pixel scale
    assert np.all(np.diff(fss) >= -1e-6)      # more tolerance, never worse
    assert fss[-1] > 0.8                      # large window forgives the displacement


def test_fss_nan_without_events():
    z = np.zeros((1, 1, 16, 16), dtype=np.float32)
    acc = FSSAccumulator(1, (74,), (1, 3))
    acc.update(z, z)
    assert np.isnan(acc.compute()).all()
