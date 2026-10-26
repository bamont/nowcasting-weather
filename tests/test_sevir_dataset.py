"""Tests the dataset on a tiny synthetic SEVIR-like layout (no download needed)."""

import h5py
import numpy as np
import pandas as pd
import pytest

from nowcasting.data.sevir import SEVIRDataset


@pytest.fixture()
def fake_sevir(tmp_path):
    rel = "vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"
    h5_path = tmp_path / "data" / rel
    h5_path.parent.mkdir(parents=True)
    with h5py.File(h5_path, "w") as f:
        f["vil"] = np.random.randint(0, 255, (3, 384, 384, 49), dtype=np.uint8)
    rows = [
        {"id": f"e{i}", "file_name": rel, "file_index": i, "img_type": "vil",
         "time_utc": "2019-08-01 12:00:00", "pct_missing": 0.0}
        for i in range(3)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "CATALOG.csv", index=False)
    return tmp_path


def test_shapes_and_range(fake_sevir):
    ds = SEVIRDataset(fake_sevir, "test", image_size=128)
    assert len(ds) == 3 * 3  # 3 events x 3 windows (stride 12, seq_len 25)
    x, y = ds[0]
    assert x.shape == (13, 1, 128, 128)
    assert y.shape == (12, 1, 128, 128)
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_empty_split_raises(fake_sevir):
    with pytest.raises(RuntimeError):
        SEVIRDataset(fake_sevir, "train")
