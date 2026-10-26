import h5py
import numpy as np
import pandas as pd
import pytest

from nowcasting.data.cache import CachedSEVIRDataset, build_split_cache
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


def test_cache_matches_raw_dataset(fake_sevir, tmp_path):
    cache_dir = tmp_path / "cache"
    build_split_cache(fake_sevir, cache_dir, "test", image_size=128)

    raw = SEVIRDataset(fake_sevir, "test", image_size=128)
    cached = CachedSEVIRDataset(cache_dir, "test", image_size=128)
    assert len(raw) == len(cached)

    for i in (0, 4, len(raw) - 1):
        (xr, yr), (xc, yc) = raw[i], cached[i]
        assert xc.shape == xr.shape and yc.shape == yr.shape
        # the cache stores rounded uint8, so allow half a quantisation step
        assert (xr - xc).abs().max() <= 0.5 / 255 + 1e-5
        assert (yr - yc).abs().max() <= 0.5 / 255 + 1e-5


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CachedSEVIRDataset(tmp_path, "train")
