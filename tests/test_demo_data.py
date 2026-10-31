import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_data.py"


def test_make_demo_data(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    data = np.random.default_rng(0).integers(0, 255, size=(12, 49, 16, 16), dtype=np.uint8)
    np.save(cache / "test_16.npy", data)
    out = tmp_path / "demo.npz"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--cache-dir", str(cache), "--image-size", "16",
         "--n-intense", "2", "--n-random", "3", "--out", str(out)],
        check=True,
    )
    z = np.load(out)
    assert z["frames"].shape == (5, 49, 16, 16) and z["frames"].dtype == np.uint8
    assert list(z["kind"]) == ["intense"] * 2 + ["random"] * 3
    assert len(set(z["event_index"].tolist())) == 5            # no duplicated event
    np.testing.assert_array_equal(z["frames"][0], data[z["event_index"][0]])
