"""PyTorch Dataset for SEVIR VIL nowcasting.

Expected layout on disk (as produced by scripts/download_sevir.py):

    <root>/CATALOG.csv
    <root>/data/vil/<year>/SEVIR_VIL_*.h5

Each VIL event is a (384, 384, 49) uint8 array: 49 frames, 5 minutes apart.
Events are split by time (protocol similar to the Earthformer paper).
Only events whose HDF5 file exists locally are used, so a partial download works.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SPLITS: dict[str, tuple[str | None, str | None]] = {
    "train": (None, "2019-03-01"),
    "val": ("2019-03-01", "2019-06-01"),
    "test": ("2019-06-01", None),
}
FRAMES_PER_EVENT = 49
NATIVE_SIZE = 384


def load_catalog(root: str | Path, img_type: str = "vil") -> pd.DataFrame:
    root = Path(root)
    cat = pd.read_csv(root / "CATALOG.csv", parse_dates=["time_utc"], low_memory=False)
    cat = cat[cat["img_type"] == img_type].copy()
    cat["path"] = cat["file_name"].map(lambda f: root / "data" / f)
    return cat


class SEVIRDataset(Dataset):
    """Yields (x, y) with shapes (in_len, 1, H, W) and (out_len, 1, H, W), values in [0, 1]."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        in_len: int = 13,
        out_len: int = 12,
        stride: int = 12,
        image_size: int = 128,
        max_pct_missing: float = 0.0,
        name_contains: str | None = "STORMEVENTS",
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}")
        if NATIVE_SIZE % image_size != 0:
            raise ValueError(f"image_size must divide {NATIVE_SIZE} (e.g. 64, 96, 128, 192, 384)")
        self.in_len, self.out_len = in_len, out_len
        self.seq_len = in_len + out_len
        self.pool = NATIVE_SIZE // image_size

        cat = load_catalog(root)
        start, end = SPLITS[split]
        if start is not None:
            cat = cat[cat["time_utc"] >= pd.Timestamp(start)]
        if end is not None:
            cat = cat[cat["time_utc"] < pd.Timestamp(end)]
        cat = cat[cat["pct_missing"] <= max_pct_missing]
        if name_contains:
            cat = cat[cat["file_name"].str.contains(name_contains)]
        cat = cat[cat["path"].map(Path.exists)].reset_index(drop=True)
        if cat.empty:
            raise RuntimeError(
                f"No events found for split '{split}'. Is the data downloaded, and do the "
                "downloaded files cover this split's date range?"
            )
        self.events = list(zip(cat["path"], cat["file_index"].astype(int), strict=True))

        offsets = range(0, FRAMES_PER_EVENT - self.seq_len + 1, stride)
        self.index = [(e, o) for e in range(len(self.events)) for o in offsets]
        self._files: dict[str, h5py.File] = {}  # opened lazily, once per worker process

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, path: Path) -> h5py.File:
        key = str(path)
        if key not in self._files:
            self._files[key] = h5py.File(key, "r")
        return self._files[key]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        event_idx, offset = self.index[i]
        path, file_index = self.events[event_idx]
        raw = self._file(path)["vil"][file_index, :, :, offset : offset + self.seq_len]
        seq = torch.from_numpy(np.ascontiguousarray(raw.transpose(2, 0, 1)))  # (T, H, W)
        seq = seq.float().div_(255.0).unsqueeze(1)  # (T, 1, H, W)
        if self.pool > 1:
            seq = F.avg_pool2d(seq, self.pool)
        return seq[: self.in_len], seq[self.in_len :]
