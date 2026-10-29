"""Compact on-disk cache of SEVIR VIL at reduced resolution, and a Dataset that reads it.

Why: the raw HDF5 files are ~26 GB at 384x384. At 128x128 / uint8 the whole
train+val+test set is ~2.8 GB, which is easy to upload to Kaggle / Drive and much
faster to read during training (no HDF5 access, no pooling at runtime).

Layout: <cache_dir>/<split>_<size>.npy with shape (N_events, 49, size, size), uint8.
"""

from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from nowcasting.data.sevir import FRAMES_PER_EVENT, NATIVE_SIZE, SEVIRDataset


# Windows of 25 frames start at these offsets when stride=12 (the default used everywhere).
ADV_OFFSETS = (0, 12, 24)


def cache_path(cache_dir: str | Path, split: str, image_size: int) -> Path:
    return Path(cache_dir) / f"{split}_{image_size}.npy"


def advection_cache_path(cache_dir: str | Path, split: str, image_size: int) -> Path:
    return Path(cache_dir) / f"adv_{split}_{image_size}.npy"


def build_split_cache(
    root: str | Path, cache_dir: str | Path, split: str, image_size: int = 128
) -> Path:
    """Convert one split of the raw SEVIR files into a single uint8 .npy file."""
    if NATIVE_SIZE % image_size != 0:
        raise ValueError(f"image_size must divide {NATIVE_SIZE}")
    factor = NATIVE_SIZE // image_size

    # Reuse the dataset's own event selection so the cache matches it exactly.
    events = SEVIRDataset(root, split, image_size=NATIVE_SIZE).events
    events = sorted(events, key=lambda e: (str(e[0]), e[1]))  # sequential disk access

    out_path = cache_path(cache_dir, split, image_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8,
        shape=(len(events), FRAMES_PER_EVENT, image_size, image_size),
    )

    i = 0
    with tqdm(total=len(events), desc=f"cache {split}") as bar:
        for path, group in groupby(events, key=lambda e: e[0]):
            with h5py.File(path, "r") as f:
                vil = f["vil"]
                for _, idx in group:
                    raw = vil[int(idx)].astype(np.float32)  # (384, 384, 49)
                    pooled = raw.reshape(
                        image_size, factor, image_size, factor, FRAMES_PER_EVENT
                    ).mean(axis=(1, 3))  # (size, size, 49)
                    arr[i] = np.rint(pooled).astype(np.uint8).transpose(2, 0, 1)
                    i += 1
                    bar.update(1)
    arr.flush()

    meta = pd.DataFrame(
        {"file": [Path(p).name for p, _ in events], "file_index": [int(k) for _, k in events]}
    )
    meta.to_csv(out_path.with_suffix(".csv"), index=False)
    return out_path


def build_advection_cache(
    cache_dir: str | Path,
    split: str,
    image_size: int = 128,
    lookback: int = 4,
    winsize: int = 31,
    batch_size: int = 32,
) -> Path:
    """Precompute the advection forecast of every window (offsets 0, 12, 24) of one split.

    Output: <cache_dir>/adv_<split>_<size>.npy, uint8, shape (n_events, 3, 12, size, size).
    Each forecast only uses the 13 input frames of its own window (no future information).
    """
    from nowcasting.models.baselines import OpticalFlowAdvection  # needs OpenCV

    data = np.load(cache_path(cache_dir, split, image_size), mmap_mode="r")
    n_events = data.shape[0]
    out_path = advection_cache_path(cache_dir, split, image_size)
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(n_events, len(ADV_OFFSETS), 12, image_size, image_size)
    )
    model = OpticalFlowAdvection(out_len=12, lookback=lookback, winsize=winsize).eval()
    jobs = [(e, k) for e in range(n_events) for k in range(len(ADV_OFFSETS))]
    for start in tqdm(range(0, len(jobs), batch_size), desc=f"advection cache {split}"):
        chunk = jobs[start : start + batch_size]
        x = np.stack([data[e, ADV_OFFSETS[k] : ADV_OFFSETS[k] + 13] for e, k in chunk]).astype(np.float32) / 255.0
        with torch.no_grad():
            pred = model(torch.from_numpy(x).unsqueeze(2))  # (B, 12, 1, H, W)
        pred = np.rint(pred[:, :, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        for (e, k), frames in zip(chunk, pred, strict=True):
            arr[e, k] = frames
    arr.flush()
    out_path.with_suffix(".json").write_text(
        json.dumps({"lookback": lookback, "winsize": winsize, "offsets": list(ADV_OFFSETS)})
    )
    return out_path


class CachedSEVIRDataset(Dataset):
    """Same output as SEVIRDataset, read from the cache: (x, y) in [0, 1], shape (T, 1, H, W).

    With advection=True, the 12 advected frames precomputed by scripts/build_advection_cache.py are
    appended to the inputs along the time axis, so x has 13 + 12 = 25 frames. This needs the default
    13 -> 12 setup with stride 12, because the advection cache holds windows starting at 0, 12 and 24.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        in_len: int = 13,
        out_len: int = 12,
        stride: int = 12,
        image_size: int = 128,
        advection: bool = False,
    ) -> None:
        self.path = cache_path(cache_dir, split, image_size)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Build it with scripts/build_cache.py first."
            )
        self.advection = advection
        self.adv_path = advection_cache_path(cache_dir, split, image_size)
        if advection:
            if (in_len, out_len, stride) != (13, 12, 12):
                raise ValueError("advection input needs in_len=13, out_len=12 and stride=12")
            if not self.adv_path.exists():
                raise FileNotFoundError(
                    f"{self.adv_path} not found. Build it with scripts/build_advection_cache.py first."
                )
        self.in_len, self.out_len = in_len, out_len
        self.seq_len = in_len + out_len
        n_events = np.load(self.path, mmap_mode="r").shape[0]
        offsets = range(0, FRAMES_PER_EVENT - self.seq_len + 1, stride)
        self.index = [(e, o) for e in range(n_events) for o in offsets]
        self._data: np.ndarray | None = None  # opened lazily in each worker process
        self._adv: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.index)

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = np.load(self.path, mmap_mode="r")
        return self._data

    @property
    def adv(self) -> np.ndarray:
        if self._adv is None:
            self._adv = np.load(self.adv_path, mmap_mode="r")
        return self._adv

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        e, o = self.index[i]
        seq = torch.from_numpy(np.array(self.data[e, o : o + self.seq_len]))  # (T, H, W) uint8
        seq = seq.float().div_(255.0).unsqueeze(1)  # (T, 1, H, W)
        x, y = seq[: self.in_len], seq[self.in_len :]
        if self.advection:
            adv = torch.from_numpy(np.array(self.adv[e, ADV_OFFSETS.index(o)])).float().div_(255.0).unsqueeze(1)
            x = torch.cat([x, adv], dim=0)  # (25, 1, H, W)
        return x, y
