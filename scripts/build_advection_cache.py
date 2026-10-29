"""Precompute the advection forecast of every window, to feed it to the U-Net as extra input channels.

Example:
    uv run python scripts/build_advection_cache.py --cache-dir data/cache --lookback 4 --winsize 31
Output: <cache-dir>/adv_<split>_128.npy (about 2.1 GB for the three splits).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nowcasting.data.cache import build_advection_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--lookback", type=int, default=None, help="default: read results/advection_tuned_val.json, else 4")
    p.add_argument("--winsize", type=int, default=None, help="default: read results/advection_tuned_val.json, else 31")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    lookback, winsize = args.lookback, args.winsize
    tuned = Path("results/advection_tuned_val.json")
    if (lookback is None or winsize is None) and tuned.exists():
        cfg = json.loads(tuned.read_text()).get("config", {})
        lookback = lookback or cfg.get("lookback")
        winsize = winsize or cfg.get("winsize")
    lookback, winsize = lookback or 4, winsize or 31
    print(f"advection config: lookback={lookback}, winsize={winsize}")

    for split in args.splits:
        out = build_advection_cache(args.cache_dir, split, args.image_size, lookback, winsize, args.batch_size)
        print(f"{split}: {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
