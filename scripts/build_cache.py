"""Build the compact uint8 cache (default 128x128) from the raw SEVIR files.

Example:
    uv run python scripts/build_cache.py --root data/sevir --cache-dir data/cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nowcasting.data.cache import build_split_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/sevir"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    for split in args.splits:
        out = build_split_cache(args.root, args.cache_dir, split, args.image_size)
        print(f"{split}: {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
