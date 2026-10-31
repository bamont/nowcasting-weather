"""Create a small sample file (a few events) for the demo notebook, from the cache of one split.

The file is meant to be uploaded next to the model checkpoints (e.g. as release assets), so that the
notebook runs without downloading the dataset. Check the SEVIR terms of use before redistributing samples.

Example:
    uv run python scripts/make_demo_data.py --cache-dir data/cache --split test --out demo_samples.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-intense", type=int, default=4, help="events with the largest area above VIL 133")
    p.add_argument("--n-random", type=int, default=4, help="events drawn at random (representative cases)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("demo_samples.npz"))
    args = p.parse_args()

    data = np.load(args.cache_dir / f"{args.split}_{args.image_size}.npy", mmap_mode="r")  # (N, 49, H, W) uint8
    n = data.shape[0]
    if args.n_intense + args.n_random > n:
        raise SystemExit(f"only {n} events available")

    # area above VIL 133 over the 36 frames that can be targets (frames 13-48)
    scores = np.array([int((data[e, 13:] >= 133).sum()) for e in range(n)])
    intense = np.argsort(-scores, kind="stable")[: args.n_intense]
    rest = np.setdiff1d(np.arange(n), intense)
    random = np.random.default_rng(args.seed).choice(rest, size=args.n_random, replace=False)
    idx = np.concatenate([intense, random]).astype(int)

    frames = np.stack([np.asarray(data[e]) for e in idx])
    kind = np.array(["intense"] * len(intense) + ["random"] * len(random))
    np.savez_compressed(args.out, frames=frames, kind=kind, event_index=idx)
    print(f"saved {args.out}: {frames.shape}, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
