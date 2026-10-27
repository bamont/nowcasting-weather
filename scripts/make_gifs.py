"""Side-by-side forecast comparison (GIF + static PNG): observation vs. each model.

Samples are taken from the cache. By default the most intense storms of the split are
shown ('intense'); use --select random for representative samples (report both in the README).

Examples:
    uv run python scripts/make_gifs.py --cache-dir data/cache --split val --n 3
    uv run python scripts/make_gifs.py --cache-dir data/cache --select random --n 3 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.metrics import THRESHOLDS
from nowcasting.models.loading import DEFAULT_UNETS, load_models

CMAP = "turbo"


def pick_samples(ds: CachedSEVIRDataset, n: int, mode: str, seed: int, thr: int = 133) -> list[int]:
    """Indices of n samples from n distinct events."""
    if mode == "random":
        order = np.random.default_rng(seed).permutation(len(ds)).tolist()
    else:  # most pixels at or above `thr` in the target frames
        scores = []
        for i in range(len(ds)):
            e, o = ds.index[i]
            y = ds.data[e, o + ds.in_len : o + ds.seq_len]
            scores.append(int((y >= thr).sum()))
        order = np.argsort(scores)[::-1].tolist()
    chosen, seen = [], set()
    for i in order:
        event = ds.index[i][0]
        if event not in seen:
            seen.add(event)
            chosen.append(i)
        if len(chosen) == n:
            break
    return chosen


def make_gif(path: Path, obs: np.ndarray, preds: dict[str, np.ndarray], fps: int = 3) -> None:
    arrays = {"Observation": obs, **preds}
    fig, axes = plt.subplots(1, len(arrays), figsize=(2.6 * len(arrays), 3.2))
    ims = []
    for ax, (name, arr) in zip(axes, arrays.items(), strict=True):
        ims.append(ax.imshow(arr[0], vmin=0, vmax=1, cmap=CMAP))
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    sup = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    def update(t: int):
        for im, arr in zip(ims, arrays.values(), strict=True):
            im.set_data(arr[t])
        sup.set_text(f"Lead time: +{5 * (t + 1)} min")
        return ims

    FuncAnimation(fig, update, frames=obs.shape[0]).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_png(path: Path, last_input: np.ndarray, obs: np.ndarray, preds: dict[str, np.ndarray],
             leads: tuple[int, ...] = (2, 5, 8, 11)) -> None:
    rows = {"Observation": obs, **preds}
    n_cols = 2 + len(leads)  # label, last input, leads
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(2.1 * n_cols, 2.1 * len(rows)), constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9] + [1] * (n_cols - 1)},
    )
    for r, (name, arr) in enumerate(rows.items()):
        for ax in axes[r]:
            ax.axis("off")
        axes[r, 0].text(0.95, 0.5, name, ha="right", va="center", fontsize=9,
                        transform=axes[r, 0].transAxes)
        if r == 0:
            axes[r, 1].imshow(last_input, vmin=0, vmax=1, cmap=CMAP)
            axes[r, 1].set_title("Last input", fontsize=8)
        for c, lead in enumerate(leads):
            axes[r, 2 + c].imshow(arr[lead], vmin=0, vmax=1, cmap=CMAP)
            if r == 0:
                axes[r, 2 + c].set_title(f"+{5 * (lead + 1)} min", fontsize=8)
    sm = ScalarMappable(norm=Normalize(0, 255), cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="VIL (0-255)")
    cbar.set_ticks(list(THRESHOLDS))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--select", choices=["intense", "random"], default="intense")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--indices", type=int, nargs="+", default=None, help="explicit sample indices")
    p.add_argument("--unet", action="append", help="'Label=checkpoint.ckpt' (repeatable)")
    p.add_argument("--diffusion", action="append", help="'Label=checkpoint.ckpt' (repeatable)")
    p.add_argument("--sample-steps", type=int, default=None, help="DDIM steps for diffusion models")
    p.add_argument("--lookback", type=int, default=3, help="advection lookback")
    p.add_argument("--winsize", type=int, default=15, help="advection Farneback window")
    p.add_argument("--out-dir", type=Path, default=Path("assets"))
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    ds = CachedSEVIRDataset(args.cache_dir, args.split, image_size=args.image_size)
    indices = args.indices or pick_samples(ds, args.n, args.select, args.seed)
    models = load_models(args.unet or DEFAULT_UNETS, ds.out_len, args.lookback, args.winsize,
                         args.diffusion, args.sample_steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        x, y = ds[i]
        with torch.no_grad():
            preds = {name: m(x[None])[0, :, 0].contiguous().numpy() for name, m in models.items()}
        obs = y[:, 0].numpy()
        tag = f"{args.split}_{args.select}_{i}"
        make_png(args.out_dir / f"forecast_{tag}.png", x[-1, 0].numpy(), obs, preds)
        if not args.no_gif:
            make_gif(args.out_dir / f"forecast_{tag}.gif", obs, preds)
        print(f"saved forecast_{tag}.png" + ("" if args.no_gif else " and .gif"))


if __name__ == "__main__":
    main()
