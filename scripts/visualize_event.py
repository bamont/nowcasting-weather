"""Save a GIF of one SEVIR VIL sample (inputs followed by targets).

Example:
    uv run python scripts/visualize_event.py --root data/sevir --split train --index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from nowcasting.data.sevir import SEVIRDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/sevir"))
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("assets/sample.gif"))
    args = p.parse_args()

    ds = SEVIRDataset(args.root, args.split, image_size=args.image_size)
    x, y = ds[args.index]
    seq = torch.cat([x, y]).squeeze(1).numpy()
    n_in = x.shape[0]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(seq[0], vmin=0, vmax=1, cmap="turbo")
    title = ax.set_title("")
    ax.axis("off")

    def update(t: int):
        im.set_data(seq[t])
        kind = "input" if t < n_in else "target"
        title.set_text(f"{kind} (t{(t - n_in + 1) * 5:+d} min)")
        return im, title

    args.out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(seq)).save(args.out, writer=PillowWriter(fps=4))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
