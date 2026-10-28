"""Sharpness diagnostics: radial power spectrum and Fractions Skill Score, per model.

Examples:
    uv run python scripts/evaluate_sharpness.py --cache-dir data/cache --split val
    uv run python scripts/evaluate_sharpness.py --max-batches 5          # quick check
    uv run python scripts/evaluate_sharpness.py --plot-only              # redraw from the JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.models.loading import DEFAULT_UNETS, load_models
from nowcasting.sharpness import FSSAccumulator, SpectrumAccumulator

SCALES = (1, 3, 9, 17, 33)  # neighbourhood sizes in pixels (about 3 km per pixel at 128x128)
BANDS = {"large (k 1-4)": (1, 4), "medium (k 5-15)": (5, 15), "small (k 16-64)": (16, 64)}


def read_advection_config(path: Path) -> tuple[int, int]:
    if path.exists():
        cfg = json.loads(path.read_text()).get("config", {})
        if "lookback" in cfg and "winsize" in cfg:
            print(f"advection config from {path}: {cfg}")
            return cfg["lookback"], cfg["winsize"]
    print("advection config not found, using lookback=4, winsize=31")
    return 4, 31


def compute(args: argparse.Namespace) -> dict:
    ds = CachedSEVIRDataset(args.cache_dir, args.split, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    lookback, winsize = read_advection_config(args.advection_config)
    unets = args.unet or ([] if args.no_baselines else DEFAULT_UNETS)
    models = load_models(unets, ds.out_len, lookback, winsize, args.diffusion, args.sample_steps,
                         include_baselines=not args.no_baselines)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {name: m.to(device).eval() for name, m in models.items()}

    obs_spec = SpectrumAccumulator(ds.out_len, args.image_size)
    spec = {n: SpectrumAccumulator(ds.out_len, args.image_size) for n in models}
    fss = {n: FSSAccumulator(ds.out_len, tuple(args.fss_thresholds), SCALES) for n in models}
    n_seen = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc=f"sharpness / {args.split}")):
            if args.max_batches is not None and i >= args.max_batches:
                break
            obs = y[:, :, 0].numpy()
            obs_spec.update(obs)
            n_seen += len(obs)
            obs_frac = next(iter(fss.values())).observed_fractions(obs)
            x_dev = x.to(device)
            for name, model in models.items():
                pred = model(x_dev)[:, :, 0].contiguous().cpu().numpy()
                spec[name].update(pred)
                fss[name].update(pred, obs, obs_frac)

    return {
        "split": args.split, "n_samples": n_seen, "scales": list(SCALES),
        "fss_thresholds": list(args.fss_thresholds),
        "k": list(range(obs_spec.n_bins)),
        "observation": {"spectrum": obs_spec.compute().tolist()},
        "models": {
            n: {"spectrum": spec[n].compute().tolist(), "fss": fss[n].compute().tolist()}
            for n in models
        },
    }


def band_ratio(pred_spec: np.ndarray, obs_spec: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Predicted / observed power summed over wavenumbers lo..hi, per lead time."""
    return pred_spec[:, lo : hi + 1].sum(axis=1) / obs_spec[:, lo : hi + 1].sum(axis=1)


def lead_indices(n_leads: int) -> list[int]:
    return sorted({min(5, n_leads - 1), n_leads - 1})  # +30 min and +60 min


def plot_spectra(res: dict, out: Path) -> None:
    obs = np.array(res["observation"]["spectrum"])
    k = np.array(res["k"])[1:]
    leads = lead_indices(obs.shape[0])
    fig, axes = plt.subplots(1, len(leads) + 1, figsize=(5 * (len(leads) + 1), 4.2))
    for ax, t in zip(axes, leads, strict=False):
        for name, m in res["models"].items():
            ax.plot(k, (np.array(m["spectrum"])[t] / obs[t])[1:], label=name)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set(xscale="log", yscale="log", xlabel="Wavenumber k (cycles per image)",
               ylabel="Power, forecast / observation", title=f"Lead time +{5 * (t + 1)} min")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    t = leads[-1]
    ax = axes[-1]
    ax.plot(k, obs[t][1:], color="k", lw=2, label="Observation")
    for name, m in res["models"].items():
        ax.plot(k, np.array(m["spectrum"])[t][1:], label=name)
    ax.set(xscale="log", yscale="log", xlabel="Wavenumber k (cycles per image)", ylabel="Power",
           title=f"Absolute spectra, +{5 * (t + 1)} min")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_fss(res: dict, out: Path) -> None:
    thresholds, scales = res["fss_thresholds"], res["scales"]
    n_leads = np.array(res["observation"]["spectrum"]).shape[0]
    t = n_leads - 1
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.6 * len(thresholds), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for j, (ax, thr) in enumerate(zip(axes, thresholds, strict=True)):
        for name, m in res["models"].items():
            ax.plot(scales, np.array(m["fss"], dtype=float)[t, j], marker="o", label=name)
        ax.set(xscale="log", xlabel="Neighbourhood size (pixels)", title=f"Threshold {thr}, +{5 * (t + 1)} min")
        ax.set_xticks(scales)
        ax.set_xticklabels(scales)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("FSS")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def print_summary(res: dict) -> None:
    obs = np.array(res["observation"]["spectrum"])
    leads = lead_indices(obs.shape[0])
    print("\nPower ratio forecast / observation (1 = same amount of detail, <1 = blurrier)")
    print(f"{'model':24s} {'lead':>5s} " + " ".join(f"{b:>16s}" for b in BANDS))
    for name, m in res["models"].items():
        for t in leads:
            ratios = [band_ratio(np.array(m["spectrum"]), obs, lo, hi)[t] for lo, hi in BANDS.values()]
            print(f"{name:24s} +{5 * (t + 1):<3d}m " + " ".join(f"{r:16.3f}" for r in ratios))

    t = obs.shape[0] - 1
    scales = res["scales"]
    print(f"\nFSS at +{5 * (t + 1)} min (rows: model / threshold, columns: neighbourhood size in pixels)")
    print(f"{'model':24s} {'thr':>4s} " + " ".join(f"{s:>6d}" for s in scales))
    for name, m in res["models"].items():
        fss = np.array(m["fss"], dtype=float)
        for j, thr in enumerate(res["fss_thresholds"]):
            print(f"{name:24s} {thr:4d} " + " ".join(f"{v:6.3f}" for v in fss[t, j]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--unet", action="append", help="'Label=checkpoint.ckpt' (repeatable)")
    p.add_argument("--diffusion", action="append", help="'Label=checkpoint.ckpt' (repeatable)")
    p.add_argument("--sample-steps", type=int, default=None, help="DDIM steps for diffusion models")
    p.add_argument("--no-baselines", action="store_true", help="skip persistence, advection and default U-Nets")
    p.add_argument("--advection-config", type=Path, default=Path("results/advection_tuned_val.json"))
    p.add_argument("--fss-thresholds", type=int, nargs="+", default=[74, 133, 181])
    p.add_argument("--out", type=Path, default=None, help="JSON path (default results/sharpness_<split>.json)")
    p.add_argument("--fig-dir", type=Path, default=Path("assets"))
    p.add_argument("--plot-only", action="store_true", help="redraw from an existing JSON")
    args = p.parse_args()

    out = args.out or Path(f"results/sharpness_{args.split}.json")
    if args.plot_only:
        res = json.loads(out.read_text())
    else:
        res = compute(args)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res))
        print(f"saved {out}")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_spectra(res, args.fig_dir / "sharpness_spectrum.png")
    plot_fss(res, args.fig_dir / "sharpness_fss.png")
    print_summary(res)


if __name__ == "__main__":
    main()
