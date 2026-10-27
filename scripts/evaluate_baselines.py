"""Evaluate a model (baseline or trained U-Net) on a SEVIR split: CSI/POD/FAR/MSE.

Examples:
    uv run python scripts/evaluate_baselines.py --model persistence --split val
    uv run python scripts/evaluate_baselines.py --model advection --cache-dir data/cache --split val
    python scripts/evaluate_baselines.py --model unet --checkpoint best.ckpt --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.data.sevir import SEVIRDataset
from nowcasting.metrics import THRESHOLDS, NowcastingMetrics
from nowcasting.models.baselines import OpticalFlowAdvection, Persistence


def fmt(v: float) -> str:
    return "  n/a " if v != v else f"{v:6.3f}"


def build_model(args: argparse.Namespace, out_len: int) -> nn.Module:
    if args.model == "persistence":
        return Persistence(out_len)
    if args.model == "advection":
        return OpticalFlowAdvection(out_len, lookback=args.lookback, winsize=args.winsize)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for --model unet / diffusion")
    if args.model == "diffusion":
        from nowcasting.models.loading import load_diffusion

        return load_diffusion(args.checkpoint, args.sample_steps)
    from nowcasting.lit import LitNowcaster

    return LitNowcaster.load_from_checkpoint(args.checkpoint, map_location="cpu").model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["persistence", "advection", "unet", "diffusion"], default="persistence")
    p.add_argument("--checkpoint", type=Path, default=None, help="Lightning .ckpt (unet only)")
    p.add_argument("--root", type=Path, default=Path("data/sevir"), help="raw SEVIR (no cache)")
    p.add_argument("--cache-dir", type=Path, default=None, help="use the compact cache instead")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lookback", type=int, default=3, help="advection only")
    p.add_argument("--winsize", type=int, default=15, help="advection only")
    p.add_argument("--sample-steps", type=int, default=None, help="DDIM steps (diffusion only)")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="JSON results path")
    args = p.parse_args()

    if args.cache_dir:
        ds = CachedSEVIRDataset(args.cache_dir, args.split, image_size=args.image_size)
    else:
        ds = SEVIRDataset(args.root, args.split, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, ds.out_len).to(device).eval()
    metrics = NowcastingMetrics(ds.out_len)

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc=f"{args.model} / {args.split}")):
            if args.max_batches is not None and i >= args.max_batches:
                break
            metrics.update(model(x.to(device)), y)

    res = metrics.compute()
    print(f"\n{args.model} on '{args.split}' ({args.image_size}x{args.image_size}), "
          f"scores pooled over all lead times")
    print("threshold |   CSI    POD    FAR")
    for k, thr in enumerate(THRESHOLDS):
        print(f"{thr:9d} | {fmt(res['csi_pooled'][k].item())} "
              f"{fmt(res['pod_pooled'][k].item())} {fmt(res['far_pooled'][k].item())}")
    print(f"\nCSI-M (mean over thresholds): {torch.nanmean(res['csi_pooled']).item():.3f}")
    print(f"MSE: {res['mse'].mean().item():.5f}   MAE: {res['mae'].mean().item():.5f}")
    print("CSI@16 by lead time (min):",
          {5 * (t + 1): round(res["csi"][t, 0].item(), 3) for t in (0, 5, 11)})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.tolist() for k, v in res.items()}
        payload["thresholds"] = list(THRESHOLDS)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
