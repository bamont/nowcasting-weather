"""Average the forecasts of several independently trained U-Nets and score the mean.

Prints each member and the ensemble mean on the same samples (CSI-M, CSI and bias at 181 / 219, MSE).

Example (U-Nets trained with the advection forecast as input):
    python scripts/evaluate_seed_ensemble.py --cache-dir /tmp/sevir-cache --split val --advection-input \
        --checkpoints /kaggle/working/adv_s42/unet_mse_adv/version_0/checkpoints/final.ckpt \
                      /kaggle/working/adv_s43/unet_mse_adv/version_0/checkpoints/final.ckpt \
                      /kaggle/working/adv_s44/unet_mse_adv/version_0/checkpoints/final.ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.lit import LitNowcaster
from nowcasting.metrics import THRESHOLDS, NowcastingMetrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--advection-input", action="store_true", help="the checkpoints were trained with advection input")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("give at least two checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [LitNowcaster.load_from_checkpoint(str(c), map_location="cpu").model.to(device).eval()
              for c in args.checkpoints]
    ds = CachedSEVIRDataset(args.cache_dir, args.split, image_size=args.image_size, advection=args.advection_input)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    singles = [NowcastingMetrics(ds.out_len) for _ in models]
    mean_metrics = NowcastingMetrics(ds.out_len)
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc=f"seed ensemble / {args.split}")):
            if args.max_batches is not None and i >= args.max_batches:
                break
            preds = [m(x.to(device)) for m in models]
            for metric, pred in zip(singles, preds, strict=True):
                metric.update(pred, y)
            mean_metrics.update(torch.stack(preds).mean(dim=0), y)

    i181, i219 = THRESHOLDS.index(181), THRESHOLDS.index(219)
    print(f"\n{'forecast':20s} {'CSI-M':>6s} {'CSI@181':>8s} {'CSI@219':>8s} {'bias181':>8s} {'bias219':>8s} {'MSE':>9s}")
    out: dict = {"split": args.split, "checkpoints": [str(c) for c in args.checkpoints]}
    rows = [(f"member {k + 1}", m) for k, m in enumerate(singles)] + [(f"mean of {len(models)}", mean_metrics)]
    for name, m in rows:
        res, bias = m.compute(), m.frequency_bias()
        csi = res["csi_pooled"]
        print(f"{name:20s} {float(torch.nanmean(csi)):6.4f} {float(csi[i181]):8.4f} {float(csi[i219]):8.4f} "
              f"{float(bias[i181]):8.2f} {float(bias[i219]):8.2f} {float(res['mse'].mean()):9.5f}")
        out[name] = {k: v.tolist() for k, v in res.items()} | {"bias": bias.tolist()}

    path = args.out or Path(f"results/seed_ensemble_{args.split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
