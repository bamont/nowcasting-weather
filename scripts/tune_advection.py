"""Grid search of the advection baseline hyper-parameters on the validation split.

Tune on 'val' only; never on 'test'. Writes:
  results/advection_sweep_val.json   all configurations with their scores
  results/advection_tuned_val.json   full metrics of the best one (same format as evaluate script)

Example:
    uv run python scripts/tune_advection.py --cache-dir data/cache
    uv run python scripts/tune_advection.py --lookbacks 1 2 3 4 6 --winsizes 15 31 --max-batches 20
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.metrics import THRESHOLDS, NowcastingMetrics
from nowcasting.models.baselines import OpticalFlowAdvection


def evaluate(model, loader, out_len: int, max_batches: int | None) -> dict[str, torch.Tensor]:
    metrics = NowcastingMetrics(out_len)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            metrics.update(model(x), y)
    return metrics.compute()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lookbacks", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    p.add_argument("--winsizes", type=int, nargs="+", default=[15, 31])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    ds = CachedSEVIRDataset(args.cache_dir, args.split, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows, best = [], None
    for lookback, winsize in tqdm(list(itertools.product(args.lookbacks, args.winsizes)), desc="sweep"):
        model = OpticalFlowAdvection(ds.out_len, lookback=lookback, winsize=winsize).eval()
        res = evaluate(model, loader, ds.out_len, args.max_batches)
        csi_m = float(torch.nanmean(res["csi_pooled"]))
        rows.append({
            "lookback": lookback, "winsize": winsize, "csi_m": csi_m,
            "mse": float(res["mse"].mean()), "csi_pooled": res["csi_pooled"].tolist(),
        })
        if best is None or csi_m > best[0]:
            best = (csi_m, lookback, winsize, res)

    print("\n lookback  winsize  CSI-M     MSE    " + " ".join(f"CSI@{t:<3d}" for t in THRESHOLDS))
    for r in sorted(rows, key=lambda r: -r["csi_m"]):
        print(f"{r['lookback']:9d} {r['winsize']:8d}  {r['csi_m']:.3f}  {r['mse']:.5f}  "
              + " ".join(f"{v:7.3f}" for v in r["csi_pooled"]))

    csi_m, lookback, winsize, res = best
    print(f"\nbest: lookback={lookback}, winsize={winsize} (CSI-M {csi_m:.3f})")
    edges = [(lookback, args.lookbacks), (winsize, args.winsizes)]
    if any(len(grid) > 1 and v in (min(grid), max(grid)) for v, grid in edges):
        print("note: the best value lies on the edge of the grid, consider extending it.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "advection_sweep_val.json").write_text(json.dumps(rows, indent=2))
    payload = {k: v.tolist() for k, v in res.items()}
    payload["thresholds"] = list(THRESHOLDS)
    payload["config"] = {"lookback": lookback, "winsize": winsize}
    (args.out_dir / "advection_tuned_val.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {args.out_dir}/advection_sweep_val.json and advection_tuned_val.json")


if __name__ == "__main__":
    main()
