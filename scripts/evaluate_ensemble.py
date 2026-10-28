"""Ensemble evaluation of the residual diffusion model: CRPS, spread-skill, Brier, ensemble-mean scores.

Compares, on the same samples: the U-Net backbone alone, one diffusion member, and the ensemble mean.
Run it on a GPU (about 2 s per batch per member).

Example:
    python scripts/evaluate_ensemble.py --checkpoint diffusion_final.ckpt --cache-dir /tmp/sevir-cache \
        --members 8 --out ensemble_val.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.ensemble import EnsembleScores
from nowcasting.lit_diffusion import LitResidualDiffusion
from nowcasting.metrics import THRESHOLDS, NowcastingMetrics

BRIER_THRESHOLDS = (74, 133, 181)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.members < 2:
        raise SystemExit("--members must be at least 2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit = LitResidualDiffusion.load_from_checkpoint(str(args.checkpoint), map_location=device).to(device).eval()
    ds = CachedSEVIRDataset(args.cache_dir, args.split, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    out_len = ds.out_len
    cat = {name: NowcastingMetrics(out_len) for name in ("backbone", "member", "mean")}
    scores = EnsembleScores(out_len, BRIER_THRESHOLDS)

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc=f"ensemble x{args.members} / {args.split}")):
            if args.max_batches is not None and i >= args.max_batches:
                break
            x = x.to(device)
            members = lit.sample(x, n_members=args.members, steps=args.sample_steps, generator=gen)  # (B,M,T,1,H,W)
            mu = lit.backbone(x).float()
            cat["backbone"].update(mu, y)
            cat["member"].update(members[:, 0], y)
            cat["mean"].update(members.mean(dim=1), y)
            scores.update(members[:, :, :, 0].cpu().numpy(), y[:, :, 0].numpy(), reference=mu[:, :, 0].cpu().numpy())

    names = {"backbone": "U-Net backbone", "member": "Diffusion, 1 member", "mean": f"Diffusion, mean of {args.members}"}
    i181, i219 = THRESHOLDS.index(181), THRESHOLDS.index(219)
    print(f"\n{'forecast':26s} {'CSI-M':>6s} {'CSI@181':>8s} {'CSI@219':>8s} {'bias181':>8s} {'bias219':>8s} {'MSE':>8s}")
    out: dict = {"members": args.members, "split": args.split, "thresholds": list(THRESHOLDS)}
    for key, m in cat.items():
        res, bias = m.compute(), m.frequency_bias()
        csi = res["csi_pooled"]
        print(f"{names[key]:26s} {float(torch.nanmean(csi)):6.3f} {float(csi[i181]):8.3f} {float(csi[i219]):8.3f} "
              f"{float(bias[i181]):8.2f} {float(bias[i219]):8.2f} {float(res['mse'].mean()):8.5f}")
        out[key] = {k: v.tolist() for k, v in res.items()} | {"bias": bias.tolist()}

    r = scores.compute()
    leads = sorted({min(5, out_len - 1), out_len - 1})
    print("\nCRPS of the ensemble vs MAE of the deterministic U-Net (CRPS = MAE for a deterministic forecast)")
    for t in leads:
        print(f"  +{5 * (t + 1):2d} min: CRPS {r['crps'][t]:.4f}   U-Net MAE {r['mae_reference'][t]:.4f}   "
              f"spread/skill {r['spread_skill'][t]:.2f}")
    print(f"  all leads: CRPS {r['crps'].mean():.4f}   U-Net MAE {r['mae_reference'].mean():.4f}   "
          f"spread/skill {r['spread_skill'].mean():.2f}  (1 = reliable, < 1 = under-dispersive)")
    print("\nBrier score (lower is better), ensemble probability vs deterministic U-Net 0/1 forecast")
    for k, thr in enumerate(BRIER_THRESHOLDS):
        print(f"  threshold {thr:3d}: ensemble {r['brier'][:, k].mean():.4f}   U-Net {r['brier_reference'][:, k].mean():.4f}")
    out["ensemble_scores"] = {k: np.asarray(v).tolist() for k, v in r.items()}
    out["brier_thresholds"] = list(BRIER_THRESHOLDS)

    path = args.out or Path(f"results/ensemble_{args.split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
