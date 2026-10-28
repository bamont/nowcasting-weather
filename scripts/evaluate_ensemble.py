"""Ensemble evaluation of the residual diffusion model: CRPS, spread-skill, Brier, ensemble-mean scores.

Compares, on the same samples: the U-Net backbone alone, one diffusion member, and the ensemble mean.
With --gaussian-baseline it also scores a control ensemble that perturbs the U-Net forecast with
independent Gaussian noise (std fitted per lead time and per intensity bin on training data). This
tells whether the diffusion model brings more than "some spread around the U-Net".
Run it on a GPU (about 2 s per batch per member).

Example:
    python scripts/evaluate_ensemble.py --checkpoint diffusion_final.ckpt --cache-dir /tmp/sevir-cache \
        --members 8 --gaussian-baseline --out ensemble_val.json
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


@torch.no_grad()
def fit_sigma(lit: LitResidualDiffusion, loader: DataLoader, device: str, n_batches: int, n_bins: int) -> torch.Tensor:
    """RMS error of the backbone per lead time and per forecast-intensity bin, from training data: (T, n_bins)."""
    sq = torch.zeros(lit.hparams.out_len, n_bins, dtype=torch.float64, device=device)
    cnt = torch.zeros_like(sq)
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        mu = lit.backbone(x.to(device)).float()[:, :, 0]
        err2 = ((y.to(device)[:, :, 0] - mu) ** 2).double()
        idx = (mu * n_bins).long().clamp(0, n_bins - 1)
        for t in range(mu.shape[1]):
            flat = idx[:, t].flatten()
            sq[t].index_add_(0, flat, err2[:, t].flatten())
            cnt[t].index_add_(0, flat, torch.ones_like(err2[:, t].flatten()))
    return (sq / cnt.clamp(min=1)).sqrt().float()


def gaussian_members(mu: torch.Tensor, sigma: torch.Tensor, n_members: int, gen: torch.Generator) -> torch.Tensor:
    """mu (B, T, 1, H, W) -> (B, M, T, H, W): mu + sigma(lead, intensity) * N(0, 1), clipped to [0, 1]."""
    mu2 = mu[:, :, 0]
    b = mu2.shape[0]
    idx = (mu2 * sigma.shape[1]).long().clamp(0, sigma.shape[1] - 1)
    sig = torch.gather(sigma.unsqueeze(0).expand(b, -1, -1), 2, idx.flatten(2)).view_as(mu2)
    noise = torch.randn((b, n_members, *mu2.shape[1:]), device=mu.device, generator=gen)
    return (mu2.unsqueeze(1) + sig.unsqueeze(1) * noise).clamp(0.0, 1.0)


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
    p.add_argument("--gaussian-baseline", action="store_true", help="also score the Gaussian-noise control ensemble")
    p.add_argument("--sigma-batches", type=int, default=30, help="training batches used to fit the noise std")
    p.add_argument("--sigma-bins", type=int, default=8)
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

    sigma = None
    if args.gaussian_baseline:
        train_ds = CachedSEVIRDataset(args.cache_dir, "train", image_size=args.image_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        sigma = fit_sigma(lit, train_loader, device, args.sigma_batches, args.sigma_bins)
        print(f"noise std fitted on {args.sigma_batches} training batches, mean over lead/intensity: {sigma.mean():.4f}")

    out_len = ds.out_len
    cat = {name: NowcastingMetrics(out_len) for name in ("backbone", "member", "mean")}
    scores = EnsembleScores(out_len, BRIER_THRESHOLDS)
    scores_g = EnsembleScores(out_len, BRIER_THRESHOLDS) if sigma is not None else None

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc=f"ensemble x{args.members} / {args.split}")):
            if args.max_batches is not None and i >= args.max_batches:
                break
            x = x.to(device)
            members = lit.sample(x, n_members=args.members, steps=args.sample_steps, generator=gen)  # (B,M,T,1,H,W)
            mu = lit.backbone(x).float()
            obs, ref = y[:, :, 0].numpy(), mu[:, :, 0].cpu().numpy()
            cat["backbone"].update(mu, y)
            cat["member"].update(members[:, 0], y)
            cat["mean"].update(members.mean(dim=1), y)
            scores.update(members[:, :, :, 0].cpu().numpy(), obs, reference=ref)
            if scores_g is not None:
                scores_g.update(gaussian_members(mu, sigma, args.members, gen).cpu().numpy(), obs, reference=ref)

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
    header = f"{'forecast':30s} {'+30min':>7s} {'+60min':>7s} {'all':>7s} {'spr/skill':>9s} " + " ".join(
        f"{'Brier' + str(t):>9s}" for t in BRIER_THRESHOLDS)
    print("\nCRPS (a deterministic forecast has CRPS = MAE) and Brier score: lower is better")
    print(header)

    def row(name: str, crps: np.ndarray, spread_skill: float | None, brier: np.ndarray) -> None:
        ss = f"{spread_skill:9.2f}" if spread_skill is not None else f"{'-':>9s}"
        print(f"{name:30s} {crps[leads[0]]:7.4f} {crps[leads[-1]]:7.4f} {crps.mean():7.4f} {ss} "
              + " ".join(f"{b:9.4f}" for b in brier.mean(axis=0)))

    row("U-Net (deterministic: MAE)", r["mae_reference"], None, r["brier_reference"])
    row(f"Diffusion ensemble (M={args.members})", r["crps"], float(r["spread_skill"].mean()), r["brier"])
    out["ensemble_scores"] = {k: np.asarray(v).tolist() for k, v in r.items()}
    if scores_g is not None:
        rg = scores_g.compute()
        row(f"Gaussian noise on U-Net (M={args.members})", rg["crps"], float(rg["spread_skill"].mean()), rg["brier"])
        out["gaussian_baseline_scores"] = {k: np.asarray(v).tolist() for k, v in rg.items()}
    print("spread/skill: 1 = reliable ensemble, < 1 = under-dispersive")
    out["brier_thresholds"] = list(BRIER_THRESHOLDS)

    path = args.out or Path(f"results/ensemble_{args.split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
