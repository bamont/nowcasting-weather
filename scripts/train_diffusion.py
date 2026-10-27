"""Train the residual diffusion model on top of a frozen, pre-trained U-Net.

Examples:
    # local smoke test (CPU, tiny network)
    uv run python scripts/train_diffusion.py --backbone-ckpt checkpoints/unet_mse_best.ckpt \
        --cache-dir data/cache --debug --base 16 --num-workers 0
    # Kaggle: first measure the time per epoch with --epochs 2
    python scripts/train_diffusion.py --backbone-ckpt /kaggle/input/<ds>/unet_mse_best.ckpt \
        --cache-dir /tmp/sevir-cache --out-dir /kaggle/working --epochs 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.lit import LitNowcaster
from nowcasting.lit_diffusion import LitResidualDiffusion


@torch.no_grad()
def estimate_res_std(backbone: torch.nn.Module, loader: DataLoader, device: str, n_batches: int) -> float:
    """Root-mean-square of y - mu on training samples, used to normalise the residual."""
    backbone = backbone.to(device).eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        r = y - backbone(x.to(device)).float().cpu()
        total += float((r**2).sum())
        count += r.numel()
    return (total / count) ** 0.5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone-ckpt", type=Path, required=True, help="Lightning checkpoint of the U-Net")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64, help="diffusion U-Net base channels (multiple of 8)")
    p.add_argument("--cond-frames", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=20, help="DDIM steps used in validation")
    p.add_argument("--eval-batches", type=int, default=4, help="validation batches that are sampled")
    p.add_argument("--train-stride", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true", help="1 epoch, 10 train / 5 val batches")
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)
    train_ds = CachedSEVIRDataset(args.cache_dir, "train", stride=args.train_stride, image_size=args.image_size)
    val_ds = CachedSEVIRDataset(args.cache_dir, "val", image_size=args.image_size)
    kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **kw)
    print(f"train samples: {len(train_ds)}   val samples: {len(val_ds)}")

    backbone = LitNowcaster.load_from_checkpoint(args.backbone_ckpt, map_location="cpu")
    module = LitResidualDiffusion(
        backbone_base=backbone.hparams.base, base=args.base, cond_frames=args.cond_frames,
        sample_steps=args.sample_steps, lr=args.lr, eval_batches=1 if args.debug else args.eval_batches,
    )
    module.backbone.load_state_dict(backbone.model.state_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    std = estimate_res_std(module.backbone, train_loader, device, n_batches=3 if args.debug else 30)
    module.res_std.fill_(std)
    module.backbone.cpu()
    print(f"residual RMS (normalisation constant): {std:.4f}")

    ckpt = ModelCheckpoint(monitor="val/csi_m", mode="max", save_top_k=1,
                           filename="best-{epoch:02d}", save_last=True)
    trainer = pl.Trainer(
        max_epochs=1 if args.debug else args.epochs,
        accelerator="auto", devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[ckpt],
        logger=CSVLogger(str(args.out_dir), name="diffusion"),
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        limit_train_batches=10 if args.debug else 1.0,
        limit_val_batches=5 if args.debug else 1.0,
        log_every_n_steps=20,
    )
    trainer.fit(module, train_loader, val_loader)

    final = Path(trainer.logger.log_dir) / "checkpoints" / "final.ckpt"
    trainer.save_checkpoint(str(final))
    print(f"final checkpoint: {final} | global_step {trainer.global_step}")
    print("best checkpoint:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
