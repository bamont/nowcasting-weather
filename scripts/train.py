"""Train the temporal U-Net on the SEVIR cache.

Examples:
    uv run python scripts/train.py --cache-dir data/cache --debug      # quick smoke test
    python scripts/train.py --cache-dir /kaggle/input/sevir-cache --epochs 20 --num-workers 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from nowcasting.data.cache import CachedSEVIRDataset
from nowcasting.lit import LitNowcaster


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32, help="U-Net base channels")
    p.add_argument("--loss", choices=["mse", "weighted_mse"], default="mse")
    p.add_argument("--train-stride", type=int, default=12, help="smaller = more (overlapping) samples")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true", help="1 epoch, 10 train / 5 val batches")
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)
    train_ds = CachedSEVIRDataset(args.cache_dir, "train", stride=args.train_stride,
                                  image_size=args.image_size)
    val_ds = CachedSEVIRDataset(args.cache_dir, "val", image_size=args.image_size)
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **loader_kwargs)
    print(f"train samples: {len(train_ds)}   val samples: {len(val_ds)}")

    model = LitNowcaster(base=args.base, lr=args.lr, loss=args.loss)
    ckpt = ModelCheckpoint(monitor="val/csi_m", mode="max", save_top_k=1,
                           filename="best-{epoch:02d}", save_last=True)
    trainer = pl.Trainer(
        max_epochs=1 if args.debug else args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger(str(args.out_dir), name=f"unet_{args.loss}"),
        limit_train_batches=10 if args.debug else 1.0,
        limit_val_batches=5 if args.debug else 1.0,
        log_every_n_steps=20,
    )
    trainer.fit(model, train_loader, val_loader)
    print("best checkpoint:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
