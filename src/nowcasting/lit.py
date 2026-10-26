"""LightningModule: trains the U-Net and tracks CSI/POD/FAR on the validation set."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nowcasting.metrics import THRESHOLDS, NowcastingMetrics
from nowcasting.models.unet import UNet


class LitNowcaster(pl.LightningModule):
    def __init__(
        self,
        in_len: int = 13,
        out_len: int = 12,
        base: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "mse",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_len, out_len, base)
        self.val_metrics = NowcastingMetrics(out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.hparams.loss == "mse":
            return F.mse_loss(pred, y)
        if self.hparams.loss == "weighted_mse":  # up-weights intense pixels (weight 1..6)
            return ((1.0 + 5.0 * y) * (pred - y) ** 2).mean()
        raise ValueError(f"unknown loss {self.hparams.loss!r}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metrics = NowcastingMetrics(self.hparams.out_len)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        self.log("val/loss", self._loss(pred, y), prog_bar=True)
        self.val_metrics.update(pred, y)

    def on_validation_epoch_end(self) -> None:
        res = self.val_metrics.compute()
        csi = res["csi_pooled"]
        self.log("val/csi_m", float(torch.nanmean(csi)), prog_bar=True)
        for k, thr in enumerate(THRESHOLDS):
            self.log(f"val/csi_{thr}", float(csi[k]))
        self.log("val/mse", float(res["mse"].mean()))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
