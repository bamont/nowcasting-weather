"""Residual diffusion: a frozen U-Net gives the mean forecast mu, a conditional diffusion model
learns the residual r = y - mu (the small-scale detail that the MSE-trained U-Net smooths out).

Final forecast: clip(mu + res_std * z0, 0, 1) where z0 is a sample of the normalised residual.
Simplified two-stage variant inspired by DiffCast (Yu et al., CVPR 2024), which trains the
deterministic and diffusion parts jointly.
"""

from __future__ import annotations

import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nowcasting.metrics import THRESHOLDS, NowcastingMetrics
from nowcasting.models.diffusion import (
    DiffusionUNet,
    cosine_alphas_cumprod,
    ddim_sample,
    q_sample,
    v_target,
)
from nowcasting.models.unet import UNet
from nowcasting.sharpness import SpectrumAccumulator


class LitResidualDiffusion(pl.LightningModule):
    def __init__(
        self,
        in_len: int = 13,
        out_len: int = 12,
        backbone_base: int = 32,
        base: int = 64,
        cond_frames: int = 4,
        timesteps: int = 1000,
        sample_steps: int = 20,
        lr: float = 2e-4,
        ema_decay: float = 0.999,
        eval_batches: int = 4,
        res_std: float = 0.08,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = UNet(in_len, out_len, base=backbone_base)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.net = DiffusionUNet(in_ch=2 * out_len + cond_frames, out_ch=out_len, base=base)
        self.ema_net = copy.deepcopy(self.net)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(timesteps), persistent=False)
        self.register_buffer("res_std", torch.tensor(float(res_std)))
        self.val_metrics = NowcastingMetrics(out_len)
        self.bb_metrics = NowcastingMetrics(out_len)  # same samples, backbone forecast only
        self._spec_pred: SpectrumAccumulator | None = None
        self._spec_obs: SpectrumAccumulator | None = None
        self._spec_bb: SpectrumAccumulator | None = None
        self.backbone.eval()
        self.ema_net.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen: keep BatchNorm statistics fixed
        self.ema_net.eval()
        return self

    # ----------------------------------------------------------------------------- helpers
    def _cond(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Conditioning channels: backbone forecast + the last observed frames."""
        return torch.cat([mu[:, :, 0], x[:, -self.hparams.cond_frames :, 0]], dim=1)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu = self.backbone(x).float()
        z0 = (y - mu)[:, :, 0] / self.res_std  # normalised residual, (B, T, H, W)
        cond = self._cond(x, mu)
        b = z0.shape[0]
        t = torch.randint(0, self.hparams.timesteps, (b,), device=z0.device)
        noise = torch.randn_like(z0)
        ab = self.alphas_cumprod[t].view(b, 1, 1, 1)
        v_hat = self.net(torch.cat([q_sample(ab, z0, noise), cond], dim=1), t)
        return F.mse_loss(v_hat.float(), v_target(ab, z0, noise))

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n_members: int = 1, steps: int | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """x: (B, T_in, 1, H, W) -> ensemble forecast (B, n_members, T_out, 1, H, W) in [0, 1]."""
        steps = steps or self.hparams.sample_steps
        mu = self.backbone(x).float()
        cond = self._cond(x, mu)
        mu2 = mu[:, :, 0]
        lo, hi = -mu2 / self.res_std, (1.0 - mu2) / self.res_std  # keeps mu + residual inside [0, 1]

        def model_fn(z, t, c):
            return self.ema_net(torch.cat([z, c], dim=1), t)

        members = []
        for _ in range(n_members):
            z0 = ddim_sample(model_fn, self.alphas_cumprod, cond, tuple(mu2.shape), steps, lo, hi, generator)
            members.append((mu2 + self.res_std * z0).clamp(0.0, 1.0).unsqueeze(2))
        return torch.stack(members, dim=1)

    # ------------------------------------------------------------------------- lightning
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.compute_loss(x, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        step = self.global_step
        decay = min(self.hparams.ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for p_ema, p in zip(self.ema_net.parameters(), self.net.parameters(), strict=True):
                p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)

    def on_validation_epoch_start(self) -> None:
        self.val_metrics = NowcastingMetrics(self.hparams.out_len)
        self.bb_metrics = NowcastingMetrics(self.hparams.out_len)
        self._spec_pred = self._spec_obs = self._spec_bb = None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val/loss", self.compute_loss(x, y))
        if batch_idx < self.hparams.eval_batches:  # sampling is slow: only a few batches
            pred = self.sample(x, n_members=1)[:, 0]
            mu = self.backbone(x).float()
            self.val_metrics.update(pred, y)
            self.bb_metrics.update(mu, y)  # paired reference: the same samples without diffusion
            if self._spec_pred is None:
                n, size = self.hparams.out_len, y.shape[-1]
                self._spec_pred = SpectrumAccumulator(n, size)
                self._spec_obs = SpectrumAccumulator(n, size)
                self._spec_bb = SpectrumAccumulator(n, size)
            self._spec_pred.update(pred[:, :, 0].float().cpu().numpy())
            self._spec_obs.update(y[:, :, 0].float().cpu().numpy())
            self._spec_bb.update(mu[:, :, 0].cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        res, res_bb = self.val_metrics.compute(), self.bb_metrics.compute()
        self.log("val/csi_m", float(torch.nanmean(res["csi_pooled"])), prog_bar=True)
        self.log("val/csi_m_backbone", float(torch.nanmean(res_bb["csi_pooled"])))
        bias, bias_bb = self.val_metrics.frequency_bias(), self.bb_metrics.frequency_bias()
        for k, thr in enumerate(THRESHOLDS):
            self.log(f"val/csi_{thr}", float(res["csi_pooled"][k]))
            self.log(f"val/bias_{thr}", float(bias[k]))
            self.log(f"val/bias_bb_{thr}", float(bias_bb[k]))
        self.log("val/mse", float(res["mse"].mean()))
        if self._spec_pred is not None:
            obs = self._spec_obs.compute()

            def hf(acc: SpectrumAccumulator) -> float:  # small scales (k >= 16), all lead times
                return float(acc.compute()[:, 16:].sum() / obs[:, 16:].sum())

            self.log("val/hf_ratio", hf(self._spec_pred), prog_bar=True)
            self.log("val/hf_ratio_backbone", hf(self._spec_bb))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
