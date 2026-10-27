"""Build the dict of models compared in the evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from nowcasting.models.baselines import OpticalFlowAdvection, Persistence

DEFAULT_UNETS = [
    "U-Net (MSE)=checkpoints/unet_mse_best.ckpt",
    "U-Net (weighted MSE)=checkpoints/unet_weighted_mse_best.ckpt",
]


class SingleSampleForecaster(nn.Module):
    """Wraps the residual diffusion model: one random sample per input, same interface as the others."""

    def __init__(self, lit: nn.Module, steps: int | None = None, seed: int = 0) -> None:
        super().__init__()
        self.lit, self.steps, self.seed = lit, steps, seed
        self._gen: torch.Generator | None = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gen is None or self._gen.device != x.device:
            self._gen = torch.Generator(device=x.device)
            self._gen.manual_seed(self.seed)
        return self.lit.sample(x, n_members=1, steps=self.steps, generator=self._gen)[:, 0]


def load_diffusion(path: str | Path, steps: int | None = None) -> nn.Module:
    from nowcasting.lit_diffusion import LitResidualDiffusion

    lit = LitResidualDiffusion.load_from_checkpoint(str(path), map_location="cpu").eval()
    return SingleSampleForecaster(lit, steps=steps)


def load_models(
    unet_specs: list[str],
    out_len: int,
    lookback: int = 4,
    winsize: int = 31,
    diffusion_specs: list[str] | None = None,
    sample_steps: int | None = None,
) -> dict[str, nn.Module]:
    """Specs are 'Label=path/to/checkpoint.ckpt'. Missing checkpoints are skipped with a warning."""
    from nowcasting.lit import LitNowcaster

    models: dict[str, nn.Module] = {
        "Persistence": Persistence(out_len),
        "Advection": OpticalFlowAdvection(out_len, lookback=lookback, winsize=winsize),
    }
    for spec in unet_specs:
        label, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            print(f"warning: checkpoint for '{label}' not found ({path!r}), skipping")
            continue
        models[label] = LitNowcaster.load_from_checkpoint(path, map_location="cpu").model.eval()
    for spec in diffusion_specs or []:
        label, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            print(f"warning: checkpoint for '{label}' not found ({path!r}), skipping")
            continue
        models[label] = load_diffusion(path, sample_steps)
    return models
