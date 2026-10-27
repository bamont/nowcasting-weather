"""Building blocks for the residual diffusion model (pixel space, v-prediction, DDIM sampling).

The 12 forecast frames are treated as 12 channels (like the U-Net). The network receives the
noisy residual concatenated with conditioning channels and the diffusion step.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


# ------------------------------------------------------------------------------- schedule
def cosine_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cumulative alphas (alpha_bar) of the cosine schedule (Nichol & Dhariwal), shape (T,)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


def q_sample(ab: torch.Tensor, x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Noisy sample z_t = sqrt(ab) x0 + sqrt(1 - ab) noise. `ab` broadcastable to x0."""
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def v_target(ab: torch.Tensor, x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """v-prediction target (Salimans & Ho): v = sqrt(ab) noise - sqrt(1 - ab) x0."""
    return ab.sqrt() * noise - (1 - ab).sqrt() * x0


def x0_from_v(ab: torch.Tensor, z_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return ab.sqrt() * z_t - (1 - ab).sqrt() * v


def ddim_sample(
    model_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    alphas_cumprod: torch.Tensor,
    cond: torch.Tensor,
    shape: tuple[int, ...],
    steps: int,
    lo: torch.Tensor | None = None,
    hi: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Deterministic DDIM sampling. model_fn(z_t, t, cond) returns the predicted v.

    `lo` / `hi` optionally clamp the running estimate of x0 per pixel (feasible range).
    Returns the final x0 estimate, shape `shape`.
    """
    device = cond.device
    n_train = alphas_cumprod.shape[0]
    ts = torch.linspace(n_train - 1, 0, steps, device=device).round().long()
    z = torch.randn(shape, device=device, generator=generator)
    for i, t in enumerate(ts):
        ab_t = alphas_cumprod[t]
        ab_prev = alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=device)
        v = model_fn(z, t.expand(shape[0]), cond).float()
        x0 = x0_from_v(ab_t, z, v)
        if lo is not None and hi is not None:
            x0 = torch.maximum(torch.minimum(x0, hi), lo)
        eps = (z - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        z = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return z


# ------------------------------------------------------------------------------- network
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.n1 = nn.GroupNorm(groups, c_in)
        self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t = nn.Linear(t_dim, c_out)
        self.n2 = nn.GroupNorm(groups, c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(t_emb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, c: int, heads: int = 4, groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        z, _ = self.attn(z, z, z, need_weights=False)
        return x + z.transpose(1, 2).reshape(b, c, h, w)


class Upsample(nn.Sequential):
    def __init__(self, c: int) -> None:
        super().__init__(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1))


class DiffusionUNet(nn.Module):
    """(B, in_ch, H, W), t (B,) -> (B, out_ch, H, W). H and W must be divisible by 2**(levels-1)."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 64,
                 mults: tuple[int, ...] = (1, 2, 3, 4), groups: int = 8) -> None:
        super().__init__()
        self.base = base
        self.n_levels = len(mults)
        t_dim = base * 4
        chs = [base * m for m in mults]
        n = self.n_levels

        self.time_mlp = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inc = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc = nn.ModuleList(ResBlock(chs[max(i - 1, 0)], chs[i], t_dim, groups) for i in range(n))
        self.down = nn.ModuleList(nn.Conv2d(chs[i], chs[i], 3, stride=2, padding=1) for i in range(n - 1))
        self.mid1 = ResBlock(chs[-1], chs[-1], t_dim, groups)
        self.attn = SelfAttention(chs[-1], groups=groups)
        self.mid2 = ResBlock(chs[-1], chs[-1], t_dim, groups)
        self.dec = nn.ModuleList(ResBlock(2 * chs[i], chs[max(i - 1, 0)], t_dim, groups) for i in range(n))
        self.up = nn.ModuleList(Upsample(chs[j]) for j in range(n - 1))
        self.out = nn.Sequential(nn.GroupNorm(groups, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        factor = 2 ** (self.n_levels - 1)
        if x.shape[-1] % factor or x.shape[-2] % factor:
            raise ValueError(f"H and W must be divisible by {factor}")
        emb = self.time_mlp(timestep_embedding(t, self.base))
        x = self.inc(x)
        skips = []
        for i in range(self.n_levels):
            x = self.enc[i](x, emb)
            skips.append(x)
            if i < self.n_levels - 1:
                x = self.down[i](x)
        x = self.mid2(self.attn(self.mid1(x, emb)), emb)
        for i in reversed(range(self.n_levels)):
            x = self.dec[i](torch.cat([x, skips.pop()], dim=1), emb)
            if i > 0:
                x = self.up[i - 1](x)
        return self.out(x)
