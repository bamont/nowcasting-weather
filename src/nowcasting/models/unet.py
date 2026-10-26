"""A plain temporal U-Net: input frames are stacked as channels, output frames are channels."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """(B, T_in, 1, H, W) -> (B, T_out, 1, H, W), values in [0, 1]. H and W divisible by 2**depth."""

    def __init__(self, in_len: int = 13, out_len: int = 12, base: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        chans = [base * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_len, chans[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(chans[i], chans[i + 1])) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.up_convs = nn.ModuleList(
            DoubleConv(2 * chans[i], chans[i]) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(chans[0], out_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        if c != 1:
            raise ValueError("UNet expects single-channel frames")
        if h % 2**self.depth or w % 2**self.depth:
            raise ValueError(f"H and W must be divisible by {2**self.depth}")
        z = self.inc(x.reshape(b, t * c, h, w))
        skips = []
        for down in self.downs:
            skips.append(z)
            z = down(z)
        for up, conv in zip(self.ups, self.up_convs, strict=True):
            z = up(z)
            z = conv(torch.cat([z, skips.pop()], dim=1))
        return torch.sigmoid(self.head(z)).unsqueeze(2)
