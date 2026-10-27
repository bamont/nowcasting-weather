import torch

from nowcasting.lit_diffusion import LitResidualDiffusion
from nowcasting.models.diffusion import (
    DiffusionUNet,
    cosine_alphas_cumprod,
    ddim_sample,
    q_sample,
    v_target,
    x0_from_v,
)


def test_schedule_is_decreasing_in_unit_interval():
    ab = cosine_alphas_cumprod(100)
    assert ab.shape == (100,)
    assert torch.all(ab[1:] < ab[:-1])
    assert ab[0] < 1.0 and ab[-1] > 0.0


def test_x0_is_recovered_from_v():
    ab = cosine_alphas_cumprod(50)[[3, 25, 49]].view(3, 1, 1, 1)
    x0, noise = torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8)
    z = q_sample(ab, x0, noise)
    torch.testing.assert_close(x0_from_v(ab, z, v_target(ab, x0, noise)), x0, atol=1e-4, rtol=1e-4)


def _oracle(x0, ab_all):
    def fn(z, t, cond):
        ab = ab_all[t].view(-1, 1, 1, 1)
        eps = (z - ab.sqrt() * x0) / (1 - ab).sqrt()
        return ab.sqrt() * eps - (1 - ab).sqrt() * x0

    return fn


def test_ddim_recovers_target_with_oracle_model():
    ab = cosine_alphas_cumprod(100)
    x0 = torch.randn(2, 3, 8, 8)
    out = ddim_sample(_oracle(x0, ab), ab, torch.zeros(2, 1, 8, 8), tuple(x0.shape), steps=10)
    torch.testing.assert_close(out, x0, atol=1e-3, rtol=1e-3)


def test_ddim_respects_bounds():
    ab = cosine_alphas_cumprod(100)
    x0 = torch.full((1, 2, 8, 8), 5.0)  # outside the feasible range
    lo, hi = torch.full_like(x0, -1.0), torch.full_like(x0, 1.0)
    out = ddim_sample(_oracle(x0, ab), ab, torch.zeros(1, 1, 8, 8), tuple(x0.shape), 10, lo, hi)
    assert out.max() <= 1.0 + 1e-6 and out.min() >= -1.0 - 1e-6


def test_diffusion_unet_shape():
    net = DiffusionUNet(in_ch=28, out_ch=12, base=16)
    out = net(torch.randn(2, 28, 32, 32), torch.tensor([3, 40]))
    assert out.shape == (2, 12, 32, 32)


def _tiny():
    return LitResidualDiffusion(backbone_base=8, base=16, timesteps=50, sample_steps=3, eval_batches=1)


def test_loss_trains_diffusion_net_only():
    lit = _tiny()
    x, y = torch.rand(2, 13, 1, 32, 32), torch.rand(2, 12, 1, 32, 32)
    loss = lit.compute_loss(x, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in lit.net.parameters())
    assert all(p.grad is None for p in lit.backbone.parameters())


def test_sample_shape_range_and_determinism():
    lit = _tiny().eval()
    x = torch.rand(2, 13, 1, 32, 32)
    gens = [torch.Generator().manual_seed(0) for _ in range(2)]
    a = lit.sample(x, n_members=2, steps=3, generator=gens[0])
    b = lit.sample(x, n_members=2, steps=3, generator=gens[1])
    assert a.shape == (2, 2, 12, 1, 32, 32)
    assert a.min() >= 0.0 and a.max() <= 1.0
    torch.testing.assert_close(a, b)
