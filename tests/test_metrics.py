import math

import torch

from nowcasting.metrics import NowcastingMetrics
from nowcasting.models.baselines import Persistence


def test_hand_computed_counts():
    # 1 sample, 1 lead time, 2x4 image. Values 20/255 are above threshold 16, below 74.
    v = 20 / 255
    target = torch.zeros(1, 1, 1, 2, 4)
    target[0, 0, 0, 0, :] = v  # 4 true events
    pred = torch.zeros(1, 1, 1, 2, 4)
    pred[0, 0, 0, 0, :2] = v  # 2 hits
    pred[0, 0, 0, 1, 0] = v  # 1 false alarm

    m = NowcastingMetrics(out_len=1)
    m.update(pred, target)
    res = m.compute()

    assert m.counts[0, 0].tolist() == [2, 2, 1]  # hits, misses, false alarms at threshold 16
    assert math.isclose(res["csi"][0, 0].item(), 2 / 5)
    assert math.isclose(res["pod"][0, 0].item(), 2 / 4)
    assert math.isclose(res["far"][0, 0].item(), 1 / 3)
    assert torch.isnan(res["csi"][0, 1])  # no event at threshold 74 -> undefined


def test_perfect_prediction():
    target = torch.rand(2, 3, 1, 8, 8)
    m = NowcastingMetrics(out_len=3)
    m.update(target.clone(), target)
    res = m.compute()
    assert res["mse"].sum().item() == 0.0
    valid = ~torch.isnan(res["csi"])
    assert torch.all(res["csi"][valid] == 1.0)
    assert torch.all(res["far"][valid] == 0.0)


def test_counts_accumulate_across_batches():
    target = torch.rand(4, 2, 1, 8, 8)
    pred = torch.rand(4, 2, 1, 8, 8)
    a, b = NowcastingMetrics(2), NowcastingMetrics(2)
    a.update(pred, target)
    b.update(pred[:2], target[:2])
    b.update(pred[2:], target[2:])
    assert torch.equal(a.counts, b.counts)


def test_persistence_repeats_last_frame():
    x = torch.rand(2, 13, 1, 16, 16)
    out = Persistence(out_len=12)(x)
    assert out.shape == (2, 12, 1, 16, 16)
    assert all(torch.equal(out[:, t], x[:, -1]) for t in range(12))
