"""ATSP environment tests."""

import torch

from neuro_co.problems.atsp.env import ATSPEnv


def test_atsp_reset_shapes() -> None:
    env = ATSPEnv(size=8)
    s = env.reset(4)
    assert s.dist.shape == (4, 8, 8)
    # Zero diagonal.
    assert torch.allclose(torch.diagonal(s.dist, dim1=1, dim2=2), torch.zeros(4, 8))
    assert s.visited[:, 0].all()
    assert env.encoder_in_dim == 8


def test_atsp_asymmetric() -> None:
    env = ATSPEnv(size=6)
    s = env.reset(2, generator=torch.Generator().manual_seed(0))
    d = s.dist
    # Almost surely asymmetric.
    assert not torch.allclose(d, d.transpose(1, 2))


def test_atsp_tour_length_directed() -> None:
    env = ATSPEnv(size=4)
    s = env.reset(1)
    # Fixed dist matrix.
    dist = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [9.0, 0.0, 1.0, 1.0], [9.0, 9.0, 0.0, 1.0], [1.0, 9.0, 9.0, 0.0]]]
    )
    s = s.replace(dist=dist)
    # tour 0->1->2->3->0 : 1 + 1 + 1 + 1 = 4
    for a in [1, 2, 3]:
        s, _, done = env.step(s, torch.tensor([a]))
    assert done.all()
    assert torch.allclose(s.tour_length, torch.tensor([4.0]), atol=1e-5)
