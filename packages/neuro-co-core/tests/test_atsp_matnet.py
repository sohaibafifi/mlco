"""ATSP env + MatNet encoder tests."""

import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.envs.atsp import ATSPEnv
from neuro_co.core.models import MatNetEncoder, MatNetModel
from neuro_co.core.models.encoder import Encoder


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


def test_matnet_encoder_satisfies_protocol() -> None:
    enc = MatNetEncoder(problem_size=8, hidden_dim=16, num_layers=1, num_heads=2)
    assert isinstance(enc, Encoder)


def test_matnet_encode_shapes() -> None:
    enc = MatNetEncoder(problem_size=8, hidden_dim=16, num_layers=2, num_heads=4)
    d = torch.rand(3, 8, 8)
    node_embs, graph_emb = enc(d)
    assert node_embs.shape == (3, 8, 16)
    assert graph_emb.shape == (3, 16)


def test_reinforce_on_atsp_with_matnet() -> None:
    env = ATSPEnv(size=6)
    model = MatNetModel(problem_size=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    algo = REINFORCE(model, env, REINFORCEConfig(batch_size=8, eval_batch_size=8))
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert m["loss"] == m["loss"]
    ev = algo.eval_step(rng)
    assert ev["eval_tour_length"] > 0


def test_pomo_on_atsp_with_matnet() -> None:
    env = ATSPEnv(size=6)
    model = MatNetModel(problem_size=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    algo = POMO(model, env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=8))
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert m["loss"] == m["loss"]
