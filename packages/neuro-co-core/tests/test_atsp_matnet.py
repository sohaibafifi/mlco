"""MatNet encoder and ATSP training tests."""

import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.models import MatNetEncoder, MatNetModel
from neuro_co.core.models.encoder import Encoder
from neuro_co.problems.atsp.env import ATSPEnv


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
