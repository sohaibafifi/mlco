"""SSMEncoder satisfies the neuro-co-core Encoder Protocol + trains on core envs."""

import pytest
import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import ConstructivePolicy, MambaModel, PointerDecoder, SSMEncoder
from neuro_co.core.models.encoder import Encoder

pytest.importorskip("mambapy")


def test_ssm_encoder_satisfies_protocol() -> None:
    enc = SSMEncoder(backend="mambapy", in_dim=2, hidden_dim=16, num_layers=1, bidirectional=False)
    assert isinstance(enc, Encoder)


def test_ssm_encode_shapes() -> None:
    enc = SSMEncoder(backend="mambapy", in_dim=2, hidden_dim=16, num_layers=2, bidirectional=True)
    x = torch.rand(3, 7, 2)
    node_embs, graph_emb = enc(x)
    assert node_embs.shape == (3, 7, 16)
    assert graph_emb.shape == (3, 16)


def test_mamba_model_is_constructive_policy() -> None:
    m = MambaModel(backend="mambapy", in_dim=2, hidden_dim=16, num_layers=1, num_heads=2)
    assert isinstance(m, ConstructivePolicy)
    assert isinstance(m.encoder, SSMEncoder)
    assert isinstance(m.decoder, PointerDecoder)


def test_reinforce_tsp_with_mamba() -> None:
    env = TSPEnv(size=6)
    model = MambaModel(
        backend="mambapy", in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2
    )
    algo = REINFORCE(model, env, REINFORCEConfig(batch_size=8, eval_batch_size=8))
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert m["loss"] == m["loss"]
    ev = algo.eval_step(rng)
    assert ev["eval_tour_length"] > 0


def test_pomo_tsp_with_mamba() -> None:
    env = TSPEnv(size=6)
    model = MambaModel(
        backend="mambapy", in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2
    )
    algo = POMO(model, env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=8))
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert m["loss"] == m["loss"]
