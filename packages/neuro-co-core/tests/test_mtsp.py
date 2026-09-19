"""mTSP env tests."""

import torch

from neuro_co.core.envs.mtsp import MTSPEnv


def test_reset_shapes() -> None:
    env = MTSPEnv(size=8, num_agents=2)
    s = env.reset(4)
    assert s.coords.shape == (4, 9, 2)
    assert (s.routes_done == 0).all()


def test_route_closes_on_depot_return() -> None:
    env = MTSPEnv(size=5, num_agents=2)
    s = env.reset(1)
    s, _, _ = env.step(s, torch.tensor([1]))  # leave depot
    assert s.routes_done.item() == 0
    s, _, _ = env.step(s, torch.tensor([0]))  # back to depot -> close route 1
    assert s.routes_done.item() == 1


def test_full_rollout_two_routes() -> None:
    env = MTSPEnv(size=4, num_agents=2)
    s = env.reset(1)
    # route1: depot->1->2->depot ; route2: ->3->4->depot
    for a in [1, 2, 0, 3, 4, 0]:
        s, reward, done = env.step(s, torch.tensor([a]))
    assert done.all()
    assert reward.item() < 0
    assert s.routes_done.item() == 2


def test_features_dim() -> None:
    env = MTSPEnv(size=6, num_agents=2)
    s = env.reset(2)
    assert env.build_features(s).shape == (2, 7, 2)
    assert env.encoder_in_dim == 2
