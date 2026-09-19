"""OP (Orienteering) env tests."""

import torch

from neuro_co.problems.op.env import OPEnv


def test_reset_shapes() -> None:
    env = OPEnv(size=8, budget=3.0)
    s = env.reset(4)
    assert s.coords.shape == (4, 9, 2)
    assert s.prize.shape == (4, 9)
    assert (s.prize[:, 0] == 0).all()  # depot prize 0
    assert (s.prize[:, 1:] > 0).all()
    assert (s.collected == 0).all()


def test_budget_mask_excludes_unreachable() -> None:
    env = OPEnv(size=5, budget=0.01)  # tiny budget
    s = env.reset(2)
    mask = env.action_mask(s)
    # With ~zero budget, no customer reachable (can't go + return). Depot ok.
    assert not mask[:, 1:].any()
    assert mask[:, 0].all()


def test_collect_prize_on_visit() -> None:
    env = OPEnv(size=4, budget=10.0)
    s = env.reset(1)
    prize1 = s.prize[0, 1].item()
    s2, _, done = env.step(s, torch.tensor([1]))
    assert abs(s2.collected.item() - prize1) < 1e-6
    assert not done.any()  # not back at depot yet


def test_return_depot_terminates_with_prize_reward() -> None:
    env = OPEnv(size=4, budget=10.0)
    s = env.reset(1)
    s, _, _ = env.step(s, torch.tensor([1]))
    collected = s.collected.item()
    s, reward, done = env.step(s, torch.tensor([0]))  # back to depot
    assert done.all()
    assert abs(reward.item() - collected) < 1e-6  # reward = collected prize


def test_features_dim() -> None:
    env = OPEnv(size=5)
    s = env.reset(2)
    assert env.build_features(s).shape == (2, 6, 3)
    assert env.encoder_in_dim == 3
