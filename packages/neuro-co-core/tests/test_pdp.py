"""PDP (Pickup-Delivery) env tests."""

import torch

from neuro_co.core.envs.pdp import PDPEnv


def test_reset_shapes_and_types() -> None:
    env = PDPEnv(size=3)  # 3 pairs -> 7 nodes
    s = env.reset(4)
    assert s.coords.shape == (4, 7, 2)
    assert (s.node_type[:, 0] == 0).all()  # depot
    assert (s.node_type[:, 1:4] == 1).all()  # pickups
    assert (s.node_type[:, 4:] == 2).all()  # deliveries


def test_delivery_masked_until_pickup() -> None:
    env = PDPEnv(size=2)  # nodes: 0 depot, 1,2 pickups, 3,4 deliveries
    s = env.reset(1)
    mask = env.action_mask(s)
    # Initially: pickups allowed, deliveries forbidden.
    assert mask[0, 1].item() and mask[0, 2].item()
    assert not mask[0, 3].item() and not mask[0, 4].item()
    # Visit pickup 1 -> delivery 3 (paired) becomes allowed.
    s, _, _ = env.step(s, torch.tensor([1]))
    mask = env.action_mask(s)
    assert mask[0, 3].item()  # delivery of pair 1 now ok
    assert not mask[0, 4].item()  # pair 2 pickup not visited yet


def test_first_mask_only_pickups() -> None:
    env = PDPEnv(size=3)
    s = env.reset(2)
    fm = env.pomo_first_mask(s)
    assert fm[:, 1:4].all()  # pickups
    assert not fm[:, 0].any()
    assert not fm[:, 4:].any()  # deliveries excluded


def test_full_rollout_completes() -> None:
    env = PDPEnv(size=2)
    s = env.reset(1)
    # pickup1, delivery1, pickup2, delivery2, depot
    for a in [1, 3, 2, 4, 0]:
        s, reward, done = env.step(s, torch.tensor([a]))
    assert done.all()
    assert reward.item() < 0  # -tour_length


def test_features_dim() -> None:
    env = PDPEnv(size=3)
    s = env.reset(2)
    assert env.build_features(s).shape == (2, 7, 4)
    assert env.encoder_in_dim == 4
