"""CVRPTW env tests."""

import torch

from neuro_co.problems.vrptw.env import CVRPTWEnv


def test_reset_shapes() -> None:
    env = CVRPTWEnv(size=8, capacity=30.0, horizon=4.0, window_width=0.5)
    s = env.reset(batch_size=4)
    n_plus_1 = 9
    assert s.coords.shape == (4, n_plus_1, 2)
    assert s.demand.shape == (4, n_plus_1)
    assert s.tw_early.shape == (4, n_plus_1)
    assert s.tw_late.shape == (4, n_plus_1)
    assert s.current_time.shape == (4,)
    assert (s.demand[:, 0] == 0).all()
    assert (s.tw_early[:, 0] == 0).all()
    assert (s.tw_late[:, 0] == 4.0).all()
    # Customer windows: late = early + width.
    assert torch.allclose(s.tw_late[:, 1:] - s.tw_early[:, 1:], torch.full((4, 8), 0.5))


def test_action_mask_excludes_late_customers() -> None:
    env = CVRPTWEnv(size=4, capacity=30.0, horizon=4.0, window_width=0.5)
    s = env.reset(2)
    # Force a state where current_time is very late so most customers infeasible.
    s = s.replace(current_time=torch.full((2,), 10.0))  # past horizon
    mask = env.action_mask(s)
    # All customers should be infeasible (out of window). Depot must be permitted.
    assert not mask[:, 1:].any()
    assert mask[:, 0].all()


def test_step_advances_time_and_capacity() -> None:
    env = CVRPTWEnv(size=4, capacity=30.0, horizon=10.0, window_width=0.5)
    s = env.reset(1)
    # Set deterministic geometry: depot at (0,0), customer 1 at (3,4).
    coords = s.coords.clone()
    coords[0, 0] = torch.tensor([0.0, 0.0])
    coords[0, 1] = torch.tensor([3.0, 4.0])
    # Make customer 1's window open early.
    tw_early = s.tw_early.clone()
    tw_late = s.tw_late.clone()
    tw_early[0, 1] = 0.0
    tw_late[0, 1] = 10.0
    s = s.replace(coords=coords, tw_early=tw_early, tw_late=tw_late)
    s2, _, done = env.step(s, torch.tensor([1]))
    # Distance = 5, arrival = 5, > tw_early=0 so no wait => current_time = 5.
    assert torch.allclose(s2.current_time, torch.tensor([5.0]), atol=1e-5)
    assert torch.allclose(s2.tour_length, torch.tensor([5.0]), atol=1e-5)
    assert not done.any()


def test_pomo_first_mask_excludes_depot() -> None:
    env = CVRPTWEnv(size=5, capacity=30.0)
    s = env.reset(2)
    m = env.pomo_first_mask(s)
    assert not m[:, 0].any()
