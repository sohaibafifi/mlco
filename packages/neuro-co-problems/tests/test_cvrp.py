"""CVRP env tests."""

import torch

from neuro_co.problems.cvrp.env import CVRPEnv


def test_reset_shapes() -> None:
    env = CVRPEnv(size=10, capacity=30.0, max_demand=9)
    s = env.reset(batch_size=4)
    assert s.coords.shape == (4, 11, 2)
    assert s.demand.shape == (4, 11)
    assert torch.equal(s.demand[:, 0], torch.zeros(4))  # depot demand 0
    assert (s.demand[:, 1:] >= 1).all()
    assert (s.demand[:, 1:] <= 9).all()
    assert torch.equal(s.remaining_capacity, torch.full((4,), 30.0))
    assert (s.current == 0).all()


def test_initial_mask_excludes_depot() -> None:
    env = CVRPEnv(size=5)
    s = env.reset(3)
    m = env.action_mask(s)
    # At depot, cannot return to depot immediately unless no customer fits.
    assert not m[:, 0].any() or (s.demand[:, 1:].min(dim=1).values > s.remaining_capacity).any()
    # All customers permitted at start if demand fits capacity.
    fits = s.demand[:, 1:] <= s.remaining_capacity.unsqueeze(1)
    assert torch.equal(m[:, 1:], fits)


def test_step_capacity_decreases() -> None:
    env = CVRPEnv(size=4, capacity=10.0)
    s = env.reset(2)
    a = torch.tensor([1, 2])
    pre_cap = s.remaining_capacity.clone()
    delivered = s.demand.gather(1, a.unsqueeze(1)).squeeze(1)
    s2, reward, done = env.step(s, a)
    assert torch.allclose(s2.remaining_capacity, pre_cap - delivered)
    assert not done.any()
    assert (reward == 0).all()


def test_dynamic_decoder_context_tracks_normalized_remaining_capacity() -> None:
    env = CVRPEnv(size=4, capacity=10.0)
    state = env.reset(2, generator=torch.Generator().manual_seed(0))
    assert torch.equal(env.dynamic_decoder_context(state), torch.ones(2, 1))

    actions = torch.tensor([1, 2])
    delivered = state.demand.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_state, _, _ = env.step(state, actions)
    expected = ((10.0 - delivered) / 10.0).unsqueeze(-1)
    assert torch.allclose(env.dynamic_decoder_context(next_state), expected)


def test_step_depot_refills() -> None:
    env = CVRPEnv(size=4, capacity=20.0)
    s = env.reset(1)
    # Visit a customer first to drop capacity.
    s, _, _ = env.step(s, torch.tensor([1]))
    cap_before = s.remaining_capacity.clone()
    s2, _, _ = env.step(s, torch.tensor([0]))
    assert s2.remaining_capacity.item() == 20.0
    assert cap_before.item() < 20.0


def test_capacity_constraint_in_mask() -> None:
    env = CVRPEnv(size=3, capacity=5.0, max_demand=9)
    s = env.reset(1)
    # Force a known demand vector: depot=0, others=[2, 9, 4].
    demand = torch.tensor([[0.0, 2.0, 9.0, 4.0]])
    s = s.replace(demand=demand, remaining_capacity=torch.tensor([3.0]))
    mask = env.action_mask(s)
    # Capacity=3: only customer with demand<=3 i.e. index 1 (demand=2) permitted.
    assert mask[0, 1].item() is True
    assert mask[0, 2].item() is False  # demand=9 > 3
    assert mask[0, 3].item() is False  # demand=4 > 3
