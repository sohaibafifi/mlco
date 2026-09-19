"""TSP env tests."""

import torch

from neuro_co.problems.tsp.env import TSPEnv


def test_reset_shapes() -> None:
    env = TSPEnv(size=10)
    s = env.reset(batch_size=4)
    assert s.coords.shape == (4, 10, 2)
    assert s.visited.shape == (4, 10)
    assert s.current.shape == (4,)
    assert s.first.shape == (4,)
    assert s.visited[:, 0].all()  # start city always visited
    assert not s.visited[:, 1:].any()


def test_reset_seeded() -> None:
    env = TSPEnv(size=8)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    s1 = env.reset(2, generator=g1)
    s2 = env.reset(2, generator=g2)
    assert torch.equal(s1.coords, s2.coords)


def test_action_mask_initial() -> None:
    env = TSPEnv(size=6)
    s = env.reset(3)
    m = env.action_mask(s)
    assert not m[:, 0].any()  # start city not allowed
    assert m[:, 1:].all()


def test_step_advances() -> None:
    env = TSPEnv(size=5)
    s = env.reset(2)
    a = torch.tensor([2, 3])
    s2, reward, done = env.step(s, a)
    assert torch.equal(s2.current, a)
    assert s2.visited[0, 2].item() is True
    assert s2.visited[1, 3].item() is True
    assert (s2.step_count == 1).all()
    assert not done.any()  # not done after 1 step on size-5
    assert (reward == 0).all()


def test_full_rollout_terminal() -> None:
    """Visit all cities; final step should produce non-zero reward and done=True."""
    env = TSPEnv(size=4)
    s = env.reset(1)
    actions_seq = [1, 2, 3]
    for i, a in enumerate(actions_seq):
        s, reward, done = env.step(s, torch.tensor([a]))
        if i < len(actions_seq) - 1:
            assert not done.any()
            assert (reward == 0).all()
        else:
            assert done.all()
            assert (reward < 0).all()  # tour_length > 0


def test_tour_length_correct() -> None:
    """Manually compute tour length for fixed coords and verify."""
    env = TSPEnv(size=3)
    s = env.reset(1)
    # Override coords to a fixed triangle.
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    s = s.replace(coords=coords)
    s, _, _ = env.step(s, torch.tensor([1]))
    s, _, done = env.step(s, torch.tensor([2]))
    assert done.all()
    expected = 1.0 + (2**0.5) + 1.0
    assert torch.allclose(s.tour_length, torch.tensor([expected]), atol=1e-5)
