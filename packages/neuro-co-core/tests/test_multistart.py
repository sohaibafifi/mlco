"""Multi-start (POMO) helper tests."""

import torch

from neuro_co.core.algos.multistart import distinct_first_actions, pomo_advantage, replicate
from neuro_co.core.envs.tsp import TSPEnv


def test_replicate_shapes() -> None:
    env = TSPEnv(size=5)
    s = env.reset(3)
    s2 = replicate(s, n_starts=4)
    assert s2.coords.shape == (12, 5, 2)
    # Each consecutive group of 4 shares the same problem.
    for i in range(3):
        for j in range(1, 4):
            assert torch.equal(s2.coords[i * 4], s2.coords[i * 4 + j])


def test_pomo_advantage_zero_mean_per_group() -> None:
    reward = torch.tensor([1.0, 3.0, 5.0, 7.0, 0.0, 2.0])  # b=2, n_starts=3
    adv = pomo_advantage(reward, n_starts=3)
    assert adv.shape == (6,)
    g0 = adv[:3]
    g1 = adv[3:]
    assert torch.allclose(g0.mean(), torch.tensor(0.0))
    assert torch.allclose(g1.mean(), torch.tensor(0.0))


def test_distinct_first_actions() -> None:
    # b=2, a=5, all permitted -> picks 3 distinct per problem.
    mask = torch.ones(2, 5, dtype=torch.bool)
    rng = torch.Generator().manual_seed(0)
    picks = distinct_first_actions(mask, n_starts=3, generator=rng)
    assert picks.shape == (6,)
    # Distinct within each problem.
    g0 = picks[:3]
    g1 = picks[3:]
    assert len(set(g0.tolist())) == 3
    assert len(set(g1.tolist())) == 3


def test_distinct_first_actions_respects_mask() -> None:
    mask = torch.tensor([[False, True, True, False, True]])  # only 3 permitted
    rng = torch.Generator().manual_seed(0)
    picks = distinct_first_actions(mask, n_starts=3, generator=rng)
    assert picks.shape == (3,)
    permitted = {1, 2, 4}
    assert set(picks.tolist()).issubset(permitted)


def test_distinct_first_actions_wraps_when_few_valid() -> None:
    """n_starts > valid: cycle through valid set, all picks valid, size fixed."""
    mask = torch.tensor([[False, True, True, False, True]])  # 3 permitted
    rng = torch.Generator().manual_seed(0)
    picks = distinct_first_actions(mask, n_starts=5, generator=rng)  # want 5 from 3
    assert picks.shape == (5,)
    permitted = {1, 2, 4}
    assert set(picks.tolist()).issubset(permitted)
    # All 3 distinct valid actions should appear (cycle covers them).
    assert set(picks.tolist()) == permitted


def test_distinct_first_actions_wrap_per_problem() -> None:
    """Different valid counts per problem in same batch."""
    mask = torch.tensor(
        [
            [False, True, True, True, True],  # 4 valid
            [False, True, True, False, False],  # 2 valid
        ]
    )
    rng = torch.Generator().manual_seed(0)
    picks = distinct_first_actions(mask, n_starts=4, generator=rng).view(2, 4)
    assert set(picks[0].tolist()).issubset({1, 2, 3, 4})
    assert set(picks[1].tolist()).issubset({1, 2})  # only 2 valid, wrapped
