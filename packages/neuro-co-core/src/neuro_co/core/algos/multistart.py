"""POMO-style multi-start rollout helper.

For each problem in a batch, roll out from `n_starts` distinct first
actions in parallel. Use group-mean reward as baseline: no separate
baseline network needed.

Algo-agnostic: takes a `rollout_fn(state, first_actions) -> (reward, sum_logp)`.
"""

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from ..state import State


def replicate(state: State, n_starts: int) -> State:
    """Tile each item in the batch `n_starts` times along the batch dim.

    Output ordering: [s0_start0, s0_start1, ..., s1_start0, s1_start1, ...].
    """
    from torch.utils import _pytree as pytree

    def _tile(x):
        if not isinstance(x, torch.Tensor):
            return x
        return x.repeat_interleave(n_starts, dim=0)

    return pytree.tree_map(_tile, state)


def pomo_advantage(reward: Float[Tensor, "bn"], n_starts: int) -> Float[Tensor, "bn"]:
    """reward minus group-mean baseline. Output shape matches input."""
    bn = reward.shape[0]
    if bn % n_starts != 0:
        raise ValueError(f"reward shape {bn} not divisible by n_starts={n_starts}")
    b = bn // n_starts
    grouped = reward.view(b, n_starts)
    mean = grouped.mean(dim=1, keepdim=True)
    return (grouped - mean).view(bn)


def distinct_first_actions(
    mask: torch.Tensor,
    n_starts: int,
    generator: torch.Generator | None = None,
) -> Int[Tensor, "bn"]:
    """Pick `n_starts` permitted first actions per problem.

    Picks are distinct when a problem has >= `n_starts` permitted actions.
    When fewer are permitted (e.g. CVRPTW time windows close some customers
    even at t=0), cycle through the valid set so every pick stays valid and
    the group size stays exactly `n_starts` (some repeats). Group-mean POMO
    baseline still works with repeats: just less start diversity.

    Args:
        mask: (b, a) boolean. True = permitted.
        n_starts: number of starts per problem.

    Returns:
        (b * n_starts,) flat tensor of action indices, grouped per problem.
        Every entry indexes a permitted action (unless a row has zero
        permitted actions, which the caller must prevent).
    """
    b, a = mask.shape
    scores = torch.rand(b, a, generator=generator, device=mask.device).masked_fill(~mask, -1.0)
    perm = torch.argsort(scores, dim=1, descending=True)  # permitted actions sort first
    valid_count = mask.sum(dim=1).clamp_min(1)  # (b,) avoid div-by-zero
    pos = torch.arange(n_starts, device=mask.device).unsqueeze(0).expand(b, n_starts)
    wrapped = pos % valid_count.unsqueeze(1)  # cycle within the permitted prefix
    picks = perm.gather(1, wrapped)  # (b, n_starts), all valid
    return picks.reshape(b * n_starts)


def multistart_rollout(
    state: State,
    n_starts: int,
    *,
    rollout_fn: Callable[
        [State, Int[Tensor, "bn"] | None], tuple[Float[Tensor, "bn"], Float[Tensor, "bn"]]
    ],
    first_action_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "bn"], Float[Tensor, "bn"], Float[Tensor, "bn"]]:
    """Run `n_starts` rollouts per problem with distinct first actions.

    Returns:
        reward: (b*n_starts,)
        sum_logp: (b*n_starts,)
        advantage: (b*n_starts,): reward minus group mean.
    """
    expanded = replicate(state, n_starts)
    first_actions = distinct_first_actions(first_action_mask, n_starts, generator)
    reward, sum_logp = rollout_fn(expanded, first_actions)
    adv = pomo_advantage(reward, n_starts)
    return reward, sum_logp, adv
