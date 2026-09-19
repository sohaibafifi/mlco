"""Comparison strategies for multiplier-based cost shaping.

`shape_reward_scalar` applies uniform family weights. `AdaptiveLagrangian`
updates family weights from observed violations without solving an LP.
`shape_reward_oracle_optimum` currently delegates to LP-based shaping;
despite its name, it does not compute optimal-solution multipliers.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuro_co.cax.constraint_map import get_constraints


def shape_reward_scalar(
    trajectory_cost: Tensor,
    family_slack: dict[str, Tensor],
    *,
    alpha: float = 0.1,
) -> Tensor:
    """Constant-penalty baseline.

    Equivalent to `shape_reward_global` with `mu_k(x) = 1` for all
    families. Single scalar hyperparameter `alpha`; can be tuned
    by sweep but cannot adapt per-instance.
    """
    if not family_slack:
        return trajectory_cost
    total_slack = sum(s.to(trajectory_cost) for s in family_slack.values())
    # Sign convention matches `shape_reward_global`: slack on a
    # binding family is *waste*, so it is added to the cost (= a
    # penalty), not subtracted. With `mu_k = 1` this reduces to
    # `c + alpha * sum_k slack_k`.
    return trajectory_cost + alpha * total_slack


class AdaptiveLagrangian:
    """Online dual ascent on per-family violations.

    Keeps a learnable `mu_k >= 0` per constraint family, updated
    each training step::

        mu_k <- max(0, mu_k + lr * mean_batch[violation_k])

    The shaped reward is `R - sum_k mu_k * violation_k`. Equivalent
    to a Lagrangian-PPO baseline where `mu` is appended to the
    optimisation variables.

    Parameters
    ----------
    problem
        Used to enumerate constraint families from
        `neuro_co.cax.constraint_map`.
    lr
        Dual ascent step size.
    mu_init
        Initial multiplier value (clamped to >= 0).
    mu_max
        Optional cap on `mu` to keep gradients bounded. `None`
        for unconstrained ascent.

    State
    -----
    `self.mu : dict[str, Tensor]`  the running multipliers.
    """

    def __init__(
        self,
        problem: str,
        *,
        lr: float = 0.01,
        mu_init: float = 0.0,
        mu_max: float | None = 100.0,
    ):
        self.problem = problem
        self.lr = lr
        self.mu_max = mu_max
        families = [name for name, _ in get_constraints(problem)]
        self.mu: dict[str, Tensor] = {
            name: torch.tensor(float(max(mu_init, 0.0))) for name in families
        }

    def update(self, family_violation: dict[str, Tensor]) -> None:
        """One dual-ascent step on observed batch-mean violations."""
        for name in self.mu:
            if name not in family_violation:
                continue
            v = family_violation[name]
            step = self.lr * float(v.float().mean())
            new = self.mu[name].item() + step
            new = max(0.0, new)
            if self.mu_max is not None:
                new = min(new, self.mu_max)
            self.mu[name] = torch.tensor(new)

    def shape(
        self,
        trajectory_cost: Tensor,
        family_violation: dict[str, Tensor],
        *,
        alpha: float = 1.0,
    ) -> Tensor:
        """Shape the trajectory cost with current `mu`."""
        if not family_violation:
            return trajectory_cost
        device = trajectory_cost.device
        penalty = torch.zeros_like(trajectory_cost)
        for name, v in family_violation.items():
            mu = self.mu.get(name)
            if mu is None:
                continue
            penalty = penalty + mu.to(device) * v.to(device).float()
        return trajectory_cost + alpha * penalty

    def state(self) -> dict[str, float]:
        """Snapshot of current multipliers (for logging)."""
        return {name: float(t.item()) for name, t in self.mu.items()}


def shape_reward_oracle_optimum(
    state: Any,
    env: Any,
    trajectory_cost: Tensor,
    family_slack: dict[str, Tensor],
    *,
    problem: str,
    alpha: float = 0.1,
    cp_kwargs: dict[str, Any] | None = None,
) -> Tensor:
    """Return LP-based shaping through the compatibility oracle entry point.

    This function delegates to `shape_reward_global` with `multipliers="lp"`.
    It does not solve the integer problem or extract optimal-solution multipliers.
    """
    from neuro_co.dual.shaping import shape_reward_global

    cp_kwargs = cp_kwargs or {}
    return shape_reward_global(
        state,
        env,
        trajectory_cost,
        family_slack,
        problem=problem,
        alpha=alpha,
        multipliers="lp",
        duals_kwargs=cp_kwargs,
    )
