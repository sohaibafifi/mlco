"""Cost and advantage shaping with constraint multipliers.

`shape_reward_global` adds `alpha * sum(mu * slack)` to trajectory cost.
`shape_advantage_per_step` subtracts a weighted violation penalty from the
advantage. Both obtain family weights from `neuro_co.cax.duals`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuro_co.cax import duals as _cax_duals
from neuro_co.cax.constraint_map import get_constraints
from neuro_co.cax.duals import state_to_instance


def shape_reward_global(
    state: Any,
    env: Any,
    trajectory_cost: Tensor,
    family_slack: dict[str, Tensor],
    *,
    problem: str,
    alpha: float = 0.1,
    multipliers: str = "lp",
    per_instance_mu: bool = True,
    duals_kwargs: dict[str, Any] | None = None,
) -> Tensor:
    """Trajectory-level reward shaping.

    Parameters
    ----------
    state, env
        Initial core state and its environment, used to extract the
        instance passed to the multiplier backend.
    trajectory_cost
        `FloatTensor[B]` cost (negative reward) of each rollout.
    family_slack
        Output of `neuro_co.dual.slack.family_slack`: dict
        `{family_name: FloatTensor[B]}`.
    problem
        Problem name (`'vrptw'`, `'op'`, `'fjsp'`).
    alpha
        Shaping coefficient. `0.0` leaves the trajectory cost unchanged.
    multipliers
        Backend for `get_multipliers`: `'lp'` (GLOP) or
        `'subgrad'` (Beasley).
    per_instance_mu
        If True (default), resolve `mu` once per instance in the
        batch. If False, resolve only for `batch_idx=0` and reuse
        across the batch. Shared multipliers are an approximation when
        the batch contains different instances.
    duals_kwargs
        Forwarded to the dual backend (`time_limit_s`,
        `max_iters`, ...).

    Returns
    -------
    Tensor
        `FloatTensor[B]` shaped cost `c + alpha * sum_k mu_k * slack_k`.
        Negating this result gives the shaped reward. Family aggregation
        and slack normalization make this a shaping heuristic, not an
        optimality-gap certificate.
    """
    duals_kwargs = duals_kwargs or {}
    families = [name for name, _ in get_constraints(problem)]
    B = trajectory_cost.shape[0]
    device = trajectory_cost.device

    mu_table = _resolve_mu(
        state=state,
        env=env,
        problem=problem,
        method=multipliers,
        batch_size=B,
        per_instance=per_instance_mu,
        duals_kwargs=duals_kwargs,
    )  # dict[family] -> Tensor[B]

    bonus = torch.zeros(B, device=device)
    for name in families:
        if name not in family_slack or name not in mu_table:
            continue
        slack = family_slack[name].to(device=device, dtype=torch.float32)
        mu = mu_table[name].to(device=device, dtype=torch.float32)
        bonus = bonus + mu * slack

    return trajectory_cost + alpha * bonus


def shape_advantage_per_step(
    state: Any,
    env: Any,
    advantage_per_step: Tensor,
    step_violation: dict[str, Tensor],
    *,
    problem: str,
    alpha: float = 0.1,
    multipliers: str = "lp",
    per_instance_mu: bool = True,
    duals_kwargs: dict[str, Any] | None = None,
) -> Tensor:
    """Per-step advantage shaping.

    Parameters
    ----------
    advantage_per_step
        `FloatTensor[B, T]` advantage estimates per (instance,
        step).
    step_violation
        `{family_name: FloatTensor[B, T]}` per-step violation of
        each constraint family.

    Returns
    -------
    Tensor
        `FloatTensor[B, T]` shaped advantage
        `A - alpha * sum_k mu_k(x) * violation_k(t)`.
    """
    duals_kwargs = duals_kwargs or {}
    families = [name for name, _ in get_constraints(problem)]
    B, T = advantage_per_step.shape
    device = advantage_per_step.device

    mu_table = _resolve_mu(
        state=state,
        env=env,
        problem=problem,
        method=multipliers,
        batch_size=B,
        per_instance=per_instance_mu,
        duals_kwargs=duals_kwargs,
    )

    penalty = torch.zeros(B, T, device=device)
    for name in families:
        if name not in step_violation or name not in mu_table:
            continue
        v = step_violation[name].to(device=device, dtype=torch.float32)
        mu = mu_table[name].to(device=device, dtype=torch.float32).unsqueeze(-1)
        penalty = penalty + mu * v

    return advantage_per_step - alpha * penalty


def _resolve_mu(
    *,
    state: Any,
    env: Any,
    problem: str,
    method: str,
    batch_size: int,
    per_instance: bool,
    duals_kwargs: dict[str, Any],
) -> dict[str, Tensor]:
    """Build a `{family: Tensor[B]}` table of dual multipliers."""
    families = [name for name, _ in get_constraints(problem)]
    table: dict[str, list[float]] = {name: [] for name in families}

    n_solve = batch_size if per_instance else 1
    for b in range(n_solve):
        try:
            instance = state_to_instance(state, env, problem, batch_idx=b)
            mu = _cax_duals.get_multipliers(problem, instance, method=method, **duals_kwargs)
        except (NotImplementedError, ValueError, RuntimeError):
            mu = {name: 0.0 for name in families}
        for name in families:
            table[name].append(float(mu.get(name, 0.0)))

    out: dict[str, Tensor] = {}
    for name, vals in table.items():
        t = torch.tensor(vals, dtype=torch.float32)
        if not per_instance:
            t = t.repeat(batch_size)
        out[name] = t
    return out
