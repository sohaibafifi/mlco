"""Constraint slack and violation magnitudes for completed trajectories.

CVRPTW uses the peak route load and the smallest customer time-window margin.
OP uses route length including the return to the depot. Capacity, time, and
budget margins are normalized by the corresponding instance scale. Public
slacks clamp these margins at zero; violations retain their negative part.

Spatial entries are placeholders. OP prize is an objective term, and FJSP
eligibility is a coarse instance statistic. Neither measures violations.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuro_co.cax._instance import to_batch_instance


def family_slack(
    state: Any,
    env: Any,
    actions: Tensor,
    problem: str,
) -> dict[str, Tensor]:
    """Per-family slack of a finished rollout.

    Parameters
    ----------
    state, env
        Core `State` (pre-rollout) + its `Env`. Converted to a canonical
        instance dict internally.
    actions
        `LongTensor[B, T]` action sequence produced by the policy.
    problem
        Problem name; dispatches to a per-problem vectorised slack.

    Returns
    -------
    dict[str, Tensor]
        `{family_name: FloatTensor[B]}` non-negative slack values.
    """
    prob = problem.lower()
    inst = to_batch_instance(state, env, problem)
    if prob in ("vrptw", "cvrptw"):
        return _vrptw_slack(inst, actions)
    if prob == "op":
        return _op_slack(inst, actions)
    if prob == "fjsp":
        return _fjsp_slack(inst, actions)
    raise NotImplementedError(
        f"family_slack: problem={problem!r} not wired. Supports vrptw, op, fjsp."
    )


def family_violation(
    state: Any,
    env: Any,
    actions: Tensor,
    problem: str,
) -> dict[str, Tensor]:
    """Return normalized peak capacity excess, customer lateness, or budget excess.

    Each supported constraint returns zero on a feasible trajectory. CVRPTW
    resets load and time at depot visits, matching the core environment.
    OP prize, spatial entries, and the coarse FJSP statistics return zero;
    this function does not check FJSP trajectory feasibility.
    """
    prob = problem.lower()
    inst = to_batch_instance(state, env, problem)
    if prob in ("vrptw", "cvrptw"):
        margins = _vrptw_margins(inst, actions)
    elif prob == "op":
        margins = _op_margins(inst, actions)
        margins["prize"] = torch.zeros_like(margins["prize"])
    elif prob == "fjsp":
        return {name: torch.zeros_like(value) for name, value in _fjsp_slack(inst, actions).items()}
    else:
        raise NotImplementedError(
            f"family_violation: problem={problem!r} not wired. Supports vrptw, op, fjsp."
        )
    return {name: (-value).clamp(min=0.0) for name, value in margins.items()}


# --------------------------------------------------------------------- VRPTW


def _vrptw_slack(td: Any, actions: Tensor) -> dict[str, Tensor]:
    return {name: value.clamp(min=0.0) for name, value in _vrptw_margins(td, actions).items()}


def _vrptw_margins(td: Any, actions: Tensor) -> dict[str, Tensor]:
    """Signed capacity and customer time-window margins.

    Convention (the `CVRPTWEnv` with `size=N`):

        locs:           [B, N+1, 2]   depot at index 0
        demand:         [B, N+1]      depot included, or [B, N] customers only
        time_windows:   [B, N+1, 2]
        durations:      [B, N+1]
        vehicle_cap:    [B, 1] | [B] | scalar
        actions:        [B, T]        0 = depot, 1..N = customers
    """
    demand = _get(td, "demand")
    locs = _get(td, "locs")  # [B, N+1, 2]
    tw = _get(td, "time_windows")  # [B, N+1, 2]
    dur = _get(td, "durations")  # [B, N+1]
    cap = _get(td, "vehicle_capacity")  # [B, 1] or [B]

    actions = actions.long()
    B, T = actions.shape
    device = actions.device
    N_dem = demand.shape[1]
    N_loc = locs.shape[1]
    N_tw = tw.shape[1]
    N_dur = dur.shape[1]

    # ---- capacity slack ---------------------------------------------------
    # Core states include the depot demand; customer-only inputs omit it.
    depot = actions == 0
    demand_offset = 0 if N_dem == N_loc else 1
    cust_idx = (actions - demand_offset).clamp_min(0).clamp_max(N_dem - 1)
    demand_step = torch.gather(demand, 1, cust_idx)  # [B, T]
    demand_step = demand_step.masked_fill(depot, 0.0)

    # Cumulative load with reset at depot: cum - last_cum_at_depot.
    # The running maximum records the cumulative demand at the last depot.
    cum = demand_step.cumsum(dim=1)
    cum_at_depot = torch.where(depot, cum, torch.zeros_like(cum))
    running_dep = cum_at_depot.cummax(dim=1).values
    load = cum - running_dep  # [B, T]
    peak_load = load.max(dim=1).values  # [B]

    if cap.dim() == 0:
        cap_b = cap.expand(B).to(device=device, dtype=load.dtype)
    elif cap.dim() == 1:
        cap_b = cap.to(device=device, dtype=load.dtype)
    else:
        cap_b = cap[..., 0].to(device=device, dtype=load.dtype)
    # Express capacity headroom or excess as a fraction of vehicle capacity.
    cap_slack = (cap_b - peak_load) / cap_b.clamp(min=1e-9)

    # ---- time-window slack ------------------------------------------------
    # Per-step gather of node features (depot = index 0).
    loc_idx = actions.clamp_min(0).clamp_max(N_loc - 1)
    tw_idx = actions.clamp_min(0).clamp_max(N_tw - 1)
    dur_idx = actions.clamp_min(0).clamp_max(N_dur - 1)

    pos = torch.gather(locs, 1, loc_idx.unsqueeze(-1).expand(-1, -1, 2))  # [B, T, 2]
    prev_pos = torch.cat([locs[:, 0:1].expand(-1, 1, 2), pos[:, :-1]], dim=1)
    travel = torch.linalg.norm(pos - prev_pos, dim=-1)  # [B, T]

    tw_open = torch.gather(tw[..., 0], 1, tw_idx)
    tw_close = torch.gather(tw[..., 1], 1, tw_idx)
    service = torch.gather(dur, 1, dur_idx)

    # Advance arrival times across the batch, resetting at depot visits.
    prev_t = torch.zeros(B, device=device, dtype=travel.dtype)
    margins: list[Tensor] = []
    inf_margin = torch.full((B,), float("inf"), device=device, dtype=travel.dtype)
    zeros_b = torch.zeros(B, device=device, dtype=travel.dtype)

    for t in range(T):
        depot_t = depot[:, t]
        base = torch.where(depot_t, zeros_b, prev_t + travel[:, t])
        arrive = torch.maximum(base, tw_open[:, t])
        margin_t = tw_close[:, t] - arrive
        margin_t = torch.where(depot_t, inf_margin, margin_t)
        margins.append(margin_t)
        prev_t = torch.where(depot_t, zeros_b, arrive + service[:, t])

    margins_t = torch.stack(margins, dim=1)  # [B, T]
    tw_slack_abs = margins_t.min(dim=1).values
    # Express time-window headroom or lateness as a fraction of the depot horizon.
    horizon = tw[:, 0, 1].to(device=device, dtype=travel.dtype)
    tw_slack = tw_slack_abs / horizon.clamp(min=1e-9)

    return {
        "capacity": cap_slack,
        "time_window": tw_slack,
        "spatial": torch.zeros(B, device=device, dtype=cap_slack.dtype),
    }


# ------------------------------------------------------------------------ OP


def _op_slack(td: Any, actions: Tensor) -> dict[str, Tensor]:
    margins = _op_margins(td, actions)
    margins["budget"] = margins["budget"].clamp(min=0.0)
    return margins


def _op_margins(td: Any, actions: Tensor) -> dict[str, Tensor]:
    """Signed route-budget headroom and collected prize."""
    locs = _get(td, "locs")  # [B, N, 2] (depot at 0)
    prize = _get(td, "prize")  # [B, N]
    budget = _get(td, "max_length")  # [B] or scalar
    actions = actions.long()
    B = actions.shape[0]
    device = actions.device
    N_loc = locs.shape[1]
    N_pr = prize.shape[1]

    loc_idx = actions.clamp_min(0).clamp_max(N_loc - 1)
    pos = torch.gather(locs, 1, loc_idx.unsqueeze(-1).expand(-1, -1, 2))
    prev_pos = torch.cat([locs[:, 0:1].expand(-1, 1, 2), pos[:, :-1]], dim=1)
    # Append return-to-depot leg
    last_to_depot = torch.linalg.norm(pos[:, -1] - locs[:, 0], dim=-1, keepdim=True)
    travel = torch.linalg.norm(pos - prev_pos, dim=-1)
    used = travel.sum(dim=1) + last_to_depot.squeeze(-1)

    pr_idx = actions.clamp_min(0).clamp_max(N_pr - 1)
    pr_step = torch.gather(prize, 1, pr_idx)
    pr_step = pr_step.masked_fill(actions == 0, 0.0)
    prize_sum = pr_step.sum(dim=1)

    if budget.dim() == 0:
        budget_b = budget.expand(B).to(device=device, dtype=used.dtype)
    elif budget.dim() == 1:
        budget_b = budget.to(device=device, dtype=used.dtype)
    else:
        budget_b = budget[..., 0].to(device=device, dtype=used.dtype)

    # Normalised slacks (see VRPTW for rationale).
    return {
        "budget": (budget_b - used) / budget_b.clamp(min=1e-9),
        "prize": prize_sum,  # objective term, not a normalised constraint
        "spatial": torch.zeros(B, device=device, dtype=used.dtype),
    }


# ---------------------------------------------------------------------- FJSP


def _fjsp_slack(td: Any, actions: Tensor) -> dict[str, Tensor]:
    """FJSP coarse slack (eligibility headroom)."""
    n_elig = _get(td, "num_eligible")  # [B, O]
    B = actions.shape[0]
    device = actions.device
    prec_slack = torch.zeros(B, device=device)
    elig_slack = n_elig.float().mean(dim=-1) if n_elig.ndim > 1 else n_elig.float()
    return {"precedence": prec_slack, "eligibility": elig_slack}


# --------------------------------------------------------------------- utils


def _get(td: Any, key: str) -> Tensor:
    """Convert an instance field to a tensor if necessary."""
    v = td[key] if hasattr(td, "__getitem__") else getattr(td, key)
    if not isinstance(v, Tensor):
        v = torch.as_tensor(v)
    return v
