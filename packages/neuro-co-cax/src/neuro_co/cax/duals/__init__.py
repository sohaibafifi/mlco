"""Constraint-family multipliers for CVRPTW, OP, and FJSP.

The ``lp`` backend aggregates shadow prices from an OR-Tools GLOP
relaxation. The ``subgrad`` backend updates multipliers using a relaxed
subproblem. Both return a mapping from constraint family to multiplier.
"""

from __future__ import annotations

from typing import Any

import torch

from neuro_co.cax.constraint_map import get_constraints


def get_multipliers(
    problem: str,
    instance: dict[str, Any],
    *,
    method: str = "lp",
    **kwargs: Any,
) -> dict[str, float]:
    """Dispatch to the per-problem dual extractor.

    Parameters
    ----------
    problem
        Problem name (`'vrptw'`, `'fjsp'`, `'op'`).
    instance
        Flat dict of the instance fields the backend needs
        (`'locs'`, `'demand'`, `'time_windows'`, ...). Usually
        built with `state_to_instance(state, env, problem)`.
    method
        `'lp'` (GLOP backend, default) or `'subgrad'` (Beasley).
    **kwargs
        Forwarded to the backend (`time_limit_s`, `max_iters`,
        `step_size`, ...).

    Returns
    -------
    dict[str, float]
        Constraint-family name -> Lagrangian multiplier (absolute
        value). Order matches `PROBLEM_CONSTRAINTS[problem]`.

    Raises
    ------
    NotImplementedError
        If the (problem, method) pair isn't wired yet.
    """
    key = (problem.lower(), method.lower())
    if key in (("vrptw", "lp"), ("cvrptw", "lp")):
        from neuro_co.cax.duals.vrptw_lp import vrptw_lp_duals

        return vrptw_lp_duals(instance, **kwargs)
    if key in (("vrptw", "subgrad"), ("cvrptw", "subgrad")):
        from neuro_co.cax.duals.vrptw_subgrad import vrptw_subgrad_duals

        return vrptw_subgrad_duals(instance, **kwargs)
    if key == ("op", "lp"):
        from neuro_co.cax.duals.op_lp import op_lp_duals

        return op_lp_duals(instance, **kwargs)
    if key == ("op", "subgrad"):
        from neuro_co.cax.duals.op_subgrad import op_subgrad_duals

        return op_subgrad_duals(instance, **kwargs)
    if key == ("fjsp", "lp"):
        from neuro_co.cax.duals.fjsp_lp import fjsp_lp_duals

        return fjsp_lp_duals(instance, **kwargs)
    if key == ("fjsp", "subgrad"):
        from neuro_co.cax.duals.fjsp_subgrad import fjsp_subgrad_duals

        return fjsp_subgrad_duals(instance, **kwargs)
    raise NotImplementedError(
        f"No multiplier estimator for problem={problem!r}, method={method!r}. "
        f"Supported problems are VRPTW, OP, and FJSP with lp or subgrad."
    )


def state_to_instance(state: Any, env: Any, problem: str, batch_idx: int = 0) -> dict[str, Any]:
    """Single-instance numpy dict from a core `State` + `Env`.

    The LP / subgradient encoders are per-instance; this takes only the
    requested batch element. Use :func:`get_batch_multipliers` when an
    attribution batch contains distinct instances.
    """
    from neuro_co.cax._instance import to_instance

    return to_instance(state, env, problem, batch_idx=batch_idx)


def normalize_multipliers(
    raw: dict[str, float],
    family_names: list[str],
    *,
    floor: float = 0.1,
) -> dict[str, float]:
    """Return non-negative, mean-one dual weights in declared family order.

    The floor prevents a zero LP dual from deleting an entire explanatory
    family. If all duals are zero, the neutral all-ones weighting is used.
    """
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must lie in [0, 1), got {floor}")
    values = torch.tensor([max(0.0, float(raw.get(n, 0.0))) for n in family_names])
    total = float(values.sum())
    if total <= 0.0:
        return dict.fromkeys(family_names, 1.0)
    scaled = values * (len(family_names) / total)
    weights = floor + (1.0 - floor) * scaled
    return {name: float(weights[i]) for i, name in enumerate(family_names)}


def get_batch_multipliers(
    problem: str,
    state: Any,
    env: Any,
    *,
    method: str = "lp",
    floor: float = 0.1,
    **kwargs: Any,
) -> tuple[torch.Tensor, list[dict[str, float]], list[dict[str, float]]]:
    """Solve and normalize one dual vector for every instance in a batch.

    Returns ``(weights, raw_records, normalized_records)`` where ``weights``
    has shape ``[B, K]`` in the order declared by ``get_constraints``.
    """
    batch = int(state.coords.shape[0] if hasattr(state, "coords") else state.proc_times.shape[0])
    names = [name for name, _ in get_constraints(problem)]
    raw_records: list[dict[str, float]] = []
    normalized_records: list[dict[str, float]] = []
    for batch_idx in range(batch):
        instance = state_to_instance(state, env, problem, batch_idx=batch_idx)
        raw = get_multipliers(problem, instance, method=method, **kwargs)
        normalized = normalize_multipliers(raw, names, floor=floor)
        raw_records.append(raw)
        normalized_records.append(normalized)
    weights = torch.tensor(
        [[record[name] for name in names] for record in normalized_records],
        dtype=torch.float32,
    )
    return weights, raw_records, normalized_records


__all__ = [
    "get_batch_multipliers",
    "get_multipliers",
    "normalize_multipliers",
    "state_to_instance",
]
