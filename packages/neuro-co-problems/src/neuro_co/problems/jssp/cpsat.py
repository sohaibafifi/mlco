"""OR-Tools CP-SAT baseline for the Job-Shop Scheduling Problem.

Takes a single-instance the `JSSPEnv` TensorDict slice and returns
`(schedule, makespan)`. CP-SAT can solve JSSP to optimality in seconds
at the sizes we use for AET (10x5 …
20x10).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Literal, overload

import numpy as np
from ortools.sat.python import cp_model

from neuro_co.core.result import SolveResult
from neuro_co.problems.solver_result import cp_result, validate_schedule


def _td_to_jobs(instance: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-op (machine, duration, job, sequence-order) arrays.

    Reads `proc_times` `[M, N]`, `ops_ma_adj` `[M, N]`,
    `ops_job_map` `[N]`, `ops_sequence_order` `[N]` from the TensorDict.
    """
    proc_times = instance["proc_times"].cpu().numpy()  # [M, N]
    ops_ma_adj = instance["ops_ma_adj"].cpu().numpy()  # [M, N], binary
    ops_job_map = instance["ops_job_map"].cpu().numpy()  # [N]
    order = instance["ops_sequence_order"].cpu().numpy()  # [N]
    machine = ops_ma_adj.argmax(axis=0)  # [N]
    duration = (proc_times * ops_ma_adj).sum(axis=0)  # [N]
    return machine, duration, ops_job_map, order


@overload
def solve_jssp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
    return_result: Literal[False] = False,
) -> tuple[list[tuple[int, float]], float]: ...


@overload
def solve_jssp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
    return_result: Literal[True],
) -> SolveResult: ...


def solve_jssp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
    return_result: bool = False,
) -> tuple[list[tuple[int, float]], float] | SolveResult:
    """Solve a single JSSP instance with CP-SAT.

    Parameters
    ----------
    instance
        TensorDict slice from the upstream env.
    max_runtime
        Wall-clock budget in seconds for CP-SAT.
    scale
        Multiplier for converting float durations to integers (CP-SAT
        needs integer intervals). `1000` keeps millisecond precision.

    Returns
    -------
    schedule
        List of `(op_index, start_time)` ordered by op index.
    makespan
        Best makespan found (`float("nan")` on timeout with no feasible
        solution).
    """
    if return_result:
        proc = instance["proc_times"].cpu().numpy()
        adjacency = instance["ops_ma_adj"].cpu().numpy()
        if (
            proc.ndim != 2
            or not proc.size
            or proc.shape != adjacency.shape
            or not np.isfinite(proc).all()
            or np.any(proc < 0)
            or not np.isin(adjacency, [0, 1]).all()
            or not np.all(adjacency.sum(axis=0) == 1)
            or np.any((proc * adjacency).sum(axis=0) <= 0)
        ):
            raise ValueError(
                "JSSP requires finite processing times and exactly one eligible machine per operation"
            )
    machine, duration, job_map, order = _td_to_jobs(instance)
    if return_result and (
        job_map.shape != duration.shape
        or order.shape != duration.shape
        or not np.isfinite(job_map).all()
        or not np.isfinite(order).all()
        or np.any(job_map < 0)
        or np.any(order < 0)
        or not np.array_equal(job_map, np.floor(job_map))
        or not np.array_equal(order, np.floor(order))
        or len(set(zip(job_map.tolist(), order.tolist(), strict=True))) != len(duration)
    ):
        raise ValueError(
            "JSSP job and sequence arrays must be nonnegative integral operation vectors"
        )
    n_ops = int(duration.shape[0])

    dur_int = np.maximum(1, np.round(duration * scale).astype(np.int64))
    horizon = int(dur_int.sum())

    model = cp_model.CpModel()
    starts = [model.NewIntVar(0, horizon, f"s_{i}") for i in range(n_ops)]
    ends = [model.NewIntVar(0, horizon, f"e_{i}") for i in range(n_ops)]
    intervals = [
        model.NewIntervalVar(starts[i], int(dur_int[i]), ends[i], f"iv_{i}") for i in range(n_ops)
    ]

    # No-overlap per machine.
    by_machine: dict[int, list[Any]] = {}
    for i in range(n_ops):
        by_machine.setdefault(int(machine[i]), []).append(intervals[i])
    for ivals in by_machine.values():
        if len(ivals) > 1:
            model.AddNoOverlap(ivals)

    # Precedence within each job.
    by_job: dict[int, list[tuple[int, int]]] = {}
    for i in range(n_ops):
        by_job.setdefault(int(job_map[i]), []).append((int(order[i]), i))
    for ops in by_job.values():
        ops.sort()
        for (_, i1), (_, i2) in pairwise(ops):
            model.Add(starts[i2] >= ends[i1])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_runtime)
    solver.parameters.num_search_workers = 1  # respect parallel mode at the runner level
    status = solver.Solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not feasible:
        if return_result:
            return cp_result(
                "jssp",
                solver,
                status,
                scale=scale,
                metadata={"model_matches_source": bool(np.array_equal(dur_int / scale, duration))},
            )
        return [], float("nan")

    schedule = [(i, solver.Value(starts[i]) / scale) for i in range(n_ops)]
    ms = solver.Value(makespan) / scale
    if return_result:
        processing = np.zeros((int(machine.max()) + 1, n_ops), dtype=float)
        processing[machine, np.arange(n_ops)] = duration
        validation = validate_schedule(
            [(i, int(machine[i]), start) for i, start in schedule],
            processing,
            job_map,
            order,
            list(range(n_ops)),
        )
        return cp_result(
            "jssp",
            solver,
            status,
            solution={"schedule": schedule},
            objective=float(ms),
            scale=scale,
            validation=validation,
            metadata={
                "scale": scale,
                "model_matches_source": bool(np.array_equal(dur_int / scale, duration)),
            },
        )
    # the reward convention is `-makespan` (higher is better); return
    # negative so callers can `abs()` uniformly across solvers.
    return schedule, -float(ms)


def instance2data(instance: Any) -> dict[str, Any]:
    """Return the per-op arrays as a dict."""
    machine, duration, job_map, order = _td_to_jobs(instance)
    return {
        "machine": machine,
        "duration": duration,
        "job": job_map,
        "order": order,
        "n_ops": int(duration.shape[0]),
        "n_machines": int(instance["proc_times"].shape[0]),
        "n_jobs": int(np.unique(job_map).size),
    }
