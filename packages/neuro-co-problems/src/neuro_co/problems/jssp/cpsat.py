"""OR-Tools CP-SAT baseline for the Job-Shop Scheduling Problem.

Takes a single-instance the `JSSPEnv` TensorDict slice and returns
`(schedule, makespan)`. CP-SAT can solve JSSP to optimality in seconds
at the sizes we use for AET (10x5 …
20x10).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from ortools.sat.python import cp_model


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


def solve_jssp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
) -> tuple[list[tuple[int, float]], float]:
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
    machine, duration, job_map, order = _td_to_jobs(instance)
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
        return [], float("nan")

    schedule = [(i, solver.Value(starts[i]) / scale) for i in range(n_ops)]
    ms = solver.Value(makespan) / scale
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
