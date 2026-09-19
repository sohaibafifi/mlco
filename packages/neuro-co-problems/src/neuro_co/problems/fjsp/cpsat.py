"""CP-SAT baseline for FJSP (flexible job-shop).

Uses the `ortools.sat.python.cp_model` API. Extends the JSSP CP
model with alternative intervals: each op picks exactly one
eligible machine via `AddExactlyOne` over per-machine optional
intervals; the chosen machine's interval participates in that
machine's `AddNoOverlap`. Standard FJSP CP formulation.

Registered as `BASELINE_SOLVERS[("fjsp", "cpsat")]`. The engine key
is `cpsat` because this uses the CP-SAT model API (not the OR-Tools
routing/scheduling specialised API).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from ortools.sat.python import cp_model


def _td_to_jobs(instance: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proc = instance["proc_times"].cpu().numpy()
    pad = (
        instance["pad_mask"].cpu().numpy()
        if "pad_mask" in instance
        else np.zeros(proc.shape[-1], dtype=bool)
    )
    job_map = instance["ops_job_map"].cpu().numpy()
    order = instance["ops_sequence_order"].cpu().numpy()
    # Accept both the historical solver layout [M, O] and the core
    # environment layout [O, M]. The job map identifies the op axis.
    n_ops = int(job_map.shape[-1])
    if proc.shape[0] == n_ops:
        proc = proc.T
    elif proc.shape[1] != n_ops:
        raise ValueError(f"cannot locate operation axis in proc_times {proc.shape} for {n_ops} ops")
    return proc, pad, job_map, order


def solve_fjsp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
    feasibility_only: bool = False,
) -> tuple[list[tuple[int, int, float]], float]:
    """Solve a single FJSP instance with CP-SAT.

    Parameters
    ----------
    feasibility_only
        If True, drop the makespan-minimising objective and halt at
        the first feasible schedule. Mirrors the VRPTW solver's flag;
        used by `neuro_co.cax.feasibility.fjsp_cp.fjsp_cp_is_feasible`
        to convert a COP pass into a CSP feasibility-decision.

    Returns
    -------
    schedule
        List of `(op_index, chosen_machine, start_time)`. When
        `feasibility_only=True` the returned `cost` is `0.0` on
        success and `nan` on infeasibility.
    cost
        `-makespan` (the reward convention).
    """
    proc, pad, job_map, order = _td_to_jobs(instance)
    n_machines, n_ops = proc.shape
    proc_int = np.maximum(0, np.round(proc * scale).astype(np.int64))
    horizon = max(1, int(proc_int.sum()))

    active = [i for i in range(n_ops) if not pad[i]]
    if not active:
        return [], float("nan")

    model = cp_model.CpModel()
    starts: dict[int, Any] = {i: model.NewIntVar(0, horizon, f"s_{i}") for i in active}
    ends: dict[int, Any] = {i: model.NewIntVar(0, horizon, f"e_{i}") for i in active}

    # Per (op, machine) optional interval. machine_choices[i] = list of
    # (machine_idx, bool_var) for downstream resolution.
    machine_choices: dict[int, list[tuple[int, Any]]] = {i: [] for i in active}
    intervals_per_machine: dict[int, list[Any]] = {m: [] for m in range(n_machines)}
    for i in active:
        for m in range(n_machines):
            d = int(proc_int[m, i])
            if d <= 0:
                continue
            chosen = model.NewBoolVar(f"x_{i}_{m}")
            ivar = model.NewOptionalIntervalVar(starts[i], d, ends[i], chosen, f"iv_{i}_{m}")
            intervals_per_machine[m].append(ivar)
            machine_choices[i].append((m, chosen))
        if not machine_choices[i]:
            return [], float("nan")  # op with no eligible machine
        model.AddExactlyOne([c for _, c in machine_choices[i]])

    for ivals in intervals_per_machine.values():
        if len(ivals) > 1:
            model.AddNoOverlap(ivals)

    # Precedence within job: ops sharing a job_map id, ordered by
    # ops_sequence_order, must run sequentially.
    by_job: dict[int, list[tuple[int, int]]] = {}
    for i in active:
        by_job.setdefault(int(job_map[i]), []).append((int(order[i]), i))
    for ops in by_job.values():
        ops.sort()
        for (_, i1), (_, i2) in pairwise(ops):
            model.Add(starts[i2] >= ends[i1])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [ends[i] for i in active])
    # Drop the minimisation objective under `feasibility_only=True`:
    # the solver halts at the first valid schedule (any schedule with
    # makespan <= horizon, where horizon is the worst-case sum of
    # processing times). Matches the VRPTW solver's flag.
    if not feasibility_only:
        model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_runtime)
    solver.parameters.num_search_workers = 1
    if feasibility_only:
        solver.parameters.stop_after_first_solution = True
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], float("nan")
    if feasibility_only:
        return [], 0.0

    schedule: list[tuple[int, int, float]] = []
    for i in active:
        chosen_machine = next((m for m, c in machine_choices[i] if solver.Value(c)), -1)
        schedule.append((i, chosen_machine, solver.Value(starts[i]) / scale))
    ms = solver.Value(makespan) / scale
    return schedule, -float(ms)
