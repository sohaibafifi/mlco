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
from typing import Any, Literal, overload

import numpy as np
from ortools.sat.python import cp_model

from neuro_co.core.result import SolveResult, SolveStatus
from neuro_co.problems.solver_result import cp_result, validate_schedule


def _td_to_jobs(instance: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proc = instance["proc_times"].cpu().numpy()
    job_map = instance["ops_job_map"].cpu().numpy()
    order = instance["ops_sequence_order"].cpu().numpy()
    n_ops = int(job_map.shape[-1])
    pad = (
        instance["pad_mask"].cpu().numpy()
        if "pad_mask" in instance
        else np.zeros(n_ops, dtype=bool)
    )
    # Accept both the historical solver layout [M, O] and the core
    # environment layout [O, M]. The job map identifies the op axis.
    if proc.shape[0] == n_ops:
        proc = proc.T
    elif proc.shape[1] != n_ops:
        raise ValueError(f"cannot locate operation axis in proc_times {proc.shape} for {n_ops} ops")
    return proc, pad, job_map, order


@overload
def solve_fjsp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
    feasibility_only: bool = False,
    return_result: Literal[False] = False,
) -> tuple[list[tuple[int, int, float]], float]: ...


@overload
def solve_fjsp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
    feasibility_only: bool = False,
    return_result: Literal[True],
) -> SolveResult: ...


def solve_fjsp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 1000,
    feasibility_only: bool = False,
    return_result: bool = False,
) -> tuple[list[tuple[int, int, float]], float] | SolveResult:
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
    if return_result and (
        n_ops == 0
        or n_machines == 0
        or not np.isfinite(proc).all()
        or np.any(proc < 0)
        or job_map.shape != (n_ops,)
        or order.shape != (n_ops,)
        or pad.shape != (n_ops,)
        or not np.isfinite(job_map).all()
        or not np.isfinite(order).all()
        or np.any(job_map < 0)
        or np.any(order < 0)
        or not np.array_equal(job_map, np.floor(job_map))
        or not np.array_equal(order, np.floor(order))
        or len(set(zip(job_map.tolist(), order.tolist(), strict=True))) != n_ops
    ):
        raise ValueError("FJSP requires finite processing times and valid operation/job arrays")
    proc_int = np.maximum(0, np.round(proc * scale).astype(np.int64))
    horizon = max(1, int(proc_int.sum()))

    active = [i for i in range(n_ops) if not pad[i]]
    if not active:
        if return_result:
            return SolveResult(
                problem="fjsp",
                method="cpsat",
                status=SolveStatus.INVALID,
                solver_status="NO_ACTIVE_OPERATIONS",
            )
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
            if return_result:
                removed = bool(np.any(proc[:, i] > 0))
                return SolveResult(
                    problem="fjsp",
                    method="cpsat",
                    status=SolveStatus.UNKNOWN if removed else SolveStatus.INFEASIBLE,
                    feasible=None if removed else False,
                    solver_status="ROUNDING_REMOVED_ELIGIBILITY"
                    if removed
                    else "NO_ELIGIBLE_MACHINE",
                )
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
        if return_result:
            return cp_result(
                "fjsp",
                solver,
                status,
                scale=scale,
                metadata={"model_matches_source": bool(np.array_equal(proc_int / scale, proc))},
            )
        return [], float("nan")
    if feasibility_only and not return_result:
        return [], 0.0

    schedule: list[tuple[int, int, float]] = []
    for i in active:
        chosen_machine = next((m for m, c in machine_choices[i] if solver.Value(c)), -1)
        schedule.append((i, chosen_machine, solver.Value(starts[i]) / scale))
    ms = solver.Value(makespan) / scale
    if return_result:
        return cp_result(
            "fjsp",
            solver,
            status,
            solution={"schedule": schedule},
            objective=float(ms),
            scale=scale,
            feasibility_only=feasibility_only,
            validation=validate_schedule(schedule, proc, job_map, order, active),
            metadata={
                "scale": scale,
                "model_matches_source": bool(np.array_equal(proc_int / scale, proc)),
            },
        )
    return schedule, -float(ms)
