"""Raw CP-SAT (cp_model) solver for the CVRPTW env.

The model uses AddMultipleCircuit with MTZ-style load and time-window
propagation. CP-SAT can certify optimality when its status is OPTIMAL;
a feasible incumbent at the time limit has no such guarantee.

Model::

    variables
        lit[i, j]   in {0, 1}    "arc (i, j) is used"
        load[i]     in [0, Q]    cumulative load at node i
        arr[i]      in [open_i, close_i]   arrival time at node i

    objective
        min  sum_{i != j}  d_ij * lit[i, j]

    constraints
        AddMultipleCircuit({(i, j, lit[i, j]) : i, j in [0, N), i != j}
                          U {(i, i, skip_i) : i in [0, N)})
        load[j] >= load[i] + demand[j]                    if lit[i, j]
        arr[j]  >= arr[i] + d_ij + service[i]             if lit[i, j]
        load[0] = 0          (depot start)
        arr[0]  in [0, max_close]   (depot start time -- soft)

`AddMultipleCircuit` lets the model pick the number of vehicles
implicitly (whatever respects the capacity / TW budget); each
circuit must start and end at the depot.

The cost returned is **negative** (the reward convention):
`-total_distance`. Distances are scaled to ints by `_DIST_SCALE`
because CP-SAT requires integer coefficients.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ortools.sat.python import cp_model

# Capacity gets its own scale (demand often in [0, 1] -> need >0
# resolution); travel-time stays unscaled because the CVRPTW
# expresses time-windows in the same units as Euclidean distance,
# so scaling distance would push arrival times past tw_close and
# make every model infeasible.
_CAP_SCALE = 1000
_DIST_SCALE = 1  # objective uses raw int distances (lossy by ~0.5%)


def solve_cvrptw_cpsat(
    instance: Any,
    *,
    max_runtime: float = 10.0,
    feasibility_only: bool = False,
) -> tuple[list[list[int]], float]:
    """Solve one CVRPTW instance with raw CP-SAT.

    Parameters
    ----------
    instance
        1-element TensorDict with `locs`, `demand`, `time_windows`
        (or `tw_open`/`tw_close`), `durations`, `vehicle_capacity`.
        Depot is index 0; the strips it from demand/durations
        (shape N-1) -- we left-pad with zeros to length N.
    max_runtime
        Wall-clock budget in seconds per instance.

    Returns
    -------
    routes : list[list[int]]
        One list of node indices per vehicle route (each route
        starts and ends at the depot, depots stripped from the
        list for compactness).
    cost : float
        `-total_distance` (the reward sign). `nan` on solver
        failure / time-out without an incumbent.
    """
    locs, demand, tw, service, Q = _unpack(instance)
    n = locs.shape[0]
    if n < 3:
        return [], float("nan")

    diff = locs[:, None, :] - locs[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    # Distances stay in raw int units to match the TW scale.
    d_int = np.round(dist).astype(np.int64)

    # Capacity gets its own (independent) scaling because the
    # CVRPTW normalises demand/capacity to [0, 1].
    Q_int = round(float(Q) * _CAP_SCALE)
    demand_int = np.round(demand * _CAP_SCALE).astype(np.int64)
    # Use ceil for TW close so we never accidentally bar the depot
    # from returning by 1 unit when scaling rounds down.
    tw_open_int = np.floor(tw[:, 0]).astype(np.int64)
    tw_close_int = np.ceil(tw[:, 1]).astype(np.int64)
    service_int = np.round(service).astype(np.int64)
    # `big_t` reserved for a future big-M-style relaxation of TW
    # constraints; current MTZ form uses OnlyEnforceIf instead.
    _ = int(tw_close_int.max() + d_int.max() * n + 1)

    model = cp_model.CpModel()

    # ---- Variables ----
    lit = {}
    arcs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = model.NewBoolVar(f"arc_{i}_{j}")
            lit[i, j] = v
            arcs.append((i, j, v))
    # AddMultipleCircuit handles VRP routing: all nodes start /
    # end at depot (index 0), each non-depot node visited exactly
    # once. No self-loop / skip literals -- those belong to
    # `AddCircuit` (single-circuit variant). Explicitly require
    # each customer to have exactly one outgoing arc (enforced
    # below) to lock visit-all semantics.
    model.AddMultipleCircuit(arcs)
    for i in range(1, n):
        model.Add(sum(lit[i, j] for j in range(n) if j != i) == 1)
        model.Add(sum(lit[j, i] for j in range(n) if j != i) == 1)

    # Load + arrival vars.
    load = [model.NewIntVar(0, Q_int, f"load_{i}") for i in range(n)]
    arr = [model.NewIntVar(int(tw_open_int[i]), int(tw_close_int[i]), f"arr_{i}") for i in range(n)]

    # ---- Constraints ----
    model.Add(load[0] == 0)
    for (i, j), used in lit.items():
        if j == 0:
            # Returning to depot: load resets, no arrival propagation.
            continue
        model.Add(load[j] >= load[i] + int(demand_int[j])).OnlyEnforceIf(used)
        model.Add(arr[j] >= arr[i] + int(d_int[i, j]) + int(service_int[i])).OnlyEnforceIf(used)
    # Depot start time can be 0 (allowed by tw_open_int[0] = 0 typically).

    # ---- Objective ----
    # feasibility_only=True: skip Minimize() and stop at first feasible
    # solution. Used by the CF feasibility check
    # (`neuro_co.cax.feasibility.vrptw_cp.vrptw_cp_is_feasible`): it
    # only needs a yes/no on "does a feasible routing exist for this
    # perturbed instance?". CP-SAT in feasibility mode terminates in
    # ~milliseconds on N=50 vs ~seconds when optimising distance.
    if not feasibility_only:
        model.Minimize(sum(int(d_int[i, j]) * lit[i, j] for (i, j) in lit))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_runtime)
    # Outer ProcessPoolExecutor already parallelises over instances,
    # so keep CP-SAT single-threaded to avoid oversubscription.
    solver.parameters.num_search_workers = 1
    solver.parameters.log_search_progress = False
    if feasibility_only:
        solver.parameters.stop_after_first_solution = True

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], float("nan")

    # Reconstruct routes by following arc literals. The depot has
    # one outgoing arc per vehicle, so `succ[0]` is a list; every
    # other node has at most one successor.
    succ_depot: list[int] = []
    succ: dict[int, int] = {}
    for (i, j), used in lit.items():
        if solver.Value(used) != 1:
            continue
        if i == 0:
            succ_depot.append(j)
        else:
            succ[i] = j
    routes: list[list[int]] = []
    for start in succ_depot:
        route = []
        cur = start
        while cur != 0:
            route.append(cur)
            nxt = succ.get(cur, 0)
            if nxt == 0 or nxt in route:
                break
            cur = nxt
        if route:
            routes.append(route)

    cost = -float(solver.ObjectiveValue()) / max(1, _DIST_SCALE)  # the sign
    return routes, cost


# ---------------------------------------------------------------------------
# Instance unpacker. the conventions:
#   locs                [N, 2]       depot at index 0
#   demand              [N-1]        depot stripped (we pad with 0)
#   durations           [N-1]        depot stripped (we pad with 0)
#   time_windows        [N, 2]       (open, close), depot has wide TW
#   vehicle_capacity    scalar       (Q)
# ---------------------------------------------------------------------------


def _unpack(instance: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    locs = instance["locs"].cpu().numpy()
    if locs.ndim == 3 and locs.shape[0] == 1:
        locs = locs[0]
    n = locs.shape[0]
    demand = instance["demand"].cpu().numpy()
    if demand.ndim == 2 and demand.shape[0] == 1:
        demand = demand[0]
    if demand.shape[0] == n - 1:
        demand = np.concatenate([[0.0], demand])
    tw = instance["time_windows"].cpu().numpy()
    if tw.ndim == 3 and tw.shape[0] == 1:
        tw = tw[0]
    if tw.shape[0] == n - 1:
        depot_tw = np.array([[0.0, float(tw[:, 1].max())]])
        tw = np.concatenate([depot_tw, tw], axis=0)
    dur = instance["durations"].cpu().numpy()
    if dur.ndim == 2 and dur.shape[0] == 1:
        dur = dur[0]
    if dur.shape[0] == n - 1:
        dur = np.concatenate([[0.0], dur])
    Q_arr = instance.get("vehicle_capacity", 1.0)
    Q = float(Q_arr.cpu().numpy().flatten()[0]) if hasattr(Q_arr, "cpu") else float(Q_arr)
    return (
        locs.astype(float),
        demand.astype(float),
        tw.astype(float),
        dur.astype(float),
        Q,
    )
