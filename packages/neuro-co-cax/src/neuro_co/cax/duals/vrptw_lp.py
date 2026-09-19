"""Build the arc-based CVRPTW LP relaxation.

Variables represent fractional arcs, arrival times, and cumulative loads.
The objective is total travel cost. Constraint families group load-MTZ and
load bounds under capacity, time-MTZ rows under time windows, and degree
and depot-flow rows under spatial constraints.

`lp.py` solves the model and aggregates absolute row duals by family. These
weights describe this relaxation and depend on its formulation and scaling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuro_co.cax.duals.lp import solve_lp_and_extract_duals


def vrptw_lp_duals(
    instance: dict[str, Any],
    *,
    time_limit_s: float = 5.0,
    big_m: float | None = None,
    capacity_default: float = 1.0,
    aggregate: str = "mean",
) -> dict[str, float]:
    """Solve arc-based CVRPTW LP, return per-family mean |dual|."""
    locs = np.asarray(instance["locs"], dtype=float)  # [N, 2]
    N = locs.shape[0]
    if N < 3:
        return {"capacity": 0.0, "time_window": 0.0, "spatial": 0.0}

    demand = _pad_depot(np.asarray(instance.get("demand", np.zeros(N)), dtype=float), N)
    tw = _pad_depot_tw(
        np.asarray(
            instance.get("time_windows", np.tile([0.0, 1e6], (N, 1))),
            dtype=float,
        ),
        N,
    )
    dur = _pad_depot(np.asarray(instance.get("durations", np.zeros(N)), dtype=float), N)
    Q = _to_scalar(instance.get("vehicle_capacity", capacity_default))
    d = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)
    bigM = float(big_m if big_m is not None else tw[:, 1].max() + d.max() * N)

    def _build(solver: Any) -> dict[str, list[Any]]:
        # ---- Variables ----
        x = {}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                x[i, j] = solver.NumVar(0.0, 1.0, f"x_{i}_{j}")
        a = [solver.NumVar(float(tw[i, 0]), float(tw[i, 1]), f"a_{i}") for i in range(N)]
        u = [solver.NumVar(0.0, Q, f"u_{i}") for i in range(N)]

        # ---- Objective ----
        solver.Minimize(solver.Sum(d[i, j] * x[i, j] for i, j in x))

        # ---- Constraints, partitioned by family ----
        family_rows: dict[str, list[Any]] = {
            "capacity": [],
            "time_window": [],
            "spatial": [],
        }

        # Degree: each non-depot customer visited exactly once + depot balance.
        for i in range(1, N):
            family_rows["spatial"].append(
                solver.Add(
                    solver.Sum(x[i, j] for j in range(N) if j != i) == 1,
                    f"deg_out_{i}",
                )
            )
            family_rows["spatial"].append(
                solver.Add(
                    solver.Sum(x[j, i] for j in range(N) if j != i) == 1,
                    f"deg_in_{i}",
                )
            )
        family_rows["spatial"].append(
            solver.Add(
                solver.Sum(x[0, j] for j in range(1, N)) - solver.Sum(x[j, 0] for j in range(1, N))
                == 0,
                "depot_balance",
            )
        )

        # Capacity MTZ + load lower bounds.
        for (i, j), var in x.items():
            if j == 0:
                continue
            family_rows["capacity"].append(
                solver.Add(u[j] >= u[i] + demand[j] - Q * (1 - var), f"cap_mtz_{i}_{j}")
            )
        for i in range(1, N):
            family_rows["capacity"].append(solver.Add(u[i] >= demand[i], f"cap_lb_{i}"))

        # Time-window MTZ.
        for (i, j), var in x.items():
            if j == 0:
                continue
            family_rows["time_window"].append(
                solver.Add(
                    a[j] >= a[i] + d[i, j] + dur[i] - bigM * (1 - var),
                    f"tw_mtz_{i}_{j}",
                )
            )

        return family_rows

    result = solve_lp_and_extract_duals(_build, time_limit_s=time_limit_s, aggregate=aggregate)
    return result.multipliers


# ---------------------------------------------------------------------------
# the convention helpers: depot is index 0, customer features are length
# (N-1) (depot stripped). Pad with a depot row (zero demand / [0, big_T] TW
# / zero duration) so downstream LP / subgrad code sees uniform N-length
# arrays.
# ---------------------------------------------------------------------------


def _pad_depot(arr: np.ndarray, N: int) -> np.ndarray:
    """Left-pad a length-(N-1) customer vector with a zero depot entry."""
    if arr.shape[0] == N:
        return arr
    if arr.shape[0] == N - 1:
        return np.concatenate([[0.0], arr])
    raise ValueError(f"unexpected demand shape {arr.shape}; expected ({N},) or ({N - 1},)")


def _pad_depot_tw(tw: np.ndarray, N: int) -> np.ndarray:
    """Left-pad a length-(N-1) TW matrix with a wide [0, max] depot window."""
    if tw.shape[0] == N:
        return tw
    if tw.shape[0] == N - 1:
        depot = np.array([[0.0, float(tw[:, 1].max())]])
        return np.concatenate([depot, tw], axis=0)
    raise ValueError(f"unexpected time_windows shape {tw.shape}")


def _to_scalar(v: Any, default: float = 1.0) -> float:
    """Coerce a `(1,)` numpy array / 0-d ndarray / Python number to `float`."""
    if v is None:
        return float(default)
    if hasattr(v, "shape") and getattr(v, "shape", ()) != ():
        return float(np.asarray(v).reshape(-1)[0])
    return float(v)
