"""OP (Orienteering Problem) LP-relaxation dual extractor.

Encoding: prize-collecting TSP with travel-budget constraint and
Miller-Tucker-Zemlin subtour elimination.

Variables (LP-relaxed):
  x_ij in [0, 1]  forall (i, j), i != j   (use arc)
  z_i  in [0, 1]  forall i                (visit indicator)
  u_i  >= 0       forall i                (MTZ order)

Objective:
  maximise  sum_i prize_i * z_i

Constraints (by family, surfaced as duals):
  travel_budget : node-degree balance for every i,
             sum_j x_ji == z_i  and  sum_j x_ij == z_i,
             plus MTZ subtour elimination
             u_j >= u_i + 1 - N*(1 - x_ij), and
             sum_{i,j} d_ij * x_ij <= max_length
  prize    : 0 <= z_i <= 1 (already in var bounds);
             one explicit row pinning the depot z_0 = 1 anchors
             the family so dual extraction has something to read.

The OP LP relaxation is loose (continuous arc usage with MTZ),
but the per-family aggregate duals are still informative as
relative constraint pressure -- the same pattern as
`vrptw_lp.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuro_co.cax.duals.lp import solve_lp_and_extract_duals


def op_lp_duals(
    instance: dict[str, Any],
    *,
    time_limit_s: float = 5.0,
    aggregate: str = "mean",
) -> dict[str, float]:
    """Solve OP LP, return per-family mean |dual|."""
    locs = np.asarray(instance["locs"], dtype=float)  # [N, 2]
    N = locs.shape[0]
    if N < 3:
        return {"travel_budget": 0.0, "prize": 0.0}

    prize = np.asarray(
        instance.get("prize", np.ones(N)),
        dtype=float,
    )
    # the OP encodes max_length as either a per-node tensor (filled
    # with the same scalar) or a single scalar; reduce to one budget.
    ml = instance.get("max_length", np.array(2.0))
    ml_arr = np.asarray(ml, dtype=float)
    max_length = float(ml_arr.flatten()[0]) if ml_arr.size else 2.0

    d = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)

    def _build(solver: Any) -> dict[str, list[Any]]:
        # Variables.
        x = {
            (i, j): solver.NumVar(0.0, 1.0, f"x_{i}_{j}")
            for i in range(N)
            for j in range(N)
            if i != j
        }
        # z_i bound widened to [0, +inf); the upper bound z_i <= 1
        # is lifted to an explicit row so its dual is exposed to the
        # extractor (variable-bound multipliers are not read).
        z = [solver.NumVar(0.0, solver.infinity(), f"z_{i}") for i in range(N)]
        u = [solver.NumVar(0.0, float(N), f"u_{i}") for i in range(N)]

        # Objective: maximise total prize collected.
        solver.Maximize(solver.Sum(float(prize[i]) * z[i] for i in range(N)))

        family_rows: dict[str, list[Any]] = {
            "travel_budget": [],
            "prize": [],
        }

        # Depot anchored as visited, plus z_i <= 1 explicit caps for
        # non-depot nodes (their bound multipliers carry the
        # prize-collection pressure).
        family_rows["prize"].append(solver.Add(z[0] == 1.0, "depot_visited"))
        for i in range(1, N):
            family_rows["prize"].append(solver.Add(z[i] <= 1.0, f"visit_cap_{i}"))

        # Degree balance: sum incoming == sum outgoing == z_i, for every node.
        for i in range(N):
            family_rows["travel_budget"].append(
                solver.Add(
                    solver.Sum(x[j, i] for j in range(N) if j != i) == z[i],
                    f"deg_in_{i}",
                )
            )
            family_rows["travel_budget"].append(
                solver.Add(
                    solver.Sum(x[i, j] for j in range(N) if j != i) == z[i],
                    f"deg_out_{i}",
                )
            )

        # MTZ subtour elimination for non-depot pairs.
        for (i, j), arc in x.items():
            if i == 0 or j == 0:
                continue
            family_rows["travel_budget"].append(
                solver.Add(
                    u[j] >= u[i] + 1 - float(N) * (1 - arc),
                    f"mtz_{i}_{j}",
                )
            )

        # Budget constraint: total travel <= max_length.
        family_rows["travel_budget"].append(
            solver.Add(
                solver.Sum(float(d[i, j]) * x[i, j] for (i, j) in x) <= max_length,
                "budget",
            )
        )

        return family_rows

    result = solve_lp_and_extract_duals(_build, time_limit_s=time_limit_s, aggregate=aggregate)
    return result.multipliers
