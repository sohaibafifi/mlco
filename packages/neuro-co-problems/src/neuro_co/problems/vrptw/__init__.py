"""CVRPTW plug-in: ConceptBank registration (+ optional baseline solvers).

Importing this module registers a `ConceptBank` for `"cvrptw"` (alias
`"vrptw"`) into the core concept registry. Baseline solvers are registered
only when the `cp` extra (pyvrp / OR-Tools) is installed.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import BASELINE_SOLVERS
from neuro_co.problems.vrptw.concepts import CONCEPTS

# Feature-column slices into CVRPTWEnv.build_features:
# [x, y, demand, tw_early, tw_late, t_now].
BANK = ConceptBank(
    problem="cvrptw",
    concepts=CONCEPTS,
    feature_slices={"demand": 2, "tw_early": 3, "tw_late": 4},
)
register_concept_bank(BANK)
register_concept_bank(
    ConceptBank(problem="vrptw", concepts=CONCEPTS, feature_slices=BANK.feature_slices)
)

try:  # optional pyvrp solver (cp extra)
    from neuro_co.problems.vrptw.pyvrp import solve_cvrptw

    BASELINE_SOLVERS[("cvrptw", "pyvrp")] = solve_cvrptw
    BASELINE_SOLVERS[("vrptw", "pyvrp")] = solve_cvrptw
except ImportError:  # pragma: no cover
    pass

try:  # optional CP-SAT solver (cp extra)
    from neuro_co.problems.vrptw.cpsat import solve_cvrptw_cpsat

    BASELINE_SOLVERS[("cvrptw", "cpsat")] = solve_cvrptw_cpsat
    BASELINE_SOLVERS[("vrptw", "cpsat")] = solve_cvrptw_cpsat
except ImportError:  # pragma: no cover
    pass

__all__ = ["BANK"]
