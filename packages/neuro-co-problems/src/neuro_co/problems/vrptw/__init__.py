"""CVRPTW plug-in: ConceptBank registration (+ optional baseline solvers).

Importing this module registers a `ConceptBank` for `"cvrptw"` (alias
`"vrptw"`) into the core concept registry. Baseline solvers are registered
only when the `cp` extra (pyvrp / OR-Tools) is installed.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import _register_lazy_solver
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

_solver = _register_lazy_solver(
    "cvrptw",
    "pyvrp",
    "neuro_co.problems.vrptw.pyvrp",
    "solve_cvrptw",
    dependency="pyvrp",
    aliases=("vrptw",),
)
if _solver is not None:
    solve_cvrptw = _solver

_solver = _register_lazy_solver(
    "cvrptw",
    "cpsat",
    "neuro_co.problems.vrptw.cpsat",
    "solve_cvrptw_cpsat",
    dependency="ortools",
    aliases=("vrptw",),
)
if _solver is not None:
    solve_cvrptw_cpsat = _solver

__all__ = ["BANK"]
