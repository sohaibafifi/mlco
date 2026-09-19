"""OP (orienteering) plug-in: ConceptBank registration (+ optional solver).

Registers `ConceptBank(problem="op")` into the core concept registry. The
CP-SAT solver is registered only when the `cp` extra is installed.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import _register_lazy_solver
from neuro_co.problems.op.concepts import CONCEPTS

# OPEnv.build_features columns: [x, y, prize].
BANK = ConceptBank(problem="op", concepts=CONCEPTS, feature_slices={"prize": 2})
register_concept_bank(BANK)

_solver = _register_lazy_solver(
    "op",
    "cpsat",
    "neuro_co.problems.op.cpsat",
    "solve_op",
    dependency="ortools",
)
if _solver is not None:
    solve_op = _solver

__all__ = ["BANK"]
