"""PDP plug-in: ConceptBank registration (+ optional OR-Tools solver)."""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import _register_lazy_solver
from neuro_co.problems.pdp.concepts import CONCEPTS

# PDPEnv.build_features columns: [x, y, is_pickup, is_delivery].
BANK = ConceptBank(problem="pdp", concepts=CONCEPTS, feature_slices={})
register_concept_bank(BANK)

_solver = _register_lazy_solver(
    "pdp",
    "ortools",
    "neuro_co.problems.pdp.ortools",
    "solve_pdp",
)
if _solver is not None:
    solve_pdp = _solver

__all__ = ["BANK"]
