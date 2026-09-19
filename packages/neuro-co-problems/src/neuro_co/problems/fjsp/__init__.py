"""FJSP plug-in: ConceptBank registration (+ optional OR-Tools solver)."""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import _register_lazy_solver
from neuro_co.problems.fjsp.concepts import CONCEPTS

# FJSPEnv.build_features columns: [mean_proc, num_eligible, ready, job_progress].
BANK = ConceptBank(
    problem="fjsp", concepts=CONCEPTS, feature_slices={"mean_proc": 0, "num_eligible": 1}
)
register_concept_bank(BANK)

_solver = _register_lazy_solver(
    "fjsp",
    "cpsat",
    "neuro_co.problems.fjsp.cpsat",
    "solve_fjsp",
    dependency="ortools",
)
if _solver is not None:
    solve_fjsp = _solver

__all__ = ["BANK"]
