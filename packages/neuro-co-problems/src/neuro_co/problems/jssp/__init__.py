"""JSSP plug-in: ConceptBank registration (+ optional CP-SAT solver).

Note: JSSP has no core env yet; the bank registers so tooling can discover
it, but its concept extractors need a core JSSP `State` to run.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import _register_lazy_solver
from neuro_co.problems.jssp.concepts import CONCEPTS

BANK = ConceptBank(problem="jssp", concepts=CONCEPTS)
register_concept_bank(BANK)

_solver = _register_lazy_solver(
    "jssp",
    "cpsat",
    "neuro_co.problems.jssp.cpsat",
    "solve_jssp",
    dependency="ortools",
)
if _solver is not None:
    solve_jssp = _solver

__all__ = ["BANK"]
