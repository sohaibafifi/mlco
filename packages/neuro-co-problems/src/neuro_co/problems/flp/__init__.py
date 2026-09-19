"""FLP (facility location) plug-in: ConceptBank registration (+ optional solver).

Note: FLP has no core env yet; the bank registers for discovery, but its
concept extractors need a core FLP `State` to run.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import _register_lazy_solver
from neuro_co.problems.flp.concepts import CONCEPTS

BANK = ConceptBank(problem="flp", concepts=CONCEPTS)
register_concept_bank(BANK)

_solver = _register_lazy_solver(
    "flp",
    "lp",
    "neuro_co.problems.flp.lp",
    "solve_flp",
)
if _solver is not None:
    solve_flp = _solver

__all__ = ["BANK"]
