"""JSSP plug-in: ConceptBank registration (+ optional CP-SAT solver).

Note: JSSP has no core env yet; the bank registers so tooling can discover
it, but its concept extractors need a core JSSP `State` to run.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import BASELINE_SOLVERS
from neuro_co.problems.jssp.concepts import CONCEPTS

BANK = ConceptBank(problem="jssp", concepts=CONCEPTS)
register_concept_bank(BANK)

try:  # optional CP-SAT solver (cp extra)
    from neuro_co.problems.jssp.cpsat import solve_jssp

    BASELINE_SOLVERS[("jssp", "cpsat")] = solve_jssp
except ImportError:  # pragma: no cover
    pass

__all__ = ["BANK"]
