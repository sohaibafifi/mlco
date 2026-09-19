"""FLP (facility location) plug-in: ConceptBank registration (+ optional solver).

Note: FLP has no core env yet; the bank registers for discovery, but its
concept extractors need a core FLP `State` to run.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import BASELINE_SOLVERS
from neuro_co.problems.flp.concepts import CONCEPTS

BANK = ConceptBank(problem="flp", concepts=CONCEPTS)
register_concept_bank(BANK)

try:  # optional MIP solver via OR-Tools linear_solver / pywraplp (cp extra)
    from neuro_co.problems.flp.lp import solve_flp

    BASELINE_SOLVERS[("flp", "lp")] = solve_flp
except ImportError:  # pragma: no cover
    pass

__all__ = ["BANK"]
