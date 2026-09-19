"""PDP plug-in: ConceptBank registration (+ optional OR-Tools solver)."""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import BASELINE_SOLVERS
from neuro_co.problems.pdp.concepts import CONCEPTS

# PDPEnv.build_features columns: [x, y, is_pickup, is_delivery].
BANK = ConceptBank(problem="pdp", concepts=CONCEPTS, feature_slices={})
register_concept_bank(BANK)

try:  # optional OR-Tools solver (cp extra)
    from neuro_co.problems.pdp.ortools import solve_pdp

    BASELINE_SOLVERS[("pdp", "ortools")] = solve_pdp
except ImportError:  # pragma: no cover
    pass

__all__ = ["BANK"]
