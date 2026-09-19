"""FJSP plug-in: ConceptBank registration (+ optional OR-Tools solver)."""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import BASELINE_SOLVERS
from neuro_co.problems.fjsp.concepts import CONCEPTS

# FJSPEnv.build_features columns: [mean_proc, num_eligible, ready, job_progress].
BANK = ConceptBank(
    problem="fjsp", concepts=CONCEPTS, feature_slices={"mean_proc": 0, "num_eligible": 1}
)
register_concept_bank(BANK)

try:  # optional CP-SAT solver (cp extra)
    from neuro_co.problems.fjsp.cpsat import solve_fjsp

    BASELINE_SOLVERS[("fjsp", "cpsat")] = solve_fjsp
except ImportError:  # pragma: no cover
    pass

__all__ = ["BANK"]
