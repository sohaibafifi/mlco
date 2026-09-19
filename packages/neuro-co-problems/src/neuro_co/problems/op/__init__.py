"""OP (orienteering) plug-in: ConceptBank registration (+ optional solver).

Registers `ConceptBank(problem="op")` into the core concept registry. The
CP-SAT solver is registered only when the `cp` extra is installed.
"""

from __future__ import annotations

from neuro_co.core.concepts import ConceptBank, register_concept_bank
from neuro_co.problems import BASELINE_SOLVERS
from neuro_co.problems.op.concepts import CONCEPTS

# OPEnv.build_features columns: [x, y, prize].
BANK = ConceptBank(problem="op", concepts=CONCEPTS, feature_slices={"prize": 2})
register_concept_bank(BANK)

try:  # optional CP-SAT solver (cp extra)
    from neuro_co.problems.op.cpsat import solve_op

    BASELINE_SOLVERS[("op", "cpsat")] = solve_op
except ImportError:  # pragma: no cover
    pass

__all__ = ["BANK"]
