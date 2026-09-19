"""Constraint attribution, feasible counterfactuals, and sufficient node subsets.

CP feasibility checks and LP dual estimators require the ``cp`` extra.
"""

from __future__ import annotations

from neuro_co.cax.constraint_map import (
    PROBLEM_CONSTRAINTS,
    aggregate_trace_by_family,
    get_constraints,
)
from neuro_co.cax.cp_minimal_subset import (
    MinimalSubsetReport,
    cp_minimal_subset,
    pac_sample_count,
)
from neuro_co.cax.lambda_attribution import LambdaAttribution, lambda_attribution

__version__ = "0.2.0"

# Load solver and benchmark modules when requested.
_LAZY_CP = {"cp_counterfactual", "CounterfactualReport"}
_LAZY_BENCH = {"benchmark_run", "benchmark_runs", "BenchmarkRow"}
_LAZY_INTERVENTION = {
    "ConstraintInterventionAttribution",
    "constraint_intervention_attribution",
}


def __getattr__(name: str):
    from importlib import import_module

    for module, names in (
        ("cp_counterfactual", _LAZY_CP),
        ("benchmark", _LAZY_BENCH),
        ("constraint_intervention", _LAZY_INTERVENTION),
    ):
        if name in names:
            implementation = import_module(f"neuro_co.cax.{module}")
            globals().update({key: getattr(implementation, key) for key in names})
            return globals()[name]
    raise AttributeError(f"module 'neuro_co.cax' has no attribute {name!r}")


__all__ = [
    "PROBLEM_CONSTRAINTS",
    "ConstraintInterventionAttribution",  # lazy
    "CounterfactualReport",  # lazy
    "LambdaAttribution",
    "MinimalSubsetReport",
    "__version__",
    "aggregate_trace_by_family",
    "benchmark_runs",  # lazy
    "constraint_intervention_attribution",  # lazy
    "cp_counterfactual",  # lazy
    "cp_minimal_subset",
    "get_constraints",
    "lambda_attribution",
    "pac_sample_count",
]
