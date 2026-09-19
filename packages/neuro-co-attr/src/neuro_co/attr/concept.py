"""Concept-bank contract + registry.

Re-exported from `neuro_co.core.concepts`. The contract lives in core so
every analysis package (attr, probe, cax) shares one registry without a
cross-package dependency cycle. This module is kept for back-compatible
imports (`from neuro_co.attr.concept import ConceptFn`).
"""

from __future__ import annotations

from neuro_co.core.concepts import (
    ConceptBank,
    ConceptFn,
    InstanceRegistry,
    concept_registry,
    infer_problem_name,
    register_concept_bank,
)

__all__ = [
    "ConceptBank",
    "ConceptFn",
    "InstanceRegistry",
    "concept_registry",
    "infer_problem_name",
    "register_concept_bank",
]
