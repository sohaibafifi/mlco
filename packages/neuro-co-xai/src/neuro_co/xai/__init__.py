"""Compatibility imports for attribution, probes, and constraint explanations.

New code can import directly from ``neuro_co.attr``, ``neuro_co.probe``,
and ``neuro_co.cax``.
"""

from __future__ import annotations

import sys
from typing import Any

from neuro_co.attr import *  # noqa: F403  (re-export attr public API)
from neuro_co.probe import *  # noqa: F403  (re-export probe public API)

__version__ = "0.2.0"

# Submodules importable as `neuro_co.xai.<name>` for back-compat paths.
_SUBMODULES = {
    "attribution": "neuro_co.attr.attribution",
    "concept": "neuro_co.attr.concept",
    "faithfulness": "neuro_co.attr.faithfulness",
    "stability": "neuro_co.attr.stability",
    "probes": "neuro_co.probe.probes",
    "discovered": "neuro_co.probe.discovered",
    "viz": "neuro_co.probe.viz",
    "report": "neuro_co.probe.report",
    "cax": "neuro_co.cax",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy resolution
    import importlib

    if name in _SUBMODULES:
        mod = importlib.import_module(_SUBMODULES[name])
        sys.modules[f"neuro_co.xai.{name}"] = mod
        return mod
    # Lazy public symbols that live behind optional extras.
    if name == "explain_policy":
        from neuro_co.probe import explain_policy

        return explain_policy
    raise AttributeError(f"module 'neuro_co.xai' has no attribute {name!r}")
