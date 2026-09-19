"""Per-problem plug-ins for neuro-co.

This package is a namespace container. Each subpackage
(`neuro_co.problems.<name>`) registers itself on import via
`neuro_co.core.concepts.register_concept_bank(...)` plus optional
`BASELINE_SOLVERS[(name, engine)] = solver_fn` entries for any
classical solvers it wraps.

Generic tooling discovers installed plug-ins through the
`neuro_co.problems` entry-point group. See `load_plugins()`.

Public surface:

- `BASELINE_SOLVERS`: `(problem, engine) -> Callable` registry.
- `get_solver(problem, engine)`: resolver with a
  KeyError listing available pairs.
- `load_plugins()` / `force_reload()`: entry-point discovery
  (idempotent; failed plug-ins logged at WARNING).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


# (problem, engine) -> solver callable.
# Signature: `(instance: Any, *, max_runtime: float) -> (solution, cost)`.
# `cost` follows the reward convention (negative = lower-is-better,
# so callers can `abs(cost)` to compare across solvers).
BASELINE_SOLVERS: dict[tuple[str, str], Callable[..., Any]] = {}


def get_solver(problem: str, engine: str) -> Callable[..., Any]:
    """Resolve `(problem, engine) -> solver_fn`.

    Raises a `KeyError` listing the currently registered pairs when
    the requested combination isn't available (typically because the
    matching extra wasn't installed).
    """
    key = (problem.lower(), engine.lower())
    if key not in BASELINE_SOLVERS:
        load_plugins()
    if key not in BASELINE_SOLVERS:
        available = ", ".join(f"{p}/{e}" for p, e in sorted(BASELINE_SOLVERS))
        raise KeyError(
            f"no baseline solver registered for problem={problem!r}, engine={engine!r}. "
            f"Available: {available or '<none; install pyvrp/ortools extras>'}"
        )
    return BASELINE_SOLVERS[key]


_loaded: bool = False


def load_plugins() -> list[str]:
    """Import every distribution declaring a `neuro_co.problems` entry point.

    Subsequent calls do nothing. Failed
    plug-ins are logged at WARNING but do not raise; this keeps the
    workspace usable when an optional extra (pyvrp, ortools) is
    missing. Returns the list of plug-in names actually loaded.
    """
    global _loaded
    if _loaded:
        return []
    loaded: list[str] = []
    try:
        eps = importlib.metadata.entry_points(group="neuro_co.problems")
    except TypeError:  # pragma: no cover - <3.10 fallback
        eps = importlib.metadata.entry_points().get("neuro_co.problems", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            ep.load()
        except Exception as exc:  # pragma: no cover - surface only at debug
            log.warning("failed to load neuro_co.problems plug-in %r: %s", ep.name, exc)
            continue
        loaded.append(ep.name)
    _loaded = True
    return loaded


def force_reload() -> list[str]:
    """Re-run plug-in discovery. Mainly for tests adding new entry points."""
    global _loaded
    _loaded = False
    return load_plugins()


def _import_builtin_problems() -> None:
    """Fallback for editable checkouts without entry-point installation."""
    for name in ("vrptw", "jssp"):
        importlib.import_module(f"neuro_co.problems.{name}")


def get_bank(name: str) -> Any:
    """Resolve `name` → `ConceptBank`. Triggers plug-in discovery if needed."""
    from neuro_co.attr import concept_registry

    load_plugins()
    return concept_registry.get(name.lower())


__all__ = [
    "BASELINE_SOLVERS",
    "force_reload",
    "get_bank",
    "get_solver",
    "load_plugins",
]
