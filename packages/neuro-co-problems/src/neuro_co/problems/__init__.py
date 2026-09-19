"""Problem environments, concept banks, and classical solver adapters.

Environment constructors use the ``neuro_co.envs`` entry-point group.
``load_plugins()`` discovers concept banks and baseline solvers through
``neuro_co.problems``. Solver backends are imported when called.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# (problem, engine) -> solver callable.
# Signature: `(instance: Any, *, max_runtime: float) -> (solution, cost)`.
# `cost` follows the reward convention (negative = lower-is-better,
# so callers can `abs(cost)` to compare across solvers).
BASELINE_SOLVERS: dict[tuple[str, str], Callable[..., Any]] = {}


@dataclass(frozen=True, slots=True)
class _LazySolver:
    """Pickleable adapter reference for local or worker-process execution."""

    module: str
    function: str

    @property
    def __name__(self) -> str:
        return self.function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(importlib.import_module(self.module), self.function)(*args, **kwargs)


def _register_lazy_solver(
    problem: str,
    engine: str,
    module: str,
    function: str,
    *,
    dependency: str | None = None,
    aliases: tuple[str, ...] = (),
) -> Callable[..., Any] | None:
    """Register an adapter without importing its optional solver backend."""
    if dependency is not None and importlib.util.find_spec(dependency) is None:
        return None

    solve = _LazySolver(module, function)
    for name in (problem, *aliases):
        BASELINE_SOLVERS[(name, engine)] = solve
    return solve


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
    from neuro_co.core.concepts import concept_registry

    load_plugins()
    return concept_registry.get(name.lower())


__all__ = [
    "BASELINE_SOLVERS",
    "force_reload",
    "get_bank",
    "get_solver",
    "load_plugins",
]
