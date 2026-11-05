"""Serializable results shared by learned policies and classical solvers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from numbers import Integral, Real
from typing import Any

import numpy as np

SOLVE_SCHEMA = "neuro-co-solve/v1"


class SolveStatus(StrEnum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    INVALID = "invalid"
    ERROR = "error"


def _finite(name: str, value: Any, *, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Result JSON mappings must have string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    raise ValueError(f"Result contains a non-JSON or nonfinite value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """An independent check of a returned solution in the input's units."""

    feasible: bool
    objective: float | None
    violations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise ValueError("validation.feasible must be a boolean")
        object.__setattr__(self, "objective", _finite("validation.objective", self.objective))
        object.__setattr__(self, "violations", tuple(self.violations))
        if self.feasible and self.violations:
            raise ValueError("A feasible validation report cannot contain violations")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class SolveResult:
    """One method's outcome, without conflating failure with infeasibility.

    ``objective`` and ``bound`` use the positive problem objective and ``sense``
    states whether it is minimized or maximized. Missing bounds and gaps remain
    null. ``UNKNOWN`` means the solver did not establish feasibility or
    infeasibility; it is never a proof that the problem is infeasible.
    """

    problem: str
    method: str
    status: SolveStatus
    solution: dict[str, Any] | None = None
    objective: float | None = None
    sense: str = "min"
    bound: float | None = None
    gap: float | None = None
    runtime_s: float = 0.0
    feasible: bool | None = None
    validation: ValidationReport | None = None
    solver_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.problem or not self.method:
            raise ValueError("problem and method must be nonempty")
        object.__setattr__(self, "status", SolveStatus(self.status))
        if self.sense not in {"min", "max"}:
            raise ValueError("sense must be 'min' or 'max'")
        for name in ("objective", "bound", "gap", "runtime_s"):
            value = _finite(name, getattr(self, name), nonnegative=name in {"gap", "runtime_s"})
            if name == "runtime_s" and value is None:
                raise ValueError("runtime_s must be a finite nonnegative number")
            object.__setattr__(self, name, value)
        if self.feasible is not None and not isinstance(self.feasible, bool):
            raise ValueError("feasible must be a boolean or None")
        if self.feasible is None:
            if self.status in {SolveStatus.FEASIBLE, SolveStatus.OPTIMAL}:
                object.__setattr__(self, "feasible", True)
            elif self.status in {SolveStatus.INFEASIBLE, SolveStatus.INVALID}:
                object.__setattr__(self, "feasible", False)
        if self.status in {SolveStatus.FEASIBLE, SolveStatus.OPTIMAL} and self.feasible is not True:
            raise ValueError("A feasible or optimal result must be feasible")
        if (
            self.status in {SolveStatus.INFEASIBLE, SolveStatus.INVALID}
            and self.feasible is not False
        ):
            raise ValueError("An infeasible or invalid result must have feasible=False")
        if self.status == SolveStatus.UNKNOWN and (
            self.feasible is not None or self.solution is not None or self.objective is not None
        ):
            raise ValueError("An unknown result cannot assert feasibility or carry an incumbent")
        if self.status == SolveStatus.INFEASIBLE and (
            self.solution is not None or self.objective is not None
        ):
            raise ValueError("An infeasibility result cannot carry an incumbent solution")
        if self.gap is not None and (self.objective is None or self.bound is None):
            raise ValueError("A gap requires an objective and a corresponding bound")
        if (
            self.validation is not None
            and self.feasible is not None
            and self.validation.feasible != self.feasible
        ):
            raise ValueError("Result feasibility disagrees with independent validation")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SOLVE_SCHEMA, **_json_value(asdict(self))}

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False) + "\n"


__all__ = ["SOLVE_SCHEMA", "SolveResult", "SolveStatus", "ValidationReport"]
