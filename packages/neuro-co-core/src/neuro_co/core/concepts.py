"""Concept labels and registry shared by XAI tools.

A ConceptFn labels each node of a State with 1, 0, or -1 (ignore). Linear probes
and PCA/ICA direction matching use these labels. Problem packages register
ConceptBanks so analysis code can retrieve labels through a common interface."""

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class ConceptFn(Protocol):
    """Per-node binary concept labeller over a `State`.

    Takes a `State` and returns a `[B, N]` long tensor with values in
    `{0, 1, -1}` (`-1` = ignore that position, e.g. depot), or `None` to
    opt out when the state lacks the needed fields.
    """

    def __call__(self, state) -> object: ...


@dataclass(frozen=True)
class ConceptBank:
    """Per-problem bundle consumed by generic XAI tooling.

    Attributes:
        problem: canonical name ("tsp", "cvrp", "cvrptw", ...).
        concepts: named per-node concept extractors.
        feature_slices: optional map name -> column index into
            `env.build_features(state)`, so attribution can report
            per-named-feature scores from the single feature tensor.
    """

    problem: str
    concepts: dict[str, ConceptFn] = field(default_factory=dict)
    feature_slices: dict[str, int] = field(default_factory=dict)


class InstanceRegistry(Generic[T]):
    """Name → instance mapping (values that aren't no-arg classes)."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        if name in self._items:
            raise ValueError(f"{self._kind} {name!r} already registered")
        self._items[name] = value
        return value

    def get(self, name: str) -> T:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"{self._kind} {name!r} not registered. Available: {available}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


concept_registry: InstanceRegistry[ConceptBank] = InstanceRegistry("concept_bank")


def register_concept_bank(bank: ConceptBank) -> ConceptBank:
    """Register `bank` under `bank.problem`. Re-registration disallowed."""
    return concept_registry.register(bank.problem, bank)


def infer_problem_name(target: str, *, aliases: dict[str, str] | None = None) -> str | None:
    """Substring-match `target` against registered concept-bank names."""
    target = target.lower()
    aliases = aliases or {}
    for name in concept_registry.names():
        if name in target:
            return name
        alias = aliases.get(name)
        if alias and alias in target:
            return name
    return None


__all__ = [
    "ConceptBank",
    "ConceptFn",
    "InstanceRegistry",
    "concept_registry",
    "infer_problem_name",
    "register_concept_bank",
]
