"""Discover environment constructors without importing problem implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping
from importlib import metadata
from typing import Any

EnvBuilder = Callable[..., Any]


class EnvironmentRegistry(MutableMapping[str, EnvBuilder]):
    """Lazy constructors from ``neuro_co.envs`` entries named ``backend.problem``.

    Listing names reads package metadata only. Accessing a constructor imports
    its module. Explicit registrations can replace installed constructors.
    """

    def __init__(self, backend: str = "torch") -> None:
        self.backend = backend.lower()
        self._entries: dict[str, EnvBuilder | metadata.EntryPoint] = {}
        self._discovered = False

    def _discover(self) -> None:
        if self._discovered:
            return
        entries: dict[str, metadata.EntryPoint] = {}
        prefix = f"{self.backend}."
        for entry in metadata.entry_points(group="neuro_co.envs"):
            if not entry.name.startswith(prefix):
                continue
            name = entry.name.removeprefix(prefix).lower()
            if name in entries and entries[name].value != entry.value:
                raise ValueError(f"multiple environment providers for {entry.name!r}")
            entries[name] = entry
        self._entries.update(entries)
        self._discovered = True

    def __getitem__(self, name: str) -> EnvBuilder:
        self._discover()
        name = name.lower()
        entry = self._entries[name]
        if isinstance(entry, metadata.EntryPoint):
            builder = entry.load()
            if not callable(builder):
                raise TypeError(f"environment entry point {entry.name!r} is not callable")
            self._entries[name] = builder
            return builder
        return entry

    def __setitem__(self, name: str, builder: EnvBuilder) -> None:
        if not callable(builder):
            raise TypeError("environment constructor must be callable")
        self._discover()
        self._entries[name.lower()] = builder

    def __delitem__(self, name: str) -> None:
        self._discover()
        del self._entries[name.lower()]

    def __iter__(self) -> Iterator[str]:
        self._discover()
        return iter(self._entries)

    def __len__(self) -> int:
        self._discover()
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        self._discover()
        return isinstance(name, str) and name.lower() in self._entries


ENV_BUILDERS = EnvironmentRegistry()
_registries: dict[str, EnvironmentRegistry] = {"torch": ENV_BUILDERS}


def _registry(backend: str) -> EnvironmentRegistry:
    key = backend.lower()
    if key not in _registries:
        _registries[key] = EnvironmentRegistry(key)
    return _registries[key]


def available_envs(*, backend: str = "torch") -> list[str]:
    """List installed or registered problems without importing their backends."""
    return sorted(_registry(backend))


def register_env(problem: str, builder: EnvBuilder, *, backend: str = "torch") -> None:
    """Register a constructor for this process, replacing an existing entry."""
    _registry(backend)[problem] = builder


def make_env(problem: str, *, backend: str = "torch", **kwargs: Any) -> Any:
    """Build an installed environment, for example ``make_env('tsp', size=20)``.

    Built-in problems are provided by ``neuro-co-problems``. JAX environments
    also require its ``jax`` extra.
    """
    registry = _registry(backend)
    if problem not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise KeyError(
            f"unknown problem {problem!r} for backend {backend!r}. Available: {available}. "
            "Install neuro-co-problems for the built-in environments."
        )
    return registry[problem](**kwargs)


__all__ = ["ENV_BUILDERS", "EnvironmentRegistry", "available_envs", "make_env", "register_env"]
