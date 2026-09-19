"""Discovery must leave unrequested problem backends unloaded."""

from importlib.metadata import EntryPoint

import pytest

from neuro_co.core import env_registry
from neuro_co.core.env_registry import EnvironmentRegistry


def test_discovery_loads_only_the_requested_constructor(monkeypatch) -> None:
    entries = (
        EntryPoint(name="torch.toy", value="builtins:dict", group="neuro_co.envs"),
        EntryPoint(name="torch.optional", value="missing_backend:Env", group="neuro_co.envs"),
        EntryPoint(name="jax.toy", value="missing_jax_backend:Env", group="neuro_co.envs"),
    )
    monkeypatch.setattr(env_registry.metadata, "entry_points", lambda **kwargs: entries)
    registry = EnvironmentRegistry()

    assert sorted(registry) == ["optional", "toy"]
    assert "OPTIONAL" in registry
    assert registry["TOY"](size=5) == {"size": 5}
    with pytest.raises(ModuleNotFoundError, match="missing_backend"):
        registry["optional"]


def test_custom_environment_without_a_problem_distribution(monkeypatch) -> None:
    monkeypatch.setattr(env_registry.metadata, "entry_points", lambda **kwargs: ())
    monkeypatch.setattr(env_registry, "_registries", {})

    assert env_registry.available_envs() == []
    with pytest.raises(KeyError, match="Install neuro-co-problems"):
        env_registry.make_env("missing")
    env_registry.register_env("Toy", dict)
    assert env_registry.available_envs() == ["toy"]
    assert env_registry.available_envs(backend="jax") == []
    assert env_registry.make_env("TOY", size=5) == {"size": 5}


def test_conflicting_providers_are_reported(monkeypatch) -> None:
    entries = (
        EntryPoint(name="torch.toy", value="builtins:dict", group="neuro_co.envs"),
        EntryPoint(name="torch.toy", value="builtins:list", group="neuro_co.envs"),
    )
    monkeypatch.setattr(env_registry.metadata, "entry_points", lambda **kwargs: entries)
    with pytest.raises(ValueError, match="multiple environment providers"):
        list(EnvironmentRegistry())
