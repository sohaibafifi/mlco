"""Build environments, models, and algorithms from names and keyword arguments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.ppo import PPO, PPOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.envs.atsp import ATSPEnv
from neuro_co.core.envs.cvrp import CVRPEnv
from neuro_co.core.envs.cvrptw import CVRPTWEnv
from neuro_co.core.envs.fjsp import FJSPEnv
from neuro_co.core.envs.mtsp import MTSPEnv
from neuro_co.core.envs.op import OPEnv
from neuro_co.core.envs.pdp import PDPEnv
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel, GNNModel

ENV_BUILDERS: dict[str, Callable[..., Any]] = {
    "tsp": TSPEnv,
    "atsp": ATSPEnv,
    "cvrp": CVRPEnv,
    "cvrptw": CVRPTWEnv,
    "op": OPEnv,
    "pdp": PDPEnv,
    "mtsp": MTSPEnv,
    "fjsp": FJSPEnv,
}

_ALGOS: dict[str, tuple[Any, Any]] = {
    "reinforce": (REINFORCE, REINFORCEConfig),
    "pomo": (POMO, POMOConfig),
    "ppo": (PPO, PPOConfig),
}


def make_env(problem: str, **kwargs: Any) -> Any:
    """Construct a core env by problem name (e.g. `make_env("cvrptw", size=50)`)."""
    key = problem.lower()
    if key not in ENV_BUILDERS:
        raise KeyError(f"unknown problem {problem!r}. Available: {sorted(ENV_BUILDERS)}")
    return ENV_BUILDERS[key](**kwargs)


def make_model(
    env: Any,
    *,
    backbone: str = "am",
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
) -> Any:
    """Build an AM, sparse GNN, MatNet, or Mamba policy for the environment.

    GNN and Mamba require the corresponding optional core dependencies.
    """
    kw = {
        "in_dim": env.encoder_in_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }
    key = backbone.lower()
    if key in ("am", "attention", "transformer"):
        return AttentionModel(**kw)
    if key == "matnet":
        from neuro_co.core.models import MatNetModel

        # MatNet is edge/matrix-based: it takes the matrix width, not in_dim.
        return MatNetModel(
            problem_size=env.size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    if key == "gnn":
        return GNNModel(**kw)
    if key == "mamba":
        from neuro_co.core.models import MambaModel

        return MambaModel(**kw)
    raise KeyError(f"unknown backbone {backbone!r}. Use 'am', 'gnn', 'matnet', or 'mamba'.")


def make_algo(name: str, model: Any, env: Any, *, device: str = "cpu", **cfg_kwargs: Any) -> Any:
    """Construct a training algo by name with its config from `cfg_kwargs`."""
    key = name.lower()
    if key not in _ALGOS:
        raise KeyError(f"unknown algo {name!r}. Available: {sorted(_ALGOS)}")
    algo_cls, cfg_cls = _ALGOS[key]
    return algo_cls(model=model, env=env, cfg=cfg_cls(**cfg_kwargs), device=device)


__all__ = ["ENV_BUILDERS", "make_algo", "make_env", "make_model"]
