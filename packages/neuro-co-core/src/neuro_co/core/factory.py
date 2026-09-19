"""Build environments, models, and algorithms from names and keyword arguments."""

from __future__ import annotations

from typing import Any

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.ppo import PPO, PPOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.env_registry import ENV_BUILDERS, available_envs, make_env, register_env
from neuro_co.core.models import AttentionModel, GNNModel

_ALGOS: dict[str, tuple[Any, Any]] = {
    "reinforce": (REINFORCE, REINFORCEConfig),
    "pomo": (POMO, POMOConfig),
    "ppo": (PPO, PPOConfig),
}


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


__all__ = ["ENV_BUILDERS", "available_envs", "make_algo", "make_env", "make_model", "register_env"]
