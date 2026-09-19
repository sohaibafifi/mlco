"""JAX POMO adapter for the shared training protocol."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from neuro_co.core.env_registry import make_env
from neuro_co.core.jax_backend.am import JaxAttentionModel
from neuro_co.core.jax_backend.optim import AdamConfig, init_adam
from neuro_co.core.jax_backend.pomo import JaxPOMO, POMOTrainState
from neuro_co.core.jax_backend.train import Config, load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from .training import TrainingConfig


def _device(name: str) -> Any:
    platform, _, index_text = name.partition(":")
    if platform not in {"cpu", "cuda"}:
        raise ValueError("JAX training requires --device cpu or cuda")
    index = int(index_text) if index_text else 0
    if index < 0:
        raise ValueError("device index must be non-negative")
    try:
        devices = jax.devices("gpu" if platform == "cuda" else "cpu")
    except RuntimeError as error:
        raise RuntimeError(f"requested JAX device {name!r} is unavailable") from error
    if index >= len(devices):
        raise RuntimeError(f"requested JAX device {name!r} is unavailable")
    return devices[index]


class JaxTrainer:
    checkpoint_suffix = ".npz"

    def __init__(self, config: TrainingConfig, weights: dict[str, np.ndarray], device: str) -> None:
        self.config = config
        self.device = _device(device)
        self.info = {
            "backend": "jax",
            "device": str(self.device),
            "platform": "cuda" if self.device.platform == "gpu" else self.device.platform,
            "framework_version": jax.__version__,
        }
        environment: dict[str, Any] = {"size": config.size}
        if config.problem == "cvrp":
            environment.update(capacity=config.capacity, max_demand=config.max_demand)
        self.env = make_env(config.problem, backend="jax", **environment)
        model = JaxAttentionModel(
            in_dim=self.env.encoder_in_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=0.0,
            precision="fp32",
        )
        self.algorithm = JaxPOMO(
            model=model,
            env=self.env,
            n_starts=cast(int, config.n_starts),
            optimizer=AdamConfig(
                learning_rate=config.lr,
                weight_decay=config.weight_decay,
                decoupled_weight_decay=config.optimizer == "adamw",
                grad_clip=config.grad_clip,
            ),
        )
        with jax.default_device(self.device):
            params = model.from_torch_state_dict(weights)
            self.state = POMOTrainState(params=params, optimizer_state=init_adam(params))
            self.key = jax.random.key(config.seed)
        self._train_step = jax.jit(self.algorithm.train_step)
        self._evaluate = jax.jit(self.algorithm.greedy_rollout)
        self.synchronize()

    def _problem(self, batch: dict[str, np.ndarray]) -> Any:
        coords = jnp.asarray(batch["coords"], dtype=jnp.float32)
        if self.config.problem == "cvrp":
            demand = jnp.asarray(batch["demand"], dtype=jnp.float32)
            return self.env.reset_from_data(coords, demand)
        return self.env.reset_from_coords(coords)

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        with jax.default_device(self.device):
            problem = self._problem(batch)
            self.key, step_key = jax.random.split(self.key)
            self.state, metrics = self._train_step(self.state, problem, step_key)
        jax.block_until_ready((self.state, metrics))
        return {name: float(value) for name, value in metrics._asdict().items()}

    def evaluate(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        with jax.default_device(self.device):
            reward = self._evaluate(self.state.params, self._problem(batch))
        return -np.asarray(jax.block_until_ready(reward))

    def save(self, path: Path, step: int) -> None:
        config = self.config
        legacy_config = Config(
            problem=cast(Literal["cvrp", "tsp"], config.problem),
            steps=config.epochs * config.steps_per_epoch,
            size=config.size,
            batch_size=config.batch_size,
            n_starts=cast(int, config.n_starts),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
            grad_clip=config.grad_clip,
            precision="fp32",
            seed=config.seed,
            eval_batch_size=config.eval_batch_size,
        )
        save_checkpoint(Path(path), self.state, self.key, step, legacy_config)

    def load(self, path: Path) -> int:
        with jax.default_device(self.device):
            self.state, self.key, step, _ = load_checkpoint(Path(path), self.state)
        self.synchronize()
        return step

    def synchronize(self) -> None:
        jax.block_until_ready((self.state, self.key))
