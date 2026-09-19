"""Train a JAX attention policy with POMO on generated CVRP or TSP instances.

Run ``python -m neuro_co.core.jax_backend.train --help`` for options.
Checkpoints store parameters, Adam moments, configuration, and the RNG state.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from neuro_co.core.envs.jax_backend import JaxCVRPEnv, JaxTSPEnv

from .am import JaxAttentionModel
from .optim import AdamConfig
from .pomo import JaxPOMO, POMOTrainState


@dataclass(frozen=True)
class Config:
    problem: Literal["cvrp", "tsp"] = "cvrp"
    steps: int = 100
    size: int = 50
    batch_size: int = 256
    n_starts: int = 50
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    grad_clip: float = 1.0
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    seed: int = 0
    log_every: int = 10
    save_every: int = 100
    eval_batch_size: int = 128

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "size",
            "batch_size",
            "n_starts",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "log_every",
            "save_every",
            "eval_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.problem not in ("cvrp", "tsp"):
            raise ValueError("problem must be cvrp or tsp")
        if self.n_starts < 2:
            raise ValueError("POMO needs at least two starts for a nonzero training advantage")
        valid_starts = self.size if self.problem == "cvrp" else self.size - 1
        if self.n_starts > valid_starts:
            raise ValueError(f"n_starts must not exceed {valid_starts} for this problem")


def save_checkpoint(
    path: Path, state: POMOTrainState, key: jax.Array, step: int, config: Config
) -> None:
    """Write a pickle-free NumPy archive, replacing the previous file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"version": 1, "step": step, "config": asdict(config)}
    arrays = {
        f"leaf_{index}": np.asarray(value) for index, value in enumerate(jax.tree.leaves(state))
    }
    arrays["key"] = np.asarray(jax.random.key_data(key))
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    if metadata.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    return metadata


def load_checkpoint(
    path: Path, template: POMOTrainState
) -> tuple[POMOTrainState, jax.Array, int, Config]:
    """Rebuild the training state against the model's expected tree and shapes."""
    metadata = checkpoint_metadata(path)
    expected, structure = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected) + 2:
            raise ValueError("checkpoint array count does not match the model")
        leaves = []
        for index, value in enumerate(expected):
            restored = archive[f"leaf_{index}"]
            if restored.shape != value.shape or restored.dtype != value.dtype:
                raise ValueError(f"checkpoint leaf {index} has an incompatible shape or dtype")
            leaves.append(jnp.asarray(restored))
        key = jax.random.wrap_key_data(jnp.asarray(archive["key"]))
    return (
        jax.tree.unflatten(structure, leaves),
        key,
        int(metadata["step"]),
        Config(**metadata["config"]),
    )


def train(config: Config, output: Path, resume: Path | None = None) -> Path:
    """Train to the requested total step count and return the checkpoint path."""
    output = Path(output)
    checkpoint = output / "checkpoint.npz"
    same_run = resume is not None and Path(resume).resolve().parent == output.resolve()
    artifacts = ("checkpoint.npz", "metrics.jsonl", "config.json", "evaluation.json")
    if not same_run and any((output / name).exists() for name in artifacts):
        raise FileExistsError(
            f"{output} contains a run; use --resume from that directory or a new --output"
        )
    env = JaxCVRPEnv(size=config.size) if config.problem == "cvrp" else JaxTSPEnv(config.size)
    model = JaxAttentionModel(
        in_dim=env.encoder_in_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        precision=config.precision,
    )
    algorithm = JaxPOMO(
        model=model,
        env=env,
        n_starts=config.n_starts,
        optimizer=AdamConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            grad_clip=config.grad_clip,
        ),
    )
    key, init_key = jax.random.split(jax.random.key(config.seed))
    state = algorithm.init(init_key)
    start_step = 0
    if resume is not None:
        state, key, start_step, saved_config = load_checkpoint(Path(resume), state)
        runtime_fields = {"steps", "log_every", "save_every", "eval_batch_size"}
        for field in fields(config):
            if field.name not in runtime_fields and getattr(config, field.name) != getattr(
                saved_config, field.name
            ):
                raise ValueError(f"--resume cannot change {field.name}")
        if config.steps <= start_step:
            raise ValueError(f"steps must exceed the checkpoint step ({start_step})")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    metrics_path = output / "metrics.jsonl"
    if resume is not None and metrics_path.exists():
        records = [
            line
            for line in metrics_path.read_text().splitlines()
            if int(json.loads(line)["step"]) <= start_step
        ]
        metrics_path.write_text("".join(line + "\n" for line in records))

    train_step = jax.jit(
        lambda current, step_key: algorithm.sample_train_step(
            current, step_key, batch_size=config.batch_size
        )
    )
    for step in range(start_step + 1, config.steps + 1):
        key, step_key = jax.random.split(key)
        state, metrics = train_step(state, step_key)
        if step == start_step + 1 or step % config.log_every == 0 or step == config.steps:
            record = {
                "step": step,
                **{name: float(value) for name, value in metrics._asdict().items()},
            }
            if not all(np.isfinite(value) for value in record.values()):
                raise FloatingPointError(f"training produced non-finite metrics at step {step}")
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"step={step:6d} loss={record['loss']:+.6f} "
                f"tour_length={-record['reward_mean']:.6f} "
                f"grad_norm={record['grad_norm']:.6f}",
                flush=True,
            )
        if step % config.save_every == 0 or step == config.steps:
            save_checkpoint(checkpoint, state, key, step, config)

    eval_key = jax.random.fold_in(jax.random.key(config.seed), 1)
    eval_state = env.reset(eval_key, config.eval_batch_size)
    reward = jax.jit(algorithm.greedy_rollout)(state.params, eval_state)
    tour_length = float(-jnp.mean(reward))
    if not np.isfinite(tour_length):
        raise FloatingPointError("greedy evaluation produced a non-finite tour length")
    result = {"step": config.steps, "greedy_tour_length": tour_length}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"greedy_tour_length={result['greedy_tour_length']:.6f} checkpoint={checkpoint}")
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--resume", type=Path)
    initial, _ = preliminary.parse_known_args(argv)
    defaults = (
        Config(**checkpoint_metadata(initial.resume)["config"]) if initial.resume else Config()
    )
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument(
        "--output",
        type=Path,
        default=initial.resume.parent if initial.resume else Path("outputs/jax"),
        help="Run directory containing configuration, metrics, and checkpoint.npz.",
    )
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        kwargs = {"type": type(value), "default": value}
        if field.name == "n_starts" and initial.resume is None:
            kwargs["default"] = None
            kwargs["help"] = "POMO starts (default: up to 50 valid first actions)."
        if field.name == "problem":
            kwargs["choices"] = ("cvrp", "tsp")
        if field.name == "precision":
            kwargs["choices"] = ("fp32", "bf16", "fp16")
        if field.name == "steps":
            kwargs["help"] = "Total training steps, including steps restored with --resume."
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    arguments = vars(parser.parse_args(argv))
    output = arguments.pop("output")
    resume = arguments.pop("resume")
    if arguments["n_starts"] is None:
        valid_starts = arguments["size"] - (arguments["problem"] == "tsp")
        arguments["n_starts"] = min(defaults.n_starts, valid_starts)
    train(Config(**arguments), output, resume)


if __name__ == "__main__":
    main()
