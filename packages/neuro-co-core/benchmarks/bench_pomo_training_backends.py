"""Quick paired throughput benchmark for Torch and JAX AM/POMO training.

The parent process creates one fixed CVRP batch and one Torch-initialized AM
state. Each backend then runs in a fresh child process. The result compares the
current Torch eager training path with the current whole-step JAX JIT path.

This is a throughput diagnostic. It does not compare convergence because the
two frameworks use different random-number generators for sampled actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "neuro-co-paired-training-benchmark/v1"
RESULT_PREFIX = "NEURO_CO_BACKEND_RESULT="


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    size: int = 50
    capacity: float = 40.0
    max_demand: int = 9
    batch_size: int = 32
    n_starts: int = 50
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    precision: str = "bf16"
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    grad_clip: float = 1.0
    seed: int = 27_211
    warmup_steps: int = 3
    measure_blocks: int = 3
    steps_per_block: int = 5


def _validate(config: BenchmarkConfig) -> None:
    integer_fields = {
        "size": config.size,
        "batch_size": config.batch_size,
        "n_starts": config.n_starts,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "warmup_steps": config.warmup_steps,
        "measure_blocks": config.measure_blocks,
        "steps_per_block": config.steps_per_block,
    }
    for name, value in integer_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.max_demand <= 0 or config.capacity < config.max_demand:
        raise ValueError("capacity must be at least max_demand > 0")
    if config.hidden_dim % config.num_heads:
        raise ValueError("num_heads must divide hidden_dim")
    if config.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")
    if config.grad_clip < 0.0:
        raise ValueError("grad_clip must be non-negative")


def _sha256_arrays(arrays: dict[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        metadata = json.dumps(
            {"dtype": str(value.dtype), "name": name, "shape": list(value.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _prepare_artifacts(root: Path, config: BenchmarkConfig) -> dict[str, Any]:
    import numpy as np
    import torch

    from neuro_co.core.models import AttentionModel

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    coords = rng.random((config.batch_size, config.size + 1, 2), dtype=np.float32)
    customer_demand = rng.integers(
        1,
        config.max_demand + 1,
        size=(config.batch_size, config.size),
        dtype=np.int32,
    ).astype(np.float32)
    demand = np.concatenate(
        (np.zeros((config.batch_size, 1), dtype=np.float32), customer_demand),
        axis=1,
    )
    batch = {"coords": coords, "demand": demand}
    savez = cast(Any, np.savez)
    savez(root / "batch.npz", **batch)

    torch.manual_seed(config.seed)
    model = AttentionModel(
        in_dim=3,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=0.0,
    )
    weights = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    savez(root / "weights.npz", **weights)
    return {
        "batch_sha256": _sha256_arrays(batch),
        "weights_sha256": _sha256_arrays(weights),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _block_statistics(
    durations_s: list[float], config: BenchmarkConfig
) -> dict[str, float | list[float]]:
    per_step = [duration / config.steps_per_block for duration in durations_s]
    median = statistics.median(per_step)
    return {
        "block_duration_s": durations_s,
        "step_s_by_block": per_step,
        "median_step_s": median,
        "mean_step_s": statistics.fmean(per_step),
        "min_step_s": min(per_step),
        "max_step_s": max(per_step),
        "instances_per_s": config.batch_size / median,
        "pomo_rollouts_per_s": config.batch_size * config.n_starts / median,
    }


def _torch_worker(
    artifact_root: Path, config: BenchmarkConfig, *, allow_cpu: bool
) -> dict[str, Any]:
    import numpy as np
    import torch

    from neuro_co.core.algos.pomo import POMO, POMOConfig
    from neuro_co.core.envs.cvrp import CVRPEnv, CVRPState
    from neuro_co.core.models import AttentionModel

    if not allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.ones(1, device=device).sum().item()

    batch = _load_npz(artifact_root / "batch.npz")
    weights = _load_npz(artifact_root / "weights.npz")
    coords = torch.as_tensor(batch["coords"], device=device)
    demand = torch.as_tensor(batch["demand"], device=device)

    class FixedCVRPEnv(CVRPEnv):
        def __init__(self) -> None:
            super().__init__(
                size=config.size,
                capacity=config.capacity,
                max_demand=config.max_demand,
            )
            batch_size = coords.shape[0]
            self._batch_size = batch_size
            self._device = coords.device
            state_type = cast(Any, CVRPState)
            self.initial_state: CVRPState = state_type(
                coords=coords,
                demand=demand,
                visited=torch.zeros((batch_size, config.size + 1), dtype=torch.bool, device=device),
                current=torch.zeros(batch_size, dtype=torch.long, device=device),
                remaining_capacity=torch.full((batch_size,), config.capacity, device=device),
                tour_length=torch.zeros(batch_size, device=device),
                step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            )

        def reset(
            self,
            batch_size: int,
            *,
            generator: torch.Generator | None = None,
            device: torch.device | str = "cpu",
        ) -> CVRPState:
            del generator
            requested_device = torch.device(device)
            if batch_size != self._batch_size:
                raise ValueError("fixed benchmark batch size changed")
            if requested_device != self._device:
                raise ValueError("fixed benchmark device changed")
            return self.initial_state

    def build() -> POMO:
        model = AttentionModel(
            in_dim=3,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=0.0,
        )
        model.load_state_dict(
            {name: torch.from_numpy(np.asarray(value)) for name, value in weights.items()}
        )
        return POMO(
            model=model,
            env=FixedCVRPEnv(),
            cfg=POMOConfig(
                batch_size=config.batch_size,
                n_starts=config.n_starts,
                lr=config.learning_rate,
                optimizer="adam",
                weight_decay=config.weight_decay,
                grad_clip=config.grad_clip,
                precision=config.precision,  # type: ignore[arg-type]
                eval_batch_size=1,
            ),
            device=device,
        )

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    algorithm = build()
    warmup_generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    synchronize()
    started = time.perf_counter()
    algorithm.train_step(warmup_generator)
    synchronize()
    first_execution_s = time.perf_counter() - started
    for _ in range(config.warmup_steps):
        algorithm.train_step(warmup_generator)
    synchronize()

    # Adam allocates its moment buffers lazily on the first step. Keep those
    # allocations, then restore the initial model and zero optimizer state so
    # the timed region starts from the same weights as JAX without charging
    # Torch for one-time optimizer setup.
    algorithm.model.load_state_dict(
        {name: torch.from_numpy(np.asarray(value)) for name, value in weights.items()}
    )
    for optimizer_state in algorithm.opt.state.values():
        for value in optimizer_state.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
    algorithm.opt.zero_grad(set_to_none=True)
    algorithm._step = 0
    generator = torch.Generator(device=device).manual_seed(config.seed + 2)
    durations: list[float] = []
    metrics: dict[str, float] = {}
    for _ in range(config.measure_blocks):
        synchronize()
        started = time.perf_counter()
        for _ in range(config.steps_per_block):
            metrics = algorithm.train_step(generator)
        synchronize()
        durations.append(time.perf_counter() - started)

    finite = all(math.isfinite(float(value)) for value in metrics.values())
    if not finite:
        raise RuntimeError("Torch produced non-finite metrics")
    return {
        "backend": "torch_eager",
        "framework_version": torch.__version__,
        "device_platform": device.type,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "compile_s": 0.0,
        "first_execution_s": first_execution_s,
        "timing": _block_statistics(durations, config),
        "last_metrics": {name: float(value) for name, value in metrics.items()},
        "optimizer_steps": config.measure_blocks * config.steps_per_block,
        "metrics_finite": finite,
    }


def _jax_worker(artifact_root: Path, config: BenchmarkConfig, *, allow_cpu: bool) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from neuro_co.core.envs.jax_backend import JaxCVRPEnv
    from neuro_co.core.jax_backend import (
        AdamConfig,
        JaxAttentionModel,
        JaxPOMO,
        POMOTrainState,
        init_adam,
    )

    backend = jax.default_backend()
    if not allow_cpu and backend != "gpu":
        raise RuntimeError(f"JAX GPU backend is unavailable: {backend}")
    batch = _load_npz(artifact_root / "batch.npz")
    weights = _load_npz(artifact_root / "weights.npz")
    env = JaxCVRPEnv(
        size=config.size,
        capacity=config.capacity,
        max_demand=config.max_demand,
    )
    model = JaxAttentionModel(
        in_dim=3,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=0.0,
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

    problem = env.reset_from_data(
        jnp.asarray(np.asarray(batch["coords"])),
        jnp.asarray(np.asarray(batch["demand"])),
    )
    params = model.from_torch_state_dict(weights)
    state = POMOTrainState(params=params, optimizer_state=init_adam(params))
    measurement_state = jax.tree.map(
        lambda value: jnp.array(value, copy=True),
        state,
    )
    jax.block_until_ready((state, measurement_state, problem))
    step = jax.jit(
        lambda train_state, problem_state, key: algorithm.train_step(
            train_state,
            problem_state,
            key,
        ),
        donate_argnums=(0,),
    )
    compile_key = jax.random.key(config.seed + 1)
    started = time.perf_counter()
    compiled = step.lower(state, problem, compile_key).compile()
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    state, metrics = compiled(state, problem, compile_key)
    jax.block_until_ready((state, metrics))
    first_execution_s = time.perf_counter() - started
    for index in range(config.warmup_steps):
        state, metrics = compiled(
            state,
            problem,
            jax.random.fold_in(compile_key, index + 1),
        )
    jax.block_until_ready((state, metrics))

    state = measurement_state
    key_count = config.measure_blocks * config.steps_per_block
    keys = tuple(
        jax.random.fold_in(jax.random.key(config.seed + 2), index) for index in range(key_count)
    )
    jax.block_until_ready(keys)
    durations: list[float] = []
    key_index = 0
    for _ in range(config.measure_blocks):
        started = time.perf_counter()
        for _ in range(config.steps_per_block):
            state, metrics = compiled(state, problem, keys[key_index])
            key_index += 1
        jax.block_until_ready((state, metrics))
        durations.append(time.perf_counter() - started)

    metric_values = {name: float(getattr(metrics, name)) for name in metrics._fields}
    finite = all(math.isfinite(value) for value in metric_values.values())
    if not finite:
        raise RuntimeError("JAX produced non-finite metrics")
    device = jax.devices()[0]
    return {
        "backend": "jax_jit",
        "framework_version": jax.__version__,
        "device_platform": backend,
        "device": str(device),
        "compile_s": compile_s,
        "first_execution_s": first_execution_s,
        "timing": _block_statistics(durations, config),
        "last_metrics": metric_values,
        "optimizer_steps": config.measure_blocks * config.steps_per_block,
        "metrics_finite": finite,
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _config_arguments(config: BenchmarkConfig) -> list[str]:
    arguments: list[str] = []
    for name, value in asdict(config).items():
        arguments.extend((f"--{name.replace('_', '-')}", str(value)))
    return arguments


def _run_child(
    backend: str,
    artifact_root: Path,
    config: BenchmarkConfig,
    *,
    allow_cpu: bool,
    gpu_index: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        backend,
        "--artifact-root",
        str(artifact_root),
        *_config_arguments(config),
    ]
    if allow_cpu:
        command.append("--allow-cpu")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(f"{backend} worker did not emit a result")


def _paired_result(
    config: BenchmarkConfig,
    artifacts: dict[str, Any],
    torch_result: dict[str, Any],
    jax_result: dict[str, Any],
    *,
    gpu_index: int,
) -> dict[str, Any]:
    torch_platform = str(torch_result["device_platform"])
    jax_platform = str(jax_result["device_platform"])
    normalized_platforms = {
        "torch": "gpu" if torch_platform == "cuda" else torch_platform,
        "jax": jax_platform,
    }
    if normalized_platforms["torch"] != normalized_platforms["jax"]:
        raise RuntimeError(
            f"backend device platforms differ: Torch={torch_platform}, JAX={jax_platform}"
        )
    torch_step = float(torch_result["timing"]["median_step_s"])
    jax_step = float(jax_result["timing"]["median_step_s"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(config),
        "pairing": {
            **artifacts,
            "same_initial_weights": True,
            "same_fixed_problem_batch": True,
            "same_action_trajectories": False,
            "normalized_device_platform": normalized_platforms["torch"],
            "requested_gpu_index": gpu_index,
        },
        "scope": {
            "comparison": "current_torch_eager_vs_current_jax_jit",
            "throughput_diagnostic": True,
            "convergence_comparison": False,
            "energy_evidence": False,
            "caveats": [
                "Torch and JAX use different random-number generators for actions.",
                "Torch can stop once all rollouts finish; JAX uses a fixed-length scan.",
                "Torch uses autocast BF16; JAX uses explicit compute casts.",
                "Torch synchronizes during decoding and metric extraction; JAX blocks at each timing-block boundary.",
                "Backends run once in Torch-then-JAX order; the result is a diagnostic, not a backend-wide ranking.",
            ],
        },
        "backends": {"torch": torch_result, "jax": jax_result},
        "comparison": {
            "torch_step_s_over_jax_step_s": torch_step / jax_step,
            "lower_median_backend_in_this_run": ("jax" if jax_step < torch_step else "torch"),
            "median_step_s_difference": abs(torch_step - jax_step),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("torch", "jax"), default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--capacity", type=float, default=40.0)
    parser.add_argument("--max-demand", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-starts", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=27_211)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-blocks", type=int, default=3)
    parser.add_argument("--steps-per-block", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BenchmarkConfig(
        size=args.size,
        capacity=args.capacity,
        max_demand=args.max_demand,
        batch_size=args.batch_size,
        n_starts=args.n_starts,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        precision=args.precision,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        measure_blocks=args.measure_blocks,
        steps_per_block=args.steps_per_block,
    )
    try:
        _validate(config)
        if args.worker is not None:
            if args.artifact_root is None:
                raise ValueError("worker mode requires --artifact-root")
            result = (
                _torch_worker(args.artifact_root, config, allow_cpu=args.allow_cpu)
                if args.worker == "torch"
                else _jax_worker(args.artifact_root, config, allow_cpu=args.allow_cpu)
            )
            print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))
            return 0

        with tempfile.TemporaryDirectory(prefix="neuro-co-paired-training-") as temp:
            artifact_root = Path(temp)
            artifacts = _prepare_artifacts(artifact_root, config)
            torch_result = _run_child(
                "torch",
                artifact_root,
                config,
                allow_cpu=args.allow_cpu,
                gpu_index=args.gpu_index,
            )
            jax_result = _run_child(
                "jax",
                artifact_root,
                config,
                allow_cpu=args.allow_cpu,
                gpu_index=args.gpu_index,
            )
        result = _paired_result(
            config,
            artifacts,
            torch_result,
            jax_result,
            gpu_index=args.gpu_index,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"paired training benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
