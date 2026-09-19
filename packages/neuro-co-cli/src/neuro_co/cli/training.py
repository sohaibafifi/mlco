"""A shared training protocol for Torch and JAX attention policies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro-co-training/v1"


@dataclass(frozen=True)
class TrainingConfig:
    problem: str = "tsp"
    size: int = 20
    epochs: int = 10
    steps_per_epoch: int = 100
    batch_size: int = 128
    eval_batch_size: int = 512
    n_starts: int | None = None
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    lr: float = 1e-4
    optimizer: str = "adam"
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    capacity: float = 50.0
    max_demand: int = 9
    seed: int = 42
    eval_seed: int = 12345
    test_seed: int = 54321
    algo: str = "pomo"
    backbone: str = "am"
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.problem not in {"tsp", "cvrp"}:
            raise ValueError("Shared Torch/JAX training supports tsp and cvrp")
        if (self.algo, self.backbone, self.precision) != ("pomo", "am", "fp32"):
            raise ValueError("Shared Torch/JAX training uses POMO, AM, and fp32")
        for name in (
            "epochs",
            "steps_per_epoch",
            "batch_size",
            "eval_batch_size",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "max_demand",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.size < 3 or self.hidden_dim % self.num_heads:
            raise ValueError("size must be at least 3 and num_heads must divide hidden_dim")
        valid_starts = self.size - (self.problem == "tsp")
        if self.n_starts is None:
            object.__setattr__(self, "n_starts", min(20, valid_starts))
        if self.n_starts is None or not 2 <= self.n_starts <= valid_starts:
            raise ValueError(f"n_starts must be between 2 and {valid_starts}")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        if not all(
            math.isfinite(x) for x in (self.lr, self.weight_decay, self.grad_clip, self.capacity)
        ):
            raise ValueError("Optimizer and capacity settings must be finite")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("Invalid optimizer settings")
        if self.capacity < self.max_demand:
            raise ValueError("capacity must be at least max_demand")
        if min(self.seed, self.eval_seed, self.test_seed) < 0:
            raise ValueError("Seeds must be nonnegative")

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch


def batch(config: TrainingConfig, split: str, step: int = 0) -> dict[str, np.ndarray]:
    """Generate the same fresh batch for either backend without storing a dataset."""
    stream, seed = {
        "train": (0, config.seed),
        "validation": (1, config.eval_seed),
        "test": (2, config.test_seed),
    }[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream, step]))
    count = config.batch_size if split == "train" else config.eval_batch_size
    nodes = config.size + (config.problem == "cvrp")
    data = {"coords": rng.random((count, nodes, 2), dtype=np.float32)}
    if config.problem == "cvrp":
        demand = np.zeros((count, nodes), dtype=np.float32)
        demand[:, 1:] = rng.integers(1, config.max_demand + 1, size=(count, config.size))
        data["demand"] = demand
    return data


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def initial_weights(config: TrainingConfig) -> dict[str, np.ndarray]:
    import torch

    from neuro_co.core.models import AttentionModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = AttentionModel(
            in_dim=3 if config.problem == "cvrp" else 2,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=0.0,
        )
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def make_trainer(
    config: TrainingConfig, backend: str, device: str, weights: dict[str, np.ndarray]
) -> Any:
    if device == "auto":
        device = "cpu"
    if backend == "torch":
        from .training_torch import TorchTrainer

        return TorchTrainer(config, weights, device)
    if backend == "jax":
        from .training_jax import JaxTrainer

        return JaxTrainer(config, weights, device)
    raise ValueError("backend must be torch or jax")


def device_info(trainer: Any, requested: str) -> dict:
    return {
        **trainer.info,
        "host": platform.node(),
        "machine": platform.machine(),
        "requested": "cpu" if requested == "auto" else requested,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _progress(
    backend: str, config: TrainingConfig, step: int, started: float, train_s: float, phase: str
) -> None:
    eta = f"{train_s / step * (config.total_steps - step):.0f}s" if step else "?"
    print(
        f"[{backend}/{config.problem}] {step}/{config.total_steps} ({100 * step / config.total_steps:.1f}%) | elapsed {time.perf_counter() - started:.0f}s | training ETA {eta} | {phase}",
        file=sys.stderr,
        flush=True,
    )


def _evaluate(trainer: Any, data: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    trainer.synchronize()
    started = time.perf_counter()
    costs = np.asarray(trainer.evaluate(data), dtype=np.float64)
    trainer.synchronize()
    elapsed = time.perf_counter() - started
    if costs.shape != (len(data["coords"]),) or not np.isfinite(costs).all():
        raise FloatingPointError("Evaluation returned invalid costs")
    return costs, elapsed


def train(config: TrainingConfig, backend: str, device: str, output: Path) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; select a new output directory")
    started = time.perf_counter()
    weights = initial_weights(config)
    trainer = make_trainer(config, backend, device, weights)
    trainer.synchronize()
    setup_s = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    _write(
        output / "config.json",
        {"schema": SCHEMA, "backend": backend, "device": device, "training": asdict(config)},
    )
    validation = batch(config, "validation")
    test = batch(config, "test")
    history = []
    step = 0
    training_s = validation_s = 0.0
    first_step_s = 0.0
    best_cost = float("inf")
    best_step = 0
    train_digest = hashlib.sha256()
    best_path = output / ("best" + trainer.checkpoint_suffix)
    last_path = output / ("latest.pt" if backend == "torch" else "checkpoint.npz")
    last_report = 0.0
    for epoch in range(config.epochs):
        _progress(backend, config, step, started, training_s, f"epoch {epoch + 1}/{config.epochs}")
        for _ in range(config.steps_per_epoch):
            trainer.synchronize()
            step_started = time.perf_counter()
            data = batch(config, "train", step)
            metrics = trainer.train_step(data)
            trainer.synchronize()
            duration = time.perf_counter() - step_started
            training_s += duration
            if step == 0:
                first_step_s = duration
            train_digest.update(array_digest(data).encode())
            step += 1
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError(f"Non-finite training metrics at step {step}")
            now = time.perf_counter()
            if step in (1, config.total_steps) or now - last_report >= 5:
                _progress(backend, config, step, started, training_s, "training")
                last_report = now
        _progress(backend, config, step, started, training_s, "validation")
        costs, duration = _evaluate(trainer, validation)
        validation_s += duration
        cost = float(costs.mean())
        history.append(
            {
                "epoch": epoch,
                "step": step,
                "eval_reward": -cost,
                "eval_tour_length": cost,
                "training_s": training_s,
                **metrics,
            }
        )
        trainer.save(last_path, step)
        if cost < best_cost:
            best_cost, best_step = cost, step
            trainer.save(best_path, step)
        _progress(backend, config, step, started, training_s, f"validation cost={cost:.6f}")
    trainer.load(best_path)
    _progress(backend, config, step, started, training_s, "testing best checkpoint")
    costs, test_first_s = _evaluate(trainer, test)
    warm_costs, test_warm_s = _evaluate(trainer, test)
    if not np.allclose(costs, warm_costs):
        raise RuntimeError("Repeated greedy evaluation changed costs")
    evaluation = {
        "schema": SCHEMA,
        "backend": backend,
        "problem": config.problem,
        "checkpoint": best_path.name,
        "device": device_info(trainer, device),
        "step": best_step,
        "num_instances": config.eval_batch_size,
        "greedy_tour_length": float(costs.mean()),
        "costs": costs.tolist(),
        "test_sha256": array_digest(test),
        "first_call_s": test_first_s,
        "warm_call_s": test_warm_s,
    }
    _write(output / "evaluation.json", evaluation)
    payload = {
        "schema": SCHEMA,
        "backend": backend,
        "args": {**asdict(config), "out_dir": str(output)},
        "device": device_info(trainer, device),
        "best_reward": -best_cost,
        "best_step": best_step,
        "history": history,
        "protocol": {
            "initial_weights_sha256": array_digest(weights),
            "training_data_sha256": train_digest.hexdigest(),
            "validation_sha256": array_digest(validation),
            "test_sha256": array_digest(test),
            "checkpoint_selection": "lowest validation cost",
            "evaluation": "single greedy on separate test instances",
            "sampling": "backend-specific random action sampling",
        },
        "timing": {
            "setup_s": setup_s,
            "training_s": training_s,
            "first_train_step_s": first_step_s,
            "warm_train_step_mean_s": (training_s - first_step_s) / (step - 1)
            if step > 1
            else None,
            "validation_s": validation_s,
            "test_first_call_s": test_first_s,
            "test_warm_call_s": test_warm_s,
            "total_s": time.perf_counter() - started,
        },
    }
    _write(output / "metrics.json", payload)
    _progress(
        backend,
        config,
        step,
        started,
        training_s,
        f"finished | test cost={costs.mean():.6f} | training={training_s:.2f}s",
    )
    return payload


def evaluate_saved_run(checkpoint: Path, device: str = "cpu") -> dict:
    saved = json.loads((checkpoint.parent / "config.json").read_text())
    config = TrainingConfig(**saved["training"])
    trainer = make_trainer(config, saved["backend"], device, initial_weights(config))
    step = trainer.load(checkpoint)
    data = batch(config, "test")
    costs, first_s = _evaluate(trainer, data)
    _, warm_s = _evaluate(trainer, data)
    result = {
        "schema": SCHEMA,
        "backend": saved["backend"],
        "problem": config.problem,
        "checkpoint": checkpoint.name,
        "device": device_info(trainer, device),
        "step": step,
        "num_instances": config.eval_batch_size,
        "greedy_tour_length": float(costs.mean()),
        "costs": costs.tolist(),
        "test_sha256": array_digest(data),
        "first_call_s": first_s,
        "warm_call_s": warm_s,
    }
    filename = (
        "evaluation.json" if checkpoint.stem == "best" else f"evaluation-{checkpoint.stem}.json"
    )
    _write(checkpoint.parent / filename, result)
    print(json.dumps({key: value for key, value in result.items() if key != "costs"}, indent=2))
    return result


def summarize(
    root: Path, problems: list[str] | None = None, backends: list[str] | None = None
) -> dict:
    rows = []
    by_problem: dict[str, list[dict]] = {}
    for path in sorted(root.glob("*/*/metrics.json")):
        run = json.loads(path.read_text())
        if run.get("schema") != SCHEMA:
            continue
        if problems is not None and run["args"]["problem"] not in problems:
            continue
        if backends is not None and run["backend"] not in backends:
            continue
        evaluation = json.loads((path.parent / "evaluation.json").read_text())
        expected_checkpoint = "best.pt" if run["backend"] == "torch" else "best.npz"
        if (
            evaluation["test_sha256"] != run["protocol"]["test_sha256"]
            or evaluation["step"] != run["best_step"]
            or evaluation["checkpoint"] != expected_checkpoint
        ):
            raise ValueError(
                f"Evaluation does not match the selected checkpoint and test set: {path.parent}"
            )
        run["evaluation_device"] = evaluation["device"]
        row = {
            "problem": run["args"]["problem"],
            "backend": run["backend"],
            "test_cost": evaluation["greedy_tour_length"],
            "training_s": run["timing"]["training_s"],
            "first_train_step_s": run["timing"]["first_train_step_s"],
            "warm_train_step_mean_s": run["timing"]["warm_train_step_mean_s"],
            "test_warm_call_s": evaluation["warm_call_s"],
        }
        rows.append(row)
        by_problem.setdefault(row["problem"], []).append(run)
    if not rows:
        raise ValueError(f"No completed shared-protocol runs under {root}")
    matched = {}
    timing_comparable = {}
    for problem, runs in by_problem.items():
        configs = [{k: v for k, v in run["args"].items() if k != "out_dir"} for run in runs]
        matched[problem] = (
            len(runs) == 2
            and {run["backend"] for run in runs} == {"torch", "jax"}
            and configs[0] == configs[1]
            and runs[0]["protocol"] == runs[1]["protocol"]
        )

        def hardware(info: dict) -> tuple:
            return info.get("host"), info.get("machine"), info.get("requested")

        timing_comparable[problem] = (
            matched[problem]
            and hardware(runs[0]["device"]) == hardware(runs[1]["device"])
            and hardware(runs[0]["evaluation_device"]) == hardware(runs[1]["evaluation_device"])
        )
        print(
            f"[compare/{problem}] matching protocol={matched[problem]} | same hardware={timing_comparable[problem]}",
            file=sys.stderr,
        )
    result = {
        "matched_protocol": matched,
        "timing_comparable": timing_comparable,
        "runs": rows,
        "timing_note": "Training includes data generation, transfer, updates, and the first JAX compilation. Warm step time excludes the first update. Hardware is recorded in each metrics.json; action samples can differ across backends.",
    }
    _write(root / "comparison.json", result)
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['problem']:5} {row['backend']:5} | test cost {row['test_cost']:.6f} | train {row['training_s']:.2f}s | warm eval {row['test_warm_call_s']:.4f}s",
            file=sys.stderr,
        )
    return result
