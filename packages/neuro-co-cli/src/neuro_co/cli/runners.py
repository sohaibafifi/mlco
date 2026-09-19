"""Core-native run drivers for the `neuroco` CLI.

Build environments, models, and algorithms with `neuro_co.core.factory`.
Training saves checkpoints and metrics in the selected output directory.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from neuro_co.core.factory import make_algo, make_env, make_model


def _pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _env_kwargs(problem: str) -> dict[str, Any]:
    """Per-problem extra constructor kwargs beyond `size`."""
    extra: dict[str, dict[str, Any]] = {
        "cvrp": {"capacity": 50.0},
        "cvrptw": {"capacity": 50.0, "horizon": 10.0, "window_width": 2.0},
        "op": {"budget": 3.0},
        "mtsp": {"num_agents": 5},
        "fjsp": {"ops_per_job": 3, "num_machines": 5},
    }
    return extra.get(problem.lower(), {})


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


@dataclass
class TrainArgs:
    problem: str
    algo: str = "pomo"
    backbone: str = "am"
    backend: str = "torch"
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
    precision: str = "fp32"
    out_dir: str = "outputs/run"
    device: str = "cpu"


def train_run(args: TrainArgs) -> dict[str, Any]:
    """Train a policy and save checkpoints and run metrics."""
    from .training import TrainingConfig, train

    shared = args.problem in {"tsp", "cvrp"} and (args.algo, args.backbone) == ("pomo", "am")
    if shared:
        config = TrainingConfig(
            **{field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
        )
        return train(config, args.backend, args.device, Path(args.out_dir))
    if args.backend != "torch":
        raise ValueError("JAX training supports tsp/cvrp with --algo pomo --backbone am")
    if args.precision != "fp32":
        raise ValueError("The training CLI currently supports fp32 precision")
    started = time.monotonic()
    device = _pick_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    env_kwargs = _env_kwargs(args.problem)
    if args.problem in {"cvrp", "cvrptw"}:
        env_kwargs["capacity"] = args.capacity
        env_kwargs["max_demand"] = args.max_demand
    env = make_env(args.problem, size=args.size, **env_kwargs)
    model = make_model(
        env,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    algo_kwargs: dict[str, Any] = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_clip": args.grad_clip,
        "eval_seed": args.eval_seed,
        "precision": args.precision,
    }
    if args.algo in {"pomo", "reinforce"}:
        algo_kwargs.update(optimizer=args.optimizer, weight_decay=args.weight_decay)
    if args.algo == "pomo":
        algo_kwargs["n_starts"] = args.n_starts if args.n_starts is not None else 20
    if args.algo == "ppo":
        algo_kwargs["hidden_dim"] = args.hidden_dim
    algo = make_algo(
        args.algo,
        model,
        env,
        device=device,
        **algo_kwargs,
    )

    rng = torch.Generator(device=device).manual_seed(args.seed)
    best_reward = -float("inf")
    history: list[dict[str, float]] = []
    total_steps = args.epochs * args.steps_per_epoch
    completed_steps = 0
    training_seconds = validation_seconds = first_step_seconds = 0.0
    last_update = time.monotonic()
    setup_seconds = last_update - started

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elif device == "mps":
            torch.mps.synchronize()

    def report(epoch: int, phase: str) -> None:
        nonlocal last_update
        last_update = time.monotonic()
        percent = 100.0 * completed_steps / total_steps if total_steps else 0.0
        remaining = (
            _duration(training_seconds / completed_steps * (total_steps - completed_steps))
            if completed_steps
            else "--:--"
        )
        print(
            f"[{args.problem}/{args.algo}] epoch {epoch + 1}/{args.epochs} | "
            f"steps {completed_steps}/{total_steps} ({percent:.1f}%) | "
            f"elapsed {_duration(last_update - started)} | training ETA {remaining} | {phase}",
            file=sys.stderr,
            flush=True,
        )

    for epoch in range(args.epochs):
        report(epoch, "training")
        for _ in range(args.steps_per_epoch):
            synchronize()
            step_started = time.monotonic()
            algo.train_step(rng)
            synchronize()
            now = time.monotonic()
            training_seconds += now - step_started
            if completed_steps == 0:
                first_step_seconds = now - step_started
            completed_steps += 1
            if completed_steps in (1, total_steps) or now - last_update >= 5.0:
                report(epoch, "training")
        report(epoch, f"validating {args.eval_batch_size} instances")
        synchronize()
        validation_started = time.monotonic()
        ev = algo.eval_step(rng)
        synchronize()
        validation_seconds += time.monotonic() - validation_started
        reward = float(ev.get("eval_reward", float("nan")))
        history.append({"epoch": epoch, **{k: float(v) for k, v in ev.items()}})
        arch = {
            "backbone": args.backbone,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        }
        torch.save({"model": model.state_dict(), "arch": arch, "epoch": epoch}, out / "latest.pt")
        if reward > best_reward:
            best_reward = reward
            torch.save({"model": model.state_dict(), "arch": arch, "epoch": epoch}, out / "best.pt")
        report(epoch, f"validation complete, reward={reward:.4f}")

    payload = {
        "args": asdict(args),
        "device": device,
        "best_reward": best_reward,
        "history": history,
        "timing": {
            "setup_s": setup_seconds,
            "training_s": training_seconds,
            "first_train_step_s": first_step_seconds,
            "warm_train_step_mean_s": (
                (training_seconds - first_step_seconds) / (completed_steps - 1)
                if completed_steps > 1
                else None
            ),
            "validation_s": validation_seconds,
            "total_s": time.monotonic() - started,
        },
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


@dataclass
class EvalArgs:
    problem: str
    ckpt_path: str
    backend: str = "torch"
    algo: str = "reinforce"
    size: int = 50
    eval_batch_size: int = 512
    seed: int = 42
    device: str = "cpu"


def eval_run(args: EvalArgs) -> dict[str, Any]:
    """Load a checkpoint and report eval metrics."""
    from .training import SCHEMA, evaluate_saved_run

    checkpoint = Path(args.ckpt_path)
    config_path = checkpoint.parent / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved.get("schema") == SCHEMA:
            if saved["training"]["problem"] != args.problem:
                raise ValueError(
                    "Requested problem does not match the saved training configuration"
                )
            if saved["backend"] != args.backend:
                raise ValueError(
                    "Requested backend does not match the saved training configuration"
                )
            return evaluate_saved_run(checkpoint, device=args.device)
    if args.backend != "torch":
        raise ValueError("JAX evaluation requires a shared-protocol checkpoint and config.json")
    device = _pick_device(args.device)
    env = make_env(args.problem, size=args.size, **_env_kwargs(args.problem))
    state = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = make_model(env, **state.get("arch", {}))
    model.load_state_dict(state.get("model", state), strict=False)
    algo = make_algo(args.algo, model, env, device=device, eval_batch_size=args.eval_batch_size)
    print(
        f"[{args.problem}/{args.algo}] evaluating {args.eval_batch_size} instances on {device}",
        file=sys.stderr,
        flush=True,
    )
    ev = algo.eval_step(torch.Generator(device=device).manual_seed(args.seed))
    metrics = {k: float(v) for k, v in ev.items()}
    print(json.dumps(metrics, indent=2))
    return metrics
