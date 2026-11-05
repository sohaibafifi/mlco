"""Core-native run drivers for the `neuroco` CLI.

Build environments, models, and algorithms with `neuro_co.core.factory`.
Training saves checkpoints and metrics in the selected output directory.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from neuro_co.core.checkpoint import (
    atomic_save,
    capture_rng_state,
    restore_rng_state,
    restore_training_state,
    training_state,
    validate_rng_state,
    validate_training_state,
)
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
    resume: str | None = None
    checkpoint_every_steps: int | None = None


def train_run(args: TrainArgs) -> dict[str, Any]:
    """Train a policy and save checkpoints and run metrics."""
    from .training import TrainingConfig, train

    for name in ("epochs", "steps_per_epoch", "batch_size", "eval_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if min(args.seed, args.eval_seed, args.test_seed) < 0:
        raise ValueError("Seeds must be nonnegative")
    shared = args.problem in {"tsp", "cvrp"} and (args.algo, args.backbone) == ("pomo", "am")
    if shared:
        config = TrainingConfig(
            **{field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
        )
        return train(
            config,
            args.backend,
            args.device,
            Path(args.out_dir),
            resume_from=args.resume,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
    if args.checkpoint_every_steps is not None and (
        type(args.checkpoint_every_steps) is not int or args.checkpoint_every_steps <= 0
    ):
        raise ValueError("checkpoint_every_steps must be positive or None")
    if args.backend != "torch":
        raise ValueError("JAX training supports tsp/cvrp with --algo pomo --backbone am")
    if args.precision != "fp32":
        raise ValueError("The training CLI currently supports fp32 precision")
    started = time.monotonic()
    device = _pick_device(args.device)
    from .artifacts import atomic_write_json, write_run
    from .training import _copy_checkpoint, _resume_path

    out = Path(args.out_dir)
    run_config = {
        k: v
        for k, v in asdict(args).items()
        if k not in {"resume", "out_dir", "checkpoint_every_steps"}
    }
    resume_path = None
    if args.resume is not None:
        resume_path = _resume_path(out, args.resume)
        saved_config = json.loads((out / "config.json").read_text())
        if (
            saved_config.get("schema") != "neuro-co-cli-training/v1"
            or saved_config.get("device") != device
            or {k: v for k, v in saved_config.get("training", {}).items() if k != "epochs"}
            != {k: v for k, v in run_config.items() if k != "epochs"}
        ):
            raise ValueError("Resume configuration or device differs from the saved run")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; select a new output directory or --resume")

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
    best_checkpoint = None
    best_step = 0
    history: list[dict[str, float]] = []
    total_steps = args.epochs * args.steps_per_epoch
    completed_steps = 0
    training_seconds = validation_seconds = first_step_seconds = previous_elapsed = 0.0
    last_metrics: dict[str, float] = {}
    previous_setup = 0.0
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        progress = saved.get("run_state")
        if not isinstance(progress, dict) or progress.get("version") != 1:
            raise ValueError("Checkpoint lacks resumable CLI progress; start a new run")
        if progress["step"] > total_steps:
            raise ValueError("Total update target precedes the saved checkpoint")
        validate_training_state(algo, saved["training_state"])
        validate_rng_state(saved["global_rng_state"], device)
        torch.Generator(device=device).set_state(saved["rng_state"].cpu())
        algo.load_state_dict(saved["algo_state"])
        restore_training_state(algo, saved["training_state"])
        restore_rng_state(saved["global_rng_state"], device)
        rng.set_state(saved["rng_state"].cpu())
        completed_steps = progress["step"]
        history = list(progress["history"])
        best_reward = (
            progress["best_reward"] if progress["best_reward"] is not None else -float("inf")
        )
        training_seconds = progress["training_s"]
        validation_seconds = progress["validation_s"]
        first_step_seconds = progress["first_step_s"]
        previous_elapsed = progress["elapsed_s"]
        last_metrics = progress["last_metrics"]
        previous_setup = progress.get("setup_s", 0.0)
        best_checkpoint = progress.get("best_checkpoint")
        best_step = progress.get("best_step", 0)
        if history and (best_checkpoint is None or not (out / best_checkpoint).is_file()):
            raise ValueError("Best checkpoint is missing from the saved run")
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        out / "config.json",
        {
            "schema": "neuro-co-cli-training/v1",
            "backend": "torch",
            "device": device,
            "training": run_config,
        },
    )
    last_update = time.monotonic()
    setup_seconds = previous_setup + last_update - started
    arch = {
        "backbone": args.backbone,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elif device == "mps":
            torch.mps.synchronize()

    def manifest(status: str, error: str | None = None) -> None:
        elapsed = previous_elapsed + time.monotonic() - started
        values = {
            "step": completed_steps,
            "training_s": training_seconds,
            "validation_s": validation_seconds,
            "setup_s": setup_seconds,
            "first_train_step_s": first_step_seconds,
            "warm_train_step_mean_s": (training_seconds - first_step_seconds)
            / (completed_steps - 1)
            if completed_steps > 1
            else None,
            "duration_s": elapsed,
            "total_s": elapsed,
        }
        if best_reward != -float("inf"):
            values["best_reward"] = best_reward
            values["best_step"] = best_step
        write_run(
            out,
            kind="train",
            problem=args.problem,
            seed=args.seed,
            backend="torch",
            config=run_config,
            metrics=values,
            artifacts={
                "config": "config.json",
                "metrics": "metrics.json",
                "latest": "latest.pt",
                "best": "best.pt",
            },
            status=status,
            error=error,
        )

    def save_latest() -> None:
        atomic_save(
            {
                "checkpoint_version": 1,
                "model": model.state_dict(),
                "arch": arch,
                "algo_state": algo.state_dict(),
                "step": completed_steps,
                "epoch": max(0, (completed_steps - 1) // args.steps_per_epoch),
                "training_state": training_state(algo),
                "global_rng_state": capture_rng_state(device),
                "rng_state": rng.get_state().cpu(),
                "run_state": {
                    "version": 1,
                    "step": completed_steps,
                    "history": history,
                    "best_reward": best_reward if best_reward != -float("inf") else None,
                    "training_s": training_seconds,
                    "validation_s": validation_seconds,
                    "setup_s": setup_seconds,
                    "first_step_s": first_step_seconds,
                    "last_metrics": last_metrics,
                    "best_checkpoint": best_checkpoint,
                    "best_step": best_step,
                    "elapsed_s": previous_elapsed + time.monotonic() - started,
                },
            },
            out / "latest.pt",
        )

    def report(phase: str) -> None:
        nonlocal last_update
        last_update = time.monotonic()
        percent = 100.0 * completed_steps / total_steps if total_steps else 0.0
        remaining = (
            _duration(training_seconds / completed_steps * (total_steps - completed_steps))
            if completed_steps
            else "--:--"
        )
        print(
            f"[{args.problem}/{args.algo}] steps {completed_steps}/{total_steps} ({percent:.1f}%) | "
            f"elapsed {_duration(last_update - started)} | training ETA {remaining} | {phase}",
            file=sys.stderr,
            flush=True,
        )

    manifest("running")
    try:
        if completed_steps == 0:
            save_latest()
        while True:
            validated = history[-1]["step"] if history else 0
            if (
                completed_steps > 0
                and completed_steps % args.steps_per_epoch == 0
                and validated < completed_steps
            ):
                report(f"validating {args.eval_batch_size} instances")
                synchronize()
                validation_started = time.monotonic()
                ev = algo.eval_step(rng)
                synchronize()
                validation_seconds += time.monotonic() - validation_started
                reward = float(ev.get("eval_reward", float("nan")))
                if not all(math.isfinite(float(v)) for v in ev.values()):
                    raise FloatingPointError("Non-finite validation metrics")
                epoch = completed_steps // args.steps_per_epoch - 1
                history.append(
                    {
                        "epoch": epoch,
                        "step": completed_steps,
                        **{k: float(v) for k, v in ev.items()},
                    }
                )
                previous_best = best_checkpoint
                if reward > best_reward:
                    best_reward = reward
                    best_step = completed_steps
                    best_checkpoint = f"best-step-{completed_steps}.pt"
                    atomic_save(
                        {
                            "model": model.state_dict(),
                            "arch": arch,
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        out / best_checkpoint,
                    )
                    _copy_checkpoint(out / best_checkpoint, out / "best.pt")
                save_latest()
                if previous_best is not None and previous_best != best_checkpoint:
                    (out / previous_best).unlink(missing_ok=True)
                continue
            if completed_steps >= total_steps:
                break
            synchronize()
            step_started = time.monotonic()
            last_metrics = algo.train_step(rng)
            synchronize()
            now = time.monotonic()
            if not all(math.isfinite(float(v)) for v in last_metrics.values()):
                raise FloatingPointError("Non-finite training metrics")
            training_seconds += now - step_started
            if completed_steps == 0:
                first_step_seconds = now - step_started
            completed_steps += 1
            if (
                args.checkpoint_every_steps is not None
                and completed_steps % args.checkpoint_every_steps == 0
            ):
                save_latest()
            if completed_steps in (1, total_steps) or now - last_update >= 5.0:
                report("training")
        if best_checkpoint is None:
            raise RuntimeError("No validated checkpoint was selected")
        _copy_checkpoint(out / best_checkpoint, out / "best.pt")
        payload = {
            "args": asdict(args),
            "device": device,
            "best_reward": best_reward,
            "history": history,
            "timing": {
                "setup_s": setup_seconds,
                "training_s": training_seconds,
                "first_train_step_s": first_step_seconds,
                "warm_train_step_mean_s": (training_seconds - first_step_seconds)
                / (completed_steps - 1)
                if completed_steps > 1
                else None,
                "validation_s": validation_seconds,
                "total_s": previous_elapsed + time.monotonic() - started,
            },
        }
        atomic_write_json(out / "metrics.json", payload)
        manifest("complete")
        return payload
    except BaseException as error:
        manifest("failed", f"{type(error).__name__}: {error}")
        raise


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
    from .artifacts import atomic_write_json, write_run

    evaluation_dir = checkpoint.parent / "evaluations" / checkpoint.stem
    atomic_write_json(evaluation_dir / "evaluation.json", metrics)
    write_run(
        evaluation_dir,
        kind="eval",
        problem=args.problem,
        seed=args.seed,
        backend="torch",
        config=asdict(args),
        metrics=metrics,
        artifacts={"evaluation": "evaluation.json", "checkpoint": checkpoint.resolve()},
    )
    print(json.dumps(metrics, indent=2))
    return metrics
