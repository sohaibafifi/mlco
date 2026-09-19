"""Core-native run drivers for the `neuroco` CLI.

Build environments, models, and algorithms with `neuro_co.core.factory`.
Training saves checkpoints and metrics in the selected output directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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


@dataclass
class TrainArgs:
    problem: str
    algo: str = "reinforce"
    backbone: str = "am"
    size: int = 50
    epochs: int = 10
    steps_per_epoch: int = 100
    batch_size: int = 512
    eval_batch_size: int = 512
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    lr: float = 1e-4
    seed: int = 42
    out_dir: str = "outputs/run"
    device: str = "auto"


def train_run(args: TrainArgs) -> dict[str, Any]:
    """Train a policy; write `best.pt` / `latest.pt` + `metrics.json`."""
    device = _pick_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(args.problem, size=args.size, **_env_kwargs(args.problem))
    model = make_model(
        env,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    algo = make_algo(
        args.algo,
        model,
        env,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    rng = torch.Generator(device=device).manual_seed(args.seed)
    best_reward = -float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            algo.train_step(rng)
        ev = algo.eval_step(rng)
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
        print(f"[{args.problem}/{args.algo}] epoch {epoch}: reward={reward:.4f}")

    payload = {
        "args": asdict(args),
        "device": device,
        "best_reward": best_reward,
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


@dataclass
class EvalArgs:
    problem: str
    ckpt_path: str
    algo: str = "reinforce"
    size: int = 50
    eval_batch_size: int = 512
    seed: int = 42
    device: str = "auto"


def eval_run(args: EvalArgs) -> dict[str, float]:
    """Load a checkpoint and report eval metrics."""
    device = _pick_device(args.device)
    env = make_env(args.problem, size=args.size, **_env_kwargs(args.problem))
    state = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = make_model(env, **state.get("arch", {}))
    model.load_state_dict(state.get("model", state), strict=False)
    algo = make_algo(args.algo, model, env, device=device, eval_batch_size=args.eval_batch_size)
    ev = algo.eval_step(torch.Generator(device=device).manual_seed(args.seed))
    metrics = {k: float(v) for k, v in ev.items()}
    print(json.dumps(metrics, indent=2))
    return metrics
