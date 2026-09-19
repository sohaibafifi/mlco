"""Seeded benchmark suite. Reproducible, JSON-emitting.

Single env-agnostic algo per kind: `reinforce`, `pomo`, `ppo`: works
on any problem (`tsp`, `cvrp`, `cvrptw`). No `*_cvrp` variants needed.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.ppo import PPO, PPOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.env import Env
from neuro_co.core.envs.cvrp import CVRPEnv
from neuro_co.core.envs.cvrptw import CVRPTWEnv
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel

RESULTS_SCHEMA_VERSION = "1"


@dataclass(slots=True)
class BenchSpec:
    problem: str  # "tsp" | "cvrp" | "cvrptw"
    size: int
    algo: str  # "reinforce" | "pomo" | "ppo"
    steps: int = 200
    batch_size: int = 64
    eval_batch_size: int = 256
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    lr: float = 1e-4
    seed: int = 0
    n_starts: int = 20
    device: str = "cpu"
    precision: str = "fp32"  # "fp32" | "bf16" | "fp16"
    log_every: int = 10


@dataclass(slots=True)
class BenchResult:
    schema_version: str = RESULTS_SCHEMA_VERSION
    timestamp: str = ""
    git_sha: str | None = None
    host: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _host_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }


def _build_env(spec: BenchSpec) -> Env:
    if spec.problem == "tsp":
        return TSPEnv(size=spec.size)
    if spec.problem == "cvrp":
        return CVRPEnv(size=spec.size)
    if spec.problem == "cvrptw":
        return CVRPTWEnv(size=spec.size)
    raise ValueError(f"unknown problem: {spec.problem}")


def _build_algo(spec: BenchSpec, env: Env):
    model = AttentionModel(
        in_dim=env.encoder_in_dim,
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
    )
    if spec.algo == "reinforce":
        return REINFORCE(
            model=model,
            env=env,
            cfg=REINFORCEConfig(
                batch_size=spec.batch_size,
                lr=spec.lr,
                eval_batch_size=spec.eval_batch_size,
                precision=spec.precision,  # type: ignore[arg-type]
            ),
            device=spec.device,
        )
    if spec.algo == "pomo":
        return POMO(
            model=model,
            env=env,
            cfg=POMOConfig(
                batch_size=spec.batch_size,
                n_starts=spec.n_starts,
                lr=spec.lr,
                eval_batch_size=spec.eval_batch_size,
                precision=spec.precision,  # type: ignore[arg-type]
            ),
            device=spec.device,
        )
    if spec.algo == "ppo":
        return PPO(
            model=model,
            env=env,
            cfg=PPOConfig(
                batch_size=spec.batch_size,
                lr=spec.lr,
                eval_batch_size=spec.eval_batch_size,
                hidden_dim=spec.hidden_dim,
                precision=spec.precision,  # type: ignore[arg-type]
            ),
            device=spec.device,
        )
    raise ValueError(f"unknown algo: {spec.algo}")


def run(spec: BenchSpec) -> BenchResult:
    torch.manual_seed(spec.seed)
    env = _build_env(spec)
    algo = _build_algo(spec, env)

    rng = torch.Generator(device=spec.device).manual_seed(spec.seed)
    eval_rng = torch.Generator(device=spec.device).manual_seed(spec.seed + 99)

    initial_eval = algo.eval_step(eval_rng)["eval_tour_length"]
    loss_curve: list[tuple[int, float]] = []
    best_eval = initial_eval

    algo.train_step(rng)  # warmup

    tag = f"{spec.problem}{spec.size} {spec.algo} s{spec.seed}"
    eval_period = max(1, spec.steps // 10)  # ~10 eval points across the run
    t0 = time.perf_counter()
    bar = tqdm(range(spec.steps), desc=tag, unit="step", dynamic_ncols=True)
    last_eval = initial_eval
    for step in bar:
        m = algo.train_step(rng)
        if (step + 1) % spec.log_every == 0:
            loss_curve.append((step + 1, m.get("loss", float("nan"))))
        if (step + 1) % eval_period == 0:
            last_eval = algo.eval_step(eval_rng)["eval_tour_length"]
            best_eval = min(best_eval, last_eval)
        bar.set_postfix(
            loss=f"{m.get('loss', float('nan')):+.3f}",
            tour=f"{last_eval:.3f}",
            best=f"{best_eval:.3f}",
        )
    bar.close()
    dt = time.perf_counter() - t0

    final_eval = algo.eval_step(eval_rng)["eval_tour_length"]
    best_eval = min(best_eval, final_eval)

    return BenchResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_sha=_git_sha(),
        host=_host_info(),
        config=asdict(spec),
        metrics={
            "wall_s": dt,
            "steps_per_s": spec.steps / dt,
            "eval_tour_length_initial": initial_eval,
            "eval_tour_length_final": final_eval,
            "eval_tour_length_best": best_eval,
            "loss_curve": loss_curve,
        },
    )


def write(res: BenchResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(res), f, indent=2, default=str)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=["tsp", "cvrp", "cvrptw"], required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--algo", choices=["reinforce", "pomo", "ppo"], required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_starts", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    spec = BenchSpec(**{k: v for k, v in vars(args).items() if k != "output"})
    res = run(spec)
    write(res, args.output)
    print(json.dumps(asdict(res)["metrics"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
