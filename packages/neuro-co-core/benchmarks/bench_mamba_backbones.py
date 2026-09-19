"""Backbone comparison: AM vs Mamba (TSP) and MatNet vs AM (ATSP).

Same algo (POMO), same hyperparams, same seed: only the encoder differs.
Reports wall-clock per training run + final greedy eval tour length on a
fixed held-out set.

MatNet only applies to matrix problems (ATSP), so it is compared there
against AM fed the distance-matrix rows as node features.

Usage:
    uv run python packages/neuro-co-core/benchmarks/bench_mamba_backbones.py \\
        --size 20 --steps 100 --batch 64 --device cpu
"""

import argparse
import sys
import time

import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.models import AttentionModel, MambaModel, MatNetModel
from neuro_co.problems.atsp.env import ATSPEnv
from neuro_co.problems.tsp.env import TSPEnv


def _run(model, env, *, steps: int, batch: int, n_starts: int, lr: float, device: str) -> dict:
    algo = POMO(
        model=model,
        env=env,
        cfg=POMOConfig(batch_size=batch, n_starts=n_starts, lr=lr, eval_batch_size=256),
        device=device,
    )
    rng = torch.Generator().manual_seed(0)
    algo.train_step(rng)  # warmup (compile / lazy init)
    t0 = time.perf_counter()
    for _ in range(steps):
        algo.train_step(rng)
    dt = time.perf_counter() - t0
    tour = algo.eval_step(rng)["eval_tour_length"]
    n_params = sum(p.numel() for p in model.parameters())
    return {"wall_s": dt, "steps_per_s": steps / dt, "tour": tour, "params": n_params}


def _print(name: str, m: dict) -> None:
    print(
        f"  {name:18s} wall={m['wall_s']:6.2f}s  steps/s={m['steps_per_s']:5.2f}  "
        f"tour={m['tour']:.4f}  params={m['params'] / 1e3:.0f}k"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    n_starts_tsp = args.size - 1  # city 0 pre-visited
    kw = dict(hidden_dim=args.hidden, num_layers=args.layers, num_heads=args.heads)

    print(
        f"\nTSP-{args.size}  steps={args.steps}  batch={args.batch}  hidden={args.hidden}  layers={args.layers}"
    )
    print("[encoder on TSP: coords]")
    torch.manual_seed(0)
    tsp = TSPEnv(size=args.size)
    _print(
        "AM (attention)",
        _run(
            AttentionModel(in_dim=tsp.encoder_in_dim, **kw),
            tsp,
            steps=args.steps,
            batch=args.batch,
            n_starts=n_starts_tsp,
            lr=args.lr,
            device=args.device,
        ),
    )
    torch.manual_seed(0)
    _print(
        "Mamba (SSM)",
        _run(
            MambaModel(in_dim=tsp.encoder_in_dim, **kw),
            tsp,
            steps=args.steps,
            batch=args.batch,
            n_starts=n_starts_tsp,
            lr=args.lr,
            device=args.device,
        ),
    )

    print(f"\nATSP-{args.size}  (asymmetric matrix)")
    print("[encoder on ATSP: distance matrix]")
    torch.manual_seed(0)
    atsp = ATSPEnv(size=args.size)
    _print(
        "MatNet (edge-aware)",
        _run(
            MatNetModel(problem_size=atsp.encoder_in_dim, **kw),
            atsp,
            steps=args.steps,
            batch=args.batch,
            n_starts=args.size,
            lr=args.lr,
            device=args.device,
        ),
    )
    torch.manual_seed(0)
    _print(
        "AM (matrix rows)",
        _run(
            AttentionModel(in_dim=atsp.encoder_in_dim, **kw),
            atsp,
            steps=args.steps,
            batch=args.batch,
            n_starts=args.size,
            lr=args.lr,
            device=args.device,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
