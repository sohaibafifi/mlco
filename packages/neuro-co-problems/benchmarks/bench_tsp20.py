"""Compare TSP REINFORCE training with eager or compiled execution and POMO.

Requires neuro-co-core and neuro-co-problems. Each run uses the same model
size and training-step budget, with one untimed warmup step. Evaluation uses
a shared random seed. Results report steps per second and greedy tour length.
"""

import argparse
import time
from dataclasses import dataclass

import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.models import AttentionModel as CoreAM
from neuro_co.problems.tsp.env import TSPEnv as CoreTSPEnv


@dataclass
class BenchCfg:
    size: int = 20
    steps: int = 100
    batch_size: int = 64
    eval_batch: int = 256
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    lr: float = 1e-4
    seed: int = 0


def bench_core_pomo(cfg: BenchCfg, n_starts: int) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    env = CoreTSPEnv(size=cfg.size)
    model = CoreAM(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
    )
    algo = POMO(
        model=model,
        env=env,
        cfg=POMOConfig(
            batch_size=cfg.batch_size,
            n_starts=n_starts,
            lr=cfg.lr,
            eval_batch_size=cfg.eval_batch,
        ),
    )
    rng = torch.Generator().manual_seed(cfg.seed)
    algo.train_step(rng)  # warmup
    t0 = time.perf_counter()
    for _ in range(cfg.steps):
        algo.train_step(rng)
    dt = time.perf_counter() - t0
    eval_rng = torch.Generator().manual_seed(99)
    eval_m = algo.eval_step(eval_rng)
    return {
        "wall_s": dt,
        "steps_per_s": cfg.steps / dt,
        "eval_tour_length": eval_m["eval_tour_length"],
    }


def bench_core(cfg: BenchCfg, compile_mode: str | None) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    env = CoreTSPEnv(size=cfg.size)
    model = CoreAM(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
    )
    if compile_mode:
        model.encode = torch.compile(model.encode, mode=compile_mode, fullgraph=True)  # type: ignore[assignment]
        model.decode_step = torch.compile(model.decode_step, mode=compile_mode, fullgraph=True)  # type: ignore[assignment]
    algo = REINFORCE(
        model=model,
        env=env,
        cfg=REINFORCEConfig(batch_size=cfg.batch_size, lr=cfg.lr, eval_batch_size=cfg.eval_batch),
    )
    rng = torch.Generator().manual_seed(cfg.seed)
    # Warmup (compile traces here).
    algo.train_step(rng)

    t0 = time.perf_counter()
    for _ in range(cfg.steps):
        algo.train_step(rng)
    dt = time.perf_counter() - t0

    eval_rng = torch.Generator().manual_seed(99)
    eval_m = algo.eval_step(eval_rng)
    return {
        "wall_s": dt,
        "steps_per_s": cfg.steps / dt,
        "eval_tour_length": eval_m["eval_tour_length"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--skip_compile", action="store_true")
    p.add_argument("--skip_pomo", action="store_true")
    p.add_argument("--n_starts", type=int, default=20)
    args = p.parse_args()

    cfg = BenchCfg(steps=args.steps, size=args.size, batch_size=args.batch_size)
    print(
        f"\nTSP-{cfg.size}  steps={cfg.steps}  batch={cfg.batch_size}  "
        f"hidden={cfg.hidden_dim}  layers={cfg.num_layers}\n"
    )

    print("[neuro-co-core REINFORCE eager]")
    core_eager = bench_core(cfg, compile_mode=None)
    _print(core_eager)

    if not args.skip_compile:
        print("\n[neuro-co-core REINFORCE compiled (default)]")
        core_compiled = bench_core(cfg, compile_mode="default")
        _print(core_compiled)

    if not args.skip_pomo:
        print(f"\n[neuro-co-core POMO (n_starts={args.n_starts})]")
        core_pomo = bench_core_pomo(cfg, n_starts=args.n_starts)
        _print(core_pomo)


def _print(m: dict[str, float]) -> None:
    print(
        f"  wall={m['wall_s']:6.2f}s  steps/s={m['steps_per_s']:5.2f}  "
        f"eval_tour_length={m['eval_tour_length']:.4f}"
    )


if __name__ == "__main__":
    main()
