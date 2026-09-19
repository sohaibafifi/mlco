"""Throughput benchmark: PyTorch TSP env vs pure-JAX TSP env.

Measures wall-clock per N random-policy rollouts. Excludes any policy
compute: env-only throughput, no neural network involved.

Requires:
    uv pip install -e 'packages/neuro-co-core[jax]' -e packages/neuro-co-problems

Usage:
    uv run python packages/neuro-co-core/benchmarks/bench_jax_env.py \\
        --size 50 --batch 1024 --rollouts 100
"""

import argparse
import sys
import time


def bench_torch(size: int, batch: int, rollouts: int, device: str) -> float:
    import torch

    from neuro_co.core.env_registry import make_env

    env = make_env("tsp", size=size)
    gen = torch.Generator(device=device).manual_seed(0)

    # Warmup
    state = env.reset(batch, generator=gen, device=device)
    for _ in range(size - 1):
        mask = env.action_mask(state)
        # Random valid action: pick the first permitted index.
        action = mask.float().argmax(dim=-1)
        state, _, _ = env.step(state, action)

    t0 = time.perf_counter()
    for _ in range(rollouts):
        state = env.reset(batch, generator=gen, device=device)
        for _ in range(size - 1):
            mask = env.action_mask(state)
            action = mask.float().argmax(dim=-1)
            state, _, _ = env.step(state, action)
    return time.perf_counter() - t0


def bench_jax(size: int, batch: int, rollouts: int) -> float:
    import jax
    import jax.numpy as jnp

    from neuro_co.core.env_registry import make_env

    env = make_env("tsp", backend="jax", size=size)
    key = jax.random.PRNGKey(0)

    @jax.jit
    def one_rollout(k: jax.Array) -> jax.Array:
        state = env.reset(k, batch_size=batch)

        def body(carry, _):
            s = carry
            mask = env.action_mask(s)
            action = jnp.argmax(mask.astype(jnp.float32), axis=-1)
            s2, _, _ = env.step(s, action)
            return s2, None

        final, _ = jax.lax.scan(body, state, None, length=size - 1)
        return final.tour_length

    # Warmup (jit compile)
    keys = jax.random.split(key, rollouts + 1)
    _ = one_rollout(keys[0]).block_until_ready()

    t0 = time.perf_counter()
    for i in range(rollouts):
        _ = one_rollout(keys[i + 1]).block_until_ready()
    return time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--rollouts", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip_jax", action="store_true")
    args = p.parse_args()

    print(f"TSP-{args.size}  batch={args.batch}  rollouts={args.rollouts}  device={args.device}")

    dt = bench_torch(args.size, args.batch, args.rollouts, args.device)
    print(f"  [torch] wall={dt:.3f}s  rollouts/s={args.rollouts / dt:.1f}")

    if not args.skip_jax:
        try:
            dtj = bench_jax(args.size, args.batch, args.rollouts)
            print(f"  [jax]   wall={dtj:.3f}s  rollouts/s={args.rollouts / dtj:.1f}")
            print(f"  speedup = {dt / dtj:.2f}x")
        except ImportError as e:
            print(f"  [jax] skipped: {e}")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
