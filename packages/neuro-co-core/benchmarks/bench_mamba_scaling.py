"""Encoder forward-time + memory scaling vs problem size n: AM (O(n^2)) vs Mamba (O(n)).

Measures per-forward wall-clock (and peak CUDA memory if available) of the
encoder alone, across increasing n. Shows the crossover where attention's
quadratic cost overtakes the SSM's linear cost.

Usage:
    uv run python packages/neuro-co-core/benchmarks/bench_mamba_scaling.py \\
        --sizes 50 100 200 500 1000 --batch 16 --device cpu
"""

import argparse
import sys
import time

import torch

from neuro_co.core.models import SSMEncoder
from neuro_co.core.models.encoders.am import AMEncoder


def _time_fwd(enc, x, iters: int, device: str) -> tuple[float, float]:
    enc = enc.to(device).eval()
    x = x.to(device)
    with torch.no_grad():
        enc(x)  # warmup
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            enc(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    return dt * 1e3, mem  # ms, MB


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    print(
        f"\nEncoder forward scaling  batch={args.batch}  hidden={args.hidden}  layers={args.layers}  device={args.device}"
    )
    print(
        f"{'n':>6} | {'AM ms':>9} {'AM MB':>8} | {'Mamba ms':>9} {'Mamba MB':>9} | {'speedup':>8}"
    )
    print("-" * 64)
    for n in args.sizes:
        x = torch.rand(args.batch, n, 2)
        am = AMEncoder(
            in_dim=2, hidden_dim=args.hidden, num_layers=args.layers, num_heads=args.heads
        )
        mb = SSMEncoder(
            in_dim=2, hidden_dim=args.hidden, num_layers=args.layers, bidirectional=True
        )
        am_ms, am_mem = _time_fwd(am, x, args.iters, args.device)
        mb_ms, mb_mem = _time_fwd(mb, x, args.iters, args.device)
        ratio = am_ms / mb_ms if mb_ms > 0 else float("nan")
        flag = "  <-- Mamba wins" if ratio > 1 else ""
        print(
            f"{n:>6} | {am_ms:>9.2f} {am_mem:>8.0f} | {mb_ms:>9.2f} {mb_mem:>9.0f} | {ratio:>7.2f}x{flag}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
