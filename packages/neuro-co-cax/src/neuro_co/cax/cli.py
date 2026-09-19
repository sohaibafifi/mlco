"""`neuroco-cax-benchmark` CLI.

Drives `benchmark_runs` over one or more run directories produced
by `neuroco-train`. Lands per-run `lambda_attribution.json` plus
an optional aggregate `--out parquet`.

Usage::

    neuroco-cax-benchmark outputs/vrptw/train_seed0
    neuroco-cax-benchmark outputs/vrptw/train_seed* \
        --out experiments/cax_m1/lambda.parquet \
        --num-instances 8 --max-steps 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuro_co.cax.benchmark import benchmark_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neuroco-cax-benchmark",
        description="Run Lambda-attribution on one or more run dirs.",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more run directories produced by neuroco-train.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=8,
        help="Batch size sampled from each env (default 8).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="Cap on decoding steps per rollout (default 16).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional aggregate parquet path. Per-run JSONs always written.",
    )
    parser.add_argument(
        "--multipliers",
        type=str,
        default=None,
        choices=("proxy", "lp", "subgrad"),
        help=(
            "Lagrangian-multiplier backend for Lambda-attribution scaling. "
            "'proxy' (default None) = equal weight per family; 'lp' = "
            "OR-Tools GLOP LP duals; 'subgrad' = Beasley subgradient."
        ),
    )
    args = parser.parse_args(argv)

    mult_arg: str | None = None if args.multipliers in (None, "proxy") else args.multipliers
    benchmark_runs(
        run_dirs=args.run_dirs,
        num_instances=args.num_instances,
        max_steps=args.max_steps,
        out_parquet=args.out,
        multipliers=mult_arg,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
