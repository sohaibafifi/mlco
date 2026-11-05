"""Solve a supplied instance and export a common, validated result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import atomic_write_json, write_run


def main_cli(argv: list[str] | None = None) -> int:
    from neuro_co.problems import solve

    parser = argparse.ArgumentParser(prog="neuroco solve", description=__doc__)
    parser.add_argument("instance", type=Path, help="One JSON, NPZ, .tsp, or .vrp instance.")
    parser.add_argument("--problem", help="Problem name when it is absent from the input file.")
    parser.add_argument("--method", help="neural, nearest_neighbour, pyvrp, ortools, or cpsat.")
    parser.add_argument("--ckpt", "--ckpt-path", dest="checkpoint", type=Path)
    parser.add_argument("--backend", choices=["torch", "jax"], default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-runtime", type=float, default=5.0, help="Classical solver budget in seconds."
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--out", type=Path, help="Write the result as JSON (also printed to stdout)."
    )
    output.add_argument(
        "--out-dir", type=Path, help="Create an isolated solution.json and run.json bundle."
    )
    args = parser.parse_args(argv)
    if (
        args.out_dir is not None
        and args.out_dir.exists()
        and (not args.out_dir.is_dir() or any(args.out_dir.iterdir()))
    ):
        parser.error("--out-dir must be empty; choose a new solve directory")
    method = args.method or ("neural" if args.checkpoint is not None else "nearest_neighbour")
    try:
        result = solve(
            args.instance,
            problem=args.problem,
            method=method,
            checkpoint=args.checkpoint,
            backend=args.backend,
            device=args.device,
            seed=args.seed,
            max_runtime=args.max_runtime,
        )
        payload = result.to_dict()
    except (ValueError, TypeError, OSError, ImportError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    if args.out is not None:
        atomic_write_json(args.out, payload)
    if args.out_dir is not None:
        atomic_write_json(args.out_dir / "solution.json", payload)
        write_run(
            args.out_dir,
            kind="solve",
            problem=result.problem,
            seed=args.seed,
            backend=args.backend if method == "neural" else method,
            config={
                "method": result.method,
                "instance": str(args.instance.resolve()),
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "solve_status": payload["status"],
                "sense": result.sense,
                "device": args.device,
            },
            metrics={
                "objective": result.objective,
                "bound": result.bound,
                "gap": result.gap,
                "runtime_s": result.runtime_s,
            },
            artifacts={"solution": "solution.json"},
            status="failed" if payload["status"] in {"invalid", "error"} else "complete",
        )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return {"optimal": 0, "feasible": 0, "infeasible": 2, "unknown": 3, "invalid": 1, "error": 1}[
        payload["status"]
    ]


if __name__ == "__main__":
    raise SystemExit(main_cli())
