"""`neuroco`, the single entry point for core-native experiment drivers.

Subcommands::

    neuroco train   --problem <name> [--algo reinforce|pomo|ppo] [--size N] ...
    neuroco eval    --problem <name> --ckpt-path PATH [--algo ...] [--size N]
    neuroco explain --problem <name> [--ckpt-path PATH] [--method gradient|ig|contrastive]

The core factory builds environments, models, and algorithms. The environment
registry defines the available problems.
"""

from __future__ import annotations

import argparse
from typing import Any

from neuro_co.cli.runners import EvalArgs, TrainArgs, eval_run, train_run


def _add_train(sub: Any) -> None:
    p = sub.add_parser("train", help="train a policy")
    p.add_argument("--problem", required=True)
    p.add_argument("--algo", default="reinforce", choices=["reinforce", "pomo", "ppo"])
    p.add_argument("--backbone", default="am", choices=["am", "gnn", "matnet", "mamba"])
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs/run")
    p.add_argument("--device", default="auto")


def _add_eval(sub: Any) -> None:
    p = sub.add_parser("eval", help="evaluate a checkpoint")
    p.add_argument("--problem", required=True)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--algo", default="reinforce", choices=["reinforce", "pomo", "ppo"])
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")


def _add_explain(sub: Any) -> None:
    p = sub.add_parser("explain", help="attribute + faithfulness for a policy")
    p.add_argument("--problem", required=True)
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--method", default="gradient", choices=["gradient", "ig", "contrastive"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--num-instances", type=int, default=8)
    p.add_argument("--out-dir", default="outputs/explain")


def main(argv: list[str] | None = None) -> None:
    import sys

    # Subcommands with their own parsers delegate lazily to keep core deps light.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "probe":
        from neuro_co.probe.probes_cli import main as probe_main

        probe_main(argv[1:])
        return
    if argv and argv[0] in {"results", "figures", "stats", "experiment", "budget"}:
        cmd, rest = argv[0], argv[1:]
        if cmd == "results":
            from neuro_co.cli.results import main_cli
        elif cmd == "figures":
            from neuro_co.cli.figures import main_cli
        elif cmd == "stats":
            from neuro_co.cli.stats import main_cli
        elif cmd == "experiment":
            from neuro_co.cli.experiment import main_run_cli as main_cli
        else:  # budget
            from neuro_co.cli.experiment import main_budget_cli as main_cli
        raise SystemExit(main_cli(rest))

    parser = argparse.ArgumentParser(
        prog="neuroco",
        epilog="Other commands: probe, results, figures, stats, experiment, budget. "
        "Use COMMAND --help for their options.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_train(sub)
    _add_eval(sub)
    _add_explain(sub)
    args = parser.parse_args(argv)

    if args.cmd == "train":
        train_run(
            TrainArgs(
                problem=args.problem,
                algo=args.algo,
                backbone=args.backbone,
                size=args.size,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                lr=args.lr,
                seed=args.seed,
                out_dir=args.out_dir,
                device=args.device,
            )
        )
    elif args.cmd == "eval":
        eval_run(
            EvalArgs(
                problem=args.problem,
                ckpt_path=args.ckpt_path,
                algo=args.algo,
                size=args.size,
                seed=args.seed,
                device=args.device,
            )
        )
    elif args.cmd == "explain":
        import torch

        from neuro_co.core.factory import make_env, make_model
        from neuro_co.probe import explain_policy

        env = make_env(args.problem, size=args.size)
        arch: dict[str, Any] = {}
        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
            arch = ckpt.get("arch", {}) if isinstance(ckpt, dict) else {}
        model = make_model(env, **arch)
        explain_policy(
            model,
            env,
            ckpt_path=args.ckpt_path,
            num_instances=args.num_instances,
            top_k=args.top_k,
            method=args.method,
            output_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
