"""Search POMO hyperparameters with Optuna.

Install neuro-co-core[sweep] and neuro-co-problems. From the repository root:
    uv run --no-sync python packages/neuro-co-core/examples/sweep_optuna.py \\
        --n_trials 20 --study_name tsp20_pomo --storage sqlite:///optuna.db

Stores trials in SQLite and prints the best parameters.
"""

import argparse
import sys
from pathlib import Path


def objective(trial, problem: str, size: int, steps: int):
    """Optuna objective: returns eval tour length (minimize)."""
    from benchmarks.suite import BenchSpec, run  # type: ignore[import-not-found]

    spec = BenchSpec(
        problem=problem,
        size=size,
        algo="pomo",
        steps=steps,
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        num_layers=trial.suggest_int("num_layers", 2, 4),
        num_heads=8,
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        n_starts=trial.suggest_categorical("n_starts", [10, 20, 30]),
        seed=0,
    )
    result = run(spec)
    return result.metrics["eval_tour_length_final"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--study_name", type=str, default="neuro_co_core_sweep")
    p.add_argument("--storage", type=str, default="sqlite:///optuna.db")
    args = p.parse_args()

    try:
        import optuna
    except ImportError:
        print("optuna not installed. Install with: pip install optuna", file=sys.stderr)
        return 1

    # Make `benchmarks.suite` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, args.problem, args.size, args.steps),
        n_trials=args.n_trials,
    )

    print("\n=== best trial ===")
    print(f"value={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
