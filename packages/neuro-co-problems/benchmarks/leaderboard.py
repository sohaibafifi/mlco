"""Aggregate benchmark JSON files into a sorted table.

Usage:
    uv run --no-sync python packages/neuro-co-problems/benchmarks/leaderboard.py \\
        --results outputs/benchmarks/ --markdown outputs/benchmarks/leaderboard.md
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(root: Path) -> list[dict]:
    results = []
    for p in sorted(root.glob("**/*.json")):
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
    return results


def format_table(results: list[dict]) -> str:
    """Render as markdown table grouped by (problem, size), sorted by best tour len."""
    headers = [
        "problem",
        "size",
        "algo",
        "n_starts",
        "steps",
        "wall_s",
        "best_tour_len",
        "final_tour_len",
        "seed",
    ]
    rows = []
    for r in results:
        cfg = r.get("config", {})
        mts = r.get("metrics", {})
        rows.append(
            [
                cfg.get("problem", "?"),
                cfg.get("size", "?"),
                cfg.get("algo", "?"),
                cfg.get("n_starts", "-"),
                cfg.get("steps", "-"),
                f"{mts.get('wall_s', float('nan')):.2f}",
                f"{mts.get('eval_tour_length_best', float('nan')):.4f}",
                f"{mts.get('eval_tour_length_final', float('nan')):.4f}",
                cfg.get("seed", "-"),
            ]
        )
    # Sort: problem asc, size asc, best_tour_len asc.
    rows.sort(key=lambda r: (str(r[0]), int(r[1]) if str(r[1]).isdigit() else 0, float(r[6])))

    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--markdown", type=Path, default=None)
    args = p.parse_args()

    if not args.results.exists():
        print(f"no such directory: {args.results}", file=sys.stderr)
        return 1
    results = load_results(args.results)
    table = format_table(results)
    print(table)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n")
        print(f"\nwrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
