#!/usr/bin/env bash
# Benchmark grid: TSP × CVRP × algos × seeds.
# Writes one JSON per cell to benchmarks/results/grid/.
#
# Usage: bash packages/neuro-co-core/benchmarks/run_grid.sh [device]
# Default device: mps.

set -euo pipefail
DEVICE="${1:-mps}"
OUT_DIR="packages/neuro-co-core/benchmarks/results/grid"
mkdir -p "$OUT_DIR"

STEPS="${STEPS:-50}"
BATCH="${BATCH:-32}"
HIDDEN="${HIDDEN:-32}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
EVAL_BATCH="${EVAL_BATCH:-64}"
PRECISION="${PRECISION:-fp32}"

run() {
  local problem="$1" size="$2" algo="$3" seed="$4" extra="${5:-}"
  local fname="${OUT_DIR}/${problem}${size}_${algo}_seed${seed}.json"
  echo "===> $fname"
  # shellcheck disable=SC2086
  uv run python packages/neuro-co-core/benchmarks/suite.py \
    --problem "$problem" --size "$size" --algo "$algo" \
    --steps "$STEPS" --batch_size "$BATCH" --eval_batch_size "$EVAL_BATCH" \
    --hidden_dim "$HIDDEN" --num_layers "$LAYERS" --num_heads "$HEADS" \
    --seed "$seed" --device "$DEVICE" --precision "$PRECISION" \
    --output "$fname" $extra > /dev/null
}

SEEDS="${SEEDS:-0 1 2}"
for seed in $SEEDS; do
  # TSP
  for size in 20 50; do
    run tsp "$size" reinforce "$seed"
    run tsp "$size" pomo      "$seed" "--n_starts $((size - 1))"
    run tsp "$size" ppo       "$seed"
  done
  # CVRP
  for size in 20 50; do
    run cvrp "$size" reinforce "$seed"
    run cvrp "$size" pomo      "$seed" "--n_starts $size"
    run cvrp "$size" ppo       "$seed"
  done
  # CVRPTW
  for size in 20; do
    run cvrptw "$size" reinforce "$seed"
    run cvrptw "$size" pomo      "$seed" "--n_starts $size"
  done
done

echo
echo "Done. Render leaderboard:"
echo "  uv run python packages/neuro-co-core/benchmarks/leaderboard.py --results $OUT_DIR"
