#!/usr/bin/env bash
#
# train_all.sh: train one policy per core problem via the `neuroco` CLI.
#
# Each run writes outputs/<problem>/best.pt + latest.pt + metrics.json
# (the run-dir convention consumed by `neuroco eval/explain/probe` and
# `cax benchmark/adjudicate`). The script uses the core APIs directly.
#
# Override any knob from the environment, e.g.:
#   EPOCHS=100 ALGO=pomo BACKBONE=mamba SIZE=100 bash scripts/train_all.sh
#
set -euo pipefail

ALGO="${ALGO:-reinforce}"        # reinforce | pomo | ppo
BACKBONE="${BACKBONE:-am}"       # am | matnet | mamba
EPOCHS="${EPOCHS:-50}"
STEPS="${STEPS:-100}"            # steps per epoch
BATCH="${BATCH:-512}"
SIZE="${SIZE:-50}"               # routing problem size; fjsp uses FJSP_SIZE
FJSP_SIZE="${FJSP_SIZE:-10}"     # fjsp = number of jobs
DEVICE="${DEVICE:-auto}"         # auto | cuda | mps | cpu
OUT_ROOT="${OUT_ROOT:-outputs}"

IFS=' ' read -ra PROBLEMS <<< "${PROBLEMS:-tsp atsp cvrp cvrptw op pdp mtsp fjsp}"

for problem in "${PROBLEMS[@]}"; do
  size="$SIZE"
  [ "$problem" = "fjsp" ] && size="$FJSP_SIZE"
  echo "=================================================================="
  echo "[train_all] $problem  algo=$ALGO backbone=$BACKBONE size=$size epochs=$EPOCHS"
  echo "=================================================================="
  uv run neuroco train \
    --problem "$problem" \
    --algo "$ALGO" \
    --backbone "$BACKBONE" \
    --size "$size" \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS" \
    --batch-size "$BATCH" \
    --device "$DEVICE" \
    --out-dir "$OUT_ROOT/$problem"
done

echo "[train_all] done: checkpoints under $OUT_ROOT/<problem>/best.pt"
