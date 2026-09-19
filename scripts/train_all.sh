#!/usr/bin/env bash
# Train one policy per registered problem with the installed neuroco CLI.
# Each run writes best.pt, latest.pt, and metrics.json under OUT_ROOT/<problem>.
# Install the workspace before running this script.
# Example: EPOCHS=1 STEPS=2 BATCH=8 DEVICE=cpu bash scripts/train_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
UV="${UV_BIN:-uv}"

ALGO="${ALGO:-reinforce}"        # reinforce | pomo | ppo
BACKBONE="${BACKBONE:-am}"       # am | gnn | matnet | mamba
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
  echo "[train_all] $problem  algo=$ALGO backbone=$BACKBONE size=$size epochs=$EPOCHS"
  "$UV" run --no-sync neuroco train \
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
