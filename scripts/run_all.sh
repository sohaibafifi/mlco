#!/usr/bin/env bash
# Train and evaluate registered problems using outputs/<problem>/ run directories.
# Usage:
#   PROBLEMS="cvrptw fjsp" bash scripts/run_all.sh
#   EPOCHS=1 STEPS=2 BATCH=8 DEVICE=cpu bash scripts/run_all.sh
#   SKIP_TRAIN=1 bash scripts/run_all.sh
# Training options: ALGO BACKBONE EPOCHS STEPS BATCH SIZE FJSP_SIZE DEVICE OUT_ROOT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROBLEMS="${PROBLEMS:-cvrptw op pdp fjsp}"
IFS=' ' read -ra problem_names <<< "$PROBLEMS"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  PROBLEMS="$PROBLEMS" bash "$HERE/train_all.sh"
fi
bash "$HERE/run_local_eval.sh" "${problem_names[@]}"
