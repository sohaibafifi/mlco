#!/usr/bin/env bash
# Train and evaluate registered problems using outputs/<problem>/ run directories.
# Usage:
#   PROBLEMS="cvrptw fjsp" bash scripts/run_all.sh
#   EPOCHS=1 STEPS=2 BATCH=8 DEVICE=cpu bash scripts/run_all.sh
#   SKIP_TRAIN=1 bash scripts/run_all.sh
# Training options: ALGO BACKBONE EPOCHS STEPS BATCH SIZE FJSP_SIZE DEVICE OUT_ROOT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/progress.sh"
PROBLEMS="${PROBLEMS:-cvrptw op pdp fjsp}"
IFS=' ' read -ra problem_names <<< "$PROBLEMS"
problem_count=${#problem_names[@]}
if [ "$problem_count" -eq 0 ]; then
  echo '[run_all] PROBLEMS must contain at least one problem' >&2
  exit 2
fi

completed=0
stages=2
stage='training'
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  stages=1
  stage='evaluation'
fi

report_failure() {
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "[run_all] failed during $stage after $(mlco_elapsed) (exit $status)" >&2
  fi
}
trap report_failure EXIT

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  mlco_progress run_all "$completed" "$stages" "training $problem_count problems: ${problem_names[*]}"
  PROBLEMS="$PROBLEMS" bash "$HERE/train_all.sh"
  completed=$((completed + 1))
fi
stage='evaluation'
mlco_progress run_all "$completed" "$stages" "evaluating checkpoints for $problem_count problems"
bash "$HERE/run_local_eval.sh" "${problem_names[@]}"
completed=$((completed + 1))
mlco_progress run_all "$completed" "$stages" 'finished'
