#!/usr/bin/env bash
#
# run_all.sh: end-to-end neuro-co experimentation, core-native.
#
#   1. train  all problems   (scripts/train_all.sh -> neuroco train)
#   2. eval + XAI suite       (scripts/run_local_eval.sh)
#
# Both stages share the run-dir convention outputs/<problem>/ (best.pt +
# metrics.json). The script uses the core APIs directly.
#
# Usage:
#   scripts/run_all.sh                          # train + eval, default problems
#   PROBLEMS="cvrptw fjsp" scripts/run_all.sh   # subset
#   EPOCHS=100 ALGO=pomo scripts/run_all.sh     # forwarded to train_all.sh
#   SKIP_TRAIN=1 scripts/run_all.sh             # eval-only (reuse existing ckpts)
#
# Train knobs (forwarded to train_all.sh): ALGO BACKBONE EPOCHS STEPS BATCH
# SIZE FJSP_SIZE DEVICE OUT_ROOT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROBLEMS="${PROBLEMS:-cvrptw op pdp fjsp}"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "########## STAGE 1: train ##########"
  # train_all.sh trains its own fixed problem set; to honor PROBLEMS here,
  # loop train per problem via the same env knobs.
  ALGO="${ALGO:-reinforce}" BACKBONE="${BACKBONE:-am}" \
  EPOCHS="${EPOCHS:-50}" PROBLEMS="$PROBLEMS" \
    bash "$HERE/train_all.sh"
else
  echo "########## STAGE 1: train SKIPPED (SKIP_TRAIN=1) ##########"
fi

echo "########## STAGE 2: eval + XAI ##########"
# shellcheck disable=SC2086
bash "$HERE/run_local_eval.sh" $PROBLEMS

echo "########## run_all done ##########"
