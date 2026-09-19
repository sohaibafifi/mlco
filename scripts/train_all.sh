#!/usr/bin/env bash
# One set of options for both backends. Checkpoints go to OUT_ROOT/backend/problem.
# Example: EPOCHS=1 STEPS=2 BATCH=8 SIZE=6 bash scripts/train_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/progress.sh"
cd "$HERE/.."
UV="${UV_BIN:-uv}"
OUT_ROOT="${OUT_ROOT:-outputs/matched}"
DEVICE="${DEVICE:-cpu}"
IFS=' ' read -ra problems <<< "${PROBLEMS:-tsp cvrp}"
IFS=' ' read -ra backends <<< "${BACKENDS:-torch jax}"
if [ "${#problems[@]}" -eq 0 ] || [ "${#backends[@]}" -eq 0 ]; then
  echo '[train_all] PROBLEMS and BACKENDS must not be empty' >&2
  exit 2
fi
# Validate the entire request before starting the first run.
for backend in "${backends[@]}"; do
  case "$backend" in torch|jax) ;; *) echo "Unknown backend: $backend" >&2; exit 2 ;; esac
  for problem in "${problems[@]}"; do
    if [ "$backend" = jax ]; then
      case "$problem" in tsp|cvrp) ;; *) echo "JAX supports tsp and cvrp; use BACKENDS=torch for $problem" >&2; exit 2 ;; esac
      if [ "${ALGO:-pomo}" != pomo ] || [ "${BACKBONE:-am}" != am ]; then
        echo 'Shared Torch/JAX training requires ALGO=pomo BACKBONE=am' >&2; exit 2
      fi
      case "$DEVICE" in cpu|cuda|auto) ;; *) echo 'Use DEVICE=cpu or cuda for both backends' >&2; exit 2 ;; esac
    fi
    directory="$OUT_ROOT/$backend/$problem"
    if [ -d "$directory" ] && [ -n "$(ls -A "$directory")" ]; then
      echo "Run directory is not empty: $directory; choose a new OUT_ROOT" >&2
      exit 2
    fi
  done
done

# Unset options use the shared CLI defaults.
options=(--device "$DEVICE")
for pair in EPOCHS:epochs STEPS:steps-per-epoch BATCH:batch-size SIZE:size \
  EVAL_BATCH:eval-batch-size N_STARTS:n-starts HIDDEN_DIM:hidden-dim \
  NUM_LAYERS:num-layers NUM_HEADS:num-heads LR:lr SEED:seed \
  EVAL_SEED:eval-seed TEST_SEED:test-seed CAPACITY:capacity MAX_DEMAND:max-demand \
  OPTIMIZER:optimizer WEIGHT_DECAY:weight-decay GRAD_CLIP:grad-clip \
  ALGO:algo BACKBONE:backbone PRECISION:precision; do
  variable="${pair%%:*}"
  flag="${pair#*:}"
  value="${!variable-}"
  if [ -n "$value" ]; then options+=("--$flag" "$value"); fi
done
total=$((${#backends[@]} * ${#problems[@]}))
completed=0
for problem in "${problems[@]}"; do
  for backend in "${backends[@]}"; do
    mlco_progress train "$completed" "$total" "starting $backend/$problem"
    command=("$UV" run --no-sync neuroco train --backend "$backend" --problem "$problem"
      --out-dir "$OUT_ROOT/$backend/$problem" "${options[@]}")
    if [ "$problem" = fjsp ]; then command+=(--size "${FJSP_SIZE:-10}"); fi
    "${command[@]}"
    completed=$((completed + 1))
    mlco_progress train "$completed" "$total" "finished $backend/$problem"
  done
done
