#!/usr/bin/env bash
# Evaluate each backend's selected checkpoint using its saved protocol.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/progress.sh"
cd "$HERE/.."
UV="${UV_BIN:-uv}"
OUT_ROOT="${OUT_ROOT:-outputs/matched}"
DEVICE="${DEVICE:-cpu}"
IFS=' ' read -ra backends <<< "${BACKENDS:-torch jax}"
if [ "$#" -gt 0 ]; then problems=("$@"); else problems=(tsp cvrp); fi

directories=()
checkpoints=()
eval_problems=()
eval_backends=()
for problem in "${problems[@]}"; do
  for backend in "${backends[@]}"; do
    found=0
    suffix=pt
    [ "$backend" = jax ] && suffix=npz
    directory="$OUT_ROOT/$backend/$problem"
    if [ -f "$directory/best.$suffix" ]; then
      directories+=("$directory"); checkpoints+=("$directory/best.$suffix")
      eval_problems+=("$problem"); eval_backends+=("$backend"); found=1
    elif [ "$backend" = torch ]; then
      # Older Torch runs used flat directories or train_seed subdirectories.
      found=0
      for directory in "$OUT_ROOT/$problem"/train_seed*; do
        if [ -f "$directory/best.pt" ]; then
          directories+=("$directory"); checkpoints+=("$directory/best.pt")
          eval_problems+=("$problem"); eval_backends+=(torch); found=1
        fi
      done
      if [ "$found" -eq 0 ] && [ -f "$OUT_ROOT/$problem/best.pt" ]; then
        directory="$OUT_ROOT/$problem"
        directories+=("$directory"); checkpoints+=("$directory/best.pt")
        eval_problems+=("$problem"); eval_backends+=(torch); found=1
      fi
    fi
    if [ "$found" -eq 0 ]; then
      echo "Missing checkpoint for $backend/$problem under $OUT_ROOT" >&2
      exit 1
    fi
  done
done
total=${#directories[@]}
if [ "$total" -eq 0 ]; then echo "No checkpoints found under $OUT_ROOT" >&2; exit 1; fi
count=0
for directory in "${directories[@]}"; do
  backend="${eval_backends[$count]}"
  problem="${eval_problems[$count]}"
  metadata="$("$UV" run --no-sync python - "$directory" "$problem" <<'PY'
import json
import sys
from pathlib import Path
run, problem = Path(sys.argv[1]), sys.argv[2]
args = json.loads((run / 'metrics.json').read_text())['args']
if args['problem'] != problem:
    raise ValueError('Saved problem differs from requested problem')
print(args['size'], args['algo'], args['seed'])
PY
)"
  read -r size algo seed <<< "$metadata"
  mlco_progress eval "$count" "$total" "evaluating $backend/$problem"
  "$UV" run --no-sync neuroco eval --backend "$backend" --problem "$problem" \
    --size "$size" --algo "$algo" --seed "$seed" --device "$DEVICE" \
    --ckpt-path "${checkpoints[$count]}"
  count=$((count + 1))
  mlco_progress eval "$count" "$total" "finished $backend/$problem"
done
