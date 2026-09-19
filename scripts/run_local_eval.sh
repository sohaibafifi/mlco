#!/usr/bin/env bash
# Evaluate checkpoints using the problem, size, algorithm, and seed saved by training.
# Run from an installed checkout; use OUT_ROOT and DEVICE to select runs and hardware.
# Usage: bash scripts/run_local_eval.sh [cvrptw op pdp fjsp]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/progress.sh"
cd "$HERE/.."
UV="${UV_BIN:-uv}"
OUT_ROOT="${OUT_ROOT:-outputs}"
DEVICE="${DEVICE:-auto}"
if [ "$#" -gt 0 ]; then
  PROBLEMS=("$@")
else
  PROBLEMS=(cvrptw op pdp fjsp)
fi

run_dirs_for() {
  local problem="$1" directory
  local found=0
  for directory in "$OUT_ROOT/$problem"/train_seed*; do
    if [ -f "$directory/best.pt" ]; then
      printf '%s\n' "$directory"
      found=1
    fi
  done
  if [ "$found" -eq 0 ] && [ -f "$OUT_ROOT/$problem/best.pt" ]; then
    printf '%s\n' "$OUT_ROOT/$problem"
  fi
  return 0
}

eval_directories=()
eval_problems=()
skipped=0
for problem in "${PROBLEMS[@]}"; do
  runs=()
  while IFS= read -r directory; do
    [ -n "$directory" ] && runs+=("$directory")
  done < <(run_dirs_for "$problem")
  if [ "${#runs[@]}" -eq 0 ]; then
    echo "[local-eval] skipped $problem: no checkpoint under $OUT_ROOT/$problem" >&2
    skipped=$((skipped + 1))
    continue
  fi

  for directory in "${runs[@]}"; do
    eval_directories+=("$directory")
    eval_problems+=("$problem")
  done
done

total=${#eval_directories[@]}
if [ "$total" -eq 0 ]; then
  echo "[local-eval] no checkpoints found under $OUT_ROOT" >&2
  exit 1
fi

count=0
for directory in "${eval_directories[@]}"; do
  problem="${eval_problems[$count]}"
  mlco_progress eval "$count" "$total" "loading $directory ($problem)"
  metadata="$("$UV" run --no-sync python - "$directory" "$problem" <<'PY'
import json
import sys
from pathlib import Path

run_dir, problem = Path(sys.argv[1]), sys.argv[2]
args = json.loads((run_dir / "metrics.json").read_text())["args"]
if args["problem"] != problem:
    raise ValueError(f"Run problem {args['problem']!r} differs from {problem!r}")
print(int(args["size"]), args["algo"], int(args["seed"]))
PY
)"
  read -r size algo seed <<< "$metadata"
  mlco_progress eval "$count" "$total" "running $problem | size=$size algo=$algo seed=$seed"
  "$UV" run --no-sync neuroco eval \
    --problem "$problem" --size "$size" --algo "$algo" --seed "$seed" \
    --device "$DEVICE" --ckpt-path "$directory/best.pt"
  count=$((count + 1))
  mlco_progress eval "$count" "$total" "finished $directory"
done

if [ "$skipped" -gt 0 ]; then
  echo "[local-eval] evaluated $count run(s); skipped $skipped problem(s) without checkpoints" >&2
else
  echo "[local-eval] evaluated $count run(s)" >&2
fi
