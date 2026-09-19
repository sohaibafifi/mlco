#!/usr/bin/env bash
# Full local post-train experimentation pipeline (no OAR; laptop / single box).
#
# For each problem + run dir produced by training, runs the core-native
# evaluation + XAI suite:
#   1. neuroco eval        -> policy metrics (stdout)
#   2. neuroco explain     -> explanation.json (gradient + IG attribution + faithfulness)
#   3. neuroco probe       -> probes.json / probes.md (encoder concept probes)
#   4. cax adjudicate      -> adjudication.json (lambda modes vs CP-counterfactual)
#   5. KBS B2 / B3         -> deletion-faithfulness + PAC sufficient-subset
#   6. KBS B4 (fjsp only)  -> family-magnitude profile (needs step 4 first)
#
# Run-dir layout is auto-detected: either outputs/<problem>/ (from
# `train_all.sh`) or outputs/<problem>/train_seed<S>/ (from OAR / sweeps).
#
# Usage:
#   scripts/run_local_eval.sh                 # default core problems
#   scripts/run_local_eval.sh cvrptw fjsp     # subset
#
# Optional: install the `cp` extra for OR-Tools LP duals (adjudicate `lp`
# mode) + classical baselines: `uv sync --package neuro-co-cax --extra cp`.
set -euo pipefail

ALL=(cvrptw op pdp fjsp)
PROBLEMS=("${@:-${ALL[@]}}")

# Discover run dirs for a problem: prefer seed dirs, else the bare dir.
run_dirs_for() {
  local p="$1" dirs=()
  for d in "outputs/$p"/train_seed*; do
    [ -f "$d/best.pt" ] && dirs+=("$d")
  done
  if [ ${#dirs[@]} -eq 0 ] && [ -f "outputs/$p/best.pt" ]; then
    dirs+=("outputs/$p")
  fi
  printf '%s\n' "${dirs[@]}"
}

for P in "${PROBLEMS[@]}"; do
  echo
  echo "================= $P ================="
  RUNS=()
  while IFS= read -r line; do
    [ -n "$line" ] && RUNS+=("$line")
  done < <(run_dirs_for "$P")
  if [ ${#RUNS[@]} -eq 0 ]; then
    echo "[skip] no trained run dir under outputs/$P (run scripts/train_all.sh first)"
    continue
  fi

  for RUN in "${RUNS[@]}"; do
    echo "--- run dir: $RUN ---"

    echo "[1/6] eval"
    uv run neuroco eval --problem "$P" --ckpt-path "$RUN/best.pt" || true

    echo "[2/6] explain (gradient + ig)"
    for M in gradient ig; do
      uv run neuroco explain --problem "$P" --ckpt-path "$RUN/best.pt" \
          --method "$M" --num-instances 16 --top-k 5 \
          --out-dir "$RUN/explain_$M" || true
    done

    echo "[3/6] probe (+ figures)"
    uv run neuroco probe "$RUN" --num-instances 16 --epochs 100 || true

    echo "[4/6] adjudicate (cax)"
    uv run python - "$RUN" "$P" <<'PY' || true
import sys
from neuro_co.cax.adjudicate import adjudicate_run
adjudicate_run(sys.argv[1], problem=sys.argv[2], modes=("proxy",),
               num_instances=16, max_steps=12, cf_shots=64)
print(f"[adjudicate] wrote {sys.argv[1]}/adjudication.json")
PY

    echo "[5/6] KBS B2 (faithfulness) + B3 (PAC subset)"
    uv run python scripts/kbs/run_b2_faithfulness.py "$P" 0 "$RUN" || true
    uv run python scripts/kbs/run_b3_pac_subset.py  "$P" 0 "$RUN" || true

    if [ "$P" = "fjsp" ]; then
      echo "[6/6] KBS B4 (fjsp family profile)"
      uv run python scripts/kbs/run_b4_fjsp_cosine.py 0 "$RUN" || true
    fi
  done
done

echo
echo "===== figures ====="
uv run --package neuro-co-probe python scripts/plot_results.py || true

echo
echo "[local-eval] done."
echo "  per-run:  <run>/explain_*/explanation.json, <run>/probes/probes.{json,md},"
echo "            <run>/probes/figures/*.png, <run>/adjudication.json"
echo "  summary:  experiments/kbs_*/*.json + experiments/figures/*.png"
