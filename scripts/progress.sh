#!/usr/bin/env bash
# Shared progress output for the training and evaluation scripts.

MLCO_PROGRESS_STARTED=$SECONDS

mlco_elapsed() {
  local elapsed=$((SECONDS - MLCO_PROGRESS_STARTED))
  printf '%02d:%02d:%02d' "$((elapsed / 3600))" "$((elapsed / 60 % 60))" "$((elapsed % 60))"
}

mlco_progress() {
  local label="$1" completed="$2" total="$3" message="$4"
  local filled=$((20 * completed / total))
  local full='####################' empty='....................'
  printf '[%s] [%s%s] %d/%d (%d%%) | elapsed %s | %s\n' \
    "$label" "${full:0:filled}" "${empty:filled}" \
    "$completed" "$total" "$((100 * completed / total))" \
    "$(mlco_elapsed)" "$message" >&2
}
