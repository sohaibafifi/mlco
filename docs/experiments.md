# Experiments and run artifacts

Install the reporting dependencies and run the short CPU example:

```bash
uv sync --all-packages --extra report
uv run --no-sync neuroco experiment recipes/smoke.yaml
```

The recipe trains a small TSP policy, solves the same fixed test instances with
a nearest-neighbour baseline and the selected policy checkpoint, explains the
policy, and writes `experiments/smoke/report.html`. This is a workflow check;
the short training run is not intended to produce a competitive policy.

## Campaign isolation

All files belong to the selected campaign directory. Defaults are
`experiments/<recipe-name>/`, with a manifest, per-stage runs under `runs/`,
fixed datasets under `datasets/`, `results.csv`, `results.parquet`, figures,
and the HTML report. Results aggregation uses only that campaign's runs.

```bash
uv run --no-sync neuroco experiment recipes/smoke.yaml --dry-run
uv run --no-sync neuroco experiment recipes/smoke.yaml --out-dir experiments/check-1
uv run --no-sync neuroco experiment recipes/smoke.yaml --out-dir experiments/check-1 --resume
uv run --no-sync neuroco experiment recipes/smoke.yaml --out-dir experiments/check-1 --retry-failed
```

An existing campaign requires an explicit resume or retry. Completed stages are
preserved; interrupted or failed work is recorded in `manifest.json`. Resuming
requires the same recipe. Use a new output directory when changing the recipe.
The executor is sequential and requires `parallel: 1`.

## Recipe format

Use explicit training arguments in snake case:

```yaml
name: routing_cpu
parallel: 1
problems:
  cvrp:
    seeds: [0, 1]
    train:
      backend: torch
      algo: pomo
      backbone: am
      size: 20
      epochs: 2
      steps_per_epoch: 10
      batch_size: 8
      eval_batch_size: 16
      device: cpu
    baseline: {}
    eval: {}
    explain:
      methods: [gradient]
      num_instances: 2
      top_k: 2
      max_steps: 2
```

The complete baseline/evaluation path supports the shared TSP/CVRP training
protocol. Its nearest-neighbour heuristic is a reference solution, not an
optimality bound or a best-known solution. A reported gap to that baseline is
labelled accordingly. Baseline and policy evaluation use the same saved test
instances and record their data hash. Explanations use the trained checkpoint.

Unsupported combinations and unknown settings are rejected before execution.
Historical recipes using Hydra `overrides` need conversion to the explicit
format above; those overrides are not silently ignored. Start from `smoke.yaml`
for the executable format.

## Versioned run records

New runs write `run.json` with schema `neuro-co-run/v1`. It records a persistent
`run_id`, stage (`kind`), problem, seed, backend, status, timestamps, configuration,
provenance, scalar metrics, and artifact references. Status is one of `pending`,
`running`, `complete`, or `failed`. A resume preserves the run identity.
Artifact paths are relative to the run directory. Existing `config.json`,
`metrics.json`, and `evaluation.json` remain available.

```bash
uv run --no-sync neuroco results experiments/smoke/runs --out results.csv
uv run --no-sync neuroco results outputs/my-run --out single-run.parquet
```

The collector discovers modern records regardless of directory naming. It also
reads historical training metrics and the previous energy, explanation, and
probe layouts. A modern record takes precedence over its corresponding legacy
metrics to avoid duplicate values. Runs with no metrics remain visible with an
empty metric and value; their status is not converted into a quality score.
