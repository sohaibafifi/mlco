# Solving supplied instances

`solve` operates on supplied data rather than generating a random evaluation set.
The Python API and `neuroco solve` return the same versioned result.

## Command line

```bash
uv run --no-sync neuroco solve instance.vrp --ckpt outputs/cvrp/best.pt --out solution.json
uv run --no-sync neuroco solve instance.vrp --method nearest_neighbour --out solution.json
uv run --no-sync neuroco solve instance.vrp --method ortools --max-runtime 2 \
  --out-dir outputs/solve-instance
uv run --no-sync neuroco results outputs/solve-instance --out results.csv
```

When no method is specified, a checkpoint selects `neural`; otherwise the command
uses `nearest_neighbour`. `--ckpt-path` is an alias for `--ckpt`. Optional classical
backends require the corresponding `neuro-co-problems` extras. The command prints
JSON to stdout; `--out` also writes it atomically. `--out-dir` writes
`solution.json` and `run.json` and requires an empty output directory.

Exit codes are 0 for a feasible incumbent, 2 for proven infeasibility or an input
error, 3 for an unknown result without an incumbent, and 1 for an invalid solution
or a solver error. The JSON status distinguishes solver outcomes from input errors.

## Python API

```python
import numpy as np
from neuro_co.problems import RoutingInstance, load_instance, solve

instance = RoutingInstance(
    problem="cvrp",
    coords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    demand=np.array([0, 2, 2, 2]),
    capacity=4,
)
result = solve(instance, method="nearest_neighbour")
print(result.objective, result.status, result.validation)
payload = result.to_dict()

instance = load_instance("instance.vrp")
result = solve(instance, method="neural", checkpoint="outputs/cvrp/best.pt")
```

Routing indices are zero based internally. Index 0 is the depot or fixed TSP
origin. Routes store the other visited nodes, with departure and return to index 0
implicit. `node_ids` preserves the mapping to source-file node identifiers.

## Input formats

A JSON file describes one routing instance. For example:

```json
{
  "problem": "cvrp",
  "coords": [[0, 0], [1, 0], [1, 1], [0, 1]],
  "demand": [0, 2, 2, 2],
  "capacity": 4,
  "distance_convention": "euclidean"
}
```

NPZ accepts the same arrays for a single instance, with a scalar `problem` string
or the `--problem` argument. Batched training datasets must be split into
individual instances. TSPLIB `.tsp` and CVRPLIB `.vrp` input retain their declared
distance convention. Unsupported formats or ambiguous dimensions are rejected.
For time windows, `time_windows` has shape `(nodes, 2)` and `service_time` has one
entry per node. All time and distance units must be consistent.

The Python API and JSON/NPZ command also accept the existing problem-specific
arrays for JSSP/FJSP, OP, PDP, and FLP. For example, a two-operation FJSP instance:

```json
{
  "problem": "fjsp",
  "proc_times": [[1.0], [1.0]],
  "ops_job_map": [0, 0],
  "ops_sequence_order": [0, 1]
}
```

Solve it with `neuroco solve schedule.json --method cpsat --out schedule-result.json`.
`proc_times` here uses the core FJSP layout `(operations, machines)`; zero means
an ineligible machine. Other solver inputs retain their existing problem array
contracts.

| Method | Supported supplied problems |
|---|---|
| `neural` | TSP and CVRP, Torch or shared-protocol JAX checkpoints |
| `nearest_neighbour` | TSP, CVRP, CVRPTW |
| `pyvrp` | CVRP, CVRPTW |
| `ortools` | CVRP, PDP |
| `cpsat` | CVRPTW, JSSP, FJSP, OP |
| `lp` | FLP |

CVRPTW classical adapters currently require `euc_2d` distances. Neural coordinate
policies reject explicit distance matrices. The requested seed is recorded;
legacy adapters that do not expose a seed report that limitation in metadata.

No implicit coordinate normalization is applied. A checkpoint trained on unit
square coordinates can produce poor solutions on coordinates in other units.
The result reports the objective in the supplied instance's units and distance
convention. Use a checkpoint appropriate for the target distribution.

## Common result

`SolveResult.to_dict()` uses schema `neuro-co-solve/v1` and contains:

| Field | Meaning |
|---|---|
| `problem`, `method` | Problem and selected method |
| `status` | `optimal`, `feasible`, `infeasible`, `unknown`, `invalid`, or `error` |
| `solution` | Routes, schedule, or another problem-specific witness |
| `objective`, `sense` | Objective in its documented units, and `min` or `max` |
| `feasible`, `validation` | Feasibility evidence and independent checks when available |
| `bound`, `gap` | Certified bound and relative gap when applicable to that objective |
| `runtime_s`, `solver_status` | Measured runtime and original backend status |
| `metadata` | Distance/scaling conventions, backend details, and provenance |

Missing objectives and bounds are JSON `null`, never NaN. An unknown result is
not an infeasibility proof. A heuristic incumbent is `feasible`, not `optimal`.
Bounds from a rounded or relaxed model are not automatically certificates for the
original problem. Inspect the recorded convention before comparing results.

Routing solutions are checked independently for visit coverage, node indices,
vehicle loads, time windows, and depot return where applicable. Their objective
is recalculated from the instance. A solver witness that fails these checks is
reported as `invalid` with the violations attached.

Historical solver adapters keep their original default return values. Supported
adapters additionally accept `return_result=True` to return the common result
without discarding their native solver status.
CVRP corpus functions retain their existing result dataclasses; their
`solve_instance_result` entry points provide the common result used by `solve`.
