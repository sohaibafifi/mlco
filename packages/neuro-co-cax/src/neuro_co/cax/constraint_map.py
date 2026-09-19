"""Per-problem constraint -> feature-column mapping.

`lambda_attribution` partitions the policy's per-step gradient over the
input feature tensor `env.build_features(state)` by CO constraint family.
On the core backbone every node carries a single feature vector, so a
family maps to the *columns* of that vector it parameterises.

Column layouts (must match each core env's `build_features`):

- cvrptw: [x, y, demand, tw_early, tw_late, t_now]      (d_in = 6)
- op:     [x, y, prize]                                  (d_in = 3)
- fjsp:   [mean_proc, num_eligible, ready, job_progress] (d_in = 4)

The groups are interpretable views, not necessarily a disjoint partition.
For OP, coordinate sensitivity is exposed as travel-budget pressure because
travel is the quantity consumed by the global budget. Family aggregation
uses a mean over columns, so wider groups do not receive an automatic scale
advantage.
"""

from __future__ import annotations

import torch

from neuro_co.attr.attribution import AttributionTrace

# problem -> [(constraint_family, (feature_column_indices,)), ...]
PROBLEM_CONSTRAINTS: dict[str, list[tuple[str, tuple[int, ...]]]] = {
    "cvrptw": [
        ("capacity", (2,)),  # demand
        ("time_window", (3, 4, 5)),  # tw_early, tw_late, current time
        ("spatial", (0, 1)),  # coords
    ],
    "vrptw": [  # alias for cvrptw
        ("capacity", (2,)),
        ("time_window", (3, 4, 5)),
        ("spatial", (0, 1)),
    ],
    "op": [
        ("prize", (2,)),
        ("travel_budget", (0, 1)),
    ],
    "fjsp": [
        ("processing", (0,)),  # mean processing time
        ("eligibility", (1,)),  # num eligible machines
        ("precedence", (2, 3)),  # ready flag and job progress / op order
    ],
}


def get_constraints(problem: str) -> list[tuple[str, tuple[int, ...]]]:
    """Return the `(family_name, feature_columns)` list for `problem`.

    Raises `KeyError` for an unregistered problem. The caller must declare
    its column decomposition here.
    """
    key = problem.lower()
    if key not in PROBLEM_CONSTRAINTS:
        available = ", ".join(sorted(PROBLEM_CONSTRAINTS))
        raise KeyError(f"no constraint map for problem={problem!r}. Available: {available}")
    return PROBLEM_CONSTRAINTS[key]


def aggregate_trace_by_family(trace: AttributionTrace, problem: str) -> torch.Tensor:
    """Aggregate element-wise attribution into normalized family scores.

    Returns ``[B, T, K]`` scores. A mean over both nodes and selected feature
    columns prevents group width and instance size from determining rank.
    """
    if trace.feature_scores is None:
        raise ValueError("trace has no feature_scores; recompute with a supported method")
    family_scores = []
    for _name, cols in get_constraints(problem):
        col_idx = torch.tensor(cols, device=trace.feature_scores.device)
        score = trace.feature_scores.index_select(-1, col_idx).mean(dim=(-1, -2))
        family_scores.append(score)
    return torch.stack(family_scores, dim=-1)
