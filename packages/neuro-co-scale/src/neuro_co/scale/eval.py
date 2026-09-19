"""Compare hierarchical policies and baselines on fixed held-out instances.

Evaluation computes per-instance costs and feasibility. Paired comparisons
report relative gaps and win rates on the same instance set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from .compose import compose_nearest_neighbor
from .decode import DecodeMode, HierarchicalPolicy
from .metrics import evaluate_solution
from .types import Partition, VRPInstance

PartitionFn = Callable[[VRPInstance], Partition]


@dataclass(frozen=True, slots=True)
class MethodEval:
    name: str
    mean_cost: float
    feasible_rate: float
    mean_routes: float
    costs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PairedGap:
    method: str
    reference: str
    mean_gap_pct: float
    win_rate: float


def evaluate_policy(
    policy: HierarchicalPolicy,
    instances: list[VRPInstance],
    partition_fn: PartitionFn,
    *,
    name: str = "policy",
    mode: DecodeMode = "greedy",
    num_samples: int = 1,
    generator: torch.Generator | None = None,
) -> MethodEval:
    """Evaluate the policy on a fixed instance set (no gradient)."""

    policy.eval()
    costs: list[float] = []
    feasible: list[bool] = []
    routes: list[int] = []
    with torch.no_grad():
        for instance in instances:
            rollout = policy.rollout(
                instance,
                partition_fn(instance),
                num_samples=num_samples,
                mode=mode,
                generator=generator,
            )
            solution = rollout.best_solution
            metrics = evaluate_solution(instance, solution)
            costs.append(metrics.cost)
            feasible.append(metrics.feasible)
            routes.append(metrics.num_routes)
    return _summarize(name, costs, feasible, routes)


def evaluate_constructor(
    instances: list[VRPInstance],
    partition_fn: PartitionFn,
    *,
    name: str,
) -> MethodEval:
    """Evaluate a partition + nearest-neighbour heuristic baseline."""

    costs: list[float] = []
    feasible: list[bool] = []
    routes: list[int] = []
    for instance in instances:
        solution = compose_nearest_neighbor(instance, partition_fn(instance))
        metrics = evaluate_solution(instance, solution)
        costs.append(metrics.cost)
        feasible.append(metrics.feasible)
        routes.append(metrics.num_routes)
    return _summarize(name, costs, feasible, routes)


def evaluate_pyvrp(
    paths: list[Path],
    *,
    time_limit: float,
    seed: int = 0,
    name: str = "pyvrp",
) -> MethodEval:
    """Run PyVRP on each CVRPLIB file with a fixed time budget.

    Read distances with EUC_2D rounding, then report the best solution's distance
    and feasibility. At a short budget the solver may return an infeasible result.
    Reference costs must use the same distance convention for gaps to be meaningful.
    """

    from pyvrp import Model, read
    from pyvrp.stop import MaxRuntime

    costs: list[float] = []
    feasible: list[bool] = []
    routes: list[int] = []
    for path in paths:
        data = read(str(path), round_func="round")
        result = Model.from_data(data).solve(MaxRuntime(time_limit), seed=seed, display=False)
        best = result.best
        costs.append(float(best.distance()))
        feasible.append(bool(result.is_feasible()))
        routes.append(int(best.num_routes()))
    return _summarize(name, costs, feasible, routes)


def paired_gap(method: MethodEval, reference: MethodEval) -> PairedGap:
    """Per-instance gap of ``method`` over ``reference`` (lower cost is better)."""

    if len(method.costs) != len(reference.costs):
        raise ValueError("methods must be evaluated on the same instances")
    if not method.costs:
        raise ValueError("no instances to compare")
    gaps = [
        100.0 * (m - r) / r for m, r in zip(method.costs, reference.costs, strict=True) if r > 0
    ]
    wins = sum(m < r for m, r in zip(method.costs, reference.costs, strict=True))
    return PairedGap(
        method=method.name,
        reference=reference.name,
        mean_gap_pct=sum(gaps) / len(gaps) if gaps else 0.0,
        win_rate=wins / len(method.costs),
    )


def _summarize(
    name: str, costs: list[float], feasible: list[bool], routes: list[int]
) -> MethodEval:
    n = max(len(costs), 1)
    return MethodEval(
        name=name,
        mean_cost=sum(costs) / n,
        feasible_rate=sum(feasible) / n,
        mean_routes=sum(routes) / n,
        costs=tuple(costs),
    )
