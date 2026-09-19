"""Small benchmark drivers for decomposition baselines."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .compatibility import CompatibilityWeights
from .compose import compose_nearest_neighbor, compose_time_window_aware
from .datasets import sample_core_instances
from .metrics import evaluate_solution
from .partition import (
    capacity_aware_sweep_partition,
    feature_aware_partition,
    grid_partition,
    morton_partition,
    morton_refined_partition,
    sweep_partition,
)
from .partition_metrics import evaluate_partition
from .types import ScaleMetrics, VRPInstance

PartitionMethod = Literal[
    "sweep",
    "capacity_sweep",
    "grid",
    "morton",
    "morton_refined",
    "feature_aware",
]
ProblemName = Literal["cvrp", "cvrptw"]
LocalConstructor = Literal["nearest_neighbor", "time_window"]
RefinementScoreMode = Literal["compactness", "route", "hybrid"]


@dataclass(frozen=True, slots=True)
class PartitionBaselineConfig:
    method: PartitionMethod = "capacity_sweep"
    max_customers: int = 100
    capacity_fraction: float = 0.95
    repair_capacity: bool = True
    local_constructor: LocalConstructor = "nearest_neighbor"
    refinement_score_mode: RefinementScoreMode = "hybrid"
    refinement_hybrid_shortlist: int = 2
    feature_weights: CompatibilityWeights | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkGridConfig:
    """Configuration for repeatable partition baseline sweeps."""

    problem: ProblemName = "cvrp"
    sizes: tuple[int, ...] = (50, 100, 200)
    methods: tuple[PartitionMethod, ...] = (
        "sweep",
        "capacity_sweep",
        "grid",
        "morton",
        "morton_refined",
        "feature_aware",
    )
    seeds: tuple[int, ...] = (0,)
    num_instances: int = 1
    max_customers: int = 10
    capacity: float = 50.0
    capacity_fraction: float = 0.95
    repair_capacity: bool = True
    local_constructor: LocalConstructor = "nearest_neighbor"
    refinement_score_mode: RefinementScoreMode = "hybrid"
    refinement_hybrid_shortlist: int = 2
    horizon: float = 10.0
    window_width: float = 2.0
    feature_weights: CompatibilityWeights | None = None


def run_partition_baseline(
    instance: VRPInstance,
    cfg: PartitionBaselineConfig,
    *,
    reference_cost: float | None = None,
) -> tuple[ScaleMetrics, dict]:
    """Run a deterministic partition plus nearest-neighbor baseline."""

    t0 = time.perf_counter()
    if cfg.method == "sweep":
        partition = sweep_partition(instance, max_customers=cfg.max_customers)
    elif cfg.method == "capacity_sweep":
        partition = capacity_aware_sweep_partition(
            instance,
            max_customers=cfg.max_customers,
            capacity_fraction=cfg.capacity_fraction,
        )
    elif cfg.method == "grid":
        partition = grid_partition(instance, max_customers=cfg.max_customers)
    elif cfg.method == "morton":
        partition = morton_partition(
            instance,
            max_customers=cfg.max_customers,
            capacity_fraction=cfg.capacity_fraction,
        )
    elif cfg.method == "morton_refined":
        partition = morton_refined_partition(
            instance,
            max_customers=cfg.max_customers,
            capacity_fraction=cfg.capacity_fraction,
            score_mode=cfg.refinement_score_mode,
            hybrid_shortlist=cfg.refinement_hybrid_shortlist,
        )
    elif cfg.method == "feature_aware":
        partition = feature_aware_partition(
            instance,
            max_customers=cfg.max_customers,
            capacity_fraction=cfg.capacity_fraction,
            weights=cfg.feature_weights,
        )
    else:
        raise ValueError(f"unknown partition method {cfg.method!r}")
    partition_quality = evaluate_partition(
        instance,
        partition,
        max_demand=partition.metadata.get("max_demand"),
        weights=cfg.feature_weights,
    )
    if cfg.local_constructor == "nearest_neighbor":
        solution = compose_nearest_neighbor(
            instance,
            partition,
            repair_capacity=cfg.repair_capacity,
        )
    elif cfg.local_constructor == "time_window":
        solution = compose_time_window_aware(
            instance,
            partition,
            repair_capacity=cfg.repair_capacity,
        )
    else:
        raise ValueError(f"unknown local constructor {cfg.local_constructor!r}")
    runtime = time.perf_counter() - t0
    metrics = evaluate_solution(
        instance,
        solution,
        reference_cost=reference_cost,
        runtime_s=runtime,
    )
    payload = {
        "config": asdict(cfg),
        "partition": {
            "method": partition.method,
            "num_clusters": partition.num_clusters,
            "cluster_sizes": [len(c) for c in partition.clusters],
            "metadata": partition.metadata,
        },
        "partition_quality": asdict(partition_quality),
        "solution": {
            "num_routes": solution.num_routes,
            "route_sizes": [len(r) for r in solution.routes],
        },
        "metrics": asdict(metrics),
    }
    return metrics, payload


def iter_benchmark_records(cfg: BenchmarkGridConfig) -> Iterator[dict]:
    """Yield JSON-serializable records for a benchmark grid."""

    _validate_grid_config(cfg)
    for size in cfg.sizes:
        for seed in cfg.seeds:
            instances = sample_core_instances(
                cfg.problem,
                size=size,
                num_instances=cfg.num_instances,
                seed=seed,
                **_env_kwargs(cfg),
            )
            for instance_idx, instance in enumerate(instances):
                for method in cfg.methods:
                    baseline_cfg = PartitionBaselineConfig(
                        method=method,
                        max_customers=cfg.max_customers,
                        capacity_fraction=cfg.capacity_fraction,
                        repair_capacity=cfg.repair_capacity,
                        local_constructor=cfg.local_constructor,
                        refinement_score_mode=cfg.refinement_score_mode,
                        refinement_hybrid_shortlist=cfg.refinement_hybrid_shortlist,
                        feature_weights=cfg.feature_weights,
                    )
                    _, payload = run_partition_baseline(instance, baseline_cfg)
                    yield {
                        "problem": cfg.problem,
                        "size": size,
                        "seed": seed,
                        "instance_idx": instance_idx,
                        "instance": {
                            "name": instance.name,
                            "num_customers": instance.num_customers,
                            "capacity": instance.capacity,
                        },
                        **payload,
                    }


def write_benchmark_jsonl(path: Path, cfg: BenchmarkGridConfig) -> int:
    """Write benchmark records to JSONL and return the number of records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in iter_benchmark_records(cfg):
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def _env_kwargs(cfg: BenchmarkGridConfig) -> dict[str, float]:
    kwargs = {"capacity": cfg.capacity}
    if cfg.problem == "cvrptw":
        kwargs.update({"horizon": cfg.horizon, "window_width": cfg.window_width})
    return kwargs


def _validate_grid_config(cfg: BenchmarkGridConfig) -> None:
    if not cfg.sizes:
        raise ValueError("sizes must not be empty")
    if not cfg.methods:
        raise ValueError("methods must not be empty")
    if not cfg.seeds:
        raise ValueError("seeds must not be empty")
    if cfg.num_instances <= 0:
        raise ValueError("num_instances must be positive")
    if cfg.max_customers <= 0:
        raise ValueError("max_customers must be positive")
    if cfg.capacity <= 0:
        raise ValueError("capacity must be positive")
    if not 0 < cfg.capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    if cfg.refinement_hybrid_shortlist <= 0:
        raise ValueError("refinement_hybrid_shortlist must be positive")
