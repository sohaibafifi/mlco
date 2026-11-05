"""Dedicated runners for the shared-protocol experiment stages."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json, write_run
from .training import TrainingConfig, _evaluate, array_digest, initial_weights, make_trainer


def _save(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _nearest_neighbour(
    coords: np.ndarray, demands: np.ndarray | None, capacity: float
) -> tuple[list[list[int]], float]:
    """Construct a deterministic feasible route, using node index to break ties."""
    remaining = set(range(1, len(coords)))
    current, load, cost = 0, capacity, 0.0
    routes: list[list[int]] = []
    route: list[int] = []
    while remaining:
        allowed = [node for node in sorted(remaining) if demands is None or demands[node] <= load]
        if not allowed:
            if not route:
                raise ValueError("A customer demand exceeds vehicle capacity")
            cost += float(np.linalg.norm(coords[current] - coords[0]))
            routes.append(route)
            route, current, load = [], 0, capacity
            continue
        node = min(allowed, key=lambda idx: float(np.linalg.norm(coords[current] - coords[idx])))
        cost += float(np.linalg.norm(coords[current] - coords[node]))
        route.append(node)
        remaining.remove(node)
        if demands is not None:
            load -= float(demands[node])
        current = node
    cost += float(np.linalg.norm(coords[current] - coords[0]))
    routes.append(route)
    return routes, cost


def run_baseline(
    output: Path,
    config: TrainingConfig,
    backend: str,
    dataset: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    with np.load(dataset, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    routes, costs = [], []
    for idx, coords in enumerate(data["coords"]):
        solution, cost = _nearest_neighbour(
            coords.astype(np.float64),
            data["demand"][idx] if config.problem == "cvrp" else None,
            config.capacity,
        )
        routes.append(solution)
        costs.append(cost)
    payload = {
        "schema": "neuro-co-baseline/v1",
        "method": "nearest_neighbour",
        "reference_kind": "heuristic",
        "problem": config.problem,
        "seed": config.seed,
        "num_instances": len(costs),
        "test_sha256": array_digest(data),
        "distance_convention": "euclidean_unrounded",
        "costs": costs,
        "routes": routes,
        "average_cost": float(np.mean(costs)),
        "duration_s": time.perf_counter() - started,
    }
    _save(output / "baseline.json", payload)
    write_run(
        output,
        kind="baseline",
        problem=config.problem,
        seed=config.seed,
        backend=backend,
        config={**asdict(config), "method": "nearest_neighbour"},
        metrics={key: payload[key] for key in ("average_cost", "duration_s", "num_instances")},
        artifacts={"baseline": "baseline.json", "dataset": dataset.resolve()},
    )
    return payload


def run_evaluation(
    output: Path,
    config: TrainingConfig,
    backend: str,
    device: str,
    checkpoint: Path,
    dataset: Path,
    baseline: Path | None = None,
) -> dict[str, Any]:
    trainer = make_trainer(config, backend, device, initial_weights(config))
    step = (
        trainer.load(checkpoint, resume=False) if backend == "torch" else trainer.load(checkpoint)
    )
    with np.load(dataset, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    costs, first_s = _evaluate(trainer, data)
    warm_costs, warm_s = _evaluate(trainer, data)
    if not np.allclose(costs, warm_costs):
        raise RuntimeError("Repeated evaluation changed costs")
    payload = {
        "schema": "neuro-co-evaluation/v1",
        "backend": backend,
        "problem": config.problem,
        "seed": config.seed,
        "checkpoint": str(checkpoint),
        "step": step,
        "num_instances": len(costs),
        "test_sha256": array_digest(data),
        "costs": costs.tolist(),
        "greedy_tour_length": float(costs.mean()),
        "first_call_s": first_s,
        "warm_call_s": warm_s,
    }
    metrics = {
        "average_cost": payload["greedy_tour_length"],
        "first_call_s": first_s,
        "warm_call_s": warm_s,
        "num_instances": len(costs),
    }
    if baseline is not None:
        reference = json.loads(baseline.read_text())
        if reference["test_sha256"] != payload["test_sha256"]:
            raise ValueError("Baseline and policy evaluation use different test instances")
        denominator = np.asarray(reference["costs"], dtype=np.float64)
        gap = float(np.mean((costs - denominator) / denominator) * 100)
        payload["baseline_method"] = reference["method"]
        payload["gap_to_baseline_percent"] = gap
        metrics["gap_to_baseline_percent"] = gap
    _save(output / "evaluation.json", payload)
    write_run(
        output,
        kind="eval",
        problem=config.problem,
        seed=config.seed,
        backend=backend,
        config=asdict(config),
        metrics=metrics,
        artifacts={
            "evaluation": "evaluation.json",
            "checkpoint": checkpoint.resolve(),
            "dataset": dataset.resolve(),
            **({"baseline": baseline.resolve()} if baseline is not None else {}),
        },
    )
    return payload


def run_explanation(
    output: Path,
    config: TrainingConfig,
    backend: str,
    device: str,
    checkpoint: Path,
    options: dict[str, Any],
) -> dict[str, Any]:
    from neuro_co.probe import explain_policy

    trainer = make_trainer(config, backend, device, initial_weights(config))
    trainer.load(checkpoint, resume=False)
    started = time.perf_counter()
    payload = explain_policy(
        trainer.model,
        trainer.env,
        ckpt_path=checkpoint,
        output_dir=output,
        seed=config.test_seed,
        device=device,
        **options,
    )
    metrics: dict[str, Any] = {"duration_s": time.perf_counter() - started}
    for section, key, name in (
        ("deletion", "mean_flip_rate", "deletion.mean_flip_rate"),
        ("sufficiency", "mean_keep_rate", "sufficiency.mean_keep_rate"),
        ("sanity_check", "mean_jaccard", "sanity.mean_jaccard"),
        ("sanity_check", "chance_jaccard", "sanity.chance_jaccard"),
    ):
        if section in payload["faithfulness"]:
            metrics[name] = payload["faithfulness"][section][key]
    write_run(
        output,
        kind="explain",
        problem=config.problem,
        seed=config.seed,
        backend=backend,
        config={**asdict(config), **options},
        metrics=metrics,
        artifacts={
            "explanation": "explanation.json",
            "instances": "instances.pt",
            "checkpoint": checkpoint.resolve(),
        },
    )
    return payload
