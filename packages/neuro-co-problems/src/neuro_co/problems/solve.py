"""Resolve supplied instances through one checked neural or classical interface."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any, cast

import numpy as np

from neuro_co.core.result import SolveResult, SolveStatus

from .instances import RoutingInstance, load_instance
from .validation import validate_routes

_ROUTING = frozenset({"tsp", "cvrp", "cvrptw", "vrptw"})
_METHODS = frozenset({"neural", "nearest_neighbour", "pyvrp", "ortools", "cpsat", "lp"})


def _options(seed: int, max_runtime: float, backend: str) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if (
        isinstance(max_runtime, bool)
        or not isinstance(max_runtime, (int, float))
        or not math.isfinite(max_runtime)
        or max_runtime <= 0
    ):
        raise ValueError("max_runtime must be finite and positive")
    if backend not in {"torch", "jax"}:
        raise ValueError("backend must be torch or jax")


def _metadata(instance: RoutingInstance, seed: int) -> dict[str, Any]:
    fingerprint = hashlib.sha256()
    fingerprint.update(
        json.dumps(
            {
                "problem": instance.problem,
                "capacity": instance.capacity,
                "distance_convention": instance.distance_convention,
                "node_ids": instance.node_ids,
            },
            sort_keys=True,
        ).encode()
    )
    for key in ("coords", "demand", "time_windows", "service_time", "cost_matrix"):
        value = getattr(instance, key)
        if value is not None:
            fingerprint.update(key.encode())
            fingerprint.update(str(value.shape).encode())
            fingerprint.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return {
        "instance_name": instance.name,
        "source": str(instance.metadata.get("source", "memory")),
        "instance_sha256": fingerprint.hexdigest(),
        "distance_convention": instance.distance_convention,
        "node_ids": list(instance.node_ids or range(len(instance.coords))),
        "seed": seed,
        "normalization": "none; coordinates and demands are supplied in input units",
    }


def _checked(
    result: SolveResult, instance: RoutingInstance, started: float, seed: int
) -> SolveResult:
    """Never accept a solver's feasibility or reported cost without replaying routes."""
    metadata = {**result.metadata, **_metadata(instance, seed)}
    if result.solution is None:
        return replace(result, runtime_s=time.perf_counter() - started, metadata=metadata)
    if "routes" not in result.solution:
        raise ValueError("Routing solver returned no routes in its solution")
    routes = result.solution["routes"]
    report = validate_routes(instance, routes)
    solution = {**result.solution, "node_ids": metadata["node_ids"]}
    status = result.status
    objective, bound, gap = report.objective, result.bound, result.gap
    if not report.feasible:
        status, objective, bound, gap = SolveStatus.INVALID, None, None, None
    elif status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
        status = SolveStatus.FEASIBLE
    # A bound from a rounded/scaled objective cannot be applied to a different
    # objective reconstructed from the original instance.
    if (
        objective is not None
        and result.objective is not None
        and not math.isclose(objective, result.objective, rel_tol=1e-10, abs_tol=1e-10)
    ):
        metadata["reported_objective"] = result.objective
        bound, gap = None, None
        if status == SolveStatus.OPTIMAL:
            status = SolveStatus.FEASIBLE
    return replace(
        result,
        status=status,
        solution=solution,
        objective=objective,
        bound=bound,
        gap=gap,
        feasible=report.feasible,
        validation=report,
        runtime_s=time.perf_counter() - started,
        metadata=metadata,
    )


def _nearest_neighbour(instance: RoutingInstance, seed: int, max_runtime: float) -> SolveResult:
    started = time.perf_counter()
    matrix = instance.distance_matrix()
    remaining = set(range(1, len(matrix)))
    demands = instance.demand
    capacity = instance.capacity
    if demands is not None and capacity is not None and np.any(demands[1:] > capacity):
        return SolveResult(
            instance.problem,
            "nearest_neighbour",
            SolveStatus.INFEASIBLE,
            solver_status="CUSTOMER_DEMAND_EXCEEDS_CAPACITY",
            metadata=_metadata(instance, seed),
            runtime_s=time.perf_counter() - started,
        )
    windows = instance.time_windows
    service = instance.service_time
    route: list[int] = []
    routes: list[list[int]] = []
    current = 0
    load = float(capacity) if capacity is not None else math.inf
    departure = (float(windows[0, 0]) if windows is not None else 0.0) + (
        float(service[0]) if service is not None else 0.0
    )
    clock = departure

    def possible(node: int) -> bool:
        if demands is not None and demands[node] > load + 1e-10:
            return False
        if windows is None:
            return True
        arrival = max(clock + matrix[current, node], windows[node, 0])
        departure = arrival + (float(service[node]) if service is not None else 0.0)
        return bool(
            arrival <= windows[node, 1] + 1e-10
            and departure + matrix[node, 0] <= windows[0, 1] + 1e-10
        )

    while remaining:
        if time.perf_counter() - started > max_runtime:
            return SolveResult(
                instance.problem,
                "nearest_neighbour",
                SolveStatus.UNKNOWN,
                solver_status="TIME_LIMIT_WITHOUT_COMPLETE_SOLUTION",
                runtime_s=time.perf_counter() - started,
                metadata={**_metadata(instance, seed), "unserved": sorted(remaining)},
            )
        allowed = [node for node in sorted(remaining) if possible(node)]
        if not allowed:
            if not route or instance.problem == "tsp":
                return SolveResult(
                    instance.problem,
                    "nearest_neighbour",
                    SolveStatus.UNKNOWN,
                    solver_status="HEURISTIC_STUCK",
                    runtime_s=time.perf_counter() - started,
                    metadata={**_metadata(instance, seed), "unserved": sorted(remaining)},
                )
            routes.append(route)
            route, current = [], 0
            load = float(capacity) if capacity is not None else math.inf
            clock = departure
            continue
        node = min(allowed, key=lambda item: (matrix[current, item], item))
        clock += float(matrix[current, node])
        if windows is not None:
            clock = max(clock, float(windows[node, 0]))
        if service is not None:
            clock += float(service[node])
        if demands is not None:
            load -= float(demands[node])
        route.append(node)
        current = node
        remaining.remove(node)
    if route:
        routes.append(route)
    return _checked(
        SolveResult(
            instance.problem,
            "nearest_neighbour",
            SolveStatus.FEASIBLE,
            solution={"routes": routes},
            solver_status="HEURISTIC_COMPLETE",
        ),
        instance,
        started,
        seed,
    )


def _saved_configuration(checkpoint: Path, problem: str, backend: str) -> dict[str, Any]:
    config_path = checkpoint.parent / "config.json"
    if not config_path.exists():
        return {}
    saved = json.loads(config_path.read_text())
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint config.json must contain an object")
    config = saved.get("training", saved.get("args", saved))
    if not isinstance(config, dict):
        raise ValueError("Checkpoint configuration must contain a training mapping")
    if config.get("problem") not in {None, problem}:
        raise ValueError("Checkpoint problem does not match the supplied instance")
    if saved.get("backend", config.get("backend", "torch")) != backend:
        raise ValueError("Checkpoint backend does not match the selected backend")
    return config


def _neural_torch(instance: RoutingInstance, checkpoint: Path, device: str) -> list[list[int]]:
    import torch

    from neuro_co.core.factory import make_env, make_model
    from neuro_co.core.inference import greedy_rollout_actions
    from neuro_co.problems.cvrp.env import CVRPState
    from neuro_co.problems.tsp.env import TSPState

    saved_config = _saved_configuration(checkpoint, instance.problem, "torch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Expected a checkpoint containing model weights and architecture metadata")
    arch = payload.get("arch")
    if not isinstance(arch, dict):
        arch = {
            key: saved_config[key]
            for key in ("backbone", "hidden_dim", "num_layers", "num_heads")
            if key in saved_config
        }
    required = {"backbone", "hidden_dim", "num_layers", "num_heads"}
    if set(arch) != required:
        raise ValueError("Checkpoint must provide complete, supported architecture metadata")
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    target = torch.device(device)
    nodes = len(instance.coords)
    env_options: dict[str, Any] = (
        {"capacity": instance.capacity} if instance.problem == "cvrp" else {}
    )
    env = make_env(
        instance.problem,
        size=nodes if instance.problem == "tsp" else nodes - 1,
        **env_options,
    )
    model = make_model(env, **arch).to(target)
    model.load_state_dict(payload["model"], strict=True)
    coords = torch.as_tensor(np.array(instance.coords), dtype=torch.float32, device=target)[None]
    if not bool(torch.isfinite(coords).all()):
        raise ValueError("Coordinates cannot be represented by the neural model's float32 dtype")
    visited = torch.zeros((1, nodes), dtype=torch.bool, device=target)
    index = torch.zeros(1, dtype=torch.long, device=target)
    length = torch.zeros(1, dtype=torch.float32, device=target)
    if instance.problem == "tsp":
        visited[:, 0] = True
        state = cast(Any, TSPState)(
            coords=coords,
            visited=visited,
            current=index,
            first=index.clone(),
            step_count=index.clone(),
            tour_length=length,
        )
    else:
        assert instance.capacity is not None and instance.demand is not None
        demand = torch.as_tensor(np.array(instance.demand), dtype=torch.float32, device=target)[
            None
        ]
        if not bool(torch.isfinite(demand).all()) or not bool(
            torch.isfinite(torch.tensor(instance.capacity, dtype=torch.float32))
        ):
            raise ValueError("Demand and capacity must be representable by float32")
        state = cast(Any, CVRPState)(
            coords=coords,
            demand=demand,
            visited=visited,
            current=index,
            remaining_capacity=torch.full((1,), float(instance.capacity), device=target),
            tour_length=length,
            step_count=index.clone(),
        )
    actions = greedy_rollout_actions(model, env, state).detach().cpu().numpy()[0]
    return _action_routes(actions, instance.problem)


def _action_routes(actions: Any, problem: str) -> list[list[int]]:
    selected = [int(action) for action in actions if int(action) >= 0]
    if problem == "tsp":
        return [selected]
    routes: list[list[int]] = []
    current: list[int] = []
    for action in selected:
        if action == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(action)
    if current:
        # Keep an unfinished return leg visible to the independent checker.
        raise RuntimeError("Neural CVRP rollout did not return to the depot")
    return routes


def _neural_jax(instance: RoutingInstance, checkpoint: Path, device: str) -> list[list[int]]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("Neural JAX solve requires neuro-co-core[jax]") from exc

    from neuro_co.core.env_registry import make_env
    from neuro_co.core.jax_backend.am import JaxAttentionModel
    from neuro_co.core.jax_backend.pomo import JaxPOMO
    from neuro_co.core.jax_backend.train import checkpoint_metadata, load_checkpoint

    saved = checkpoint_metadata(checkpoint)["config"]
    if saved.get("problem") != instance.problem:
        raise ValueError("Checkpoint problem does not match the supplied instance")
    _saved_configuration(checkpoint, instance.problem, "jax")
    platform, _, index = ("cpu" if device == "auto" else device).partition(":")
    if platform not in {"cpu", "cuda"}:
        raise ValueError("JAX solve supports cpu or cuda devices")
    try:
        device_index = int(index or 0)
    except ValueError as exc:
        raise ValueError("JAX device index must be a nonnegative integer") from exc
    if device_index < 0:
        raise ValueError("JAX device index must be a nonnegative integer")
    try:
        devices = jax.devices("gpu" if platform == "cuda" else "cpu")
    except RuntimeError as exc:
        raise ValueError(f"Requested JAX device {device!r} is unavailable") from exc
    if device_index >= len(devices):
        raise ValueError(f"Requested JAX device {device!r} is unavailable")
    target = devices[device_index]
    with jax.default_device(target):
        nodes = len(instance.coords)
        env_options: dict[str, Any] = (
            {"capacity": instance.capacity} if instance.problem == "cvrp" else {}
        )
        env = make_env(
            instance.problem,
            backend="jax",
            size=nodes if instance.problem == "tsp" else nodes - 1,
            **env_options,
        )
        model = JaxAttentionModel(
            in_dim=env.encoder_in_dim,
            hidden_dim=saved["hidden_dim"],
            num_layers=saved["num_layers"],
            num_heads=saved["num_heads"],
            dropout=0.0,
            precision=saved.get("precision", "fp32"),
        )
        algo = JaxPOMO(model, env, n_starts=1)
        state, _, _, _ = load_checkpoint(checkpoint, algo.init(jax.random.key(0)))
        coords = jnp.asarray(instance.coords, dtype=jnp.float32)[None]
        if not bool(jnp.isfinite(coords).all()):
            raise ValueError(
                "Coordinates cannot be represented by the neural model's float32 dtype"
            )
        data = (
            env.reset_from_coords(coords)
            if instance.problem == "tsp"
            else env.reset_from_data(coords, jnp.asarray(instance.demand, dtype=jnp.float32)[None])
        )
        actions = jax.jit(algo.greedy_rollout_actions)(state.params, data)
        return _action_routes(np.asarray(jax.block_until_ready(actions))[0], instance.problem)


def _neural(
    instance: RoutingInstance, checkpoint: Path, backend: str, device: str, seed: int
) -> SolveResult:
    if instance.problem not in {"tsp", "cvrp"}:
        raise ValueError("Neural solve currently supports TSP and CVRP checkpoints")
    if instance.cost_matrix is not None or instance.distance_convention == "explicit":
        raise ValueError("Coordinate neural policies do not support explicit distance matrices")
    if instance.problem == "cvrp":
        assert instance.demand is not None and instance.capacity is not None
    if instance.problem == "cvrp" and np.any(
        cast(np.ndarray, instance.demand)[1:] > cast(float, instance.capacity)
    ):
        return SolveResult(
            instance.problem,
            "neural",
            SolveStatus.INFEASIBLE,
            solver_status="CUSTOMER_DEMAND_EXCEEDS_CAPACITY",
            metadata=_metadata(instance, seed),
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    started = time.perf_counter()
    routes = (
        _neural_torch(instance, checkpoint, device)
        if backend == "torch"
        else _neural_jax(instance, checkpoint, device)
    )
    result = SolveResult(
        instance.problem,
        "neural",
        SolveStatus.FEASIBLE,
        solution={"routes": routes},
        solver_status="GREEDY_COMPLETE",
        metadata={
            "backend": backend,
            "device": device,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "budget_scope": "max_runtime applies to classical search, not neural inference",
        },
    )
    return _checked(result, instance, started, seed)


def _legacy_data(instance: RoutingInstance) -> dict[str, Any]:
    import torch

    assert instance.capacity is not None
    return {
        "locs": torch.as_tensor(np.array(instance.coords)),
        "demand": torch.as_tensor(np.array(instance.demand)),
        "time_windows": torch.as_tensor(np.array(instance.time_windows)),
        "durations": torch.as_tensor(np.array(instance.service_time)),
        "vehicle_capacity": float(instance.capacity),
    }


def _classical(
    instance: RoutingInstance, method: str, seed: int, max_runtime: float
) -> SolveResult:
    started = time.perf_counter()
    if instance.problem == "cvrp" and method in {"pyvrp", "ortools"}:
        from importlib import import_module

        try:
            adapter = import_module(f"neuro_co.problems.cvrp.{method}")
            result = adapter.solve_instance_result(
                instance.coords,
                instance.demand,
                instance.capacity,
                seed=seed,
                max_runtime_s=max_runtime,
                cost_matrix=instance.distance_matrix(),
            )
        except ImportError as exc:
            raise ImportError(f"{method} solve requires neuro-co-problems[{method}]") from exc
    elif instance.problem == "cvrptw" and method in {"cpsat", "pyvrp"}:
        if instance.distance_convention != "euc_2d":
            raise ValueError("CVRPTW classical solve currently requires EUC_2D distances")
        from . import get_solver

        solver = get_solver("cvrptw", method)
        result = solver(
            _legacy_data(instance),
            max_runtime=max_runtime,
            return_result=True,
            cost_matrix=instance.distance_matrix(),
        )
    else:
        raise ValueError(f"Unsupported solve combination: {instance.problem}/{method}")
    if not isinstance(result, SolveResult):
        raise TypeError(f"{method} adapter did not return a SolveResult")
    return _checked(result, instance, started, seed)


def _generic_classical(
    data: Mapping[str, Any], problem: str, method: str, max_runtime: float, seed: int
) -> SolveResult:
    started = time.perf_counter()
    import torch

    from . import get_solver

    supported = {
        ("jssp", "cpsat"),
        ("fjsp", "cpsat"),
        ("op", "cpsat"),
        ("pdp", "ortools"),
        ("flp", "lp"),
    }
    if (problem, method) not in supported:
        raise ValueError(f"Unsupported solve combination: {problem}/{method}")
    values = {
        key: torch.as_tensor(np.asarray(value))
        if isinstance(value, (list, tuple, np.ndarray, Real))
        else value
        for key, value in data.items()
        if key != "problem"
    }
    result = get_solver(problem, method)(values, max_runtime=max_runtime, return_result=True)
    if not isinstance(result, SolveResult):
        raise TypeError(f"{method} adapter did not return a SolveResult")
    metadata = data.get("metadata", {})
    fingerprint = hashlib.sha256(problem.encode())
    for key, value in sorted(values.items()):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            fingerprint.update(key.encode())
            fingerprint.update(str((array.shape, array.dtype)).encode())
            fingerprint.update(np.ascontiguousarray(array).tobytes())
    return replace(
        result,
        metadata={
            **result.metadata,
            "solver_runtime_s": result.runtime_s,
            "instance_sha256": fingerprint.hexdigest(),
            "instance_name": str(data.get("name", "")),
            "requested_seed": seed,
            "solver_seed_not_configurable": True,
            "source": str(metadata.get("source", "memory"))
            if isinstance(metadata, dict)
            else "memory",
        },
        runtime_s=time.perf_counter() - started,
    )


def _file_input(path: Path, problem: str | None) -> RoutingInstance | Mapping[str, Any]:
    """Scheduling and other problems use the same JSON/NPZ entry point."""
    if path.suffix.lower() not in {".json", ".npz"}:
        return load_instance(path, problem=problem)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        for key in ("problem", "name", "schema"):
            if key in data and data[key].ndim == 0:
                data[key] = data[key].item()
    if not isinstance(data, dict):
        raise ValueError("Instance file must contain an object")
    declared = data.get("problem")
    if (
        problem is not None
        and declared is not None
        and problem != declared
        and {problem, declared} != {"cvrptw", "vrptw"}
    ):
        raise ValueError("Requested problem does not match the supplied instance")
    selected = problem or declared
    if selected in _ROUTING or selected is None:
        return load_instance(path, problem=problem)
    if data.get("schema", "neuro-co-instance/v1") != "neuro-co-instance/v1":
        raise ValueError("Unsupported instance schema")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Instance metadata must be an object")
    return {**data, "problem": selected, "metadata": {**metadata, "source": str(path)}}


def solve(
    instance: RoutingInstance | Mapping[str, Any] | str | Path,
    *,
    method: str = "neural",
    checkpoint: str | Path | None = None,
    problem: str | None = None,
    device: str = "cpu",
    backend: str = "torch",
    seed: int = 0,
    max_runtime: float = 5.0,
) -> SolveResult:
    """Solve one supplied instance and return a common, serializable result.

    Routing inputs may be an instance, a mapping, or JSON/NPZ/TSPLIB paths.
    Learned policies use their checkpoint and the supplied coordinates without
    sampling replacement instances or normalizing input units. Classical
    methods retain solver statuses, while routing incumbents are independently
    validated. ``max_runtime`` limits heuristic/classical search; neural
    inference runs to completion and records its measured runtime.
    """
    _options(seed, max_runtime, backend)
    method = {"nn": "nearest_neighbour", "nearest_neighbor": "nearest_neighbour"}.get(
        method, method
    )
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(_METHODS)}")
    if checkpoint is not None and method != "neural":
        raise ValueError("checkpoint is only accepted with method='neural'")
    if method != "neural" and backend != "torch":
        raise ValueError("backend selects neural inference; classical methods use backend='torch'")
    if isinstance(instance, (str, Path)):
        instance = _file_input(Path(instance), problem)
    if isinstance(instance, RoutingInstance):
        supplied = instance
    elif isinstance(instance, Mapping):
        key = problem or instance.get("problem")
        declared = instance.get("problem")

        def normalized(value: Any) -> Any:
            return "cvrptw" if value == "vrptw" else value

        if (
            problem is not None
            and declared is not None
            and normalized(problem) != normalized(declared)
        ):
            raise ValueError("Requested problem does not match the supplied instance")
        if key not in _ROUTING:
            if not isinstance(key, str):
                raise ValueError("A problem must be specified for the input instance")
            return _generic_classical(instance, key, method, max_runtime, seed)
        values = {key: value for key, value in instance.items() if key != "schema"}
        if instance.get("schema", "neuro-co-instance/v1") != "neuro-co-instance/v1":
            raise ValueError("Unsupported routing instance schema")
        supplied = RoutingInstance(**{**values, "problem": key})
    else:
        raise TypeError("instance must be a RoutingInstance, mapping, or file path")
    requested = "cvrptw" if problem == "vrptw" else problem
    if requested is not None and supplied.problem != requested:
        raise ValueError("Requested problem does not match the supplied instance")
    if method == "nearest_neighbour":
        return _nearest_neighbour(supplied, seed, max_runtime)
    if method == "neural":
        if checkpoint is None:
            raise ValueError("Neural solve requires a checkpoint")
        return _neural(supplied, Path(checkpoint), backend, device, seed)
    return _classical(supplied, method, seed, max_runtime)


__all__ = ["solve"]
