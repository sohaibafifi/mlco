"""Validated, dependency-light routing instances and local file adapters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and shaped {shape}, got {result.shape}")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class RoutingInstance:
    """One routing instance, with internal origin/depot index zero.

    ``node_ids`` maps internal indices back to the identifiers in the input.
    Routing solutions contain internal customer indices only; both origin legs
    are implicit. Coordinates are retained in the original input units.
    """

    problem: str
    coords: np.ndarray
    demand: np.ndarray | None = None
    capacity: float | None = None
    time_windows: np.ndarray | None = None
    service_time: np.ndarray | None = None
    cost_matrix: np.ndarray | None = None
    distance_convention: str = "euclidean"
    node_ids: tuple[int, ...] | None = None
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problem = str(self.problem).lower()
        problem = "cvrptw" if problem == "vrptw" else problem
        if problem not in {"tsp", "cvrp", "cvrptw"}:
            raise ValueError("RoutingInstance supports tsp, cvrp, and cvrptw")
        object.__setattr__(self, "problem", problem)
        try:
            coords = np.asarray(self.coords, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("coords must contain numeric values") from exc
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
            raise ValueError("coords must have shape (nodes, 2) with at least two nodes")
        count = len(coords)
        object.__setattr__(self, "coords", _array(coords, "coords", (count, 2)))
        convention = str(self.distance_convention).lower()
        if convention not in {"euclidean", "euc_2d", "explicit"}:
            raise ValueError("distance_convention must be euclidean, euc_2d, or explicit")
        if (self.cost_matrix is not None) != (convention == "explicit"):
            raise ValueError("cost_matrix requires distance_convention='explicit' and vice versa")
        object.__setattr__(self, "distance_convention", convention)
        if self.cost_matrix is not None:
            matrix = _array(self.cost_matrix, "cost_matrix", (count, count))
            if np.any(matrix < 0) or np.any(np.diag(matrix) != 0):
                raise ValueError("cost_matrix must be nonnegative with a zero diagonal")
            object.__setattr__(self, "cost_matrix", matrix)
        if problem == "tsp":
            if any(
                value is not None
                for value in (self.demand, self.capacity, self.time_windows, self.service_time)
            ):
                raise ValueError(
                    "TSP instances cannot declare capacity, demand, or time-window constraints"
                )
        else:
            if self.demand is None:
                raise ValueError("CVRP/CVRPTW instances require depot-first demand")
            demand = _array(self.demand, "demand", (count,))
            if np.any(demand < 0) or demand[0] != 0:
                raise ValueError("demand must be nonnegative and zero at the depot")
            object.__setattr__(self, "demand", demand)
            if (
                isinstance(self.capacity, bool)
                or not isinstance(self.capacity, Real)
                or not math.isfinite(self.capacity)
                or self.capacity <= 0
            ):
                raise ValueError("capacity must be a finite positive number")
            object.__setattr__(self, "capacity", float(self.capacity))
        if problem == "cvrp" and (self.time_windows is not None or self.service_time is not None):
            raise ValueError("Use problem='cvrptw' when declaring time windows or service times")
        if problem == "cvrptw":
            if self.time_windows is None:
                raise ValueError("CVRPTW requires time_windows for every node including the depot")
            windows = _array(self.time_windows, "time_windows", (count, 2))
            if np.any(windows < 0) or np.any(windows[:, 0] > windows[:, 1]):
                raise ValueError("time_windows must satisfy 0 <= earliest <= latest")
            services = _array(
                np.zeros(count) if self.service_time is None else self.service_time,
                "service_time",
                (count,),
            )
            if np.any(services < 0):
                raise ValueError("service_time must be nonnegative")
            object.__setattr__(self, "time_windows", windows)
            object.__setattr__(self, "service_time", services)
        node_ids = tuple(range(count)) if self.node_ids is None else tuple(self.node_ids)
        if (
            len(node_ids) != count
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in node_ids)
            or len(set(node_ids)) != count
        ):
            raise ValueError("node_ids must contain one distinct integer per node")
        object.__setattr__(self, "node_ids", tuple(int(item) for item in node_ids))
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_nodes(self) -> int:
        return len(self.coords)

    def distance_matrix(self) -> np.ndarray:
        if self.cost_matrix is not None:
            return self.cost_matrix.copy()
        matrix = np.linalg.norm(self.coords[:, None] - self.coords[None, :], axis=-1)
        if not np.isfinite(matrix).all():
            raise ValueError("Coordinates produce nonfinite travel distances")
        if self.distance_convention == "euc_2d":
            matrix = np.floor(matrix + 0.5)
        return matrix


def _mapping_instance(data: dict[str, Any], path: Path, problem: str | None) -> RoutingInstance:
    allowed = {item.name for item in fields(RoutingInstance)}
    unknown = set(data) - allowed - {"schema"}
    if unknown:
        raise ValueError(f"Unknown routing instance fields: {', '.join(sorted(unknown))}")
    if "schema" in data and data["schema"] != "neuro-co-instance/v1":
        raise ValueError("Unsupported routing instance schema")
    data = {key: value for key, value in data.items() if key != "schema"}
    if problem is not None:
        normalized = "cvrptw" if problem.lower() == "vrptw" else problem.lower()
        declared = data.get("problem")
        if (
            declared is not None
            and ("cvrptw" if str(declared).lower() == "vrptw" else str(declared).lower())
            != normalized
        ):
            raise ValueError("Requested problem differs from the instance's declared problem")
        data["problem"] = normalized
    if "problem" not in data:
        raise ValueError("Instance must declare problem, or supply problem explicitly")
    data.setdefault("name", path.stem)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    data["metadata"] = {**metadata, "source": str(path)}
    try:
        return RoutingInstance(**data)
    except TypeError as exc:
        raise ValueError(f"Missing or invalid routing instance fields: {exc}") from exc


def load_instance(path: str | Path, *, problem: str | None = None) -> RoutingInstance:
    """Load one JSON, NPZ, TSPLIB TSP, or CVRPLIB CVRP instance."""
    path = Path(path)
    if path.suffix.lower() in {".vrp", ".tsp"}:
        instance = _load_tsplib(path)
        if problem is not None and instance.problem != problem.lower():
            raise ValueError("Requested problem differs from the TSPLIB instance type")
        return instance
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid JSON instance: {path}") from exc
        if not isinstance(data, dict):
            raise ValueError("A routing instance JSON must be an object")
    elif path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
        except (ValueError, EOFError) as exc:
            raise ValueError(f"Invalid or object-bearing NPZ instance: {path}") from exc
        for key in ("problem", "name", "capacity", "distance_convention", "schema"):
            if key in data and data[key].ndim == 0:
                data[key] = data[key].item()
    else:
        raise ValueError("Supported instance extensions are .json, .npz, .tsp, and .vrp")
    return _mapping_instance(data, path, problem)


def _load_tsplib(path: Path) -> RoutingInstance:
    headers: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    section: str | None = None
    supported_sections = {
        "NODE_COORD_SECTION",
        "DEMAND_SECTION",
        "DEPOT_SECTION",
        "EDGE_WEIGHT_SECTION",
    }
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper() == "EOF":
            break
        if line.upper().endswith("_SECTION"):
            section = line.upper()
            if section not in supported_sections or section in sections:
                raise ValueError(f"Unsupported or repeated TSPLIB section: {section}")
            sections[section] = []
        elif section is not None:
            sections[section].append(line)
        else:
            parts = line.split(":", 1) if ":" in line else line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid TSPLIB header: {line}")
            key, value = (part.strip() for part in parts)
            key = key.upper()
            if key in headers and key != "COMMENT":
                raise ValueError(f"Duplicate TSPLIB header {key}")
            headers[key] = value
    try:
        count = int(headers["DIMENSION"])
    except (KeyError, ValueError) as exc:
        raise ValueError("TSPLIB DIMENSION must be an integer") from exc
    if count < 2:
        raise ValueError("TSPLIB DIMENSION must be at least two")
    kind = headers.get("TYPE", "CVRP" if path.suffix.lower() == ".vrp" else "TSP").upper()
    if kind not in {"TSP", "CVRP"}:
        raise ValueError(f"Unsupported TSPLIB TYPE {kind!r}")
    coords: dict[int, tuple[float, float]] = {}
    demand: dict[int, float] = {}
    try:
        for line in sections.get("NODE_COORD_SECTION", []):
            tokens = line.split()
            if len(tokens) != 3:
                raise ValueError("NODE_COORD_SECTION rows require id, x, y")
            node = int(tokens[0])
            if node <= 0 or node in coords:
                raise ValueError("Coordinate node ids must be distinct positive integers")
            coords[node] = (float(tokens[1]), float(tokens[2]))
        for line in sections.get("DEMAND_SECTION", []):
            tokens = line.split()
            if len(tokens) != 2:
                raise ValueError("DEMAND_SECTION rows require id, demand")
            node = int(tokens[0])
            if node <= 0 or node in demand:
                raise ValueError("Demand node ids must be distinct positive integers")
            demand[node] = float(tokens[1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid TSPLIB node section: {exc}") from exc
    weight = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if weight not in {"EUC_2D", "EXPLICIT"}:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE {weight!r}")
    if coords and len(coords) != count:
        raise ValueError("NODE_COORD_SECTION does not match DIMENSION")
    if not coords and weight != "EXPLICIT":
        raise ValueError("EUC_2D requires NODE_COORD_SECTION")
    identifiers = sorted(coords) if coords else list(range(1, count + 1))
    depots = [
        int(token)
        for line in sections.get("DEPOT_SECTION", [])
        for token in line.split()
        if token != "-1"
    ]
    if kind == "CVRP" and (len(depots) != 1 or depots[0] not in identifiers):
        raise ValueError("CVRP requires one valid depot in DEPOT_SECTION")
    if kind == "TSP" and (depots or demand):
        raise ValueError("TSP cannot declare depot or demand sections")
    ordered = (
        ([depots[0]] + [node for node in identifiers if node != depots[0]])
        if depots
        else identifiers
    )
    kwargs: dict[str, Any] = {}
    if kind == "CVRP":
        if set(demand) != set(identifiers):
            raise ValueError("Demand ids must exactly match the coordinate/matrix node ids")
        try:
            kwargs["capacity"] = float(headers["CAPACITY"])
        except (KeyError, ValueError) as exc:
            raise ValueError("CVRP requires numeric CAPACITY") from exc
        kwargs["demand"] = np.array([demand[node] for node in ordered])
    if weight == "EXPLICIT":
        if headers.get("EDGE_WEIGHT_FORMAT", "").upper() != "FULL_MATRIX":
            raise ValueError("EXPLICIT currently requires EDGE_WEIGHT_FORMAT FULL_MATRIX")
        try:
            values = [
                float(token)
                for line in sections.get("EDGE_WEIGHT_SECTION", [])
                for token in line.split()
            ]
        except ValueError as exc:
            raise ValueError("EDGE_WEIGHT_SECTION must contain numeric values") from exc
        if len(values) != count * count:
            raise ValueError("FULL_MATRIX size does not match DIMENSION")
        permutation = [identifiers.index(node) for node in ordered]
        kwargs["cost_matrix"] = np.array(values).reshape(count, count)[
            np.ix_(permutation, permutation)
        ]
    elif "EDGE_WEIGHT_SECTION" in sections:
        raise ValueError("EUC_2D cannot include an explicit edge weight section")
    return RoutingInstance(
        problem=kind.lower(),
        coords=np.array([coords.get(node, (0.0, 0.0)) for node in ordered]),
        distance_convention="explicit" if weight == "EXPLICIT" else "euc_2d",
        node_ids=tuple(ordered),
        name=headers.get("NAME", path.stem),
        metadata={"source": str(path), "format": "tsplib", "headers": headers},
        **kwargs,
    )


__all__ = ["RoutingInstance", "load_instance"]
