"""Dataset adapters for scale experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuro_co.core.factory import make_env

from .types import Solution, VRPInstance


def load_cvrplib_instance(path: str | Path) -> VRPInstance:
    """Parse a TSPLIB-format CVRP ``.vrp`` file (CVRPLIB X / XL sets).

    Node 1 is the depot, so CVRPLIB node ``i`` maps to instance index ``i - 1``
    and customer ``k`` (as written in the ``.sol`` files) maps to index ``k``.
    """

    coords: list[tuple[float, float]] = []
    demands: list[float] = []
    depots: list[int] = []
    header: dict[str, str] = {}
    section: str | None = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if line.endswith("SECTION"):
            section = line
            continue
        if section is None:
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()
            continue
        parts = line.split()
        if section == "NODE_COORD_SECTION":
            coords.append((float(parts[1]), float(parts[2])))
        elif section == "DEMAND_SECTION":
            demands.append(float(parts[1]))
        elif section == "DEPOT_SECTION" and parts[0] != "-1":
            depots.append(int(parts[0]))

    weight_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D")
    if weight_type != "EUC_2D":
        raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {weight_type!r}")
    if depots != [1]:
        raise ValueError(f"expected a single depot at node 1, got {depots}")
    if len(coords) != len(demands):
        raise ValueError("coord and demand sections disagree")

    return VRPInstance(
        coords=np.asarray(coords, dtype=np.float64),
        demand=np.asarray(demands, dtype=np.float64),
        capacity=float(header["CAPACITY"]),
        round_distances=True,
        name=header.get("NAME", Path(path).stem),
        metadata={"source": "cvrplib", "path": str(path)},
    )


def load_cvrplib_solution(path: str | Path) -> tuple[Solution, float | None]:
    """Parse a CVRPLIB ``.sol`` file into a solution and its reported cost."""

    routes: list[tuple[int, ...]] = []
    cost: float | None = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line.startswith("Route"):
            routes.append(tuple(int(x) for x in line.split(":", 1)[1].split()))
        elif line.lower().startswith("cost"):
            cost = float(line.split()[1])
    return Solution(routes=tuple(routes), metadata={"source": "cvrplib_bks"}), cost


def sample_clustered_instances(
    *,
    size: int,
    num_instances: int,
    seed: int = 0,
    capacity: float = 50.0,
    num_clusters: int = 0,
    spread: float = 0.04,
) -> list[VRPInstance]:
    """Gaussian-clustered CVRP instances (customers grouped around random centers).

    The uniform generator is the easy case for nearest-neighbour construction;
    clustered demand is where a spatial heuristic gets trapped. ``num_clusters=0``
    scales the centre count with the instance size.
    """

    if size < 1:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    centres = num_clusters or max(2, size // 100)
    out: list[VRPInstance] = []
    for i in range(num_instances):
        middles = rng.random((centres, 2))
        assign = rng.integers(0, centres, size)
        customers = np.clip(middles[assign] + rng.normal(0.0, spread, (size, 2)), 0.0, 1.0)
        coords = np.vstack([rng.random((1, 2)), customers])
        demand = np.zeros(size + 1)
        demand[1:] = rng.integers(1, 10, size)
        out.append(
            VRPInstance(
                coords=coords,
                demand=demand,
                capacity=capacity,
                name=f"clustered{size}_seed{seed}_{i}",
                metadata={"source": "clustered", "num_clusters": centres, "spread": spread},
            )
        )
    return out


def core_state_to_instance(
    state: Any, env: Any, *, batch_idx: int = 0, name: str = ""
) -> VRPInstance:
    """Convert a core CVRP/CVRPTW state into a `VRPInstance`."""

    coords = _to_numpy(state.coords[batch_idx])
    demand = _to_numpy(state.demand[batch_idx])
    tw_early = _maybe_numpy(getattr(state, "tw_early", None), batch_idx)
    tw_late = _maybe_numpy(getattr(state, "tw_late", None), batch_idx)
    service_time = None
    if tw_early is not None:
        service_time = np.zeros_like(demand)
    return VRPInstance(
        coords=coords,
        demand=demand,
        capacity=float(getattr(env, "capacity", 1.0)),
        tw_early=tw_early,
        tw_late=tw_late,
        service_time=service_time,
        name=name,
        metadata={"source": "neuro-co-core", "batch_idx": batch_idx},
    )


def sample_core_instances(
    problem: str,
    *,
    size: int,
    num_instances: int,
    seed: int = 0,
    **env_kwargs: Any,
) -> list[VRPInstance]:
    """Sample fixed instances from an existing core environment."""

    env = make_env(problem, size=size, **env_kwargs)
    state = env.reset(num_instances, generator=torch.Generator().manual_seed(seed))
    return [
        core_state_to_instance(state, env, batch_idx=i, name=f"{problem}{size}_seed{seed}_{i}")
        for i in range(num_instances)
    ]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _maybe_numpy(value: Any, batch_idx: int) -> np.ndarray | None:
    if value is None:
        return None
    return _to_numpy(value[batch_idx])
