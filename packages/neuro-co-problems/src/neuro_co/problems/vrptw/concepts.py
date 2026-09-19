"""CVRPTW concept extractors (core `CVRPTWState`).

Each returns `[B, N]` long labels in `{0, 1, -1}` (`-1` = depot, ignored by
the probe trainer), or `None` when the state lacks the needed field. Node 0
is the depot; customers are 1..N.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def _above_median(values: Tensor) -> Tensor:
    """Binary label: value above the customer (non-depot) median; depot = -1."""
    cust = values[:, 1:]
    thr = cust.median()
    labels = (values > thr).long()
    labels[:, 0] = -1
    return labels


def _below_median(values: Tensor) -> Tensor:
    cust = values[:, 1:]
    thr = cust.median()
    labels = (values < thr).long()
    labels[:, 0] = -1
    return labels


def tight_tw(state: Any) -> Tensor | None:
    """Time-window width below the customer median."""
    tw_early = getattr(state, "tw_early", None)
    tw_late = getattr(state, "tw_late", None)
    if tw_early is None or tw_late is None:
        return None
    return _below_median(tw_late - tw_early)


def high_demand(state: Any) -> Tensor | None:
    demand = getattr(state, "demand", None)
    if demand is None:
        return None
    return _above_median(demand)


def far_from_depot(state: Any) -> Tensor | None:
    coords = getattr(state, "coords", None)
    if coords is None:
        return None
    depot = coords[:, 0:1, :]
    return _above_median((coords - depot).norm(dim=-1))


CONCEPTS = {
    "tight_tw": tight_tw,
    "high_demand": high_demand,
    "far_from_depot": far_from_depot,
}
