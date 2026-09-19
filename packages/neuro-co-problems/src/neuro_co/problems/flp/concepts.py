"""FLP (facility location) concept extractors.

the `FLPEnv` state:
  - `locs[B, N, 2]` candidate facility / customer coordinates.
  - `orig_distances[B, N, N]` pairwise distances.
  - `distances[B, N]` per-node distance to nearest already-chosen
    facility (infinity initially).
  - `to_choose[B]` cardinality budget k of facilities to open.

Concepts:
  - `central`         - node distance to centroid below median
    (geometric centre; likely first-opened facility).
  - `peripheral`      - node distance to centroid above median.
  - `dense_neighbour` - mean distance to 5 nearest neighbours
    below median (densely surrounded; high coverage potential).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _binary_above_median(values: Tensor) -> Tensor:
    threshold = values.median()
    return (values > threshold).long()


def _binary_below_median(values: Tensor) -> Tensor:
    threshold = values.median()
    return (values < threshold).long()


def _centroid_distance(td: Any) -> Tensor:
    locs = td["locs"]
    centroid = locs.mean(dim=1, keepdim=True)
    return (locs - centroid).norm(dim=-1)


def central(td: Any) -> Tensor | None:
    if "locs" not in td:
        return None
    locs = td["locs"]
    if locs.ndim != 3 or locs.shape[-1] != 2:
        return None
    return _binary_below_median(_centroid_distance(td))


def peripheral(td: Any) -> Tensor | None:
    if "locs" not in td:
        return None
    locs = td["locs"]
    if locs.ndim != 3 or locs.shape[-1] != 2:
        return None
    return _binary_above_median(_centroid_distance(td))


def dense_neighbour(td: Any) -> Tensor | None:
    if "orig_distances" not in td:
        return None
    d = td["orig_distances"]
    if d.ndim != 3:
        return None
    # mean of 5 smallest non-self distances per node
    k = min(5, d.shape[-1] - 1)
    eye = torch.eye(d.shape[-1], device=d.device).bool().unsqueeze(0)
    d_no_self = d.masked_fill(eye, float("inf"))
    nearest, _ = d_no_self.topk(k, dim=-1, largest=False)
    mean_near = nearest.mean(dim=-1)
    return _binary_below_median(mean_near)


CONCEPTS = {
    "central": central,
    "peripheral": peripheral,
    "dense_neighbour": dense_neighbour,
}
