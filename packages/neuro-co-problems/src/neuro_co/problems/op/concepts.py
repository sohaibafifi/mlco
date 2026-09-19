"""OP (orienteering) concept extractors (core `OPState`).

Node 0 = depot; `state.prize[B, N]` per-node reward (depot = 0);
`state.coords[B, N, 2]`. Each concept returns `[B, N]` labels in
`{0, 1, -1}` (`-1` = depot), or `None` when the field is absent.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def _mark_depot(labels: Tensor) -> Tensor:
    out = labels.clone()
    out[:, 0] = -1
    return out


def _customer_distances(coords: Tensor) -> Tensor:
    depot = coords[:, 0:1, :]
    return (coords - depot).norm(dim=-1)  # [B, N], dist[:, 0] = 0


def _above_median_customers(values: Tensor) -> Tensor:
    threshold = values[:, 1:].median()
    return _mark_depot((values > threshold).long())


def high_prize(state: Any) -> Tensor | None:
    prize = getattr(state, "prize", None)
    if prize is None or prize.ndim != 2:
        return None
    return _above_median_customers(prize)


def far_from_depot(state: Any) -> Tensor | None:
    coords = getattr(state, "coords", None)
    if coords is None or coords.ndim != 3:
        return None
    return _above_median_customers(_customer_distances(coords))


def prize_over_distance(state: Any) -> Tensor | None:
    """Greedy-pick score: prize / (distance + eps). High = visit early."""
    prize = getattr(state, "prize", None)
    coords = getattr(state, "coords", None)
    if prize is None or coords is None:
        return None
    score = prize / (_customer_distances(coords) + 1e-6)
    return _above_median_customers(score)


def low_prize_far(state: Any) -> Tensor | None:
    """Skip candidates: prize below median AND distance above median."""
    prize = getattr(state, "prize", None)
    coords = getattr(state, "coords", None)
    if prize is None or coords is None:
        return None
    dist = _customer_distances(coords)
    p_thr = prize[:, 1:].median()
    d_thr = dist[:, 1:].median()
    skip = (prize < p_thr) & (dist > d_thr)
    return _mark_depot(skip.long())


CONCEPTS = {
    "high_prize": high_prize,
    "far_from_depot": far_from_depot,
    "prize_over_distance": prize_over_distance,
    "low_prize_far": low_prize_far,
}
