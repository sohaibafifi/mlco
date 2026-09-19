"""PDP (pickup-delivery) concept extractors (core `PDPState`).

Layout: node 0 = depot, nodes 1..P = pickups, nodes P+1..2P = deliveries
(delivery `i` pairs with pickup `i - P`). Labels `[B, N]` in `{0, 1, -1}`
(`-1` = depot), or `None` when `coords` is absent.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _mark_depot(labels: Tensor) -> Tensor:
    out = labels.clone()
    out[:, 0] = -1
    return out


def _coords(state: Any) -> Tensor | None:
    c = getattr(state, "coords", None)
    if c is None or c.ndim != 3 or c.shape[-1] != 2:
        return None
    return c


def is_pickup(state: Any) -> Tensor | None:
    """1 iff node is a pickup (index 1..P); 0 deliveries; -1 depot."""
    coords = _coords(state)
    if coords is None:
        return None
    b, n, _ = coords.shape
    p = (n - 1) // 2
    labels = torch.zeros(b, n, dtype=torch.long, device=coords.device)
    labels[:, 1 : p + 1] = 1
    return _mark_depot(labels)


def is_delivery(state: Any) -> Tensor | None:
    pickup = is_pickup(state)
    if pickup is None:
        return None
    return _mark_depot((pickup == 0).long())


def far_paired_distance(state: Any) -> Tensor | None:
    """Pickup-delivery distance above the median pair (same label both ends)."""
    coords = _coords(state)
    if coords is None:
        return None
    b, n, _ = coords.shape
    p = (n - 1) // 2
    pair_dist = (coords[:, 1 : p + 1, :] - coords[:, p + 1 :, :]).norm(dim=-1)  # [B, P]
    pair_label = (pair_dist > pair_dist.median()).long()
    out = torch.zeros(b, n, dtype=torch.long, device=coords.device)
    out[:, 1 : p + 1] = pair_label
    out[:, p + 1 :] = pair_label
    return _mark_depot(out)


def far_from_depot(state: Any) -> Tensor | None:
    coords = _coords(state)
    if coords is None:
        return None
    dist = (coords - coords[:, 0:1, :]).norm(dim=-1)
    return _mark_depot((dist > dist[:, 1:].median()).long())


CONCEPTS = {
    "is_pickup": is_pickup,
    "is_delivery": is_delivery,
    "far_paired_distance": far_paired_distance,
    "far_from_depot": far_from_depot,
}
