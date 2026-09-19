"""Instance augmentation for eval-time inference boosting (POMO trick).

The 8 dihedral transforms of the unit square preserve all pairwise
euclidean distances, so the optimal tour length is invariant. The neural
policy is NOT invariant, so each transform yields a different greedy tour;
taking the best across transforms is a free quality boost at eval.

Used at eval only. Training uses POMO's multi-start instead.

Reference: Kwon et al. 2020 (POMO), §4 "instance augmentation".
"""

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils import _pytree as pytree

from .state import State

N_DIHEDRAL = 8


def dihedral8(coords: Float[Tensor, "b n 2"]) -> Float[Tensor, "b8 n 2"]:
    """Apply the 8 dihedral transforms. Output is aug-major:

        [aug0 over all b, aug1 over all b, ..., aug7 over all b]

    so reshape to `(8, b, n, 2)` recovers the transform axis first.
    """
    x = coords[..., 0]
    y = coords[..., 1]
    variants = [
        (x, y),
        (y, x),
        (x, 1.0 - y),
        (y, 1.0 - x),
        (1.0 - x, y),
        (1.0 - y, x),
        (1.0 - x, 1.0 - y),
        (1.0 - y, 1.0 - x),
    ]
    stacked = torch.stack([torch.stack([a, b], dim=-1) for a, b in variants], dim=0)  # (8,b,n,2)
    return stacked.reshape(N_DIHEDRAL * coords.shape[0], *coords.shape[1:])


def augment_state(state: State, n_aug: int) -> State:
    """Replicate `state` `n_aug` times (aug-major) with dihedral-transformed
    coords. Non-coord fields are tiled unchanged.

    `n_aug` must be 1 (no-op) or 8. State must expose a `coords` field of
    shape `(b, n, 2)`: true for TSP/CVRP/CVRPTW.
    """
    if n_aug == 1:
        return state
    if n_aug != N_DIHEDRAL:
        raise ValueError(f"n_aug must be 1 or {N_DIHEDRAL}, got {n_aug}")
    coords = getattr(state, "coords", None)
    if coords is None:
        raise AttributeError("augment_state requires a `coords` field on the state")

    def _tile(t):
        if not isinstance(t, torch.Tensor):
            return t
        # Block tile (aug-major): [batch, batch, ...]; matches dihedral8 layout.
        return t.repeat(n_aug, *([1] * (t.dim() - 1)))

    tiled = pytree.tree_map(_tile, state)
    return tiled.replace(coords=dihedral8(coords))


def best_over_aug(reward: Float[Tensor, "b8"], n_aug: int, batch: int) -> Float[Tensor, "b"]:
    """Reduce aug-major rewards `(n_aug * batch,)` to per-instance best `(batch,)`."""
    if n_aug == 1:
        return reward
    return reward.view(n_aug, batch).max(dim=0).values


__all__ = ["N_DIHEDRAL", "augment_state", "best_over_aug", "dihedral8"]
