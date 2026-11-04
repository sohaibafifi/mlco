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
    return _dihedral(coords, N_DIHEDRAL)


def _dihedral(coords: Tensor, n_aug: int) -> Tensor:
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
    stacked = torch.stack([torch.stack([a, b], dim=-1) for a, b in variants[:n_aug]], dim=0)
    return stacked.reshape(n_aug * coords.shape[0], *coords.shape[1:])


def augment_state(state: State, n_aug: int) -> State:
    """Replicate `state` `n_aug` times (aug-major) with dihedral-transformed
    coords. Non-coord fields are tiled unchanged.

    `n_aug` must be 1 (no-op), 2, 4, or 8. Subsets use the first transforms
    in :func:`dihedral8`, so candidate sets are nested. State must expose
    a `coords` field of shape `(b, n, 2)`.
    """
    if n_aug == 1:
        return state
    if n_aug not in (2, 4, N_DIHEDRAL):
        raise ValueError(f"n_aug must be 1, 2, 4, or {N_DIHEDRAL}, got {n_aug}")
    coords = getattr(state, "coords", None)
    if coords is None:
        raise AttributeError("augment_state requires a `coords` field on the state")

    def _tile(t):
        if not isinstance(t, torch.Tensor):
            return t
        # Block tile (aug-major): [batch, batch, ...]; matches dihedral8 layout.
        return t.repeat(n_aug, *([1] * (t.dim() - 1)))

    tiled = pytree.tree_map(_tile, state)
    return tiled.replace(coords=_dihedral(coords, n_aug))


def best_over_aug(reward: Float[Tensor, "b8"], n_aug: int, batch: int) -> Float[Tensor, "b"]:
    """Reduce aug-major rewards `(n_aug * batch,)` to per-instance best `(batch,)`."""
    if n_aug == 1:
        return reward
    return reward.view(n_aug, batch).max(dim=0).values


__all__ = ["N_DIHEDRAL", "augment_state", "best_over_aug", "dihedral8"]
