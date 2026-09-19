"""Compile-friendly model base.

Plain `nn.Module`. No TensorDict, no Lightning. Use
`F.scaled_dot_product_attention` for attention. Mask via large negative
bias, not boolean indexing.
"""

from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch
from jaxtyping import Bool, Float
from torch import nn

from .state import State

NEG_INF: float = -1e9

S_contra = TypeVar("S_contra", bound=State, contravariant=True)


@runtime_checkable
class Policy(Protocol, Generic[S_contra]):
    """Maps state to action logits."""

    def __call__(
        self,
        state: S_contra,
        mask: Bool[torch.Tensor, "b a"],
    ) -> Float[torch.Tensor, "b a"]: ...


def apply_mask(
    logits: Float[torch.Tensor, "b a"],
    mask: Bool[torch.Tensor, "b a"],
) -> Float[torch.Tensor, "b a"]:
    """Replace masked positions with large negative bias.

    Boolean indexing breaks `torch.compile` graph; `where` stays in graph.
    """
    return torch.where(mask, logits, logits.new_full((), NEG_INF))


class PolicyModule(nn.Module):
    """Base class for parametric policies. Implement `forward(state, mask)`."""

    def forward(
        self,
        state: State,
        mask: Bool[torch.Tensor, "b a"],
    ) -> Float[torch.Tensor, "b a"]:
        raise NotImplementedError
