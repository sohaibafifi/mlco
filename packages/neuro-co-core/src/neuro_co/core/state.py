"""Frozen-dataclass `State` base + pytree helpers.

Replaces TensorDict. Each problem defines its own `State` subclass with
shape-annotated tensor fields. Compile-friendly: static field set, no
dynamic key access.
"""

from dataclasses import dataclass, fields, replace
from typing import Self

import torch
from torch.utils import _pytree as pytree


@dataclass(frozen=True, slots=True)
class State:
    """Base for problem states. Subclass with tensor fields.

    Subclasses MUST:
      - declare every tensor field with a `jaxtyping` annotation
      - be registered as pytree via `register_state` decorator
    """

    def replace(self, **kw) -> Self:
        return replace(self, **kw)

    def to(self, device: torch.device | str) -> Self:
        moved = {f.name: _to(getattr(self, f.name), device) for f in fields(self)}
        return replace(self, **moved)

    def detach(self) -> Self:
        moved = {f.name: _detach(getattr(self, f.name)) for f in fields(self)}
        return replace(self, **moved)


def _to(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else x


def _detach(x):
    return x.detach() if isinstance(x, torch.Tensor) else x


def register_state(cls: type[State]) -> type[State]:
    """Class decorator to register a `State` subclass as a pytree node.

    Allows `torch.utils._pytree.tree_map` to traverse all tensor fields.
    """
    names = [f.name for f in fields(cls)]

    def flatten(s: State):
        return [getattr(s, n) for n in names], names

    def unflatten(values, context):
        return cls(**dict(zip(context, values, strict=True)))

    pytree.register_pytree_node(cls, flatten, unflatten)
    return cls
