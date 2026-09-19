"""Training algorithm protocol.

Concrete algorithms in algos/ own their optimizer, scheduler, and rollout loop.
They expose training and evaluation steps that return scalar metrics."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Algo(Protocol):
    """Training algorithm. Owns optimizer + rollout."""

    def train_step(self, rng: torch.Generator) -> dict[str, float]:
        """Run one optimization step. Return scalar metrics."""
        ...

    def eval_step(self, rng: torch.Generator) -> dict[str, float]:
        """Run one evaluation pass. Return scalar metrics."""
        ...
