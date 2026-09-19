"""Utilities for evaluating explanations after an executed decision prefix."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from neuro_co.attr.attribution._common import _decode_logp


@dataclass(frozen=True)
class PrefixRollout:
    """State and completion metadata after a greedy policy prefix."""

    state: Any
    completed: torch.Tensor
    completion_step: torch.Tensor
    requested_steps: int
    executed_steps: int

    @property
    def active(self) -> torch.Tensor:
        """Boolean CPU mask for instances still active after the prefix."""
        return ~self.completed


@dataclass(frozen=True)
class DecisionActivity:
    """Greedy actions and pre-decision activity for a rollout window."""

    active: torch.Tensor
    actions: torch.Tensor


def select_state(state: Any, index: torch.Tensor) -> Any:
    """Select a batch subset from a frozen dataclass state."""
    selected = {
        field.name: getattr(state, field.name)[index.to(getattr(state, field.name).device)]
        for field in fields(state)
    }
    return state.replace(**selected)


def greedy_prefix(
    policy: Any,
    env: Any,
    state: Any,
    *,
    steps: int,
) -> PrefixRollout:
    """Execute ``steps`` greedy decisions and retain terminal metadata.

    Completed rows remain in the batched state while other rows finish their
    prefix. Callers must use ``active`` and ``select_state`` before evaluating
    explanations. Keeping the original batch until the end preserves stable
    instance identifiers across phase-specific runs.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")

    device = next(policy.parameters()).device
    current = state.to(device)
    batch_size = int(env.build_features(current).shape[0])
    completed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    completion_step = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    executed = 0

    policy.eval()
    with torch.no_grad():
        for step in range(steps):
            action = _decode_logp(
                policy,
                env,
                current,
                env.build_features(current),
            ).argmax(dim=-1)
            current, _, done = env.step(current, action)
            newly_completed = (~completed) & done
            completion_step = torch.where(
                newly_completed,
                torch.full_like(completion_step, step + 1),
                completion_step,
            )
            completed |= done
            executed = step + 1
            if bool(completed.all()):
                break

    return PrefixRollout(
        state=current,
        completed=completed.cpu(),
        completion_step=completion_step.cpu(),
        requested_steps=int(steps),
        executed_steps=int(executed),
    )


def decision_activity(
    policy: Any,
    env: Any,
    state: Any,
    *,
    max_steps: int,
) -> DecisionActivity:
    """Return which rows are active before each greedy decision."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    device = next(policy.parameters()).device
    current = state.to(device)
    batch_size = int(env.build_features(current).shape[0])
    completed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    active_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []

    policy.eval()
    with torch.no_grad():
        for _ in range(min(int(max_steps), int(env.max_steps(current)))):
            active_steps.append((~completed).cpu())
            action = _decode_logp(
                policy,
                env,
                current,
                env.build_features(current),
            ).argmax(dim=-1)
            action_steps.append(action.cpu())
            current, _, done = env.step(current, action)
            completed |= done
            if bool(completed.all()):
                break

    return DecisionActivity(
        active=torch.stack(active_steps, dim=1),
        actions=torch.stack(action_steps, dim=1),
    )
