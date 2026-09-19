"""ATSP env: Asymmetric TSP. Pure functional.

Distances given by a random asymmetric matrix `D (b, n, n)` (D[i,j] !=
D[j,i] in general), not by coordinates. Visit every city once, return to
the start, minimize total directed length.

Pairs with the MatNet encoder (`models/encoders/matnet.py`), which consumes
the distance matrix directly via edge-aware attention. `encoder_in_dim`
equals the problem size (each row is an n-vector of out-distances).

Reward (higher = better) = -tour_length at episode end.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..state import State, register_state


@register_state
@dataclass(frozen=True, slots=True)
class ATSPState(State):
    dist: Float[Tensor, "b n n"]  # asymmetric distance matrix
    visited: Bool[Tensor, "b n"]
    current: Int[Tensor, "b"]
    first: Int[Tensor, "b"]
    tour_length: Float[Tensor, "b"]
    step_count: Int[Tensor, "b"]


class ATSPEnv:
    """Pure-functional Asymmetric TSP env."""

    def __init__(self, size: int) -> None:
        if size < 3:
            raise ValueError(f"ATSP size must be >=3, got {size}")
        self.size = size
        self.encoder_in_dim = size  # MatNet consumes the n-wide distance rows

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> ATSPState:
        device = torch.device(device)
        n = self.size
        gen_dev = generator.device if generator is not None else device
        dist = torch.rand(batch_size, n, n, generator=generator, device=gen_dev)
        if gen_dev != device:
            dist = dist.to(device)
        # Zero diagonal (no self-loops).
        eye = torch.eye(n, dtype=torch.bool, device=device)
        dist = dist.masked_fill(eye.unsqueeze(0), 0.0)
        visited = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
        first = torch.zeros(batch_size, dtype=torch.long, device=device)
        visited[:, 0] = True
        return ATSPState(
            dist=dist,
            visited=visited,
            current=first.clone(),
            first=first,
            tour_length=torch.zeros(batch_size, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def step(
        self,
        state: ATSPState,
        action: Int[Tensor, "b"],
    ) -> tuple[ATSPState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        n = state.visited.shape[1]
        b = state.visited.shape[0]
        ar = torch.arange(b, device=state.dist.device)
        edge = state.dist[ar, state.current, action]

        new_visited = state.visited.scatter(1, action.unsqueeze(1), True)
        new_step = state.step_count + 1
        new_length = state.tour_length + edge

        done = new_step >= (n - 1)
        close_edge = state.dist[ar, action, state.first]
        new_length = torch.where(done, new_length + close_edge, new_length)
        reward = torch.where(done, -new_length, torch.zeros_like(new_length))

        return (
            state.replace(
                visited=new_visited,
                current=action,
                tour_length=new_length,
                step_count=new_step,
            ),
            reward,
            done,
        )

    def action_mask(self, state: ATSPState) -> Bool[Tensor, "b n"]:
        return ~state.visited

    # --- Env Protocol extensions.

    def build_features(self, state: ATSPState) -> Float[Tensor, "b n n"]:
        return state.dist

    def decoder_context(self, state: ATSPState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        return state.first, state.current

    def max_steps(self, state: ATSPState) -> int:
        return self.size - 1

    def pomo_first_mask(self, state: ATSPState) -> Bool[Tensor, "b n"]:
        return self.action_mask(state)
