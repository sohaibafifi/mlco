"""TSP env. Pure functional, fixed-shape, compile-friendly.

Cities sampled uniform in [0, 1]^2. State carries coords, visited mask,
current node, first node, cumulative tour length. Reward delivered only
at episode end as `-tour_length`.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..state import State, register_state


@register_state
@dataclass(frozen=True, slots=True)
class TSPState(State):
    coords: Float[Tensor, "b n 2"]
    visited: Bool[Tensor, "b n"]
    current: Int[Tensor, "b"]
    first: Int[Tensor, "b"]
    step_count: Int[Tensor, "b"]
    tour_length: Float[Tensor, "b"]


class TSPEnv:
    """Pure-functional TSP env. Holds only constants (size); no batch state."""

    encoder_in_dim: int = 2

    def __init__(self, size: int) -> None:
        if size < 3:
            raise ValueError(f"TSP size must be >=3, got {size}")
        self.size = size

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> TSPState:
        device = torch.device(device)
        gen_dev = generator.device if generator is not None else device
        if gen_dev == device:
            coords = torch.rand(batch_size, self.size, 2, generator=generator, device=device)
        else:
            coords = torch.rand(batch_size, self.size, 2, generator=generator, device=gen_dev).to(
                device
            )
        visited = torch.zeros(batch_size, self.size, dtype=torch.bool, device=device)
        first = torch.zeros(batch_size, dtype=torch.long, device=device)
        visited[:, 0] = True
        return TSPState(
            coords=coords,
            visited=visited,
            current=first.clone(),
            first=first,
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            tour_length=torch.zeros(batch_size, device=device),
        )

    def step(
        self,
        state: TSPState,
        action: Int[Tensor, "b"],
    ) -> tuple[TSPState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        n = state.visited.shape[1]
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = torch.linalg.vector_norm(to_xy - from_xy, dim=-1)

        new_visited = state.visited.scatter(1, action.unsqueeze(1), True)
        new_step = state.step_count + 1
        new_length = state.tour_length + edge

        done = new_step >= (n - 1)
        close_xy = _gather_rows(state.coords, state.first)
        close_edge = torch.linalg.vector_norm(to_xy - close_xy, dim=-1)
        new_length = torch.where(done, new_length + close_edge, new_length)
        reward = torch.where(done, -new_length, torch.zeros_like(new_length))

        return (
            state.replace(
                visited=new_visited,
                current=action,
                step_count=new_step,
                tour_length=new_length,
            ),
            reward,
            done,
        )

    def action_mask(self, state: TSPState) -> Bool[Tensor, "b a"]:
        """True = action permitted. Cannot revisit."""
        return ~state.visited

    # --- Env Protocol extensions.

    def build_features(self, state: TSPState) -> Float[Tensor, "b n 2"]:
        return state.coords

    def decoder_context(self, state: TSPState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        return state.first, state.current

    def max_steps(self, state: TSPState) -> int:
        return self.size - 1

    def pomo_first_mask(self, state: TSPState) -> Bool[Tensor, "b n"]:
        """First action picks a node other than the start city (idx 0)."""
        return self.action_mask(state)


def _gather_rows(
    x: Float[Tensor, "b n d"],
    idx: Int[Tensor, "b"],
) -> Float[Tensor, "b d"]:
    """Index dim=1 with per-row indices."""
    b, _, d = x.shape
    return x.gather(1, idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
