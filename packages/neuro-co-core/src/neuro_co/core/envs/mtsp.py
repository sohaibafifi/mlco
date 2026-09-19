"""mTSP env: multiple Travelling Salesmen. Pure functional.

Node 0 = depot, nodes 1..n = cities. `m` salesmen, each a closed route
through the depot. Modelled single-agent: returning to the depot closes
the current route and starts the next. Exactly `m` routes are used; all
cities visited. Objective: min-sum of all route lengths.

Reward (higher = better) = -total_length at episode end.

(Min-sum variant. Min-max is a different objective: left as future work.)
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..state import State, register_state


@register_state
@dataclass(frozen=True, slots=True)
class MTSPState(State):
    coords: Float[Tensor, "b n_plus_1 2"]
    visited: Bool[Tensor, "b n_plus_1"]
    current: Int[Tensor, "b"]
    routes_done: Int[Tensor, "b"]  # completed routes (depot returns)
    tour_length: Float[Tensor, "b"]
    step_count: Int[Tensor, "b"]


class MTSPEnv:
    """Pure-functional min-sum mTSP env."""

    encoder_in_dim: int = 2

    def __init__(self, size: int, num_agents: int = 2) -> None:
        if size < num_agents:
            raise ValueError(f"size {size} must be >= num_agents {num_agents}")
        self.size = size
        self.m = num_agents

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> MTSPState:
        device = torch.device(device)
        n1 = self.size + 1
        gen_dev = generator.device if generator is not None else device
        coords = torch.rand(batch_size, n1, 2, generator=generator, device=gen_dev)
        if gen_dev != device:
            coords = coords.to(device)
        visited = torch.zeros(batch_size, n1, dtype=torch.bool, device=device)
        return MTSPState(
            coords=coords,
            visited=visited,
            current=torch.zeros(batch_size, dtype=torch.long, device=device),
            routes_done=torch.zeros(batch_size, dtype=torch.long, device=device),
            tour_length=torch.zeros(batch_size, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def step(
        self,
        state: MTSPState,
        action: Int[Tensor, "b"],
    ) -> tuple[MTSPState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = torch.linalg.vector_norm(to_xy - from_xy, dim=-1)

        is_depot = action == 0
        # A depot visit after having left it closes a route.
        closing = is_depot & (state.current != 0)
        new_routes = state.routes_done + closing.long()

        new_visited = state.visited.scatter(1, action.unsqueeze(1), True)
        new_visited[:, 0] = False
        new_length = state.tour_length + edge
        new_step = state.step_count + 1

        all_visited = new_visited[:, 1:].all(dim=1)
        done = all_visited & is_depot
        reward = torch.where(done, -new_length, torch.zeros_like(new_length))

        return (
            state.replace(
                visited=new_visited,
                current=action,
                routes_done=new_routes,
                tour_length=new_length,
                step_count=new_step,
            ),
            reward,
            done,
        )

    def action_mask(self, state: MTSPState) -> Bool[Tensor, "b n_plus_1"]:
        cust_ok = ~state.visited
        cust_ok[:, 0] = False

        at_depot = state.current == 0
        all_visited = state.visited[:, 1:].all(dim=1)
        routes_left = state.routes_done < self.m

        # Depot allowed to close a route (if not already at depot and routes remain)
        # or to terminate once all cities visited.
        depot_ok = (~at_depot & routes_left) | all_visited
        # If no city is left and not all visited (shouldn't happen) keep depot open.
        no_cust = ~cust_ok[:, 1:].any(dim=1)
        depot_ok = depot_ok | no_cust

        mask = cust_ok.clone()
        mask[:, 0] = depot_ok
        return mask

    # --- Env Protocol extensions.

    def build_features(self, state: MTSPState) -> Float[Tensor, "b n_plus_1 2"]:
        return state.coords

    def decoder_context(self, state: MTSPState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        depot = torch.zeros_like(state.current)
        return depot, state.current

    def max_steps(self, state: MTSPState) -> int:
        return self.size + self.m + 1

    def pomo_first_mask(self, state: MTSPState) -> Bool[Tensor, "b n_plus_1"]:
        m = self.action_mask(state).clone()
        m[:, 0] = False
        return m


def _gather_rows(
    x: Float[Tensor, "b n d"],
    idx: Int[Tensor, "b"],
) -> Float[Tensor, "b d"]:
    b, _, d = x.shape
    return x.gather(1, idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
