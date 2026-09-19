"""CVRP env. Pure functional, fixed-shape, compile-friendly.

Layout: node 0 = depot, nodes 1..n = customers with demands. Vehicle
starts at depot with full capacity. May return to depot to refill
(capacity reset). Episode ends when all customers visited AND vehicle
back at depot.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from neuro_co.core.state import State, register_state

CAPACITY_BY_SIZE: dict[int, float] = {
    10: 20.0,
    20: 30.0,
    50: 40.0,
    100: 50.0,
}


@register_state
@dataclass(frozen=True, slots=True)
class CVRPState(State):
    coords: Float[Tensor, "b n_plus_1 2"]
    demand: Float[Tensor, "b n_plus_1"]
    visited: Bool[Tensor, "b n_plus_1"]
    current: Int[Tensor, "b"]
    remaining_capacity: Float[Tensor, "b"]
    tour_length: Float[Tensor, "b"]
    step_count: Int[Tensor, "b"]


class CVRPEnv:
    """Pure-functional CVRP env."""

    encoder_in_dim: int = 3  # coords (2) + normalized demand (1)

    def __init__(self, size: int, capacity: float | None = None, max_demand: int = 9) -> None:
        if size < 2:
            raise ValueError(f"CVRP size must be >=2, got {size}")
        self.size = size
        self.capacity = float(
            capacity if capacity is not None else CAPACITY_BY_SIZE.get(size, 30.0)
        )
        self.max_demand = max_demand

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> CVRPState:
        device = torch.device(device)
        n_plus_1 = self.size + 1
        gen_dev = generator.device if generator is not None else device
        if gen_dev == device:
            coords = torch.rand(batch_size, n_plus_1, 2, generator=generator, device=device)
            cust_demand = torch.randint(
                1, self.max_demand + 1, (batch_size, self.size), generator=generator, device=device
            ).float()
        else:
            coords = torch.rand(batch_size, n_plus_1, 2, generator=generator, device=gen_dev).to(
                device
            )
            cust_demand = (
                torch.randint(
                    1,
                    self.max_demand + 1,
                    (batch_size, self.size),
                    generator=generator,
                    device=gen_dev,
                )
                .float()
                .to(device)
            )
        demand = torch.cat([torch.zeros(batch_size, 1, device=device), cust_demand], dim=1)
        visited = torch.zeros(batch_size, n_plus_1, dtype=torch.bool, device=device)
        return CVRPState(
            coords=coords,
            demand=demand,
            visited=visited,
            current=torch.zeros(batch_size, dtype=torch.long, device=device),
            remaining_capacity=torch.full((batch_size,), self.capacity, device=device),
            tour_length=torch.zeros(batch_size, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def step(
        self,
        state: CVRPState,
        action: Int[Tensor, "b"],
    ) -> tuple[CVRPState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = torch.linalg.vector_norm(to_xy - from_xy, dim=-1)

        is_depot = action == 0
        action_demand = state.demand.gather(1, action.unsqueeze(1)).squeeze(1)
        new_capacity = torch.where(
            is_depot,
            torch.full_like(state.remaining_capacity, self.capacity),
            state.remaining_capacity - action_demand,
        )

        new_visited = state.visited.scatter(1, action.unsqueeze(1), True)
        new_visited[:, 0] = False  # depot stays unvisited

        new_tour_length = state.tour_length + edge
        new_step = state.step_count + 1

        all_customers = new_visited[:, 1:].all(dim=1)
        done = all_customers & is_depot
        reward = torch.where(done, -new_tour_length, torch.zeros_like(new_tour_length))

        return (
            state.replace(
                visited=new_visited,
                current=action,
                remaining_capacity=new_capacity,
                tour_length=new_tour_length,
                step_count=new_step,
            ),
            reward,
            done,
        )

    def action_mask(self, state: CVRPState) -> Bool[Tensor, "b n_plus_1"]:
        fits = state.demand <= state.remaining_capacity.unsqueeze(1)
        cust_ok = (~state.visited) & fits
        cust_ok[:, 0] = False

        at_depot = state.current == 0
        all_customers_done = state.visited[:, 1:].all(dim=1)
        depot_ok = (~at_depot) | all_customers_done

        no_cust_left = ~cust_ok[:, 1:].any(dim=1)
        depot_ok = depot_ok | no_cust_left

        mask = cust_ok.clone()
        mask[:, 0] = depot_ok
        return mask

    # --- Env Protocol extensions.

    def build_features(self, state: CVRPState) -> Float[Tensor, "b n_plus_1 3"]:
        cap = state.remaining_capacity.unsqueeze(-1).clamp_min(1e-6)
        demand_norm = (state.demand / cap.expand_as(state.demand)).unsqueeze(-1)
        return torch.cat([state.coords, demand_norm], dim=-1)

    def decoder_context(self, state: CVRPState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        depot = torch.zeros_like(state.current)
        return depot, state.current

    def dynamic_decoder_context(self, state: CVRPState) -> Float[Tensor, "b 1"]:
        """Normalized remaining capacity used by the CVRP decoder query."""

        return (state.remaining_capacity / self.capacity).unsqueeze(-1)

    def max_steps(self, state: CVRPState) -> int:
        return 2 * self.size

    def pomo_first_mask(self, state: CVRPState) -> Bool[Tensor, "b n_plus_1"]:
        """First action picks a customer (not depot, must fit capacity)."""
        m = self.action_mask(state).clone()
        m[:, 0] = False
        return m


def _gather_rows(
    x: Float[Tensor, "b n d"],
    idx: Int[Tensor, "b"],
) -> Float[Tensor, "b d"]:
    b, _, d = x.shape
    return x.gather(1, idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
