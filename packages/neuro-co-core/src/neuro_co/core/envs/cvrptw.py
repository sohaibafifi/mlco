"""CVRPTW env: Capacitated VRP with Time Windows.

Each customer has a demand AND a time window `[e_i, l_i]`. Vehicle has
capacity + speed (constant 1.0: travel time = euclidean distance).
Service time per customer = 0 (instantaneous). Vehicle starts at depot
at time 0 with full capacity, must return to depot. May arrive at a
customer *before* its window opens and wait (no penalty). Arriving
*after* `l_i` is infeasible.

State adds `current_time` per batch element; mask filters customers
whose `arrival = current_time + dist(current, j)` exceeds `l_j`.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..state import State, register_state

CAPACITY_BY_SIZE: dict[int, float] = {
    10: 20.0,
    20: 30.0,
    50: 40.0,
    100: 50.0,
}


@register_state
@dataclass(frozen=True, slots=True)
class CVRPTWState(State):
    coords: Float[Tensor, "b n_plus_1 2"]
    demand: Float[Tensor, "b n_plus_1"]
    tw_early: Float[Tensor, "b n_plus_1"]
    tw_late: Float[Tensor, "b n_plus_1"]
    visited: Bool[Tensor, "b n_plus_1"]
    current: Int[Tensor, "b"]
    current_time: Float[Tensor, "b"]
    remaining_capacity: Float[Tensor, "b"]
    tour_length: Float[Tensor, "b"]
    step_count: Int[Tensor, "b"]


class CVRPTWEnv:
    """Pure-functional CVRPTW env."""

    encoder_in_dim: int = 6  # coords(2) + demand_norm(1) + tw_early(1) + tw_late(1) + time_now(1)

    def __init__(
        self,
        size: int,
        capacity: float | None = None,
        max_demand: int = 9,
        horizon: float = 4.0,
        window_width: float = 0.5,
    ) -> None:
        if size < 2:
            raise ValueError(f"CVRPTW size must be >=2, got {size}")
        self.size = size
        self.capacity = float(
            capacity if capacity is not None else CAPACITY_BY_SIZE.get(size, 30.0)
        )
        self.max_demand = max_demand
        self.horizon = horizon  # depot tw_late and tw_early max
        self.window_width = window_width

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> CVRPTWState:
        device = torch.device(device)
        n_plus_1 = self.size + 1
        gen_dev = generator.device if generator is not None else device

        def rand(*shape):
            t = torch.rand(*shape, generator=generator, device=gen_dev)
            return t if gen_dev == device else t.to(device)

        def randint(low, high, *shape):
            t = torch.randint(low, high, shape, generator=generator, device=gen_dev)
            return t if gen_dev == device else t.to(device)

        coords = rand(batch_size, n_plus_1, 2)

        # Demand: depot = 0; customers uniform [1, max_demand].
        cust_demand = randint(1, self.max_demand + 1, batch_size, self.size).float()
        demand = torch.cat([torch.zeros(batch_size, 1, device=device), cust_demand], dim=1)

        # Time windows: depot = [0, horizon]; customers e_i in [0, horizon - window_width],
        # l_i = e_i + window_width.
        cust_early = rand(batch_size, self.size) * (self.horizon - self.window_width)
        cust_late = cust_early + self.window_width
        tw_early = torch.cat([torch.zeros(batch_size, 1, device=device), cust_early], dim=1)
        tw_late = torch.cat(
            [torch.full((batch_size, 1), self.horizon, device=device), cust_late], dim=1
        )

        visited = torch.zeros(batch_size, n_plus_1, dtype=torch.bool, device=device)
        return CVRPTWState(
            coords=coords,
            demand=demand,
            tw_early=tw_early,
            tw_late=tw_late,
            visited=visited,
            current=torch.zeros(batch_size, dtype=torch.long, device=device),
            current_time=torch.zeros(batch_size, device=device),
            remaining_capacity=torch.full((batch_size,), self.capacity, device=device),
            tour_length=torch.zeros(batch_size, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def step(
        self,
        state: CVRPTWState,
        action: Int[Tensor, "b"],
    ) -> tuple[CVRPTWState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = torch.linalg.vector_norm(to_xy - from_xy, dim=-1)

        # Travel time = edge length. Then wait if early.
        arrival = state.current_time + edge
        action_early = state.tw_early.gather(1, action.unsqueeze(1)).squeeze(1)
        new_time = torch.maximum(arrival, action_early)

        is_depot = action == 0
        action_demand = state.demand.gather(1, action.unsqueeze(1)).squeeze(1)
        new_capacity = torch.where(
            is_depot,
            torch.full_like(state.remaining_capacity, self.capacity),
            state.remaining_capacity - action_demand,
        )
        # Reset clock on depot return.
        new_time = torch.where(is_depot, torch.zeros_like(new_time), new_time)

        new_visited = state.visited.scatter(1, action.unsqueeze(1), True)
        new_visited[:, 0] = False

        new_tour_length = state.tour_length + edge
        new_step = state.step_count + 1

        all_customers = new_visited[:, 1:].all(dim=1)
        done = all_customers & is_depot
        reward = torch.where(done, -new_tour_length, torch.zeros_like(new_tour_length))

        return (
            state.replace(
                visited=new_visited,
                current=action,
                current_time=new_time,
                remaining_capacity=new_capacity,
                tour_length=new_tour_length,
                step_count=new_step,
            ),
            reward,
            done,
        )

    def action_mask(self, state: CVRPTWState) -> Bool[Tensor, "b n_plus_1"]:
        # Distance from current to each node.
        cur_xy = _gather_rows(state.coords, state.current).unsqueeze(1)  # (b, 1, 2)
        d = torch.linalg.vector_norm(state.coords - cur_xy, dim=-1)  # (b, n+1)
        arrival = state.current_time.unsqueeze(1) + d
        in_window = arrival <= state.tw_late + 1e-6

        fits_capacity = state.demand <= state.remaining_capacity.unsqueeze(1)
        cust_ok = (~state.visited) & fits_capacity & in_window
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

    def build_features(self, state: CVRPTWState) -> Float[Tensor, "b n_plus_1 6"]:
        cap = state.remaining_capacity.unsqueeze(-1).clamp_min(1e-6)
        demand_norm = (state.demand / cap.expand_as(state.demand)).unsqueeze(-1)
        tw_e = (state.tw_early / self.horizon).unsqueeze(-1)
        tw_l = (state.tw_late / self.horizon).unsqueeze(-1)
        t_now = (
            (state.current_time / self.horizon).unsqueeze(1).expand_as(state.demand).unsqueeze(-1)
        )
        return torch.cat([state.coords, demand_norm, tw_e, tw_l, t_now], dim=-1)

    def decoder_context(self, state: CVRPTWState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        depot = torch.zeros_like(state.current)
        return depot, state.current

    def max_steps(self, state: CVRPTWState) -> int:
        return 2 * self.size

    def pomo_first_mask(self, state: CVRPTWState) -> Bool[Tensor, "b n_plus_1"]:
        m = self.action_mask(state).clone()
        m[:, 0] = False
        return m


def _gather_rows(
    x: Float[Tensor, "b n d"],
    idx: Int[Tensor, "b"],
) -> Float[Tensor, "b d"]:
    b, _, d = x.shape
    return x.gather(1, idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
