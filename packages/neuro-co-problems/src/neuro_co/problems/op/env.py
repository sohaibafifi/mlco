"""OP env: Orienteering Problem. Pure functional, compile-friendly.

Node 0 = depot, nodes 1..n = customers with prizes. Travel budget caps
total tour length. Start at depot, visit a subset, return to depot with
total length <= budget. Maximize collected prize.

Reward (higher = better) delivered at episode end = collected prize.
Mask forbids any customer the vehicle could not visit and still return to
the depot within budget.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from neuro_co.core.state import State, register_state

# Budget grows with problem size (Kool 2019 OP constants, approx).
BUDGET_BY_SIZE: dict[int, float] = {10: 2.0, 20: 3.0, 50: 4.0, 100: 5.0}


@register_state
@dataclass(frozen=True, slots=True)
class OPState(State):
    coords: Float[Tensor, "b n_plus_1 2"]
    prize: Float[Tensor, "b n_plus_1"]  # depot prize = 0
    visited: Bool[Tensor, "b n_plus_1"]
    current: Int[Tensor, "b"]
    length_used: Float[Tensor, "b"]
    collected: Float[Tensor, "b"]
    step_count: Int[Tensor, "b"]


class OPEnv:
    """Pure-functional Orienteering env."""

    encoder_in_dim: int = 3  # coords (2) + prize (1)

    def __init__(self, size: int, budget: float | None = None) -> None:
        if size < 2:
            raise ValueError(f"OP size must be >=2, got {size}")
        self.size = size
        self.budget = float(budget if budget is not None else BUDGET_BY_SIZE.get(size, 3.0))

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> OPState:
        device = torch.device(device)
        n1 = self.size + 1
        gen_dev = generator.device if generator is not None else device

        def rand(*shape):
            t = torch.rand(*shape, generator=generator, device=gen_dev)
            return t if gen_dev == device else t.to(device)

        coords = rand(batch_size, n1, 2)
        # Prizes uniform (0, 1] for customers; depot = 0.
        cust_prize = rand(batch_size, self.size) * 0.99 + 0.01
        prize = torch.cat([torch.zeros(batch_size, 1, device=device), cust_prize], dim=1)
        visited = torch.zeros(batch_size, n1, dtype=torch.bool, device=device)
        return OPState(
            coords=coords,
            prize=prize,
            visited=visited,
            current=torch.zeros(batch_size, dtype=torch.long, device=device),
            length_used=torch.zeros(batch_size, device=device),
            collected=torch.zeros(batch_size, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def step(
        self,
        state: OPState,
        action: Int[Tensor, "b"],
    ) -> tuple[OPState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = torch.linalg.vector_norm(to_xy - from_xy, dim=-1)

        is_depot = action == 0
        action_prize = state.prize.gather(1, action.unsqueeze(1)).squeeze(1)
        new_visited = state.visited.scatter(1, action.unsqueeze(1), True)
        new_visited[:, 0] = False  # depot stays available as the terminal
        new_length = state.length_used + edge
        new_collected = state.collected + action_prize
        new_step = state.step_count + 1

        done = is_depot & (state.step_count > 0)  # returned to depot after moving
        reward = torch.where(done, new_collected, torch.zeros_like(new_collected))

        return (
            state.replace(
                visited=new_visited,
                current=action,
                length_used=new_length,
                collected=new_collected,
                step_count=new_step,
            ),
            reward,
            done,
        )

    def action_mask(self, state: OPState) -> Bool[Tensor, "b n_plus_1"]:
        cur_xy = _gather_rows(state.coords, state.current).unsqueeze(1)  # (b,1,2)
        to_all = torch.linalg.vector_norm(state.coords - cur_xy, dim=-1)  # (b, n+1)
        depot_xy = state.coords[:, 0:1, :]  # (b,1,2)
        all_to_depot = torch.linalg.vector_norm(state.coords - depot_xy, dim=-1)  # (b,n+1)

        # Feasible to visit j and still return to depot within budget.
        projected = state.length_used.unsqueeze(1) + to_all + all_to_depot
        feasible = projected <= self.budget + 1e-6
        cust_ok = (~state.visited) & feasible
        cust_ok[:, 0] = False

        # Depot permitted once we've left it (to terminate), or when stuck.
        moved = state.step_count > 0
        no_cust = ~cust_ok[:, 1:].any(dim=1)
        depot_ok = moved | no_cust

        mask = cust_ok.clone()
        mask[:, 0] = depot_ok
        return mask

    # --- Env Protocol extensions.

    def build_features(self, state: OPState) -> Float[Tensor, "b n_plus_1 3"]:
        return torch.cat([state.coords, state.prize.unsqueeze(-1)], dim=-1)

    def decoder_context(self, state: OPState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        depot = torch.zeros_like(state.current)
        return depot, state.current

    def max_steps(self, state: OPState) -> int:
        return self.size + 1

    def pomo_first_mask(self, state: OPState) -> Bool[Tensor, "b n_plus_1"]:
        m = self.action_mask(state).clone()
        m[:, 0] = False
        return m


def _gather_rows(
    x: Float[Tensor, "b n d"],
    idx: Int[Tensor, "b"],
) -> Float[Tensor, "b d"]:
    b, _, d = x.shape
    return x.gather(1, idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
