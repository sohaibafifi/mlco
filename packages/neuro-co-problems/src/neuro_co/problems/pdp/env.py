"""PDP env: Pickup and Delivery Problem. Pure functional, compile-friendly.

Node 0 = depot. Nodes 1..k = pickups, nodes k+1..2k = deliveries, where
pickup i is paired with delivery i+k. A delivery may only be visited after
its pickup. Single vehicle, no capacity. Visit all nodes, return to depot,
minimize tour length.

Reward (higher = better) = -tour_length at episode end.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from neuro_co.core.state import State, register_state


@register_state
@dataclass(frozen=True, slots=True)
class PDPState(State):
    coords: Float[Tensor, "b n_plus_1 2"]
    node_type: Float[Tensor, "b n_plus_1"]  # 0 depot, 1 pickup, 2 delivery
    visited: Bool[Tensor, "b n_plus_1"]
    current: Int[Tensor, "b"]
    tour_length: Float[Tensor, "b"]
    step_count: Int[Tensor, "b"]


class PDPEnv:
    """Pure-functional Pickup-Delivery env. `size` = number of pickup-delivery pairs."""

    encoder_in_dim: int = 4  # coords (2) + is_pickup (1) + is_delivery (1)

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"PDP needs >=1 pair, got {size}")
        self.pairs = size
        self.n_nodes = 2 * size + 1  # depot + pickups + deliveries

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> PDPState:
        device = torch.device(device)
        gen_dev = generator.device if generator is not None else device
        coords = torch.rand(batch_size, self.n_nodes, 2, generator=generator, device=gen_dev)
        if gen_dev != device:
            coords = coords.to(device)
        node_type = torch.zeros(batch_size, self.n_nodes, device=device)
        node_type[:, 1 : self.pairs + 1] = 1.0  # pickups
        node_type[:, self.pairs + 1 :] = 2.0  # deliveries
        visited = torch.zeros(batch_size, self.n_nodes, dtype=torch.bool, device=device)
        return PDPState(
            coords=coords,
            node_type=node_type,
            visited=visited,
            current=torch.zeros(batch_size, dtype=torch.long, device=device),
            tour_length=torch.zeros(batch_size, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def step(
        self,
        state: PDPState,
        action: Int[Tensor, "b"],
    ) -> tuple[PDPState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = torch.linalg.vector_norm(to_xy - from_xy, dim=-1)

        new_visited = state.visited.scatter(1, action.unsqueeze(1), True)
        new_visited[:, 0] = False
        new_length = state.tour_length + edge
        new_step = state.step_count + 1

        all_visited = new_visited[:, 1:].all(dim=1)
        is_depot = action == 0
        done = all_visited & is_depot
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

    def action_mask(self, state: PDPState) -> Bool[Tensor, "b n_plus_1"]:
        k = self.pairs
        not_visited = ~state.visited

        # Pickups (1..k): allowed if not visited.
        # Deliveries (k+1..2k): allowed if not visited AND paired pickup visited.
        pickups_visited = state.visited[:, 1 : k + 1]  # (b, k)
        delivery_ok = (~state.visited[:, k + 1 :]) & pickups_visited  # (b, k)

        mask = torch.zeros_like(state.visited)
        mask[:, 1 : k + 1] = not_visited[:, 1 : k + 1]
        mask[:, k + 1 :] = delivery_ok

        all_visited = state.visited[:, 1:].all(dim=1)
        at_depot = state.current == 0
        no_node_left = ~mask[:, 1:].any(dim=1)
        # Depot only to terminate (all visited) or if genuinely stuck.
        mask[:, 0] = (all_visited & ~at_depot) | (no_node_left & ~at_depot)
        return mask

    # --- Env Protocol extensions.

    def build_features(self, state: PDPState) -> Float[Tensor, "b n_plus_1 4"]:
        is_pickup = (state.node_type == 1.0).float().unsqueeze(-1)
        is_delivery = (state.node_type == 2.0).float().unsqueeze(-1)
        return torch.cat([state.coords, is_pickup, is_delivery], dim=-1)

    def decoder_context(self, state: PDPState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        depot = torch.zeros_like(state.current)
        return depot, state.current

    def max_steps(self, state: PDPState) -> int:
        return self.n_nodes + 1

    def pomo_first_mask(self, state: PDPState) -> Bool[Tensor, "b n_plus_1"]:
        # First action must be a pickup (deliveries need their pickup first).
        m = torch.zeros_like(state.visited)
        m[:, 1 : self.pairs + 1] = True
        return m


def _gather_rows(
    x: Float[Tensor, "b n d"],
    idx: Int[Tensor, "b"],
) -> Float[Tensor, "b d"]:
    b, _, d = x.shape
    return x.gather(1, idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
