"""Pure-JAX CVRP environment with the same transitions as the Torch backend."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

CAPACITY_BY_SIZE: dict[int, float] = {
    10: 20.0,
    20: 30.0,
    50: 40.0,
    100: 50.0,
}


class JaxCVRPState(NamedTuple):
    coords: jax.Array
    demand: jax.Array
    visited: jax.Array
    current: jax.Array
    remaining_capacity: jax.Array
    tour_length: jax.Array
    step_count: jax.Array


class JaxCVRPEnv:
    """Pure-functional JAX CVRP environment."""

    encoder_in_dim: int = 3

    def __init__(self, size: int, capacity: float | None = None, max_demand: int = 9) -> None:
        if size < 2:
            raise ValueError(f"CVRP size must be >=2, got {size}")
        self.size = size
        self.capacity = float(
            capacity if capacity is not None else CAPACITY_BY_SIZE.get(size, 30.0)
        )
        self.max_demand = max_demand

    def reset(self, key: jax.Array, batch_size: int) -> JaxCVRPState:
        coords_key, demand_key = jax.random.split(key)
        n_plus_1 = self.size + 1
        coords = jax.random.uniform(coords_key, (batch_size, n_plus_1, 2))
        customer_demand = jax.random.randint(
            demand_key,
            (batch_size, self.size),
            minval=1,
            maxval=self.max_demand + 1,
        ).astype(coords.dtype)
        demand = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=coords.dtype), customer_demand], axis=1
        )
        return self.reset_from_data(coords, demand)

    def reset_from_data(self, coords: jax.Array, demand: jax.Array) -> JaxCVRPState:
        coords = jnp.asarray(coords)
        demand = jnp.asarray(demand)
        expected_nodes = self.size + 1
        if coords.ndim != 3 or coords.shape[1:] != (expected_nodes, 2):
            raise ValueError(
                f"coords must have shape (batch, {expected_nodes}, 2), got {coords.shape}"
            )
        if demand.ndim != 2 or demand.shape != coords.shape[:2]:
            raise ValueError(f"demand must have shape {coords.shape[:2]}, got {demand.shape}")
        if not jnp.issubdtype(coords.dtype, jnp.floating):
            raise TypeError(f"coords must have a floating dtype, got {coords.dtype}")
        if not jnp.issubdtype(demand.dtype, jnp.floating):
            raise TypeError(f"demand must have a floating dtype, got {demand.dtype}")
        if demand.dtype != coords.dtype:
            raise TypeError(
                f"coords and demand must have the same dtype, got {coords.dtype} and {demand.dtype}"
            )
        batch_size = coords.shape[0]
        return JaxCVRPState(
            coords=coords,
            demand=demand,
            visited=jnp.zeros((batch_size, expected_nodes), dtype=jnp.bool_),
            current=jnp.zeros((batch_size,), dtype=jnp.int32),
            remaining_capacity=jnp.full((batch_size,), self.capacity, dtype=coords.dtype),
            tour_length=jnp.zeros((batch_size,), dtype=coords.dtype),
            step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def step(
        self, state: JaxCVRPState, action: jax.Array
    ) -> tuple[JaxCVRPState, jax.Array, jax.Array]:
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = jnp.linalg.norm(to_xy - from_xy, axis=-1)

        is_depot = action == 0
        action_demand = _gather_values(state.demand, action)
        new_capacity = jnp.where(
            is_depot,
            jnp.full_like(state.remaining_capacity, self.capacity),
            state.remaining_capacity - action_demand,
        )

        rows = jnp.arange(action.shape[0])
        new_visited = state.visited.at[rows, action].set(True)
        new_visited = new_visited.at[:, 0].set(False)
        new_tour_length = state.tour_length + edge
        new_step = state.step_count + 1

        all_customers = jnp.all(new_visited[:, 1:], axis=1)
        done = all_customers & is_depot
        reward = jnp.where(done, -new_tour_length, jnp.zeros_like(new_tour_length))

        return (
            state._replace(
                visited=new_visited,
                current=action,
                remaining_capacity=new_capacity,
                tour_length=new_tour_length,
                step_count=new_step,
            ),
            reward,
            done,
        )

    def action_mask(self, state: JaxCVRPState) -> jax.Array:
        fits = state.demand <= state.remaining_capacity[:, None]
        customer_ok = (~state.visited) & fits
        customer_ok = customer_ok.at[:, 0].set(False)

        at_depot = state.current == 0
        all_customers_done = jnp.all(state.visited[:, 1:], axis=1)
        depot_ok = (~at_depot) | all_customers_done
        no_customer_left = ~jnp.any(customer_ok[:, 1:], axis=1)
        depot_ok = depot_ok | no_customer_left
        return customer_ok.at[:, 0].set(depot_ok)

    def build_features(self, state: JaxCVRPState) -> jax.Array:
        capacity = jnp.maximum(state.remaining_capacity[:, None], 1e-6)
        normalized_demand = (state.demand / capacity)[:, :, None]
        return jnp.concatenate([state.coords, normalized_demand], axis=-1)

    def decoder_context(self, state: JaxCVRPState) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros_like(state.current), state.current

    def dynamic_decoder_context(self, state: JaxCVRPState) -> jax.Array:
        return (state.remaining_capacity / self.capacity)[:, None]

    def max_steps(self, state: JaxCVRPState) -> int:
        del state
        return 2 * self.size

    def pomo_first_mask(self, state: JaxCVRPState) -> jax.Array:
        return self.action_mask(state).at[:, 0].set(False)


def _gather_rows(x: jax.Array, idx: jax.Array) -> jax.Array:
    return x[jnp.arange(idx.shape[0]), idx]


def _gather_values(x: jax.Array, idx: jax.Array) -> jax.Array:
    return x[jnp.arange(idx.shape[0]), idx]
