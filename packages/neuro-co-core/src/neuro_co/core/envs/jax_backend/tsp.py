"""Pure-JAX TSP environment with the same transitions as the Torch backend.

The named-tuple state is a JAX pytree and can be used with transformations
such as ``jit``, ``vmap``, and ``scan`` without additional registration.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class JaxTSPState(NamedTuple):
    coords: jax.Array  # (b, n, 2)
    visited: jax.Array  # (b, n) bool
    current: jax.Array  # (b,) int32
    first: jax.Array  # (b,) int32
    step_count: jax.Array  # (b,) int32
    tour_length: jax.Array  # (b,)


class JaxTSPEnv:
    """Pure-functional JAX TSP env. Constants only; state carried externally."""

    encoder_in_dim: int = 2

    def __init__(self, size: int) -> None:
        if size < 3:
            raise ValueError(f"TSP size must be >=3, got {size}")
        self.size = size

    def reset(self, key: jax.Array, batch_size: int) -> JaxTSPState:
        coords = jax.random.uniform(key, (batch_size, self.size, 2))
        return self.reset_from_coords(coords)

    def reset_from_coords(self, coords: jax.Array) -> JaxTSPState:
        coords = jnp.asarray(coords)
        if coords.ndim != 3 or coords.shape[1:] != (self.size, 2):
            raise ValueError(f"coords must have shape (batch, {self.size}, 2), got {coords.shape}")
        if not jnp.issubdtype(coords.dtype, jnp.floating):
            raise TypeError(f"coords must have a floating dtype, got {coords.dtype}")
        batch_size = coords.shape[0]
        visited = jnp.zeros((batch_size, self.size), dtype=jnp.bool_)
        first = jnp.zeros((batch_size,), dtype=jnp.int32)
        visited = visited.at[:, 0].set(True)
        return JaxTSPState(
            coords=coords,
            visited=visited,
            current=first,
            first=first,
            step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            tour_length=jnp.zeros((batch_size,), dtype=coords.dtype),
        )

    def step(
        self, state: JaxTSPState, action: jax.Array
    ) -> tuple[JaxTSPState, jax.Array, jax.Array]:
        n = state.visited.shape[1]
        from_xy = _gather_rows(state.coords, state.current)
        to_xy = _gather_rows(state.coords, action)
        edge = jnp.linalg.norm(to_xy - from_xy, axis=-1)

        new_visited = state.visited.at[jnp.arange(action.shape[0]), action].set(True)
        new_step = state.step_count + 1
        new_length = state.tour_length + edge

        done = new_step >= (n - 1)
        close_xy = _gather_rows(state.coords, state.first)
        close_edge = jnp.linalg.norm(to_xy - close_xy, axis=-1)
        new_length = jnp.where(done, new_length + close_edge, new_length)
        reward = jnp.where(done, -new_length, jnp.zeros_like(new_length))

        new_state = state._replace(
            visited=new_visited,
            current=action,
            step_count=new_step,
            tour_length=new_length,
        )
        return new_state, reward, done

    def action_mask(self, state: JaxTSPState) -> jax.Array:
        return ~state.visited

    def build_features(self, state: JaxTSPState) -> jax.Array:
        return state.coords

    def decoder_context(self, state: JaxTSPState) -> tuple[jax.Array, jax.Array]:
        return state.first, state.current

    def max_steps(self, state: JaxTSPState) -> int:
        del state
        return self.size - 1

    def pomo_first_mask(self, state: JaxTSPState) -> jax.Array:
        return self.action_mask(state)


def _gather_rows(x: jax.Array, idx: jax.Array) -> jax.Array:
    """x: (b, n, d); idx: (b,) -> (b, d)."""
    b = idx.shape[0]
    return x[jnp.arange(b), idx]
