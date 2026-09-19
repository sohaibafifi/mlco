"""JAX CVRP environment tests, skipped if JAX is not installed."""

from typing import Any, cast

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neuro_co.core.envs.cvrp import CVRPEnv, CVRPState  # noqa: E402
from neuro_co.core.envs.jax_backend import JaxCVRPEnv, JaxCVRPState  # noqa: E402


def _fixed_states() -> tuple[CVRPState, JaxCVRPState]:
    coords = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    demand = np.asarray([[0.0, 2.0, 4.0, 3.0], [0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    visited = np.asarray([[False, True, False, False], [False, True, True, True]], dtype=np.bool_)
    current = np.asarray([1, 3], dtype=np.int32)
    remaining_capacity = np.asarray([3.0, 1.0], dtype=np.float32)
    tour_length = np.asarray([2.0, 3.0], dtype=np.float32)
    step_count = np.asarray([2, 4], dtype=np.int32)
    torch_state_type = cast(Any, CVRPState)
    return (
        torch_state_type(
            torch.from_numpy(coords),
            torch.from_numpy(demand),
            torch.from_numpy(visited),
            torch.from_numpy(current).long(),
            torch.from_numpy(remaining_capacity),
            torch.from_numpy(tour_length),
            torch.from_numpy(step_count).long(),
        ),
        JaxCVRPState(
            coords=jnp.asarray(coords),
            demand=jnp.asarray(demand),
            visited=jnp.asarray(visited),
            current=jnp.asarray(current),
            remaining_capacity=jnp.asarray(remaining_capacity),
            tour_length=jnp.asarray(tour_length),
            step_count=jnp.asarray(step_count),
        ),
    )


def test_reset_shapes_and_ranges_under_jit() -> None:
    env = JaxCVRPEnv(size=10, capacity=30.0, max_demand=9)
    state = jax.jit(lambda key: env.reset(key, batch_size=4))(jax.random.PRNGKey(0))
    state.coords.block_until_ready()

    assert state.coords.shape == (4, 11, 2)
    assert state.demand.shape == (4, 11)
    assert state.visited.shape == (4, 11)
    assert bool(jnp.all(state.demand[:, 0] == 0))
    assert bool(jnp.all((state.demand[:, 1:] >= 1) & (state.demand[:, 1:] <= 9)))
    assert bool(jnp.all(state.remaining_capacity == 30.0))
    assert bool(jnp.all(state.current == 0))


def test_reset_from_data_is_deterministic_jittable_and_validated() -> None:
    env = JaxCVRPEnv(size=2, capacity=5.0)
    coords = jnp.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    demand = jnp.asarray([[0.0, 2.0, 3.0]], dtype=jnp.float32)
    state = jax.jit(env.reset_from_data)(coords, demand)
    state.coords.block_until_ready()

    np.testing.assert_array_equal(np.asarray(state.coords), np.asarray(coords))
    np.testing.assert_array_equal(np.asarray(state.demand), np.asarray(demand))
    assert bool(jnp.all(state.remaining_capacity == 5.0))
    with pytest.raises(ValueError, match="coords must have shape"):
        env.reset_from_data(jnp.zeros((1, 2, 2)), demand)
    with pytest.raises(ValueError, match="demand must have shape"):
        env.reset_from_data(coords, jnp.zeros((1, 2)))
    with pytest.raises(TypeError, match="same dtype"):
        env.reset_from_data(coords, demand.astype(jnp.float16))


def test_fixed_state_semantics_match_torch() -> None:
    torch_env = CVRPEnv(size=3, capacity=5.0, max_demand=4)
    jax_env = JaxCVRPEnv(size=3, capacity=5.0, max_demand=4)
    torch_state, jax_state = _fixed_states()

    np.testing.assert_array_equal(
        np.asarray(jax_env.action_mask(jax_state)),
        torch_env.action_mask(torch_state).numpy(),
    )
    np.testing.assert_allclose(
        np.asarray(jax_env.build_features(jax_state)),
        torch_env.build_features(torch_state).numpy(),
    )
    jax_first, jax_current = jax_env.decoder_context(jax_state)
    torch_first, torch_current = torch_env.decoder_context(torch_state)
    np.testing.assert_array_equal(np.asarray(jax_first), torch_first.numpy())
    np.testing.assert_array_equal(np.asarray(jax_current), torch_current.numpy())
    np.testing.assert_allclose(
        np.asarray(jax_env.dynamic_decoder_context(jax_state)),
        torch_env.dynamic_decoder_context(torch_state).numpy(),
    )
    np.testing.assert_array_equal(
        np.asarray(jax_env.pomo_first_mask(jax_state)),
        torch_env.pomo_first_mask(torch_state).numpy(),
    )
    assert jax_env.max_steps(jax_state) == torch_env.max_steps(torch_state)

    actions = np.asarray([3, 0], dtype=np.int32)
    torch_next, torch_reward, torch_done = torch_env.step(
        torch_state, torch.from_numpy(actions).long()
    )
    jax_next, jax_reward, jax_done = jax_env.step(jax_state, jnp.asarray(actions))
    for name in JaxCVRPState._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(jax_next, name)), getattr(torch_next, name).numpy()
        )
    np.testing.assert_allclose(np.asarray(jax_reward), torch_reward.numpy())
    np.testing.assert_array_equal(np.asarray(jax_done), torch_done.numpy())
    assert not bool(jax_done[0])
    assert bool(jax_done[1])


def test_transition_and_generic_methods_are_jittable() -> None:
    env = JaxCVRPEnv(size=3, capacity=5.0, max_demand=4)
    _, state = _fixed_states()
    actions = jnp.asarray([3, 0], dtype=jnp.int32)

    @jax.jit
    def transition(s, a):
        next_state, reward, done = env.step(s, a)
        first, current = env.decoder_context(next_state)
        return (
            next_state,
            reward,
            done,
            env.action_mask(next_state),
            env.build_features(next_state),
            first,
            current,
            env.dynamic_decoder_context(next_state),
            env.pomo_first_mask(next_state),
        )

    compiled = transition(state, actions)
    direct_next, direct_reward, direct_done = env.step(state, actions)
    direct = (
        direct_next,
        direct_reward,
        direct_done,
        env.action_mask(direct_next),
        env.build_features(direct_next),
        *env.decoder_context(direct_next),
        env.dynamic_decoder_context(direct_next),
        env.pomo_first_mask(direct_next),
    )
    compiled[1].block_until_ready()
    for compiled_value, direct_value in zip(
        jax.tree.leaves(compiled), jax.tree.leaves(direct), strict=True
    ):
        np.testing.assert_allclose(np.asarray(compiled_value), np.asarray(direct_value))
    assert env.max_steps(state) == 6
