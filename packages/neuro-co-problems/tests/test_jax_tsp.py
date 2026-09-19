"""JAX TSP environment tests, skipped if JAX is not installed."""

from typing import Any, cast

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neuro_co.problems.tsp.env import TSPEnv, TSPState  # noqa: E402
from neuro_co.problems.tsp.jax_env import (  # noqa: E402
    JaxTSPEnv,
    JaxTSPState,
)


def test_reset_shapes() -> None:
    env = JaxTSPEnv(size=10)
    key = jax.random.PRNGKey(0)
    s = env.reset(key, batch_size=4)
    assert s.coords.shape == (4, 10, 2)
    assert s.visited.shape == (4, 10)
    assert bool(jnp.all(s.visited[:, 0]))
    assert not bool(jnp.any(s.visited[:, 1:]))


def test_step_advances() -> None:
    env = JaxTSPEnv(size=5)
    key = jax.random.PRNGKey(7)
    s = env.reset(key, batch_size=2)
    a = jnp.asarray([2, 3], dtype=jnp.int32)
    s2, reward, done = env.step(s, a)
    assert bool(jnp.all(s2.current == a))
    assert bool(jnp.all(s2.step_count == 1))
    assert not bool(jnp.any(done))
    assert bool(jnp.all(reward == 0))


def test_generic_methods_match_torch_on_fixed_state() -> None:
    coords = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    visited = np.asarray([[True, True, False, False], [True, False, True, False]], dtype=np.bool_)
    current = np.asarray([1, 2], dtype=np.int32)
    first = np.zeros(2, dtype=np.int32)
    step_count = np.asarray([1, 2], dtype=np.int32)
    tour_length = np.asarray([1.0, 2.0], dtype=np.float32)
    actions = np.asarray([2, 3], dtype=np.int32)

    torch_state_type = cast(Any, TSPState)
    torch_state = torch_state_type(
        torch.from_numpy(coords),
        torch.from_numpy(visited),
        torch.from_numpy(current).long(),
        torch.from_numpy(first).long(),
        torch.from_numpy(step_count).long(),
        torch.from_numpy(tour_length),
    )
    jax_state = JaxTSPState(
        coords=jnp.asarray(coords),
        visited=jnp.asarray(visited),
        current=jnp.asarray(current),
        first=jnp.asarray(first),
        step_count=jnp.asarray(step_count),
        tour_length=jnp.asarray(tour_length),
    )
    torch_env = TSPEnv(size=4)
    jax_env = JaxTSPEnv(size=4)

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
    np.testing.assert_array_equal(
        np.asarray(jax_env.pomo_first_mask(jax_state)),
        torch_env.pomo_first_mask(torch_state).numpy(),
    )
    assert jax_env.max_steps(jax_state) == torch_env.max_steps(torch_state)

    torch_next, torch_reward, torch_done = torch_env.step(
        torch_state, torch.from_numpy(actions).long()
    )
    jax_next, jax_reward, jax_done = jax_env.step(jax_state, jnp.asarray(actions))
    for name in JaxTSPState._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(jax_next, name)), getattr(torch_next, name).numpy()
        )
    np.testing.assert_allclose(np.asarray(jax_reward), torch_reward.numpy())
    np.testing.assert_array_equal(np.asarray(jax_done), torch_done.numpy())


def test_generic_methods_are_jittable() -> None:
    env = JaxTSPEnv(size=5)
    state = env.reset(jax.random.PRNGKey(3), batch_size=2)

    @jax.jit
    def generic_values(s):
        first, current = env.decoder_context(s)
        return (
            env.build_features(s),
            first,
            current,
            env.pomo_first_mask(s),
        )

    features, first, current, first_mask = generic_values(state)
    features.block_until_ready()
    assert features.shape == (2, 5, 2)
    assert first.shape == current.shape == (2,)
    assert first_mask.shape == (2, 5)
    assert env.max_steps(state) == 4


def test_reset_from_coords_is_deterministic_jittable_and_validated() -> None:
    env = JaxTSPEnv(size=3)
    coords = jnp.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    state = jax.jit(env.reset_from_coords)(coords)
    state.coords.block_until_ready()

    np.testing.assert_array_equal(np.asarray(state.coords), np.asarray(coords))
    assert bool(state.visited[0, 0])
    assert not bool(jnp.any(state.visited[0, 1:]))
    with pytest.raises(ValueError, match="coords must have shape"):
        env.reset_from_coords(jnp.zeros((1, 4, 2), dtype=jnp.float32))
    with pytest.raises(TypeError, match="floating dtype"):
        env.reset_from_coords(jnp.zeros((1, 3, 2), dtype=jnp.int32))


def test_jit_full_rollout() -> None:
    env = JaxTSPEnv(size=6)
    key = jax.random.PRNGKey(0)
    s = env.reset(key, batch_size=3)

    @jax.jit
    def rollout(s):
        def body(carry, _):
            st = carry
            mask = env.action_mask(st)
            action = jnp.argmax(mask.astype(jnp.float32), axis=-1)
            st2, _, _ = env.step(st, action)
            return st2, None

        final, _ = jax.lax.scan(body, s, None, length=5)
        return final.tour_length

    out = rollout(s)
    out.block_until_ready()
    assert out.shape == (3,)
    assert bool(jnp.all(out > 0))
