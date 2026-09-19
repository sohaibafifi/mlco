"""POMO loss, rollout, gradient, and optimizer tests for the JAX backend."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neuro_co.core.envs.jax_backend import JaxCVRPEnv, JaxTSPEnv  # noqa: E402
from neuro_co.core.jax_backend import (  # noqa: E402
    AdamConfig,
    JaxAttentionModel,
    JaxPOMO,
    distinct_first_actions,
    pomo_advantage,
    pomo_loss,
)


def test_pomo_advantage_and_loss_have_expected_values() -> None:
    reward = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 7.0, 1.0]])
    sum_log_probability = jnp.asarray([[-0.2, -0.5, -0.7], [-0.1, -0.3, -0.9]])
    advantage = pomo_advantage(reward)
    np.testing.assert_allclose(np.asarray(jnp.mean(advantage, axis=1)), 0.0, atol=1e-7)
    expected = -np.mean(np.asarray(advantage) * np.asarray(sum_log_probability))
    assert float(pomo_loss(reward, sum_log_probability)) == pytest.approx(expected)


def test_distinct_first_actions_are_valid_and_wrap() -> None:
    mask = jnp.asarray([[False, True, True, True], [False, False, True, False]])
    actions = distinct_first_actions(mask, 5, jax.random.key(3)).reshape(2, 5)
    selected = np.asarray(actions)
    assert set(selected[0, :3]) == {1, 2, 3}
    assert set(selected[0]).issubset({1, 2, 3})
    assert np.all(selected[1] == 2)


def test_distinct_first_actions_rejects_empty_row() -> None:
    mask = jnp.asarray([[False, True], [False, False]])
    with pytest.raises(ValueError, match="zero permitted first actions"):
        distinct_first_actions(mask, 2, jax.random.key(4))


def test_forced_first_action_has_zero_log_probability_contribution() -> None:
    env = JaxTSPEnv(size=3)
    model = JaxAttentionModel(hidden_dim=8, num_layers=1, num_heads=2)
    algo = JaxPOMO(model=model, env=env, n_starts=2)
    params = model.init(jax.random.key(1))
    params = jax.tree.map(jnp.zeros_like, params)
    problem_state = env.reset(jax.random.key(2), batch_size=2)
    _reward, sum_log_probability = jax.jit(algo.rollout)(
        params,
        problem_state,
        jax.random.key(4),
    )
    np.testing.assert_allclose(np.asarray(sum_log_probability), 0.0, atol=1e-7)


@pytest.mark.parametrize(
    ("env", "in_dim", "n_starts"),
    [
        (JaxTSPEnv(size=5), 2, 4),
        (JaxCVRPEnv(size=5, capacity=20.0), 3, 3),
    ],
)
def test_jitted_train_step_is_finite_and_updates_parameters(env, in_dim, n_starts) -> None:
    model = JaxAttentionModel(
        in_dim=in_dim,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
    )
    algo = JaxPOMO(
        model=model,
        env=env,
        n_starts=n_starts,
        optimizer=AdamConfig(
            learning_rate=1.0e-3,
            weight_decay=1.0e-6,
            grad_clip=1.0,
        ),
    )
    train_state = algo.init(jax.random.key(5))
    before = train_state.params
    compiled_step = jax.jit(lambda state, key: algo.sample_train_step(state, key, batch_size=3))
    train_state, metrics = compiled_step(train_state, jax.random.key(6))

    assert all(
        np.isfinite(float(value))
        for value in (
            metrics.loss,
            metrics.reward_mean,
            metrics.reward_max_per_group,
            metrics.advantage_std,
            metrics.grad_norm,
        )
    )
    assert float(metrics.grad_norm) > 0.0
    assert int(train_state.optimizer_state.count) == 1
    assert any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for old, new in zip(
            jax.tree.leaves(before), jax.tree.leaves(train_state.params), strict=True
        )
    )


@pytest.mark.parametrize(
    ("env", "in_dim"),
    [(JaxTSPEnv(size=5), 2), (JaxCVRPEnv(size=5, capacity=20.0), 3)],
)
def test_jitted_greedy_rollout_returns_finite_cost(env, in_dim) -> None:
    model = JaxAttentionModel(
        in_dim=in_dim,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
    )
    algo = JaxPOMO(model=model, env=env, n_starts=3)
    train_state = algo.init(jax.random.key(7))
    problems = env.reset(jax.random.key(8), batch_size=4)
    reward = jax.jit(algo.greedy_rollout)(train_state.params, problems)
    assert reward.shape == (4,)
    assert np.all(np.isfinite(np.asarray(reward)))
    assert np.all(np.asarray(reward) < 0.0)


def test_loss_gradient_does_not_flow_through_reward_baseline() -> None:
    log_probability = jnp.asarray([[-0.2, -0.4, -0.8]])

    def objective(reward):
        return pomo_loss(reward, log_probability)

    gradient = jax.grad(objective)(jnp.asarray([[1.0, 2.0, 3.0]]))
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((1, 3), dtype=np.float32))
