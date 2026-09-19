"""Tests for the small pure-JAX optimizer implementation."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neuro_co.core.jax_backend import (  # noqa: E402
    AdamConfig,
    adam_update,
    clip_by_global_norm,
    global_norm,
    init_adam,
)


def test_global_norm_and_clipping() -> None:
    grads = {"a": jnp.asarray([3.0, 4.0]), "b": jnp.asarray([0.0])}
    assert float(global_norm(grads)) == pytest.approx(5.0)
    clipped, original = clip_by_global_norm(grads, 2.0)
    assert float(original) == pytest.approx(5.0)
    expected = 2.0 * 5.0 / (5.0 + 1.0e-6)
    assert float(global_norm(clipped)) == pytest.approx(expected)


def test_adam_moments_do_not_alias_for_buffer_donation() -> None:
    params = {"value": jnp.asarray([1.0, 2.0])}
    state = init_adam(params)
    assert state.first_moment["value"] is not state.second_moment["value"]


def test_global_norm_clipping_matches_torch() -> None:
    values = np.asarray([3.0, 4.0], dtype=np.float32)
    torch_param = torch.nn.Parameter(torch.zeros_like(torch.from_numpy(values)))
    torch_param.grad = torch.from_numpy(values.copy())
    expected_norm = torch.nn.utils.clip_grad_norm_([torch_param], 2.0)

    clipped, actual_norm = jax.jit(lambda grads: clip_by_global_norm(grads, 2.0))(
        {"value": jnp.asarray(values)}
    )

    assert float(actual_norm) == pytest.approx(float(expected_norm))
    np.testing.assert_allclose(
        np.asarray(clipped["value"]), torch_param.grad.numpy(), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("decoupled", [False, True])
def test_adam_matches_torch_for_prescribed_updates(decoupled: bool) -> None:
    initial = np.asarray([0.25, -0.5, 1.5], dtype=np.float32)
    torch_param = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    optimizer_type = torch.optim.AdamW if decoupled else torch.optim.Adam
    torch_optimizer = optimizer_type(
        [torch_param],
        lr=3.0e-3,
        betas=(0.8, 0.95),
        eps=1.0e-7,
        weight_decay=2.0e-2,
    )
    config = AdamConfig(
        learning_rate=3.0e-3,
        beta1=0.8,
        beta2=0.95,
        eps=1.0e-7,
        weight_decay=2.0e-2,
        decoupled_weight_decay=decoupled,
        grad_clip=0.0,
    )
    params = {"value": jnp.asarray(initial)}
    state = init_adam(params)
    update = jax.jit(lambda p, g, s: adam_update(p, g, s, config))

    for values in ([0.1, -0.2, 0.3], [-0.4, 0.2, 0.05], [0.2, 0.1, -0.1]):
        gradient = np.asarray(values, dtype=np.float32)
        torch_optimizer.zero_grad(set_to_none=True)
        torch_param.grad = torch.from_numpy(gradient.copy())
        torch_optimizer.step()
        params, state, _norm = update(params, {"value": jnp.asarray(gradient)}, state)

    np.testing.assert_allclose(
        np.asarray(params["value"]), torch_param.detach().numpy(), rtol=2e-6, atol=2e-7
    )
    torch_state = torch_optimizer.state[torch_param]
    np.testing.assert_allclose(
        np.asarray(state.first_moment["value"]),
        torch_state["exp_avg"].numpy(),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(state.second_moment["value"]),
        torch_state["exp_avg_sq"].numpy(),
        rtol=2e-6,
        atol=2e-7,
    )
    assert int(state.count) == 3


def test_adam_step_reduces_quadratic_objective() -> None:
    config = AdamConfig(learning_rate=0.05, grad_clip=1.0)
    params = {"value": jnp.asarray([2.0, -1.0])}
    state = init_adam(params)

    @jax.jit
    def step(current_params, current_state):
        loss, grads = jax.value_and_grad(lambda tree: jnp.sum(jnp.square(tree["value"])))(
            current_params
        )
        new_params, new_state, grad_norm = adam_update(current_params, grads, current_state, config)
        return new_params, new_state, loss, grad_norm

    before = float(jnp.sum(jnp.square(params["value"])))
    params, state, _loss, norm = step(params, state)
    after = float(jnp.sum(jnp.square(params["value"])))
    assert after < before
    assert float(norm) > 1.0
    assert int(state.count) == 1
