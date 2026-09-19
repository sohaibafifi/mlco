"""Parity and compilation tests for the pure-JAX Attention Model."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neuro_co.core.jax_backend import JaxAttentionModel  # noqa: E402
from neuro_co.core.models import AttentionModel  # noqa: E402


def _paired_models(in_dim: int = 3):
    torch.manual_seed(7)
    torch_model = AttentionModel(
        in_dim=in_dim,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    ).eval()
    jax_model = JaxAttentionModel(
        in_dim=in_dim,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    params = jax_model.from_torch_state_dict(torch_model.state_dict())
    return torch_model, jax_model, params


def test_encoder_matches_torch_with_same_weights() -> None:
    torch_model, jax_model, params = _paired_models()
    values = np.random.default_rng(11).random((3, 7, 3), dtype=np.float32)
    with torch.inference_mode():
        torch_nodes, torch_graph = torch_model.encode(torch.from_numpy(values))
    jax_nodes, jax_graph = jax.jit(jax_model.encode)(params, jnp.asarray(values))

    np.testing.assert_allclose(np.asarray(jax_nodes), torch_nodes.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(jax_graph), torch_graph.numpy(), rtol=2e-5, atol=2e-6)


def test_cvrp50_parameter_count_matches_torch() -> None:
    torch_model = AttentionModel(in_dim=3)
    jax_model = JaxAttentionModel(in_dim=3)
    params = jax_model.from_torch_state_dict(torch_model.state_dict())
    torch_count = sum(parameter.numel() for parameter in torch_model.parameters())
    jax_count = sum(array.size for array in jax.tree.leaves(params))
    assert jax_count == torch_count == 741_248


def test_pointer_decoder_matches_torch_with_dynamic_context() -> None:
    torch_model, jax_model, params = _paired_models()
    values = np.random.default_rng(13).random((2, 6, 3), dtype=np.float32)
    torch_features = torch.from_numpy(values)
    mask = torch.tensor(
        [[False, True, True, False, True, True], [True, False, True, True, False, True]]
    )
    first = torch.tensor([0, 0])
    current = torch.tensor([2, 3])
    dynamic = torch.tensor([[0.25], [0.75]])
    with torch.inference_mode():
        torch_nodes, torch_graph = torch_model.encode(torch_features)
        expected = torch_model.decode_step(
            torch_nodes,
            torch_graph,
            first,
            current,
            mask,
            dynamic_context=dynamic,
        )

    nodes, graph = jax_model.encode(params, jnp.asarray(values))
    cache = jax_model.precompute_decoder_cache(params, nodes)
    actual = jax.jit(jax_model.decode_step)(
        params,
        nodes,
        graph,
        jnp.asarray(first.numpy()),
        jnp.asarray(current.numpy()),
        jnp.asarray(mask.numpy()),
        dynamic_context=jnp.asarray(dynamic.numpy()),
        decoder_cache=cache,
    )

    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=3e-5, atol=3e-5)
    assert np.all(np.asarray(actual)[~mask.numpy()] == -1.0e9)


def test_cached_and_uncached_decoder_are_identical() -> None:
    _torch_model, model, params = _paired_models(in_dim=2)
    features = jax.random.uniform(jax.random.key(3), (4, 8, 2))
    nodes, graph = model.encode(params, features)
    mask = jnp.ones((4, 8), dtype=jnp.bool_).at[:, 0].set(False)
    first = jnp.zeros((4,), dtype=jnp.int32)
    current = jnp.zeros((4,), dtype=jnp.int32)
    uncached = model.decode_step(params, nodes, graph, first, current, mask)
    cached = model.decode_step(
        params,
        nodes,
        graph,
        first,
        current,
        mask,
        decoder_cache=model.precompute_decoder_cache(params, nodes),
    )
    np.testing.assert_array_equal(np.asarray(cached), np.asarray(uncached))


def test_all_parameter_gradients_are_finite() -> None:
    model = JaxAttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    params = model.init(jax.random.key(5))
    features = jax.random.uniform(jax.random.key(6), (2, 5, 2))
    mask = jnp.ones((2, 5), dtype=jnp.bool_).at[:, 0].set(False)
    indices = jnp.zeros((2,), dtype=jnp.int32)

    def objective(current_params):
        nodes, graph = model.encode(current_params, features)
        logits = model.decode_step(current_params, nodes, graph, indices, indices, mask)
        return jnp.sum(logits[:, 1:])

    grads = jax.jit(jax.grad(objective))(params)
    leaves = jax.tree.leaves(grads)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in leaves)


def test_parameter_gradients_match_torch() -> None:
    torch_model, jax_model, params = _paired_models()
    values = np.random.default_rng(17).random((2, 6, 3), dtype=np.float32)
    mask = np.ones((2, 6), dtype=np.bool_)
    first = np.asarray([0, 1], dtype=np.int64)
    current = np.asarray([2, 3], dtype=np.int64)
    dynamic = np.asarray([[0.25], [0.75]], dtype=np.float32)

    torch_nodes, torch_graph = torch_model.encode(torch.from_numpy(values))
    torch_logits = torch_model.decode_step(
        torch_nodes,
        torch_graph,
        torch.from_numpy(first),
        torch.from_numpy(current),
        torch.from_numpy(mask),
        dynamic_context=torch.from_numpy(dynamic),
    )
    torch_loss = torch.mean(torch.square(torch_logits / 10.0))
    torch_loss.backward()

    def objective(current_params):
        nodes, graph = jax_model.encode(current_params, jnp.asarray(values))
        logits = jax_model.decode_step(
            current_params,
            nodes,
            graph,
            jnp.asarray(first),
            jnp.asarray(current),
            jnp.asarray(mask),
            dynamic_context=jnp.asarray(dynamic),
        )
        return jnp.mean(jnp.square(logits / 10.0))

    jax_grads = jax.jit(jax.grad(objective))(params)
    for torch_parameter, jax_grad in zip(
        torch_model.parameters(), jax.tree.leaves(jax_grads), strict=True
    ):
        assert torch_parameter.grad is not None
        np.testing.assert_allclose(
            np.asarray(jax_grad),
            torch_parameter.grad.numpy(),
            rtol=5e-5,
            atol=5e-6,
        )


def test_bfloat16_compute_keeps_float32_parameters() -> None:
    model = JaxAttentionModel(
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        precision="bf16",
    )
    params = model.init(jax.random.key(8))
    nodes, graph = jax.jit(model.encode)(params, jax.random.uniform(jax.random.key(9), (2, 5, 2)))
    assert nodes.dtype == jnp.bfloat16
    assert graph.dtype == jnp.bfloat16
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))


def test_fp16_fails_clearly_on_cpu() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-specific validation")
    model = JaxAttentionModel(
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        precision="fp16",
    )
    params = model.init(jax.random.key(10))
    features = jax.random.uniform(jax.random.key(11), (2, 5, 2))
    with pytest.raises(ValueError, match="requires a non-CPU JAX backend"):
        model.encode(params, features)
