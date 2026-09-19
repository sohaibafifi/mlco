"""AttentionModel tests: shapes + mask correctness."""

import torch

from neuro_co.core.models import AttentionModel
from neuro_co.problems.tsp.env import TSPEnv


def test_encode_shapes() -> None:
    m = AttentionModel(hidden_dim=32, num_layers=2, num_heads=4)
    coords = torch.rand(3, 10, 2)
    node_embs, graph_emb = m.encode(coords)
    assert node_embs.shape == (3, 10, 32)
    assert graph_emb.shape == (3, 32)


def test_decode_step_shapes_and_mask() -> None:
    m = AttentionModel(hidden_dim=32, num_layers=2, num_heads=4)
    env = TSPEnv(size=8)
    s = env.reset(4)
    node_embs, graph_emb = m.encode(s.coords)
    mask = env.action_mask(s)
    logits = m.decode_step(node_embs, graph_emb, s.first, s.current, mask)
    assert logits.shape == (4, 8)
    # Masked entries should be very negative.
    assert (logits[:, 0] < -1e6).all()
    # Permitted entries should not be NEG_INF.
    assert (logits[:, 1:] > -1e6).all()


def test_decode_step_accepts_dynamic_scalar_without_changing_legacy_calls() -> None:
    torch.manual_seed(0)
    model = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    env = TSPEnv(size=5)
    state = env.reset(2, generator=torch.Generator().manual_seed(1))
    node_embs, graph_emb = model.encode(state.coords)
    mask = env.action_mask(state)

    legacy = model.decode_step(node_embs, graph_emb, state.first, state.current, mask)
    explicit_none = model.decode_step(
        node_embs,
        graph_emb,
        state.first,
        state.current,
        mask,
        dynamic_context=None,
    )
    lower_capacity = model.decode_step(
        node_embs,
        graph_emb,
        state.first,
        state.current,
        mask,
        dynamic_context=torch.full((2, 1), 0.25),
    )
    higher_capacity = model.decode_step(
        node_embs,
        graph_emb,
        state.first,
        state.current,
        mask,
        dynamic_context=torch.full((2, 1), 0.75),
    )

    assert torch.equal(legacy, explicit_none)
    assert not torch.allclose(lower_capacity, higher_capacity)


def test_grad_flow() -> None:
    m = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    env = TSPEnv(size=5)
    s = env.reset(2)
    node_embs, graph_emb = m.encode(s.coords)
    mask = env.action_mask(s)
    logits = m.decode_step(node_embs, graph_emb, s.first, s.current, mask)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
