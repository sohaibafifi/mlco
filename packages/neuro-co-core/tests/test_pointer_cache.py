"""Pointer decoder fixed-projection cache tests."""

import torch
from torch import Tensor

from neuro_co.core.models import PointerDecoder


def _forward_and_gradients(
    decoder: PointerDecoder,
    node_embs: Tensor,
    graph_emb: Tensor,
    dynamic_contexts: tuple[Tensor, Tensor],
    *,
    cached: bool,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, ...]]:
    batch, nodes, _ = node_embs.shape
    first_idx = torch.zeros(batch, dtype=torch.long)
    current_indices = (
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
    )
    mask = torch.ones(batch, nodes, dtype=torch.bool)
    mask[:, 0] = False
    weights = (
        torch.linspace(-0.7, 0.9, batch * nodes).view(batch, nodes),
        torch.linspace(0.8, -0.6, batch * nodes).view(batch, nodes),
    )
    decoder_cache = decoder.precompute(node_embs) if cached else None
    logits: list[Tensor] = []
    loss = node_embs.new_zeros(())
    for current_idx, dynamic_context, weight in zip(
        current_indices, dynamic_contexts, weights, strict=True
    ):
        step_logits = decoder(
            node_embs,
            graph_emb,
            first_idx,
            current_idx,
            mask,
            dynamic_context=dynamic_context,
            decoder_cache=decoder_cache,
        )
        logits.append(step_logits)
        loss = loss + (step_logits * weight.masked_fill(~mask, 0.0)).sum()

    differentiated = (
        node_embs,
        graph_emb,
        *dynamic_contexts,
        *tuple(decoder.parameters()),
    )
    gradients = torch.autograd.grad(loss, differentiated)
    return (logits[0], logits[1]), gradients


def test_cached_logits_and_gradients_match_uncached_decoder() -> None:
    torch.manual_seed(0)
    decoder = PointerDecoder(hidden_dim=16, num_heads=2)
    node_embs = torch.randn(2, 5, 16, requires_grad=True)
    graph_emb = torch.randn(2, 16, requires_grad=True)
    dynamic_contexts = (
        torch.tensor([[0.25], [0.5]], requires_grad=True),
        torch.tensor([[0.75], [1.0]], requires_grad=True),
    )
    state_before = {name: value.detach().clone() for name, value in decoder.state_dict().items()}

    uncached_logits, uncached_gradients = _forward_and_gradients(
        decoder,
        node_embs,
        graph_emb,
        dynamic_contexts,
        cached=False,
    )
    cached_logits, cached_gradients = _forward_and_gradients(
        decoder,
        node_embs,
        graph_emb,
        dynamic_contexts,
        cached=True,
    )

    for uncached, cached in zip(uncached_logits, cached_logits, strict=True):
        assert torch.equal(uncached, cached)
    for uncached, cached in zip(uncached_gradients, cached_gradients, strict=True):
        assert torch.allclose(uncached, cached, atol=1e-6, rtol=1e-5)
    assert tuple(decoder.state_dict()) == tuple(state_before)
    for name, value in decoder.state_dict().items():
        assert torch.equal(value, state_before[name])
