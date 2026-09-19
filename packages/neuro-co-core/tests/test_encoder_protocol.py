"""Encoder Protocol satisfaction tests + DI sanity."""

import torch
from torch import nn

from neuro_co.core.models import (
    AMEncoder,
    AttentionModel,
    ConstructivePolicy,
    Decoder,
    Encoder,
    PointerDecoder,
)
from neuro_co.core.models.encoder import pool_max, pool_mean


def test_am_encoder_satisfies_encoder_protocol() -> None:
    enc = AMEncoder(hidden_dim=16, num_layers=1, num_heads=2)
    assert isinstance(enc, Encoder)


def test_pointer_decoder_satisfies_decoder_protocol() -> None:
    dec = PointerDecoder(hidden_dim=16, num_heads=2)
    assert isinstance(dec, Decoder)


def test_pool_mean_max_shapes() -> None:
    x = torch.rand(3, 5, 8)
    assert pool_mean(x).shape == (3, 8)
    assert pool_max(x).shape == (3, 8)


class _IdentityEncoder(nn.Module):
    """Trivial Encoder Protocol impl: project to hidden_dim, mean-pool."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        h = self.lin(x)
        return h, h.mean(dim=1)


def test_constructive_policy_accepts_custom_encoder() -> None:
    """DI: any Encoder Protocol impl plugs in. No string flag, no registry."""
    enc = _IdentityEncoder(in_dim=2, hidden_dim=16)
    assert isinstance(enc, Encoder)
    dec = PointerDecoder(hidden_dim=16, num_heads=2)
    pol = ConstructivePolicy(encoder=enc, decoder=dec)

    coords = torch.rand(2, 4, 2)
    node_embs, graph_emb = pol.encode(coords)
    assert node_embs.shape == (2, 4, 16)
    assert graph_emb.shape == (2, 16)


def test_attention_model_factory_returns_constructive_policy() -> None:
    m = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    assert isinstance(m, ConstructivePolicy)
    assert isinstance(m.encoder, AMEncoder)
    assert isinstance(m.decoder, PointerDecoder)
