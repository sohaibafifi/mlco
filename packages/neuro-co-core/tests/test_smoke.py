"""Core state and decoding primitives."""

from dataclasses import dataclass

import torch

from neuro_co.core import (
    NEG_INF,
    State,
    apply_mask,
    greedy,
    log_prob,
    register_state,
    sample,
)


@register_state
@dataclass(frozen=True, slots=True)
class _Toy(State):
    x: torch.Tensor
    y: torch.Tensor


def test_register_state_pytree() -> None:
    s = _Toy(x=torch.zeros(4), y=torch.ones(4))
    s2 = s.replace(x=torch.ones(4))
    assert torch.equal(s2.x, torch.ones(4))
    assert torch.equal(s2.y, torch.ones(4))


def test_apply_mask() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    out = apply_mask(logits, mask)
    assert out[0, 1].item() == NEG_INF
    assert out[0, 0].item() == 1.0
    assert out[0, 2].item() == 3.0


def test_greedy_decode() -> None:
    logits = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
    a = greedy(logits)
    assert a.tolist() == [1, 0]


def test_sample_decode_seeded() -> None:
    logits = torch.tensor([[0.1, 0.7, 0.2]])
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a1 = sample(logits, generator=g1)
    a2 = sample(logits, generator=g2)
    assert torch.equal(a1, a2)


def test_log_prob_matches_softmax() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    action = torch.tensor([2])
    expected = logits.log_softmax(-1)[0, 2]
    got = log_prob(logits, action)[0]
    assert torch.allclose(got, expected)
