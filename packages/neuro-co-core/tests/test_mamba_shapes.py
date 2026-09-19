"""Shape invariants for SSMBlock + SSMEncoder."""

import pytest
import torch

from neuro_co.core.models import SSMEncoder
from neuro_co.core.models.encoders.ssm_block import SSMBlock

pytest.importorskip("mambapy")


@pytest.fixture(scope="module")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_ssm_block_preserves_shape(device: torch.device) -> None:
    b, n, d = 4, 32, 64
    block = SSMBlock(backend="mambapy", d_model=d, n_layers=2).to(device)
    x = torch.randn(b, n, d, device=device)
    y = block(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_ssm_block_rejects_bad_shape(device: torch.device) -> None:
    block = SSMBlock(backend="mambapy", d_model=16, n_layers=1).to(device)
    with pytest.raises(ValueError):
        block(torch.randn(8, 16, device=device))  # missing batch dim


def test_ssm_block_rejects_dim_mismatch(device: torch.device) -> None:
    block = SSMBlock(backend="mambapy", d_model=16, n_layers=1).to(device)
    with pytest.raises(ValueError):
        block(torch.randn(2, 8, 32, device=device))  # D=32 but d_model=16


def test_ssm_encoder_shapes(device: torch.device) -> None:
    enc = SSMEncoder(
        backend="mambapy", in_dim=2, hidden_dim=64, num_layers=2, bidirectional=True
    ).to(device)
    x = torch.rand(2, 20, 2, device=device)
    node_embs, graph_emb = enc(x)
    assert node_embs.shape == (2, 20, 64)
    assert graph_emb.shape == (2, 64)


def test_ssm_encoder_unidirectional(device: torch.device) -> None:
    enc = SSMEncoder(
        backend="mambapy", in_dim=3, hidden_dim=32, num_layers=1, bidirectional=False
    ).to(device)
    x = torch.rand(2, 16, 3, device=device)
    node_embs, graph_emb = enc(x)
    assert node_embs.shape == (2, 16, 32)
    assert graph_emb.shape == (2, 32)
