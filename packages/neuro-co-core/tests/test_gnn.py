"""Regression tests for the GNN encoder."""

from __future__ import annotations

import torch
from torch_geometric.nn import TransformerConv

from neuro_co.core.factory import make_model
from neuro_co.core.models import GNNEncoder, PointerDecoder
from neuro_co.problems.cvrp.env import CVRPEnv


def _encoder(**kwargs) -> GNNEncoder:
    return GNNEncoder(
        in_dim=3,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        k_sparse=3,
        dropout=0.1,
        **kwargs,
    )


def test_gnn_factory_uses_real_pyg_transformerconv_and_common_decoder() -> None:
    env = CVRPEnv(size=10)
    model = make_model(env, backbone="gnn", hidden_dim=32, num_layers=2, num_heads=4)

    assert isinstance(model.encoder, GNNEncoder)
    assert isinstance(model.decoder, PointerDecoder)
    assert all(isinstance(block.gnn, TransformerConv) for block in model.encoder.blocks)


def test_gnn_shapes_finite_outputs_and_gradients() -> None:
    torch.manual_seed(7)
    encoder = _encoder()
    features = torch.rand(3, 11, 3, requires_grad=True)

    node_embeddings, graph_embedding = encoder(features)

    assert node_embeddings.shape == (3, 11, 32)
    assert graph_embedding.shape == (3, 32)
    assert torch.isfinite(node_embeddings).all()
    assert torch.isfinite(graph_embedding).all()
    (node_embeddings.square().mean() + graph_embedding.square().mean()).backward()
    gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
    assert gradients
    assert features.grad is not None
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(features.grad).all()


def test_gnn_eval_is_deterministic() -> None:
    torch.manual_seed(11)
    encoder = _encoder().eval()
    features = torch.rand(2, 9, 3)

    with torch.inference_mode():
        first = encoder(features)
        second = encoder(features)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_sparse_edges_have_knn_and_bidirectional_depot_connectivity() -> None:
    encoder = _encoder()
    coordinates = torch.tensor(
        [
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.3, 0.0],
                [0.6, 0.0],
                [1.0, 0.0],
            ]
        ]
    )

    source, destination = encoder.sparse_edge_index(coordinates)

    nodes = coordinates.shape[1]
    assert source.shape == destination.shape == (1, nodes * 3 + 2 * (nodes - 1))
    edges = set(zip(source[0].tolist(), destination[0].tolist(), strict=True))
    for customer in range(1, nodes):
        assert (0, customer) in edges
        assert (customer, 0) in edges
    for node in range(nodes):
        assert int((destination[0, : nodes * 3] == node).sum()) == 3
        assert not any(
            int(source[0, index]) == int(destination[0, index]) for index in range(nodes * 3)
        )


def test_edge_feature_order_is_cos_then_sin() -> None:
    torch.manual_seed(13)
    encoder = _encoder()
    coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]])
    source = torch.tensor([[0, 0]])
    destination = torch.tensor([[1, 2]])

    observed = encoder.raw_geometric_edge_features(coordinates, source, destination)

    raw_distance = torch.tensor([[1.0, 2.0]])
    maximum = torch.tensor([[torch.sqrt(torch.tensor(5.0))]])
    normalized = raw_distance / maximum
    distance_features = encoder.distance_encoding(normalized)
    angles = torch.tensor([[0.0, torch.pi / 2]])
    expected = torch.cat(
        (
            distance_features,
            torch.cos(angles).unsqueeze(-1),
            torch.sin(angles).unsqueeze(-1),
        ),
        dim=-1,
    )

    assert observed.shape[-1] == 22
    assert torch.allclose(observed, expected, atol=1e-7, rtol=0.0)
    assert torch.equal(observed[..., -2], torch.cos(angles))
    assert torch.equal(observed[..., -1], torch.sin(angles))
