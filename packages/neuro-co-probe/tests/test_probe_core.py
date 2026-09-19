"""Encoder probes and discovered directions on core states."""

import torch

from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel
from neuro_co.probe.discovered import DiscoveredDirections, discover_directions
from neuro_co.probe.probes import ProbeResult, encoder_layer_count, fit_concept_probes


def _left_half(state) -> torch.Tensor:
    """Concept: node lies in the left half of the unit square (x < 0.5)."""
    return (state.coords[..., 0] < 0.5).long()  # [B, N] in {0,1}


def _top_half(state) -> torch.Tensor:
    return (state.coords[..., 1] < 0.5).long()


CONCEPTS = {"left_half": _left_half, "top_half": _top_half}


def _setup(size: int = 10, batch: int = 64):
    torch.manual_seed(0)
    env = TSPEnv(size=size)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=32, num_layers=2, num_heads=4)
    state = env.reset(batch, generator=torch.Generator().manual_seed(0))
    return model, env, state


def test_encoder_layer_count() -> None:
    model, _, _ = _setup()
    assert encoder_layer_count(model) == 2  # num_layers


def test_fit_concept_probes_runs() -> None:
    model, env, state = _setup()
    results = fit_concept_probes(model, env, state, concepts=CONCEPTS, epochs=50)
    assert results
    assert all(isinstance(r, ProbeResult) for r in results)
    names = {r.concept for r in results}
    assert names == {"left_half", "top_half"}
    for r in results:
        assert 0.0 <= r.val_acc <= 1.0
        assert r.layer == -1  # final output only


def test_fit_concept_probes_layers() -> None:
    model, env, state = _setup()
    results = fit_concept_probes(
        model, env, state, concepts=CONCEPTS, epochs=30, layer_indices=[0, 1]
    )
    layers = {r.layer for r in results}
    assert -1 in layers and 0 in layers and 1 in layers  # final + 2 intermediate


def test_probe_recovers_coordinate_concept() -> None:
    """Encoder sees coords directly → left_half should be highly decodable."""
    model, env, state = _setup(size=12, batch=128)
    results = fit_concept_probes(model, env, state, concepts={"left_half": _left_half}, epochs=150)
    final = next(r for r in results if r.layer == -1)
    assert final.val_acc > 0.8  # coordinate concept is linearly present


def test_discover_directions_pca() -> None:
    model, env, state = _setup()
    d = discover_directions(model, env, state, concepts=CONCEPTS, n_components=5, method="pca")
    assert isinstance(d, DiscoveredDirections)
    assert d.method == "pca"
    assert d.scores.shape[1] == d.n_components
    assert len(d.alignment) == d.n_components
    assert d.embed_dim == 32


def test_explain_policy_end_to_end(tmp_path):
    """Capstone orchestrator: attribution + faithfulness on core, writes JSON."""

    from neuro_co.core.envs.tsp import TSPEnv
    from neuro_co.core.models import AttentionModel
    from neuro_co.probe import explain_policy

    env = TSPEnv(size=6)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    payload = explain_policy(
        model, env, num_instances=4, top_k=3, method="gradient", output_dir=str(tmp_path)
    )
    assert payload["num_steps"] == 5
    assert "deletion" in payload["faithfulness"]
    assert (tmp_path / "explanation.json").exists()
    assert (tmp_path / "instances.pt").exists()  # coords persisted
