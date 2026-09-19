"""torch.compile verification.

Checks `AttentionModel.encode` and `.decode_step` compile cleanly and
produce numerically equivalent output to eager mode.
"""

import pytest
import torch

from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel


@pytest.fixture
def setup():
    torch.manual_seed(0)
    env = TSPEnv(size=8)
    model = AttentionModel(hidden_dim=32, num_layers=2, num_heads=4)
    model.eval()
    state = env.reset(4, generator=torch.Generator().manual_seed(0))
    return env, model, state


def test_encode_compiles(setup) -> None:
    _, model, state = setup
    with torch.no_grad():
        eager_ne, eager_ge = model.encode(state.coords)
        compiled = torch.compile(model.encode, mode="default", fullgraph=True)
        comp_ne, comp_ge = compiled(state.coords)
    assert torch.allclose(eager_ne, comp_ne, atol=1e-4)
    assert torch.allclose(eager_ge, comp_ge, atol=1e-4)


def test_decode_step_compiles(setup) -> None:
    env, model, state = setup
    with torch.no_grad():
        node_embs, graph_emb = model.encode(state.coords)
        mask = env.action_mask(state)
        eager = model.decode_step(node_embs, graph_emb, state.first, state.current, mask)
        compiled = torch.compile(model.decode_step, mode="default", fullgraph=True)
        comp = compiled(node_embs, graph_emb, state.first, state.current, mask)
    assert torch.allclose(eager, comp, atol=1e-4)


def test_env_step_compiles(setup) -> None:
    """Pure-functional env.step should compile without graph breaks."""
    env, _, state = setup
    action = torch.tensor([1, 2, 3, 4])
    eager_next, eager_r, eager_d = env.step(state, action)
    compiled_step = torch.compile(env.step, mode="default", fullgraph=True)
    comp_next, comp_r, comp_d = compiled_step(state, action)
    assert torch.allclose(eager_next.tour_length, comp_next.tour_length, atol=1e-5)
    assert torch.allclose(eager_r, comp_r, atol=1e-5)
    assert torch.equal(eager_d, comp_d)
