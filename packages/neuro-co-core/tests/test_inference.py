"""Exported actions must reproduce feasible routes and evaluation costs."""

import pytest
import torch

from neuro_co.core.algos.pomo import _greedy_rollout
from neuro_co.core.envs.cvrp import CVRPEnv
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.inference import greedy_rollout_actions
from neuro_co.core.models import AttentionModel
from neuro_co.core.models.decoders.pointer import PointerDecoder


def _model(env):
    torch.manual_seed(7)
    return AttentionModel(
        in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2
    ).eval()


@pytest.mark.parametrize("env", [TSPEnv(size=7), CVRPEnv(size=6, capacity=15.0)])
def test_exported_actions_are_feasible_and_match_greedy_cost(env) -> None:
    model = _model(env)
    state = env.reset(4, generator=torch.Generator().manual_seed(3))
    actions = greedy_rollout_actions(model, env, state)
    with torch.no_grad():
        reference = _greedy_rollout(model, env, state)

    finished = torch.zeros(4, dtype=torch.bool)
    reward = torch.zeros(4)
    initial_state = state
    for action in actions.unbind(dim=1):
        assert torch.equal(action == -1, finished)
        active = ~finished
        safe_action = action.clamp_min(0)
        allowed = env.action_mask(state).gather(1, safe_action[:, None]).squeeze(1)
        assert allowed[active].all()
        state, step_reward, done = env.step(state, safe_action)
        reward += step_reward * active
        finished |= done

    assert finished.all()
    torch.testing.assert_close(reward, reference)
    for row, selected in enumerate(actions):
        selected = selected[selected >= 0]
        if isinstance(env, TSPEnv):
            assert selected.sort().values.tolist() == list(range(1, env.size))
            route = torch.cat([initial_state.first[row : row + 1], selected])
            xy = initial_state.coords[row, route]
            length = torch.linalg.vector_norm(xy - xy.roll(1, 0), dim=-1).sum()
        else:
            assert selected[-1] == 0
            assert selected[selected != 0].sort().values.tolist() == list(range(1, env.size + 1))
            route = torch.cat([torch.zeros(1, dtype=torch.long), selected])
            xy = initial_state.coords[row, route]
            length = torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1).sum()
        torch.testing.assert_close(length, -reference[row])


def test_export_pads_only_finished_rows(monkeypatch) -> None:
    env = CVRPEnv(size=2, capacity=3.0)
    state = env.reset(2).replace(demand=torch.tensor([[0.0, 1.0, 1.0], [0.0, 3.0, 3.0]]))
    model = _model(env)

    def highest_feasible(_nodes, _graph, _first, _current, mask, **_kwargs):
        return torch.arange(mask.shape[1], dtype=torch.float32).expand_as(mask)

    monkeypatch.setattr(model, "decode_step", highest_feasible)
    actions = greedy_rollout_actions(model, env, state)
    assert actions.tolist() == [[2, 1, 0, -1], [2, 0, 1, 0]]


def test_export_caches_projections_and_restores_modes_on_failure(monkeypatch) -> None:
    env = TSPEnv(size=5)
    model = _model(env).train()
    assert isinstance(model.decoder, PointerDecoder)
    decoder = model.decoder
    decoder.eval()
    original_modes = [module.training for module in model.modules()]
    projection_calls = []

    def record_projection(_module, _inputs, output):
        assert not torch.is_grad_enabled()
        assert not model.training
        assert not output.requires_grad
        projection_calls.append(1)

    handle = decoder.k_proj.register_forward_hook(record_projection)
    try:
        greedy_rollout_actions(model, env, env.reset(2))
        assert len(projection_calls) == 1
        assert [module.training for module in model.modules()] == original_modes
        monkeypatch.setattr(env, "max_steps", lambda _state: 1)
        with pytest.raises(RuntimeError, match="did not finish within 1 steps"):
            greedy_rollout_actions(model, env, env.reset(2))
        assert [module.training for module in model.modules()] == original_modes
    finally:
        handle.remove()
