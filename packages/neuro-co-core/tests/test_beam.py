"""Beam search smoke tests."""

import torch

from neuro_co.core.decode import beam_search
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel


def test_beam_runs_on_tsp() -> None:
    env = TSPEnv(size=6)
    model = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    model.eval()
    state = env.reset(2, generator=torch.Generator().manual_seed(0))
    node_embs, graph_emb = model.encode(state.coords)
    orig_b = state.coords.size(0)

    def step_logits(s, mask):
        bn = s.coords.size(0)
        ne = node_embs.repeat_interleave(bn // orig_b, dim=0)
        ge = graph_emb.repeat_interleave(bn // orig_b, dim=0)
        return model.decode_step(ne, ge, s.first, s.current, mask)

    initial_mask = env.action_mask(state)
    with torch.no_grad():
        best_reward, best_actions = beam_search(
            initial_state=state,
            initial_mask=initial_mask,
            step_logits_fn=step_logits,
            env_step_fn=env.step,
            env_mask_fn=env.action_mask,
            n_steps=5,
            beam_width=3,
        )
    assert best_reward.shape == (2,)
    assert best_actions.shape == (2, 5)
    # Reward should be <= 0 (TSP reward is -tour_length).
    assert (best_reward <= 0).all()


def test_beam_better_or_equal_than_greedy() -> None:
    """Beam with width 1 should ~match greedy; wider beam should not be worse."""
    env = TSPEnv(size=5)
    model = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    model.eval()
    state = env.reset(2, generator=torch.Generator().manual_seed(0))
    node_embs, graph_emb = model.encode(state.coords)

    orig_b = state.coords.size(0)

    def step_logits(s, mask):
        bn = s.coords.size(0)
        ne = node_embs.repeat_interleave(bn // orig_b, dim=0)
        ge = graph_emb.repeat_interleave(bn // orig_b, dim=0)
        return model.decode_step(ne, ge, s.first, s.current, mask)

    initial_mask = env.action_mask(state)
    with torch.no_grad():
        r1, _ = beam_search(
            initial_state=state,
            initial_mask=initial_mask,
            step_logits_fn=step_logits,
            env_step_fn=env.step,
            env_mask_fn=env.action_mask,
            n_steps=4,
            beam_width=1,
        )
        r3, _ = beam_search(
            initial_state=state,
            initial_mask=initial_mask,
            step_logits_fn=step_logits,
            env_step_fn=env.step,
            env_mask_fn=env.action_mask,
            n_steps=4,
            beam_width=3,
        )
    # Wider beam should never be worse (higher = better since reward = -length).
    assert (r3 >= r1 - 1e-5).all()
