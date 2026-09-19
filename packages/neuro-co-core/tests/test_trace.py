"""rollout_trace: the introspection primitive used by xai/cax/probe."""

import torch

from neuro_co.core.envs.cvrp import CVRPEnv
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel
from neuro_co.core.models.decoders.pointer import PointerDecoder
from neuro_co.core.trace import Trace, layer_activations, rollout_trace


def _model(env):
    return AttentionModel(
        in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2
    ).eval()


def test_trace_records_every_step_tsp() -> None:
    env = TSPEnv(size=8)
    model = _model(env)
    state = env.reset(4, generator=torch.Generator().manual_seed(0))
    tr = rollout_trace(model, env, state)
    assert isinstance(tr, Trace)
    assert len(tr.steps) == env.max_steps(state)  # n-1 fixed
    assert tr.node_embs.shape == (4, 8, 16)
    assert tr.graph_emb.shape == (4, 16)
    assert tr.actions.shape == (4, len(tr.steps))
    assert tr.reward is not None
    assert tr.reward.shape == (4,)
    tour_length = tr.tour_length
    assert tour_length is not None
    assert (tour_length > 0).all()


def test_trace_precomputes_fixed_pointer_projections_once() -> None:
    env = TSPEnv(size=8)
    model = _model(env)
    assert isinstance(model.decoder, PointerDecoder)
    decoder = model.decoder
    calls = {"decoder": 0, "k_proj": 0, "v_proj": 0, "point_k": 0}

    def count(name: str):
        def hook(*_args) -> None:
            calls[name] += 1

        return hook

    handles = [
        decoder.register_forward_hook(count("decoder")),
        decoder.k_proj.register_forward_hook(count("k_proj")),
        decoder.v_proj.register_forward_hook(count("v_proj")),
        decoder.point_k.register_forward_hook(count("point_k")),
    ]
    try:
        state = env.reset(2, generator=torch.Generator().manual_seed(0))
        rollout_trace(model, env, state)
    finally:
        for handle in handles:
            handle.remove()

    assert calls["decoder"] == env.size - 1
    assert calls["k_proj"] == 1
    assert calls["v_proj"] == 1
    assert calls["point_k"] == 1


def test_trace_step_fields_shapes() -> None:
    env = TSPEnv(size=6)
    model = _model(env)
    state = env.reset(3, generator=torch.Generator().manual_seed(0))
    tr = rollout_trace(model, env, state)
    s0 = tr.steps[0]
    assert s0.mask.shape == (3, 6)
    assert s0.logits.shape == (3, 6)
    assert s0.action.shape == (3,)
    assert s0.logp.shape == (3,)
    assert s0.active.all()  # nobody finished at step 0


def test_trace_masks_finished_cvrp() -> None:
    """Variable-length env: late steps have some inactive episodes."""
    env = CVRPEnv(size=6, capacity=20.0)
    model = _model(env)
    state = env.reset(16, generator=torch.Generator().manual_seed(0))
    tr = rollout_trace(model, env, state)
    # Some episode finishes before the last recorded step -> active shrinks.
    actives = torch.stack([s.active for s in tr.steps], dim=1)  # (b, T)
    assert actives[:, 0].all()
    assert actives.float().sum() < actives.numel()  # at least one finished early


def test_trace_grad_flows_to_features() -> None:
    """Attribution needs d(logit)/d(input). grad=True makes features a leaf."""
    env = TSPEnv(size=6)
    model = _model(env)
    state = env.reset(2, generator=torch.Generator().manual_seed(0))
    tr = rollout_trace(model, env, state, grad=True)
    assert tr.features.requires_grad
    # Backprop a scalar from step-0 logits to the input features.
    tr.steps[0].logits.sum().backward()
    assert tr.features.grad is not None
    assert torch.isfinite(tr.features.grad).all()


def test_trace_sample_deterministic_with_seed() -> None:
    env = TSPEnv(size=6)
    model = _model(env)
    state = env.reset(2, generator=torch.Generator().manual_seed(0))
    a = rollout_trace(model, env, state, decode="sample", rng=torch.Generator().manual_seed(1))
    b = rollout_trace(model, env, state, decode="sample", rng=torch.Generator().manual_seed(1))
    assert torch.equal(a.actions, b.actions)


def test_layer_activations_per_block() -> None:
    """Probe primitive: one activation per encoder block, right shape."""
    env = TSPEnv(size=8)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=3, num_heads=2)
    state = env.reset(4, generator=torch.Generator().manual_seed(0))
    feats = env.build_features(state)
    acts = layer_activations(model.encoder, feats)
    assert len(acts) == 3  # one per block (num_layers)
    for a in acts:
        assert a.shape == (4, 8, 16)


def test_trace_greedy_matches_eval() -> None:
    """Greedy trace reward == algo greedy eval reward (same rollout)."""
    from neuro_co.core.algos.pomo import _greedy_rollout

    env = TSPEnv(size=8)
    model = _model(env)
    state = env.reset(8, generator=torch.Generator().manual_seed(3))
    tr = rollout_trace(model, env, state)
    ref = _greedy_rollout(model, env, state)
    assert tr.reward is not None
    assert torch.allclose(tr.reward, ref, atol=1e-5)
