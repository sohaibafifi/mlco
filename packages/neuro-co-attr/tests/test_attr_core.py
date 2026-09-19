"""attr on neuro-co-core: attribution + faithfulness, no TensorDict."""

import torch

from neuro_co.attr.attribution import (
    AttributionTrace,
    contrastive_attribution,
    deeplift_attribution,
    gradient_attribution,
    integrated_gradients,
)
from neuro_co.attr.faithfulness import (
    deletion_flip_rate,
    sanity_check,
    sufficiency_keep_rate,
)
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel


def _setup(size: int = 8, batch: int = 16):
    torch.manual_seed(0)
    env = TSPEnv(size=size)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    state = env.reset(batch, generator=torch.Generator().manual_seed(0))
    return model, env, state


def test_gradient_attribution_shapes() -> None:
    model, env, state = _setup()
    tr = gradient_attribution(model, env, state, top_k=3)
    assert isinstance(tr, AttributionTrace)
    assert tr.num_steps == env.max_steps(state)  # n-1
    assert tr.node_scores.shape == (16, tr.num_steps, 8)
    assert tr.top_k_nodes.shape == (16, tr.num_steps, 3)
    assert tr.feature_scores is not None
    assert tr.feature_scores.shape == (16, tr.num_steps, 8, env.encoder_in_dim)
    assert torch.isfinite(tr.node_scores).all()


def test_contrastive_attribution_runs() -> None:
    model, env, state = _setup()
    tr = contrastive_attribution(model, env, state, top_k=3)
    assert tr.node_scores.shape[-1] == 8
    assert torch.isfinite(tr.node_scores).all()


def test_integrated_gradients_runs() -> None:
    model, env, state = _setup()
    tr = integrated_gradients(model, env, state, top_k=3, ig_steps=8, baseline="zero")
    assert tr.num_steps == env.max_steps(state)
    assert torch.isfinite(tr.node_scores).all()


def test_deeplift_runs_and_reports_completeness() -> None:
    model, env, state = _setup(size=6, batch=4)
    tr = deeplift_attribution(model, env, state, top_k=3, max_steps=2, baseline="zero")
    assert tr.node_scores.shape == (4, 2, 6)
    assert tr.feature_scores is not None
    assert tr.feature_scores.shape == (4, 2, 6, env.encoder_in_dim)
    assert tr.convergence_delta is not None
    assert tr.convergence_delta.shape == (4, 2)
    assert tr.relative_convergence_delta is not None
    assert tr.relative_convergence_delta.shape == (4, 2)
    assert torch.isfinite(tr.node_scores).all()
    assert torch.isfinite(tr.convergence_delta).all()


def test_deeplift_baseline_validation() -> None:
    model, env, state = _setup(size=6, batch=4)
    try:
        deeplift_attribution(model, env, state, baseline="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_ig_baseline_validation() -> None:
    model, env, state = _setup()
    try:
        integrated_gradients(model, env, state, baseline="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_deletion_and_sufficiency() -> None:
    model, env, state = _setup()
    tr = gradient_attribution(model, env, state, top_k=3)
    d = deletion_flip_rate(tr, model, env, state, top_k=3)
    s = sufficiency_keep_rate(tr, model, env, state, top_k=3)
    assert 0.0 <= d.mean_flip_rate <= 1.0
    assert 0.0 <= s.mean_keep_rate <= 1.0
    assert d.num_steps > 0 and s.num_steps > 0


def test_sanity_check_runs() -> None:
    model, env, state = _setup()
    tr = gradient_attribution(model, env, state, top_k=3)
    rep = sanity_check(tr, model, env, state, top_k=3, num_trials=1)
    assert 0.0 <= rep.mean_jaccard <= 1.0
    assert rep.mode == "random_weights"
