"""POMO-REINFORCE smoke tests."""

import torch
from torch import Tensor, nn

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.envs.cvrp import CVRPEnv
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel, PointerDecoder
from neuro_co.core.models.policy import ConstructivePolicy


def _build(size: int = 6, batch: int = 8, n_starts: int = 4):
    env = TSPEnv(size=size)
    model = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    algo = POMO(
        model=model,
        env=env,
        cfg=POMOConfig(batch_size=batch, n_starts=n_starts, eval_batch_size=batch),
    )
    return algo


class _StepSeparatedPolicy(ConstructivePolicy):
    """Expose distinct parameters for the forced and sampled decode states."""

    def __init__(self, size: int) -> None:
        inner = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
        super().__init__(inner.encoder, inner.decoder)
        self.forced_step_logits = nn.Parameter(torch.linspace(-0.3, 0.3, size))
        self.sampled_step_logits = nn.Parameter(torch.linspace(0.3, -0.3, size))

    def decode_step(
        self,
        node_embs: Tensor,
        graph_emb: Tensor,
        first_idx: Tensor,
        current_idx: Tensor,
        mask: Tensor,
        dynamic_context: Tensor | None = None,
        decoder_cache: object | None = None,
    ) -> Tensor:
        del node_embs, graph_emb, first_idx, dynamic_context, decoder_cache
        step_logits = (
            self.forced_step_logits if bool((current_idx == 0).all()) else self.sampled_step_logits
        )
        logits = step_logits.unsqueeze(0).expand(mask.shape[0], -1)
        return torch.where(mask, logits, logits.new_full((), -1e9))


def test_train_step_runs() -> None:
    algo = _build()
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    for k in ("loss", "reward_mean", "reward_max_per_group", "advantage_std"):
        assert k in m
        assert m[k] == m[k]  # not NaN


def test_forced_first_action_has_no_score_function_gradient() -> None:
    size = 5
    env = TSPEnv(size=size)
    model = _StepSeparatedPolicy(size)
    algo = POMO(
        model=model,
        env=env,
        cfg=POMOConfig(batch_size=1, n_starts=2, eval_batch_size=1),
    )

    _, sum_logp = algo._pomo_rollout(
        rng=torch.Generator().manual_seed(0),
        sample_actions=True,
    )
    sum_logp.sum().backward()

    assert model.forced_step_logits.grad is None
    assert model.sampled_step_logits.grad is not None


def test_pomo_rollout_precomputes_fixed_pointer_projections_once() -> None:
    env = CVRPEnv(size=6, capacity=20.0)
    model = AttentionModel(
        in_dim=env.encoder_in_dim,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
    )
    assert isinstance(model.decoder, PointerDecoder)
    decoder = model.decoder
    algo = POMO(
        model=model,
        env=env,
        cfg=POMOConfig(batch_size=2, n_starts=3, eval_batch_size=2),
    )
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
        algo._pomo_rollout(
            rng=torch.Generator().manual_seed(0),
            sample_actions=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls["decoder"] > 1
    assert calls["k_proj"] == 1
    assert calls["v_proj"] == 1
    assert calls["point_k"] == 1


def test_eval_step_runs() -> None:
    algo = _build()
    rng = torch.Generator().manual_seed(0)
    m = algo.eval_step(rng)
    assert m["eval_tour_length"] > 0


def test_n_starts_exceeds_valid_wraps_ok() -> None:
    """n_starts > valid first actions: cycle through valid set, no error.

    TSP-5 has 4 valid first actions (city 0 pre-visited). n_starts=10 wraps."""
    algo = _build(size=5, n_starts=10)
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)  # must not raise
    assert m["loss"] == m["loss"]  # not NaN


def test_short_training_loop() -> None:
    algo = _build(size=6, batch=8, n_starts=4)
    rng = torch.Generator().manual_seed(0)
    initial = algo.eval_step(rng)["eval_tour_length"]
    for _ in range(3):
        algo.train_step(rng)
    final = algo.eval_step(rng)["eval_tour_length"]
    assert final > 0
    assert final < initial * 3  # no divergence


def test_advantage_zero_mean_per_group() -> None:
    """Group baseline = mean over n_starts. Advantage = reward - baseline => mean 0 per group."""
    from neuro_co.core.algos.multistart import pomo_advantage

    reward = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # b=2, n_starts=3
    adv = pomo_advantage(reward, n_starts=3)
    assert torch.allclose(adv[:3].mean(), torch.tensor(0.0))
    assert torch.allclose(adv[3:].mean(), torch.tensor(0.0))
