"""Dynamic decoder context propagation for state-dependent routing inputs."""

import torch
from torch import Tensor

from neuro_co.core.algos.pomo import POMO, POMOConfig, _greedy_rollout
from neuro_co.core.algos.ppo import PPO, PPOConfig
from neuro_co.core.algos.reinforce import _rollout
from neuro_co.core.env import get_dynamic_decoder_context
from neuro_co.core.models import AttentionModel
from neuro_co.core.models.policy import ConstructivePolicy
from neuro_co.core.trace import rollout_trace
from neuro_co.problems.cvrp.env import CVRPEnv
from neuro_co.problems.tsp.env import TSPEnv


class _RecordingPolicy(ConstructivePolicy):
    def __init__(self, env: CVRPEnv) -> None:
        inner = AttentionModel(
            in_dim=env.encoder_in_dim,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
        )
        super().__init__(inner.encoder, inner.decoder)
        self.contexts: list[Tensor | None] = []

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
        self.contexts.append(None if dynamic_context is None else dynamic_context.detach().clone())
        return super().decode_step(
            node_embs,
            graph_emb,
            first_idx,
            current_idx,
            mask,
            dynamic_context=dynamic_context,
            decoder_cache=decoder_cache,
        )


def _env() -> CVRPEnv:
    return CVRPEnv(size=4, capacity=10.0, max_demand=4)


def _assert_capacity_contexts(contexts: list[Tensor | None]) -> None:
    observed = [context for context in contexts if context is not None]
    assert observed
    assert all(context.ndim == 2 and context.shape[1] == 1 for context in observed)
    assert all(bool(((0.0 <= context) & (context <= 1.0)).all()) for context in observed)
    assert any(bool((context < 1.0).any()) for context in observed[1:])


def test_optional_context_preserves_environments_without_dynamic_state() -> None:
    env = TSPEnv(size=5)
    state = env.reset(2, generator=torch.Generator().manual_seed(0))
    assert get_dynamic_decoder_context(env, state) is None


def test_reinforce_rollout_forwards_cvrp_capacity_context() -> None:
    env = _env()
    model = _RecordingPolicy(env)
    state = env.reset(2, generator=torch.Generator().manual_seed(0))

    _rollout(model, env, state, sample_actions=False, rng=None)

    _assert_capacity_contexts(model.contexts)


def test_pomo_training_and_greedy_rollouts_forward_cvrp_capacity_context() -> None:
    env = _env()
    model = _RecordingPolicy(env)
    algo = POMO(
        model=model,
        env=env,
        cfg=POMOConfig(batch_size=2, n_starts=2, eval_batch_size=2),
    )

    algo._pomo_rollout(torch.Generator().manual_seed(0), sample_actions=False)
    _assert_capacity_contexts(model.contexts)

    model.contexts.clear()
    state = env.reset(2, generator=torch.Generator().manual_seed(1))
    _greedy_rollout(model, env, state)
    _assert_capacity_contexts(model.contexts)


def test_ppo_rollout_and_update_forward_cvrp_capacity_context() -> None:
    env = _env()
    model = _RecordingPolicy(env)
    algo = PPO(
        model=model,
        env=env,
        cfg=PPOConfig(batch_size=2, eval_batch_size=2, hidden_dim=16, epochs_per_rollout=1),
    )
    state = env.reset(2, generator=torch.Generator().manual_seed(0))
    trajectory = algo._rollout(state, rng=None)
    _assert_capacity_contexts(model.contexts)

    model.contexts.clear()
    algo._ppo_update_inner(state, trajectory)
    _assert_capacity_contexts(model.contexts)


def test_trace_forwards_cvrp_capacity_context() -> None:
    env = _env()
    model = _RecordingPolicy(env).eval()
    state = env.reset(2, generator=torch.Generator().manual_seed(0))

    rollout_trace(model, env, state)

    _assert_capacity_contexts(model.contexts)
