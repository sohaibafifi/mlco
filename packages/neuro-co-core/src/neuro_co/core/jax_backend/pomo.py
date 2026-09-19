"""Pure-JAX POMO rollout, REINFORCE loss, and gradient step."""

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp

from .am import AttentionModelParams, JaxAttentionModel
from .optim import AdamConfig, AdamState, adam_update, init_adam


class POMOLossMetrics(NamedTuple):
    reward_mean: jax.Array
    reward_max_per_group: jax.Array
    advantage_std: jax.Array


class POMOTrainMetrics(NamedTuple):
    loss: jax.Array
    reward_mean: jax.Array
    reward_max_per_group: jax.Array
    advantage_std: jax.Array
    grad_norm: jax.Array


class POMOTrainState(NamedTuple):
    params: AttentionModelParams
    optimizer_state: AdamState


@dataclass(frozen=True, slots=True)
class JaxPOMO:
    """Problem-independent POMO training built from a JAX model and environment."""

    model: JaxAttentionModel
    env: Any
    n_starts: int
    optimizer: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        if self.n_starts <= 0:
            raise ValueError("n_starts must be positive")
        env_in_dim = getattr(self.env, "encoder_in_dim", None)
        if env_in_dim != self.model.in_dim:
            raise ValueError(
                f"model in_dim={self.model.in_dim} does not match env encoder_in_dim={env_in_dim}"
            )

    def init(self, key: jax.Array) -> POMOTrainState:
        """Initialize model parameters and Adam moments."""

        params = self.model.init(key)
        return POMOTrainState(params=params, optimizer_state=init_adam(params))

    def rollout(
        self,
        params: AttentionModelParams,
        problem_state: Any,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Run sampled training rollouts and return ``(batch, starts)`` tensors."""

        encoder_key, first_key, scan_key = jax.random.split(key, 3)
        features = self.env.build_features(problem_state)
        batch = features.shape[0]
        node_embs, graph_emb = self.model.encode(
            params,
            features,
            training=True,
            key=encoder_key,
        )
        node_embs = jnp.repeat(node_embs, self.n_starts, axis=0)
        graph_emb = jnp.repeat(graph_emb, self.n_starts, axis=0)
        decoder_cache = self.model.precompute_decoder_cache(params, node_embs)

        first_mask = self.env.pomo_first_mask(problem_state)
        first_actions = distinct_first_actions(first_mask, self.n_starts, first_key)
        expanded_state = jax.tree.map(
            lambda value: jnp.repeat(value, self.n_starts, axis=0), problem_state
        )
        expanded_state, reward, done = self.env.step(expanded_state, first_actions)
        batch_starts = batch * self.n_starts
        initial = (
            expanded_state,
            reward,
            jnp.zeros((batch_starts,), dtype=jnp.float32),
            done,
            scan_key,
        )

        def body(carry: tuple[Any, ...], _unused: None) -> tuple[tuple[Any, ...], None]:
            state, total_reward, sum_log_probability, done_acc, rng = carry
            active = ~done_acc
            mask = self.env.action_mask(state)
            first_idx, current_idx = self.env.decoder_context(state)
            logits = self.model.decode_step(
                params,
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=_dynamic_decoder_context(self.env, state),
                decoder_cache=decoder_cache,
            )
            rng, action_key = jax.random.split(rng)
            stable_logits = logits.astype(jnp.float32)
            action = jax.random.categorical(action_key, stable_logits, axis=-1).astype(jnp.int32)
            selected_log_probability = jnp.take_along_axis(
                jax.nn.log_softmax(stable_logits, axis=-1), action[:, None], axis=1
            )[:, 0]
            sum_log_probability = sum_log_probability + jnp.where(
                active, selected_log_probability, 0.0
            )
            state, step_reward, step_done = self.env.step(state, action)
            total_reward = total_reward + jnp.where(active, step_reward, 0.0)
            return (
                state,
                total_reward,
                sum_log_probability,
                done_acc | step_done,
                rng,
            ), None

        remaining_steps = self.env.max_steps(expanded_state) - 1
        final, _ = jax.lax.scan(body, initial, xs=None, length=remaining_steps)
        reward = final[1].reshape(batch, self.n_starts)
        sum_log_probability = final[2].reshape(batch, self.n_starts)
        return reward, sum_log_probability

    def loss(
        self,
        params: AttentionModelParams,
        problem_state: Any,
        key: jax.Array,
    ) -> tuple[jax.Array, POMOLossMetrics]:
        """Compute the POMO REINFORCE objective for one explicit problem batch."""

        reward, sum_log_probability = self.rollout(
            params,
            problem_state,
            key,
        )
        loss = pomo_loss(reward, sum_log_probability)
        advantage = pomo_advantage(reward)
        return loss, POMOLossMetrics(
            reward_mean=jnp.mean(reward),
            reward_max_per_group=jnp.mean(jnp.max(reward, axis=1)),
            advantage_std=jnp.std(advantage),
        )

    def train_step(
        self,
        train_state: POMOTrainState,
        problem_state: Any,
        key: jax.Array,
        *,
        learning_rate: float | jax.Array | None = None,
    ) -> tuple[POMOTrainState, POMOTrainMetrics]:
        """Differentiate the POMO objective and apply one optimizer update."""

        (loss, loss_metrics), grads = jax.value_and_grad(self.loss, has_aux=True)(
            train_state.params, problem_state, key
        )
        params, optimizer_state, grad_norm = adam_update(
            train_state.params,
            grads,
            train_state.optimizer_state,
            self.optimizer,
            learning_rate=learning_rate,
        )
        return (
            POMOTrainState(params=params, optimizer_state=optimizer_state),
            POMOTrainMetrics(
                loss=loss,
                reward_mean=loss_metrics.reward_mean,
                reward_max_per_group=loss_metrics.reward_max_per_group,
                advantage_std=loss_metrics.advantage_std,
                grad_norm=grad_norm,
            ),
        )

    def sample_train_step(
        self,
        train_state: POMOTrainState,
        key: jax.Array,
        batch_size: int,
        *,
        learning_rate: float | jax.Array | None = None,
    ) -> tuple[POMOTrainState, POMOTrainMetrics]:
        """Generate a fresh problem batch and apply one training step.

        ``batch_size`` determines array shapes and should be captured by a
        closure when this method is JIT-compiled.
        """

        data_key, rollout_key = jax.random.split(key)
        problem_state = self.env.reset(data_key, batch_size)
        return self.train_step(
            train_state,
            problem_state,
            rollout_key,
            learning_rate=learning_rate,
        )

    def greedy_rollout(self, params: AttentionModelParams, problem_state: Any) -> jax.Array:
        """Decode one greedy solution per problem without forced POMO starts."""

        reward, _ = self._greedy_rollout(params, problem_state, export_actions=False)
        return reward

    def greedy_rollout_actions(self, params: AttentionModelParams, problem_state: Any) -> jax.Array:
        """Return action indices shaped ``(batch, max_steps)``, padded with -1.

        Actions follow the environment's convention. The initial node and any
        implicit closing edge are not included. Completed rows receive -1 in
        subsequent steps.
        """

        _, actions = self._greedy_rollout(params, problem_state, export_actions=True)
        return cast(jax.Array, actions)

    def _greedy_rollout(
        self,
        params: AttentionModelParams,
        problem_state: Any,
        *,
        export_actions: bool,
    ) -> tuple[jax.Array, jax.Array | None]:
        features = self.env.build_features(problem_state)
        node_embs, graph_emb = self.model.encode(params, features, training=False)
        decoder_cache = self.model.precompute_decoder_cache(params, node_embs)
        batch = features.shape[0]
        initial = (
            problem_state,
            jnp.zeros((batch,), dtype=jnp.float32),
            jnp.zeros((batch,), dtype=jnp.bool_),
        )

        def body(carry: tuple[Any, ...], _unused: None) -> tuple[tuple[Any, ...], jax.Array | None]:
            state, total_reward, done_acc = carry
            active = ~done_acc
            mask = self.env.action_mask(state)
            first_idx, current_idx = self.env.decoder_context(state)
            logits = self.model.decode_step(
                params,
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=_dynamic_decoder_context(self.env, state),
                decoder_cache=decoder_cache,
            )
            action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            state, reward, done = self.env.step(state, action)
            total_reward = total_reward + jnp.where(active, reward, 0.0)
            actions = jnp.where(active, action, -1) if export_actions else None
            return (state, total_reward, done_acc | done), actions

        final, actions = jax.lax.scan(
            body, initial, xs=None, length=self.env.max_steps(problem_state)
        )
        return final[1], jnp.swapaxes(actions, 0, 1) if actions is not None else None


def distinct_first_actions(mask: jax.Array, n_starts: int, key: jax.Array) -> jax.Array:
    """Select valid starts, cycling when fewer than ``n_starts`` are available."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got {mask.shape}")
    try:
        has_valid_start = bool(jnp.all(jnp.any(mask, axis=1)))
    except jax.errors.TracerBoolConversionError:
        has_valid_start = True
    if not has_valid_start:
        raise ValueError("a problem has zero permitted first actions")
    scores = jax.random.uniform(key, mask.shape)
    scores = jnp.where(mask, scores, -1.0)
    permutation = jnp.argsort(-scores, axis=1)
    valid_count = jnp.maximum(jnp.sum(mask, axis=1), 1)
    positions = jnp.broadcast_to(jnp.arange(n_starts), (mask.shape[0], n_starts))
    wrapped = positions % valid_count[:, None]
    actions = jnp.take_along_axis(permutation, wrapped, axis=1)
    return actions.reshape(-1).astype(jnp.int32)


def pomo_advantage(reward: jax.Array) -> jax.Array:
    """Return the detached reward minus group mean baseline."""

    if reward.ndim != 2:
        raise ValueError(f"reward must have shape (batch, starts), got {reward.shape}")
    centered = reward - jnp.mean(reward, axis=1, keepdims=True)
    return jax.lax.stop_gradient(centered)


def pomo_loss(reward: jax.Array, sum_log_probability: jax.Array) -> jax.Array:
    """POMO REINFORCE objective with a group-mean baseline."""

    if reward.shape != sum_log_probability.shape:
        raise ValueError(
            f"reward and sum_log_probability differ: {reward.shape} != {sum_log_probability.shape}"
        )
    return -jnp.mean(pomo_advantage(reward) * sum_log_probability)


def _dynamic_decoder_context(env: Any, state: Any) -> jax.Array | None:
    dynamic = getattr(env, "dynamic_decoder_context", None)
    return cast(jax.Array, dynamic(state)) if callable(dynamic) else None


__all__ = [
    "JaxPOMO",
    "POMOLossMetrics",
    "POMOTrainMetrics",
    "POMOTrainState",
    "distinct_first_actions",
    "pomo_advantage",
    "pomo_loss",
]
