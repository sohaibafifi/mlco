"""Decoding strategies. Pure functions, no class hierarchy.

All decoders consume masked logits and a generator (for reproducibility).
Caller is responsible for applying `apply_mask` upstream.
"""

from collections.abc import Callable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def greedy(logits: Float[Tensor, "b a"]) -> Int[Tensor, "b"]:
    """Argmax decode."""
    return logits.argmax(dim=-1)


def sample(
    logits: Float[Tensor, "b a"],
    generator: torch.Generator | None = None,
) -> Int[Tensor, "b"]:
    """Categorical sample. `multinomial` with `num_samples=1`."""
    probs = logits.softmax(dim=-1)
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    return idx.squeeze(-1)


def log_prob(
    logits: Float[Tensor, "b a"],
    action: Int[Tensor, "b"],
) -> Float[Tensor, "b"]:
    """Log-prob of a taken action under categorical(logits)."""
    return logits.log_softmax(dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)


def beam_search(
    *,
    initial_state,
    initial_mask: Bool[Tensor, "b a"],
    step_logits_fn: Callable,
    env_step_fn: Callable,
    env_mask_fn: Callable,
    n_steps: int,
    beam_width: int,
) -> tuple[Float[Tensor, "b"], Int[Tensor, "b T"]]:
    """Batched beam search for sequential CO problems.

    Args:
        initial_state: pytree-registered State (batch dim = b).
        initial_mask: (b, a) mask at step 0.
        step_logits_fn: (state, mask) -> logits (B, a). May reference encoder cache.
        env_step_fn:   (state, action) -> (next_state, reward, done).
        env_mask_fn:   (state) -> mask (B, a).
        n_steps: total decode steps.
        beam_width: number of candidates kept per problem.

    Returns:
        best_reward: (b,): best cumulative reward across beams.
        best_actions: (b, n_steps): actions of the best beam.

    Implementation: state is replicated to (b*beam_width). At each step, expand
    each beam over all actions, pick top-K cumulative-logp candidates per problem.
    """
    from torch.utils import _pytree as pytree

    state = initial_state
    b = initial_mask.size(0)
    a = initial_mask.size(1)

    # Step 0: expand b -> b * beam_width by picking top-K initial actions.
    logits0 = step_logits_fn(state, initial_mask)  # (b, a)
    logp0 = logits0.log_softmax(dim=-1)
    topk_logp, topk_act = logp0.topk(beam_width, dim=-1)  # (b, K)
    state = _repeat(state, beam_width)  # (b*K,)
    cum_logp = topk_logp.reshape(b * beam_width)
    actions_hist = topk_act.reshape(b * beam_width, 1)
    cum_reward = torch.zeros(b * beam_width, device=initial_mask.device)
    flat_first_action = topk_act.reshape(b * beam_width)
    state, reward, _ = env_step_fn(state, flat_first_action)
    cum_reward = cum_reward + reward

    for _ in range(n_steps - 1):
        mask = env_mask_fn(state)
        logits = step_logits_fn(state, mask)
        logp = logits.log_softmax(dim=-1)  # (b*K, a)
        # Score = cum_logp + logp; reshape to (b, K*a) to pick top-K per problem.
        score = (cum_logp.unsqueeze(-1) + logp).view(b, beam_width * a)
        top_score, top_idx = score.topk(beam_width, dim=-1)  # (b, K)
        beam_idx = (top_idx // a).reshape(b * beam_width)  # which beam survived (0..K-1)
        action = (top_idx % a).reshape(b * beam_width)

        # Re-gather state along beam dim.
        flat_beam_idx = (
            beam_idx
            + torch.arange(b, device=beam_idx.device).repeat_interleave(beam_width) * beam_width
        )
        state = pytree.tree_map(
            lambda x, fi=flat_beam_idx: x[fi] if isinstance(x, torch.Tensor) else x, state
        )
        actions_hist = actions_hist[flat_beam_idx]
        cum_reward = cum_reward[flat_beam_idx]
        cum_logp = top_score.reshape(b * beam_width)
        actions_hist = torch.cat([actions_hist, action.unsqueeze(-1)], dim=-1)
        state, reward, _ = env_step_fn(state, action)
        cum_reward = cum_reward + reward

    cum_reward_grouped = cum_reward.view(b, beam_width)
    best_idx = cum_reward_grouped.argmax(dim=-1)  # (b,)
    best_reward = cum_reward_grouped.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    flat_best = best_idx + torch.arange(b, device=best_idx.device) * beam_width
    best_actions = actions_hist[flat_best]
    return best_reward, best_actions


def _repeat(state, times: int):
    from torch.utils import _pytree as pytree

    return pytree.tree_map(
        lambda x: x.repeat_interleave(times, dim=0) if isinstance(x, torch.Tensor) else x, state
    )
