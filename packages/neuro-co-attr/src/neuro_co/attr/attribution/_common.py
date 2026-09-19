"""Shared primitives for attribution methods.

`AttributionTrace`, the per-step gradient-x-feature rollout driver, and the
trace-packing tail. Built on `neuro-co-core`: a policy is a
`ConstructivePolicy`, an env satisfies the `Env` protocol, and the state is
a frozen `State`. Node features come from a single `env.build_features(state)`
tensor `[B, N, d_in]`, so attribution differentiates one input tensor, with no
named-key bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class AttributionTrace:
    """Per-step attribution for one rollout.

    Attributes
    ----------
    actions
        `[batch, T]` long tensor of taken action indices.
    log_probs
        `[batch, T]` float tensor of `log pi(a_t | s_t)` (or the contrast
        margin, depending on the method).
    node_scores
        `[batch, T, N]` float tensor containing attribution per node per step.
    top_k_nodes
        `[batch, T, K]` long tensor of top-k node indices per step.
    top_k_scores
        `[batch, T, K]` float tensor of their attribution values.
    feature_scores
        Optional `[batch, T, N, d_in]` tensor containing the absolute
        element-wise attribution. This preserves the information required
        for feature-group and constraint-family comparisons.
    convergence_delta
        Optional `[batch, T]` completeness residual. DeepLIFT populates it
        with `sum(attributions) - (output - reference_output)`.
    relative_convergence_delta
        Optional `[batch, T]` absolute completeness residual divided by the
        absolute explained output difference.
    """

    actions: Tensor
    log_probs: Tensor
    node_scores: Tensor
    top_k_nodes: Tensor
    top_k_scores: Tensor
    feature_scores: Tensor | None = None
    convergence_delta: Tensor | None = None
    relative_convergence_delta: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_nodes(self) -> int:
        return int(self.node_scores.shape[-1])


def _node_scores(grad: Tensor, feats: Tensor) -> Tensor:
    """Reduce grad-x-feature `[B, N, d_in]` to per-node `[B, N]`.

    Sum absolute element-wise contributions over the feature dimension.
    Keeping magnitudes separate prevents unrelated signed feature effects
    from cancelling before a node is ranked.
    """
    return (grad * feats).abs().sum(dim=-1)


def _decode_logp(policy: Any, env: Any, state: Any, feats: Tensor) -> Tensor:
    """Encode `feats`, decode one step, return masked log-probabilities."""
    node_embs, graph_emb = policy.encode(feats)
    mask = env.action_mask(state)
    first_idx, current_idx = env.decoder_context(state)
    logits = policy.decode_step(node_embs, graph_emb, first_idx, current_idx, mask)
    return torch.log_softmax(logits, dim=-1)


def _pack_trace(
    actions: list[Tensor],
    logp: list[Tensor],
    scores: list[Tensor],
    top_k: int,
    feature_scores: list[Tensor] | None = None,
) -> AttributionTrace:
    """Stack per-step lists, compute top-k, return on CPU."""
    if not actions:
        raise RuntimeError("Policy terminated without taking any action")
    a = torch.stack(actions, dim=1)
    lp = torch.stack(logp, dim=1)
    sc = torch.stack(scores, dim=1)
    k = min(int(top_k), sc.shape[-1])
    top_scores, top_nodes = sc.topk(k=k, dim=-1)
    return AttributionTrace(
        actions=a.cpu(),
        log_probs=lp.cpu(),
        node_scores=sc.cpu(),
        top_k_nodes=top_nodes.cpu(),
        top_k_scores=top_scores.cpu(),
        feature_scores=(torch.stack(feature_scores, dim=1).cpu() if feature_scores else None),
    )


# Driver-callback signature:
#   target_fn(log_p, step_idx) -> (action, scalar_to_backward, value_to_record)
TargetFn = Callable[[Tensor, int], tuple[Tensor, Tensor, Tensor]]


def _rollout_grad_x_feats(
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int,
    max_steps: int | None,
    target_fn: TargetFn,
) -> AttributionTrace:
    """Single-pass greedy rollout with per-step gradient-x-feature attribution.

    The method-specific `target_fn` decides which action drives the env
    step, which scalar is backpropagated to the input features, and which
    value is recorded. Used by `gradient_attribution` and
    `contrastive_attribution`.
    """
    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    cap = env.max_steps(state) if max_steps is None else min(max_steps, env.max_steps(state))

    actions_per_step: list[Tensor] = []
    logp_per_step: list[Tensor] = []
    scores_per_step: list[Tensor] = []
    feature_scores_per_step: list[Tensor] = []
    done_acc: Tensor | None = None

    for step in range(cap):
        feats = env.build_features(state).detach().requires_grad_(True)
        log_p = _decode_logp(policy, env, state, feats)
        action, scalar, value = target_fn(log_p, step)
        (grad,) = torch.autograd.grad(scalar.sum(), feats, retain_graph=False, allow_unused=True)
        if grad is None:
            node_score = torch.zeros(feats.shape[0], feats.shape[1], device=device)
            feature_score = torch.zeros_like(feats)
        else:
            node_score = _node_scores(grad.detach(), feats.detach())
            feature_score = (grad.detach() * feats.detach()).abs()
        actions_per_step.append(action.detach())
        logp_per_step.append(value.detach())
        scores_per_step.append(node_score)
        feature_scores_per_step.append(feature_score)

        state, _, done = env.step(state, action.detach())
        done_acc = done if done_acc is None else (done_acc | done)
        if done_acc is not None and bool(done_acc.all()):
            break

    return _pack_trace(
        actions_per_step,
        logp_per_step,
        scores_per_step,
        top_k,
        feature_scores_per_step,
    )
