"""DeepLIFT-Rescale attribution for autoregressive CO policies.

Each decoding decision is wrapped as a scalar per-example function of the
current node-feature tensor. Captum's DeepLift implementation propagates
Rescale multipliers from the chosen action log-probability to the input and
returns a completeness residual for auditing.

The state-dependent mask and decoder indices are held fixed between the
input and reference. This makes the explanation local to the current
decision state and matches the conditioning used by the other attribution
methods in this package.
"""

from __future__ import annotations

from typing import Any

import torch
from captum.attr import DeepLift
from torch import Tensor, nn

from neuro_co.attr.attribution._common import AttributionTrace, _decode_logp, _node_scores


class _StepLogProbability(nn.Module):
    """Expose one chosen log-probability per example to Captum."""

    def __init__(
        self,
        policy: nn.Module,
        env: Any,
        state: Any,
        action: Tensor,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.state = state
        self.register_buffer("action", action.detach())

    @staticmethod
    def _repeat_batch(x: Tensor, target_batch: int) -> Tensor:
        """Repeat state tensors for Captum's joint input/reference batch."""
        source_batch = int(x.shape[0])
        if source_batch == target_batch:
            return x
        if target_batch % source_batch:
            raise ValueError(f"cannot expand auxiliary batch {source_batch} to {target_batch}")
        repeats = target_batch // source_batch
        return x.repeat((repeats,) + (1,) * (x.ndim - 1))

    def forward(self, feats: Tensor) -> Tensor:
        batch = int(feats.shape[0])
        mask = self._repeat_batch(self.env.action_mask(self.state), batch)
        first_idx, current_idx = self.env.decoder_context(self.state)
        first_idx = self._repeat_batch(first_idx, batch)
        current_idx = self._repeat_batch(current_idx, batch)
        action = self._repeat_batch(self.action, batch)

        node_embs, graph_emb = self.policy.encode(feats)
        logits = self.policy.decode_step(
            node_embs,
            graph_emb,
            first_idx,
            current_idx,
            mask,
        )
        log_p = torch.log_softmax(logits, dim=-1)
        return log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)


def deeplift_attribution(
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int = 5,
    max_steps: int | None = None,
    baseline: str = "zero",
) -> AttributionTrace:
    """Compute per-step DeepLIFT-Rescale attributions.

    Parameters are aligned with :func:`integrated_gradients`. ``baseline``
    can be ``"zero"`` or ``"mean"``. The returned trace includes signed
    completeness residuals in ``convergence_delta`` so experimental code can
    detect unsupported or numerically unstable model operations.
    """
    if baseline not in ("zero", "mean"):
        raise ValueError(f"baseline must be 'zero' or 'mean', got {baseline!r}")

    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    cap = env.max_steps(state) if max_steps is None else min(max_steps, env.max_steps(state))

    actions_per_step: list[Tensor] = []
    logp_per_step: list[Tensor] = []
    scores_per_step: list[Tensor] = []
    feature_scores_per_step: list[Tensor] = []
    deltas_per_step: list[Tensor] = []
    relative_deltas_per_step: list[Tensor] = []
    done_acc: Tensor | None = None

    for _step in range(cap):
        x = env.build_features(state).detach().requires_grad_(True)
        if baseline == "zero":
            x0 = torch.zeros_like(x)
        else:
            x0 = x.mean(dim=(0, 1), keepdim=True).expand_as(x).clone()

        with torch.no_grad():
            log_p = _decode_logp(policy, env, state, x)
            action = log_p.argmax(dim=-1)
            chosen_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        target = _StepLogProbability(policy, env, state, action)
        explainer = DeepLift(target, multiply_by_inputs=True)
        attr, delta = explainer.attribute(
            x,
            baselines=x0,
            return_convergence_delta=True,
        )
        feature_score = attr.detach().abs()
        node_score = _node_scores(feature_score, torch.ones_like(feature_score))

        actions_per_step.append(action.detach())
        logp_per_step.append(chosen_logp.detach())
        scores_per_step.append(node_score)
        feature_scores_per_step.append(feature_score)
        deltas_per_step.append(delta.detach())
        attributed_sum = attr.detach().flatten(1).sum(dim=-1)
        output_difference = attributed_sum - delta.detach()
        relative_deltas_per_step.append(
            delta.detach().abs() / output_difference.abs().clamp_min(1e-8)
        )

        state, _, done = env.step(state, action.detach())
        done_acc = done if done_acc is None else (done_acc | done)
        if done_acc is not None and bool(done_acc.all()):
            break

    if not actions_per_step:
        raise RuntimeError("Policy terminated without taking any action")

    actions = torch.stack(actions_per_step, dim=1)
    log_probs = torch.stack(logp_per_step, dim=1)
    scores = torch.stack(scores_per_step, dim=1)
    features = torch.stack(feature_scores_per_step, dim=1)
    convergence_delta = torch.stack(deltas_per_step, dim=1)
    relative_convergence_delta = torch.stack(relative_deltas_per_step, dim=1)
    k = min(int(top_k), scores.shape[-1])
    top_scores, top_nodes = scores.topk(k=k, dim=-1)

    return AttributionTrace(
        actions=actions.cpu(),
        log_probs=log_probs.cpu(),
        node_scores=scores.cpu(),
        top_k_nodes=top_nodes.cpu(),
        top_k_scores=top_scores.cpu(),
        feature_scores=features.cpu(),
        convergence_delta=convergence_delta.cpu(),
        relative_convergence_delta=relative_convergence_delta.cpu(),
    )
