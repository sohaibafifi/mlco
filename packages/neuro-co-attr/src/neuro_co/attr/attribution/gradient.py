"""Gradient x input attribution for autoregressive CO policies.

For each decoding step `t` we record:

- the action taken `a_t` (index of the chosen node);
- the log-probability `log pi(a_t | s_t)` under the policy;
- the gradient of that log-prob w.r.t. the node-feature tensor
  `env.build_features(state)`;
- the per-node attribution score = magnitude of the summed
  `(grad x feature)` over feature dims;
- the top-k attributed node indices and their scores.

Works with any core `ConstructivePolicy` + `Env`.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from neuro_co.attr.attribution._common import AttributionTrace, _rollout_grad_x_feats


def gradient_attribution(
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int = 5,
    max_steps: int | None = None,
) -> AttributionTrace:
    """Roll out `policy` on `env` and record gradient x input per step.

    The policy is rolled out greedily (argmax). At each decision step we
    backpropagate `log pi(a_t | s_t)` through the input feature tensor,
    then aggregate per-node scores as the magnitude of summed
    grad x feature over feature dims.
    """

    def target_fn(log_p: Tensor, _step: int) -> tuple[Tensor, Tensor, Tensor]:
        action = log_p.detach().argmax(dim=-1)
        chosen_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return action, chosen_logp, chosen_logp

    return _rollout_grad_x_feats(
        policy, env, state, top_k=top_k, max_steps=max_steps, target_fn=target_fn
    )
