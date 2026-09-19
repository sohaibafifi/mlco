"""Contrastive attribution: features that push toward `a` vs `b`.

For each decoding step we attribute the log-probability margin
`log pi(a_t) - log pi(b_t)` to the input feature tensor. We aggregate
`|grad x feature|` (same shape as `gradient_attribution`) and store the
signed log-probability margin in `log_probs`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuro_co.attr.attribution._common import AttributionTrace, _rollout_grad_x_feats


def contrastive_attribution(
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int = 5,
    max_steps: int | None = None,
    action_b: Tensor | None = None,
) -> AttributionTrace:
    """Attribute `log pi(a_t) - log pi(b_t)` per step.

    Parameters
    ----------
    action_b
        Optional `[batch, T]` long tensor of alternatives to contrast
        against. If `None`, the second-best action per step is used.
    """

    def target_fn(log_p: Tensor, step: int) -> tuple[Tensor, Tensor, Tensor]:
        action = log_p.detach().argmax(dim=-1)
        if action_b is not None and step < action_b.shape[1]:
            alt = action_b[:, step].to(log_p.device)
        else:
            masked = log_p.detach().clone()
            masked.scatter_(-1, action.unsqueeze(-1), float("-inf"))
            # A row with a single valid action has all-`-inf` here; fall back
            # to the primary action (margin 0) rather than emit NaN gradients.
            alt_max_vals, alt_argmax = masked.max(dim=-1)
            no_alt = ~torch.isfinite(alt_max_vals)
            alt = torch.where(no_alt, action, alt_argmax)
        a_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        b_logp = log_p.gather(-1, alt.unsqueeze(-1)).squeeze(-1)
        margin = a_logp - b_logp
        return action, margin, margin

    return _rollout_grad_x_feats(
        policy, env, state, top_k=top_k, max_steps=max_steps, target_fn=target_fn
    )
