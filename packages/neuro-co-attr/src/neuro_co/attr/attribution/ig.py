"""Integrated Gradients (Sundararajan et al. 2017).

For each decoding step `t`:

    IG_i(x) = (x_i - x'_i) * integral_{alpha=0..1}
              d log pi(a_t | s_t(x' + alpha (x - x')))/dx_i  dalpha

The baseline `x'` is built from the single feature tensor
`env.build_features(state)`:

- ``"zero"``: all-zero features,
- ``"mean"``: per-feature batch mean broadcast over nodes.

Approximated as a midpoint Riemann sum with `ig_steps` samples per
decoding step. Cost: `ig_steps x num_steps` backward passes.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuro_co.attr.attribution._common import (
    AttributionTrace,
    _decode_logp,
    _node_scores,
    _pack_trace,
)


def integrated_gradients(
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int = 5,
    max_steps: int | None = None,
    ig_steps: int = 20,
    baseline: str = "zero",
) -> AttributionTrace:
    """Integrated Gradients with a zero or mean feature baseline."""
    if baseline not in ("zero", "mean"):
        raise ValueError(f"baseline must be 'zero' or 'mean', got {baseline!r}")
    if ig_steps < 1:
        raise ValueError("ig_steps must be >= 1")

    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    cap = env.max_steps(state) if max_steps is None else min(max_steps, env.max_steps(state))

    actions_per_step: list[Tensor] = []
    logp_per_step: list[Tensor] = []
    scores_per_step: list[Tensor] = []
    feature_scores_per_step: list[Tensor] = []
    done_acc: Tensor | None = None

    for _step in range(cap):
        x = env.build_features(state).detach()  # [B, N, d_in]
        if baseline == "zero":
            x0 = torch.zeros_like(x)
        else:
            x0 = x.mean(dim=(0, 1), keepdim=True).expand_as(x).clone()

        # Greedy action from the true input (no path needed for the choice).
        with torch.no_grad():
            log_p = _decode_logp(policy, env, state, x)
            action = log_p.argmax(dim=-1)
            chosen_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        accum = torch.zeros_like(x)
        for i in range(ig_steps):
            alpha = (i + 0.5) / ig_steps  # midpoint rule
            interp = (x0 + alpha * (x - x0)).detach().requires_grad_(True)
            lp = _decode_logp(policy, env, state, interp)
            sel = lp.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            (grad,) = torch.autograd.grad(sel.sum(), interp, allow_unused=True)
            if grad is not None:
                accum = accum + grad.detach() / ig_steps

        attr = accum * (x - x0)  # [B, N, d_in]
        node_score = _node_scores(attr, torch.ones_like(attr))

        actions_per_step.append(action.detach())
        logp_per_step.append(chosen_logp.detach())
        scores_per_step.append(node_score)
        feature_scores_per_step.append(attr.abs())

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
