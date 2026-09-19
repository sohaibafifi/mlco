"""Lambda-attribution: decompose a policy's per-step gradient by constraint family.

For a CO problem with constraint families `c_1, ..., c_K` (capacity,
time-window, precedence, ...) the Lambda-attribution of decision `a_t` to
constraint `c_k` is the gradient-x-feature mass aggregated over the feature
*columns* that parameterise `c_k`::

    Lambda_k(t) = mean_{j in nodes, i in cols(c_k)} |grad_ij * x_ij|

The column groups come from `neuro_co.cax.constraint_map.PROBLEM_CONSTRAINTS`.
A single rollout suffices: at each step we differentiate `log pi(a_t)` w.r.t.
the input feature tensor once, then split the per-node grad-x-feature by
family columns. Optional Lagrangian weighting scales each family's score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.attr.attribution import AttributionTrace
from neuro_co.attr.attribution._common import _decode_logp
from neuro_co.cax.constraint_map import get_constraints


@dataclass
class LambdaAttribution:
    """Per-step attribution decomposed by constraint family.

    Attributes
    ----------
    constraint_names
        Ordered list of constraint family names.
    scores
        `[batch, T, K]` non-negative `Lambda_k(t)`.
    multipliers
        `[K]` or `[batch, K]` Lagrangian weights when supplied, else `None`.
    per_family_node_scores
        `[K, B, T, N]` per-family per-node attribution.
    feature_cols_per_family
        `[K]` tuples of feature-column indices used per family.
    actions, log_probs
        Greedy rollout actions and their log probabilities, both `[B, T]`.
    """

    constraint_names: list[str]
    scores: torch.Tensor
    multipliers: torch.Tensor | None
    per_family_node_scores: torch.Tensor
    feature_cols_per_family: list[tuple[int, ...]]
    actions: torch.Tensor
    log_probs: torch.Tensor

    @property
    def num_families(self) -> int:
        return int(self.scores.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self.scores.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.scores.shape[1])

    def top_family_per_step(self) -> torch.Tensor:
        """`[B, T]` long tensor of the argmax constraint family per step."""
        return self.scores.argmax(dim=-1)

    def to_attribution_trace(self, top_k: int = 5) -> AttributionTrace:
        """Collapse weighted family-node scores for deletion diagnostics."""
        nodes = self.per_family_node_scores.permute(1, 2, 0, 3)  # [B, T, K, N]
        if self.multipliers is None:
            weights = torch.ones(nodes.shape[0], nodes.shape[2])
        elif self.multipliers.ndim == 1:
            weights = self.multipliers.unsqueeze(0).expand(nodes.shape[0], -1)
        else:
            weights = self.multipliers
        combined = (nodes * weights[:, None, :, None]).mean(dim=2)
        k = min(int(top_k), int(combined.shape[-1]))
        top_scores, top_nodes = combined.topk(k=k, dim=-1)
        return AttributionTrace(
            actions=self.actions,
            log_probs=self.log_probs,
            node_scores=combined,
            top_k_nodes=top_nodes,
            top_k_scores=top_scores,
        )


def lambda_attribution(
    policy: Any,
    env: Any,
    state: Any,
    *,
    problem: str,
    max_steps: int | None = None,
    multipliers: dict[str, float] | torch.Tensor | None = None,
    require_multipliers: bool = False,
) -> LambdaAttribution:
    """Compute Lambda-attribution per (step, constraint family).

    Parameters
    ----------
    policy, env, state
        Core `ConstructivePolicy`, `Env`, and `State`.
    problem
        Constraint-map key (`'cvrptw'`, `'op'`, `'fjsp'`).
    max_steps
        Optional cap on decoding-step count.
    multipliers
        Optional `family -> weight` dict or `[batch, K]` tensor scaling each
        family's score. LP duals are computed by `neuro_co.cax.duals` and
        passed in here by the caller.
    require_multipliers
        Fail instead of silently falling back to unweighted group scores.
    """
    families = get_constraints(problem)
    family_names = [name for name, _ in families]
    family_cols = [cols for _, cols in families]
    k_families = len(families)

    if require_multipliers and multipliers is None:
        raise ValueError("CAX requires explicit constraint multipliers")

    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    cap = env.max_steps(state) if max_steps is None else min(max_steps, env.max_steps(state))
    batch_size = int(env.build_features(state).shape[0])

    if isinstance(multipliers, torch.Tensor):
        weights = multipliers.to(device=device, dtype=torch.float32)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(batch_size, -1)
        if weights.shape != (batch_size, k_families):
            raise ValueError(
                "multiplier tensor must have shape "
                f"[{batch_size}, {k_families}], got {tuple(weights.shape)}"
            )
    elif multipliers is not None:
        weights = (
            torch.tensor(
                [float(multipliers.get(n, 0.0)) for n in family_names],
                device=device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
    else:
        weights = torch.ones(batch_size, k_families, device=device)

    per_step_scores: list[torch.Tensor] = []  # each [B, K]
    per_step_nodes: list[torch.Tensor] = []  # each [K, B, N]
    actions_per_step: list[torch.Tensor] = []
    logp_per_step: list[torch.Tensor] = []
    done_acc: torch.Tensor | None = None

    for _ in range(cap):
        feats = env.build_features(state).detach().requires_grad_(True)  # [B, N, d_in]
        log_p = _decode_logp(policy, env, state, feats)
        action = log_p.argmax(dim=-1)
        chosen = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        (grad,) = torch.autograd.grad(chosen.sum(), feats, allow_unused=True)
        gxf = (grad.detach() * feats.detach()) if grad is not None else torch.zeros_like(feats)

        step_nodes = torch.zeros(k_families, feats.shape[0], feats.shape[1], device=device)
        step_scores = torch.zeros(feats.shape[0], k_families, device=device)
        for k_idx, cols in enumerate(family_cols):
            col_idx = torch.tensor(cols, device=device)
            node = gxf.index_select(-1, col_idx).abs().mean(dim=-1)  # [B, N]
            w = weights[:, k_idx]
            step_nodes[k_idx] = node
            step_scores[:, k_idx] = node.mean(dim=-1) * w
        per_step_nodes.append(step_nodes)
        per_step_scores.append(step_scores)
        actions_per_step.append(action.detach())
        logp_per_step.append(chosen.detach())

        state, _, done = env.step(state, action.detach())
        done_acc = done if done_acc is None else (done_acc | done)
        if done_acc is not None and bool(done_acc.all()):
            break

    scores = torch.stack(per_step_scores, dim=1).cpu()  # [B, T, K]
    per_family_node = torch.stack(per_step_nodes, dim=2).cpu()  # [K, B, T, N]
    mult_tensor = weights.detach().cpu() if multipliers is not None else None
    return LambdaAttribution(
        constraint_names=family_names,
        scores=scores,
        multipliers=mult_tensor,
        per_family_node_scores=per_family_node,
        feature_cols_per_family=family_cols,
        actions=torch.stack(actions_per_step, dim=1).cpu(),
        log_probs=torch.stack(logp_per_step, dim=1).cpu(),
    )
