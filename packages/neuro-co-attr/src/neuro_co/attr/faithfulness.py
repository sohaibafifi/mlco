"""Faithfulness metrics for attribution traces.

- `deletion_flip_rate`: mask the top-k attributed nodes' features, re-decode
  the same step, count how often the argmax flips. High = faithful.
- `sufficiency_keep_rate`: keep ONLY the top-k nodes' features, mask the
  rest, count how often the argmax is unchanged. High = faithful.
- `sanity_check`: randomise model weights, recompute attribution, report
  Jaccard overlap of top-k vs original (Adebayo et al. 2018). Low = faithful.

Masking operates on the single feature tensor `env.build_features(state)`:
selected node rows are replaced by a zero / batch-mean baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.attr.attribution import AttributionTrace
from neuro_co.attr.attribution._common import _decode_logp


@dataclass
class DeletionReport:
    mean_flip_rate: float
    per_step_flip_rate: list[float]
    top_k_used: int
    num_steps: int
    num_instances: int


@dataclass
class SufficiencyReport:
    mean_keep_rate: float
    per_step_keep_rate: list[float]
    top_k_used: int
    num_steps: int
    num_instances: int


@dataclass
class SanityCheckReport:
    mode: str
    mean_jaccard: float
    chance_jaccard: float
    per_step_jaccard: list[float]
    top_k_used: int
    num_trials: int


def _baseline_like(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=(0, 1), keepdim=True).expand_as(x).clone()
    return torch.zeros_like(x)


def _mask_nodes(
    feats: torch.Tensor, nodes: torch.Tensor, *, keep: bool, baseline: str
) -> torch.Tensor:
    """Replace feature rows by a baseline.

    `nodes` is `[B, k]` of node indices. `keep=True` keeps only those rows
    (mask the rest); `keep=False` masks those rows (keep the rest).
    """
    b, n, _ = feats.shape
    sel = torch.zeros(b, n, dtype=torch.bool, device=feats.device)
    sel.scatter_(1, nodes.clamp(0, n - 1), True)
    keep_mask = sel if keep else ~sel
    ref = _baseline_like(feats, baseline)
    return torch.where(keep_mask.unsqueeze(-1), feats, ref)


def _greedy_action(policy: Any, env: Any, state: Any, feats: torch.Tensor) -> torch.Tensor:
    return _decode_logp(policy, env, state, feats).argmax(dim=-1)


def deletion_flip_rate(
    trace: AttributionTrace,
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int = 5,
    baseline: str = "zero",
) -> DeletionReport:
    """Mask the top-k attributed nodes' features per step; report argmax flips."""
    device = next(policy.parameters()).device
    policy.eval()
    k = min(int(top_k), trace.top_k_nodes.shape[-1])
    state = state.to(device)
    flips: list[float] = []
    done_acc: torch.Tensor | None = None

    with torch.no_grad():
        for t in range(trace.num_steps):
            feats = env.build_features(state)
            orig = _greedy_action(policy, env, state, feats)
            top_nodes = trace.top_k_nodes[:, t, :k].to(device)
            masked = _mask_nodes(feats, top_nodes, keep=False, baseline=baseline)
            pert = _greedy_action(policy, env, state, masked)
            flips.append((pert != orig).float().mean().item())
            state, _, done = env.step(state, orig)
            done_acc = done if done_acc is None else (done_acc | done)
            if done_acc is not None and bool(done_acc.all()):
                break

    mean = float(sum(flips) / len(flips)) if flips else 0.0
    return DeletionReport(mean, flips, k, len(flips), int(trace.batch_size))


def sufficiency_keep_rate(
    trace: AttributionTrace,
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int = 5,
    baseline: str = "zero",
) -> SufficiencyReport:
    """Keep only top-k attributed nodes' features per step; report unchanged argmax."""
    device = next(policy.parameters()).device
    policy.eval()
    k = min(int(top_k), trace.top_k_nodes.shape[-1])
    state = state.to(device)
    keeps: list[float] = []
    done_acc: torch.Tensor | None = None

    with torch.no_grad():
        for t in range(trace.num_steps):
            feats = env.build_features(state)
            orig = _greedy_action(policy, env, state, feats)
            top_nodes = trace.top_k_nodes[:, t, :k].to(device)
            masked = _mask_nodes(feats, top_nodes, keep=True, baseline=baseline)
            kept = _greedy_action(policy, env, state, masked)
            keeps.append((kept == orig).float().mean().item())
            state, _, done = env.step(state, orig)
            done_acc = done if done_acc is None else (done_acc | done)
            if done_acc is not None and bool(done_acc.all()):
                break

    mean = float(sum(keeps) / len(keeps)) if keeps else 0.0
    return SufficiencyReport(mean, keeps, k, len(keeps), int(trace.batch_size))


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = {int(x) for x in a.tolist()}, {int(x) for x in b.tolist()}
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def sanity_check(
    trace: AttributionTrace,
    policy: Any,
    env: Any,
    state: Any,
    *,
    top_k: int = 5,
    num_trials: int = 1,
    mode: str = "random_weights",
    seed: int = 0,
) -> SanityCheckReport:
    """Model-parameter-randomisation sanity check (Adebayo et al. 2018).

    Saves weights, replaces every float parameter with a Gaussian draw,
    recomputes gradient attribution, restores weights, then reports the
    Jaccard overlap between original and randomised top-k. Low = faithful.
    """
    if mode != "random_weights":
        raise ValueError(f"sanity_check mode must be 'random_weights', got {mode!r}")
    from neuro_co.attr.attribution import gradient_attribution

    device = next(policy.parameters()).device
    k = min(int(top_k), trace.top_k_nodes.shape[-1])
    num_nodes = int(trace.node_scores.shape[-1])
    if num_nodes > 0:
        ei = (k * k) / num_nodes
        eu = max(2 * k - ei, 1e-9)
        chance = ei / eu
    else:
        chance = 0.0

    saved = {n: p.detach().clone() for n, p in policy.state_dict().items()}
    overlaps: list[float] = []
    try:
        for trial in range(num_trials):
            g = torch.Generator(device="cpu").manual_seed(int(seed) + trial)
            new_state: dict[str, torch.Tensor] = {}
            for n, p in saved.items():
                if p.dtype.is_floating_point:
                    new_state[n] = (
                        torch.empty_like(p, device="cpu")
                        .normal_(0.0, 0.1, generator=g)
                        .to(p.device)
                    )
                else:
                    new_state[n] = p
            policy.load_state_dict(new_state, strict=False)
            policy.to(device)
            try:
                rnd = gradient_attribution(policy, env, state, top_k=k, max_steps=trace.num_steps)
            except (RuntimeError, AssertionError, ValueError):
                overlaps.append(chance)
                continue
            steps = min(trace.num_steps, rnd.num_steps)
            for t in range(steps):
                for b in range(int(trace.batch_size)):
                    overlaps.append(
                        _jaccard(trace.top_k_nodes[b, t, :k], rnd.top_k_nodes[b, t, :k])
                    )
    finally:
        policy.load_state_dict(saved, strict=False)
        policy.to(device)

    mean = float(sum(overlaps) / len(overlaps)) if overlaps else 0.0
    return SanityCheckReport(mode, mean, chance, overlaps, k, num_trials)
