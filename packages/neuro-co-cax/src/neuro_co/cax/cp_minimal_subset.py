"""Find sufficient node subsets along an attribution ranking.

For each prefix of the gradient-attribution ranking, sample Gaussian feature
perturbations and estimate how often the masked policy preserves the action.
Accept a subset when its Hoeffding lower bound reaches `1 - target_error`.
The sample count uses the estimation margin and a multiple-test correction.

This is a greedy search over ranked prefixes. It does not solve a global
minimum-subset problem. Its confidence statement concerns the declared
perturbation distribution and feature-masking intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.attr.attribution import AttributionTrace, gradient_attribution
from neuro_co.attr.attribution._common import _decode_logp
from neuro_co.attr.faithfulness import _mask_nodes


@dataclass
class MinimalSubsetReport:
    """PAC-minimal sufficient subset per (batch, step).

    subset_size: `[B, T]` `|S|` (0 = none <= max_k satisfied PAC).
    subset:      `[B, T, N]` bool of kept nodes.
    preserved_rate: `[B, T]` empirical `Pr[argmax preserved]` at `|S|`.
    """

    subset_size: torch.Tensor
    subset: torch.Tensor
    preserved_rate: torch.Tensor
    lower_confidence_bound: torch.Tensor
    target_error: float
    estimation_margin: float
    confidence_delta: float
    samples_drawn: int
    method: str = "greedy"

    @property
    def pac_epsilon(self) -> float:
        """Backward-compatible alias for the target error."""
        return self.target_error

    @property
    def pac_delta(self) -> float:
        """Backward-compatible alias for the confidence failure rate."""
        return self.confidence_delta


def pac_sample_count(estimation_margin: float, delta: float, *, n_tests: int = 1) -> int:
    """Bonferroni-corrected Hoeffding sample size.

    `M = ceil(log(2 n_tests / delta) / (2 eps^2))`. Pass `n_tests = k_max`
    for a family-wise `(1 - delta)`-PAC guarantee along the greedy ordering.
    """
    if not (0 < estimation_margin < 1) or not (0 < delta < 1):
        raise ValueError(
            f"estimation_margin and delta must lie in (0, 1); got {estimation_margin=}, {delta=}"
        )
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1; got {n_tests}")
    return math.ceil(
        math.log(2.0 * n_tests / delta) / (2.0 * estimation_margin * estimation_margin)
    )


def cp_minimal_subset(
    policy: Any,
    env: Any,
    state: Any,
    *,
    target_error: float = 0.2,
    confidence_delta: float = 0.05,
    estimation_margin: float = 0.05,
    sigma: float = 0.05,
    max_k: int | None = None,
    max_steps: int | None = 8,
    trace: AttributionTrace | None = None,
    bonferroni: bool = True,
    pac_epsilon: float | None = None,
    pac_delta: float | None = None,
) -> MinimalSubsetReport:
    """Find a sufficient prefix of the gradient-attribution node ranking.

    The search masks whole node-feature rows and tests success with a confidence
    bound. Minimality is limited to the tested ranking prefixes.
    """
    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)

    # Preserve old call sites while giving the parameters correct semantics.
    # The legacy epsilon is used both as target error and estimation margin,
    # exactly as requested by those callers, but acceptance still uses the
    # lower confidence bound rather than the empirical rate.
    if pac_epsilon is not None:
        target_error = float(pac_epsilon)
        estimation_margin = float(pac_epsilon)
    if pac_delta is not None:
        confidence_delta = float(pac_delta)
    for name, value in (
        ("target_error", target_error),
        ("confidence_delta", confidence_delta),
        ("estimation_margin", estimation_margin),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1), got {value}")

    if trace is None:
        n0 = env.build_features(state).shape[1]
        trace = gradient_attribution(policy, env, state, top_k=n0, max_steps=max_steps)

    b = int(trace.batch_size)
    t_steps = int(trace.num_steps)
    n = int(trace.node_scores.shape[-1])
    if max_k is None:
        max_k = n
    max_k = min(max_k, n)

    n_tests = max_k if bonferroni else 1
    m_samples = pac_sample_count(
        estimation_margin,
        confidence_delta,
        n_tests=n_tests,
    )

    subset_size = torch.zeros(b, t_steps, dtype=torch.long)
    subset = torch.zeros(b, t_steps, n, dtype=torch.bool)
    preserved_rate = torch.zeros(b, t_steps, dtype=torch.float32)
    lower_bound = torch.zeros(b, t_steps, dtype=torch.float32)
    done_acc: torch.Tensor | None = None

    with torch.no_grad():
        for t in range(t_steps):
            feats = env.build_features(state)
            orig = _decode_logp(policy, env, state, feats).argmax(dim=-1)
            order = trace.top_k_nodes[:, t, :].to(device)

            for k in range(1, max_k + 1):
                top_set = order[:, :k]
                preserved = torch.zeros(b, device=device)
                for _ in range(m_samples):
                    noisy = feats + torch.randn_like(feats) * sigma
                    masked = _mask_nodes(noisy, top_set, keep=True, baseline="zero")
                    new_action = _decode_logp(policy, env, state, masked).argmax(dim=-1)
                    preserved = preserved + (new_action == orig).float()
                rate = preserved / m_samples
                candidate_lower = (rate - estimation_margin).clamp_min(0.0)
                satisfied = candidate_lower >= (1.0 - target_error)
                first_hit = (subset_size[:, t] == 0) & satisfied.cpu()
                if first_hit.any():
                    subset_size[first_hit, t] = k
                    preserved_rate[first_hit, t] = rate[first_hit].cpu()
                    lower_bound[first_hit, t] = candidate_lower[first_hit].cpu()
                    for bi in torch.nonzero(first_hit, as_tuple=False).flatten().tolist():
                        subset[bi, t, top_set[bi].cpu()] = True
                if bool((subset_size[:, t] != 0).all()):
                    break

            state, _, done = env.step(state, orig)
            done_acc = done if done_acc is None else (done_acc | done)
            if done_acc is not None and bool(done_acc.all()):
                break

    return MinimalSubsetReport(
        subset_size=subset_size,
        subset=subset,
        preserved_rate=preserved_rate,
        lower_confidence_bound=lower_bound,
        target_error=target_error,
        estimation_margin=estimation_margin,
        confidence_delta=confidence_delta,
        samples_drawn=m_samples,
        method="greedy_hoeffding_lower_bound",
    )
