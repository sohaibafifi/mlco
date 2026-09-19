"""Projected multiplier updates with a Beasley-style step rule.

The caller supplies a subproblem value and constraint residuals for each
multiplier vector. With an exact relaxed minimizer, these can define a
Lagrangian dual value and its subgradient. Heuristic callbacks supply
multiplier estimates without a certified bound.

The routine retains the multipliers at the largest reported value. It uses
`alpha * max(UB - value, 0) / ||g||^2` when an upper bound is supplied and
`alpha / (1 + iteration)` otherwise. Stalled runs halve `alpha`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class SubgradReport:
    """Outcome of a subgradient-ascent run."""

    multipliers: dict[str, float]
    best_dual: float
    iterations_run: int
    converged: bool
    history: list[float]


def subgradient_ascent(
    family_names: list[str],
    subproblem: Callable[[dict[str, float]], tuple[float, dict[str, float]]],
    *,
    max_iters: int = 100,
    initial_step: float = 2.0,
    upper_bound: float | None = None,
    initial_multipliers: dict[str, float] | None = None,
    tol: float = 1e-4,
    stall_halve_every: int = 10,
) -> SubgradReport:
    """Beasley 1981-style step on Lagrangian duals.

    Parameters
    ----------
    family_names
        Ordered list of constraint families (becomes dict keys).
    subproblem
        Callable returning `(L_value, violations)` given the
        current multiplier dict. `violations[k]` is the residual
        of the k-th dualised constraint at the returned solution.
    max_iters
        Hard cap on iterations.
    initial_step
        `alpha` in Beasley's step rule. Halved every
        `stall_halve_every` non-improving iterations.
    upper_bound
        Primal-feasible UB used in the step rule
        `step = alpha * (UB - L) / ||g||^2`. If None we fall back
        to `step = alpha / (1 + iter)` (no UB hand-off).
    initial_multipliers
        Starting `lambda_0`. Defaults to zeros.
    tol
        Tolerance for value improvement and squared residual norm.
    stall_halve_every
        Halve `alpha` after this many non-improving steps.
    """
    lam = {k: 0.0 for k in family_names}
    if initial_multipliers:
        lam.update({k: float(v) for k, v in initial_multipliers.items() if k in lam})

    best_dual = -float("inf")
    best_lam = dict(lam)
    history: list[float] = []
    alpha = float(initial_step)
    stalls = 0

    for it in range(max_iters):
        L_val, violations = subproblem(lam)
        history.append(L_val)

        if L_val > best_dual + tol:
            best_dual = L_val
            best_lam = dict(lam)
            stalls = 0
        else:
            stalls += 1
            if stalls and stalls % stall_halve_every == 0:
                alpha = max(alpha * 0.5, 1e-6)

        # Step size.
        g = [violations.get(k, 0.0) for k in family_names]
        g_norm2 = sum(v * v for v in g)
        if g_norm2 < tol:
            return SubgradReport(
                multipliers=best_lam,
                best_dual=best_dual,
                iterations_run=it + 1,
                converged=True,
                history=history,
            )
        if upper_bound is not None:
            step = alpha * max(upper_bound - L_val, 0.0) / g_norm2
        else:
            step = alpha / (1.0 + it)

        # Multiplier update: project on R_+ (multipliers are non-negative
        # because we always penalise *positive* violations).
        for k in family_names:
            lam[k] = max(lam[k] + step * violations.get(k, 0.0), 0.0)

    return SubgradReport(
        multipliers=best_lam,
        best_dual=best_dual,
        iterations_run=max_iters,
        converged=False,
        history=history,
    )


_ = Any  # silence "imported but unused" if Any-typed kwargs land later
