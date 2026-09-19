"""Asymptotic regime analysis.

For a neural solver and a metaheuristic baseline::

    E_total_NN(N)   = E_train + N * E_NN_per_inst
    E_total_meta(N) = N * E_meta_per_inst

When `E_NN_per_inst < E_meta_per_inst`, the ratio
`E_total_NN(N) / E_total_meta(N) -> E_NN_per_inst / E_meta_per_inst < 1`
as `N -> infinity`. Training cost amortizes; the marginal advantage
is structural.
"""

from __future__ import annotations


def crossover_n(e_train: float, e_nn: float, e_meta: float) -> float | None:
    """Return the break-even N (a.k.a. AET) for a given parameter triple.

    Returns None if the neural solver is not asymptotically cheaper
    (denominator is non-positive).
    """
    denom = e_meta - e_nn
    if denom <= 0:
        return None
    return e_train / denom


def asymptotic_ratio(e_nn: float, e_meta: float) -> float | None:
    """Return the limit of NN/meta cumulative-energy ratio as N→∞."""
    if e_meta <= 0:
        return None
    return e_nn / e_meta


def energy_curves(
    e_train: float,
    e_nn: float,
    e_meta: float,
    n_values: list[int] | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Return (N values, NN cumulative energy, baseline cumulative energy).

    The plotting front-ends in `benchmarks/` consume these arrays. The
    package itself keeps `matplotlib` optional.
    """
    n_values = n_values or [10**k for k in range(0, 10)]
    nn_curve = [e_train + e_nn * n for n in n_values]
    meta_curve = [e_meta * n for n in n_values]
    return n_values, nn_curve, meta_curve
