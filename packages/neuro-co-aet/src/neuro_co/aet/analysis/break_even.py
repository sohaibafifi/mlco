"""Batch break-even estimates from paired, scope-compatible energy measurements.

These functions do not establish measurement comparability or solution validity.
Callers supply matched rounds, energy in joules per instance, and a quality
requirement established independently of this analysis. Bootstrap intervals
are descriptive and conditional on the supplied training debt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _number(value: float, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _minimum_units(debt: float, saving: float, size: int) -> int:
    units = math.ceil((debt / saving) / size)
    # Division can put an exact repayment boundary one ulp above an integer.
    # Check adjacent deployments against the debt instead of using a tolerance.
    if units > 0 and ((units - 1) * size) * saving >= debt:
        return units - 1
    if (units * size) * saving < debt:
        return units + 1
    return units


def energy_break_even(
    training_debt_j: float,
    saving_j_per_instance: float,
    *,
    batch_size: int = 1,
    debt_complete: bool = True,
) -> dict[str, Any]:
    """Return the energy-only crossing and first deployment in complete batches.

    A nonpositive marginal saving has no finite crossing. With incomplete debt,
    finite thresholds are lower bounds, not estimates of the full training cost.
    Quality is deliberately not evaluated here.
    """
    debt = _number(training_debt_j, "training_debt_j")
    saving = _number(saving_j_per_instance, "saving_j_per_instance", nonnegative=False)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(debt_complete) is not bool:
        raise ValueError("debt_complete must be a boolean")
    crossing = debt / saving if saving > 0 else None
    if crossing is not None and not math.isfinite(crossing):
        raise ValueError("break-even volume exceeds the finite numeric range")
    batches = _minimum_units(debt, saving, batch_size) if crossing is not None else None
    return {
        "energy_status": "finite" if crossing is not None else "infinite",
        "training_debt_j": debt,
        "saving_j_per_instance": saving,
        "batch_size": batch_size,
        "continuous_instances": crossing,
        "minimum_instances": _minimum_units(debt, saving, 1) if crossing is not None else None,
        "full_batches": batches,
        "full_batch_instances": batches * batch_size if batches is not None else None,
        "debt_complete": debt_complete,
        "threshold_is_lower_bound": not debt_complete and crossing is not None,
    }


def _samples(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size < 2 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain at least two numeric paired rounds")
    array = array.astype(float)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite nonnegative energies")
    return array


def paired_aet(
    neural_j_per_instance: ArrayLike,
    baseline_j_per_instance: ArrayLike,
    training_debt_j: float,
    *,
    batch_size: int = 1,
    debt_complete: bool = True,
    required_tolerance_pct: float | None = None,
    tolerances_pct: Sequence[float] = (),
    bootstrap_replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap matched energy rounds and evaluate AET over a tolerance grid.

    Position i in both arrays must describe the same experimental round and
    compatible measurement scopes. Training debt stays fixed in every draw;
    repeated inference measurements do not estimate training-seed uncertainty.
    ``required_tolerance_pct`` is the independently established quality bound
    (for example the maximum of declared mean and tail bounds). Equality passes.
    Missing quality, incomplete debt, or a saving interval crossing zero leaves
    AET unidentified. The energy-only point crossing is always retained.
    """
    neural = _samples(neural_j_per_instance, "neural_j_per_instance")
    baseline = _samples(baseline_j_per_instance, "baseline_j_per_instance")
    if neural.shape != baseline.shape:
        raise ValueError("neural and baseline measurements must have matching paired rounds")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be an integer of at least two")
    confidence = _number(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    required = (
        _number(required_tolerance_pct, "required_tolerance_pct")
        if required_tolerance_pct is not None
        else None
    )
    tolerances = [_number(tau, "tolerance_pct") for tau in tolerances_pct]
    savings = baseline - neural
    point = energy_break_even(
        training_debt_j, float(savings.mean()), batch_size=batch_size, debt_complete=debt_complete
    )
    rng = np.random.default_rng(seed)
    draws = savings[rng.integers(0, len(savings), (bootstrap_replicates, len(savings)))].mean(
        axis=1
    )
    alpha = (1 - confidence) / 2
    ordered = np.sort(draws)
    low = float(ordered[math.floor(alpha * (bootstrap_replicates - 1))])
    high = float(ordered[math.ceil((1 - alpha) * (bootstrap_replicates - 1))])
    energy_status = "finite" if low > 0 else "infinite" if high <= 0 else "unidentified"
    # Map the same noninterpolated saving endpoints through the decreasing
    # threshold function. Thus a nonpositive lower saving gives an unbounded
    # upper threshold and cannot simultaneously qualify a finite AET.
    lower, upper = [
        energy_break_even(training_debt_j, delta, batch_size=batch_size)["full_batch_instances"]
        for delta in (high, low)
    ]
    curve = []
    for tau in tolerances:
        quality = (
            "unidentified" if required is None else "feasible" if required <= tau else "infeasible"
        )
        if quality != "feasible":
            status, reason = (
                quality,
                "quality_unknown" if required is None else "quality_constraint_failed",
            )
        elif energy_status == "infinite":
            status, reason = "infinite", "no_marginal_saving"
        elif not debt_complete:
            status, reason = "unidentified", "training_debt_incomplete"
        elif energy_status == "unidentified":
            status, reason = "unidentified", "saving_interval_includes_zero"
        else:
            status, reason = "finite", "quality_passed_and_positive_saving"
        curve.append(
            {
                "tolerance_pct": tau,
                "quality_status": quality,
                "aet_status": status,
                "reason": reason,
                "continuous_instances": point["continuous_instances"]
                if status == "finite"
                else None,
                "full_batch_instances": point["full_batch_instances"]
                if status == "finite"
                else None,
                "aet_lower_bound_instances": (
                    point["continuous_instances"]
                    if quality == "feasible" and not debt_complete and energy_status == "finite"
                    else None
                ),
            }
        )
    return {
        "energy_break_even": point,
        "energy_status": energy_status,
        "neural_j_per_instance_mean": float(neural.mean()),
        "baseline_j_per_instance_mean": float(baseline.mean()),
        "paired_rounds": len(savings),
        "saving_interval_j_per_instance": [float(low), float(high)],
        "full_batch_instances_interval": [lower, upper],
        "interval_lower_unbounded": lower is None,
        "interval_upper_unbounded": upper is None,
        "bootstrap_nonpositive_saving_fraction": float((draws <= 0).mean()),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "confidence": confidence,
        "uncertainty": "descriptive_paired_round_bootstrap_conditional_on_observed_debt",
        "required_tolerance_pct": required,
        "aet_by_tolerance": curve,
    }
