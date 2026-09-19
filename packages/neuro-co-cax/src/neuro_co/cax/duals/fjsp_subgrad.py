"""Heuristic precedence-multiplier estimation for flexible job shops.

Each operation selects its fastest eligible machine. A list-scheduling pass
computes start times, and within-job precedence residuals drive multiplier
updates. Eligibility remains enforced by the machine choices, so its reported
multiplier is zero. The heuristic schedule does not certify a dual bound.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from neuro_co.cax.duals.subgrad import subgradient_ascent


def fjsp_subgrad_duals(
    instance: dict[str, Any],
    *,
    max_iters: int = 40,
    initial_step: float = 1.0,
) -> dict[str, float]:
    """Return mean |lambda| per family from a small subgradient run."""
    proc = np.asarray(instance["proc_times"], dtype=float)
    if proc.ndim != 2:
        return {"processing": 0.0, "precedence": 0.0, "eligibility": 0.0}
    job_map_hint = np.asarray(instance.get("ops_job_map", []))
    hinted_ops = int(job_map_hint.shape[-1]) if job_map_hint.size else proc.shape[0]
    if proc.shape[0] != hinted_ops and proc.shape[1] == hinted_ops:
        proc = proc.T
    n_ops, M = proc.shape
    if n_ops < 2 or M < 1:
        return {"processing": 0.0, "precedence": 0.0, "eligibility": 0.0}

    if "ops_ma_adj" in instance:
        elig = np.asarray(instance["ops_ma_adj"], dtype=float) > 0.5
        if elig.shape != proc.shape and elig.T.shape == proc.shape:
            elig = elig.T
    else:
        elig = proc > 0.0

    job_of = np.asarray(instance.get("ops_job_map", np.zeros(n_ops, dtype=int)), dtype=int)
    seq_of = np.asarray(instance.get("ops_sequence_order", np.arange(n_ops, dtype=int)), dtype=int)
    if "pad_mask" in instance:
        pad = np.asarray(instance["pad_mask"], dtype=bool)
    else:
        pad = elig.sum(axis=1) == 0
    real_ops = [int(o) for o in range(n_ops) if not bool(pad[o])]
    if len(real_ops) < 2:
        return {"processing": 0.0, "precedence": 0.0, "eligibility": 0.0}

    # Pre-group ops by job (in sequence order).
    by_job: dict[int, list[int]] = {}
    for o in real_ops:
        by_job.setdefault(int(job_of[o]), []).append(o)
    for j in by_job:
        by_job[j].sort(key=lambda o: int(seq_of[o]))

    # Per-op fastest eligible machine + duration on it. Defaults: if no
    # eligible machine recorded, fall back to the cheapest non-zero
    # entry of `proc[:, o]` to keep the subproblem well-defined under
    # mild perturbations of `ops_ma_adj`.
    fastest_m = np.zeros(n_ops, dtype=int)
    fastest_p = np.zeros(n_ops, dtype=float)
    for o in real_ops:
        cands = np.where(elig[o])[0]
        if cands.size == 0:
            cands = np.where(proc[o] > 0)[0]
        if cands.size == 0:
            fastest_p[o] = 0.0
            fastest_m[o] = 0
            continue
        durations = proc[o, cands]
        best = int(np.argmin(durations))
        fastest_m[o] = int(cands[best])
        fastest_p[o] = float(durations[best])

    def subproblem(lam: dict[str, float]) -> tuple[float, dict[str, float]]:
        """List-scheduling subproblem with soft within-job precedence penalty.

        Schedules each op on its fastest eligible machine in an
        order biased by `lambda_prec` and `ops_sequence_order`:
        when the multiplier is high, ops earlier within their job
        are dispatched first, which tightens the precedence
        violation.
        """
        l_prec = lam.get("precedence", 0.0)
        order = sorted(
            real_ops,
            key=lambda o: (int(seq_of[o]) * (1.0 + l_prec), int(fastest_m[o]), o),
        )
        s = np.zeros(n_ops, dtype=float)
        machine_busy_until: dict[int, float] = {}
        for o in order:
            m = int(fastest_m[o])
            s[o] = machine_busy_until.get(m, 0.0)
            machine_busy_until[m] = s[o] + float(fastest_p[o])

        prec_violation = 0.0
        for ops in by_job.values():
            for a, b in pairwise(ops):
                slack = float(s[a]) + float(fastest_p[a]) - float(s[b])
                if slack > 0:
                    prec_violation += slack

        makespan = float(max(s[o] + fastest_p[o] for o in real_ops))
        L_val = makespan - l_prec * prec_violation
        return L_val, {"precedence": prec_violation}

    report = subgradient_ascent(
        family_names=["precedence"],
        subproblem=subproblem,
        max_iters=max_iters,
        initial_step=initial_step,
    )
    out = dict(report.multipliers)
    out.setdefault("precedence", 0.0)
    out["processing"] = 0.0  # duration remains in the scheduling subproblem
    out["eligibility"] = 0.0  # enforced in subproblem; see module docstring
    return out
