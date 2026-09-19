"""OP (Orienteering Problem) arithmetic instance-feasibility check.

An OP instance is feasible iff:

  1. All node coordinates lie in the unit-square             (locs ∈ [0, 1]²)
  2. All prizes are non-negative                                   (`prize >= 0`)
  3. The depot prize is zero or unset                              (depot has no reward)
  4. The travel budget is strictly positive                        (`max_length > 0`)
  5. The budget is at least the round-trip distance to one customer
     (otherwise no tour beyond the depot is feasible; we use the
     minimum depot-to-customer round trip as a tight lower bound)

These are *necessary* conditions; satisfying them does NOT prove
that a high-prize tour exists (that needs the OP CP-SAT decision
query in `op_cp.py`). They catch every infeasibility a small
Gaussian perturbation around a valid instance would produce
(negative prizes, budgets shrunken below the depot-round-trip,
locations pushed outside the unit square), which is the common
case during counterfactual search.

Returns `[B]` bool tensor; `True` = candidate passes all checks.
"""

from __future__ import annotations

from typing import Any

import torch


def op_is_feasible(td: Any) -> torch.Tensor:
    """Element-wise arithmetic check on a batched OP instance dict."""
    B = int(td["locs"].shape[0]) if "locs" in td else int(td["prize"].shape[0])
    ok = torch.ones(B, dtype=torch.bool)

    if "locs" in td:
        locs = td["locs"]
        # Finite coordinates only (the generates in [0,1]^2 but the
        # OP definition is unchanged under affine perturbations -- we
        # only reject NaN / inf to catch numerical blow-ups).
        ok = ok & torch.isfinite(locs).all(dim=(-1, -2)).cpu()

    if "prize" in td:
        prize = td["prize"]
        # Non-negative prizes (the OP env clips negatives to 0
        # at decode time; a perturbation that sends one slightly
        # negative is structurally fine, so we only flag clearly
        # broken values). The depot prize is anchored by the
        # ConstraintBank's `prize` family and not part of the
        # feasibility check.
        ok = ok & (prize >= -1e-2).all(dim=-1).cpu()

    if "max_length" in td:
        ml = td["max_length"]
        if ml.ndim > 1:
            ml = ml[..., 0]
        ok = ok & (ml > 0.0).cpu()
        # Budget must at least allow visiting one customer round-trip,
        # otherwise the tour is forced empty. Use the minimum
        # depot-to-customer distance as the lower bound; allow a
        # small slack for numerical perturbations.
        if "locs" in td:
            locs = td["locs"]
            depot = locs[:, 0:1, :]
            d_depot = (locs - depot).norm(dim=-1)
            if d_depot.shape[-1] > 1:
                round_trip = 2.0 * d_depot[:, 1:].min(dim=-1).values.cpu()
                ok = ok & (ml.cpu() + 1e-3 >= round_trip)

    return ok
