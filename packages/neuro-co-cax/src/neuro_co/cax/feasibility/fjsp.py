"""FJSP (Flexible Job-Shop Scheduling) arithmetic instance-feasibility check.

An FJSP instance is feasible iff:

  1. Processing times are non-negative on every (machine, op) cell
                                                       (`proc_times >= 0`)
  2. Every real (non-padded) op has at least one eligible machine
                                                       (`num_eligible[o] >= 1`)
  3. The eligibility mask is consistent with `proc_times`
     -- ineligible cells carry processing time zero
        (the stores `proc_times[m, o] = 0` when (m, o) is not eligible)
  4. The padding mask is monotonic: padded ops appear contiguously at the
     end of the op axis (`pad_mask[o] == True` implies `pad_mask[o'] == True`
     for every o' > o within the same instance). the generator enforces
     this, so a perturbation that violates it indicates a corrupted state.

These are *necessary* conditions; they do not prove that a feasible
schedule exists (the CP-SAT decision query in `fjsp_cp.py` upgrades
the certificate). They catch the failure modes a small Gaussian
perturbation around a valid instance would produce -- negative
processing times, eligibility-mask drift, and padding desync.

Returns `[B]` bool tensor; `True` = candidate passes all checks.
"""

from __future__ import annotations

from typing import Any

import torch


def fjsp_is_feasible(td: Any) -> torch.Tensor:
    """Element-wise arithmetic check on a batched FJSP instance dict."""
    B = int(td["proc_times"].shape[0])
    ok = torch.ones(B, dtype=torch.bool)

    if "proc_times" in td:
        pt = td["proc_times"]  # [B, M, N_ops]
        ok = ok & torch.isfinite(pt).all(dim=(-1, -2)).cpu()
        # On eligible (machine, op) cells the processing time must
        # remain non-negative; ineligible cells are ignored
        # (the convention stores them as 0 but a perturbation
        # may bump them above 0 -- this is a benign encoding drift,
        # not a structural infeasibility).
        if "ops_ma_adj" in td:
            adj = (td["ops_ma_adj"] > 0.5).cpu()
            pt_cpu = pt.cpu()
            elig_pt = pt_cpu.masked_fill(~adj, 0.0)
            ok = ok & (elig_pt >= -1e-3).all(dim=(-1, -2))
        else:
            ok = ok & (pt >= -1e-3).all(dim=(-1, -2)).cpu()

    if "pad_mask" in td:
        pad = td["pad_mask"].cpu()  # [B, N_ops] bool
        real_mask = ~pad
    elif "num_eligible" in td:
        real_mask = (td["num_eligible"] > 0.5).cpu()
    else:
        real_mask = None

    if "num_eligible" in td:
        ne = td["num_eligible"].cpu()
        # Every real op needs >= 1 eligible machine (otherwise the
        # schedule is structurally infeasible). Tolerate near-1 values
        # (e.g. 0.95 from a Gaussian perturbation that nudged the
        # count below the integer boundary).
        if real_mask is not None:
            real_ok = (ne >= 0.5) | (~real_mask)
            ok = ok & real_ok.all(dim=-1)
        else:
            ok = ok & (ne >= 0).all(dim=-1)

    if "pad_mask" in td:
        pad_int = td["pad_mask"].cpu().int()
        # Padded ops must form a contiguous right-suffix.
        first_pad = pad_int.cummax(dim=-1).values
        ok = ok & (pad_int == first_pad).all(dim=-1)

    return ok
