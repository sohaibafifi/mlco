"""Top-k attribution stability across runs.

Given several `AttributionTrace`s computed from independent training
seeds (or different attribution methods) on the **same** instances,
report how stable the top-k node sets are.

Per `(instance, step)`, the Jaccard overlap between two traces is

    J(A, B) = |A ^ B| / |A U B|

The expected chance Jaccard for two independent uniform top-k draws
from N nodes is `k / (2N - k)` (≈ `k/N` when `k << N`). A pairwise
mean Jaccard far above chance means the attribution is reproducible;
near chance means it's a brittle artefact of the random seed.

Designed to consume traces in memory (from `gradient_attribution`,
`integrated_gradients`, `contrastive_attribution`) or the
`explanation.json` files written by `explain_policy`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StabilityReport:
    """Aggregate top-k Jaccard stability across `n` traces."""

    n_traces: int
    top_k_used: int
    num_steps: int
    num_instances: int
    num_nodes: int
    mean_jaccard: float
    chance_jaccard: float
    pairwise_jaccard: list[list[float]]  # [n][n] matrix
    per_step_jaccard: list[float]  # mean over (instance, pair), per step


def _to_top_k_array(trace: Any, top_k: int) -> Any:
    """Accept `AttributionTrace` or a dict-loaded JSON `attribution` block."""
    import numpy as np

    if hasattr(trace, "top_k_nodes"):
        arr = (
            trace.top_k_nodes.cpu().numpy()
            if hasattr(trace.top_k_nodes, "cpu")
            else trace.top_k_nodes
        )
    elif isinstance(trace, dict) and "attribution" in trace:
        arr = np.asarray(trace["attribution"]["top_k_nodes"])
    elif isinstance(trace, dict) and "top_k_nodes" in trace:
        arr = np.asarray(trace["top_k_nodes"])
    else:
        raise TypeError(f"Cannot extract top_k_nodes from {type(trace)!r}")
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"expected `[batch, T, K]`, got {arr.shape}")
    k = min(int(top_k), arr.shape[-1])
    return arr[..., :k]


def _jaccard(a: Any, b: Any) -> float:
    sa, sb = set(int(x) for x in a.tolist()), set(int(x) for x in b.tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def top_k_stability(
    traces: Sequence[Any],
    *,
    top_k: int = 5,
    num_nodes: int | None = None,
) -> StabilityReport:
    """Pairwise Jaccard stability of top-k attributions across traces."""
    import numpy as np

    if len(traces) < 2:
        raise ValueError("need at least 2 traces to measure stability")
    arrays = [_to_top_k_array(t, top_k) for t in traces]
    shape = arrays[0].shape
    for a in arrays[1:]:
        if a.shape != shape:
            raise ValueError(
                f"top_k_nodes shape mismatch across traces: {shape} vs {a.shape}; "
                "stability assumes traces were computed on the same instances."
            )
    n_traces, (batch, T, k) = len(arrays), shape
    pairwise = np.zeros((n_traces, n_traces), dtype=float)
    per_step = np.zeros(T, dtype=float)
    pair_count = 0
    for i in range(n_traces):
        pairwise[i, i] = 1.0
        for j in range(i + 1, n_traces):
            jaccs: list[float] = []
            for b in range(batch):
                for t in range(T):
                    jaccs.append(_jaccard(arrays[i][b, t], arrays[j][b, t]))
                    per_step[t] += _jaccard(arrays[i][b, t], arrays[j][b, t])
            pairwise[i, j] = pairwise[j, i] = float(np.mean(jaccs))
            pair_count += 1
    per_step /= max(1, pair_count * batch)
    iu = np.triu_indices(n_traces, k=1)
    mean_jaccard = float(pairwise[iu].mean()) if pair_count > 0 else 0.0
    # Chance: |A ^ B| / |A U B| for two independent uniform top-k draws
    # from `N`. Approximated as `k / (2N - k)`.
    if num_nodes is None:
        # Infer from max node index seen across all traces (depot at 0).
        max_idx = 0
        for arr in arrays:
            max_idx = max(max_idx, int(arr.max()))
        num_nodes = max_idx + 1
    chance = k / max(1, 2 * num_nodes - k)
    return StabilityReport(
        n_traces=n_traces,
        top_k_used=k,
        num_steps=T,
        num_instances=batch,
        num_nodes=num_nodes,
        mean_jaccard=mean_jaccard,
        chance_jaccard=float(chance),
        pairwise_jaccard=pairwise.tolist(),
        per_step_jaccard=per_step.tolist(),
    )


def load_explanation_traces(
    paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    """Read `explanation.json` files. Sugar for the CLI / notebook."""
    out: list[dict[str, Any]] = []
    for p in paths:
        out.append(json.loads(Path(p).read_text()))
    return out
