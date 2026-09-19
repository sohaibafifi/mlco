"""Add constraint multipliers to encoder features.

`constraint_conditioned_features(state, env, problem)` returns a `[B, K]`
multiplier tensor. `ConstraintConditionedAug` projects that tensor and adds
the result to each node embedding.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from neuro_co.cax import duals as _cax_duals
from neuro_co.cax._instance import to_batch_instance
from neuro_co.cax.constraint_map import get_constraints
from neuro_co.cax.duals import state_to_instance


def constraint_conditioned_features(
    state: Any,
    env: Any,
    problem: str,
    *,
    multipliers: str = "lp",
    per_instance: bool = True,
    duals_kwargs: dict[str, Any] | None = None,
) -> Tensor:
    """Compute `mu(x)` as a `FloatTensor[B, K]` for the batch.

    Column order matches
    `neuro_co.cax.constraint_map.PROBLEM_CONSTRAINTS[problem]`.
    """
    duals_kwargs = duals_kwargs or {}
    families = [name for name, _ in get_constraints(problem)]
    inst = to_batch_instance(state, env, problem)
    sample = next(iter(inst.values()))
    B = sample.shape[0] if hasattr(sample, "shape") and sample.ndim > 0 else 1
    device = sample.device if isinstance(sample, Tensor) else torch.device("cpu")

    rows: list[list[float]] = []
    n_solve = B if per_instance else 1
    for b in range(n_solve):
        try:
            instance = state_to_instance(state, env, problem, batch_idx=b)
            mu = _cax_duals.get_multipliers(problem, instance, method=multipliers, **duals_kwargs)
        except (NotImplementedError, ValueError, RuntimeError):
            mu = {name: 0.0 for name in families}
        rows.append([float(mu.get(name, 0.0)) for name in families])

    feats = torch.tensor(rows, dtype=torch.float32, device=device)
    if not per_instance:
        feats = feats.expand(B, -1)
    return feats  # [B, K]


class ConstraintConditionedAug(nn.Module):
    """Add a projected multiplier vector to each node embedding.

    `forward(x_nodes, state, env, problem)` returns augmented embeddings with
    shape `[B, N, d_model]`. With the default `init_scale=0`, the projection
    starts at zero and the module initially returns its input unchanged.
    """

    def __init__(self, d_model: int, n_families: int, init_scale: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(n_families, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=init_scale)

    def forward(
        self,
        x_nodes: Tensor,
        state: Any,
        env: Any,
        problem: str,
        *,
        multipliers: str = "lp",
        per_instance: bool = True,
        duals_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        feats = constraint_conditioned_features(
            state,
            env,
            problem,
            multipliers=multipliers,
            per_instance=per_instance,
            duals_kwargs=duals_kwargs,
        )  # [B, K]
        proj = self.proj(feats)  # [B, d_model]
        return x_nodes + proj.unsqueeze(1)  # broadcast over N
