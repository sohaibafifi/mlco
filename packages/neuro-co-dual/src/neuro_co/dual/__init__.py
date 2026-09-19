"""Multiplier-based cost shaping and policy conditioning.

The shaping functions use `neuro_co.cax.duals` to weight trajectory costs
or step-level violations. `constraint_conditioned_features` exposes the
same multipliers as encoder inputs. Solver overhead depends on the problem,
batch size, and multiplier backend.
"""

from __future__ import annotations

from neuro_co.dual.baselines import (
    AdaptiveLagrangian,
    shape_reward_oracle_optimum,
    shape_reward_scalar,
)
from neuro_co.dual.conditioned import (
    ConstraintConditionedAug,
    constraint_conditioned_features,
)
from neuro_co.dual.shaping import (
    shape_advantage_per_step,
    shape_reward_global,
)
from neuro_co.dual.slack import family_slack, family_violation

__all__ = [
    "AdaptiveLagrangian",
    "ConstraintConditionedAug",
    "constraint_conditioned_features",
    "family_slack",
    "family_violation",
    "shape_advantage_per_step",
    "shape_reward_global",
    "shape_reward_oracle_optimum",
    "shape_reward_scalar",
]

__version__ = "0.2.0"
