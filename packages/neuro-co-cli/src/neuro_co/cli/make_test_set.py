"""Generate a fixed test-instance file from a core env.

Resets a core env with a fixed seed and saves the instance State fields to a
`.pt` file, so eval / explain runs share one held-out set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuro_co.core.factory import make_env

_INSTANCE_FIELDS = (
    "coords",
    "demand",
    "tw_early",
    "tw_late",
    "prize",
    "proc_times",
    "ops_ma_adj",
)


def make_test_set(
    problem: str,
    *,
    num_instances: int = 1000,
    size: int = 50,
    seed: int = 12345,
    out_path: str | Path = "test_set.pt",
    **env_kwargs: Any,
) -> Path:
    """Build + persist a fixed test set; returns the output path."""
    env = make_env(problem, size=size, **env_kwargs)
    state = env.reset(num_instances, generator=torch.Generator().manual_seed(seed))
    payload = {f: getattr(state, f).cpu() for f in _INSTANCE_FIELDS if hasattr(state, f)}
    payload["_meta"] = {"problem": problem, "size": size, "seed": seed, "n": num_instances}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out
