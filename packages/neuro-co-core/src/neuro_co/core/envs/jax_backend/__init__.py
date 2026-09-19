"""Compatibility imports for the JAX environments in neuro-co-problems.

Install:
    uv pip install -e 'packages/neuro-co-core[jax]' -e packages/neuro-co-problems

Imports here will fail with a helpful message if JAX is not installed.
"""

try:
    import jax  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "JAX backend requires `jax` and `jaxlib`. Install with:\n"
        "  uv pip install -e 'packages/neuro-co-core[jax]'"
    ) from e

from neuro_co.problems.cvrp.jax_env import JaxCVRPEnv, JaxCVRPState
from neuro_co.problems.tsp.jax_env import JaxTSPEnv, JaxTSPState

__all__ = ["JaxCVRPEnv", "JaxCVRPState", "JaxTSPEnv", "JaxTSPState"]
