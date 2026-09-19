"""Optional pure-JAX environment backend for compiled rollouts.

Install:
    uv pip install -e 'packages/neuro-co-core[jax]'

Imports here will fail with a helpful message if JAX is not installed.
"""

try:
    import jax  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "JAX backend requires `jax` and `jaxlib`. Install with:\n"
        "  uv pip install -e 'packages/neuro-co-core[jax]'"
    ) from e

from .cvrp import JaxCVRPEnv, JaxCVRPState
from .tsp import JaxTSPEnv, JaxTSPState

__all__ = ["JaxCVRPEnv", "JaxCVRPState", "JaxTSPEnv", "JaxTSPState"]
