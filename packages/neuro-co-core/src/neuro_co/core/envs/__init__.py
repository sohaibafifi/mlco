"""Compatibility access to environments in neuro_co.problems.

Each problem is imported only when its environment or state is requested.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .atsp import ATSPEnv, ATSPState
    from .cvrp import CVRPEnv, CVRPState
    from .cvrptw import CVRPTWEnv, CVRPTWState
    from .fjsp import FJSPEnv, FJSPState
    from .mtsp import MTSPEnv, MTSPState
    from .op import OPEnv, OPState
    from .pdp import PDPEnv, PDPState
    from .tsp import TSPEnv, TSPState

__all__ = [
    "ATSPEnv",
    "ATSPState",
    "CVRPEnv",
    "CVRPState",
    "CVRPTWEnv",
    "CVRPTWState",
    "FJSPEnv",
    "FJSPState",
    "MTSPEnv",
    "MTSPState",
    "OPEnv",
    "OPState",
    "PDPEnv",
    "PDPState",
    "TSPEnv",
    "TSPState",
]

_PROBLEMS = {
    "ATSP": "atsp",
    "CVRP": "cvrp",
    "CVRPTW": "vrptw",
    "FJSP": "fjsp",
    "MTSP": "mtsp",
    "OP": "op",
    "PDP": "pdp",
    "TSP": "tsp",
}


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    problem = _PROBLEMS[name.removesuffix("Env").removesuffix("State")]
    value = getattr(import_module(f"neuro_co.problems.{problem}.env"), name)
    globals()[name] = value
    return value
