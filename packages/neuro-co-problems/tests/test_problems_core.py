"""problems concept banks on neuro-co-core: registration + State labels."""

import subprocess
import sys

import torch

from neuro_co.core.concepts import concept_registry
from neuro_co.problems import get_bank, load_plugins
from neuro_co.problems.fjsp.env import FJSPEnv
from neuro_co.problems.op.env import OPEnv
from neuro_co.problems.pdp.env import PDPEnv
from neuro_co.problems.vrptw.env import CVRPTWEnv


def test_environment_construction_and_solver_lookup_leave_backends_unloaded() -> None:
    script = """
import pickle
import sys
from importlib.util import find_spec
from neuro_co.core.env_registry import make_env
from neuro_co.problems import get_solver, load_plugins

for problem in ("cvrptw", "op", "pdp", "fjsp"):
    make_env(problem, size=3)
load_plugins()
for problem, engine, dependency in (
    ("cvrptw", "pyvrp", "pyvrp"),
    ("cvrptw", "cpsat", "ortools"),
    ("op", "cpsat", "ortools"),
    ("fjsp", "cpsat", "ortools"),
    ("jssp", "cpsat", "ortools"),
):
    if find_spec(dependency) is None:
        try:
            get_solver(problem, engine)
        except KeyError:
            pass
        else:
            raise AssertionError((problem, engine))
    else:
        assert callable(get_solver(problem, engine))
        if problem == "cvrptw":
            assert get_solver("vrptw", engine) is get_solver(problem, engine)
solver = get_solver("pdp", "ortools")
assert pickle.loads(pickle.dumps(solver)) == solver
assert callable(get_solver("flp", "lp"))
assert "pyvrp" not in sys.modules
assert "ortools" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_load_plugins_registers_core_problems() -> None:
    load_plugins()
    for name in ("cvrptw", "op", "pdp", "fjsp"):
        assert name in concept_registry, name
        assert get_bank(name) is concept_registry.get(name)


def test_cvrptw_concepts_label_state() -> None:
    load_plugins()
    bank = concept_registry.get("cvrptw")
    env = CVRPTWEnv(size=6, capacity=20.0, horizon=10.0, window_width=2.0)
    state = env.reset(8, generator=torch.Generator().manual_seed(0))
    got = {n: fn(state) for n, fn in bank.concepts.items()}
    assert "high_demand" in got
    labels = got["high_demand"]
    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (8, 7)  # depot + 6
    assert (labels[:, 0] == -1).all()  # depot ignored
    assert set(labels[:, 1:].unique().tolist()).issubset({0, 1})


def test_op_pdp_fjsp_concepts_run() -> None:
    load_plugins()
    op_env = OPEnv(size=6, budget=4.0)
    op_state = op_env.reset(4, generator=torch.Generator().manual_seed(0))
    assert concept_registry.get("op").concepts["high_prize"](op_state).shape == (4, 7)

    pdp_env = PDPEnv(size=3)
    pdp_state = pdp_env.reset(4, generator=torch.Generator().manual_seed(0))
    assert concept_registry.get("pdp").concepts["is_pickup"](pdp_state).shape == (4, 7)

    fj_env = FJSPEnv(size=3, ops_per_job=2, num_machines=4)
    fj_state = fj_env.reset(4, generator=torch.Generator().manual_seed(0))
    assert concept_registry.get("fjsp").concepts["long_proc_time"](fj_state).shape == (4, 6)
