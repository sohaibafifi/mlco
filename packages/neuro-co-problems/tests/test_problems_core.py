"""problems concept banks on neuro-co-core: registration + State labels."""

import torch

from neuro_co.core.concepts import concept_registry
from neuro_co.core.envs.cvrptw import CVRPTWEnv
from neuro_co.core.envs.fjsp import FJSPEnv
from neuro_co.core.envs.op import OPEnv
from neuro_co.core.envs.pdp import PDPEnv
from neuro_co.problems import load_plugins


def test_load_plugins_registers_core_problems() -> None:
    load_plugins()
    for name in ("cvrptw", "op", "pdp", "fjsp"):
        assert name in concept_registry, name


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
