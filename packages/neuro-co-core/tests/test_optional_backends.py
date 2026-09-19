"""Core import and AM construction do not load optional backend dependencies."""

import subprocess
import sys


def test_optional_backend_imports_are_lazy():
    code = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"jax", "torch_geometric", "mambapy", "mamba_ssm"}:
            raise ImportError(f"blocked optional dependency: {fullname}")

sys.meta_path.insert(0, BlockOptional())
from neuro_co.core.factory import make_env, make_model
from neuro_co.core.models import MambaModel, SSMEncoder
from neuro_co.mamba import MambaModel as LegacyMambaModel
assert LegacyMambaModel is MambaModel
make_model(make_env("tsp", size=5), hidden_dim=8, num_heads=2, num_layers=1)
try:
    MambaModel(in_dim=2, hidden_dim=8, num_heads=2, num_layers=1, backend="mambapy")
except ImportError as exc:
    assert "neuro-co-core[mamba]" in str(exc)
else:
    raise AssertionError("Mamba construction should require its optional dependency")
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
