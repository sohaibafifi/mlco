"""Lazy public imports stay callable across repeated access."""

import subprocess
import sys


def test_lazy_exports_are_stable():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import neuro_co.cax as cax; "
            "names = ('cp_counterfactual', 'benchmark_runs', "
            "'constraint_intervention_attribution'); "
            "first = [getattr(cax, name) for name in names]; "
            "assert all(callable(value) for value in first); "
            "assert all(getattr(cax, name) is value "
            "for name, value in zip(names, first))",
        ],
        check=True,
        timeout=30,
    )
