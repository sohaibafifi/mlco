#!/usr/bin/env python3
"""Install built wheels in fresh environments and exercise their public commands."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "minimal": ("core", "problems"),
    "cli": ("core", "problems", "attr", "probe", "cli"),
    "report": ("core", "problems", "attr", "probe", "cli"),
    "full": ("core", "problems", "attr", "probe", "cli", "aet", "scale", "cax", "dual", "xai"),
}

IMPORT_SMOKE = """
import importlib
import importlib.metadata
import importlib.util
import pathlib
import sys
import torch
import numpy as np
from neuro_co.core.factory import make_algo, make_env, make_model
from neuro_co.problems import RoutingInstance, SolveResult, solve

for package in sys.argv[1:]:
    module = importlib.import_module('neuro_co.' + package)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
    distribution = importlib.metadata.distribution('neuro-co-' + package)
    files = distribution.files or []
    assert distribution.metadata.get_all('License-File'), package
    assert any('.dist-info/licenses/' in str(path) and path.name == 'LICENSE' for path in files), package

if sys.argv[1:] == ['core', 'problems']:
    for optional in ('neuro_co.attr', 'neuro_co.probe', 'neuro_co.cax', 'jax', 'torch_geometric'):
        assert importlib.util.find_spec(optional) is None, optional

env = make_env('tsp', size=5)
model = make_model(env, hidden_dim=16, num_layers=1, num_heads=2)
algo = make_algo('reinforce', model, env, batch_size=2, eval_batch_size=2)
algo.train_step(torch.Generator().manual_seed(7))
assert algo.eval_step(torch.Generator().manual_seed(8))
instance = RoutingInstance(problem='tsp', coords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
result = solve(instance, method='nearest_neighbour')
assert isinstance(result, SolveResult)
assert result.feasible and result.objective == 4.0
assert result.to_dict()['schema'] == 'neuro-co-solve/v1'
if 'cax' in sys.argv[1:]:
    cvrp = RoutingInstance(problem='cvrp', coords=instance.coords,
                          demand=np.array([0, 1, 1, 1]), capacity=2)
    for method in ('ortools', 'pyvrp'):
        result = solve(cvrp, method=method, max_runtime=0.2)
        assert result.feasible and result.validation.feasible, (method, result.to_dict())
"""


def check_profile(dist: Path, profile: str, python: str, *, offline: bool = False) -> None:
    """Resolve internal packages only from the selected wheel artifacts."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv before running distribution smoke checks")
    packages = PROFILES[profile]
    requirements = []
    for package in packages:
        wheels = sorted(dist.glob(f"neuro_co_{package}-*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected exactly one {package} wheel in {dist}, found {len(wheels)}")
        wheel = wheels[0]
        extras = (
            "[report]"
            if package == "cli" and profile == "report"
            else "[cp]"
            if package == "problems" and profile == "full"
            else ""
        )
        requirements.append(f"neuro-co-{package}{extras} @ {wheel.as_uri()}")

    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    with tempfile.TemporaryDirectory(prefix=f"mlco-wheels-{profile}-") as scratch:
        work = Path(scratch)
        venv = work / "venv"
        command = [uv, "venv", "--python", python, str(venv)]
        if offline:
            command.insert(1, "--offline")
        subprocess.run(command, cwd=work, env=env, check=True)
        executable = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command = [uv, "pip", "install", "--python", str(executable), *requirements]
        if offline:
            command.insert(1, "--offline")
        subprocess.run(command, cwd=work, env=env, check=True)
        subprocess.run(
            [str(executable), "-I", "-c", IMPORT_SMOKE, *packages],
            cwd=work,
            env=env,
            check=True,
        )
        if profile != "minimal":
            neuroco = executable.parent / ("neuroco.exe" if os.name == "nt" else "neuroco")
            subprocess.run([str(neuroco), "--help"], cwd=work, env=env, check=True)
            if profile == "full":
                for command in ("neuroco-scale", "neuroco-aet-report", "neuroco-explain-report"):
                    entrypoint = executable.parent / (command + (".exe" if os.name == "nt" else ""))
                    subprocess.run([str(entrypoint), "--help"], cwd=work, env=env, check=True)
            if profile == "report":
                recipe = work / "smoke.yaml"
                shutil.copyfile(ROOT / "recipes/smoke.yaml", recipe)
                subprocess.run(
                    [str(neuroco), "experiment", str(recipe)], cwd=work, env=env, check=True
                )
            else:
                subprocess.run(
                    [
                        str(neuroco),
                        "train",
                        "--problem",
                        "tsp",
                        "--size",
                        "5",
                        "--epochs",
                        "1",
                        "--steps-per-epoch",
                        "1",
                        "--batch-size",
                        "2",
                        "--eval-batch-size",
                        "2",
                        "--hidden-dim",
                        "16",
                        "--num-layers",
                        "1",
                        "--num-heads",
                        "2",
                        "--device",
                        "cpu",
                        "--out-dir",
                        "run",
                    ],
                    cwd=work,
                    env=env,
                    check=True,
                )
                subprocess.run(
                    [str(neuroco), "eval", "--problem", "tsp", "--ckpt-path", "run/best.pt"],
                    cwd=work,
                    env=env,
                    check=True,
                )
            instance = work / "instance.json"
            nodes = 6 if profile == "report" else 5
            instance.write_text(
                json.dumps(
                    {
                        "problem": "tsp",
                        "coords": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.8]][:nodes],
                    }
                )
            )
            checkpoint = (
                "experiments/smoke/runs/tsp/train_seed0/best.pt"
                if profile == "report"
                else "run/best.pt"
            )
            for options in (["--method", "nearest_neighbour"], ["--ckpt", checkpoint]):
                subprocess.run(
                    [str(neuroco), "solve", str(instance), *options, "--out", "solution.json"],
                    cwd=work,
                    env=env,
                    check=True,
                )
                result = json.loads((work / "solution.json").read_text())
                assert result["feasible"] is True and result["status"] == "feasible"
    print(f"[wheels] {profile}: passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--offline", action="store_true", help="Use only uv's cached dependencies")
    args = parser.parse_args()
    for profile in PROFILES if args.profile == "all" else [args.profile]:
        check_profile(args.dist_dir.resolve(), profile, args.python, offline=args.offline)


if __name__ == "__main__":
    main()
