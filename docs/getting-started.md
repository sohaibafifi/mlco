# Getting started

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Run the commands below
from the repository root.

```bash
git clone https://github.com/sohaibafifi/mlco.git
cd mlco
uv sync --all-packages
```

This installs the workspace packages in editable mode. Optional backends and
solvers require extras. Select the package and extra you need:

```bash
uv sync --all-packages --extra jax
uv sync --all-packages --extra mamba
uv sync --all-packages --extra cp
```

Native Mamba kernels have separate `mamba-cuda` and `mamba-macos` extras. They
require a compatible platform and toolchain. The `mamba` extra uses the PyTorch
implementation. JAX accelerator installation depends on the target machine;
the `jax` extra alone does not select a CUDA runtime.

## Small training run

```bash
uv run --no-sync neuroco train --problem cvrp --algo pomo --size 20 \
  --epochs 1 --steps-per-epoch 2 --batch-size 8 --device cpu \
  --out-dir outputs/cvrp
uv run --no-sync neuroco eval --problem cvrp --algo pomo --size 20 \
  --ckpt-path outputs/cvrp/best.pt --device cpu
```

The run directory contains `best.pt`, `latest.pt`, and `metrics.json`. Choose a
new `--out-dir` for each run; reusing it overwrites those files. This smoke run
checks the training workflow, not solution quality.

```bash
uv run --no-sync neuroco explain --problem cvrp --size 20 \
  --ckpt-path outputs/cvrp/best.pt --out-dir outputs/cvrp-explain
uv run --no-sync neuroco probe outputs/cvrp
```

Use `--help` on each command for its arguments. See [training](training.md) for
JAX and Mamba and [energy accounting](aet.md) to record energy separately.
