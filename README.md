# MLCO

Python packages for neural combinatorial optimization: routing and scheduling
policies, energy accounting and explanation methods.

## Install

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sohaibafifi/mlco.git
cd mlco
uv sync --all-packages
```

## Train and evaluate

```bash
uv run --no-sync neuroco train --problem cvrp --algo pomo --size 20 \
  --epochs 1 --steps-per-epoch 2 --batch-size 8 --device cpu \
  --out-dir outputs/cvrp
uv run --no-sync neuroco eval --problem cvrp --algo pomo --size 20 \
  --ckpt-path outputs/cvrp/best.pt --device cpu
```

Training writes checkpoints and metrics to `--out-dir`. See
[training](docs/training.md) for Mamba and JAX, and
[getting started](docs/getting-started.md) for installation options.

## Packages

| Package | Purpose |
|---|---|
| [core](packages/neuro-co-core/README.md) | Environments, policies, training algorithms, Mamba, and optional JAX backend |
| [cli](packages/neuro-co-cli/README.md) | Training, evaluation, explanation, and experiment commands |
| [scale](packages/neuro-co-scale/README.md) | Partitioning and hierarchical policies for large routing instances |
| [aet](packages/neuro-co-aet/README.md) | Energy measurement and Amortized Efficiency Threshold analysis |
| [attr](packages/neuro-co-attr/README.md) | Action attribution, faithfulness checks, and stability |
| [probe](packages/neuro-co-probe/README.md) | Encoder probes and embedding analysis |
| [cax](packages/neuro-co-cax/README.md) | Constraint-based explanation methods |
| [dual](packages/neuro-co-dual/README.md) | Multiplier-based cost shaping and policy conditioning |
| [problems](packages/neuro-co-problems/README.md) | Problem concepts and optional classical solvers |
| [xai](packages/neuro-co-xai/README.md) | Convenience imports for the explanation packages |

[Documentation](docs/index.md) covers usage and development.
The code is licensed under [MIT](LICENSE).
