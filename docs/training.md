# Training

## PyTorch

`neuroco train` builds a policy through `neuro_co.core.factory`. Available
problems are `tsp`, `atsp`, `cvrp`, `cvrptw`, `op`, `pdp`, `mtsp`, and `fjsp`.
Algorithms are `reinforce`, `pomo`, and `ppo`; backbones are `am`, `gnn`, `matnet`,
and `mamba`. The `gnn` backbone requires the core `gnn` extra. Backbone
compatibility depends on the problem representation.

```bash
uv run --no-sync neuroco train --problem tsp --algo pomo --size 20 \
  --epochs 1 --steps-per-epoch 2 --batch-size 8 --device cpu \
  --out-dir outputs/tsp
uv run --no-sync neuroco eval --problem tsp --algo pomo --size 20 \
  --ckpt-path outputs/tsp/best.pt --device cpu
```

Evaluation restores architecture metadata from the checkpoint. Pass the same
problem, size, and algorithm as training. The CLI checkpoints contain model
weights and architecture metadata; they do not contain an optimizer state for
resuming training.

## Mamba

Install the core `mamba` extra, then select the backbone:

```bash
uv sync --all-packages --extra mamba
uv run --no-sync neuroco train --problem tsp --backbone mamba --size 20 \
  --epochs 1 --steps-per-epoch 2 --batch-size 8 \
  --hidden-dim 32 --num-layers 1 --device cpu --out-dir outputs/mamba
```

`MambaModel` and `SSMEncoder` are available from `neuro_co.core.models`.
The `mamba-cuda` and `mamba-macos` extras provide native implementations.
Automatic selection depends on the available device and installed backend.

## JAX

The JAX runner trains an attention model with POMO on CVRP or TSP. Install the
core `jax` extra, then run a small CPU-compatible configuration:

```bash
uv sync --all-packages --extra jax
uv run --no-sync python scripts/train_jax.py --problem cvrp \
  --steps 2 --size 6 --batch-size 2 --n-starts 3 \
  --hidden-dim 16 --num-layers 1 --num-heads 2 --precision fp32 \
  --output outputs/jax-cvrp
```

The module entry point accepts the same arguments:
`python -m neuro_co.core.jax_backend.train`.
The runner writes `config.json`, `metrics.jsonl`, and `checkpoint.npz` under
`--output`. Checkpoints include parameters, optimizer state, and the random key.
Resume by passing the checkpoint and a larger total step target:

```bash
uv run --no-sync python scripts/train_jax.py --problem cvrp \
  --steps 4 --size 6 --batch-size 2 --n-starts 3 \
  --hidden-dim 16 --num-layers 1 --num-heads 2 --precision fp32 \
  --output outputs/jax-cvrp --resume outputs/jax-cvrp/checkpoint.npz
```

JAX checkpoints are separate from PyTorch checkpoints. The JAX backend does not
provide the PyTorch Mamba, MatNet, explanation, or scheduling workflows.

## Exporting solutions

`neuro_co.core.inference.greedy_rollout_actions(model, env, state)` returns Torch
actions without retaining a full trace. JAX provides
`JaxPOMO.greedy_rollout_actions(params, state)`, which supports `jax.jit`.
Both accept an initially unfinished environment state and return integer arrays
with shape `(batch, steps)`. Completed rows use `-1` padding; Torch trims trailing
columns once the batch finishes. Add the initial city when exporting a TSP tour.
CVRP actions include depot returns, so split the sequence at each return.
