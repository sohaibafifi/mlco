# neuro-co-core

Environments, neural policies, and training algorithms for combinatorial
optimization. PyTorch is the default backend. State is represented by typed
dataclasses, and environments expose functional state transitions.

The factory supports `tsp`, `atsp`, `cvrp`, `cvrptw`, `op`, `pdp`, `mtsp`, and
`fjsp`. Policies use attention, GNN, MatNet, or Mamba backbones; training
algorithms include REINFORCE, POMO, and PPO. The GNN backbone requires the
optional `gnn` extra.

```python
from neuro_co.core.factory import make_algo, make_env, make_model

env = make_env("cvrp", size=50, capacity=50.0)
model = make_model(env, backbone="am", hidden_dim=128)
algo = make_algo("pomo", model, env, device="cpu")
```

Mamba lives in `neuro_co.core.models` and uses optional dependencies. Install
`neuro-co-core[mamba]` for the PyTorch fallback, `mamba-cuda` for CUDA kernels,
or `mamba-macos` for Apple Silicon. Select it with
`make_model(env, backbone="mamba")` or `neuroco train --backbone mamba`.

The `jax` extra provides functional TSP/CVRP environments, an attention model,
POMO training, and checkpoint utilities in `neuro_co.core.jax_backend`.
See [training](../../docs/training.md) for runnable commands.
