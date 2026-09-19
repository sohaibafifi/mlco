# neuro-co-core

Generic environment and state interfaces, neural policies, and training
algorithms for combinatorial optimization. Concrete environments are supplied
by [neuro-co-problems](../neuro-co-problems/README.md).

Install both packages to run the built-in problems. From a checkout:

```bash
uv pip install -e packages/neuro-co-core -e packages/neuro-co-problems
```

```python
from neuro_co.core.env_registry import available_envs, make_env
from neuro_co.core.factory import make_algo, make_model

env = make_env("cvrp", size=50, capacity=50.0)
model = make_model(env, backbone="am", hidden_dim=128)
algo = make_algo("pomo", model, env, device="cpu")
print(available_envs(backend="torch"))
```

The registry discovers installed environment providers.
`neuro_co.core.factory.make_env` remains compatible with this API.
Policies use attention, GNN, MatNet, or Mamba backbones; training algorithms
include REINFORCE, POMO, and PPO.

The `gnn` extra installs the graph encoder dependency. Use `mamba` for the
PyTorch Mamba implementation, `mamba-cuda` for CUDA kernels, or `mamba-macos`
for Apple Silicon. Select Mamba with `make_model(env, backbone="mamba")`.

The `jax` extra provides the attention model, POMO, optimizer, and training
utilities in `neuro_co.core.jax_backend` independently of concrete environments.
Install `neuro-co-problems[jax]` alongside core for TSP/CVRP JAX training; select
an environment with `make_env("cvrp", backend="jax", size=50)`.
See [training](../../docs/training.md) for runnable commands.
