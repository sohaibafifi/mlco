# neuro-co-problems

Concrete environments, problem concepts, and classical solver adapters for
`neuro-co`. Install this package with `neuro-co-core` to use the built-in
training problems.

Torch environments cover `tsp`, `atsp`, `cvrp`, `cvrptw`, `op`, `pdp`, `mtsp`,
and `fjsp`. JAX environments cover `tsp` and `cvrp` and require this package's
`jax` extra. For JAX models and training, install `neuro-co-core` alongside
`neuro-co-problems[jax]`. Other plugins provide concepts and solver adapters
for `jssp` and `flp`.

```python
from neuro_co.core.env_registry import available_envs, make_env
from neuro_co.problems.cvrp.env import CVRPEnv

env = make_env("cvrp", size=20)
print(available_envs(backend="torch"))
```

Concrete Torch classes live in `neuro_co.problems.<problem>.env`. CVRPTW uses
`neuro_co.problems.vrptw.env.CVRPTWEnv`. JAX classes live in
`neuro_co.problems.tsp.jax_env` and `neuro_co.problems.cvrp.jax_env`; the registry
accepts `backend="jax"` to select them.

```python
from neuro_co.problems import get_bank, get_solver, load_plugins

load_plugins()
solver = get_solver("vrptw", "pyvrp")
```

Install `pyvrp`, `ortools`, or the combined `cp` extra for classical solvers.
Solver arguments depend on the problem adapter. Concept and solver plugins
register through the `neuro_co.problems` entry-point group.
