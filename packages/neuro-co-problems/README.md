# neuro-co-problems

Problem concepts and classical solver adapters for `neuro-co`. Built-in plugins
cover `vrptw`, `jssp`, `pdp`, `fjsp`, `op`, and `flp`.

```python
from neuro_co.problems import get_bank, get_solver, load_plugins

load_plugins()
solver = get_solver("vrptw", "pyvrp")
```

Install the `pyvrp`, `ortools`, or combined `cp` extra for solver dependencies.
Solver arguments depend on the problem adapter. Plugins register through the
`neuro_co.problems` entry-point group.

Core training environments have their own registry. The command-line name for
routing with capacity and time windows is `cvrptw`.
