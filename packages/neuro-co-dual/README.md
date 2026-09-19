# neuro-co-dual

Multiplier-based cost shaping and policy conditioning for neural combinatorial
optimization. The package uses the dual backends in `neuro-co-cax`.

```python
from neuro_co.dual import (
    constraint_conditioned_features,
    shape_advantage_per_step,
    shape_reward_global,
)
```

Despite its name, `shape_reward_global` returns a shaped cost:
`cost + alpha * sum(mu * slack)`. Negating that cost gives the corresponding
reward. `shape_advantage_per_step` applies a violation penalty, while
`constraint_conditioned_features` adds multipliers to model inputs.

The `duals` extra installs solver dependencies. Supported constraint families and
multiplier methods depend on the problem; see the function docstrings and tests.
