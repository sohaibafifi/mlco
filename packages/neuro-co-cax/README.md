# neuro-co-cax

Constraint-based explanation methods for core policies: attribution by constraint
family, sufficient subsets, and counterfactual utilities.

```python
from neuro_co.cax import lambda_attribution, cp_minimal_subset, get_constraints
```

Constraint maps group input features by problem-specific families. Optional
solver-backed methods use the `cp` extra. Feasibility and minimality claims depend
on the method's assumptions, solver status, and termination limits; inspect the
returned report before interpreting a result.

This package contains experimental APIs. See each function's docstring and tests
for its supported problem representation.
