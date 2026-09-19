# neuro-co-attr

Action attribution for core policies: gradient times input, integrated gradients,
DeepLIFT, and contrastive attribution. Methods return `AttributionTrace` records
with node scores and top-k selections for each decision step.

```python
from neuro_co.attr import gradient_attribution, integrated_gradients
from neuro_co.attr import deletion_flip_rate, sufficiency_keep_rate, sanity_check
from neuro_co.attr import top_k_stability
```

Deletion and sufficiency tests measure changes in the selected action after
input masking. The sanity test compares explanations after weight randomization.
These diagnostics depend on the intervention and baseline choices; they do not
establish causal explanations on their own.

Use `neuroco explain --help` for the command-line workflow, or import the methods
directly. Encoder probing lives in [neuro-co-probe](../neuro-co-probe/README.md).
