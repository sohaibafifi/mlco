# neuro-co-probe

Linear concept probes and PCA, ICA, and t-SNE analysis of encoder activations.
The package also provides explanation reports for core policies.

```python
from neuro_co.probe import fit_concept_probes, discover_directions, fit_tsne
```

To inspect a training run containing `metrics.json` and `best.pt`:

```bash
uv run --no-sync neuroco probe outputs/cvrp
```

Probe reports include validation metrics and plots under the run directory.
Use `neuroco probe --help` for layer selection and analysis settings. The
`explain` extra adds action-attribution methods from `neuro-co-attr`.
