# neuro-co-aet

Energy and carbon accounting for experiments, with Amortized Efficiency
Threshold (AET) analysis. AET measures how many solved instances are needed to
recover training energy at a specified solution-quality tolerance.

`EnergyTracker` supports CodeCarbon, hardware counters, and an injected wall
meter. Records distinguish measurement scope and fallback estimates.
`EnergyReading.to_dict()` uses joules, kilograms of CO2, seconds, and watts;
`to_legacy_dict()` provides Wh and gram aliases.

```bash
uv run --no-sync neuroco-aet-report outputs/ --out aet_report --deltas 0.5 1 2 5
```

The report reads `energy_train.json`, `energy_eval.json`, and
`energy_baseline.json`. It writes a CSV table, a Markdown summary, and figures.
Without a validated `--expected-manifest`, reports are exploratory. Hardware
counter diagnostics do not replace calibrated whole-system measurements.

See [energy accounting](../../docs/aet.md) for backend controls and evidence
requirements. The optional `experiments` extra installs experiment dependencies;
`rapl` adds Linux CPU counters and `analysis` adds pandas.

For paired measurements with a declared quality bound, use the library API:

```python
from neuro_co.aet.analysis import paired_aet

result = paired_aet(
    neural_j_per_instance=[6.4, 6.6, 6.5],
    baseline_j_per_instance=[45.0, 46.0, 45.5],
    training_debt_j=2_000_000,
    batch_size=64,
    required_tolerance_pct=3.0,
    tolerances_pct=[1, 3, 5],
)
```

Positions in both arrays must identify matched rounds with compatible energy
scopes. The result retains the energy-only crossing when the quality gate fails.
Its bootstrap interval is descriptive and conditional on the supplied training
debt. With `debt_complete=False`, crossings are lower bounds and AET remains
unidentified. A saving interval crossing zero also leaves AET unidentified.
The quality bound must be established independently; this function does not
verify routes or infer quality from energy measurements.
