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
