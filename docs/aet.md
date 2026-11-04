# Energy accounting

The Amortized Efficiency Threshold (AET) is the deployment volume needed to
recover a solver's training energy relative to a baseline, subject to a
solution-quality tolerance. With comparable per-instance measurements:

```text
AET = training energy / (baseline energy per instance - policy energy per instance)
```

A finite threshold requires both an acceptable quality gap and a positive energy
saving per instance. Batch size, hardware, runtime limits, and measurement scope
are part of the comparison.

## Measure a workload

`EnergyTracker` supports `codecarbon`, `hwcounters`, and `wall_meter` backends.
Default fallback can produce a TDP estimate when counters are unavailable.
Inspect `backend`, `energy_domains`, `measurement_scope`, and `extra` in the
result before interpreting it as a measurement.

For a host that exposes both GPU and CPU counters:

```python
from neuro_co.aet import EnergyTracker

with EnergyTracker(
    backend="hwcounters",
    allow_fallback=False,
    required_domains={"gpu", "cpu"},
    gpu_indices=[0],
    pue=1.0,
    report_embodied=False,
    items=batch_size,
) as tracker:
    run_workload()

record = tracker.reading.to_dict()
```

`to_dict()` uses SI units: joules, kilograms of CO2, seconds, and watts. The
explicit `to_legacy_dict()` adapter emits Wh and gram aliases. PUE adjusts
operational energy; embodied-carbon amortization is a separate estimate.
The core training CLI does not automatically produce AET energy records.

Hardware counters cover their reported devices or domains. Whole-system AC
measurement requires a calibrated physical meter implementing `WallMeterSampler`.
The `wall_meter` backend requires an injected sampler, `pue=1.0`,
`report_embodied=False`, and `allow_fallback=False`.

## Measure a GH200 module

On Linux GH200 hosts exposing Grace hwmon power sensors, use the explicit
module tracker. It integrates the selected Grace + Hopper module's power in a
separate process so Python training cannot stall polling by holding the GIL.
The module is shared by all processes using it; run the measured workload
exclusively and bind CPU, memory, and GPU work to that module.

```python
import torch
from neuro_co.aet.energy import GraceModuleTracker, checked_module_energy

torch.cuda.synchronize(device)
with GraceModuleTracker("inference", socket=0) as tracker:
    run_workload()
    torch.cuda.synchronize(device)
    tracker.n_items = batch_size

record = checked_module_energy(tracker, socket=0, instances=batch_size)
```

The record retains monotonic timestamps and power samples. Validation checks
sensor identity, work count, sampling gaps, and the integral recomputed from
the trace. The default maximum gap is one second; a failed sensor read leaves
energy unavailable instead of estimating it. `allow_zero=True` permits an
observed zero integral for short initialization or restore windows only.

`energy_j` and `energy_module_j` describe the same undecomposed module integral.
The module includes Grace and Hopper, so do not add CPU rails or NVML energy to
it. It is neither whole-system AC energy nor a calibrated measurement. Records
use `aet-grace-module-energy/v1`; they are a separate API from `EnergyReading`
and are not accepted by `neuroco-aet-report`.

## Build a report

```bash
uv run --no-sync neuroco-aet-report outputs/ --out aet_report --deltas 0.5 1 2 5
```

The command reads `energy_train.json`, `energy_eval.json`, and
`energy_baseline.json` below the supplied root. It writes `aet_table.csv`,
`aet_summary.md`, and figures. Keep incompatible hardware configurations and
measurement scopes in separate input bundles.

Reports without an input manifest are exploratory. To validate a complete
measurement bundle, pass `--expected-manifest report-input-manifest.json`.
The manifest checksums the training, inference, baseline, provenance,
validation, instance, calibration, and preflight evidence. The report validates
these inputs before assigning confirmatory eligibility. A finite computed
threshold alone does not establish an energy advantage.

The `neuro_co.aet.experiments` modules contain the stricter experiment runners.
Study-specific recipes and platform launchers are maintained separately.
Supply the inputs required by the chosen runner. Wall-meter requirements remain
distinct from software-only counter diagnostics.
