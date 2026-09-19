"""AET (Amortized Efficiency Threshold) analysis.

Library functions for computing the break-even deployment volume
above which a neural solver becomes more energy-efficient than a
metaheuristic baseline. See `aet.py` for the math and `plots.py`
for the canonical figures.
"""

from neuro_co.aet.analysis.aet import (
    EPS,
    TrainAggregate,
    aggregate_training,
    build_aet_table,
    load_records,
)
from neuro_co.aet.analysis.asymptotic import (
    asymptotic_ratio,
    crossover_n,
    energy_curves,
)
from neuro_co.aet.analysis.cli import write_aet_report

__all__ = [
    "EPS",
    "TrainAggregate",
    "aggregate_training",
    "asymptotic_ratio",
    "build_aet_table",
    "crossover_n",
    "energy_curves",
    "load_records",
    "write_aet_report",
]
