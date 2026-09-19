"""Energy + embodied-carbon tracking, AET analysis, and run provenance.

`AETCallback` (Lightning per-epoch energy snapshots) is lazy-loaded
via `__getattr__` so `import neuro_co.aet` stays torch-free for
non-Lightning consumers.
"""

from neuro_co.aet.energy.embodied import amortize_embodied
from neuro_co.aet.energy.tracker import EnergyTracker
from neuro_co.aet.energy.types import EnergyContext, EnergyReading
from neuro_co.aet.energy.wall_meter import WallMeterResult, WallMeterSampler
from neuro_co.aet.provenance import RunMetadata, record_run

__all__ = [
    "AETCallback",
    "EnergyContext",
    "EnergyReading",
    "EnergyTracker",
    "RunMetadata",
    "WallMeterResult",
    "WallMeterSampler",
    "amortize_embodied",
    "record_run",
]

__version__ = "0.2.0"


def __getattr__(name: str):  # pragma: no cover - thin lazy-import wrapper
    if name == "AETCallback":
        from neuro_co.aet.lightning import AETCallback

        return AETCallback
    raise AttributeError(f"module 'neuro_co.aet' has no attribute {name!r}")
