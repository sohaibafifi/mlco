"""Energy and CO2 tracking primitives for AET."""

from neuro_co.aet.energy.embodied import EmbodiedCarbon, amortize_embodied
from neuro_co.aet.energy.hardware import autodetect_hardware_id
from neuro_co.aet.energy.nvml import NvmlSampler
from neuro_co.aet.energy.rapl import RaplSampler, tdp_estimate_wh
from neuro_co.aet.energy.tracker import EnergyTracker, measure
from neuro_co.aet.energy.wall_meter import WallMeterResult, WallMeterSampler
from neuro_co.aet.energy.windows_emi import WindowsEmiSampler

__all__ = [
    "EmbodiedCarbon",
    "EnergyTracker",
    "NvmlSampler",
    "RaplSampler",
    "WallMeterResult",
    "WallMeterSampler",
    "WindowsEmiSampler",
    "amortize_embodied",
    "autodetect_hardware_id",
    "measure",
    "tdp_estimate_wh",
]
