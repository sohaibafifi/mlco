"""One explicitly selected GH200 module, integrated from measured hwmon power.

The module includes Grace and Hopper and is not whole-system AC energy. Its
energy is never decomposed into fabricated CPU/GPU components or added to NVML.
Allocation exclusivity and CPU/memory/GPU binding belong to the launcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from neuro_co.aet.energy.grace import GraceProcessSampler, discover_grace_sensors


@dataclass
class _Reading:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


class GraceModuleTracker:
    """Measure one Grace + Hopper module and retain its complete power trace.

    Requires Linux hwmon sensors. Synchronize CUDA work before entering and
    before leaving the context, and set ``n_items`` to the completed work count.
    Validate the resulting record with :func:`checked_module_energy`.
    """

    def __init__(
        self,
        label: str,
        *,
        socket: int,
        poll_interval_s: float = 0.1,
        hwmon_root: str | Path = "/sys/class/hwmon",
    ) -> None:
        if type(socket) is not int or socket < 0:
            raise ValueError("grace_socket must be an explicit nonnegative integer")
        selected = [
            sensor
            for sensor in discover_grace_sensors(hwmon_root)
            if sensor.domain == "module" and sensor.socket == socket
        ]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one Module Power Socket {socket} sensor")
        self.label = label
        self.sensor = selected[0]
        self.sampler = GraceProcessSampler(selected, poll_interval_s=poll_interval_s)
        self.n_items = 0
        self.reading: _Reading | None = None

    def __enter__(self) -> GraceModuleTracker:
        self.sampler.start()
        return self

    def __exit__(self, exception_type: Any, _exception: Any, _traceback: Any) -> bool:
        sampling_error = None
        try:
            raw = self.sampler.stop()
        except Exception as exc:
            sampling_error = f"{type(exc).__name__}: {exc}"
            raw = {"valid": False, "error": sampling_error, "sensor_integrals": []}
        integrals = raw.get("sensor_integrals", [])
        integral = integrals[0] if len(integrals) == 1 else {}
        energy = integral.get("energy_j")
        self.reading = _Reading(
            {
                "schema_version": "aet-grace-module-energy/v1",
                "backend": "grace_module",
                "measurement_scope": "gh200_module",
                "measurement_mode": "grace_hwmon_power_integration",
                "energy_domains": ["module"],
                "energy_j": energy,
                "energy_module_j": energy,
                "duration_s": integral.get("duration_s"),
                "items_processed": self.n_items,
                "calibrated": False,
                "whole_system_energy": False,
                "hardware": {
                    "module_measurement": self.sensor.metadata(),
                    "attribution_scope": "physical_gh200_module_all_processes",
                    "exclusive_scope_required": True,
                },
                "raw_measurement": raw,
                "extra": {
                    "label": self.label,
                    "tdp_fallback": False,
                    "run_status": "failed" if exception_type or sampling_error else "completed",
                    "sampling_error": sampling_error,
                },
            }
        )
        return False


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def checked_module_energy(
    tracker: Any,
    *,
    socket: int,
    instances: int,
    minimum_duration_s: float = 0.0,
    max_sample_gap_s: float = 1.0,
    allow_zero: bool = False,
) -> dict[str, Any]:
    """Validate scope, trace, integral and work count without inventing components.

    Startup/restore may retain a genuinely observed zero integral. Workload
    windows require a positive integral. A failed sensor read is never zero.
    """
    if type(socket) is not int or socket < 0:
        raise ValueError("grace_socket must be an explicit nonnegative integer")
    if tracker.reading is None:
        raise ValueError("Grace module tracker returned no reading")
    payload = tracker.reading.to_dict()
    if (
        payload.get("backend") != "grace_module"
        or payload.get("measurement_scope") != "gh200_module"
        or payload.get("measurement_mode") != "grace_hwmon_power_integration"
        or payload.get("energy_domains") != ["module"]
        or any(key in payload for key in ("energy_cpu_j", "energy_gpu_j", "energy_dram_j"))
    ):
        raise ValueError("energy reading is not an undecomposed Grace module integral")
    raw = payload.get("raw_measurement", {})
    integrals = raw.get("sensor_integrals", [])
    if (
        raw.get("valid") is not True
        or len(integrals) != 1
        or raw.get("measurement_mode") != "grace_hwmon_power_integration"
        or raw.get("integration_rule") != "trapezoidal"
        or raw.get("timestamp_clock") != "monotonic"
        or raw.get("nested_domains_additive") is not False
        or raw.get("nvml_energy_included") is not False
    ):
        raise ValueError("Grace module sampling is invalid or has unexpected domains")
    integral = integrals[0]
    expected = {
        "socket": socket,
        "domain": "module",
        "sensor_id": f"module:socket{socket}",
        "label": f"Module Power Socket {socket}",
        "measurement_scope": "gh200_module",
        "raw_power_unit": "microwatt",
        "exclusive_scope_required": True,
    }
    metadata = payload.get("hardware", {}).get("module_measurement", {})
    if (
        payload.get("hardware", {}).get("attribution_scope")
        != "physical_gh200_module_all_processes"
        or payload.get("hardware", {}).get("exclusive_scope_required") is not True
        or payload.get("whole_system_energy") is not False
        or payload.get("calibrated") is not False
        or payload.get("extra", {}).get("tdp_fallback") is not False
    ):
        raise ValueError("Grace module attribution or measurement provenance is invalid")
    if any(
        integral.get(key) != value or metadata.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Grace module sensor identity or scope changed")
    if (
        integral.get("valid") is not True
        or integral.get("errors") != []
        or type(integral.get("failed_sample_count")) is not int
        or integral["failed_sample_count"] != 0
    ):
        raise ValueError("Grace module trace contains failed samples")
    averaging_interval = _finite(
        integral.get("averaging_interval_s"), "averaging_interval_s", positive=True
    )
    if averaging_interval != metadata.get("averaging_interval_s"):
        raise ValueError("Grace module averaging interval metadata changed")
    duration = _finite(payload.get("duration_s"), "duration_s", positive=True)
    if duration < _finite(minimum_duration_s, "minimum_duration_s"):
        raise ValueError("Grace module window is shorter than minimum_block_duration_s")
    energy = _finite(payload.get("energy_j"), "energy_j", positive=not allow_zero)
    if energy != payload.get("energy_module_j") or energy != integral.get("energy_j"):
        raise ValueError("module integral disagrees with reported total energy")
    if (
        type(instances) is not int
        or instances <= 0
        or type(payload.get("items_processed")) is not int
    ):
        raise ValueError("Grace module work count must be a positive integer")
    if payload["items_processed"] != instances:
        raise ValueError("energy window work count disagrees with completed instances")
    samples = integral.get("samples", [])
    if (
        len(samples) < 2
        or type(integral.get("sample_count")) is not int
        or integral["sample_count"] != len(samples)
    ):
        raise ValueError("Grace module trace needs at least two counted samples")
    for sample in samples:
        _finite(sample.get("monotonic_s"), "sample monotonic_s")
        _finite(sample.get("power_w"), "sample power_w")
    intervals = [b["monotonic_s"] - a["monotonic_s"] for a, b in pairwise(samples)]
    gap_limit = _finite(max_sample_gap_s, "max_sample_gap_s", positive=True)
    if min(intervals) <= 0 or max(intervals) > gap_limit:
        raise ValueError("Grace module trace timestamps or sampling gaps are invalid")
    integrated = sum(
        (b["monotonic_s"] - a["monotonic_s"]) * (a["power_w"] + b["power_w"]) / 2
        for a, b in pairwise(samples)
    )
    if (
        not math.isclose(integrated, energy, rel_tol=1e-10, abs_tol=1e-8)
        or not math.isclose(sum(intervals), duration, rel_tol=1e-10, abs_tol=1e-8)
        or integral.get("duration_s") != duration
        or integral.get("start_monotonic_s") != samples[0]["monotonic_s"]
        or integral.get("end_monotonic_s") != samples[-1]["monotonic_s"]
    ):
        raise ValueError("Grace module energy or duration disagrees with its retained trace")
    return payload
