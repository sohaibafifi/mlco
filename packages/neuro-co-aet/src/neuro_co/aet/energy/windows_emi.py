"""Windows CPU package energy via CodeCarbon's Energy Meter Interface.

CodeCarbon 3.3.1 exposes the Windows 11 EMI counters through
``codecarbon.core.windows_emi.WindowsEMI``.  This adapter gives that interface
the same small ``start()`` / ``stop() -> Wh`` contract as the other AET
hardware-counter samplers.

Only CPU package channels are enabled.  DRAM energy is deliberately excluded
so that an AET record never presents CodeCarbon's RAM model as a measured
hardware-counter domain.
"""

from __future__ import annotations

import math
import re
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from codecarbon.core.units import Time
from codecarbon.core.windows_emi import WindowsEMI, is_emi_available

WINDOWS_EMI_CPU_PACKAGE = "windows_emi_cpu_package_counter"
SUPPORTED_CODECARBON_VERSION = "3.3.1"
_CPU_ENERGY_KEY = re.compile(r"^Processor Energy Delta_\d+\(kWh\)$")


class WindowsEmiSampler:
    """Measure native Windows CPU package energy with CodeCarbon EMI."""

    def __init__(self) -> None:
        if not sys.platform.startswith("win"):
            raise RuntimeError(f"Windows EMI requires native Windows (got {sys.platform})")
        if not is_emi_available():
            raise RuntimeError("Windows Energy Meter Interface counters are unavailable")
        try:
            self._meter: Any = WindowsEMI(emi_include_dram=False)
        except Exception as exc:
            raise RuntimeError("Windows Energy Meter Interface initialization failed") from exc
        selected_channels = [
            channel_names[index]
            for _device_path, channel_names, selected_indices in self._meter._devices
            for index in selected_indices
        ]
        if not selected_channels or any(
            "pkg" not in str(channel_name).lower() for channel_name in selected_channels
        ):
            raise RuntimeError("Windows EMI exposes no unambiguous CPU package-only counter")
        self._channel_count = len(selected_channels)
        try:
            self._codecarbon_version = version("codecarbon")
        except PackageNotFoundError:
            self._codecarbon_version = "unknown"
        if self._codecarbon_version != SUPPORTED_CODECARBON_VERSION:
            raise RuntimeError(
                "Windows EMI requires the validated CodeCarbon version "
                f"{SUPPORTED_CODECARBON_VERSION} (got {self._codecarbon_version})"
            )
        self._started = False
        self._stopped = False
        self._start_time = 0.0
        self._last_measured_channel_count: int | None = None

    @property
    def measurement_metadata(self) -> dict[str, Any]:
        """Return stable metadata without device paths or modeled RAM energy."""

        return {
            "measurement_mode": WINDOWS_EMI_CPU_PACKAGE,
            "source": "codecarbon",
            "codecarbon_version": self._codecarbon_version,
            "includes_dram": False,
            "initial_selected_channel_count": self._channel_count,
            "last_measured_channel_count": self._last_measured_channel_count,
            "attribution_scope": "machine_wide_cpu_package",
            "exclusive_host_required": True,
        }

    @property
    def available(self) -> bool:
        """Report successful initialization of readable native EMI counters."""

        return True

    def start(self) -> None:
        """Take the initial native EMI counter snapshot."""

        if self._started:
            raise RuntimeError("Windows EMI sampler has already been started")
        if self._stopped:
            raise RuntimeError("Windows EMI sampler has already been stopped")
        try:
            self._meter.start()
        except Exception as exc:
            raise RuntimeError("Windows Energy Meter Interface start failed") from exc
        self._start_time = time.perf_counter()
        self._started = True

    def stop(self) -> float:
        """Take the final snapshot and return CPU package energy in Wh."""

        if not self._started:
            raise RuntimeError("Windows EMI sampler has not been started")
        if self._stopped:
            raise RuntimeError("Windows EMI sampler has already been stopped")
        self._stopped = True
        duration_s = max(0.0, time.perf_counter() - self._start_time)
        try:
            details = self._meter.get_cpu_details(Time.from_seconds(duration_s))
        except Exception as exc:
            raise RuntimeError("Windows Energy Meter Interface stop failed") from exc
        if not isinstance(details, dict):
            raise TypeError("Windows EMI CPU details must be a dictionary")

        energy_values_kwh = [
            value
            for name, value in details.items()
            if isinstance(name, str) and _CPU_ENERGY_KEY.fullmatch(name)
        ]
        if not energy_values_kwh:
            raise RuntimeError("Windows EMI returned no CPU package energy counters")
        self._last_measured_channel_count = len(energy_values_kwh)
        energy_kwh = 0.0
        for value in energy_values_kwh:
            if isinstance(value, bool):
                raise TypeError("Windows EMI CPU energy must be a real number, not bool")
            try:
                counter_kwh = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("Windows EMI CPU energy must be a real number") from exc
            if not math.isfinite(counter_kwh) or counter_kwh < 0.0:
                raise ValueError("Windows EMI CPU energy must be finite and non-negative")
            energy_kwh += counter_kwh
        return energy_kwh * 1000.0
