"""Energy-tracking value types + Protocol.

`EnergyReading` is the canonical SI dataclass returned by every
`EnergyContext` (joules, kg-CO2eq, watts, items/sec). `EnergyContext`
itself is a `typing.Protocol` so non-aet code can type-check against
"any context manager yielding a reading" without importing this
package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, ClassVar, Protocol, runtime_checkable

ENERGY_SCHEMA_VERSION = "1.0"

ENERGY_READING_UNITS: dict[str, str] = {
    "duration_s": "s",
    "energy_j": "J",
    "energy_gpu_j": "J",
    "energy_cpu_j": "J",
    "energy_dram_j": "J",
    "co2_operational_kg": "kgCO2eq",
    "co2_embodied_kg": "kgCO2eq",
    "co2_total_kg": "kgCO2eq",
    "avg_power_w": "W",
    "items_processed": "item",
    "throughput_items_per_s": "item/s",
}

_LEGACY_ALIAS_KEYS = frozenset(
    {
        "energy_wh",
        "energy_gpu_wh",
        "energy_cpu_wh",
        "energy_dram_wh",
        "co2_g_operational",
        "co2_g_embodied",
        "co2_g_total",
        "throughput",
        "hardware_id",
    }
)


@dataclass
class EnergyReading:
    """Energy and CO2 measured by an `EnergyContext` span.

    All fields are SI / standardized: joules for energy, kgCO2-eq for
    emissions, watts for power. Throughput is items per second.
    """

    duration_s: float
    energy_j: float
    energy_gpu_j: float = 0.0
    energy_cpu_j: float = 0.0
    energy_dram_j: float = 0.0
    co2_operational_kg: float = 0.0
    co2_embodied_kg: float = 0.0
    avg_power_w: float = 0.0
    items_processed: int = 0
    backend: str = "unknown"
    energy_domains: tuple[str, ...] = ()
    measurement_scope: str = "unspecified"
    hardware: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    _CANONICAL_OUTPUT_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "units",
            "duration_s",
            "energy_j",
            "energy_gpu_j",
            "energy_cpu_j",
            "energy_dram_j",
            "co2_operational_kg",
            "co2_embodied_kg",
            "co2_total_kg",
            "avg_power_w",
            "items_processed",
            "throughput_items_per_s",
            "backend",
            "energy_domains",
            "measurement_scope",
            "hardware",
            "extra",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "energy_j",
            "energy_gpu_j",
            "energy_cpu_j",
            "energy_dram_j",
            "co2_operational_kg",
            "co2_embodied_kg",
            "avg_power_w",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number, not bool")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a real number") from exc
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, number)

        if isinstance(self.items_processed, bool) or not isinstance(self.items_processed, int):
            raise TypeError("items_processed must be an integer")
        if self.items_processed < 0:
            raise ValueError("items_processed must be non-negative")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")
        if not isinstance(self.measurement_scope, str) or not self.measurement_scope.strip():
            raise ValueError("measurement_scope must be a non-empty string")
        if isinstance(self.energy_domains, str):
            raise TypeError("energy_domains must be a sequence of strings")
        try:
            domains = tuple(self.energy_domains)
        except TypeError as exc:
            raise TypeError("energy_domains must be a sequence of strings") from exc
        if any(not isinstance(domain, str) or not domain.strip() for domain in domains):
            raise ValueError("energy_domains must contain non-empty strings")
        if len(set(domains)) != len(domains):
            raise ValueError("energy_domains must not contain duplicates")
        self.energy_domains = domains
        if not isinstance(self.hardware, dict):
            raise TypeError("hardware must be a dictionary")
        if not isinstance(self.extra, dict):
            raise TypeError("extra must be a dictionary")
        if any(not isinstance(key, str) for key in self.hardware):
            raise TypeError("hardware keys must be strings")
        if any(not isinstance(key, str) for key in self.extra):
            raise TypeError("extra keys must be strings")

        reserved = self._CANONICAL_OUTPUT_KEYS | _LEGACY_ALIAS_KEYS
        collisions = sorted(reserved.intersection(self.extra))
        if collisions:
            names = ", ".join(collisions)
            raise ValueError(f"extra contains reserved output key(s): {names}")

        component_energy_j = self.energy_gpu_j + self.energy_cpu_j + self.energy_dram_j
        tolerance_j = max(1e-9, self.energy_j * 1e-9)
        if component_energy_j > self.energy_j + tolerance_j:
            raise ValueError("component energy cannot exceed total energy_j")

        # Detach serialized metadata from dictionaries owned by the caller.
        self.hardware = dict(self.hardware)
        self.extra = dict(self.extra)

    @property
    def co2_total_kg(self) -> float:
        return self.co2_operational_kg + self.co2_embodied_kg

    @property
    def throughput(self) -> float:
        """Compatibility property for the canonical throughput value."""
        return self.throughput_items_per_s

    @property
    def throughput_items_per_s(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.items_processed / self.duration_s

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, versioned SI record.

        Extra metadata remains nested so it cannot silently replace a
        measurement or its declared unit.
        """
        return {
            "schema_version": ENERGY_SCHEMA_VERSION,
            "units": dict(ENERGY_READING_UNITS),
            "duration_s": self.duration_s,
            "energy_j": self.energy_j,
            "energy_gpu_j": self.energy_gpu_j,
            "energy_cpu_j": self.energy_cpu_j,
            "energy_dram_j": self.energy_dram_j,
            "co2_operational_kg": self.co2_operational_kg,
            "co2_embodied_kg": self.co2_embodied_kg,
            "co2_total_kg": self.co2_total_kg,
            "avg_power_w": self.avg_power_w,
            "items_processed": self.items_processed,
            "throughput_items_per_s": self.throughput_items_per_s,
            "backend": self.backend,
            "energy_domains": list(self.energy_domains),
            "measurement_scope": self.measurement_scope,
            "hardware": dict(self.hardware),
            "extra": dict(self.extra),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return an explicit transitional record with Wh and gCO2 aliases.

        New writers should use :meth:`to_dict`. This adapter exists only for
        consumers that have not yet migrated from the original mixed-unit,
        flat JSON format.
        """
        record = self.to_dict()
        record.update(
            {
                "energy_wh": self.energy_j / 3600.0,
                "energy_gpu_wh": self.energy_gpu_j / 3600.0,
                "energy_cpu_wh": self.energy_cpu_j / 3600.0,
                "energy_dram_wh": self.energy_dram_j / 3600.0,
                "co2_g_operational": self.co2_operational_kg * 1000.0,
                "co2_g_embodied": self.co2_embodied_kg * 1000.0,
                "co2_g_total": self.co2_total_kg * 1000.0,
                "throughput": self.throughput_items_per_s,
                "hardware_id": self.hardware.get("id", "unknown"),
            }
        )
        record.update(self.extra)
        return record


@runtime_checkable
class EnergyContext(Protocol):
    """A context manager that yields an `EnergyReading` on exit.

    Conventional usage::

        with energy_tracker(items=batch_size) as ctx:
            run_one_epoch()
        reading = ctx.reading
    """

    reading: EnergyReading | None

    def __enter__(self) -> EnergyContext: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...
