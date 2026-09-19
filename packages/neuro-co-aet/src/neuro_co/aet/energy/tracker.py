"""Unified energy tracker.

Backend chain (default order):
  1. codecarbon: gives both Wh and gCO2eq via regional grid intensity.
  2. pynvml + CPU package counters: NVML with Linux RAPL or Windows EMI.
  3. tdp_fallback: TDP times wall time (last resort).

PUE (datacenter Power Usage Effectiveness) is applied to operational
energy. Embodied carbon is reported separately and added to total CO2
when `report_embodied=True`.

Results are exposed via `neuro_co.aet.energy.types.EnergyReading`;
downstream code can type-check against the matching
`EnergyContext` Protocol if it wants to remain aet-agnostic.

The ``wall_meter`` backend is separate from that fallback chain. It accepts an
injected physical-meter sampler and is always raw, strict, whole-system AC.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from neuro_co.aet.energy.embodied import amortize_embodied
from neuro_co.aet.energy.nvml import NvmlDeviceSelectionError, NvmlSampler
from neuro_co.aet.energy.rapl import RaplSampler, tdp_estimate_wh
from neuro_co.aet.energy.types import EnergyReading
from neuro_co.aet.energy.wall_meter import WallMeterResult, WallMeterSampler
from neuro_co.aet.energy.windows_emi import WindowsEmiSampler

log = logging.getLogger(__name__)

DEFAULT_PUE: float = 1.4
DEFAULT_GRID_INTENSITY_G_PER_KWH: float = 475.0  # global average
WH_TO_J: float = 3600.0
KWH_TO_J: float = 3_600_000.0


@dataclass(frozen=True)
class _BackendMeasurement:
    """Normalized backend result at the tracker boundary."""

    energy_j: float
    co2_operational_kg: float
    energy_gpu_j: float = 0.0
    energy_cpu_j: float = 0.0
    energy_dram_j: float = 0.0
    energy_domains: tuple[str, ...] = ()
    measurement_scope: str = "unspecified"

    def __post_init__(self) -> None:
        for name in (
            "energy_j",
            "co2_operational_kg",
            "energy_gpu_j",
            "energy_cpu_j",
            "energy_dram_j",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=f"backend {name}"),
            )
        component_energy_j = self.energy_gpu_j + self.energy_cpu_j + self.energy_dram_j
        tolerance_j = max(1e-9, self.energy_j * 1e-9)
        if component_energy_j > self.energy_j + tolerance_j:
            raise ValueError("backend component energy cannot exceed total energy_j")
        if any(not isinstance(domain, str) or not domain for domain in self.energy_domains):
            raise ValueError("backend energy_domains must contain non-empty strings")
        if len(set(self.energy_domains)) != len(self.energy_domains):
            raise ValueError("backend energy_domains must not contain duplicates")
        if not isinstance(self.measurement_scope, str) or not self.measurement_scope:
            raise ValueError("backend measurement_scope must be non-empty")


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


class EnergyTracker:
    """Context manager measuring energy and CO2 for a code block.

    Example::

        with EnergyTracker(
            "train",
            backend="codecarbon",
            hardware_id="nvidia-a100",
            items=200_000,
        ) as t:
            do_work()
        reading = t.reading   # EnergyReading
    """

    def __init__(
        self,
        label: str = "default",
        *,
        backend: str = "codecarbon",
        pue: float = DEFAULT_PUE,
        hardware_id: str | None = None,
        report_embodied: bool = True,
        lifetime_s: float | None = None,
        grid_intensity_g_per_kwh: float = DEFAULT_GRID_INTENSITY_G_PER_KWH,
        country_iso_code: str | None = None,
        output_dir: str | None = None,
        items: int = 0,
        allow_fallback: bool = True,
        required_domains: Collection[str] | None = None,
        gpu_indices: Collection[int] | None = None,
        gpu_uuids: Collection[str] | None = None,
        wall_meter_sampler: WallMeterSampler | None = None,
    ) -> None:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("backend must be a non-empty string")
        pue_value = _finite_nonnegative(pue, name="pue")
        if pue_value < 1.0:
            raise ValueError("pue must be greater than or equal to 1.0")
        grid_intensity = _finite_nonnegative(
            grid_intensity_g_per_kwh,
            name="grid_intensity_g_per_kwh",
        )
        if lifetime_s is not None:
            lifetime_s = _finite_nonnegative(lifetime_s, name="lifetime_s")
            if lifetime_s == 0.0:
                raise ValueError("lifetime_s must be greater than zero")
        if isinstance(items, bool) or not isinstance(items, int):
            raise TypeError("items must be an integer")
        if items < 0:
            raise ValueError("items must be non-negative")
        if not isinstance(allow_fallback, bool):
            raise TypeError("allow_fallback must be a boolean")
        if isinstance(required_domains, str):
            raise TypeError("required_domains must be a collection of domain names")
        try:
            required_domain_set = frozenset(required_domains or ())
        except TypeError as exc:
            raise TypeError("required_domains must be a collection of domain names") from exc
        if any(not isinstance(domain, str) for domain in required_domain_set):
            raise TypeError("required_domains must contain only strings")
        supported_required_domains = {"gpu", "cpu"}
        unknown_domains = sorted(required_domain_set - supported_required_domains)
        if unknown_domains:
            names = ", ".join(unknown_domains)
            raise ValueError(f"unsupported required energy domain(s): {names}")
        if required_domain_set and backend != "hwcounters":
            raise ValueError("required_domains is only supported with backend='hwcounters'")
        if gpu_indices is not None and gpu_uuids is not None:
            raise ValueError("gpu_indices and gpu_uuids are mutually exclusive")
        if isinstance(gpu_indices, (str, bytes)):
            raise TypeError("gpu_indices must be a collection of integers")
        if isinstance(gpu_uuids, (str, bytes)):
            raise TypeError("gpu_uuids must be a collection of strings")
        try:
            selected_indices = None if gpu_indices is None else tuple(gpu_indices)
        except TypeError as exc:
            raise TypeError("gpu_indices must be a collection of integers") from exc
        try:
            selected_uuids = None if gpu_uuids is None else tuple(gpu_uuids)
        except TypeError as exc:
            raise TypeError("gpu_uuids must be a collection of strings") from exc
        if selected_indices is not None:
            if not selected_indices:
                raise ValueError("gpu_indices must not be empty")
            if any(
                isinstance(index, bool) or not isinstance(index, int) for index in selected_indices
            ):
                raise TypeError("gpu_indices must contain only integers")
            if any(index < 0 for index in selected_indices):
                raise ValueError("gpu_indices must contain only non-negative integers")
            if len(set(selected_indices)) != len(selected_indices):
                raise ValueError("gpu_indices must not contain duplicates")
        if selected_uuids is not None:
            if not selected_uuids:
                raise ValueError("gpu_uuids must not be empty")
            if any(not isinstance(uuid, str) for uuid in selected_uuids):
                raise TypeError("gpu_uuids must contain only strings")
            selected_uuids = tuple(uuid.strip() for uuid in selected_uuids)
            if any(not uuid for uuid in selected_uuids):
                raise ValueError("gpu_uuids must contain only non-empty strings")
            if len({uuid.casefold() for uuid in selected_uuids}) != len(selected_uuids):
                raise ValueError("gpu_uuids must not contain duplicates")
        if (selected_indices is not None or selected_uuids is not None) and backend != "hwcounters":
            raise ValueError("GPU selection is only supported with backend='hwcounters'")
        if wall_meter_sampler is not None and backend != "wall_meter":
            raise ValueError("wall_meter_sampler is only supported with backend='wall_meter'")
        if backend == "wall_meter":
            if wall_meter_sampler is None:
                raise ValueError("backend='wall_meter' requires wall_meter_sampler")
            if not callable(getattr(wall_meter_sampler, "start", None)) or not callable(
                getattr(wall_meter_sampler, "stop", None)
            ):
                raise TypeError("wall_meter_sampler must provide callable start() and stop()")
            if pue_value != 1.0:
                raise ValueError("backend='wall_meter' requires pue=1.0")
            if report_embodied is not False:
                raise ValueError("backend='wall_meter' requires report_embodied=False")
            if allow_fallback is not False:
                raise ValueError("backend='wall_meter' requires allow_fallback=False")

        self.label = label
        self.backend = backend
        self.pue = pue_value
        self.hardware_id = hardware_id
        self.report_embodied = report_embodied
        self.lifetime_s = lifetime_s
        self.grid_intensity = grid_intensity
        self.country_iso_code = country_iso_code
        self.output_dir = output_dir
        self.n_items: int = items
        self.allow_fallback = allow_fallback
        self.required_domains = required_domain_set
        self.gpu_indices = selected_indices
        self.gpu_uuids = selected_uuids
        self._wall_meter: WallMeterSampler | None = wall_meter_sampler
        self._wall_meter_result: WallMeterResult | None = None
        self._t0: float = 0.0
        self._codecarbon: Any = None
        self._nvml: NvmlSampler | None = None
        self._rapl: RaplSampler | None = None
        self._windows_emi: WindowsEmiSampler | None = None
        self._backend_used: str = "none"
        self._tdp_fallback: bool = False
        self.reading: EnergyReading | None = None

    def __enter__(self) -> EnergyTracker:
        self.reading = None
        self._codecarbon = None
        self._nvml = None
        self._rapl = None
        self._windows_emi = None
        self._backend_used = "none"
        self._tdp_fallback = False
        self._wall_meter_result = None
        self._t0 = time.perf_counter()
        self._start_backend()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        provisional_duration = max(0.0, time.perf_counter() - self._t0)
        measurement = self._stop_backend(provisional_duration)
        duration = max(0.0, time.perf_counter() - self._t0)
        co2_emb_kg = 0.0
        if self.report_embodied and self.hardware_id is not None:
            t_lifetime = self.lifetime_s if self.lifetime_s is not None else 5 * 365 * 24 * 3600
            co2_emb_kg = amortize_embodied(
                hardware_id=self.hardware_id,
                t_used_s=duration,
                t_lifetime_s=t_lifetime,
            )

        measurement_duration_s = (
            self._wall_meter_result.duration_s if self._wall_meter_result is not None else duration
        )
        avg_power_w = (
            measurement.energy_j / measurement_duration_s if measurement_duration_s > 0 else 0.0
        )
        gpu_devices = list(getattr(self._nvml, "device_metadata", []))
        hardware_metadata: dict[str, Any] = {
            "id": self.hardware_id or "unknown",
            "pue": self.pue,
            "grid_intensity_g_per_kwh": self.grid_intensity,
            "required_energy_domains": sorted(self.required_domains),
            "gpu_devices": gpu_devices,
        }
        if self._backend_used == "hwcounters":
            hardware_metadata.update(
                {
                    "attribution_scope": "machine_wide_component_counters",
                    "exclusive_host_required": True,
                }
            )
        if self._windows_emi is not None:
            hardware_metadata["cpu_measurement"] = dict(self._windows_emi.measurement_metadata)
        extra: dict[str, Any] = {
            "label": self.label,
            "tdp_fallback": self._tdp_fallback,
            "country_iso_code": self.country_iso_code,
            "pue_applied_to_operational": self._backend_used != "wall_meter",
            "allow_fallback": self.allow_fallback,
            "run_status": "failed" if exc_type is not None else "completed",
            "failure_type": exc_type.__name__ if exc_type is not None else None,
        }
        if self._wall_meter_result is not None:
            extra["wall_meter"] = self._wall_meter_result.to_metadata_dict()
            extra["tracker_context_duration_s"] = duration
        self.reading = EnergyReading(
            duration_s=measurement_duration_s,
            energy_j=measurement.energy_j,
            energy_gpu_j=measurement.energy_gpu_j,
            energy_cpu_j=measurement.energy_cpu_j,
            energy_dram_j=measurement.energy_dram_j,
            co2_operational_kg=measurement.co2_operational_kg,
            co2_embodied_kg=co2_emb_kg,
            avg_power_w=avg_power_w,
            items_processed=self.n_items,
            backend=self._backend_used,
            energy_domains=measurement.energy_domains,
            measurement_scope=measurement.measurement_scope,
            hardware=hardware_metadata,
            extra=extra,
        )

    # --- backend dispatch ---
    def _start_backend(self) -> None:
        for name in self._resolve_chain(self.backend, allow_fallback=self.allow_fallback):
            if name == "wall_meter":
                self._start_wall_meter()
                self._backend_used = "wall_meter"
                return
            if name == "codecarbon" and self._try_start_codecarbon():
                self._backend_used = "codecarbon"
                return
            if name == "hwcounters" and self._try_start_hwcounters():
                self._backend_used = "hwcounters"
                return
            if name == "tdp":
                self._backend_used = "tdp"
                return
        self._backend_used = "none"
        if not self.allow_fallback:
            raise RuntimeError(f"requested energy backend {self.backend!r} is unavailable")

    def _stop_backend(self, duration: float) -> _BackendMeasurement:
        """Stop the selected backend and normalize its result to SI units."""
        backend = self._backend_used
        if backend == "wall_meter":
            if self._wall_meter is None:
                raise RuntimeError("wall_meter sampler is unavailable while stopping")
            try:
                result = self._wall_meter.stop()
            except Exception as exc:
                raise RuntimeError("wall_meter failed while stopping") from exc
            if not isinstance(result, WallMeterResult):
                raise TypeError("wall_meter stop() must return WallMeterResult")
            self._wall_meter_result = result
            co2_kg = result.energy_j / KWH_TO_J * self.grid_intensity / 1000.0
            return _BackendMeasurement(
                energy_j=result.energy_j,
                co2_operational_kg=co2_kg,
                energy_domains=("whole_system_ac",),
                measurement_scope="whole_system_ac",
            )
        if backend == "codecarbon" and self._codecarbon is not None:
            try:
                emissions_kg = self._codecarbon.stop()
                final = getattr(self._codecarbon, "final_emissions_data", None)
                if emissions_kg is None and final is not None:
                    emissions_kg = getattr(final, "emissions", 0.0)
                co2_it_kg = _finite_nonnegative(
                    emissions_kg if emissions_kg is not None else 0.0,
                    name="codecarbon emissions (kgCO2eq)",
                )
                energy_it_kwh = _finite_nonnegative(
                    getattr(final, "energy_consumed", 0.0) if final is not None else 0.0,
                    name="codecarbon energy_consumed (kWh)",
                )
                gpu_it_kwh = _finite_nonnegative(
                    getattr(final, "gpu_energy", 0.0) if final is not None else 0.0,
                    name="codecarbon gpu_energy (kWh)",
                )
                cpu_it_kwh = _finite_nonnegative(
                    getattr(final, "cpu_energy", 0.0) if final is not None else 0.0,
                    name="codecarbon cpu_energy (kWh)",
                )
                dram_it_kwh = _finite_nonnegative(
                    getattr(final, "ram_energy", 0.0) if final is not None else 0.0,
                    name="codecarbon ram_energy (kWh)",
                )
                component_it_kwh = gpu_it_kwh + cpu_it_kwh + dram_it_kwh
                if component_it_kwh > energy_it_kwh:
                    # CodeCarbon component and total counters can differ slightly
                    # across releases. The non-overlapping component sum is the
                    # conservative total in that case.
                    energy_it_kwh = component_it_kwh
                scale_j = KWH_TO_J * self.pue
                domains = tuple(
                    domain
                    for domain, energy_kwh in (
                        ("gpu", gpu_it_kwh),
                        ("cpu", cpu_it_kwh),
                        ("dram", dram_it_kwh),
                    )
                    if energy_kwh > 0.0
                )
                if not domains and energy_it_kwh > 0.0:
                    domains = ("aggregate",)
                return _BackendMeasurement(
                    energy_j=energy_it_kwh * scale_j,
                    energy_gpu_j=gpu_it_kwh * scale_j,
                    energy_cpu_j=cpu_it_kwh * scale_j,
                    energy_dram_j=dram_it_kwh * scale_j,
                    co2_operational_kg=co2_it_kg * self.pue,
                    energy_domains=domains,
                    measurement_scope=self._component_measurement_scope(),
                )
            except (TypeError, ValueError):
                raise
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError("codecarbon failed while stopping") from e
                log.warning("codecarbon stop failed: %s; falling back to TDP", e)
                return self._fill_tdp(duration)
        if backend == "hwcounters":
            cpu_sampler = self._windows_emi if self._windows_emi is not None else self._rapl
            gpu_raw_wh = 0.0
            cpu_raw_wh = 0.0
            stop_errors: list[tuple[str, Exception]] = []
            if self._nvml is not None:
                try:
                    gpu_raw_wh = self._nvml.stop()
                except Exception as exc:
                    stop_errors.append(("NVML", exc))
            if cpu_sampler is not None:
                try:
                    cpu_raw_wh = cpu_sampler.stop()
                except Exception as exc:
                    counter_name = "Windows EMI" if self._windows_emi is not None else "RAPL"
                    stop_errors.append((counter_name, exc))
            if stop_errors:
                failed = ", ".join(name for name, _exc in stop_errors)
                raise RuntimeError(
                    f"hardware counter failed while stopping: {failed}"
                ) from stop_errors[0][1]
            gpu_it_wh = _finite_nonnegative(gpu_raw_wh, name="NVML energy (Wh)")
            rapl_it_wh = _finite_nonnegative(
                cpu_raw_wh,
                name=(
                    "Windows EMI energy (Wh)"
                    if self._windows_emi is not None
                    else "RAPL energy (Wh)"
                ),
            )
            dram_it_wh = _finite_nonnegative(
                self._rapl.last_dram_wh if self._rapl is not None else 0.0,
                name="RAPL DRAM energy (Wh)",
            )
            cpu_it_wh = max(0.0, rapl_it_wh - dram_it_wh)
            total_it_wh = gpu_it_wh + cpu_it_wh + dram_it_wh
            total_operational_wh = total_it_wh * self.pue
            co2_kg = total_operational_wh / 1000.0 * self.grid_intensity / 1000.0
            scale_j = WH_TO_J * self.pue
            domains = tuple(
                domain
                for domain, sampler in (("gpu", self._nvml), ("cpu", cpu_sampler))
                if sampler is not None
            )
            if self._rapl is not None and dram_it_wh > 0.0:
                domains = (*domains, "dram")
            return _BackendMeasurement(
                energy_j=total_operational_wh * WH_TO_J,
                energy_gpu_j=gpu_it_wh * scale_j,
                energy_cpu_j=cpu_it_wh * scale_j,
                energy_dram_j=dram_it_wh * scale_j,
                co2_operational_kg=co2_kg,
                energy_domains=domains,
                measurement_scope=self._component_measurement_scope(),
            )
        if backend == "tdp":
            return self._fill_tdp(duration)
        return _BackendMeasurement(
            energy_j=0.0,
            co2_operational_kg=0.0,
            measurement_scope="none",
        )

    def _try_start_codecarbon(self) -> bool:
        try:
            from codecarbon import EmissionsTracker, OfflineEmissionsTracker
        except Exception as e:
            log.info("codecarbon unavailable (%s); trying hardware counters", e)
            return False
        try:
            save_to_file = self.output_dir is not None
            output_dir = self.output_dir if self.output_dir is not None else "."
            tracker_kwargs: dict[str, Any] = dict(
                project_name=self.label,
                measure_power_secs=1,
                log_level="error",
                save_to_file=save_to_file,
                output_dir=output_dir,
                allow_multiple_runs=True,
                # Normalize CodeCarbon to IT energy. This tracker applies the
                # requested PUE once, to both energy and operational carbon.
                pue=1.0,
            )
            if self.country_iso_code:
                self._codecarbon = OfflineEmissionsTracker(
                    country_iso_code=self.country_iso_code, **tracker_kwargs
                )
            else:
                self._codecarbon = EmissionsTracker(**tracker_kwargs)
            self._codecarbon.start()
            return True
        except Exception as e:
            log.warning("codecarbon start failed: %s", e)
            self._codecarbon = None
            return False

    def _start_wall_meter(self) -> None:
        if self._wall_meter is None:
            raise RuntimeError("wall_meter sampler is unavailable")
        try:
            self._wall_meter.start()
        except Exception as exc:
            raise RuntimeError("wall_meter failed while starting") from exc

    def _try_start_hwcounters(self) -> bool:
        gpu_ok = False
        cpu_ok = False
        try:
            nvml = NvmlSampler(
                gpu_indices=self.gpu_indices,
                gpu_uuids=self.gpu_uuids,
            )
            self._nvml = nvml
        except NvmlDeviceSelectionError:
            self._nvml = None
            raise
        except Exception as e:
            if self.gpu_indices is not None or self.gpu_uuids is not None:
                self._nvml = None
                raise RuntimeError("explicit NVML GPU selection could not be satisfied") from e
            log.info("NVML unavailable: %s", e)
            self._nvml = None
        try:
            if sys.platform.startswith("win"):
                windows_emi = WindowsEmiSampler()
                self._windows_emi = windows_emi
            else:
                rapl = RaplSampler()
                self._rapl = rapl
        except Exception as e:
            counter_name = "Windows EMI" if sys.platform.startswith("win") else "RAPL"
            log.info("%s unavailable: %s", counter_name, e)
            self._rapl = None
            self._windows_emi = None
        cpu_sampler = self._windows_emi if self._windows_emi is not None else self._rapl
        if cpu_sampler is not None:
            try:
                cpu_sampler.start()
                cpu_ok = True
            except Exception as e:
                counter_name = "Windows EMI" if self._windows_emi is not None else "RAPL"
                log.info("%s could not start: %s", counter_name, e)
                self._rapl = None
                self._windows_emi = None
        if "cpu" in self.required_domains and not cpu_ok:
            self._stop_partial_hwcounters()
            raise RuntimeError("required hardware-counter domain(s) unavailable: cpu")
        if self._nvml is not None:
            try:
                self._nvml.start()
                gpu_ok = True
            except Exception as e:
                self._nvml.close()
                self._nvml = None
                if self.gpu_indices is not None or self.gpu_uuids is not None:
                    self._stop_partial_hwcounters()
                    raise RuntimeError("explicit NVML GPU selection could not be started") from e
                log.info("NVML could not start: %s", e)
        available_domains = {
            domain for domain, available in (("gpu", gpu_ok), ("cpu", cpu_ok)) if available
        }
        missing_domains = self.required_domains - available_domains
        if missing_domains:
            self._stop_partial_hwcounters()
            names = ", ".join(sorted(missing_domains))
            raise RuntimeError(f"required hardware-counter domain(s) unavailable: {names}")
        return gpu_ok or cpu_ok

    def _stop_partial_hwcounters(self) -> None:
        """Best-effort cleanup after a required counter domain is missing."""
        for sampler in (self._nvml, self._rapl, self._windows_emi):
            if sampler is None:
                continue
            try:
                close = getattr(sampler, "close", None)
                if callable(close):
                    close()
                else:
                    sampler.stop()
            except Exception as exc:
                log.debug("counter cleanup failed: %s", exc)
        self._nvml = None
        self._rapl = None
        self._windows_emi = None

    def _fill_tdp(self, duration: float) -> _BackendMeasurement:
        it_wh = _finite_nonnegative(
            tdp_estimate_wh(self.hardware_id, duration),
            name="TDP energy estimate (Wh)",
        )
        operational_wh = it_wh * self.pue
        co2_kg = operational_wh / 1000.0 * self.grid_intensity / 1000.0
        self._tdp_fallback = True
        return _BackendMeasurement(
            energy_j=operational_wh * WH_TO_J,
            co2_operational_kg=co2_kg,
            energy_domains=(self._tdp_domain(),),
            measurement_scope=(
                "facility_adjusted_tdp_estimate" if self.pue > 1.0 else "it_tdp_estimate"
            ),
        )

    def _component_measurement_scope(self) -> str:
        if self.pue > 1.0:
            return "facility_adjusted_it_components"
        return "it_components"

    def _tdp_domain(self) -> str:
        hardware_id = (self.hardware_id or "").lower()
        return "gpu" if "gpu" in hardware_id or "nvidia" in hardware_id else "cpu"

    @staticmethod
    def _resolve_chain(preferred: str, *, allow_fallback: bool = True) -> list[str]:
        if preferred == "wall_meter":
            if allow_fallback:
                raise ValueError("wall_meter cannot be used with fallback enabled")
            return ["wall_meter"]
        order = ["codecarbon", "hwcounters", "tdp"]
        if preferred in order:
            if not allow_fallback:
                return [preferred]
            order.remove(preferred)
            return [preferred, *order]
        if not allow_fallback:
            raise ValueError(f"unknown energy backend: {preferred!r}")
        return order


@contextmanager
def measure(label: str = "default", **kwargs: Any) -> Iterator[EnergyTracker]:
    """Functional alias for `with EnergyTracker(...) as t:`."""
    tracker = EnergyTracker(label=label, **kwargs)
    with tracker:
        yield tracker
