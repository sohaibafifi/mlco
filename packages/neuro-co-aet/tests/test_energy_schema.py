"""Focused tests for the canonical energy record and backend unit boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import codecarbon
import pytest

from neuro_co.aet import EnergyReading, EnergyTracker
from neuro_co.aet.energy.types import ENERGY_READING_UNITS, ENERGY_SCHEMA_VERSION


def test_energy_reading_serializes_only_canonical_si_fields() -> None:
    reading = EnergyReading(
        duration_s=2.0,
        energy_j=7200.0,
        energy_gpu_j=3600.0,
        energy_cpu_j=1800.0,
        energy_dram_j=900.0,
        co2_operational_kg=0.25,
        co2_embodied_kg=0.01,
        avg_power_w=3600.0,
        items_processed=8,
        backend="fixture",
        energy_domains=("gpu", "cpu", "dram"),
        measurement_scope="it_components",
        hardware={"id": "fixture-host"},
        extra={"label": "unit-test"},
    )

    record = reading.to_dict()

    assert record["schema_version"] == ENERGY_SCHEMA_VERSION
    assert record["units"] == ENERGY_READING_UNITS
    assert record["energy_j"] == pytest.approx(7200.0)
    assert record["co2_total_kg"] == pytest.approx(0.26)
    assert record["throughput_items_per_s"] == pytest.approx(4.0)
    assert record["energy_domains"] == ["gpu", "cpu", "dram"]
    assert record["measurement_scope"] == "it_components"
    assert record["hardware"] == {"id": "fixture-host"}
    assert record["extra"] == {"label": "unit-test"}
    assert "energy_wh" not in record
    assert "co2_g_total" not in record
    assert "throughput" not in record
    assert "label" not in record


def test_legacy_units_require_explicit_conversion() -> None:
    reading = EnergyReading(
        duration_s=2.0,
        energy_j=7200.0,
        co2_operational_kg=0.25,
        co2_embodied_kg=0.01,
        items_processed=8,
        hardware={"id": "fixture-host"},
        extra={"label": "unit-test"},
    )

    record = reading.to_legacy_dict()

    assert record["energy_wh"] == pytest.approx(2.0)
    assert record["co2_g_operational"] == pytest.approx(250.0)
    assert record["co2_g_embodied"] == pytest.approx(10.0)
    assert record["co2_g_total"] == pytest.approx(260.0)
    assert record["throughput"] == pytest.approx(4.0)
    assert record["hardware_id"] == "fixture-host"
    assert record["label"] == "unit-test"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"energy_j": float("nan")}, "energy_j"),
        ({"co2_operational_kg": -0.1}, "co2_operational_kg"),
        ({"items_processed": -1}, "items_processed"),
        ({"energy_j": 1.0, "energy_gpu_j": 2.0}, "component energy"),
        ({"extra": {"energy_j": 0.0}}, "reserved output"),
        ({"extra": {"energy_wh": 0.0}}, "reserved output"),
        ({"energy_domains": ("gpu", "gpu")}, "duplicates"),
        ({"measurement_scope": ""}, "measurement_scope"),
    ],
)
def test_energy_reading_rejects_invalid_or_ambiguous_values(
    kwargs: dict[str, object],
    error: str,
) -> None:
    values: dict[str, object] = {"duration_s": 1.0, "energy_j": 0.0}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error):
        EnergyReading(**values)  # type: ignore[arg-type]


def test_codecarbon_applies_pue_to_energy_carbon_and_components() -> None:
    final = SimpleNamespace(
        energy_consumed=2.0,
        gpu_energy=1.0,
        cpu_energy=0.5,
        ram_energy=0.25,
    )

    class _CodeCarbonFixture:
        final_emissions_data = final

        @staticmethod
        def stop() -> float:
            return 0.4

    tracker = EnergyTracker(
        backend="codecarbon",
        pue=1.5,
        report_embodied=False,
    )
    tracker._backend_used = "codecarbon"
    tracker._codecarbon = _CodeCarbonFixture()

    measurement = tracker._stop_backend(duration=10.0)

    assert measurement.energy_j == pytest.approx(2.0 * 3_600_000.0 * 1.5)
    assert measurement.energy_gpu_j == pytest.approx(1.0 * 3_600_000.0 * 1.5)
    assert measurement.energy_cpu_j == pytest.approx(0.5 * 3_600_000.0 * 1.5)
    assert measurement.energy_dram_j == pytest.approx(0.25 * 3_600_000.0 * 1.5)
    assert measurement.co2_operational_kg == pytest.approx(0.4 * 1.5)
    assert measurement.energy_domains == ("gpu", "cpu", "dram")
    assert measurement.measurement_scope == "facility_adjusted_it_components"


def test_codecarbon_is_started_with_neutral_internal_pue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CodeCarbonFixture:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        @staticmethod
        def start() -> None:
            return None

    monkeypatch.setattr(codecarbon, "EmissionsTracker", _CodeCarbonFixture)
    tracker = EnergyTracker(
        backend="codecarbon",
        pue=1.6,
        report_embodied=False,
    )

    assert tracker._try_start_codecarbon() is True
    assert captured["pue"] == pytest.approx(1.0)


def test_hardware_counter_components_are_non_overlapping_and_pue_scaled() -> None:
    class _NvmlFixture:
        @staticmethod
        def stop() -> float:
            return 2.0

    class _RaplFixture:
        last_dram_wh = 1.0

        @staticmethod
        def stop() -> float:
            return 4.0

    tracker = EnergyTracker(
        backend="hwcounters",
        pue=1.25,
        grid_intensity_g_per_kwh=400.0,
        report_embodied=False,
    )
    tracker._backend_used = "hwcounters"
    tracker._nvml = _NvmlFixture()  # type: ignore[assignment]
    tracker._rapl = _RaplFixture()  # type: ignore[assignment]

    measurement = tracker._stop_backend(duration=10.0)

    assert measurement.energy_gpu_j == pytest.approx(2.0 * 3600.0 * 1.25)
    assert measurement.energy_cpu_j == pytest.approx(3.0 * 3600.0 * 1.25)
    assert measurement.energy_dram_j == pytest.approx(1.0 * 3600.0 * 1.25)
    assert measurement.energy_j == pytest.approx(6.0 * 3600.0 * 1.25)
    assert measurement.co2_operational_kg == pytest.approx(7.5 / 1000.0 * 400.0 / 1000.0)
    assert measurement.energy_domains == ("gpu", "cpu", "dram")
    assert measurement.measurement_scope == "facility_adjusted_it_components"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"pue": 0.99}, "pue"),
        ({"pue": float("inf")}, "pue"),
        ({"grid_intensity_g_per_kwh": -1.0}, "grid_intensity"),
        ({"lifetime_s": 0.0}, "lifetime_s"),
        ({"items": -1}, "items"),
        ({"allow_fallback": "no"}, "allow_fallback"),
        ({"backend": "tdp", "required_domains": {"cpu"}}, "only supported"),
        ({"backend": "hwcounters", "required_domains": {"wall"}}, "unsupported"),
        ({"backend": "hwcounters", "required_domains": {1}}, "only strings"),
    ],
)
def test_tracker_rejects_invalid_configuration(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        EnergyTracker(**kwargs)  # type: ignore[arg-type]


def test_codecarbon_rejects_non_finite_backend_values() -> None:
    final = SimpleNamespace(
        energy_consumed=float("nan"),
        gpu_energy=0.0,
        cpu_energy=0.0,
        ram_energy=0.0,
    )

    class _CodeCarbonFixture:
        final_emissions_data = final

        @staticmethod
        def stop() -> float:
            return 0.0

    tracker = EnergyTracker(backend="codecarbon", report_embodied=False)
    tracker._backend_used = "codecarbon"
    tracker._codecarbon = _CodeCarbonFixture()

    with pytest.raises(ValueError, match="energy_consumed"):
        tracker._stop_backend(duration=10.0)


def test_strict_backend_refuses_start_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = EnergyTracker(
        backend="codecarbon",
        allow_fallback=False,
        report_embodied=False,
    )
    monkeypatch.setattr(tracker, "_try_start_codecarbon", lambda: False)

    with pytest.raises(RuntimeError, match=r"codecarbon.*unavailable"):
        tracker._start_backend()

    assert tracker._backend_used == "none"


def test_strict_backend_refuses_stop_substitution() -> None:
    class _BrokenCodeCarbonFixture:
        @staticmethod
        def stop() -> float:
            raise RuntimeError("counter failed")

    tracker = EnergyTracker(
        backend="codecarbon",
        allow_fallback=False,
        report_embodied=False,
    )
    tracker._backend_used = "codecarbon"
    tracker._codecarbon = _BrokenCodeCarbonFixture()

    with pytest.raises(RuntimeError, match="failed while stopping"):
        tracker._stop_backend(duration=10.0)

    assert tracker._tdp_fallback is False


def test_required_hardware_domains_block_partial_counter_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NvmlFixture:
        def __init__(
            self,
            gpu_indices: tuple[int, ...] | None = None,
            gpu_uuids: tuple[str, ...] | None = None,
        ) -> None:
            assert gpu_indices is None
            assert gpu_uuids is None

        def start(self) -> None:
            return None

        def stop(self) -> float:
            return 0.0

    class _UnavailableRaplFixture:
        def __init__(self) -> None:
            raise RuntimeError("RAPL unavailable")

    monkeypatch.setattr("neuro_co.aet.energy.tracker.NvmlSampler", _NvmlFixture)
    monkeypatch.setattr("neuro_co.aet.energy.tracker.RaplSampler", _UnavailableRaplFixture)
    tracker = EnergyTracker(
        backend="hwcounters",
        required_domains={"gpu", "cpu"},
        report_embodied=False,
    )

    with pytest.raises(RuntimeError, match=r"required.*cpu"):
        tracker._try_start_hwcounters()

    assert tracker._nvml is None
    assert tracker._rapl is None


def test_tracker_marks_failed_workload_and_times_through_backend_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((10.0, 15.0, 17.0))
    monkeypatch.setattr(
        "neuro_co.aet.energy.tracker.time.perf_counter",
        lambda: next(clock),
    )
    tracker = EnergyTracker(
        backend="tdp",
        pue=1.0,
        report_embodied=False,
        allow_fallback=False,
    )

    with pytest.raises(RuntimeError, match="workload failed"):
        with tracker:
            raise RuntimeError("workload failed")

    assert tracker.reading is not None
    assert tracker.reading.duration_s == pytest.approx(7.0)
    assert tracker.reading.extra["run_status"] == "failed"
    assert tracker.reading.extra["failure_type"] == "RuntimeError"
