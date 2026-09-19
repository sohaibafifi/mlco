"""Tests for the strict whole-system AC wall-meter contract."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from neuro_co.aet import EnergyTracker
from neuro_co.aet.energy.wall_meter import WallMeterResult

TRACE_SHA256 = "a" * 64
DEVICE_SHA256 = "b" * 64


def _result(**changes: object) -> WallMeterResult:
    values: dict[str, object] = {
        "energy_j": 5000.0,
        "duration_s": 10.0,
        "sample_interval_s": 1.0,
        "sample_count": 11,
        "dropped_sample_count": 1,
        "trace_sha256": TRACE_SHA256,
        "calibration_id": "calibration-2026-08-21",
        "device_id_sha256": DEVICE_SHA256,
        "started_at_utc": "2026-08-21T09:00:00Z",
        "ended_at_utc": "2026-08-21T09:00:10+00:00",
        "start_monotonic_s": 100.0,
        "end_monotonic_s": 110.0,
    }
    values.update(changes)
    return WallMeterResult(**values)  # type: ignore[arg-type]


class _FakeWallMeter:
    def __init__(
        self,
        result: WallMeterResult | None = None,
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self.result = result or _result()
        self.start_error = start_error
        self.stop_error = stop_error
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> WallMeterResult:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        return self.result


def _tracker(sampler: object, **changes: object) -> EnergyTracker:
    values: dict[str, object] = {
        "backend": "wall_meter",
        "wall_meter_sampler": sampler,
        "pue": 1.0,
        "report_embodied": False,
        "allow_fallback": False,
        "grid_intensity_g_per_kwh": 400.0,
        "items": 20,
    }
    values.update(changes)
    return EnergyTracker(**values)  # type: ignore[arg-type]


def test_wall_meter_serializes_raw_ac_energy_and_hashed_metadata() -> None:
    sampler = _FakeWallMeter()
    tracker = _tracker(sampler)

    with tracker:
        pass

    assert sampler.start_calls == 1
    assert sampler.stop_calls == 1
    assert tracker.reading is not None
    record = tracker.reading.to_dict()
    assert record["backend"] == "wall_meter"
    assert record["duration_s"] == pytest.approx(10.0)
    assert record["energy_j"] == pytest.approx(5000.0)
    assert record["energy_gpu_j"] == 0.0
    assert record["energy_cpu_j"] == 0.0
    assert record["energy_dram_j"] == 0.0
    assert record["energy_domains"] == ["whole_system_ac"]
    assert record["measurement_scope"] == "whole_system_ac"
    assert record["avg_power_w"] == pytest.approx(500.0)
    assert record["co2_operational_kg"] == pytest.approx(5000.0 / 3_600_000.0 * 400.0 / 1000.0)
    assert record["co2_embodied_kg"] == 0.0
    assert record["hardware"]["pue"] == 1.0
    assert record["extra"]["pue_applied_to_operational"] is False
    assert record["extra"]["allow_fallback"] is False
    assert record["extra"]["wall_meter"] == _result().to_metadata_dict()
    assert "private-serial" not in json.dumps(record)


def test_wall_meter_stops_and_records_failed_workload() -> None:
    sampler = _FakeWallMeter()
    tracker = _tracker(sampler)

    with pytest.raises(RuntimeError, match="workload failed"):
        with tracker:
            raise RuntimeError("workload failed")

    assert sampler.stop_calls == 1
    assert tracker.reading is not None
    assert tracker.reading.extra["run_status"] == "failed"
    assert tracker.reading.extra["failure_type"] == "RuntimeError"


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"energy_j": float("nan")}, "energy_j must be finite"),
        ({"energy_j": -1.0}, "energy_j must be non-negative"),
        ({"duration_s": 0.0}, "duration_s must be greater than zero"),
        ({"sample_interval_s": 0.0}, "sample_interval_s must be greater than zero"),
        ({"sample_interval_s": 11.0}, "must not exceed duration_s"),
        ({"sample_count": 0}, "sample_count must be greater than zero"),
        ({"sample_count": True}, "sample_count must be an integer"),
        ({"dropped_sample_count": -1}, "dropped_sample_count must be non-negative"),
        ({"trace_sha256": "A" * 64}, "trace_sha256.*lowercase SHA-256"),
        ({"device_id_sha256": "short"}, "device_id_sha256.*SHA-256"),
        ({"calibration_id": " "}, "calibration_id must be a non-empty"),
        ({"ended_at_utc": None}, "must be provided together"),
        ({"started_at_utc": "2026-08-21T09:00:00"}, "include a UTC offset"),
        ({"started_at_utc": "2026-08-21T10:00:00+01:00"}, "expressed in UTC"),
        (
            {"ended_at_utc": "2026-08-21T08:59:59Z"},
            "must not precede",
        ),
        ({"end_monotonic_s": None}, "must be provided together"),
        ({"end_monotonic_s": 99.0}, "must be greater than"),
        ({"end_monotonic_s": 111.0}, "must match the monotonic"),
    ],
)
def test_wall_meter_result_rejects_invalid_or_ambiguous_values(
    changes: dict[str, object],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _result(**changes)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"wall_meter_sampler": None}, "requires wall_meter_sampler"),
        ({"pue": 1.1}, "requires pue=1.0"),
        ({"report_embodied": True}, "requires report_embodied=False"),
        ({"allow_fallback": True}, "requires allow_fallback=False"),
    ],
)
def test_wall_meter_tracker_rejects_non_strict_configuration(
    changes: dict[str, object],
    error: str,
) -> None:
    sampler = _FakeWallMeter()
    values: dict[str, object] = {
        "backend": "wall_meter",
        "wall_meter_sampler": sampler,
        "pue": 1.0,
        "report_embodied": False,
        "allow_fallback": False,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError), match=error):
        EnergyTracker(**values)  # type: ignore[arg-type]


def test_wall_meter_sampler_cannot_be_attached_to_another_backend() -> None:
    with pytest.raises(ValueError, match="only supported"):
        EnergyTracker(backend="tdp", wall_meter_sampler=_FakeWallMeter())


def test_wall_meter_rejects_sampler_without_start_stop_contract() -> None:
    with pytest.raises(TypeError, match=r"start\(\).*stop\(\)"):
        _tracker(object())


def test_wall_meter_start_exception_is_strict() -> None:
    sampler = _FakeWallMeter(start_error=OSError("transport unavailable"))
    tracker = _tracker(sampler)

    with pytest.raises(RuntimeError, match="wall_meter failed while starting"):
        with tracker:
            pass

    assert sampler.start_calls == 1
    assert sampler.stop_calls == 0
    assert tracker.reading is None
    assert tracker._tdp_fallback is False


def test_wall_meter_stop_exception_is_strict_and_never_falls_back() -> None:
    sampler = _FakeWallMeter(stop_error=OSError("trace incomplete"))
    tracker = _tracker(sampler)

    with pytest.raises(RuntimeError, match="wall_meter failed while stopping"):
        with tracker:
            pass

    assert sampler.start_calls == 1
    assert sampler.stop_calls == 1
    assert tracker.reading is None
    assert tracker._tdp_fallback is False


def test_wall_meter_rejects_unvalidated_stop_result() -> None:
    sampler = _FakeWallMeter()
    sampler.result = replace(_result(), energy_j=1.0)
    sampler.stop = lambda: {"energy_j": 1.0}  # type: ignore[method-assign]
    tracker = _tracker(sampler)

    with pytest.raises(TypeError, match="must return WallMeterResult"):
        with tracker:
            pass

    assert tracker.reading is None
    assert tracker._tdp_fallback is False
