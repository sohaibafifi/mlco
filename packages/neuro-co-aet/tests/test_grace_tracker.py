"""Explicit module accounting without fabricated CPU/GPU components."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuro_co.aet.energy import GraceModuleTracker, checked_module_energy
from neuro_co.aet.energy import grace_tracker as module
from neuro_co.aet.energy.grace import GracePowerSensor


@pytest.fixture
def runtime(monkeypatch):
    sensors = [
        GracePowerSensor(
            f"Module Power Socket {socket}",
            socket,
            "module",
            Path(f"/sys/fixture/{socket}/power1_average"),
            0.05,
        )
        for socket in (0, 1)
    ]
    sensors.append(GracePowerSensor("Grace Power Socket 0", 0, "grace", Path("/sys/grace"), 0.05))
    options = {"power_w": 100.0, "fail": False, "selected": []}

    class Sampler:
        def __init__(self, selected, *, poll_interval_s):
            assert len(selected) == 1
            self.sensor = selected[0]
            self.interval = poll_interval_s
            self.started = False
            options["selected"].append(self.sensor.sensor_id)

        def start(self):
            if self.started:
                raise RuntimeError("already started")
            self.started = True

        def stop(self):
            if options["fail"]:
                raise RuntimeError("sensor unavailable")
            return {
                "valid": True,
                "measurement_mode": "grace_hwmon_power_integration",
                "integration_rule": "trapezoidal",
                "timestamp_clock": "monotonic",
                "nested_domains_additive": False,
                "nvml_energy_included": False,
                "poll_interval_s": self.interval,
                "calibrated": False,
                "sensor_integrals": [
                    {
                        **self.sensor.metadata(),
                        "valid": True,
                        "energy_j": options["power_w"],
                        "duration_s": 1.0,
                        "start_monotonic_s": 10.0,
                        "end_monotonic_s": 11.0,
                        "sample_count": 3,
                        "failed_sample_count": 0,
                        "errors": [],
                        "minimum_sample_interval_s": 0.5,
                        "maximum_sample_interval_s": 0.5,
                        "samples": [
                            {"monotonic_s": time, "power_w": options["power_w"]}
                            for time in (10.0, 10.5, 11.0)
                        ],
                    }
                ],
            }

    monkeypatch.setattr(module, "discover_grace_sensors", lambda _: tuple(sensors))
    monkeypatch.setattr(module, "GraceProcessSampler", Sampler)
    return options


def _reading(runtime, socket=0, items=256):
    tracker = module.GraceModuleTracker("test", socket=socket)
    with tracker as active:
        active.n_items = items
    return tracker


def test_explicit_socket_selects_only_module_and_preserves_raw_trace(runtime):
    tracker = _reading(runtime, socket=1)
    payload = module.checked_module_energy(tracker, socket=1, instances=256)
    assert runtime["selected"] == ["module:socket1"]
    assert payload["energy_j"] == payload["energy_module_j"] == 100.0
    assert payload["measurement_scope"] == "gh200_module"
    assert payload["duration_s"] == 1.0
    assert payload["items_processed"] == 256
    assert payload["calibrated"] is False
    assert payload["whole_system_energy"] is False
    assert len(payload["raw_measurement"]["sensor_integrals"][0]["samples"]) == 3
    assert not {"energy_cpu_j", "energy_gpu_j", "energy_dram_j"} & payload.keys()


@pytest.mark.parametrize("socket", [-1, True, "0", 2])
def test_socket_is_explicit_and_must_exist(runtime, socket):
    with pytest.raises(ValueError):
        module.GraceModuleTracker("test", socket=socket)


def test_work_exception_keeps_valid_consumed_energy(runtime):
    tracker = module.GraceModuleTracker("test", socket=0)
    with pytest.raises(RuntimeError, match="work failed"), tracker as active:
        active.n_items = 2
        raise RuntimeError("work failed")
    payload = module.checked_module_energy(tracker, socket=0, instances=2)
    assert payload["extra"]["run_status"] == "failed"
    assert payload["energy_j"] == 100.0


def test_failed_sampling_is_not_zero_imputed(runtime):
    runtime["fail"] = True
    tracker = _reading(runtime)
    assert tracker.reading.to_dict()["energy_j"] is None
    assert "sensor unavailable" in tracker.reading.to_dict()["raw_measurement"]["error"]
    with pytest.raises(ValueError, match="sampling is invalid"):
        module.checked_module_energy(tracker, socket=0, instances=256)


def test_zero_integral_allowed_only_for_explicit_startup(runtime):
    runtime["power_w"] = 0.0
    tracker = _reading(runtime, items=1)
    with pytest.raises(ValueError, match="positive"):
        module.checked_module_energy(tracker, socket=0, instances=1)
    assert (
        module.checked_module_energy(tracker, socket=0, instances=1, allow_zero=True)["energy_j"]
        == 0
    )


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p.update(energy_cpu_j=0.0), "undecomposed"),
        (lambda p: p.update(energy_module_j=90.0), "disagrees"),
        (lambda p: p.update(items_processed=255), "work count disagrees"),
        (
            lambda p: p["raw_measurement"]["sensor_integrals"][0].update(failed_sample_count=1),
            "failed samples",
        ),
        (
            lambda p: p["raw_measurement"]["sensor_integrals"][0].update(sample_count=2),
            "counted samples",
        ),
        (
            lambda p: p["raw_measurement"]["sensor_integrals"][0]["samples"][1].update(
                power_w=200.0
            ),
            "retained trace",
        ),
        (
            lambda p: p["raw_measurement"]["sensor_integrals"][0]["samples"][1].update(
                monotonic_s=9.0
            ),
            "timestamps",
        ),
        (
            lambda p: p["raw_measurement"]["sensor_integrals"][0].update(averaging_interval_s=0.1),
            "averaging interval",
        ),
    ],
)
def test_strict_validation_rejects_corrupt_scope_trace_or_counts(runtime, change, match):
    tracker = _reading(runtime)
    change(tracker.reading.to_dict())
    with pytest.raises(ValueError, match=match):
        module.checked_module_energy(tracker, socket=0, instances=256)


def test_strict_validation_checks_identity_duration_and_poll_gaps(runtime):
    tracker = _reading(runtime)
    with pytest.raises(ValueError, match="identity or scope"):
        module.checked_module_energy(tracker, socket=1, instances=256)
    with pytest.raises(ValueError, match="shorter than"):
        module.checked_module_energy(tracker, socket=0, instances=256, minimum_duration_s=2)
    with pytest.raises(ValueError, match="sampling gaps"):
        module.checked_module_energy(tracker, socket=0, instances=256, max_sample_gap_s=0.2)


def test_public_tracker_api_preserves_module_scope(runtime):
    with GraceModuleTracker("public-api", socket=0) as tracker:
        tracker.n_items = 8
    record = checked_module_energy(tracker, socket=0, instances=8)
    assert record["energy_domains"] == ["module"]
    assert record["schema_version"] == "aet-grace-module-energy/v1"
