"""Grace power integration is diagnostic, scoped, and never sums nested domains."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from neuro_co.aet.energy.grace import GracePowerSampler, discover_grace_sensors


def _sensor(
    root: Path,
    label: str = "Module Power Socket 0",
    *,
    hwmon: int = 0,
    power_uw: str = "100000000",
    interval_ms: str = "50",
    in_device: bool = True,
) -> Path:
    directory = root / f"hwmon{hwmon}"
    if in_device:
        directory /= "device"
    directory.mkdir(parents=True)
    (directory / "power1_oem_info").write_text(label)
    (directory / "power1_average").write_text(power_uw)
    (directory / "power1_average_interval").write_text(interval_ms)
    return directory / "power1_average"


class _NoThread:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def start(self) -> None:
        pass

    def join(self, **_kwargs: object) -> None:
        pass

    def is_alive(self) -> bool:
        return False


def _manual_sampling(monkeypatch: pytest.MonkeyPatch, timestamps: list[float]) -> None:
    iterator = iter(timestamps)
    monkeypatch.setattr("neuro_co.aet.energy.grace.time.monotonic", lambda: next(iterator))
    monkeypatch.setattr("neuro_co.aet.energy.grace.threading.Thread", _NoThread)


def test_discover_sensors_by_label_on_both_hwmon_layouts(tmp_path: Path) -> None:
    for index, domain in enumerate(("Module", "Grace", "CPU", "SysIO")):
        _sensor(tmp_path, f"{domain} Power Socket 1", hwmon=index, in_device=index % 2 == 0)
    _sensor(tmp_path, "Other sensor", hwmon=9)
    sensors = discover_grace_sensors(tmp_path)

    assert len(sensors) == 4
    assert {sensor.domain for sensor in sensors} == {"module", "grace", "cpu", "sysio"}
    assert {sensor.socket for sensor in sensors} == {1}
    assert {sensor.averaging_interval_s for sensor in sensors} == {0.05}
    module = next(sensor for sensor in sensors if sensor.domain == "module")
    assert module.metadata()["measurement_scope"] == "gh200_module"
    assert module.metadata()["raw_power_unit"] == "microwatt"
    assert module.metadata()["exclusive_scope_required"] is True


def test_discover_requires_recognized_sensors(tmp_path: Path) -> None:
    _sensor(tmp_path, "Unrelated Power Socket 0")
    with pytest.raises(RuntimeError, match="no recognized Grace power sensors"):
        discover_grace_sensors(tmp_path)


def test_discover_rejects_ambiguous_domain_socket(tmp_path: Path) -> None:
    _sensor(tmp_path)
    _sensor(tmp_path, hwmon=1)
    with pytest.raises(RuntimeError, match="ambiguous Grace sensor label"):
        discover_grace_sensors(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="sysfs aliases require POSIX symlinks")
def test_discover_deduplicates_sysfs_aliases(tmp_path: Path) -> None:
    _sensor(tmp_path)
    (tmp_path / "hwmon1").symlink_to(tmp_path / "hwmon0", target_is_directory=True)
    assert len(discover_grace_sensors(tmp_path)) == 1


@pytest.mark.parametrize("power_uw", ["-1", "nan", "inf", "10.5", ""])
def test_discover_rejects_malformed_power(tmp_path: Path, power_uw: str) -> None:
    _sensor(tmp_path, power_uw=power_uw)
    with pytest.raises(RuntimeError, match="not a non-negative integer"):
        discover_grace_sensors(tmp_path)


def test_discover_requires_readable_average_interval(tmp_path: Path) -> None:
    path = _sensor(tmp_path)
    path.with_name("power1_average_interval").unlink()
    with pytest.raises(RuntimeError, match="cannot read Grace sensor"):
        discover_grace_sensors(tmp_path)


def test_discover_rejects_zero_average_interval(tmp_path: Path) -> None:
    _sensor(tmp_path, interval_ms="0")
    with pytest.raises(RuntimeError, match="averaging interval must be positive"):
        discover_grace_sensors(tmp_path)


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf"), True])
def test_sampler_rejects_invalid_poll_interval(tmp_path: Path, interval: float) -> None:
    _sensor(tmp_path)
    with pytest.raises(ValueError, match="finite and positive"):
        GracePowerSampler(discover_grace_sensors(tmp_path), poll_interval_s=interval)


def test_sampler_rejects_empty_or_duplicate_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        GracePowerSampler([])
    _sensor(tmp_path)
    sensors = discover_grace_sensors(tmp_path)
    with pytest.raises(ValueError, match="must not be duplicated"):
        GracePowerSampler([*sensors, *sensors])


def test_integrates_power_using_actual_monotonic_midpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _sensor(tmp_path)
    sampler = GracePowerSampler(discover_grace_sensors(tmp_path), poll_interval_s=0.05)
    _manual_sampling(monkeypatch, [0.0, 0.2, 1.0, 1.2, 3.0, 3.2])

    sampler.start()
    path.write_text("200000000")
    sampler._sample()
    path.write_text("300000000")
    result = sampler.stop()

    reading = result["sensor_integrals"][0]
    assert reading["energy_j"] == pytest.approx(150.0 + 500.0)
    assert reading["duration_s"] == pytest.approx(3.0)
    assert reading["start_monotonic_s"] == pytest.approx(0.1)
    assert reading["end_monotonic_s"] == pytest.approx(3.1)
    assert reading["sample_count"] == 3
    assert reading["minimum_sample_interval_s"] == pytest.approx(1.0)
    assert reading["maximum_sample_interval_s"] == pytest.approx(2.0)
    assert reading["failed_sample_count"] == 0
    assert reading["errors"] == []
    assert result["valid"] is True
    assert result["calibrated"] is False
    assert result["scientific_use"] is False
    assert result["nvml_energy_included"] is False
    assert result["timestamp_clock"] == "monotonic"
    assert result["poll_interval_s"] == 0.05


def test_preserves_module_and_grace_integrals_without_sum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sensor(tmp_path, "Module Power Socket 0", power_uw="200000000")
    _sensor(tmp_path, "Grace Power Socket 0", hwmon=1, power_uw="50000000")
    sampler = GracePowerSampler(discover_grace_sensors(tmp_path))
    _manual_sampling(monkeypatch, [0.0] * 4 + [2.0] * 4)
    sampler.start()
    result = sampler.stop()

    by_domain = {reading["domain"]: reading for reading in result["sensor_integrals"]}
    assert by_domain["module"]["energy_j"] == pytest.approx(400.0)
    assert by_domain["grace"]["energy_j"] == pytest.approx(100.0)
    assert result["nested_domains_additive"] is False
    assert "energy_j" not in result
    assert "energy_gpu_j" not in result


def test_failed_read_invalidates_integral_without_erasing_other_sensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _sensor(tmp_path, "Module Power Socket 0")
    _sensor(tmp_path, "Grace Power Socket 0", hwmon=1, power_uw="50000000")
    sampler = GracePowerSampler(discover_grace_sensors(tmp_path))
    # The missing module value consumes only its pre-read timestamp.
    _manual_sampling(monkeypatch, [0.0] * 4 + [1.0] * 3 + [2.0] * 4)
    sampler.start()
    path.unlink()
    sampler._sample()
    path.write_text("100000000")
    result = sampler.stop()

    by_domain = {reading["domain"]: reading for reading in result["sensor_integrals"]}
    module = by_domain["module"]
    assert module["valid"] is False
    assert module["energy_j"] is None
    assert module["sample_count"] == 2
    assert module["failed_sample_count"] == 1
    assert "cannot read Grace sensor" in module["errors"][0]
    assert by_domain["grace"]["valid"] is True
    assert by_domain["grace"]["energy_j"] == pytest.approx(100.0)
    assert result["valid"] is False


def test_changing_averaging_interval_invalidates_integral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _sensor(tmp_path)
    sampler = GracePowerSampler(discover_grace_sensors(tmp_path))
    _manual_sampling(monkeypatch, [0.0, 0.0, 1.0, 1.0])
    sampler.start()
    path.with_name("power1_average_interval").write_text("100")
    result = sampler.stop()

    reading = result["sensor_integrals"][0]
    assert reading["energy_j"] is None
    assert "averaging interval changed" in reading["errors"][0]


def test_backwards_clock_invalidates_integral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sensor(tmp_path)
    sampler = GracePowerSampler(discover_grace_sensors(tmp_path))
    _manual_sampling(monkeypatch, [1.0, 1.1, 0.5, 0.6])
    sampler.start()
    reading = sampler.stop()["sensor_integrals"][0]
    assert reading["energy_j"] is None
    assert "timestamps are not strictly increasing" in reading["errors"][0]


def test_zero_power_is_a_valid_diagnostic_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sensor(tmp_path, power_uw="0")
    sampler = GracePowerSampler(discover_grace_sensors(tmp_path))
    _manual_sampling(monkeypatch, [0.0, 0.0, 1.0, 1.0])
    sampler.start()
    result = sampler.stop()
    assert result["valid"] is True
    assert result["sensor_integrals"][0]["energy_j"] == 0.0


def test_start_stop_lifecycle_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sensor(tmp_path)
    sampler = GracePowerSampler(discover_grace_sensors(tmp_path))
    _manual_sampling(monkeypatch, [0.0, 0.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="has not been started"):
        sampler.stop()
    sampler.start()
    with pytest.raises(RuntimeError, match="already been started"):
        sampler.start()
    sampler.stop()
    with pytest.raises(RuntimeError, match="already been stopped"):
        sampler.stop()


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_sampling_continues_while_workload_holds_gil(tmp_path: Path) -> None:
    import ctypes
    import os
    import time

    from neuro_co.aet.energy.grace import GraceProcessSampler

    path = _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path), poll_interval_s=0.02)
    sampler.start()
    # PyDLL keeps the GIL while calling C, unlike ctypes.CDLL or time.sleep.
    libc = ctypes.PyDLL(None)
    libc.usleep.argtypes = [ctypes.c_uint]
    libc.usleep.restype = ctypes.c_int
    try:
        started = time.monotonic()
        assert libc.usleep(1_200_000) == 0
        elapsed = time.monotonic() - started
        replacement = path.with_suffix(".next")
        replacement.write_text("200000000")
        replacement.replace(path)
        assert libc.usleep(250_000) == 0
    finally:
        result = sampler.stop()
    integral = result["sensor_integrals"][0]
    assert elapsed >= 1.1
    assert result["sampling_execution"] == "independent_process"
    assert result["worker_pid"] != os.getpid()
    assert result["valid"] is True
    assert integral["sample_count"] > 40
    assert integral["maximum_sample_interval_s"] < 1.0
    assert {sample["power_w"] for sample in integral["samples"]} == {100.0, 200.0}
    assert integral["energy_j"] > 100 * integral["duration_s"]
    assert sampler._process.poll() == 0


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_drains_trace_larger_than_pipe_buffer(tmp_path: Path) -> None:
    import json
    import time

    from neuro_co.aet.energy.grace import GraceProcessSampler

    _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path), poll_interval_s=0.0005)
    sampler.start()
    time.sleep(1.5)
    result = sampler.stop()
    assert len(json.dumps(result)) > 65_536
    assert result["valid"] is True
    assert sampler._process.poll() == 0


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_detects_worker_death_and_reaps(tmp_path: Path) -> None:
    from neuro_co.aet.energy.grace import GraceProcessSampler

    _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path))
    sampler.start()
    sampler._process.kill()
    sampler._process.wait(timeout=3)
    with pytest.raises(RuntimeError, match="worker failed"):
        sampler.stop()
    assert sampler._process.poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_start_failure_is_reaped(tmp_path: Path) -> None:
    from neuro_co.aet.energy.grace import GraceProcessSampler

    path = _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path))
    path.unlink()
    with pytest.raises(RuntimeError, match="failed before readiness"):
        sampler.start()
    assert sampler._process.poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_sensor_failure_retains_invalid_trace(tmp_path: Path) -> None:
    import time

    from neuro_co.aet.energy.grace import GraceProcessSampler

    path = _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path), poll_interval_s=0.01)
    sampler.start()
    path.unlink()
    time.sleep(0.05)
    result = sampler.stop()
    assert result["valid"] is False
    assert result["sensor_integrals"][0]["energy_j"] is None
    assert result["sensor_integrals"][0]["failed_sample_count"] > 0
    assert sampler._process.poll() == 0


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_exits_when_parent_closes_control_pipe(tmp_path: Path) -> None:
    from neuro_co.aet.energy.grace import GraceProcessSampler

    _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path))
    sampler.start()
    sampler._process.stdin.close()
    sampler._process.wait(timeout=3)
    assert sampler._process.returncode == 0
    sampler._process.stdout.close()
    sampler._process.stderr.close()


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_start_stop_lifecycle(tmp_path: Path) -> None:
    from neuro_co.aet.energy.grace import GraceProcessSampler

    _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path))
    with pytest.raises(RuntimeError, match="has not been started"):
        sampler.stop()
    sampler.start()
    try:
        with pytest.raises(RuntimeError, match="already been started"):
            sampler.start()
    finally:
        sampler.stop()
    with pytest.raises(RuntimeError, match="already been stopped"):
        sampler.stop()


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_preserves_trace_when_group_termination_reaches_worker(tmp_path: Path) -> None:
    import signal
    import time

    from neuro_co.aet.energy.grace import GraceProcessSampler

    _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path), poll_interval_s=0.01)
    sampler.start()
    sampler._process.send_signal(signal.SIGTERM)
    time.sleep(0.05)
    result = sampler.stop()
    assert result["valid"] is True
    assert result["sensor_integrals"][0]["sample_count"] >= 3
    assert sampler._process.returncode == 0


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_start_timeout_is_reaped(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    import sys

    from neuro_co.aet.energy import grace

    _sensor(tmp_path)
    original = subprocess.Popen

    def stalled_worker(_args, **kwargs):
        return original([sys.executable, "-I", "-c", "import time; time.sleep(10)"], **kwargs)

    monkeypatch.setattr(grace.subprocess, "Popen", stalled_worker)
    sampler = grace.GraceProcessSampler(discover_grace_sensors(tmp_path))
    sampler._timeout_s = 0.05
    with pytest.raises(RuntimeError, match="startup timed out"):
        sampler.start()
    assert sampler._process.poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="GH200 sampling requires POSIX pipes")
def test_process_stop_timeout_is_reaped(tmp_path: Path) -> None:
    import signal

    from neuro_co.aet.energy.grace import GraceProcessSampler

    _sensor(tmp_path)
    sampler = GraceProcessSampler(discover_grace_sensors(tmp_path))
    sampler.start()
    sampler._process.send_signal(signal.SIGSTOP)
    sampler._timeout_s = 0.05
    with pytest.raises(RuntimeError, match="stop timed out"):
        sampler.stop()
    assert sampler._process.poll() is not None
