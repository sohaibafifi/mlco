"""Diagnostic integration of NVIDIA Grace hwmon power sensors.

These vendor-labelled domains are nested, not additive. In particular, a
GH200 module contains both Grace and Hopper. The integrals remain separate;
this module neither adds NVML readings nor reports calibrated system energy.
"""

from __future__ import annotations

import json
import math
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

_LABEL = re.compile(r"(Module|Grace|CPU|SysIO) Power Socket (\d+)")
_SCOPES = {
    "module": "gh200_module",
    "grace": "grace_socket",
    "cpu": "cpu_rail",
    "sysio": "sysio_rail",
}


def _read_nonnegative_integer(path: Path) -> int:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read Grace sensor {path}: {exc}") from exc
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise RuntimeError(f"Grace sensor {path} is not a non-negative integer: {raw!r}")
    return int(raw)


@dataclass(frozen=True)
class GracePowerSensor:
    """One vendor-labelled hwmon domain, with no inferred GPU association."""

    label: str
    socket: int
    domain: str
    power_path: Path
    averaging_interval_s: float

    @property
    def sensor_id(self) -> str:
        return f"{self.domain}:socket{self.socket}"

    @property
    def interval_path(self) -> Path:
        return self.power_path.with_name(self.power_path.name + "_interval")

    def metadata(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "label": self.label,
            "socket": self.socket,
            "domain": self.domain,
            "measurement_scope": _SCOPES[self.domain],
            "power_path": str(self.power_path),
            "raw_power_unit": "microwatt",
            "averaging_interval_s": self.averaging_interval_s,
            "exclusive_scope_required": True,
        }


def discover_grace_sensors(
    hwmon_root: str | Path = "/sys/class/hwmon",
) -> tuple[GracePowerSensor, ...]:
    """Find recognized sensors by OEM label, not by unstable hwmon numbering.

    Missing or malformed values on a recognized sensor raise an explicit
    error. Unrelated hwmon sensors are ignored. Duplicate domain/socket
    labels are rejected unless they resolve to the same sysfs power file.
    """

    root = Path(hwmon_root)
    labels = sorted(
        {*root.glob("hwmon*/power*_oem_info"), *root.glob("hwmon*/device/power*_oem_info")}
    )
    sensors: dict[str, GracePowerSensor] = {}
    for label_path in labels:
        try:
            label = label_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"cannot read hwmon OEM label {label_path}: {exc}") from exc
        match = _LABEL.fullmatch(label)
        if match is None:
            continue
        power_path = label_path.with_name(label_path.name.replace("_oem_info", "_average"))
        interval_ms = _read_nonnegative_integer(power_path.with_name(power_path.name + "_interval"))
        if interval_ms == 0:
            raise RuntimeError(f"Grace sensor averaging interval must be positive: {power_path}")
        _read_nonnegative_integer(power_path)
        sensor = GracePowerSensor(
            label=label,
            socket=int(match[2]),
            domain=match[1].lower(),
            power_path=power_path,
            averaging_interval_s=interval_ms / 1000.0,
        )
        previous = sensors.get(sensor.sensor_id)
        if previous is not None and previous.power_path.resolve() != power_path.resolve():
            raise RuntimeError(f"ambiguous Grace sensor label: {label}")
        sensors[sensor.sensor_id] = sensor
    if not sensors:
        raise RuntimeError(f"no recognized Grace power sensors found in {root}")
    return tuple(sorted(sensors.values(), key=lambda sensor: (sensor.socket, sensor.domain)))


@dataclass
class _Integral:
    samples: list[dict[str, float]] = field(default_factory=list)
    energy_j: float = 0.0
    errors: list[str] = field(default_factory=list)
    failed_sample_count: int = 0


class GracePowerSampler:
    """Trapezoidal integration with per-read monotonic midpoint timestamps.

    ``stop()`` returns JSON-compatible diagnostics, not an aggregate energy.
    Any failed sample invalidates that sensor's entire integral (``energy_j``
    becomes null), while preserving the successful trace and error messages.
    Sensor averaging intervals are distinct from the requested polling period.
    CUDA work must be synchronized by the caller before stopping this sampler.
    """

    def __init__(
        self,
        sensors: Sequence[GracePowerSensor],
        *,
        poll_interval_s: float = 0.1,
    ) -> None:
        if (
            isinstance(poll_interval_s, bool)
            or not math.isfinite(poll_interval_s)
            or poll_interval_s <= 0.0
        ):
            raise ValueError("poll_interval_s must be finite and positive")
        if not sensors:
            raise ValueError("at least one Grace power sensor is required")
        if len({sensor.sensor_id for sensor in sensors}) != len(sensors):
            raise ValueError("Grace sensor domains must not be duplicated")
        self.sensors = tuple(sensors)
        self.poll_interval_s = float(poll_interval_s)
        self._integrals = {sensor.sensor_id: _Integral() for sensor in self.sensors}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Grace sampler has already been started")
        self._started = True
        self._sample()
        self._thread = threading.Thread(target=self._poll, name="grace-power", daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop_event.wait(self.poll_interval_s):
            self._sample()

    def _sample(self) -> None:
        for sensor in self.sensors:
            integral = self._integrals[sensor.sensor_id]
            try:
                before = time.monotonic()
                power_w = _read_nonnegative_integer(sensor.power_path) / 1_000_000.0
                after = time.monotonic()
                if not all(math.isfinite(value) for value in (before, after, power_w)):
                    raise RuntimeError("non-finite power or monotonic time")
                if after < before:
                    raise RuntimeError("monotonic clock moved backwards during sensor read")
                interval_s = _read_nonnegative_integer(sensor.interval_path) / 1000.0
                if interval_s != sensor.averaging_interval_s:
                    raise RuntimeError("sensor averaging interval changed during measurement")
                timestamp_s = (before + after) / 2.0
                if integral.samples:
                    previous = integral.samples[-1]
                    elapsed_s = timestamp_s - previous["monotonic_s"]
                    if elapsed_s <= 0.0:
                        raise RuntimeError("sensor sample timestamps are not strictly increasing")
                    integral.energy_j += 0.5 * (previous["power_w"] + power_w) * elapsed_s
                integral.samples.append({"monotonic_s": timestamp_s, "power_w": power_w})
            except (OSError, RuntimeError, ValueError, OverflowError) as exc:
                integral.failed_sample_count += 1
                message = str(exc)
                if message not in integral.errors:
                    integral.errors.append(message)

    def stop(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Grace sampler has not been started")
        if self._stopped:
            raise RuntimeError("Grace sampler has already been stopped")
        self._stopped = True
        self._stop_event.set()
        assert self._thread is not None
        self._thread.join(timeout=max(2.0, 4.0 * self.poll_interval_s))
        if self._thread.is_alive():
            raise RuntimeError("Grace power sampling thread did not stop")
        self._sample()
        readings = []
        for sensor in self.sensors:
            integral = self._integrals[sensor.sensor_id]
            samples = integral.samples
            intervals = [
                current["monotonic_s"] - previous["monotonic_s"]
                for previous, current in pairwise(samples)
            ]
            valid = bool(intervals) and not integral.errors
            readings.append(
                {
                    **sensor.metadata(),
                    "valid": valid,
                    "energy_j": integral.energy_j if valid else None,
                    "duration_s": sum(intervals),
                    "start_monotonic_s": samples[0]["monotonic_s"] if samples else None,
                    "end_monotonic_s": samples[-1]["monotonic_s"] if samples else None,
                    "sample_count": len(samples),
                    "failed_sample_count": integral.failed_sample_count,
                    "minimum_sample_interval_s": min(intervals) if intervals else None,
                    "maximum_sample_interval_s": max(intervals) if intervals else None,
                    "errors": list(integral.errors),
                    "samples": list(samples),
                }
            )
        return {
            "measurement_mode": "grace_hwmon_power_integration",
            "integration_rule": "trapezoidal",
            "timestamp_clock": "monotonic",
            "timestamp_location": "sensor_read_midpoint",
            "poll_interval_s": self.poll_interval_s,
            "calibrated": False,
            "scientific_use": False,
            "valid": all(reading["valid"] for reading in readings),
            "nested_domains_additive": False,
            "nvml_energy_included": False,
            "sensor_integrals": readings,
        }


class GraceProcessSampler:
    """Poll outside the workload interpreter so its GIL cannot suspend sampling.

    The worker executes this stdlib-only file directly, without importing Torch
    or forking a CUDA process. Startup completes before the first sensor sample
    and readiness acknowledgement. ``stop()`` drains the trace while waiting for
    exit, including traces larger than a pipe buffer. Validation stays with the
    caller, including the unchanged maximum accepted sampling gap.
    """

    def __init__(
        self,
        sensors: Sequence[GracePowerSensor],
        *,
        poll_interval_s: float = 0.1,
    ) -> None:
        # Reuse the same argument contract without starting a local thread.
        checked = GracePowerSampler(sensors, poll_interval_s=poll_interval_s)
        self.sensors = checked.sensors
        self.poll_interval_s = checked.poll_interval_s
        self._process: subprocess.Popen[str] | None = None
        self._started = False
        self._stopped = False
        self._timeout_s = 10.0

    def _reap(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Grace sampler has already been started")
        self._started = True
        try:
            process = self._process = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--sampler-worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdin is not None and process.stdout is not None
            config = {
                "poll_interval_s": self.poll_interval_s,
                "sensors": [
                    {
                        "label": sensor.label,
                        "socket": sensor.socket,
                        "domain": sensor.domain,
                        "power_path": str(sensor.power_path),
                        "averaging_interval_s": sensor.averaging_interval_s,
                    }
                    for sensor in self.sensors
                ],
            }
            process.stdin.write(json.dumps(config) + "\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], self._timeout_s)
            if not ready:
                raise RuntimeError("Grace sampling worker startup timed out")
            acknowledgement = process.stdout.readline()
            if acknowledgement != '{"status": "ready"}\n':
                raise RuntimeError("Grace sampling worker failed before readiness")
        except BaseException:
            self._reap()
            raise

    def stop(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Grace sampler has not been started")
        if self._stopped:
            raise RuntimeError("Grace sampler has already been stopped")
        self._stopped = True
        process = self._process
        if process is None:
            raise RuntimeError("Grace sampling worker did not start")
        try:
            stdout, stderr = process.communicate("stop\n", timeout=self._timeout_s)
            if process.returncode:
                raise RuntimeError(
                    f"Grace sampling worker failed ({process.returncode}): {stderr.strip()}"
                )
            try:
                result = json.loads(stdout)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Grace sampling worker returned no valid trace") from exc
            if not isinstance(result, dict):
                raise RuntimeError("Grace sampling worker returned no valid trace")
            return result
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Grace sampling worker stop timed out") from exc
        finally:
            self._reap()


def _sampler_worker() -> None:
    # The launcher terminates the whole workload process group. Keep this child
    # alive until the parent closes the measurement, or exits and sends EOF.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    config = json.loads(sys.stdin.readline())
    sensors = [
        GracePowerSensor(**{**sensor, "power_path": Path(sensor["power_path"])})
        for sensor in config["sensors"]
    ]
    sampler = GracePowerSampler(sensors, poll_interval_s=config["poll_interval_s"])
    sampler.start()
    # Failed initial reads must stop the run before expensive workload starts.
    if any(integral.errors for integral in sampler._integrals.values()):
        sampler.stop()
        raise RuntimeError("Grace sampling worker could not read its initial sensor values")
    print('{"status": "ready"}', flush=True)
    command = sys.stdin.readline()
    result = sampler.stop()
    if not command:  # Parent exited or closed its input: do not leave a worker behind.
        return
    if command != "stop\n":
        raise RuntimeError("unexpected Grace sampling worker command")
    result["sampling_execution"] = "independent_process"
    result["worker_pid"] = os.getpid()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--sampler-worker"]:
        raise SystemExit("This module's command line is reserved for its sampling worker")
    _sampler_worker()
