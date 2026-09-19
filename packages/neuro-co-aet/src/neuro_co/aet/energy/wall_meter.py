"""Strict contract for whole-system AC wall-meter measurements.

Each device adapter must implement its meter's communication protocol and return a validated
``WallMeterResult`` from ``stop()``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _finite_number(value: object, *, name: str, strictly_positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and number <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    if not strictly_positive and number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validated_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _parse_aware_datetime(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty ISO 8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be expressed in UTC")
    return parsed


@dataclass(frozen=True)
class WallMeterResult:
    """Validated result returned by a physical wall-meter sampler.

    ``energy_j`` is raw whole-system AC energy. ``duration_s`` is the measured
    boundary duration and ``sample_interval_s`` is the meter sampling period.
    The trace and device identifiers are hashes so the canonical result does
    not disclose a serial number or raw device identity.
    """

    energy_j: float
    duration_s: float
    sample_interval_s: float
    sample_count: int
    dropped_sample_count: int
    trace_sha256: str
    calibration_id: str
    device_id_sha256: str
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    start_monotonic_s: float | None = None
    end_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        energy_j = _finite_number(self.energy_j, name="energy_j")
        duration_s = _finite_number(
            self.duration_s,
            name="duration_s",
            strictly_positive=True,
        )
        sample_interval_s = _finite_number(
            self.sample_interval_s,
            name="sample_interval_s",
            strictly_positive=True,
        )
        if sample_interval_s > duration_s:
            raise ValueError("sample_interval_s must not exceed duration_s")
        sample_count = _nonnegative_integer(self.sample_count, name="sample_count")
        dropped_sample_count = _nonnegative_integer(
            self.dropped_sample_count,
            name="dropped_sample_count",
        )
        if sample_count == 0:
            raise ValueError("sample_count must be greater than zero")
        trace_sha256 = _validated_sha256(self.trace_sha256, name="trace_sha256")
        device_id_sha256 = _validated_sha256(
            self.device_id_sha256,
            name="device_id_sha256",
        )
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("calibration_id must be a non-empty string")
        calibration_id = self.calibration_id.strip()

        if (self.started_at_utc is None) != (self.ended_at_utc is None):
            raise ValueError("started_at_utc and ended_at_utc must be provided together")
        if self.started_at_utc is not None and self.ended_at_utc is not None:
            start_utc = _parse_aware_datetime(self.started_at_utc, name="started_at_utc")
            end_utc = _parse_aware_datetime(self.ended_at_utc, name="ended_at_utc")
            if end_utc < start_utc:
                raise ValueError("ended_at_utc must not precede started_at_utc")

        if (self.start_monotonic_s is None) != (self.end_monotonic_s is None):
            raise ValueError("start_monotonic_s and end_monotonic_s must be provided together")
        start_monotonic_s = self.start_monotonic_s
        end_monotonic_s = self.end_monotonic_s
        if start_monotonic_s is not None and end_monotonic_s is not None:
            start_monotonic_s = _finite_number(
                start_monotonic_s,
                name="start_monotonic_s",
            )
            end_monotonic_s = _finite_number(
                end_monotonic_s,
                name="end_monotonic_s",
            )
            monotonic_interval_s = end_monotonic_s - start_monotonic_s
            if monotonic_interval_s <= 0.0:
                raise ValueError("end_monotonic_s must be greater than start_monotonic_s")
            tolerance_s = max(1e-6, duration_s * 1e-6)
            if not math.isclose(
                monotonic_interval_s,
                duration_s,
                rel_tol=1e-6,
                abs_tol=tolerance_s,
            ):
                raise ValueError("duration_s must match the monotonic measurement bounds")

        object.__setattr__(self, "energy_j", energy_j)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "sample_interval_s", sample_interval_s)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "dropped_sample_count", dropped_sample_count)
        object.__setattr__(self, "trace_sha256", trace_sha256)
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "device_id_sha256", device_id_sha256)
        object.__setattr__(self, "start_monotonic_s", start_monotonic_s)
        object.__setattr__(self, "end_monotonic_s", end_monotonic_s)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-compatible measurement metadata without raw identities."""
        return {
            "duration_s": self.duration_s,
            "sample_interval_s": self.sample_interval_s,
            "sample_count": self.sample_count,
            "dropped_sample_count": self.dropped_sample_count,
            "trace_sha256": self.trace_sha256,
            "calibration_id": self.calibration_id,
            "device_id_sha256": self.device_id_sha256,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "start_monotonic_s": self.start_monotonic_s,
            "end_monotonic_s": self.end_monotonic_s,
        }


@runtime_checkable
class WallMeterSampler(Protocol):
    """Injectable adapter contract for an externally calibrated AC meter."""

    def start(self) -> None: ...

    def stop(self) -> WallMeterResult: ...
