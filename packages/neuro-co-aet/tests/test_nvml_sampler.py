"""Tests for explicit NVML device locking and artifact-safe metadata."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest

from neuro_co.aet import EnergyTracker
from neuro_co.aet.energy.nvml import (
    NVML_POWER_INTEGRATION,
    NVML_TOTAL_ENERGY_COUNTER,
    NvmlDeviceSelectionError,
    NvmlSampler,
)


class _FakeNvml(SimpleNamespace):
    def __init__(self, *, counter_indices: set[int] | None = None) -> None:
        super().__init__()
        self.uuids = [b"GPU-private-zero", b"GPU-private-one"]
        self.counter_indices = counter_indices or set()
        self.energy_mj = {index: 0 for index in range(len(self.uuids))}
        self.shutdown_calls = 0

    @staticmethod
    def nvmlInit() -> None:
        return None

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self) -> int:
        return len(self.uuids)

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> int:
        return index

    def nvmlDeviceGetUUID(self, handle: int) -> bytes:
        return self.uuids[handle]

    def nvmlDeviceGetTotalEnergyConsumption(self, handle: int) -> int:
        if handle not in self.counter_indices:
            raise RuntimeError("counter unsupported")
        self.energy_mj[handle] += 3_600_000
        return self.energy_mj[handle]

    @staticmethod
    def nvmlDeviceGetPowerUsage(_handle: int) -> int:
        return 180_000


def _install_fake_nvml(
    monkeypatch: pytest.MonkeyPatch,
    *,
    counter_indices: set[int] | None = None,
) -> _FakeNvml:
    fake = _FakeNvml(counter_indices=counter_indices)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


def test_index_selection_preserves_order_and_exposes_per_device_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_nvml(monkeypatch, counter_indices={0})
    sampler = NvmlSampler(gpu_indices=[1, 0], poll_hz=1000.0)

    assert sampler.gpu_indices == [1, 0]
    assert sampler.measurement_modes == (
        NVML_POWER_INTEGRATION,
        NVML_TOTAL_ENERGY_COUNTER,
    )
    assert sampler.device_metadata == [
        {
            "index": 1,
            "device_id_sha256": hashlib.sha256(b"GPU-private-one").hexdigest(),
            "measurement_mode": NVML_POWER_INTEGRATION,
            "attribution_scope": "physical_device_all_processes",
            "exclusive_device_required": True,
            "sampling_interval_s": pytest.approx(0.001),
            "sample_count": 0,
            "dropped_sample_count": 0,
        },
        {
            "index": 0,
            "device_id_sha256": hashlib.sha256(b"GPU-private-zero").hexdigest(),
            "measurement_mode": NVML_TOTAL_ENERGY_COUNTER,
            "attribution_scope": "physical_device_all_processes",
            "exclusive_device_required": True,
            "sampling_interval_s": None,
            "sample_count": 0,
            "dropped_sample_count": 0,
        },
    ]
    assert "GPU-private" not in json.dumps(sampler.device_metadata)

    sampler.start()
    time.sleep(0.005)
    assert sampler.stop() >= 1.0
    assert sampler.device_metadata[0]["sample_count"] >= 1
    assert sampler.device_metadata[0]["dropped_sample_count"] >= 0
    assert sampler.device_metadata[1]["sample_count"] == 2
    assert fake.shutdown_calls == 1


def test_uuid_selection_is_case_insensitive_but_metadata_hashes_physical_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_nvml(monkeypatch, counter_indices={1})
    sampler = NvmlSampler(gpu_uuids=["gpu-PRIVATE-one"])

    assert sampler.gpu_indices == [1]
    assert (
        sampler.device_metadata[0]["device_id_sha256"]
        == hashlib.sha256(b"GPU-private-one").hexdigest()
    )
    sampler.start()
    assert sampler.stop() == pytest.approx(1.0)
    assert fake.shutdown_calls == 1


def test_missing_uuid_fails_strictly_without_leaking_raw_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_nvml(monkeypatch)
    raw_uuid = "GPU-secret-not-present"

    with pytest.raises(NvmlDeviceSelectionError) as caught:
        NvmlSampler(gpu_uuids=[raw_uuid])

    assert raw_uuid not in str(caught.value)
    assert hashlib.sha256(raw_uuid.encode()).hexdigest()[:12] in str(caught.value)
    assert fake.shutdown_calls == 1


def test_missing_index_fails_strictly_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_nvml(monkeypatch)

    with pytest.raises(NvmlDeviceSelectionError, match="index not found: 7"):
        NvmlSampler(gpu_indices=[7])

    assert fake.shutdown_calls == 1


def test_sampler_rejects_mutually_exclusive_or_duplicate_selectors() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        NvmlSampler(gpu_indices=[0], gpu_uuids=["GPU-private-zero"])
    with pytest.raises(ValueError, match="duplicates"):
        NvmlSampler(gpu_indices=[0, 0])
    with pytest.raises(ValueError, match="duplicates"):
        NvmlSampler(gpu_uuids=["GPU-a", "gpu-A"])


def test_tracker_passes_selection_and_serializes_only_hashed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _NvmlFixture:
        device_metadata: ClassVar[list[dict[str, object]]] = [
            {
                "index": 1,
                "device_id_sha256": "f" * 64,
                "measurement_mode": NVML_TOTAL_ENERGY_COUNTER,
            }
        ]

        def __init__(
            self,
            gpu_indices: tuple[int, ...] | None = None,
            gpu_uuids: tuple[str, ...] | None = None,
        ) -> None:
            captured["gpu_indices"] = gpu_indices
            captured["gpu_uuids"] = gpu_uuids

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def stop() -> float:
            return 0.25

    class _UnavailableRaplFixture:
        def __init__(self) -> None:
            raise RuntimeError("RAPL unavailable")

    monkeypatch.setattr("neuro_co.aet.energy.tracker.NvmlSampler", _NvmlFixture)
    monkeypatch.setattr("neuro_co.aet.energy.tracker.RaplSampler", _UnavailableRaplFixture)
    raw_uuid = "GPU-private-selected"
    tracker = EnergyTracker(
        backend="hwcounters",
        allow_fallback=False,
        required_domains={"gpu"},
        gpu_uuids=[raw_uuid],
        pue=1.0,
        report_embodied=False,
    )

    with tracker:
        pass

    assert captured == {"gpu_indices": None, "gpu_uuids": (raw_uuid,)}
    assert tracker.reading is not None
    record = tracker.reading.to_dict()
    assert record["hardware"]["gpu_devices"] == _NvmlFixture.device_metadata
    assert raw_uuid not in json.dumps(record)


def test_tracker_never_falls_back_after_explicit_selection_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rapl_constructed = False

    class _MissingNvmlFixture:
        def __init__(self, **_kwargs: object) -> None:
            raise NvmlDeviceSelectionError("selected GPU not found")

    class _RaplFixture:
        def __init__(self) -> None:
            nonlocal rapl_constructed
            rapl_constructed = True

    monkeypatch.setattr("neuro_co.aet.energy.tracker.NvmlSampler", _MissingNvmlFixture)
    monkeypatch.setattr("neuro_co.aet.energy.tracker.RaplSampler", _RaplFixture)
    tracker = EnergyTracker(
        backend="hwcounters",
        gpu_indices=[0],
        allow_fallback=True,
        report_embodied=False,
    )

    with pytest.raises(NvmlDeviceSelectionError, match="not found"):
        tracker._try_start_hwcounters()

    assert rapl_constructed is False


def test_tracker_never_falls_back_when_explicit_selection_cannot_be_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rapl_constructed = False

    class _UnavailableNvmlFixture:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("NVML driver unavailable")

    class _RaplFixture:
        def __init__(self) -> None:
            nonlocal rapl_constructed
            rapl_constructed = True

    monkeypatch.setattr("neuro_co.aet.energy.tracker.NvmlSampler", _UnavailableNvmlFixture)
    monkeypatch.setattr("neuro_co.aet.energy.tracker.RaplSampler", _RaplFixture)
    tracker = EnergyTracker(
        backend="hwcounters",
        gpu_indices=[0],
        allow_fallback=True,
        report_embodied=False,
    )

    with pytest.raises(RuntimeError, match="explicit NVML GPU selection"):
        tracker._try_start_hwcounters()

    assert rapl_constructed is False


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"gpu_indices": [0], "gpu_uuids": ["GPU-a"]}, "mutually exclusive"),
        ({"gpu_indices": []}, "must not be empty"),
        ({"gpu_indices": [-1]}, "non-negative"),
        ({"gpu_indices": [True]}, "only integers"),
        ({"gpu_uuids": "GPU-a"}, "collection of strings"),
        ({"gpu_uuids": [" "]}, "non-empty strings"),
        ({"backend": "codecarbon", "gpu_indices": [0]}, "only supported"),
    ],
)
def test_tracker_rejects_invalid_gpu_selection(
    kwargs: dict[str, object],
    error: str,
) -> None:
    values: dict[str, object] = {"backend": "hwcounters"}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error):
        EnergyTracker(**values)  # type: ignore[arg-type]
