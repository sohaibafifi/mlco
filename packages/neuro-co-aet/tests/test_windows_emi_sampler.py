"""Tests for native Windows EMI CPU package energy measurement."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from neuro_co.aet import EnergyTracker
from neuro_co.aet.energy.windows_emi import (
    WINDOWS_EMI_CPU_PACKAGE,
    WindowsEmiSampler,
)


def test_windows_emi_sampler_reads_cpu_package_energy_without_dram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _WindowsEmiFixture:
        def __init__(self, *, emi_include_dram: bool) -> None:
            captured["emi_include_dram"] = emi_include_dram
            self._devices = [("private-device-path", ["RAPL_Package0_PKG"], [0])]

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def get_cpu_details(duration: object) -> dict[str, float]:
            captured["duration"] = duration
            return {
                "Processor Energy Delta_0(kWh)": 0.001,
                "Processor Power Delta_0(kWh)": 80.0,
                "Processor Energy Delta_1(kWh)": 0.002,
            }

    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.sys.platform", "win32")
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.is_emi_available", lambda: True)
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.WindowsEMI", _WindowsEmiFixture)
    sampler = WindowsEmiSampler()

    sampler.start()
    energy_wh = sampler.stop()

    assert energy_wh == pytest.approx(3.0)
    assert captured["emi_include_dram"] is False
    assert getattr(captured["duration"], "seconds") >= 0.0
    assert sampler.measurement_metadata["measurement_mode"] == WINDOWS_EMI_CPU_PACKAGE
    assert sampler.measurement_metadata["includes_dram"] is False
    assert sampler.measurement_metadata["source"] == "codecarbon"
    assert sampler.measurement_metadata["initial_selected_channel_count"] == 1
    assert sampler.measurement_metadata["last_measured_channel_count"] == 2
    assert sampler.measurement_metadata["attribution_scope"] == "machine_wide_cpu_package"
    assert sampler.measurement_metadata["exclusive_host_required"] is True
    assert sampler.available is True


def test_windows_emi_sampler_rejects_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.sys.platform", "linux")

    with pytest.raises(RuntimeError, match="native Windows"):
        WindowsEmiSampler()


def test_windows_emi_sampler_rejects_ambiguous_nonpackage_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WindowsEmiFixture:
        def __init__(self, *, emi_include_dram: bool) -> None:
            self._devices = [
                (
                    "private-device-path",
                    ["RAPL_Package0_Core0_CORE", "RAPL_Package0_DRAM"],
                    [0],
                )
            ]

    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.sys.platform", "win32")
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.is_emi_available", lambda: True)
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.WindowsEMI", _WindowsEmiFixture)

    with pytest.raises(RuntimeError, match="package-only"):
        WindowsEmiSampler()


def test_windows_emi_sampler_rejects_unvalidated_codecarbon_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WindowsEmiFixture:
        def __init__(self, *, emi_include_dram: bool) -> None:
            self._devices = [("private-device-path", ["RAPL_Package0_PKG"], [0])]

    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.sys.platform", "win32")
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.is_emi_available", lambda: True)
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.WindowsEMI", _WindowsEmiFixture)
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.version", lambda _name: "3.4.0")

    with pytest.raises(RuntimeError, match=r"3\.3\.1.*3\.4\.0"):
        WindowsEmiSampler()


def test_windows_emi_sampler_requires_exact_energy_key_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WindowsEmiFixture:
        def __init__(self, *, emi_include_dram: bool) -> None:
            self._devices = [("private-device-path", ["RAPL_Package0_PKG"], [0])]

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def get_cpu_details(_duration: object) -> dict[str, float]:
            return {"Processor Energy Delta_0(J)": 5.0}

    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.sys.platform", "win32")
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.is_emi_available", lambda: True)
    monkeypatch.setattr("neuro_co.aet.energy.windows_emi.WindowsEMI", _WindowsEmiFixture)
    sampler = WindowsEmiSampler()
    sampler.start()

    with pytest.raises(RuntimeError, match="no CPU package energy"):
        sampler.stop()


def test_tracker_combines_nvml_and_windows_emi_without_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NvmlFixture:
        device_metadata: ClassVar[list[dict[str, object]]] = [
            {"index": 0, "measurement_mode": "nvml_total_energy_counter"}
        ]

        def __init__(self, **_kwargs: object) -> None:
            return None

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def stop() -> float:
            return 2.0

    class _WindowsEmiFixture:
        measurement_metadata: ClassVar[dict[str, object]] = {
            "measurement_mode": WINDOWS_EMI_CPU_PACKAGE,
            "source": "codecarbon",
            "codecarbon_version": "3.3.1",
            "includes_dram": False,
            "initial_selected_channel_count": 1,
            "last_measured_channel_count": 1,
            "attribution_scope": "machine_wide_cpu_package",
            "exclusive_host_required": True,
        }

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def stop() -> float:
            return 3.0

    monkeypatch.setattr("neuro_co.aet.energy.tracker.sys.platform", "win32")
    monkeypatch.setattr("neuro_co.aet.energy.tracker.NvmlSampler", _NvmlFixture)
    monkeypatch.setattr("neuro_co.aet.energy.tracker.WindowsEmiSampler", _WindowsEmiFixture)
    tracker = EnergyTracker(
        backend="hwcounters",
        required_domains={"gpu", "cpu"},
        allow_fallback=False,
        pue=1.0,
        report_embodied=False,
    )

    with tracker:
        pass

    assert tracker.reading is not None
    record = tracker.reading.to_dict()
    assert record["energy_gpu_j"] == pytest.approx(2.0 * 3600.0)
    assert record["energy_cpu_j"] == pytest.approx(3.0 * 3600.0)
    assert record["energy_dram_j"] == 0.0
    assert record["energy_domains"] == ["gpu", "cpu"]
    assert record["hardware"]["cpu_measurement"] == _WindowsEmiFixture.measurement_metadata
    assert record["hardware"]["attribution_scope"] == "machine_wide_component_counters"
    assert record["hardware"]["exclusive_host_required"] is True


def test_required_windows_cpu_domain_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(gpu_stopped=False)

    class _NvmlFixture:
        def __init__(self, **_kwargs: object) -> None:
            return None

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def stop() -> float:
            state.gpu_stopped = True
            return 0.0

    class _UnavailableWindowsEmiFixture:
        def __init__(self) -> None:
            raise RuntimeError("EMI unavailable")

    monkeypatch.setattr("neuro_co.aet.energy.tracker.sys.platform", "win32")
    monkeypatch.setattr("neuro_co.aet.energy.tracker.NvmlSampler", _NvmlFixture)
    monkeypatch.setattr(
        "neuro_co.aet.energy.tracker.WindowsEmiSampler",
        _UnavailableWindowsEmiFixture,
    )
    tracker = EnergyTracker(
        backend="hwcounters",
        required_domains={"cpu"},
        allow_fallback=True,
        report_embodied=False,
    )

    with pytest.raises(RuntimeError, match=r"required.*cpu"):
        tracker._start_backend()

    assert tracker._backend_used == "none"
    assert tracker._tdp_fallback is False
    assert state.gpu_stopped is True


def test_hardware_samplers_are_constructed_before_adjacent_reverse_ordered_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _NvmlFixture:
        device_metadata: ClassVar[list[dict[str, object]]] = []

        def __init__(self, **_kwargs: object) -> None:
            events.append("gpu:init")

        @staticmethod
        def start() -> None:
            events.append("gpu:start")

        @staticmethod
        def stop() -> float:
            events.append("gpu:stop")
            return 2.0

        @staticmethod
        def close() -> None:
            events.append("gpu:close")

    class _WindowsEmiFixture:
        measurement_metadata: ClassVar[dict[str, object]] = {
            "measurement_mode": WINDOWS_EMI_CPU_PACKAGE,
            "includes_dram": False,
        }

        def __init__(self) -> None:
            events.append("cpu:init")

        @staticmethod
        def start() -> None:
            events.append("cpu:start")

        @staticmethod
        def stop() -> float:
            events.append("cpu:stop")
            return 3.0

    monkeypatch.setattr("neuro_co.aet.energy.tracker.sys.platform", "win32")
    monkeypatch.setattr("neuro_co.aet.energy.tracker.NvmlSampler", _NvmlFixture)
    monkeypatch.setattr("neuro_co.aet.energy.tracker.WindowsEmiSampler", _WindowsEmiFixture)
    tracker = EnergyTracker(
        backend="hwcounters",
        required_domains={"gpu", "cpu"},
        allow_fallback=False,
        report_embodied=False,
    )

    with tracker:
        events.append("workload")

    assert events == [
        "gpu:init",
        "cpu:init",
        "cpu:start",
        "gpu:start",
        "workload",
        "gpu:stop",
        "cpu:stop",
    ]


def test_hardware_stop_attempts_cpu_cleanup_after_gpu_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_stopped = False

    class _BrokenNvmlFixture:
        @staticmethod
        def stop() -> float:
            raise RuntimeError("GPU counter failure")

    class _WindowsEmiFixture:
        @staticmethod
        def stop() -> float:
            nonlocal cpu_stopped
            cpu_stopped = True
            return 1.0

    tracker = EnergyTracker(backend="hwcounters", report_embodied=False)
    tracker._backend_used = "hwcounters"
    tracker._nvml = _BrokenNvmlFixture()  # type: ignore[assignment]
    tracker._windows_emi = _WindowsEmiFixture()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="NVML"):
        tracker._stop_backend(duration=1.0)

    assert cpu_stopped is True
