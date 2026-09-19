"""Regression tests for the non-executing AET host preflight."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neuro_co.aet.experiments import preflight


def _calibration_payload() -> dict[str, Any]:
    workload_classes = {}
    for name in ("idle", "cpu_only", "gpu_only", "combined"):
        workload_classes[name] = [
            {
                "duration_s": 120.0,
                "sample_count": 120,
                "raw_sha256": hashlib.sha256(f"{name}:{index}".encode()).hexdigest(),
            }
            for index in range(5)
        ]
    return {
        "schema_version": "aet-calibration/v1",
        "calibration_id": "cal-win-a4500-01",
        "status": "passed",
        "host_id": "win-a4500-01",
        "execution_layer": "wsl2",
        "created_at": "2026-08-21T00:00:00+00:00",
        "hardware": {
            "cpu": "Intel Core i7-13700",
            "logical_cpu_count": 24,
            "gpu": "NVIDIA RTX A4500",
            "gpu_index": 0,
            "gpu_device_id_sha256": "e" * 64,
            "ram_bytes": 64_000_000_000,
            "windows_build": "26100",
            "wsl_version": "2.5.10",
            "wsl_kernel": "6.6.87.2-microsoft-standard-WSL2",
            "nvidia_driver": "580.0",
            "cuda_version": "13.0",
        },
        "wall_meter": {
            "required": True,
            "model": "qualified-meter",
            "device_id_hash": "b" * 64,
            "firmware": "1.0",
            "accuracy_percent": 1.0,
            "energy_resolution_j": 1.0,
            "sample_interval_s": 1.0,
            "data_interface": "csv",
            "calibration_reference": "traceable-reference",
        },
        "counter_backends": {
            "system_primary": "wall_meter",
            "gpu_diagnostic": "nvml",
            "cpu_diagnostic": None,
            "strict_backend": True,
            "pue": 1.0,
            "report_embodied": False,
        },
        "timing": {
            "timestamp_alignment_error_s": 0.1,
            "missing_sample_percent": 0.0,
        },
        "criteria": {
            "max_meter_accuracy_percent": 2.0,
            "max_sample_interval_s": 1.0,
            "max_timestamp_alignment_error_s": 0.5,
            "max_missing_sample_percent": 0.5,
            "max_repeat_cv_percent": 5.0,
            "max_coverage_ratio_cv_percent": 5.0,
            "minimum_block_duration_s": 120.0,
            "minimum_samples_per_block": 100,
            "repeats_per_workload_class": 5,
        },
        "workload_classes": workload_classes,
        "summary": {
            "meter_repeat_cv_percent": 1.0,
            "coverage_ratio_cv_percent": 2.0,
            "nvml_positive_and_monotonic": True,
            "rapl_available": False,
            "prohibited_fallback_observed": False,
        },
        "evidence": {
            "raw_trace_manifest": "raw-traces.json",
            "raw_trace_manifest_sha256": "c" * 64,
            "analysis_script": "analyze-calibration.py",
            "analysis_script_sha256": "d" * 64,
        },
        "approval": {
            "passed_at": "2026-08-21T00:01:00+00:00",
            "approved_by": "author",
            "notes": None,
        },
    }


def _materialize_calibration_evidence(path: Path, payload: dict[str, Any]) -> None:
    traces = []
    for workload_name, records in payload["workload_classes"].items():
        for index, record in enumerate(records):
            raw_trace = Path("traces") / f"{workload_name}-{index}.jsonl"
            raw_trace_path = path.parent / raw_trace
            raw_trace_path.parent.mkdir(parents=True, exist_ok=True)
            raw_trace_path.write_text(
                json.dumps({"workload_class": workload_name, "block_index": index}) + "\n",
                encoding="utf-8",
            )
            raw_sha256 = hashlib.sha256(raw_trace_path.read_bytes()).hexdigest()
            record["raw_sha256"] = raw_sha256
            traces.append(
                {
                    "workload_class": workload_name,
                    "block_index": index,
                    "raw_trace": raw_trace.as_posix(),
                    "raw_sha256": raw_sha256,
                }
            )
    raw_manifest_path = path.parent / payload["evidence"]["raw_trace_manifest"]
    raw_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "aet-raw-trace-manifest/v1",
                "traces": traces,
            }
        ),
        encoding="utf-8",
    )
    payload["evidence"]["raw_trace_manifest_sha256"] = hashlib.sha256(
        raw_manifest_path.read_bytes()
    ).hexdigest()
    analysis_script_path = path.parent / payload["evidence"]["analysis_script"]
    analysis_script_path.write_text("# calibration analysis fixture\n", encoding="utf-8")
    payload["evidence"]["analysis_script_sha256"] = hashlib.sha256(
        analysis_script_path.read_bytes()
    ).hexdigest()


def test_calibration_probe_requires_complete_evidence(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    payload = _calibration_payload()
    _materialize_calibration_evidence(path, payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    verified_evidence_files: set[Path] = set()

    result = preflight.calibration_probe(
        path,
        verified_evidence_files=verified_evidence_files,
    )

    assert result["passed"] is True
    assert result["host_id"] == "win-a4500-01"
    assert len(verified_evidence_files) == 22
    assert tmp_path / "raw-traces.json" in verified_evidence_files
    assert tmp_path / "analyze-calibration.py" in verified_evidence_files
    payload["workload_classes"]["gpu_only"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert preflight.calibration_probe(path)["passed"] is False


def test_calibration_probe_fails_closed_when_trace_manifest_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    payload = _calibration_payload()
    _materialize_calibration_evidence(path, payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "raw-traces.json").unlink()

    result = preflight.calibration_probe(path)

    assert result["passed"] is False
    assert "cannot be read" in result["validation_error"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["criteria"].update({"repeats_per_workload_class": 0}),
        lambda value: value["workload_classes"].update({"cpu_only": []}),
        lambda value: value["workload_classes"]["gpu_only"][0].update({"duration_s": -120.0}),
        lambda value: value["wall_meter"].update({"accuracy_percent": float("inf")}),
        lambda value: value["evidence"].update({"analysis_script_sha256": "fake"}),
        lambda value: value["hardware"].update({"nvidia_driver": ""}),
        lambda value: value["criteria"].update({"max_sample_interval_s": 2.0}),
        lambda value: value["approval"].update({"passed_at": "2026-08-21T00:01:00"}),
        lambda value: value["counter_backends"].update({"strict_backend": False}),
    ],
)
def test_calibration_probe_uses_closed_shared_validator(
    mutate: Any,
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    payload = _calibration_payload()
    _materialize_calibration_evidence(path, payload)
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = preflight.calibration_probe(path)

    assert result["passed"] is False
    assert result["validation_error"]


def test_git_snapshot_excludes_generated_preflight_from_cleanliness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "preflight.json"
    report.write_text("generated", encoding="utf-8")

    def fake_command(*args: str, **_kwargs: Any) -> str | bytes:
        if args[:2] == ("git", "rev-parse"):
            return "a" * 40 + "\n"
        if args[:2] == ("git", "branch"):
            return "main\n"
        if args[:2] == ("git", "diff"):
            return b""
        if args[:2] == ("git", "ls-files"):
            return b"preflight.json\0"
        raise AssertionError(args)

    monkeypatch.setattr(preflight, "command", fake_command)

    result = preflight.git_snapshot(tmp_path, {report.resolve()})

    assert result["available"] is True
    assert result["dirty"] is False
    assert result["untracked_file_count"] == 0


def test_git_snapshot_excludes_exact_evidence_but_rejects_other_untracked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "preflight.json"
    trace = tmp_path / "traces" / "idle-0.jsonl"
    report.write_text("generated", encoding="utf-8")
    trace.parent.mkdir()
    trace.write_text("evidence", encoding="utf-8")
    untracked = [b"preflight.json\0traces/idle-0.jsonl\0"]

    def fake_command(*args: str, **_kwargs: Any) -> str | bytes:
        if args[:2] == ("git", "rev-parse"):
            return "a" * 40 + "\n"
        if args[:2] == ("git", "branch"):
            return "main\n"
        if args[:2] == ("git", "diff"):
            return b""
        if args[:2] == ("git", "ls-files"):
            return untracked[0]
        raise AssertionError(args)

    monkeypatch.setattr(preflight, "command", fake_command)

    assert preflight.git_snapshot(tmp_path, {report.resolve(), trace.resolve()})["dirty"] is False
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("other", encoding="utf-8")
    untracked[0] += b"unrelated.txt\0"
    assert preflight.git_snapshot(tmp_path, {report.resolve(), trace.resolve()})["dirty"] is True


def _fake_pynvml() -> SimpleNamespace:
    def total_energy(handle: int) -> int:
        if handle == 1:
            raise RuntimeError("counter unavailable")
        return 10_000

    return SimpleNamespace(
        NVML_CLOCK_GRAPHICS=0,
        NVML_CLOCK_MEM=2,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlSystemGetDriverVersion=lambda: b"999.1",
        nvmlSystemGetCudaDriverVersion=lambda: 12040,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: f"GPU-{handle}".encode(),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=16_000 + handle),
        nvmlDeviceGetUUID=lambda handle: f"private-uuid-{handle}".encode(),
        nvmlDeviceGetPowerManagementLimit=lambda handle: 200_000,
        nvmlDeviceGetPersistenceMode=lambda handle: 1,
        nvmlDeviceGetClockInfo=lambda handle, domain: 1_500 + domain,
        nvmlDeviceGetTotalEnergyConsumption=total_energy,
        nvmlDeviceGetPowerUsage=lambda handle: 50_000,
    )


def test_nvml_probe_selects_one_device_without_exposing_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())

    result = preflight.nvml_probe(active=False, selected_index=1)

    assert result["selection_valid"] is True
    assert result["selected_device"]["index"] == 1
    assert result["selected_device"]["measurement_mode"] == "power_integration"
    assert result["selected_device"]["power_sampling_supported"] is True
    assert "private-uuid" not in json.dumps(result)
    assert len(result["selected_device"]["device_id_sha256"]) == 64


def test_nvml_probe_rejects_unknown_selected_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())

    result = preflight.nvml_probe(active=False, selected_index=7)

    assert result["available"] is True
    assert result["selection_valid"] is False
    assert result["selected_device"] is None


def test_strict_tracker_probe_requires_wall_meter_contract() -> None:
    result = preflight.strict_tracker_probe()

    assert result["supported"] is True
    assert result["wall_meter_contract_supported"] is True
    assert result["component_contract_supported"] is True
    assert result["missing_parameters"] == []


def test_windows_emi_version_gate_accepts_only_pinned_release() -> None:
    assert preflight.version_is_exact("3.3.1", "3.3.1") is True


@pytest.mark.parametrize(
    "version",
    [None, "3.2.7", "3.3.0", "3.3.1rc1", "3.3.2", "4.0.0", "unknown"],
)
def test_windows_emi_version_gate_rejects_every_other_release(
    version: str | None,
) -> None:
    assert preflight.version_is_exact(version, "3.3.1") is False


def test_windows_emi_probe_requires_native_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight.sys, "platform", "linux")
    monkeypatch.setattr(preflight, "distribution_version", lambda _name: "3.3.1")

    result = preflight.windows_emi_probe(active=True)

    assert result["available"] is False
    assert result["platform_windows"] is False


def test_windows_emi_probe_rejects_unpinned_codecarbon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight.sys, "platform", "win32")
    monkeypatch.setattr(preflight, "distribution_version", lambda _name: "3.3.2")

    result = preflight.windows_emi_probe(active=True)

    assert result["available"] is False
    assert result["version_supported"] is False
    assert result["required_codecarbon_version"] == "3.3.1"


@pytest.mark.parametrize(
    ("sample_energy_wh", "expected_available"),
    [(0.001, True), (0.0, False)],
)
def test_windows_emi_active_probe_requires_positive_energy(
    sample_energy_wh: float,
    expected_available: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WindowsEmiFixture:
        available = True

        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def stop() -> float:
            return sample_energy_wh

    times = iter([0.0, 0.1, 0.3])
    monkeypatch.setattr(preflight.sys, "platform", "win32")
    monkeypatch.setattr(preflight, "distribution_version", lambda _name: "3.3.1")
    monkeypatch.setattr(preflight.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(
        "neuro_co.aet.energy.windows_emi.WindowsEmiSampler",
        _WindowsEmiFixture,
    )

    result = preflight.windows_emi_probe(active=True)

    assert result["available"] is expected_available
    assert result["counter_positive"] is expected_available
    assert result["ram_included"] is False


def _qualified_pynvml(
    uuids: tuple[str, ...],
    energy_sequences: dict[int, list[int]],
    events: list[str],
) -> SimpleNamespace:
    energy_offsets = {index: 0 for index in range(len(uuids))}

    def total_energy(handle: int) -> int:
        events.append(f"energy:{handle}")
        values = energy_sequences[handle]
        offset = energy_offsets[handle]
        energy_offsets[handle] += 1
        return values[min(offset, len(values) - 1)]

    return SimpleNamespace(
        NVML_CLOCK_GRAPHICS=0,
        NVML_CLOCK_MEM=2,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlSystemGetDriverVersion=lambda: b"999.1",
        nvmlSystemGetCudaDriverVersion=lambda: 12040,
        nvmlDeviceGetCount=lambda: len(uuids),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: f"GPU-{handle}".encode(),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=16_000 + handle),
        nvmlDeviceGetUUID=lambda handle: uuids[handle].encode(),
        nvmlDeviceGetPowerManagementLimit=lambda _handle: 200_000,
        nvmlDeviceGetPersistenceMode=lambda _handle: 1,
        nvmlDeviceGetClockInfo=lambda _handle, domain: 1_500 + domain,
        nvmlDeviceGetTotalEnergyConsumption=total_energy,
        nvmlDeviceGetPowerUsage=lambda _handle: 50_000,
    )


def _fake_torch(
    *,
    cuda_available: bool = True,
    uuids: tuple[str | None, ...] = ("GPU-private-uuid-0",),
    events: list[str] | None = None,
) -> SimpleNamespace:
    recorded_events = events if events is not None else []

    class _FakeTensor:
        def __matmul__(self, _other: object) -> _FakeTensor:
            recorded_events.append("compute")
            return self

        def sum(self) -> _FakeTensor:
            return self

        @staticmethod
        def item() -> float:
            return 262_144.0

    return SimpleNamespace(
        version=SimpleNamespace(cuda="13.0" if cuda_available else None),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: len(uuids) if cuda_available else 0,
            get_device_name=lambda index: f"GPU-{index}",
            get_device_properties=lambda index: SimpleNamespace(uuid=uuids[index]),
            synchronize=lambda _device: recorded_events.append("synchronize"),
        ),
        device=lambda label: label,
        ones=lambda _shape, *, device: _FakeTensor(),
    )


def test_torch_cuda_probe_executes_compute_on_selected_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    pynvml = _qualified_pynvml(
        ("GPU-private-uuid-0",),
        {0: [10_000, 10_000, 12_000]},
        events,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events=events))

    nvml, result = preflight.gpu_qualification_probe(active=True, selected_index=0)

    assert result["available"] is True
    assert result["selection_valid"] is True
    assert result["identity_verified"] is True
    assert result["compute_probe_passed"] is True
    assert result["selected_device_name"] == "GPU-0"
    assert result["nvml_counter_positive"] is True
    assert result["nvml_sample_energy_delta_mj"] == 2_000
    assert nvml["selected_device"]["counter_positive"] is True
    assert events.index("compute") > 1
    assert events[-1] == "energy:0"


def test_torch_cuda_probe_rejects_cpu_only_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        _qualified_pynvml(
            ("GPU-private-uuid-0",),
            {0: [10_000]},
            [],
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))

    _, result = preflight.gpu_qualification_probe(active=True, selected_index=0)

    assert result["available"] is False
    assert result["selection_valid"] is False
    assert result["compute_probe_passed"] is False


def test_gpu_qualification_inactive_probe_performs_no_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        _qualified_pynvml(
            ("GPU-private-uuid-0",),
            {0: [10_000]},
            events,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events=events))

    nvml, torch_cuda = preflight.gpu_qualification_probe(
        active=False,
        selected_index=0,
    )

    assert events == ["energy:0"]
    assert nvml["selected_device"]["counter_monotonic"] is None
    assert nvml["selected_device"]["counter_positive"] is None
    assert torch_cuda["selection_valid"] is True
    assert torch_cuda["compute_probe_passed"] is False
    assert torch_cuda["nvml_counter_bracketed"] is False


def test_gpu_qualification_rejects_stuck_nvml_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        _qualified_pynvml(
            ("GPU-private-uuid-0",),
            {0: [10_000, 10_000, 10_000]},
            [],
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    nvml, torch_cuda = preflight.gpu_qualification_probe(
        active=True,
        selected_index=0,
    )

    assert torch_cuda["compute_probe_passed"] is True
    assert torch_cuda["nvml_counter_bracketed"] is True
    assert torch_cuda["nvml_counter_positive"] is False
    assert torch_cuda["nvml_sample_energy_delta_mj"] == 0
    assert nvml["selected_device"]["counter_monotonic"] is True
    assert nvml["selected_device"]["counter_positive"] is False


def test_gpu_qualification_remaps_cuda_index_by_uuid_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        _qualified_pynvml(
            ("GPU-physical-zero", "GPU-physical-one"),
            {
                0: [10_000],
                1: [20_000, 20_000, 23_000],
            },
            events,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            uuids=("physical-one", "physical-zero"),
            events=events,
        ),
    )

    nvml, torch_cuda = preflight.gpu_qualification_probe(
        active=True,
        selected_index=1,
    )

    assert torch_cuda["identity_verification_mode"] == "uuid_hash"
    assert torch_cuda["identity_verified"] is True
    assert torch_cuda["selected_cuda_index"] == 0
    assert torch_cuda["visibility_remapped"] is True
    assert torch_cuda["nvml_counter_positive"] is True
    assert nvml["selected_device"]["index"] == 1
    assert events[-1] == "energy:1"


def test_gpu_qualification_rejects_mismatched_device_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        _qualified_pynvml(
            ("GPU-physical-zero", "GPU-physical-one"),
            {0: [10_000], 1: [20_000]},
            [],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(uuids=("other-zero", "other-one")),
    )

    _, torch_cuda = preflight.gpu_qualification_probe(
        active=True,
        selected_index=1,
    )

    assert torch_cuda["identity_verified"] is False
    assert torch_cuda["identity_mismatch"] is True
    assert torch_cuda["identity_verification_mode"] == "uuid_hash_mismatch"
    assert torch_cuda["selection_valid"] is False
    assert torch_cuda["compute_probe_passed"] is False


def test_gpu_qualification_allows_uuidless_single_device_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        _qualified_pynvml(
            ("GPU-private-uuid-0",),
            {0: [10_000]},
            [],
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(uuids=(None,)))

    _, torch_cuda = preflight.gpu_qualification_probe(
        active=False,
        selected_index=0,
    )

    assert torch_cuda["identity_verified"] is True
    assert torch_cuda["identity_verification_mode"] == "single_device_count"
    assert torch_cuda["selection_valid"] is True
    assert torch_cuda["compute_probe_passed"] is False
