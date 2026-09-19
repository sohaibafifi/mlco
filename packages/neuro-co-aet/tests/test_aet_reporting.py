"""Reporting must use authoritative AET classifications."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from neuro_co.aet.analysis import TrainAggregate, plots
from neuro_co.aet.analysis import cli as report_cli
from neuro_co.aet.analysis.cli import (
    INPUT_MANIFEST_SCHEMA,
    _load_expected_manifest,
    _read_jsons,
    _summary_markdown,
)
from neuro_co.aet.analysis.plots import _training_energy_wh


def test_summary_does_not_promote_point_feasible_row() -> None:
    rows = [
        {
            "feasible": True,
            "quality_status": "unidentified",
            "aet_E_status": "unidentified",
            "aet_E_reason": "quality_not_confirmatory",
            "aet_E": float("nan"),
        }
    ]

    summary = _summary_markdown([], [], [], rows)

    assert "No finite AET rows" in summary
    assert "Finite AET rows" not in summary
    assert "Exploratory input set" in summary
    assert "'unidentified': 1" in summary


def test_training_plot_prefers_canonical_joules() -> None:
    assert _training_energy_wh({"energy_j": 7200.0, "energy_wh": 999.0}) == pytest.approx(2.0)
    assert _training_energy_wh({"energy_wh": 3.0}) == pytest.approx(3.0)
    assert math.isnan(_training_energy_wh({}))


def test_asymptotic_plot_uses_primary_training_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, float] = {}

    def fake_curves(
        training_energy: float,
        nn_energy: float,
        baseline_energy: float,
        n_values: list[int] | None,
    ) -> tuple[list[int], list[float], list[float]]:
        captured["training_energy"] = training_energy
        return (
            [1, 10],
            [training_energy + nn_energy, training_energy + 10 * nn_energy],
            [
                baseline_energy,
                10 * baseline_energy,
            ],
        )

    monkeypatch.setattr(plots, "energy_curves", fake_curves)
    aggregate = TrainAggregate(
        energy_wh_median=100.0,
        energy_wh_p25=90.0,
        energy_wh_p75=110.0,
        co2_g_median=10.0,
        co2_g_p25=9.0,
        co2_g_p75=11.0,
        n_seeds=3,
        hardware_id="fixture",
        energy_wh_mean=200.0,
    )
    rows = [
        {
            "aet_E_status": "finite",
            "aet_E": 1000.0,
            "E_NN_wh_per_inst": 0.1,
            "E_meta_wh_per_inst": 0.2,
        }
    ]

    plots.plot_asymptotic(aggregate, rows, tmp_path / "asymptotic.png")

    assert captured["training_energy"] == pytest.approx(200.0)


def test_record_reader_fails_on_corrupt_json_instead_of_skipping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "energy_train.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        _read_jsons(tmp_path, "energy_train.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _calibration_payload(
    raw_trace_manifest: str,
    raw_trace_manifest_sha256: str,
    analysis_script: str,
    analysis_script_sha256: str,
    trace_hashes: dict[tuple[str, int], str],
) -> dict[str, Any]:
    return {
        "schema_version": "aet-calibration/v1",
        "calibration_id": "cal-win-a4500-01",
        "status": "passed",
        "host_id": "win-a4500-01",
        "execution_layer": "wsl2",
        "created_at": "2026-08-21T00:00:00+00:00",
        "hardware": {
            "cpu": "Intel Core i7-13700",
            "gpu": "NVIDIA RTX A4500",
            "gpu_index": 0,
            "gpu_device_id_sha256": "2" * 64,
        },
        "wall_meter": {
            "required": True,
            "model": "qualified-meter",
            "device_id_hash": "3" * 64,
            "accuracy_percent": 1.0,
            "sample_interval_s": 1.0,
            "data_interface": "csv",
        },
        "counter_backends": {
            "system_primary": "wall_meter",
            "gpu_diagnostic": "nvml",
            "strict_backend": True,
            "pue": 1.0,
            "report_embodied": False,
        },
        "timing": {
            "timestamp_alignment_error_s": 0.1,
            "missing_sample_percent": 0.0,
        },
        "criteria": {
            "max_missing_sample_percent": 0.5,
            "max_repeat_cv_percent": 5.0,
            "max_coverage_ratio_cv_percent": 5.0,
            "minimum_block_duration_s": 120.0,
            "minimum_samples_per_block": 100,
            "repeats_per_workload_class": 5,
        },
        "workload_classes": {
            name: [
                {
                    "duration_s": 120.0,
                    "sample_count": 120,
                    "raw_sha256": trace_hashes[(name, index)],
                }
                for index in range(5)
            ]
            for name in ("idle", "cpu_only", "gpu_only", "combined")
        },
        "summary": {
            "meter_repeat_cv_percent": 1.0,
            "coverage_ratio_cv_percent": 1.0,
            "nvml_positive_and_monotonic": True,
            "prohibited_fallback_observed": False,
        },
        "evidence": {
            "raw_trace_manifest": raw_trace_manifest,
            "raw_trace_manifest_sha256": raw_trace_manifest_sha256,
            "analysis_script": analysis_script,
            "analysis_script_sha256": analysis_script_sha256,
        },
        "approval": {
            "passed_at": "2026-08-21T00:01:00+00:00",
            "approved_by": "author",
        },
    }


def _preflight_payload(calibration_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "aet-preflight/v1",
        "generated_at": "2026-08-21T00:02:00+00:00",
        "host_id": "win-a4500-01",
        "system": {"execution_layer": "wsl2"},
        "git": {"available": True, "dirty": False, "sha": "6" * 40},
        "readiness": {
            "environment_ready": True,
            "git_clean": True,
            "gpu_component_ready": True,
            "calibration_compatible": True,
            "system_energy_ready": True,
            "measured_smoke_ready": True,
            "confirmatory_primary_ready": True,
        },
        "strict_tracker": {"importable": True, "supported": True},
        "backends": {
            "codecarbon_allowed_primary": False,
            "tdp_allowed_primary": False,
            "nvml": {
                "available": True,
                "active_probe": True,
                "selection_valid": True,
                "selected_index": 0,
                "selected_device": {
                    "index": 0,
                    "selected": True,
                    "measurement_mode_supported": True,
                    "device_id_sha256": "2" * 64,
                },
            },
        },
        "calibration": {
            "passed": True,
            "schema_version": "aet-calibration/v1",
            "host_id": "win-a4500-01",
            "execution_layer": "wsl2",
            "gpu_index": 0,
            "gpu_device_id_sha256": "2" * 64,
            "sha256": calibration_sha256,
        },
    }


def _provenance_payload(
    role: str,
    run_id: str,
    preflight_sha256: str,
    calibration_sha256: str,
    training_dataset_manifest_sha256: str | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {"recipe": "measurement-smoke"}
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    measurement: dict[str, Any] = {
        "primary_backend": "wall_meter",
        "pue": 1.0,
        "fallback": False,
        "preflight_sha256": preflight_sha256,
        "calibration_sha256": calibration_sha256,
        "gpu_index": 0,
        "gpu_device_id_sha256": "2" * 64,
    }
    if training_dataset_manifest_sha256 is not None:
        measurement["training_dataset_manifest_sha256"] = training_dataset_manifest_sha256
    measurement["diagnostic_backend"] = None if role == "baseline" else "nvml"
    service_mode = "training" if role == "training" else "warm"
    return {
        "schema_version": "aet-provenance/v1",
        "run_id": run_id,
        "host_id": "win-a4500-01",
        "timestamp_start": "2026-08-21T00:10:00+00:00",
        "timestamp_end": "2026-08-21T00:12:00+00:00",
        "status": "completed",
        "failure_reason": None,
        "git_sha": "6" * 40,
        "git_dirty": False,
        "dirty_patch_sha256": None,
        "execution_layer": "wsl2",
        "environment_hash": "7" * 64,
        "python_version": "3.12.3",
        "python_executable": "python.exe",
        "platform": "Windows-11-WSL2",
        "cpu": "Intel Core i7-13700",
        "memory_total_b": 64_000_000_000,
        "library_versions": {"neuro-co-aet": "0.1.0", "torch": "2.12.0"},
        "lockfiles": {"uv": {"path": "uv.lock", "size_bytes": 100, "sha256": "b" * 64}},
        "seeds": {"run": 0},
        "config": config,
        "config_sha256": config_digest,
        "artifacts": {
            "solver_output": {
                "path": f"artifacts/{run_id}.json",
                "size_bytes": 100,
                "sha256": "c" * 64,
            }
        },
        "runtime_controls": {
            "schema_version": "aet-runtime-controls/v1",
            "windows_build": "26100",
            "wsl_version": "2.5.10",
            "wsl_kernel": "6.6.87.2-microsoft-standard-WSL2",
            "wsl_memory_limit_b": 32_000_000_000,
            "cpu_logical_processors": 24,
            "cpu_threads": 1,
            "cpu_affinity": [0],
            "windows_power_plan": "balanced",
            "cpu_power_mode": "normal",
            "gpu_driver": "580.0",
            "cuda_version": "13.0",
            "gpu_power_limit_w": 200.0,
            "gpu_graphics_clock_mhz": 1800.0,
            "gpu_memory_clock_mhz": 16000.0,
            "gpu_clock_policy": "application-default",
            "gpu_persistence_mode": "enabled",
            "warmup_completed": True,
            "warmup_duration_s": 120.0,
            "warmup_stability_criterion": "throughput-cv-below-2pct",
            "warmup_trace_sha256": "d" * 64,
            "background_load_status": "clear",
            "background_load_check": "task-manager-and-process-snapshot",
            "background_load_trace_sha256": "e" * 64,
            "service_boundary": {
                "mode": service_mode,
                "initialization_reuse": role != "training",
                "includes": ["solver-workload", "solution-validation"],
                "excludes": ["environment-installation"],
            },
        },
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX A4500",
                "device_id_sha256": "2" * 64,
            }
        ],
        "measurement": measurement,
    }


def _instance_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": "aet-instance-manifest/v1",
        "manifest_id": "cvrp20-confirmatory",
        "status": "complete",
        "complete": True,
        "split": "confirmatory",
        "problem": "cvrp",
        "size": 20,
        "item_count": 100,
        "entries": [
            {"instance_id": f"cvrp20-{index:03d}", "sha256": f"{index + 8:064x}"}
            for index in range(100)
        ],
    }


def _training_dataset_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": "aet-training-dataset-manifest/v1",
        "dataset_id": "cvrp20-training",
        "status": "complete",
        "complete": True,
        "split": "training",
        "problem": "cvrp",
        "size": 20,
        "item_count": 1,
        "generation": {
            "method": "nazari-uniform",
            "seed": 1234,
            "config_sha256": "d" * 64,
        },
        "shards": [
            {
                "path": "datasets/cvrp20-train-000.npz",
                "sha256": "e" * 64,
                "item_count": 1,
            }
        ],
    }


def _calibration_trace_manifest_payload(trace_paths: list[Path], root: Path) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    manifest_parent = root / "support"
    for trace_path in trace_paths:
        workload_class, block_text = trace_path.stem.rsplit("-", 1)
        traces.append(
            {
                "raw_trace": trace_path.relative_to(manifest_parent).as_posix(),
                "raw_sha256": _sha256(trace_path),
                "workload_class": workload_class,
                "block_index": int(block_text),
            }
        )
    return {
        "schema_version": "aet-raw-trace-manifest/v1",
        "traces": traces,
    }


def _validation_payload(
    role: str,
    run_id: str,
    preflight_sha256: str,
    calibration_sha256: str,
    provenance_sha256: str,
    instance_manifest_sha256: str | None,
    training_dataset_manifest_sha256: str | None,
    purpose: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    is_training = role == "training"
    return {
        "schema_version": "aet-validation/v1",
        "run_id": run_id,
        "role": role,
        "status": "passed",
        "complete": True,
        "independent": True,
        "validator": "independent-replay-v1",
        "checks": [
            {
                "name": "solution-replay",
                "passed": True,
                "evidence_sha256": evidence_sha256,
            }
        ],
        "metrics": {"objective": 1.0},
        "host_id": "win-a4500-01",
        "execution_layer": "wsl2",
        "git_sha": "6" * 40,
        "gpu_index": 0,
        "gpu_device_id_sha256": "2" * 64,
        "preflight_sha256": preflight_sha256,
        "calibration_sha256": calibration_sha256,
        "provenance_sha256": provenance_sha256,
        "instance_manifest_sha256": None if is_training else instance_manifest_sha256,
        "training_dataset_manifest_sha256": (
            training_dataset_manifest_sha256 if is_training else None
        ),
        "expected_records": 1 if is_training else 100,
        "validated_records": 1 if is_training else 100,
        "failure_count": 0,
        "excluded_count": 0,
        "gap_to_reference_pct": None if is_training else (1.0 if role == "inference" else 0.0),
        "quality_feasible_confirmatory": (
            None if is_training or purpose == "instrumentation_smoke" else True
        ),
    }


def _energy_payload(
    role: str,
    run_id: str,
    preflight_sha256: str,
    calibration_sha256: str,
    provenance_sha256: str,
    validation_sha256: str,
    instance_manifest_sha256: str | None,
    training_dataset_manifest_sha256: str | None,
    wall_trace_sha256: str,
    nvml_trace_sha256: str | None,
    purpose: str,
) -> dict[str, Any]:
    is_training = role == "training"
    energy_j = {"training": 360_000.0, "inference": 3_600.0, "baseline": 7_200.0}[role]
    extra: dict[str, Any] = {
        "role": role,
        "run_id": run_id,
        "run_status": "completed",
        "failure_type": None,
        "measurement_qualified_confirmatory": purpose == "confirmatory",
        "measurement_qualified_instrumentation": purpose == "instrumentation_smoke",
        "purpose": purpose,
        "scientific_use": purpose == "confirmatory",
        "fallback": False,
        "allow_fallback": False,
        "tdp_fallback": False,
        "host_id": "win-a4500-01",
        "execution_layer": "wsl2",
        "git_sha": "6" * 40,
        "gpu_index": 0,
        "gpu_device_id_sha256": "2" * 64,
        "preflight_sha256": preflight_sha256,
        "calibration_sha256": calibration_sha256,
        "provenance_sha256": provenance_sha256,
        "artifact_sha256": validation_sha256,
        "problem": "cvrp",
        "wall_meter": {
            "duration_s": 120.0,
            "sample_interval_s": 1.0,
            "sample_count": 120,
            "dropped_sample_count": 0,
            "trace_sha256": wall_trace_sha256,
            "calibration_id": "cal-win-a4500-01",
            "device_id_sha256": "3" * 64,
            "started_at_utc": "2026-08-21T00:10:00+00:00",
            "ended_at_utc": "2026-08-21T00:12:00+00:00",
        },
    }
    if role in {"training", "inference"}:
        extra["diagnostics"] = {
            "nvml": {
                "schema_version": "aet-nvml-diagnostic/v1",
                "source": "independent_concurrent_sampler",
                "gpu_index": 0,
                "device_id_sha256": "2" * 64,
                "measurement_mode": "nvml_total_energy_counter",
                "energy_j": 1_000.0,
                "duration_s": 120.0,
                "trace_sha256": nvml_trace_sha256,
                "sampling_interval_s": None,
                "sample_count": 2,
                "dropped_sample_count": 0,
                "started_at_utc": "2026-08-21T00:10:00+00:00",
                "ended_at_utc": "2026-08-21T00:12:00+00:00",
            }
        }
    if is_training:
        extra.update(
            {
                "seed": 0,
                "variant": "AM",
                "size": 20,
                "training_dataset_manifest_sha256": training_dataset_manifest_sha256,
            }
        )
    elif role == "inference":
        extra.update(
            {
                "instance_manifest_sha256": instance_manifest_sha256,
                "size_key": 20,
                "variant": "AM",
                "batch_size": 10,
                "gap_to_reference_pct": 1.0,
                "quality_feasible_confirmatory": (True if purpose == "confirmatory" else None),
                "delta_energy_j_per_item_ci_low": 18.0,
                "delta_energy_j_per_item_ci_high": 54.0,
            }
        )
    else:
        extra.update(
            {
                "instance_manifest_sha256": instance_manifest_sha256,
                "size": 20,
                "thread_mode": "mono",
                "gap_to_reference_pct": 0.0,
                "quality_feasible_confirmatory": (True if purpose == "confirmatory" else None),
            }
        )
    items = 1 if is_training else 100
    return {
        "schema_version": "1.0",
        "units": {
            "duration_s": "s",
            "energy_j": "J",
            "energy_gpu_j": "J",
            "energy_cpu_j": "J",
            "energy_dram_j": "J",
            "co2_operational_kg": "kgCO2eq",
            "co2_embodied_kg": "kgCO2eq",
            "co2_total_kg": "kgCO2eq",
            "avg_power_w": "W",
            "items_processed": "item",
            "throughput_items_per_s": "item/s",
        },
        "duration_s": 120.0,
        "energy_j": energy_j,
        "energy_gpu_j": 0.0,
        "energy_cpu_j": 0.0,
        "energy_dram_j": 0.0,
        "co2_operational_kg": 0.01,
        "co2_embodied_kg": 0.0,
        "co2_total_kg": 0.01,
        "avg_power_w": energy_j / 120.0,
        "items_processed": items,
        "throughput_items_per_s": items / 120.0,
        "backend": "wall_meter",
        "energy_domains": ["whole_system_ac"],
        "measurement_scope": "whole_system_ac",
        "hardware": {"id": "win-a4500-01", "pue": 1.0},
        "extra": extra,
    }


def _refresh_manifest(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in payload["files"]:
        entry["sha256"] = _sha256(manifest.parent / entry["path"])
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_valid_bundle(
    tmp_path: Path,
    *,
    purpose: str = "confirmatory",
    confirmatory_eligible: bool = True,
) -> tuple[Path, dict[str, Path]]:
    paths: dict[str, Path] = {}
    entries: list[dict[str, str]] = []

    analysis_path = tmp_path / "support" / "calibration_analysis.py"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text("# frozen calibration analysis fixture\n", encoding="utf-8")
    paths["calibration_analysis"] = analysis_path
    trace_paths: list[Path] = []
    for workload_class in ("idle", "cpu_only", "gpu_only", "combined"):
        for block_index in range(5):
            trace_path = tmp_path / "support" / "traces" / f"{workload_class}-{block_index}.csv"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                f"workload,timestamp,power_w\n{workload_class},{block_index},{100 + block_index}\n",
                encoding="utf-8",
            )
            trace_paths.append(trace_path)
    paths["calibration_trace_manifest"] = _write_json(
        tmp_path / "support" / "calibration-traces.json",
        _calibration_trace_manifest_payload(trace_paths, tmp_path),
    )
    paths["calibration"] = _write_json(
        tmp_path / "support" / "calibration.json",
        _calibration_payload(
            paths["calibration_trace_manifest"].name,
            _sha256(paths["calibration_trace_manifest"]),
            paths["calibration_analysis"].name,
            _sha256(paths["calibration_analysis"]),
            {
                (path.stem.rsplit("-", 1)[0], int(path.stem.rsplit("-", 1)[1])): _sha256(path)
                for path in trace_paths
            },
        ),
    )
    calibration_sha = _sha256(paths["calibration"])
    paths["preflight"] = _write_json(
        tmp_path / "support" / "preflight.json", _preflight_payload(calibration_sha)
    )
    preflight_sha = _sha256(paths["preflight"])
    paths["instance_manifest"] = _write_json(
        tmp_path / "support" / "instance_manifest.json", _instance_manifest_payload()
    )
    instance_sha = _sha256(paths["instance_manifest"])
    paths["training_dataset_manifest"] = _write_json(
        tmp_path / "support" / "training_dataset_manifest.json",
        _training_dataset_manifest_payload(),
    )
    training_dataset_sha = _sha256(paths["training_dataset_manifest"])
    for role in ("training", "inference", "baseline"):
        run_id = f"{role}-run-001"
        validation_evidence_path = tmp_path / role / "run.log"
        validation_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        validation_evidence_path.write_text(f"completed {run_id}\n", encoding="utf-8")
        paths[f"validation_evidence_{role}"] = validation_evidence_path
        wall_trace_path = tmp_path / role / "wall-trace.jsonl"
        wall_trace_path.parent.mkdir(parents=True, exist_ok=True)
        wall_trace_path.write_text(f'{{"run_id":"{run_id}","source":"wall"}}\n')
        paths[f"wall_trace_{role}"] = wall_trace_path
        nvml_trace_path: Path | None = None
        if role in {"training", "inference"}:
            nvml_trace_path = tmp_path / role / "nvml-trace.jsonl"
            nvml_trace_path.write_text(f'{{"run_id":"{run_id}","source":"nvml"}}\n')
            paths[f"nvml_trace_{role}"] = nvml_trace_path
        paths[f"provenance_{role}"] = _write_json(
            tmp_path / role / "provenance.json",
            _provenance_payload(
                role,
                run_id,
                preflight_sha,
                calibration_sha,
                training_dataset_sha if role == "training" else None,
            ),
        )
        provenance_sha = _sha256(paths[f"provenance_{role}"])
        paths[f"validation_{role}"] = _write_json(
            tmp_path / role / "validation.json",
            _validation_payload(
                role,
                run_id,
                preflight_sha,
                calibration_sha,
                provenance_sha,
                instance_sha,
                training_dataset_sha if role == "training" else None,
                purpose,
                _sha256(validation_evidence_path),
            ),
        )
        validation_sha = _sha256(paths[f"validation_{role}"])
        basename = {
            "training": "energy_train.json",
            "inference": "energy_eval.json",
            "baseline": "energy_baseline.json",
        }[role]
        paths[f"energy_{role}"] = _write_json(
            tmp_path / role / basename,
            _energy_payload(
                role,
                run_id,
                preflight_sha,
                calibration_sha,
                provenance_sha,
                validation_sha,
                instance_sha,
                training_dataset_sha if role == "training" else None,
                _sha256(wall_trace_path),
                _sha256(nvml_trace_path) if nvml_trace_path is not None else None,
                purpose,
            ),
        )
        for support_role in ("provenance", "validation"):
            path = paths[f"{support_role}_{role}"]
            entries.append(
                {
                    "role": support_role,
                    "path": path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha256(path),
                }
            )
        entries.append(
            {
                "role": role,
                "path": paths[f"energy_{role}"].relative_to(tmp_path).as_posix(),
                "sha256": _sha256(paths[f"energy_{role}"]),
            }
        )
        entries.append(
            {
                "role": "wall_trace",
                "path": wall_trace_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(wall_trace_path),
                "run_id": run_id,
            }
        )
        entries.append(
            {
                "role": "validation_evidence",
                "path": validation_evidence_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(validation_evidence_path),
                "run_id": run_id,
            }
        )
        if nvml_trace_path is not None:
            entries.append(
                {
                    "role": "nvml_trace",
                    "path": nvml_trace_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha256(nvml_trace_path),
                    "run_id": run_id,
                }
            )
    for support_role in (
        "calibration",
        "preflight",
        "instance_manifest",
        "training_dataset_manifest",
        "calibration_trace_manifest",
        "calibration_analysis",
    ):
        entries.append(
            {
                "role": support_role,
                "path": paths[support_role].relative_to(tmp_path).as_posix(),
                "sha256": _sha256(paths[support_role]),
            }
        )
    entries.extend(
        {
            "role": "calibration_trace",
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": _sha256(path),
        }
        for path in trace_paths
    )
    manifest = _write_json(
        tmp_path / "report-input-manifest.json",
        {
            "schema_version": INPUT_MANIFEST_SCHEMA,
            "purpose": purpose,
            "confirmatory_eligible": confirmatory_eligible,
            "files": entries,
        },
    )
    paths["manifest"] = manifest
    return manifest, paths


def test_valid_confirmatory_bundle_links_all_content(tmp_path: Path) -> None:
    manifest, _ = _write_valid_bundle(tmp_path)

    by_role = _load_expected_manifest(tmp_path, manifest)

    assert len(by_role["training"]) == 1
    assert len(by_role["inference"]) == 1
    assert len(by_role["baseline"]) == 1
    assert len(by_role["provenance"]) == 3
    assert len(by_role["validation"]) == 3
    assert len(by_role["instance_manifest"]) == 1
    assert len(by_role["training_dataset_manifest"]) == 1
    assert len(by_role["calibration"]) == 1
    assert len(by_role["preflight"]) == 1
    assert len(by_role["calibration_trace_manifest"]) == 1
    assert len(by_role["calibration_trace"]) == 20
    assert len(by_role["calibration_analysis"]) == 1
    assert len(by_role["wall_trace"]) == 3
    assert len(by_role["nvml_trace"]) == 2
    assert len(by_role["validation_evidence"]) == 3


def test_valid_bundle_is_promoted_only_after_full_report_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _write_valid_bundle(tmp_path)
    monkeypatch.setattr(report_cli, "plot_aet_vs_delta", lambda *args: None)
    monkeypatch.setattr(report_cli, "plot_asymptotic", lambda *args: None)
    monkeypatch.setattr(report_cli, "plot_per_seed_energy", lambda *args: None)

    summary_path = report_cli.write_aet_report(
        tmp_path,
        deltas=[1.0],
        out="report",
        expected_manifest=manifest,
    )

    assert "confirmatory input manifest valid: True" in summary_path.read_text(encoding="utf-8")
    assert "finite" in (tmp_path / "report" / "aet_table.csv").read_text(encoding="utf-8")


def test_instrumentation_smoke_bundle_is_validated_but_never_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _write_valid_bundle(
        tmp_path,
        purpose="instrumentation_smoke",
        confirmatory_eligible=False,
    )
    loaded = _load_expected_manifest(tmp_path, manifest)
    assert loaded.purpose == "instrumentation_smoke"
    assert loaded.confirmatory_eligible is False
    monkeypatch.setattr(report_cli, "plot_aet_vs_delta", lambda *args: None)
    monkeypatch.setattr(report_cli, "plot_asymptotic", lambda *args: None)
    monkeypatch.setattr(report_cli, "plot_per_seed_energy", lambda *args: None)

    summary_path = report_cli.write_aet_report(
        tmp_path,
        deltas=[1.0],
        out="report",
        expected_manifest=manifest,
    )

    summary = summary_path.read_text(encoding="utf-8")
    with (tmp_path / "report" / "aet_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "confirmatory input manifest valid: False" in summary
    assert "Exploratory input set" in summary
    assert rows
    assert all(row["aet_E_status"] != "finite" for row in rows)


def test_instrumentation_smoke_cannot_claim_confirmatory_eligibility(tmp_path: Path) -> None:
    manifest, _ = _write_valid_bundle(
        tmp_path,
        purpose="instrumentation_smoke",
        confirmatory_eligible=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["confirmatory_eligible"] = True
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="cannot be confirmatory eligible"):
        _load_expected_manifest(tmp_path, manifest)


def test_smoke_energy_cannot_be_promoted_by_flipping_only_manifest_purpose(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_valid_bundle(
        tmp_path,
        purpose="instrumentation_smoke",
        confirmatory_eligible=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["purpose"] = "confirmatory"
    payload["confirmatory_eligible"] = True
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="quality_feasible_confirmatory must be a boolean"):
        _load_expected_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    ("path_key", "mutate", "message"),
    [
        (
            "energy_training",
            lambda value: value["extra"].update(
                {"run_status": "failed", "failure_type": "RuntimeError"}
            ),
            "completed measurement without failure",
        ),
        (
            "energy_baseline",
            lambda value: value.update({"avg_power_w": 999.0}),
            "avg_power_w is inconsistent",
        ),
        (
            "provenance_training",
            lambda value: value.update({"status": "failed", "failure_reason": "boom"}),
            "completed run without failure",
        ),
        (
            "validation_inference",
            lambda value: value.update({"validated_records": 99}),
            "partial, failed, or excluded",
        ),
        (
            "validation_inference",
            lambda value: value.update({"independent": False}),
            "independently validated",
        ),
        (
            "calibration",
            lambda value: value.update({"status": "pending"}),
            "must be a passed",
        ),
        (
            "preflight",
            lambda value: value["readiness"].update({"confirmatory_primary_ready": False}),
            "confirmatory_primary_ready must be true",
        ),
        (
            "instance_manifest",
            lambda value: value["entries"].pop(),
            "exactly item_count",
        ),
    ],
)
def test_confirmatory_bundle_rejects_failed_or_partial_support_content(
    tmp_path: Path,
    path_key: str,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    payload = json.loads(paths[path_key].read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(paths[path_key], payload)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match=message):
        _load_expected_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("host_id", "other-host", "host_id does not match linked provenance"),
        ("git_sha", "b" * 40, "git_sha does not match linked provenance"),
        ("gpu_device_id_sha256", "c" * 64, "gpu_device_id_sha256 does not match"),
    ],
)
def test_energy_identity_must_match_git_host_and_gpu_evidence(
    tmp_path: Path,
    field: str,
    bad_value: str,
    message: str,
) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    energy = json.loads(paths["energy_inference"].read_text(encoding="utf-8"))
    energy["extra"][field] = bad_value
    if field == "gpu_device_id_sha256":
        energy["extra"]["diagnostics"]["nvml"]["device_id_sha256"] = bad_value
    _write_json(paths["energy_inference"], energy)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match=message):
        _load_expected_manifest(tmp_path, manifest)


def test_validation_quality_must_match_energy_record(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    energy = json.loads(paths["energy_inference"].read_text(encoding="utf-8"))
    energy["extra"]["gap_to_reference_pct"] = 2.0
    _write_json(paths["energy_inference"], energy)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="quality gap does not match linked validation"):
        _load_expected_manifest(tmp_path, manifest)


def test_preflight_must_link_the_same_calibration_as_energy_records(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    preflight["calibration"]["sha256"] = "b" * 64
    _write_json(paths["preflight"], preflight)
    new_preflight_sha = _sha256(paths["preflight"])
    for role in ("training", "inference", "baseline"):
        provenance_path = paths[f"provenance_{role}"]
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["measurement"]["preflight_sha256"] = new_preflight_sha
        _write_json(provenance_path, provenance)
        new_provenance_sha = _sha256(provenance_path)

        validation_path = paths[f"validation_{role}"]
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["preflight_sha256"] = new_preflight_sha
        validation["provenance_sha256"] = new_provenance_sha
        _write_json(validation_path, validation)
        new_validation_sha = _sha256(validation_path)

        energy_path = paths[f"energy_{role}"]
        energy = json.loads(energy_path.read_text(encoding="utf-8"))
        energy["extra"]["preflight_sha256"] = new_preflight_sha
        energy["extra"]["provenance_sha256"] = new_provenance_sha
        energy["extra"]["artifact_sha256"] = new_validation_sha
        _write_json(energy_path, energy)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="calibration_sha256 does not match linked preflight"):
        _load_expected_manifest(tmp_path, manifest)


def test_shared_instance_set_allows_different_measured_item_totals(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    inference = json.loads(paths["energy_inference"].read_text(encoding="utf-8"))
    baseline = json.loads(paths["energy_baseline"].read_text(encoding="utf-8"))
    inference["items_processed"] = 1024
    baseline["items_processed"] = 128
    inference["throughput_items_per_s"] = 1024 / inference["duration_s"]
    baseline["throughput_items_per_s"] = 128 / baseline["duration_s"]
    _write_json(paths["energy_inference"], inference)
    _write_json(paths["energy_baseline"], baseline)
    _refresh_manifest(manifest)

    _load_expected_manifest(tmp_path, manifest)


def test_instance_manifest_count_must_match_validation_count(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    validation = json.loads(paths["validation_baseline"].read_text(encoding="utf-8"))
    validation["expected_records"] = 99
    validation["validated_records"] = 99
    _write_json(paths["validation_baseline"], validation)
    new_validation_sha = _sha256(paths["validation_baseline"])
    energy = json.loads(paths["energy_baseline"].read_text(encoding="utf-8"))
    energy["extra"]["artifact_sha256"] = new_validation_sha
    _write_json(paths["energy_baseline"], energy)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="validation count does not match"):
        _load_expected_manifest(tmp_path, manifest)


def test_inference_and_baseline_must_share_one_instance_manifest(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    second_payload = json.loads(paths["instance_manifest"].read_text(encoding="utf-8"))
    second_payload["manifest_id"] = "cvrp20-confirmatory-other"
    second_manifest = _write_json(
        tmp_path / "support" / "instance_manifest_other.json", second_payload
    )
    second_sha = _sha256(second_manifest)

    validation = json.loads(paths["validation_baseline"].read_text(encoding="utf-8"))
    validation["instance_manifest_sha256"] = second_sha
    _write_json(paths["validation_baseline"], validation)
    validation_sha = _sha256(paths["validation_baseline"])
    energy = json.loads(paths["energy_baseline"].read_text(encoding="utf-8"))
    energy["extra"]["instance_manifest_sha256"] = second_sha
    energy["extra"]["artifact_sha256"] = validation_sha
    _write_json(paths["energy_baseline"], energy)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["files"].append(
        {
            "role": "instance_manifest",
            "path": second_manifest.relative_to(tmp_path).as_posix(),
            "sha256": second_sha,
        }
    )
    _write_json(manifest, manifest_payload)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="must share one instance manifest"):
        _load_expected_manifest(tmp_path, manifest)


def test_training_energy_must_link_its_distinct_dataset_manifest(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    energy = json.loads(paths["energy_training"].read_text(encoding="utf-8"))
    energy["extra"]["training_dataset_manifest_sha256"] = "f" * 64
    _write_json(paths["energy_training"], energy)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="training_dataset_manifest"):
        _load_expected_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("windows_power_plan", "", "must be a non-empty string"),
        ("background_load_status", "busy", "must be 'clear'"),
        ("warmup_completed", False, "must be true"),
    ],
)
def test_provenance_rejects_incomplete_runtime_disclosures(
    tmp_path: Path,
    field: str,
    bad_value: Any,
    message: str,
) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    provenance = json.loads(paths["provenance_inference"].read_text(encoding="utf-8"))
    provenance["runtime_controls"][field] = bad_value
    _write_json(paths["provenance_inference"], provenance)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match=message):
        _load_expected_manifest(tmp_path, manifest)


def test_provenance_runtime_controls_use_a_closed_schema(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    provenance = json.loads(paths["provenance_inference"].read_text(encoding="utf-8"))
    del provenance["runtime_controls"]["gpu_power_limit_w"]
    _write_json(paths["provenance_inference"], provenance)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="closed aet-runtime-controls/v1 schema"):
        _load_expected_manifest(tmp_path, manifest)


def test_cpu_affinity_must_name_exactly_the_declared_thread_count(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    provenance = json.loads(paths["provenance_inference"].read_text(encoding="utf-8"))
    provenance["runtime_controls"]["cpu_affinity"] = [0, 1]
    _write_json(paths["provenance_inference"], provenance)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="cpu_affinity is invalid"):
        _load_expected_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("python_executable", "/usr/bin/python", "must be a basename"),
        ("library_versions", {}, "must be non-empty"),
        ("lockfiles", {}, "must be a non-empty object"),
        ("seeds", {}, "must be non-empty"),
        ("artifacts", {}, "must be a non-empty object"),
    ],
)
def test_provenance_rejects_missing_reproducibility_disclosures(
    tmp_path: Path,
    field: str,
    bad_value: Any,
    message: str,
) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    provenance = json.loads(paths["provenance_training"].read_text(encoding="utf-8"))
    provenance[field] = bad_value
    _write_json(paths["provenance_training"], provenance)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match=message):
        _load_expected_manifest(tmp_path, manifest)


def test_inference_and_baseline_must_share_service_boundary(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    provenance_path = paths["provenance_baseline"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_controls"]["service_boundary"]["mode"] = "cold"
    _write_json(provenance_path, provenance)
    provenance_sha = _sha256(provenance_path)

    validation_path = paths["validation_baseline"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["provenance_sha256"] = provenance_sha
    _write_json(validation_path, validation)
    validation_sha = _sha256(validation_path)

    energy_path = paths["energy_baseline"]
    energy = json.loads(energy_path.read_text(encoding="utf-8"))
    energy["extra"]["provenance_sha256"] = provenance_sha
    energy["extra"]["artifact_sha256"] = validation_sha
    _write_json(energy_path, energy)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="must share one service boundary"):
        _load_expected_manifest(tmp_path, manifest)


def test_wall_trace_hash_must_match_the_linked_energy_record(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    paths["wall_trace_inference"].write_text(
        '{"run_id":"inference-run-001","source":"tampered"}\n',
        encoding="utf-8",
    )
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="does not link exactly one wall trace"):
        _load_expected_manifest(tmp_path, manifest)


def test_trace_manifest_run_id_must_match_energy_run_id(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    relative = paths["nvml_trace_inference"].relative_to(tmp_path).as_posix()
    entry = next(item for item in payload["files"] if item["path"] == relative)
    entry["run_id"] = "forged-run-id"
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="does not link exactly one NVML trace"):
        _load_expected_manifest(tmp_path, manifest)


def test_validation_evidence_bytes_must_match_the_linked_check(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    paths["validation_evidence_inference"].write_text(
        "forged validation evidence\n",
        encoding="utf-8",
    )
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="validation check does not link exactly one evidence"):
        _load_expected_manifest(tmp_path, manifest)


def test_validation_evidence_run_id_must_match_validation_run(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    relative = paths["validation_evidence_inference"].relative_to(tmp_path).as_posix()
    entry = next(item for item in payload["files"] if item["path"] == relative)
    entry["run_id"] = "forged-run-id"
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="validation check does not link exactly one evidence"):
        _load_expected_manifest(tmp_path, manifest)


def test_validation_supports_multiple_manifested_check_evidence_files(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    extra_evidence = tmp_path / "inference" / "objective-replay.log"
    extra_evidence.write_text("independent objective replay passed\n", encoding="utf-8")
    extra_sha = _sha256(extra_evidence)

    validation_path = paths["validation_inference"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].append(
        {
            "name": "objective-replay",
            "passed": True,
            "evidence_sha256": extra_sha,
        }
    )
    _write_json(validation_path, validation)
    validation_sha = _sha256(validation_path)

    energy_path = paths["energy_inference"]
    energy = json.loads(energy_path.read_text(encoding="utf-8"))
    energy["extra"]["artifact_sha256"] = validation_sha
    _write_json(energy_path, energy)

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["files"].append(
        {
            "role": "validation_evidence",
            "path": extra_evidence.relative_to(tmp_path).as_posix(),
            "sha256": extra_sha,
            "run_id": "inference-run-001",
        }
    )
    _write_json(manifest, manifest_payload)
    _refresh_manifest(manifest)

    _load_expected_manifest(tmp_path, manifest)


def test_manifested_validation_evidence_cannot_be_orphaned(tmp_path: Path) -> None:
    manifest, _ = _write_valid_bundle(tmp_path)
    orphan = tmp_path / "inference" / "orphan.log"
    orphan.write_text("unused evidence\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"].append(
        {
            "role": "validation_evidence",
            "path": orphan.relative_to(tmp_path).as_posix(),
            "sha256": _sha256(orphan),
            "run_id": "inference-run-001",
        }
    )
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="validation_evidence files are not linked"):
        _load_expected_manifest(tmp_path, manifest)


def test_baseline_must_not_contain_nvml_diagnostics(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    baseline_path = paths["energy_baseline"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    inference = json.loads(paths["energy_inference"].read_text(encoding="utf-8"))
    baseline["extra"]["diagnostics"] = inference["extra"]["diagnostics"]
    _write_json(baseline_path, baseline)
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="baseline must not contain an NVML diagnostic"):
        _load_expected_manifest(tmp_path, manifest)


def test_calibration_raw_trace_bytes_are_verified(tmp_path: Path) -> None:
    manifest, _ = _write_valid_bundle(tmp_path)
    trace_path = tmp_path / "support" / "traces" / "idle-0.csv"
    trace_path.write_text("tampered\n", encoding="utf-8")
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="does not exactly cover manifested raw traces"):
        _load_expected_manifest(tmp_path, manifest)


def test_calibration_analysis_script_bytes_are_verified(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    paths["calibration_analysis"].write_text("# tampered\n", encoding="utf-8")
    _refresh_manifest(manifest)

    with pytest.raises(ValueError, match="does not link a manifested analysis script"):
        _load_expected_manifest(tmp_path, manifest)


def test_expected_manifest_rejects_checksum_forgery(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    paths["preflight"].write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_expected_manifest(tmp_path, manifest)


def test_expected_manifest_rejects_unmanifested_support_file(tmp_path: Path) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    payload = json.loads(paths["provenance_training"].read_text(encoding="utf-8"))
    payload["run_id"] = "unmanifested-run"
    _write_json(tmp_path / "extra" / "provenance.json", payload)

    with pytest.raises(ValueError, match="unmanifested AET input"):
        _load_expected_manifest(tmp_path, manifest)


def test_expected_manifest_rejects_manifested_but_unlinked_support_file(
    tmp_path: Path,
) -> None:
    manifest, paths = _write_valid_bundle(tmp_path)
    payload = copy.deepcopy(json.loads(paths["provenance_training"].read_text(encoding="utf-8")))
    payload["run_id"] = "unused-run"
    extra = _write_json(tmp_path / "extra" / "provenance.json", payload)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["files"].append(
        {
            "role": "provenance",
            "path": extra.relative_to(tmp_path).as_posix(),
            "sha256": _sha256(extra),
        }
    )
    _write_json(manifest, manifest_payload)

    with pytest.raises(ValueError, match="not linked by the bundle"):
        _load_expected_manifest(tmp_path, manifest)
