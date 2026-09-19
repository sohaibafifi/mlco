"""`neuroco-aet-report`: scan run-dir outputs, build AET table + plots.

Usage::

    neuroco-aet-report <runs_root> [--deltas 0.5 1 2 5] [--out report]

Walks `runs_root` recursively, picks up every
`energy_train.json`, `energy_eval.json`, and `energy_baseline.json`,
aggregates training across seeds, crosses inference and baseline
records per problem size, computes AET tables, writes:

- `aet_table.csv`
- `aet_summary.md`
- `figures/aet_vs_delta.png`
- `figures/asymptotic.png`
- `figures/per_seed_energy.png`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from neuro_co.aet.analysis.aet import aggregate_training, build_aet_table
from neuro_co.aet.analysis.plots import (
    plot_aet_vs_delta,
    plot_asymptotic,
    plot_per_seed_energy,
)

INPUT_MANIFEST_SCHEMA = "aet-report-input-manifest/v1"
PROVENANCE_SCHEMA = "aet-provenance/v1"
VALIDATION_SCHEMA = "aet-validation/v1"
INSTANCE_MANIFEST_SCHEMA = "aet-instance-manifest/v1"
TRAINING_DATASET_MANIFEST_SCHEMA = "aet-training-dataset-manifest/v1"
CALIBRATION_SCHEMA = "aet-calibration/v1"
PREFLIGHT_SCHEMA = "aet-preflight/v1"
CALIBRATION_TRACE_MANIFEST_SCHEMA = "aet-raw-trace-manifest/v1"
_ROLE_BASENAMES: dict[str, str | None] = {
    "training": "energy_train.json",
    "inference": "energy_eval.json",
    "baseline": "energy_baseline.json",
    "provenance": None,
    "validation": None,
    "instance_manifest": None,
    "training_dataset_manifest": None,
    "calibration": None,
    "preflight": None,
    "calibration_trace_manifest": None,
    "calibration_trace": None,
    "calibration_analysis": None,
    "wall_trace": None,
    "nvml_trace": None,
    "validation_evidence": None,
}
_ENERGY_ROLES = {"training", "inference", "baseline"}
_SUPPORT_SCHEMAS = {
    "provenance": PROVENANCE_SCHEMA,
    "validation": VALIDATION_SCHEMA,
    "instance_manifest": INSTANCE_MANIFEST_SCHEMA,
    "training_dataset_manifest": TRAINING_DATASET_MANIFEST_SCHEMA,
    "calibration": CALIBRATION_SCHEMA,
    "preflight": PREFLIGHT_SCHEMA,
    "calibration_trace_manifest": CALIBRATION_TRACE_MANIFEST_SCHEMA,
}
_OPAQUE_ROLES = {
    "calibration_trace",
    "calibration_analysis",
    "wall_trace",
    "nvml_trace",
    "validation_evidence",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_EXECUTION_LAYERS = {"windows-native", "wsl2"}
_KNOWN_SUPPORT_BASENAMES = {
    "provenance.json",
    "validation.json",
    "instance_manifest.json",
    "training_dataset_manifest.json",
}
_ENERGY_UNITS = {
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
}
_ENERGY_KEYS = {
    "schema_version",
    "units",
    *_ENERGY_UNITS,
    "backend",
    "energy_domains",
    "measurement_scope",
    "hardware",
    "extra",
}
RUNTIME_CONTROLS_SCHEMA = "aet-runtime-controls/v1"
_RUNTIME_CONTROL_KEYS = {
    "schema_version",
    "windows_build",
    "wsl_version",
    "wsl_kernel",
    "wsl_memory_limit_b",
    "cpu_logical_processors",
    "cpu_threads",
    "cpu_affinity",
    "windows_power_plan",
    "cpu_power_mode",
    "gpu_driver",
    "cuda_version",
    "gpu_power_limit_w",
    "gpu_graphics_clock_mhz",
    "gpu_memory_clock_mhz",
    "gpu_clock_policy",
    "gpu_persistence_mode",
    "warmup_completed",
    "warmup_duration_s",
    "warmup_stability_criterion",
    "warmup_trace_sha256",
    "background_load_status",
    "background_load_check",
    "background_load_trace_sha256",
    "service_boundary",
}


class LoadedInputManifest(dict[str, list[Path]]):
    """Validated input paths plus their explicit promotion eligibility."""

    purpose: str
    confirmatory_eligible: bool

    def __init__(
        self,
        paths_by_role: dict[str, list[Path]],
        *,
        purpose: str,
        confirmatory_eligible: bool,
    ) -> None:
        super().__init__(paths_by_role)
        self.purpose = purpose
        self.confirmatory_eligible = confirmatory_eligible


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read required AET input {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in required AET input {path}: {exc}") from exc
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(record, dict) for record in data):
        return data
    raise ValueError(f"AET input must contain an object or list of objects: {path}")


def _read_jsons(root: Path, basename: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob(basename)):
        records.extend(_read_json_file(path))
    return records


def _safe_manifest_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("manifest file path must be a portable relative path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe manifest file path: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest file escapes runs_root: {value!r}") from exc
    return resolved


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _required_fields(record: dict[str, Any], fields: set[str], where: str) -> None:
    missing = sorted(fields - set(record))
    if missing:
        raise ValueError(f"{where} is missing fields: {', '.join(missing)}")


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase Git commit digest")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{where} must be a finite number >= {minimum}")
    return result


def _aware_datetime(value: Any, where: str) -> datetime:
    text = _nonempty_string(value, where)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{where} must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{where} must include a timezone offset")
    return result


def _execution_layer(value: Any, where: str) -> str:
    if value not in _EXECUTION_LAYERS:
        raise ValueError(f"{where} must be 'windows-native' or 'wsl2'")
    return str(value)


def _matching_gpu(
    gpus: Any,
    *,
    gpu_index: int,
    gpu_device_id_sha256: str,
    where: str,
) -> dict[str, Any]:
    if not isinstance(gpus, list):
        raise ValueError(f"{where} must be a list")
    matches = [
        gpu
        for gpu in gpus
        if isinstance(gpu, dict)
        and gpu.get("index") == gpu_index
        and gpu.get("device_id_sha256") == gpu_device_id_sha256
    ]
    if len(matches) != 1:
        raise ValueError(f"{where} must identify exactly one selected GPU")
    return matches[0]


def _validate_calibration(record: dict[str, Any], where: str) -> dict[str, Any]:
    _required_fields(
        record,
        {
            "schema_version",
            "calibration_id",
            "status",
            "host_id",
            "execution_layer",
            "created_at",
            "hardware",
            "wall_meter",
            "counter_backends",
            "timing",
            "criteria",
            "workload_classes",
            "summary",
            "evidence",
            "approval",
        },
        where,
    )
    if record["schema_version"] != CALIBRATION_SCHEMA or record["status"] != "passed":
        raise ValueError(f"{where} must be a passed {CALIBRATION_SCHEMA} record")
    host_id = _nonempty_string(record["host_id"], f"{where}.host_id")
    layer = _execution_layer(record["execution_layer"], f"{where}.execution_layer")
    calibration_id = _nonempty_string(record["calibration_id"], f"{where}.calibration_id")
    created_at = _aware_datetime(record["created_at"], f"{where}.created_at")

    hardware = _mapping(record["hardware"], f"{where}.hardware")
    _required_fields(
        hardware,
        {"gpu_index", "gpu_device_id_sha256", "cpu", "gpu"},
        f"{where}.hardware",
    )
    gpu_index = _integer(hardware["gpu_index"], f"{where}.hardware.gpu_index")
    gpu_hash = _sha256(
        hardware["gpu_device_id_sha256"],
        f"{where}.hardware.gpu_device_id_sha256",
    )
    _nonempty_string(hardware["cpu"], f"{where}.hardware.cpu")
    _nonempty_string(hardware["gpu"], f"{where}.hardware.gpu")

    wall_meter = _mapping(record["wall_meter"], f"{where}.wall_meter")
    _required_fields(
        wall_meter,
        {
            "required",
            "model",
            "device_id_hash",
            "accuracy_percent",
            "sample_interval_s",
            "data_interface",
        },
        f"{where}.wall_meter",
    )
    if wall_meter["required"] is not True:
        raise ValueError(f"{where}.wall_meter.required must be true")
    _nonempty_string(wall_meter["model"], f"{where}.wall_meter.model")
    meter_hash = _sha256(wall_meter["device_id_hash"], f"{where}.wall_meter.device_id_hash")
    if _number(wall_meter["accuracy_percent"], f"{where}.wall_meter.accuracy_percent") > 2.0:
        raise ValueError(f"{where}.wall_meter.accuracy_percent exceeds 2.0")
    sample_interval = _number(
        wall_meter["sample_interval_s"],
        f"{where}.wall_meter.sample_interval_s",
        minimum=1e-12,
    )
    if sample_interval > 1.0:
        raise ValueError(f"{where}.wall_meter.sample_interval_s exceeds 1.0")
    if wall_meter["data_interface"] not in {"api", "csv", "json"}:
        raise ValueError(f"{where}.wall_meter.data_interface is unsupported")

    counters = _mapping(record["counter_backends"], f"{where}.counter_backends")
    _required_fields(
        counters,
        {
            "system_primary",
            "gpu_diagnostic",
            "strict_backend",
            "pue",
            "report_embodied",
        },
        f"{where}.counter_backends",
    )
    if (
        counters["system_primary"] != "wall_meter"
        or counters["gpu_diagnostic"] != "nvml"
        or counters["strict_backend"] is not True
        or counters["pue"] != 1.0
        or counters["report_embodied"] is not False
    ):
        raise ValueError(f"{where}.counter_backends violates the confirmatory contract")

    timing = _mapping(record["timing"], f"{where}.timing")
    alignment = _number(
        timing.get("timestamp_alignment_error_s"),
        f"{where}.timing.timestamp_alignment_error_s",
    )
    observed_missing_percent = _number(
        timing.get("missing_sample_percent"),
        f"{where}.timing.missing_sample_percent",
    )
    if alignment > 0.5 or observed_missing_percent > 0.5:
        raise ValueError(f"{where}.timing exceeds the confirmatory thresholds")

    criteria = _mapping(record["criteria"], f"{where}.criteria")
    repeats = _integer(
        criteria.get("repeats_per_workload_class"),
        f"{where}.criteria.repeats_per_workload_class",
        minimum=5,
    )
    if repeats != 5:
        raise ValueError(f"{where}.criteria.repeats_per_workload_class must be exactly 5")
    minimum_duration = _number(
        criteria.get("minimum_block_duration_s"),
        f"{where}.criteria.minimum_block_duration_s",
        minimum=120.0,
    )
    minimum_samples = _integer(
        criteria.get("minimum_samples_per_block"),
        f"{where}.criteria.minimum_samples_per_block",
        minimum=100,
    )
    maximum_repeat_cv = _number(
        criteria.get("max_repeat_cv_percent"),
        f"{where}.criteria.max_repeat_cv_percent",
    )
    maximum_coverage_cv = _number(
        criteria.get("max_coverage_ratio_cv_percent"),
        f"{where}.criteria.max_coverage_ratio_cv_percent",
    )
    maximum_missing_percent = _number(
        criteria.get("max_missing_sample_percent"),
        f"{where}.criteria.max_missing_sample_percent",
    )
    if maximum_repeat_cv > 5.0 or maximum_coverage_cv > 5.0:
        raise ValueError(f"{where}.criteria permits excessive variability")
    if maximum_missing_percent > 0.5 or observed_missing_percent > maximum_missing_percent:
        raise ValueError(f"{where}.criteria permits excessive missing samples")

    workload_classes = _mapping(record["workload_classes"], f"{where}.workload_classes")
    if set(workload_classes) != {"idle", "cpu_only", "gpu_only", "combined"}:
        raise ValueError(f"{where}.workload_classes must contain the four required classes")
    workload_trace_hashes: dict[tuple[str, int], str] = {}
    for workload_name, blocks in workload_classes.items():
        block_where = f"{where}.workload_classes.{workload_name}"
        if not isinstance(blocks, list) or len(blocks) != repeats:
            raise ValueError(f"{block_where} must contain exactly {repeats} blocks")
        for index, raw_block in enumerate(blocks):
            block = _mapping(raw_block, f"{block_where}[{index}]")
            if (
                _number(block.get("duration_s"), f"{block_where}[{index}].duration_s")
                < minimum_duration
            ):
                raise ValueError(f"{block_where}[{index}] is too short")
            _integer(
                block.get("sample_count"),
                f"{block_where}[{index}].sample_count",
                minimum=minimum_samples,
            )
            workload_trace_hashes[(workload_name, index)] = _sha256(
                block.get("raw_sha256"), f"{block_where}[{index}].raw_sha256"
            )

    summary = _mapping(record["summary"], f"{where}.summary")
    if (
        _number(
            summary.get("meter_repeat_cv_percent"),
            f"{where}.summary.meter_repeat_cv_percent",
        )
        > maximum_repeat_cv
        or _number(
            summary.get("coverage_ratio_cv_percent"),
            f"{where}.summary.coverage_ratio_cv_percent",
        )
        > maximum_coverage_cv
        or summary.get("nvml_positive_and_monotonic") is not True
        or summary.get("prohibited_fallback_observed") is not False
    ):
        raise ValueError(f"{where}.summary does not pass calibration")

    evidence = _mapping(record["evidence"], f"{where}.evidence")
    raw_trace_manifest_path = _nonempty_string(
        evidence.get("raw_trace_manifest"),
        f"{where}.evidence.raw_trace_manifest",
    )
    raw_trace_manifest_sha256 = _sha256(
        evidence.get("raw_trace_manifest_sha256"),
        f"{where}.evidence.raw_trace_manifest_sha256",
    )
    analysis_script_path = _nonempty_string(
        evidence.get("analysis_script"),
        f"{where}.evidence.analysis_script",
    )
    analysis_script_sha256 = _sha256(
        evidence.get("analysis_script_sha256"),
        f"{where}.evidence.analysis_script_sha256",
    )
    approval = _mapping(record["approval"], f"{where}.approval")
    passed_at = _aware_datetime(approval.get("passed_at"), f"{where}.approval.passed_at")
    if passed_at < created_at:
        raise ValueError(f"{where}.approval.passed_at precedes created_at")
    _nonempty_string(approval.get("approved_by"), f"{where}.approval.approved_by")
    return {
        "host_id": host_id,
        "execution_layer": layer,
        "calibration_id": calibration_id,
        "gpu_index": gpu_index,
        "gpu_device_id_sha256": gpu_hash,
        "wall_meter_device_id_sha256": meter_hash,
        "sample_interval_s": sample_interval,
        "minimum_block_duration_s": minimum_duration,
        "max_missing_sample_percent": maximum_missing_percent,
        "raw_trace_manifest_path": raw_trace_manifest_path,
        "raw_trace_manifest_sha256": raw_trace_manifest_sha256,
        "analysis_script_sha256": analysis_script_sha256,
        "analysis_script_path": analysis_script_path,
        "workload_trace_hashes": workload_trace_hashes,
    }


def _validate_preflight(record: dict[str, Any], where: str) -> dict[str, Any]:
    if record.get("schema_version") != PREFLIGHT_SCHEMA:
        raise ValueError(f"{where}.schema_version must be {PREFLIGHT_SCHEMA!r}")
    host_id = _nonempty_string(record.get("host_id"), f"{where}.host_id")
    _aware_datetime(record.get("generated_at"), f"{where}.generated_at")
    system = _mapping(record.get("system"), f"{where}.system")
    layer = _execution_layer(system.get("execution_layer"), f"{where}.system.execution_layer")
    git = _mapping(record.get("git"), f"{where}.git")
    if git.get("available") is not True or git.get("dirty") is not False:
        raise ValueError(f"{where}.git must be available and clean")
    git_commit = _git_sha(git.get("sha"), f"{where}.git.sha")
    readiness = _mapping(record.get("readiness"), f"{where}.readiness")
    for field in (
        "environment_ready",
        "git_clean",
        "gpu_component_ready",
        "calibration_compatible",
        "system_energy_ready",
        "measured_smoke_ready",
        "confirmatory_primary_ready",
    ):
        if readiness.get(field) is not True:
            raise ValueError(f"{where}.readiness.{field} must be true")
    strict_tracker = _mapping(record.get("strict_tracker"), f"{where}.strict_tracker")
    if strict_tracker.get("importable") is not True or strict_tracker.get("supported") is not True:
        raise ValueError(f"{where}.strict_tracker must be importable and supported")
    backends = _mapping(record.get("backends"), f"{where}.backends")
    if (
        backends.get("codecarbon_allowed_primary") is not False
        or backends.get("tdp_allowed_primary") is not False
    ):
        raise ValueError(f"{where}.backends permits a prohibited primary backend")
    nvml = _mapping(backends.get("nvml"), f"{where}.backends.nvml")
    selected = _mapping(nvml.get("selected_device"), f"{where}.backends.nvml.selected_device")
    gpu_index = _integer(nvml.get("selected_index"), f"{where}.backends.nvml.selected_index")
    if (
        nvml.get("available") is not True
        or nvml.get("active_probe") is not True
        or nvml.get("selection_valid") is not True
        or selected.get("index") != gpu_index
        or selected.get("selected") is not True
        or selected.get("measurement_mode_supported") is not True
    ):
        raise ValueError(f"{where}.backends.nvml did not pass its active selected-device probe")
    gpu_hash = _sha256(
        selected.get("device_id_sha256"),
        f"{where}.backends.nvml.selected_device.device_id_sha256",
    )
    calibration = _mapping(record.get("calibration"), f"{where}.calibration")
    if (
        calibration.get("passed") is not True
        or calibration.get("schema_version") != CALIBRATION_SCHEMA
    ):
        raise ValueError(f"{where}.calibration must be a passed {CALIBRATION_SCHEMA} record")
    calibration_sha = _sha256(calibration.get("sha256"), f"{where}.calibration.sha256")
    if (
        calibration.get("host_id") != host_id
        or calibration.get("execution_layer") != layer
        or calibration.get("gpu_index") != gpu_index
        or calibration.get("gpu_device_id_sha256") != gpu_hash
    ):
        raise ValueError(f"{where}.calibration identity does not match the preflight")
    return {
        "host_id": host_id,
        "execution_layer": layer,
        "git_sha": git_commit,
        "gpu_index": gpu_index,
        "gpu_device_id_sha256": gpu_hash,
        "calibration_sha256": calibration_sha,
    }


def _validate_fingerprints(value: Any, where: str) -> None:
    fingerprints = _mapping(value, where)
    if not fingerprints:
        raise ValueError(f"{where} must be a non-empty object")
    for name, raw_fingerprint in fingerprints.items():
        _nonempty_string(name, f"{where} key")
        fingerprint_where = f"{where}.{name}"
        fingerprint = _mapping(raw_fingerprint, fingerprint_where)
        if set(fingerprint) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"{fingerprint_where} has an invalid schema")
        path_value = _nonempty_string(fingerprint["path"], f"{fingerprint_where}.path")
        path = Path(path_value)
        if (
            "\\" in path_value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"{fingerprint_where}.path must be a portable relative path")
        _integer(fingerprint["size_bytes"], f"{fingerprint_where}.size_bytes", minimum=1)
        _sha256(fingerprint["sha256"], f"{fingerprint_where}.sha256")


def _validate_runtime_controls(
    value: Any,
    where: str,
    *,
    execution_layer: str,
) -> dict[str, Any]:
    controls = _mapping(value, where)
    if set(controls) != _RUNTIME_CONTROL_KEYS:
        raise ValueError(f"{where} must follow the closed {RUNTIME_CONTROLS_SCHEMA} schema")
    if controls["schema_version"] != RUNTIME_CONTROLS_SCHEMA:
        raise ValueError(f"{where}.schema_version must be {RUNTIME_CONTROLS_SCHEMA!r}")
    for field in (
        "windows_build",
        "windows_power_plan",
        "cpu_power_mode",
        "gpu_driver",
        "cuda_version",
        "gpu_clock_policy",
        "warmup_stability_criterion",
        "background_load_check",
    ):
        _nonempty_string(controls[field], f"{where}.{field}")
    if execution_layer == "wsl2":
        _nonempty_string(controls["wsl_version"], f"{where}.wsl_version")
        _nonempty_string(controls["wsl_kernel"], f"{where}.wsl_kernel")
        _integer(controls["wsl_memory_limit_b"], f"{where}.wsl_memory_limit_b", minimum=1)
    elif any(
        controls[field] is not None for field in ("wsl_version", "wsl_kernel", "wsl_memory_limit_b")
    ):
        raise ValueError(f"{where} Windows-native WSL controls must be null")
    logical_processors = _integer(
        controls["cpu_logical_processors"],
        f"{where}.cpu_logical_processors",
        minimum=1,
    )
    cpu_threads = _integer(controls["cpu_threads"], f"{where}.cpu_threads", minimum=1)
    if cpu_threads > logical_processors:
        raise ValueError(f"{where}.cpu_threads exceeds cpu_logical_processors")
    affinity = controls["cpu_affinity"]
    if (
        not isinstance(affinity, list)
        or len(affinity) != cpu_threads
        or any(isinstance(index, bool) or not isinstance(index, int) for index in affinity)
        or len(set(affinity)) != len(affinity)
        or any(index < 0 or index >= logical_processors for index in affinity)
    ):
        raise ValueError(f"{where}.cpu_affinity is invalid for the declared CPU controls")
    for field in (
        "gpu_power_limit_w",
        "gpu_graphics_clock_mhz",
        "gpu_memory_clock_mhz",
        "warmup_duration_s",
    ):
        _number(controls[field], f"{where}.{field}", minimum=1e-12)
    if controls["gpu_persistence_mode"] not in {"enabled", "disabled", "unsupported"}:
        raise ValueError(f"{where}.gpu_persistence_mode is invalid")
    if controls["warmup_completed"] is not True:
        raise ValueError(f"{where}.warmup_completed must be true")
    _sha256(controls["warmup_trace_sha256"], f"{where}.warmup_trace_sha256")
    if controls["background_load_status"] != "clear":
        raise ValueError(f"{where}.background_load_status must be 'clear'")
    _sha256(
        controls["background_load_trace_sha256"],
        f"{where}.background_load_trace_sha256",
    )
    service_boundary = _mapping(controls["service_boundary"], f"{where}.service_boundary")
    if set(service_boundary) != {"mode", "initialization_reuse", "includes", "excludes"}:
        raise ValueError(f"{where}.service_boundary has an invalid schema")
    if service_boundary["mode"] not in {"training", "warm", "cold"}:
        raise ValueError(f"{where}.service_boundary.mode is invalid")
    if not isinstance(service_boundary["initialization_reuse"], bool):
        raise ValueError(f"{where}.service_boundary.initialization_reuse must be boolean")
    normalized_boundary_lists: dict[str, list[str]] = {}
    for field in ("includes", "excludes"):
        values = service_boundary[field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item.strip() for item in values)
        ):
            raise ValueError(f"{where}.service_boundary.{field} must be non-empty and unique")
        normalized = [item.strip() for item in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{where}.service_boundary.{field} must be non-empty and unique")
        normalized_boundary_lists[field] = normalized
    if set(normalized_boundary_lists["includes"]).intersection(
        normalized_boundary_lists["excludes"]
    ):
        raise ValueError(f"{where}.service_boundary includes and excludes must be disjoint")
    return controls


def _validate_provenance(record: dict[str, Any], where: str) -> dict[str, Any]:
    if record.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError(f"{where}.schema_version must be {PROVENANCE_SCHEMA!r}")
    _required_fields(
        record,
        {
            "python_version",
            "python_executable",
            "platform",
            "cpu",
            "memory_total_b",
            "library_versions",
            "lockfiles",
            "seeds",
            "runtime_controls",
            "artifacts",
        },
        where,
    )
    _nonempty_string(record["python_version"], f"{where}.python_version")
    python_executable = _nonempty_string(record["python_executable"], f"{where}.python_executable")
    if (
        python_executable != Path(python_executable).name
        or "/" in python_executable
        or "\\" in python_executable
    ):
        raise ValueError(f"{where}.python_executable must be a basename")
    _nonempty_string(record["platform"], f"{where}.platform")
    _nonempty_string(record["cpu"], f"{where}.cpu")
    _integer(record["memory_total_b"], f"{where}.memory_total_b", minimum=1)
    library_versions = _mapping(record["library_versions"], f"{where}.library_versions")
    if not library_versions:
        raise ValueError(f"{where}.library_versions must be non-empty")
    for library, version in library_versions.items():
        _nonempty_string(library, f"{where}.library_versions key")
        _nonempty_string(version, f"{where}.library_versions.{library}")
    _validate_fingerprints(record["lockfiles"], f"{where}.lockfiles")
    _validate_fingerprints(record["artifacts"], f"{where}.artifacts")
    seeds = _mapping(record["seeds"], f"{where}.seeds")
    if not seeds:
        raise ValueError(f"{where}.seeds must be non-empty")
    for seed_name, seed_value in seeds.items():
        _nonempty_string(seed_name, f"{where}.seeds key")
        _integer(seed_value, f"{where}.seeds.{seed_name}")
    layer = _execution_layer(record.get("execution_layer"), f"{where}.execution_layer")
    runtime_controls = _validate_runtime_controls(
        record["runtime_controls"],
        f"{where}.runtime_controls",
        execution_layer=layer,
    )
    if record.get("status") != "completed" or record.get("failure_reason") is not None:
        raise ValueError(f"{where} must describe a completed run without failure")
    if record.get("git_dirty") is not False or record.get("dirty_patch_sha256") is not None:
        raise ValueError(f"{where} must describe a clean Git worktree")
    started = _aware_datetime(record.get("timestamp_start"), f"{where}.timestamp_start")
    ended = _aware_datetime(record.get("timestamp_end"), f"{where}.timestamp_end")
    if ended < started:
        raise ValueError(f"{where}.timestamp_end precedes timestamp_start")
    run_id = _nonempty_string(record.get("run_id"), f"{where}.run_id")
    host_id = _nonempty_string(record.get("host_id"), f"{where}.host_id")
    git_commit = _git_sha(record.get("git_sha"), f"{where}.git_sha")
    _sha256(record.get("environment_hash"), f"{where}.environment_hash")
    config = _mapping(record.get("config"), f"{where}.config")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    if record.get("config_sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError(f"{where}.config_sha256 does not match config")
    measurement = _mapping(record.get("measurement"), f"{where}.measurement")
    _required_fields(
        measurement,
        {
            "primary_backend",
            "pue",
            "fallback",
            "preflight_sha256",
            "calibration_sha256",
            "gpu_index",
            "gpu_device_id_sha256",
        },
        f"{where}.measurement",
    )
    if (
        measurement["primary_backend"] != "wall_meter"
        or measurement["pue"] != 1.0
        or measurement["fallback"] is not False
    ):
        raise ValueError(f"{where}.measurement violates the confirmatory contract")
    gpu_index = _integer(measurement["gpu_index"], f"{where}.measurement.gpu_index")
    gpu_hash = _sha256(
        measurement["gpu_device_id_sha256"],
        f"{where}.measurement.gpu_device_id_sha256",
    )
    _matching_gpu(
        record.get("gpus"),
        gpu_index=gpu_index,
        gpu_device_id_sha256=gpu_hash,
        where=f"{where}.gpus",
    )
    training_dataset_hash = measurement.get("training_dataset_manifest_sha256")
    if training_dataset_hash is not None:
        training_dataset_hash = _sha256(
            training_dataset_hash,
            f"{where}.measurement.training_dataset_manifest_sha256",
        )
    return {
        "run_id": run_id,
        "host_id": host_id,
        "execution_layer": layer,
        "git_sha": git_commit,
        "gpu_index": gpu_index,
        "gpu_device_id_sha256": gpu_hash,
        "preflight_sha256": _sha256(
            measurement["preflight_sha256"], f"{where}.measurement.preflight_sha256"
        ),
        "calibration_sha256": _sha256(
            measurement["calibration_sha256"],
            f"{where}.measurement.calibration_sha256",
        ),
        "training_dataset_manifest_sha256": training_dataset_hash,
        "diagnostic_backend": measurement.get("diagnostic_backend"),
        "service_boundary": runtime_controls["service_boundary"],
    }


def _validate_instance_manifest(record: dict[str, Any], where: str) -> dict[str, Any]:
    if record.get("schema_version") != INSTANCE_MANIFEST_SCHEMA:
        raise ValueError(f"{where}.schema_version must be {INSTANCE_MANIFEST_SCHEMA!r}")
    if record.get("status") != "complete" or record.get("complete") is not True:
        raise ValueError(f"{where} must be complete")
    if record.get("split") != "confirmatory":
        raise ValueError(f"{where}.split must be 'confirmatory'")
    item_count = _integer(record.get("item_count"), f"{where}.item_count", minimum=1)
    entries = record.get("entries")
    if not isinstance(entries, list) or len(entries) != item_count:
        raise ValueError(f"{where}.entries must contain exactly item_count records")
    identifiers: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"{where}.entries[{index}]")
        if set(entry) != {"instance_id", "sha256"}:
            raise ValueError(f"{where}.entries[{index}] has an invalid schema")
        identifier = _nonempty_string(entry["instance_id"], f"{where}.entries[{index}].instance_id")
        if identifier in identifiers:
            raise ValueError(f"{where}.entries contains duplicate instance_id values")
        identifiers.add(identifier)
        _sha256(entry["sha256"], f"{where}.entries[{index}].sha256")
    return {
        "manifest_id": _nonempty_string(record.get("manifest_id"), f"{where}.manifest_id"),
        "problem": _nonempty_string(record.get("problem"), f"{where}.problem"),
        "size": _integer(record.get("size"), f"{where}.size", minimum=1),
        "item_count": item_count,
    }


def _validate_training_dataset_manifest(record: dict[str, Any], where: str) -> dict[str, Any]:
    if record.get("schema_version") != TRAINING_DATASET_MANIFEST_SCHEMA:
        raise ValueError(f"{where}.schema_version must be {TRAINING_DATASET_MANIFEST_SCHEMA!r}")
    if record.get("status") != "complete" or record.get("complete") is not True:
        raise ValueError(f"{where} must be complete")
    if record.get("split") != "training":
        raise ValueError(f"{where}.split must be 'training'")
    item_count = _integer(record.get("item_count"), f"{where}.item_count", minimum=1)
    generation = _mapping(record.get("generation"), f"{where}.generation")
    if set(generation) != {"method", "seed", "config_sha256"}:
        raise ValueError(f"{where}.generation has an invalid schema")
    _nonempty_string(generation["method"], f"{where}.generation.method")
    _integer(generation["seed"], f"{where}.generation.seed")
    _sha256(generation["config_sha256"], f"{where}.generation.config_sha256")
    shards = record.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"{where}.shards must be a non-empty list")
    total_items = 0
    shard_paths: set[str] = set()
    for index, raw_shard in enumerate(shards):
        shard_where = f"{where}.shards[{index}]"
        shard = _mapping(raw_shard, shard_where)
        if set(shard) != {"path", "sha256", "item_count"}:
            raise ValueError(f"{shard_where} has an invalid schema")
        shard_path = _nonempty_string(shard["path"], f"{shard_where}.path")
        relative = Path(shard_path)
        if (
            "\\" in shard_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"{shard_where}.path must be a portable relative path")
        if shard_path in shard_paths:
            raise ValueError(f"{where}.shards contains duplicate paths")
        shard_paths.add(shard_path)
        _sha256(shard["sha256"], f"{shard_where}.sha256")
        total_items += _integer(shard["item_count"], f"{shard_where}.item_count", minimum=1)
    if total_items != item_count:
        raise ValueError(f"{where}.shards item counts do not sum to item_count")
    return {
        "dataset_id": _nonempty_string(record.get("dataset_id"), f"{where}.dataset_id"),
        "problem": _nonempty_string(record.get("problem"), f"{where}.problem"),
        "size": _integer(record.get("size"), f"{where}.size", minimum=1),
        "item_count": item_count,
    }


def _validate_calibration_trace_manifest(record: dict[str, Any], where: str) -> dict[str, Any]:
    if set(record) != {"schema_version", "traces"}:
        raise ValueError(f"{where} has an invalid schema")
    if record.get("schema_version") != CALIBRATION_TRACE_MANIFEST_SCHEMA:
        raise ValueError(f"{where}.schema_version must be {CALIBRATION_TRACE_MANIFEST_SCHEMA!r}")
    traces = record.get("traces")
    if not isinstance(traces, list) or len(traces) != 20:
        raise ValueError(f"{where}.traces must contain exactly 20 trace records")
    expected_blocks = {
        (workload_class, block_index)
        for workload_class in ("idle", "cpu_only", "gpu_only", "combined")
        for block_index in range(5)
    }
    observed_blocks: set[tuple[str, int]] = set()
    hashes_by_block: dict[tuple[str, int], str] = {}
    paths: dict[str, str] = {}
    unique_hashes: set[str] = set()
    for index, raw_trace in enumerate(traces):
        trace_where = f"{where}.traces[{index}]"
        trace_record = _mapping(raw_trace, trace_where)
        if set(trace_record) != {
            "raw_trace",
            "raw_sha256",
            "workload_class",
            "block_index",
        }:
            raise ValueError(f"{trace_where} has an invalid schema")
        trace_path = _nonempty_string(trace_record["raw_trace"], f"{trace_where}.raw_trace")
        relative = Path(trace_path)
        if (
            "\\" in trace_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"{trace_where}.raw_trace must be a portable relative path")
        if trace_path in paths:
            raise ValueError(f"{where}.traces contains duplicate paths")
        trace_sha = _sha256(trace_record["raw_sha256"], f"{trace_where}.raw_sha256")
        if trace_sha in unique_hashes:
            raise ValueError(f"{where}.traces contains duplicate raw_sha256 values")
        unique_hashes.add(trace_sha)
        paths[trace_path] = trace_sha
        workload_class = trace_record["workload_class"]
        if workload_class not in {"idle", "cpu_only", "gpu_only", "combined"}:
            raise ValueError(f"{trace_where}.workload_class is invalid")
        block_index = _integer(trace_record["block_index"], f"{trace_where}.block_index")
        coordinate = (str(workload_class), block_index)
        if coordinate in observed_blocks:
            raise ValueError(f"{where}.traces contains duplicate workload blocks")
        observed_blocks.add(coordinate)
        hashes_by_block[coordinate] = trace_sha
    if observed_blocks != expected_blocks:
        raise ValueError(f"{where}.traces must cover blocks 0 through 4 for all four classes")
    return {"files": paths, "hashes_by_block": hashes_by_block}


def _validate_validation(
    record: dict[str, Any],
    where: str,
    *,
    purpose: str,
) -> dict[str, Any]:
    if record.get("schema_version") != VALIDATION_SCHEMA:
        raise ValueError(f"{where}.schema_version must be {VALIDATION_SCHEMA!r}")
    required = {
        "run_id",
        "role",
        "status",
        "complete",
        "independent",
        "validator",
        "checks",
        "metrics",
        "host_id",
        "execution_layer",
        "git_sha",
        "gpu_index",
        "gpu_device_id_sha256",
        "preflight_sha256",
        "calibration_sha256",
        "provenance_sha256",
        "instance_manifest_sha256",
        "training_dataset_manifest_sha256",
        "expected_records",
        "validated_records",
        "failure_count",
        "excluded_count",
        "gap_to_reference_pct",
        "quality_feasible_confirmatory",
    }
    _required_fields(record, required, where)
    role = record["role"]
    if role not in _ENERGY_ROLES:
        raise ValueError(f"{where}.role is invalid")
    if (
        record["status"] != "passed"
        or record["complete"] is not True
        or record["independent"] is not True
    ):
        raise ValueError(f"{where} must be complete, independently validated, and passed")
    _nonempty_string(record["validator"], f"{where}.validator")
    checks = record["checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{where}.checks must be a non-empty list")
    evidence_hashes: set[str] = set()
    for index, raw_check in enumerate(checks):
        check_where = f"{where}.checks[{index}]"
        check = _mapping(raw_check, check_where)
        if set(check) != {"name", "passed", "evidence_sha256"}:
            raise ValueError(f"{check_where} has an invalid schema")
        _nonempty_string(check["name"], f"{check_where}.name")
        if check["passed"] is not True:
            raise ValueError(f"{check_where} did not pass")
        evidence_hashes.add(_sha256(check["evidence_sha256"], f"{check_where}.evidence_sha256"))
    _mapping(record["metrics"], f"{where}.metrics")
    expected = _integer(record["expected_records"], f"{where}.expected_records", minimum=1)
    validated = _integer(record["validated_records"], f"{where}.validated_records", minimum=0)
    failures = _integer(record["failure_count"], f"{where}.failure_count", minimum=0)
    exclusions = _integer(record["excluded_count"], f"{where}.excluded_count", minimum=0)
    if validated != expected or failures != 0 or exclusions != 0:
        raise ValueError(f"{where} contains partial, failed, or excluded validation data")
    instance_hash = record["instance_manifest_sha256"]
    training_dataset_hash = record["training_dataset_manifest_sha256"]
    gap = record["gap_to_reference_pct"]
    quality = record["quality_feasible_confirmatory"]
    if role == "training":
        if instance_hash is not None or gap is not None or quality is not None:
            raise ValueError(f"{where} training quality and instance fields must be null")
        training_dataset_hash = _sha256(
            training_dataset_hash, f"{where}.training_dataset_manifest_sha256"
        )
    else:
        if training_dataset_hash is not None:
            raise ValueError(
                f"{where}.training_dataset_manifest_sha256 must be null outside training"
            )
        instance_hash = _sha256(instance_hash, f"{where}.instance_manifest_sha256")
        gap = _number(gap, f"{where}.gap_to_reference_pct")
        if purpose == "confirmatory" and not isinstance(quality, bool):
            raise ValueError(f"{where}.quality_feasible_confirmatory must be a boolean")
        if purpose == "instrumentation_smoke" and quality is not None:
            raise ValueError(
                f"{where}.quality_feasible_confirmatory must be null for instrumentation smoke"
            )
    return {
        "run_id": _nonempty_string(record["run_id"], f"{where}.run_id"),
        "role": role,
        "host_id": _nonempty_string(record["host_id"], f"{where}.host_id"),
        "execution_layer": _execution_layer(record["execution_layer"], f"{where}.execution_layer"),
        "git_sha": _git_sha(record["git_sha"], f"{where}.git_sha"),
        "gpu_index": _integer(record["gpu_index"], f"{where}.gpu_index"),
        "gpu_device_id_sha256": _sha256(
            record["gpu_device_id_sha256"], f"{where}.gpu_device_id_sha256"
        ),
        "preflight_sha256": _sha256(record["preflight_sha256"], f"{where}.preflight_sha256"),
        "calibration_sha256": _sha256(record["calibration_sha256"], f"{where}.calibration_sha256"),
        "provenance_sha256": _sha256(record["provenance_sha256"], f"{where}.provenance_sha256"),
        "instance_manifest_sha256": instance_hash,
        "training_dataset_manifest_sha256": training_dataset_hash,
        "expected_records": expected,
        "gap_to_reference_pct": gap,
        "quality_feasible_confirmatory": quality,
        "evidence_sha256s": evidence_hashes,
    }


def _validate_energy_record(
    record: dict[str, Any],
    role: str,
    where: str,
    *,
    purpose: str,
) -> dict[str, Any]:
    if set(record) != _ENERGY_KEYS:
        raise ValueError(f"{where} has an invalid canonical energy schema")
    if record.get("schema_version") != "1.0":
        raise ValueError(f"{where}.schema_version must be '1.0'")
    if record.get("units") != _ENERGY_UNITS:
        raise ValueError(f"{where}.units do not match the canonical energy schema")
    if record.get("backend") != "wall_meter":
        raise ValueError(f"{where}.backend must be 'wall_meter'")
    if record.get("measurement_scope") != "whole_system_ac" or record.get("energy_domains") != [
        "whole_system_ac"
    ]:
        raise ValueError(f"{where} must measure whole-system AC energy")
    duration_s = _number(record.get("duration_s"), f"{where}.duration_s", minimum=1e-12)
    energy_j = _number(record.get("energy_j"), f"{where}.energy_j", minimum=1e-12)
    component_energy = sum(
        _number(record.get(field), f"{where}.{field}")
        for field in ("energy_gpu_j", "energy_cpu_j", "energy_dram_j")
    )
    if component_energy > energy_j + max(1e-9, energy_j * 1e-9):
        raise ValueError(f"{where} component energy exceeds total energy_j")
    operational_co2 = _number(record.get("co2_operational_kg"), f"{where}.co2_operational_kg")
    embodied_co2 = _number(record.get("co2_embodied_kg"), f"{where}.co2_embodied_kg")
    total_co2 = _number(record.get("co2_total_kg"), f"{where}.co2_total_kg")
    if not math.isclose(
        total_co2,
        operational_co2 + embodied_co2,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{where}.co2_total_kg is inconsistent")
    average_power = _number(record.get("avg_power_w"), f"{where}.avg_power_w")
    if not math.isclose(
        average_power,
        energy_j / duration_s,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{where}.avg_power_w is inconsistent")
    items = _integer(record.get("items_processed"), f"{where}.items_processed")
    if role in {"inference", "baseline"} and items < 1:
        raise ValueError(f"{where}.items_processed must be positive")
    throughput = _number(record.get("throughput_items_per_s"), f"{where}.throughput_items_per_s")
    if not math.isclose(
        throughput,
        items / duration_s,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{where}.throughput_items_per_s is inconsistent")
    hardware = _mapping(record.get("hardware"), f"{where}.hardware")
    if hardware.get("pue") != 1.0:
        raise ValueError(f"{where}.hardware.pue must be exactly 1.0")
    extra = _mapping(record.get("extra"), f"{where}.extra")
    if extra.get("run_status") != "completed" or extra.get("failure_type") is not None:
        raise ValueError(f"{where} must describe a completed measurement without failure")
    if extra.get("fallback") is not False or (
        extra.get("allow_fallback") is not False or extra.get("tdp_fallback") is not False
    ):
        raise ValueError(f"{where} permits a prohibited measurement fallback")
    if purpose == "confirmatory" and (
        extra.get("measurement_qualified_confirmatory") is not True
        or extra.get("measurement_qualified_instrumentation") not in (None, False)
        or extra.get("purpose") != "confirmatory"
        or extra.get("scientific_use") is not True
    ):
        raise ValueError(f"{where} is not a strict confirmatory measurement")
    if purpose == "instrumentation_smoke" and (
        extra.get("measurement_qualified_instrumentation") is not True
        or extra.get("measurement_qualified_confirmatory") not in (None, False)
        or extra.get("purpose") != "instrumentation_smoke"
        or extra.get("scientific_use") is not False
    ):
        raise ValueError(f"{where} is not a strict instrumentation smoke measurement")
    if extra.get("role") != role:
        raise ValueError(f"{where}.extra.role does not match its manifest role")
    run_id = _nonempty_string(extra.get("run_id"), f"{where}.extra.run_id")
    wall_meter = _mapping(extra.get("wall_meter"), f"{where}.extra.wall_meter")
    sample_count = _integer(
        wall_meter.get("sample_count"), f"{where}.extra.wall_meter.sample_count", minimum=1
    )
    dropped_samples = _integer(
        wall_meter.get("dropped_sample_count"),
        f"{where}.extra.wall_meter.dropped_sample_count",
    )
    started = _aware_datetime(
        wall_meter.get("started_at_utc"), f"{where}.extra.wall_meter.started_at_utc"
    )
    ended = _aware_datetime(
        wall_meter.get("ended_at_utc"), f"{where}.extra.wall_meter.ended_at_utc"
    )
    if started.utcoffset() != timedelta(0) or ended.utcoffset() != timedelta(0) or ended <= started:
        raise ValueError(f"{where}.extra.wall_meter timestamps must be increasing UTC values")
    duration = _number(
        wall_meter.get("duration_s"),
        f"{where}.extra.wall_meter.duration_s",
        minimum=1e-12,
    )
    if not math.isclose(duration, float(record["duration_s"]), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{where}.extra.wall_meter.duration_s disagrees with duration_s")
    elapsed = (ended - started).total_seconds()
    if not math.isclose(duration, elapsed, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{where}.extra.wall_meter timestamps disagree with duration_s")
    gpu_index = _integer(extra.get("gpu_index"), f"{where}.extra.gpu_index")
    gpu_hash = _sha256(extra.get("gpu_device_id_sha256"), f"{where}.extra.gpu_device_id_sha256")
    nvml_trace_sha: str | None = None
    diagnostics = extra.get("diagnostics")
    if role in {"training", "inference"}:
        diagnostics_mapping = _mapping(diagnostics, f"{where}.extra.diagnostics")
        if set(diagnostics_mapping) != {"nvml"}:
            raise ValueError(f"{where}.extra.diagnostics must contain exactly nvml")
        nvml = _mapping(diagnostics_mapping["nvml"], f"{where}.extra.diagnostics.nvml")
        required_nvml_fields = {
            "schema_version",
            "source",
            "gpu_index",
            "device_id_sha256",
            "measurement_mode",
            "energy_j",
            "duration_s",
            "trace_sha256",
            "sampling_interval_s",
            "sample_count",
            "dropped_sample_count",
            "started_at_utc",
            "ended_at_utc",
        }
        if set(nvml) != required_nvml_fields:
            raise ValueError(f"{where}.extra.diagnostics.nvml has an invalid schema")
        if (
            nvml["schema_version"] != "aet-nvml-diagnostic/v1"
            or nvml["source"] != "independent_concurrent_sampler"
            or nvml["gpu_index"] != gpu_index
            or nvml["device_id_sha256"] != gpu_hash
        ):
            raise ValueError(f"{where}.extra.diagnostics.nvml identity is invalid")
        mode = nvml["measurement_mode"]
        if mode not in {"nvml_total_energy_counter", "nvml_power_integration"}:
            raise ValueError(f"{where}.extra.diagnostics.nvml.measurement_mode is invalid")
        _number(nvml["energy_j"], f"{where}.extra.diagnostics.nvml.energy_j", minimum=1e-12)
        nvml_duration = _number(
            nvml["duration_s"],
            f"{where}.extra.diagnostics.nvml.duration_s",
            minimum=1e-12,
        )
        if not math.isclose(nvml_duration, duration, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{where}.extra.diagnostics.nvml.duration_s is inconsistent")
        sampling_interval = nvml["sampling_interval_s"]
        if mode == "nvml_total_energy_counter":
            if sampling_interval is not None:
                raise ValueError(
                    f"{where}.extra.diagnostics.nvml.sampling_interval_s must be null "
                    "for a total counter"
                )
        else:
            _number(
                sampling_interval,
                f"{where}.extra.diagnostics.nvml.sampling_interval_s",
                minimum=1e-12,
            )
        _integer(
            nvml["sample_count"],
            f"{where}.extra.diagnostics.nvml.sample_count",
            minimum=2,
        )
        _integer(
            nvml["dropped_sample_count"],
            f"{where}.extra.diagnostics.nvml.dropped_sample_count",
        )
        nvml_started = _aware_datetime(
            nvml["started_at_utc"], f"{where}.extra.diagnostics.nvml.started_at_utc"
        )
        nvml_ended = _aware_datetime(
            nvml["ended_at_utc"], f"{where}.extra.diagnostics.nvml.ended_at_utc"
        )
        if nvml_started != started or nvml_ended != ended:
            raise ValueError(f"{where}.extra.diagnostics.nvml timestamps are inconsistent")
        nvml_trace_sha = _sha256(
            nvml["trace_sha256"], f"{where}.extra.diagnostics.nvml.trace_sha256"
        )
    elif diagnostics not in (None, {}):
        raise ValueError(f"{where} baseline must not contain an NVML diagnostic")
    quality_value = extra.get("quality_feasible_confirmatory")
    if role in {"inference", "baseline"}:
        if purpose == "confirmatory" and not isinstance(quality_value, bool):
            raise ValueError(f"{where}.extra.quality_feasible_confirmatory must be a boolean")
        if purpose == "instrumentation_smoke" and quality_value is not None:
            raise ValueError(
                f"{where}.extra.quality_feasible_confirmatory must be null for "
                "instrumentation smoke"
            )
    return {
        "run_id": run_id,
        "role": role,
        "host_id": _nonempty_string(extra.get("host_id"), f"{where}.extra.host_id"),
        "execution_layer": _execution_layer(
            extra.get("execution_layer"), f"{where}.extra.execution_layer"
        ),
        "git_sha": _git_sha(extra.get("git_sha"), f"{where}.extra.git_sha"),
        "gpu_index": gpu_index,
        "gpu_device_id_sha256": gpu_hash,
        "preflight_sha256": _sha256(
            extra.get("preflight_sha256"), f"{where}.extra.preflight_sha256"
        ),
        "calibration_sha256": _sha256(
            extra.get("calibration_sha256"), f"{where}.extra.calibration_sha256"
        ),
        "provenance_sha256": _sha256(
            extra.get("provenance_sha256"), f"{where}.extra.provenance_sha256"
        ),
        "validation_sha256": _sha256(
            extra.get("artifact_sha256"), f"{where}.extra.artifact_sha256"
        ),
        "instance_manifest_sha256": (
            _sha256(
                extra.get("instance_manifest_sha256"),
                f"{where}.extra.instance_manifest_sha256",
            )
            if role in {"inference", "baseline"}
            else None
        ),
        "training_dataset_manifest_sha256": (
            _sha256(
                extra.get("training_dataset_manifest_sha256"),
                f"{where}.extra.training_dataset_manifest_sha256",
            )
            if role == "training"
            else None
        ),
        "items_processed": items,
        "problem": _nonempty_string(extra.get("problem"), f"{where}.extra.problem"),
        "size": (
            _integer(extra.get("size_key"), f"{where}.extra.size_key", minimum=1)
            if role == "inference"
            else _integer(extra.get("size"), f"{where}.extra.size", minimum=1)
        ),
        "gap_to_reference_pct": (
            _number(
                extra.get("gap_to_reference_pct"),
                f"{where}.extra.gap_to_reference_pct",
            )
            if role in {"inference", "baseline"}
            else None
        ),
        "quality_feasible_confirmatory": (
            quality_value if role in {"inference", "baseline"} else None
        ),
        "wall_meter_calibration_id": _nonempty_string(
            wall_meter.get("calibration_id"),
            f"{where}.extra.wall_meter.calibration_id",
        ),
        "wall_meter_device_id_sha256": _sha256(
            wall_meter.get("device_id_sha256"),
            f"{where}.extra.wall_meter.device_id_sha256",
        ),
        "wall_meter_trace_sha256": _sha256(
            wall_meter.get("trace_sha256"), f"{where}.extra.wall_meter.trace_sha256"
        ),
        "nvml_trace_sha256": nvml_trace_sha,
        "wall_meter_sample_interval_s": _number(
            wall_meter.get("sample_interval_s"),
            f"{where}.extra.wall_meter.sample_interval_s",
            minimum=1e-12,
        ),
        "wall_meter_sample_count": sample_count,
        "wall_meter_dropped_sample_count": dropped_samples,
        "duration_s": duration,
    }


def _load_expected_manifest(
    runs_root: Path,
    manifest_path: Path,
) -> LoadedInputManifest:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read AET input manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid AET input manifest JSON {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != INPUT_MANIFEST_SCHEMA:
        raise ValueError(f"input manifest schema must be {INPUT_MANIFEST_SCHEMA!r}")
    if set(payload) != {"schema_version", "purpose", "confirmatory_eligible", "files"}:
        raise ValueError("input manifest must use the closed v1 top-level schema")
    purpose = payload["purpose"]
    if purpose not in {"confirmatory", "instrumentation_smoke"}:
        raise ValueError("input manifest purpose must be confirmatory or instrumentation_smoke")
    confirmatory_eligible = payload["confirmatory_eligible"]
    if not isinstance(confirmatory_eligible, bool):
        raise ValueError("input manifest confirmatory_eligible must be a boolean")
    if purpose == "instrumentation_smoke" and confirmatory_eligible:
        raise ValueError("an instrumentation_smoke manifest cannot be confirmatory eligible")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("input manifest files must be a non-empty list")

    runs_root = runs_root.resolve()
    manifest_path = manifest_path.resolve()
    by_role: dict[str, list[Path]] = {role: [] for role in _ROLE_BASENAMES}
    seen: set[Path] = set()
    records_by_path: dict[Path, dict[str, Any]] = {}
    run_scoped_opaque_roles = {"wall_trace", "nvml_trace", "validation_evidence"}
    opaque_run_ids: dict[str, dict[str, str]] = {role: {} for role in run_scoped_opaque_roles}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"input manifest files[{index}] has an invalid schema")
        role = entry.get("role")
        if role not in _ROLE_BASENAMES:
            raise ValueError(f"input manifest files[{index}] has an unknown role")
        expected_entry_keys = {"role", "path", "sha256"}
        if role in run_scoped_opaque_roles:
            expected_entry_keys.add("run_id")
        if set(entry) != expected_entry_keys:
            raise ValueError(f"input manifest files[{index}] has an invalid schema")
        path = _safe_manifest_path(runs_root, entry["path"])
        expected_basename = _ROLE_BASENAMES[role]
        if expected_basename is not None and path.name != expected_basename:
            raise ValueError(f"input manifest role and basename disagree for {entry['path']!r}")
        if path in seen:
            raise ValueError(f"duplicate input manifest path: {entry['path']!r}")
        seen.add(path)
        expected_sha = entry["sha256"]
        if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
            raise ValueError(f"invalid SHA-256 in input manifest files[{index}]")
        if not path.is_file():
            raise ValueError(f"required AET input is missing: {entry['path']}")
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha.lower():
            raise ValueError(f"AET input checksum mismatch: {entry['path']}")
        if role in run_scoped_opaque_roles:
            opaque_run_ids[role][observed_sha] = _nonempty_string(
                entry["run_id"], f"input manifest files[{index}].run_id"
            )
        if role not in _OPAQUE_ROLES:
            file_records = _read_json_file(path)
            if len(file_records) != 1:
                raise ValueError(f"confirmatory AET input must contain one object: {entry['path']}")
            record = file_records[0]
            expected_schema = _SUPPORT_SCHEMAS.get(role)
            if expected_schema is not None and record.get("schema_version") != expected_schema:
                raise ValueError(
                    f"input manifest role {role!r} requires schema {expected_schema!r}: "
                    f"{entry['path']}"
                )
            records_by_path[path] = record
        by_role[role].append(path)

    missing_roles = sorted(role for role, paths in by_role.items() if not paths)
    if missing_roles:
        raise ValueError(f"input manifest is missing roles: {', '.join(missing_roles)}")
    discovered: set[Path] = {
        path.resolve()
        for basename in (value for value in _ROLE_BASENAMES.values() if value is not None)
        for path in runs_root.rglob(basename)
    }
    for pattern in ("*.jsonl", "*.csv", "*.log", "*.py"):
        discovered.update(path.resolve() for path in runs_root.rglob(pattern))
    for path in runs_root.rglob("*.json"):
        resolved = path.resolve()
        if resolved == manifest_path or resolved in discovered:
            continue
        support_named = (
            path.name in _KNOWN_SUPPORT_BASENAMES
            or "calibration" in path.name.lower()
            or "preflight" in path.name.lower()
        )
        try:
            candidate_records = _read_json_file(path)
        except ValueError:
            if support_named:
                discovered.add(resolved)
            continue
        if len(candidate_records) == 1 and candidate_records[0].get("schema_version") in set(
            _SUPPORT_SCHEMAS.values()
        ):
            discovered.add(resolved)
    extras = sorted(path for path in discovered - seen)
    if extras:
        rendered = ", ".join(path.relative_to(runs_root.resolve()).as_posix() for path in extras)
        raise ValueError(f"unmanifested AET input files found: {rendered}")

    paths_by_role_hash: dict[str, dict[str, Path]] = {
        role: {} for role in _ROLE_BASENAMES if role not in _ENERGY_ROLES
    }
    for role, role_paths in by_role.items():
        if role in _ENERGY_ROLES:
            continue
        for path in role_paths:
            digest = _sha256_file(path)
            if digest in paths_by_role_hash[role]:
                raise ValueError(f"duplicate manifested {role} content: {path}")
            paths_by_role_hash[role][digest] = path

    support_by_hash: dict[str, dict[str, dict[str, Any]]] = {role: {} for role in _SUPPORT_SCHEMAS}
    validators = {
        "provenance": _validate_provenance,
        "instance_manifest": _validate_instance_manifest,
        "training_dataset_manifest": _validate_training_dataset_manifest,
        "calibration": _validate_calibration,
        "preflight": _validate_preflight,
        "calibration_trace_manifest": _validate_calibration_trace_manifest,
    }
    for path in by_role["validation"]:
        digest = _sha256_file(path)
        support_by_hash["validation"][digest] = _validate_validation(
            records_by_path[path],
            path.relative_to(runs_root).as_posix(),
            purpose=purpose,
        )
    for role, validator in validators.items():
        for path in by_role[role]:
            digest = _sha256_file(path)
            support_by_hash[role][digest] = validator(
                records_by_path[path],
                path.relative_to(runs_root).as_posix(),
            )

    used_hashes: dict[str, set[str]] = {role: set() for role in _SUPPORT_SCHEMAS}
    used_opaque_hashes: dict[str, set[str]] = {role: set() for role in _OPAQUE_ROLES}
    for calibration_sha, calibration in support_by_hash["calibration"].items():
        calibration_path = paths_by_role_hash["calibration"][calibration_sha]
        trace_manifest_sha = calibration["raw_trace_manifest_sha256"]
        trace_manifest_path = paths_by_role_hash["calibration_trace_manifest"].get(
            trace_manifest_sha
        )
        trace_manifest = support_by_hash["calibration_trace_manifest"].get(trace_manifest_sha)
        if trace_manifest_path is None or trace_manifest is None:
            raise ValueError("calibration does not link a manifested raw trace manifest")
        expected_trace_manifest_path = _safe_manifest_path(
            calibration_path.parent, calibration["raw_trace_manifest_path"]
        )
        if trace_manifest_path != expected_trace_manifest_path:
            raise ValueError("calibration raw trace manifest path does not match its evidence")
        if trace_manifest["hashes_by_block"] != calibration["workload_trace_hashes"]:
            raise ValueError("calibration trace manifest does not match workload block hashes")
        used_hashes["calibration_trace_manifest"].add(trace_manifest_sha)

        analysis_sha = calibration["analysis_script_sha256"]
        analysis_path = paths_by_role_hash["calibration_analysis"].get(analysis_sha)
        expected_analysis_path = _safe_manifest_path(
            calibration_path.parent, calibration["analysis_script_path"]
        )
        if analysis_path is None or analysis_path != expected_analysis_path:
            raise ValueError("calibration does not link a manifested analysis script")
        used_opaque_hashes["calibration_analysis"].add(analysis_sha)

        trace_files = {
            _safe_manifest_path(trace_manifest_path.parent, relative_path)
            .relative_to(runs_root)
            .as_posix(): digest
            for relative_path, digest in trace_manifest["files"].items()
        }
        manifested_trace_files = {
            path.relative_to(runs_root).as_posix(): digest
            for digest, path in paths_by_role_hash["calibration_trace"].items()
        }
        if trace_files != manifested_trace_files:
            raise ValueError(
                "calibration trace manifest does not exactly cover manifested raw traces"
            )
        used_opaque_hashes["calibration_trace"].update(manifested_trace_files.values())

    bundle_identity: set[tuple[Any, ...]] = set()
    comparison_manifests: dict[tuple[str, int], dict[str, set[str]]] = {}
    comparison_boundaries: dict[
        tuple[str, int], dict[str, set[tuple[str, bool, tuple[str, ...], tuple[str, ...]]]]
    ] = {}
    for energy_role in ("training", "inference", "baseline"):
        for path in by_role[energy_role]:
            relative = path.relative_to(runs_root).as_posix()
            energy = _validate_energy_record(
                records_by_path[path],
                energy_role,
                relative,
                purpose=purpose,
            )
            wall_trace_sha = energy["wall_meter_trace_sha256"]
            wall_trace_path = paths_by_role_hash["wall_trace"].get(wall_trace_sha)
            if (
                wall_trace_path is None
                or opaque_run_ids["wall_trace"].get(wall_trace_sha) != energy["run_id"]
            ):
                raise ValueError(f"{relative} does not link exactly one wall trace for its run_id")
            used_opaque_hashes["wall_trace"].add(wall_trace_sha)
            nvml_trace_sha = energy["nvml_trace_sha256"]
            if energy_role in {"training", "inference"}:
                nvml_trace_path = paths_by_role_hash["nvml_trace"].get(nvml_trace_sha)
                if (
                    nvml_trace_path is None
                    or opaque_run_ids["nvml_trace"].get(nvml_trace_sha) != energy["run_id"]
                ):
                    raise ValueError(
                        f"{relative} does not link exactly one NVML trace for its run_id"
                    )
                used_opaque_hashes["nvml_trace"].add(nvml_trace_sha)
            elif nvml_trace_sha is not None:
                raise ValueError(f"{relative} baseline must not link an NVML trace")
            support_links = {
                "provenance": energy["provenance_sha256"],
                "validation": energy["validation_sha256"],
                "calibration": energy["calibration_sha256"],
                "preflight": energy["preflight_sha256"],
            }
            if energy_role in {"inference", "baseline"}:
                support_links["instance_manifest"] = energy["instance_manifest_sha256"]
            else:
                support_links["training_dataset_manifest"] = energy[
                    "training_dataset_manifest_sha256"
                ]
            linked: dict[str, dict[str, Any]] = {}
            for support_role, digest in support_links.items():
                support_record = support_by_hash[support_role].get(digest)
                if support_record is None:
                    raise ValueError(
                        f"{energy_role} record does not link to a manifested "
                        f"{support_role} file: {relative}"
                    )
                linked[support_role] = support_record
                used_hashes[support_role].add(digest)

            provenance = linked["provenance"]
            validation = linked["validation"]
            calibration = linked["calibration"]
            preflight = linked["preflight"]
            common_fields = (
                "host_id",
                "execution_layer",
                "git_sha",
                "gpu_index",
                "gpu_device_id_sha256",
                "preflight_sha256",
                "calibration_sha256",
            )
            for field in common_fields:
                expected = energy[field]
                for support_name, support_record in (
                    ("provenance", provenance),
                    ("validation", validation),
                ):
                    if support_record[field] != expected:
                        raise ValueError(f"{relative} {field} does not match linked {support_name}")
            for field in (
                "host_id",
                "execution_layer",
                "git_sha",
                "gpu_index",
                "gpu_device_id_sha256",
                "calibration_sha256",
            ):
                if preflight[field] != energy[field]:
                    raise ValueError(f"{relative} {field} does not match linked preflight")
            for field in (
                "host_id",
                "execution_layer",
                "gpu_index",
                "gpu_device_id_sha256",
            ):
                if calibration[field] != energy[field]:
                    raise ValueError(f"{relative} {field} does not match linked calibration")
            if provenance["run_id"] != energy["run_id"] or validation["run_id"] != energy["run_id"]:
                raise ValueError(f"{relative} run_id does not match linked run evidence")
            for evidence_sha in validation["evidence_sha256s"]:
                evidence_path = paths_by_role_hash["validation_evidence"].get(evidence_sha)
                if (
                    evidence_path is None
                    or opaque_run_ids["validation_evidence"].get(evidence_sha) != energy["run_id"]
                ):
                    raise ValueError(
                        f"{relative} validation check does not link exactly one evidence "
                        "file for its run_id"
                    )
                used_opaque_hashes["validation_evidence"].add(evidence_sha)
            expected_diagnostic = None if energy_role == "baseline" else "nvml"
            if provenance["diagnostic_backend"] != expected_diagnostic:
                raise ValueError(
                    f"{relative} provenance diagnostic_backend is invalid for {energy_role}"
                )
            boundary_mode = provenance["service_boundary"]["mode"]
            if energy_role == "training" and boundary_mode != "training":
                raise ValueError(f"{relative} training must use a training service boundary")
            if energy_role != "training" and boundary_mode not in {"warm", "cold"}:
                raise ValueError(f"{relative} deployment service boundary is invalid")
            if validation["role"] != energy_role:
                raise ValueError(f"{relative} role does not match linked validation")
            if validation["provenance_sha256"] != energy["provenance_sha256"]:
                raise ValueError(f"{relative} validation does not link the same provenance")
            if (
                calibration["calibration_id"] != energy["wall_meter_calibration_id"]
                or calibration["wall_meter_device_id_sha256"]
                != energy["wall_meter_device_id_sha256"]
            ):
                raise ValueError(f"{relative} wall-meter identity does not match calibration")
            if energy["wall_meter_sample_interval_s"] > calibration["sample_interval_s"]:
                raise ValueError(f"{relative} wall-meter sampling is slower than calibration")
            if energy["duration_s"] < calibration["minimum_block_duration_s"]:
                raise ValueError(f"{relative} measurement block is shorter than calibration gate")
            sample_total = (
                energy["wall_meter_sample_count"] + energy["wall_meter_dropped_sample_count"]
            )
            missing_percent = 100.0 * energy["wall_meter_dropped_sample_count"] / sample_total
            if missing_percent > calibration["max_missing_sample_percent"]:
                raise ValueError(f"{relative} has too many dropped wall-meter samples")
            if energy_role in {"inference", "baseline"}:
                instance_manifest = linked["instance_manifest"]
                if validation["instance_manifest_sha256"] != energy["instance_manifest_sha256"]:
                    raise ValueError(f"{relative} validation links a different instance manifest")
                if (
                    instance_manifest["problem"] != energy["problem"]
                    or instance_manifest["size"] != energy["size"]
                ):
                    raise ValueError(f"{relative} does not match its instance manifest")
                if instance_manifest["item_count"] != validation["expected_records"]:
                    raise ValueError(
                        f"{relative} validation count does not match its instance manifest"
                    )
                if validation["gap_to_reference_pct"] != energy["gap_to_reference_pct"]:
                    raise ValueError(f"{relative} quality gap does not match linked validation")
                if (
                    validation["quality_feasible_confirmatory"]
                    != energy["quality_feasible_confirmatory"]
                ):
                    raise ValueError(
                        f"{relative} quality decision does not match linked validation"
                    )
                cell = (energy["problem"], energy["size"])
                role_manifests = comparison_manifests.setdefault(
                    cell, {"inference": set(), "baseline": set()}
                )
                role_manifests[energy_role].add(energy["instance_manifest_sha256"])
                boundary = provenance["service_boundary"]
                boundary_signature = (
                    boundary["mode"],
                    boundary["initialization_reuse"],
                    tuple(sorted(item.strip() for item in boundary["includes"])),
                    tuple(sorted(item.strip() for item in boundary["excludes"])),
                )
                role_boundaries = comparison_boundaries.setdefault(
                    cell, {"inference": set(), "baseline": set()}
                )
                role_boundaries[energy_role].add(boundary_signature)
            else:
                training_dataset = linked["training_dataset_manifest"]
                if (
                    provenance["training_dataset_manifest_sha256"]
                    != energy["training_dataset_manifest_sha256"]
                    or validation["training_dataset_manifest_sha256"]
                    != energy["training_dataset_manifest_sha256"]
                ):
                    raise ValueError(
                        f"{relative} training evidence links a different dataset manifest"
                    )
                if (
                    training_dataset["problem"] != energy["problem"]
                    or training_dataset["size"] != energy["size"]
                    or training_dataset["item_count"] != validation["expected_records"]
                ):
                    raise ValueError(f"{relative} does not match its training dataset manifest")
            bundle_identity.add(
                (
                    energy["host_id"],
                    energy["execution_layer"],
                    energy["git_sha"],
                    energy["gpu_index"],
                    energy["gpu_device_id_sha256"],
                    energy["preflight_sha256"],
                    energy["calibration_sha256"],
                )
            )

    if len(bundle_identity) != 1:
        raise ValueError(
            "confirmatory bundle mixes Git, host, GPU, preflight, or calibration identities"
        )
    for cell, role_manifests in comparison_manifests.items():
        inference_hashes = role_manifests["inference"]
        baseline_hashes = role_manifests["baseline"]
        if len(inference_hashes) != 1 or inference_hashes != baseline_hashes:
            raise ValueError(
                "inference and baseline must share one instance manifest for "
                f"{cell[0]} size {cell[1]}"
            )
        role_boundaries = comparison_boundaries[cell]
        inference_boundaries = role_boundaries["inference"]
        baseline_boundaries = role_boundaries["baseline"]
        if len(inference_boundaries) != 1 or inference_boundaries != baseline_boundaries:
            raise ValueError(
                "inference and baseline must share one service boundary for "
                f"{cell[0]} size {cell[1]}"
            )
    for support_role, records in support_by_hash.items():
        unused = sorted(set(records) - used_hashes[support_role])
        if unused:
            raise ValueError(f"manifested {support_role} files are not linked by the bundle")
    for support_role, records in paths_by_role_hash.items():
        if support_role not in _OPAQUE_ROLES:
            continue
        unused = sorted(set(records) - used_opaque_hashes[support_role])
        if unused:
            raise ValueError(f"manifested {support_role} files are not linked by the bundle")
    return LoadedInputManifest(
        by_role,
        purpose=purpose,
        confirmatory_eligible=confirmatory_eligible,
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _summary_markdown(
    train_records: list[dict[str, Any]],
    inference_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    confirmatory_bundle_valid: bool = False,
) -> str:
    finite_rows = [r for r in rows if r.get("aet_E_status") == "finite"]
    aet_statuses = Counter(str(r.get("aet_E_status", "missing")) for r in rows)
    quality_statuses = Counter(str(r.get("quality_status", "missing")) for r in rows)
    parts = [
        "# AET report\n",
        "## Inputs\n",
        f"- training records: {len(train_records)} (across "
        f"{len({r.get('seed') for r in train_records})} seeds)\n",
        f"- inference records: {len(inference_records)}\n",
        f"- baseline records: {len(baseline_records)}\n",
        f"- AET rows: {len(rows)}\n",
        f"- confirmatory input manifest valid: {confirmatory_bundle_valid}\n",
        f"- quality statuses: {dict(sorted(quality_statuses.items()))}\n",
        f"- energy AET statuses: {dict(sorted(aet_statuses.items()))}\n\n",
    ]
    if not confirmatory_bundle_valid:
        parts.append(
            "> **Exploratory input set.** No checksummed input manifest was both "
            "confirmatory-purpose and explicitly eligible, so no row may be promoted "
            "as confirmatory.\n\n"
        )
    if not finite_rows and rows:
        parts.append(
            "> **No finite AET rows.** The report does not promote point-only or "
            "legacy feasibility to a finite crossover. Inspect `quality_status`, "
            "`aet_E_status`, and their reason fields in `aet_table.csv`.\n\n"
        )
    if finite_rows:
        parts.append("## Finite AET rows (head)\n\n")
        parts.append(
            "| variant | size | batch | delta% | quality | energy status | aet_E |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for r in finite_rows[:20]:
            parts.append(
                f"| {r.get('variant')} | {r.get('size')} | {r.get('batch_size')} | "
                f"{r.get('delta_pct')} | {r.get('quality_status')} | "
                f"{r.get('aet_E_status')} | {r.get('aet_E'):.1f} |\n"
            )
    parts.append("\n![AET vs δ](figures/aet_vs_delta.png)\n")
    parts.append("\n![Asymptotic regime](figures/asymptotic.png)\n")
    parts.append("\n![Per-seed energy](figures/per_seed_energy.png)\n")
    return "".join(parts)


def write_aet_report(
    runs_root: str | Path,
    *,
    deltas: Iterable[float] = (0.5, 1.0, 2.0, 5.0),
    out: str | Path = "aet_report",
    use_embodied_for_co2: bool = True,
    expected_manifest: str | Path | None = None,
) -> Path:
    """Build a report, promoting confirmatory rows only with a complete manifest."""
    runs_root = Path(runs_root).resolve()
    if not runs_root.is_dir():
        raise ValueError(f"runs_root is not a directory: {runs_root}")
    manifest_paths: LoadedInputManifest | None = None
    if expected_manifest is not None:
        manifest_paths = _load_expected_manifest(
            runs_root,
            Path(expected_manifest).resolve(),
        )

    if manifest_paths is None:
        train_records = _read_jsons(runs_root, "energy_train.json")
        inference_records = _read_jsons(runs_root, "energy_eval.json")
        baseline_records = _read_jsons(runs_root, "energy_baseline.json")
    else:
        train_records = [
            record for path in manifest_paths["training"] for record in _read_json_file(path)
        ]
        inference_records = [
            record for path in manifest_paths["inference"] for record in _read_json_file(path)
        ]
        baseline_records = [
            record for path in manifest_paths["baseline"] for record in _read_json_file(path)
        ]

    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = runs_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    confirmatory_bundle_valid = bool(
        manifest_paths is not None
        and manifest_paths.purpose == "confirmatory"
        and manifest_paths.confirmatory_eligible
    )
    train_agg = aggregate_training(train_records)
    rows = build_aet_table(
        train_agg,
        inference_records,
        baseline_records,
        deltas=list(deltas),
        use_embodied_for_co2=use_embodied_for_co2,
        confirmatory_bundle_valid=confirmatory_bundle_valid,
    )

    _write_csv(rows, out_dir / "aet_table.csv")
    plot_aet_vs_delta(rows, figures_dir / "aet_vs_delta.png")
    plot_asymptotic(train_agg, rows, figures_dir / "asymptotic.png")
    plot_per_seed_energy(train_records, figures_dir / "per_seed_energy.png")
    md = _summary_markdown(
        train_records,
        inference_records,
        baseline_records,
        rows,
        confirmatory_bundle_valid=confirmatory_bundle_valid,
    )
    (out_dir / "aet_summary.md").write_text(md)
    return out_dir / "aet_summary.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a tree of run dirs and build the AET table + plots. "
            "Picks up energy_train.json, energy_eval.json, energy_baseline.json."
        )
    )
    parser.add_argument("runs_root", type=Path, help="Root directory (e.g. ./outputs)")
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0],
        help="Quality-gap tolerances delta (in %%).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("aet_report"),
        help="Output directory (relative to runs_root unless absolute).",
    )
    parser.add_argument(
        "--operational-only",
        action="store_true",
        help="Use operational CO2 only (drop embodied amortization).",
    )
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        default=None,
        help=(
            "Checksummed input manifest required to promote any confirmatory finite row. "
            "Without it the report is exploratory."
        ),
    )
    args = parser.parse_args()
    out = write_aet_report(
        args.runs_root,
        deltas=args.deltas,
        out=args.out,
        use_embodied_for_co2=not args.operational_only,
        expected_manifest=args.expected_manifest,
    )
    print(out)


if __name__ == "__main__":
    main()
