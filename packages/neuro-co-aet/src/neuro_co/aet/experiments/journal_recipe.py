"""Strict, non-executing validation for AET journal smoke recipes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "aet-journal-recipe/v1"
RECIPE_KIND = "aet-journal-smoke"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "name",
    "platform",
    "output_root",
    "qualification",
    "measurement",
    "runs",
}
_PLATFORM_KEYS = {"host_id", "execution_layer", "gpu_index"}
_QUALIFICATION_KEYS = {"preflight_report", "calibration_manifest"}
_MEASUREMENT_KEYS = {
    "pue",
    "fallback",
    "primary_backend",
    "diagnostic_backends",
    "minimum_block_duration_s",
    "scope",
}
_RUN_KEYS = {
    "id",
    "stage",
    "problem",
    "size",
    "policy",
    "seed",
    "repetitions",
    "items_per_block",
    "max_walltime_s",
    "output_subdir",
}
_STAGES = {"training", "inference", "baseline"}
_EXECUTION_LAYERS = {"windows-native", "wsl2"}
_DIAGNOSTIC_BACKENDS = {"nvml", "rapl"}
_SCOPE_DOMAINS = {"whole_system_ac", "gpu", "cpu_package", "dram"}
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

CALIBRATION_SCHEMA_VERSION = "aet-calibration/v1"
_CALIBRATION_TOP_LEVEL_KEYS = {
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
}
_CALIBRATION_HARDWARE_KEYS = {
    "cpu",
    "logical_cpu_count",
    "gpu",
    "gpu_index",
    "gpu_device_id_sha256",
    "ram_bytes",
    "windows_build",
    "wsl_version",
    "wsl_kernel",
    "nvidia_driver",
    "cuda_version",
}
_CALIBRATION_WALL_METER_KEYS = {
    "required",
    "model",
    "device_id_hash",
    "firmware",
    "accuracy_percent",
    "energy_resolution_j",
    "sample_interval_s",
    "data_interface",
    "calibration_reference",
}
_CALIBRATION_COUNTER_KEYS = {
    "system_primary",
    "gpu_diagnostic",
    "cpu_diagnostic",
    "strict_backend",
    "pue",
    "report_embodied",
}
_CALIBRATION_TIMING_KEYS = {
    "timestamp_alignment_error_s",
    "missing_sample_percent",
}
_CALIBRATION_CRITERIA_KEYS = {
    "max_meter_accuracy_percent",
    "max_sample_interval_s",
    "max_timestamp_alignment_error_s",
    "max_missing_sample_percent",
    "max_repeat_cv_percent",
    "max_coverage_ratio_cv_percent",
    "minimum_block_duration_s",
    "minimum_samples_per_block",
    "repeats_per_workload_class",
}
_CALIBRATION_WORKLOAD_CLASSES = {"idle", "cpu_only", "gpu_only", "combined"}
_CALIBRATION_BLOCK_KEYS = {"duration_s", "sample_count", "raw_sha256"}
_CALIBRATION_SUMMARY_KEYS = {
    "meter_repeat_cv_percent",
    "coverage_ratio_cv_percent",
    "nvml_positive_and_monotonic",
    "rapl_available",
    "prohibited_fallback_observed",
}
_CALIBRATION_EVIDENCE_KEYS = {
    "raw_trace_manifest",
    "raw_trace_manifest_sha256",
    "analysis_script",
    "analysis_script_sha256",
}
_CALIBRATION_APPROVAL_KEYS = {"passed_at", "approved_by", "notes"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

RAW_TRACE_MANIFEST_SCHEMA_VERSION = "aet-raw-trace-manifest/v1"
_RAW_TRACE_MANIFEST_KEYS = {"schema_version", "traces"}
_RAW_TRACE_ENTRY_KEYS = {
    "workload_class",
    "block_index",
    "raw_trace",
    "raw_sha256",
}

_PROTOCOL_MAX_METER_ACCURACY_PERCENT = 2.0
_PROTOCOL_MAX_SAMPLE_INTERVAL_S = 1.0
_PROTOCOL_MAX_TIMESTAMP_ALIGNMENT_ERROR_S = 0.5
_PROTOCOL_MAX_MISSING_SAMPLE_PERCENT = 0.5
_PROTOCOL_MAX_REPEAT_CV_PERCENT = 5.0
_PROTOCOL_MAX_COVERAGE_RATIO_CV_PERCENT = 5.0
_PROTOCOL_MINIMUM_BLOCK_DURATION_S = 120.0
_PROTOCOL_MINIMUM_SAMPLES_PER_BLOCK = 100
_PROTOCOL_REPEATS_PER_WORKLOAD_CLASS = 5


class RecipeValidationError(ValueError):
    """Raised when a journal recipe does not satisfy the closed schema."""


class CalibrationValidationError(ValueError):
    """Raised when a calibration manifest does not satisfy its closed schema."""


@dataclass(frozen=True)
class JournalRun:
    run_id: str
    stage: str
    problem: str
    size: int
    policy: str
    seed: int
    repetitions: int
    items_per_block: int
    max_walltime_s: float
    output_subdir: str


@dataclass(frozen=True)
class AETJournalRecipe:
    name: str
    host_id: str
    execution_layer: str
    gpu_index: int
    output_root: str
    qualification: dict[str, str]
    measurement: dict[str, Any]
    runs: tuple[JournalRun, ...]


@dataclass(frozen=True)
class ValidatedCalibration:
    """Identity fields from a fully validated, passed calibration manifest."""

    calibration_id: str
    host_id: str
    execution_layer: str
    gpu_index: int
    gpu_device_id_sha256: str
    created_at: datetime
    passed_at: datetime
    evidence_files: tuple[Path, ...]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeValidationError(f"{where} must be a mapping")
    return value


def _closed_keys(
    value: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    where: str,
) -> None:
    unknown = sorted(repr(key) for key in set(value) - allowed)
    if unknown:
        raise RecipeValidationError(f"{where} has unknown keys: {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise RecipeValidationError(f"{where} is missing required keys: {', '.join(missing)}")


def _reject_legacy_overrides(value: Any, where: str = "recipe") -> None:
    if isinstance(value, dict):
        if "overrides" in value:
            raise RecipeValidationError(
                f"{where}.overrides is a legacy free-form field and is forbidden"
            )
        for key, child in value.items():
            _reject_legacy_overrides(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_legacy_overrides(child, f"{where}[{index}]")


def _slug(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise RecipeValidationError(
            f"{where} must contain only lowercase letters, digits, '.', '_' and '-'"
        )
    if value.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise RecipeValidationError(f"{where} uses a Windows-reserved name")
    return value


def _safe_relative_path(value: Any, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RecipeValidationError(f"{where} must be a non-empty relative path")
    if "\\" in value:
        raise RecipeValidationError(f"{where} must use portable '/' separators")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise RecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise RecipeValidationError(f"{where} may not contain '.' or '..' components")
    for part in posix_path.parts:
        _slug(part, f"{where} component")
    return posix_path


def _json_artifact_path(value: Any, where: str) -> str:
    path = _safe_relative_path(value, where)
    if path.suffix != ".json":
        raise RecipeValidationError(f"{where} must identify a .json file")
    return path.as_posix()


def _integer(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RecipeValidationError(f"{where} must be a positive number")
    return float(value)


def _calibration_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationValidationError(f"{where} must be a mapping")
    return value


def _calibration_closed_keys(
    value: dict[str, Any],
    *,
    keys: set[str],
    where: str,
) -> None:
    unknown = sorted(repr(key) for key in set(value) - keys)
    if unknown:
        raise CalibrationValidationError(f"{where} has unknown keys: {', '.join(unknown)}")
    missing = sorted(keys - set(value))
    if missing:
        raise CalibrationValidationError(f"{where} is missing required keys: {', '.join(missing)}")


def _calibration_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationValidationError(f"{where} must be a non-empty string")
    return value


def _calibration_optional_string(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _calibration_string(value, where)


def _calibration_slug(value: Any, where: str) -> str:
    value = _calibration_string(value, where)
    if not _SAFE_COMPONENT.fullmatch(value):
        raise CalibrationValidationError(
            f"{where} must contain only lowercase letters, digits, '.', '_' and '-'"
        )
    if value.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise CalibrationValidationError(f"{where} uses a Windows-reserved name")
    return value


def _calibration_number(
    value: Any,
    where: str,
    *,
    positive: bool = True,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise CalibrationValidationError(f"{where} must be a finite {qualifier} number")
    return float(value)


def _calibration_integer(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _calibration_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise CalibrationValidationError(f"{where} must be a boolean")
    return value


def _calibration_sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CalibrationValidationError(
            f"{where} must be a lowercase SHA-256 digest with 64 hexadecimal characters"
        )
    return value


def _calibration_datetime(value: Any, where: str) -> datetime:
    value = _calibration_string(value, where)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CalibrationValidationError(f"{where} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CalibrationValidationError(f"{where} must include a timezone offset")
    return parsed


def _calibration_relative_file(value: Any, where: str) -> PurePosixPath:
    value = _calibration_string(value, where)
    if "\\" in value:
        raise CalibrationValidationError(f"{where} must use portable '/' separators")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise CalibrationValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise CalibrationValidationError(f"{where} may not contain '.' or '..' components")
    for part in posix_path.parts:
        if not _SAFE_COMPONENT.fullmatch(part):
            raise CalibrationValidationError(
                f"{where} components may contain only lowercase letters, digits, '.', '_' and '-'"
            )
        if part.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
            raise CalibrationValidationError(f"{where} uses a Windows-reserved name")
    return posix_path


def _calibration_sha256_file(path: Path, where: str) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise CalibrationValidationError(f"{where} cannot be read: {exc}") from exc


def _calibration_evidence_file(
    root: Path,
    relative: PurePosixPath,
    where: str,
) -> Path:
    resolved_root = root.resolve()
    resolved_path = root.joinpath(*relative.parts).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise CalibrationValidationError(f"{where} must remain inside {resolved_root}") from exc
    return resolved_path


def _calibration_json_file(path: Path, where: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationValidationError(f"{where} cannot be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationValidationError(f"{where} must contain valid JSON: {exc}") from exc
    return _calibration_mapping(payload, where)


def _validate_calibration_evidence(
    evidence: dict[str, Any],
    *,
    calibration_path: Path | None,
    expected_hashes: dict[tuple[str, int], str],
) -> tuple[Path, ...]:
    if calibration_path is None:
        raise CalibrationValidationError(
            "calibration manifest path is required to verify evidence files"
        )
    calibration_path = calibration_path.resolve()
    if not calibration_path.is_file():
        raise CalibrationValidationError(f"calibration manifest does not exist: {calibration_path}")
    evidence_root = calibration_path.parent

    raw_manifest_relative = _calibration_relative_file(
        evidence["raw_trace_manifest"],
        "calibration.evidence.raw_trace_manifest",
    )
    if raw_manifest_relative.suffix != ".json":
        raise CalibrationValidationError(
            "calibration.evidence.raw_trace_manifest must identify a .json file"
        )
    raw_manifest_path = _calibration_evidence_file(
        evidence_root,
        raw_manifest_relative,
        "calibration.evidence.raw_trace_manifest",
    )
    declared_manifest_hash = _calibration_sha256(
        evidence["raw_trace_manifest_sha256"],
        "calibration.evidence.raw_trace_manifest_sha256",
    )
    observed_manifest_hash = _calibration_sha256_file(
        raw_manifest_path,
        "calibration raw trace manifest",
    )
    if observed_manifest_hash != declared_manifest_hash:
        raise CalibrationValidationError(
            "calibration.evidence.raw_trace_manifest_sha256 does not match the manifest file"
        )

    raw_manifest = _calibration_json_file(raw_manifest_path, "calibration raw trace manifest")
    _calibration_closed_keys(
        raw_manifest,
        keys=_RAW_TRACE_MANIFEST_KEYS,
        where="calibration raw trace manifest",
    )
    if raw_manifest["schema_version"] != RAW_TRACE_MANIFEST_SCHEMA_VERSION:
        raise CalibrationValidationError(
            "calibration raw trace manifest.schema_version must be exactly "
            f"{RAW_TRACE_MANIFEST_SCHEMA_VERSION!r}"
        )
    traces = raw_manifest["traces"]
    if not isinstance(traces, list) or len(traces) != len(expected_hashes):
        raise CalibrationValidationError(
            f"calibration raw trace manifest.traces must contain exactly {len(expected_hashes)} entries"
        )

    observed_hashes: dict[tuple[str, int], str] = {}
    observed_paths: set[PurePosixPath] = set()
    unique_hashes: set[str] = set()
    raw_trace_paths: list[Path] = []
    for index, raw_entry in enumerate(traces):
        where = f"calibration raw trace manifest.traces[{index}]"
        entry = _calibration_mapping(raw_entry, where)
        _calibration_closed_keys(entry, keys=_RAW_TRACE_ENTRY_KEYS, where=where)
        workload_class = entry["workload_class"]
        if workload_class not in _CALIBRATION_WORKLOAD_CLASSES:
            raise CalibrationValidationError(
                f"{where}.workload_class must be one of: "
                + ", ".join(sorted(_CALIBRATION_WORKLOAD_CLASSES))
            )
        block_index = _calibration_integer(entry["block_index"], f"{where}.block_index", minimum=0)
        coordinate = (workload_class, block_index)
        if coordinate in observed_hashes:
            raise CalibrationValidationError(
                f"calibration raw trace manifest has duplicate block {workload_class}[{block_index}]"
            )
        raw_trace_relative = _calibration_relative_file(entry["raw_trace"], f"{where}.raw_trace")
        if raw_trace_relative in observed_paths:
            raise CalibrationValidationError(
                f"calibration raw trace manifest has duplicate raw_trace path {raw_trace_relative}"
            )
        raw_sha256 = _calibration_sha256(entry["raw_sha256"], f"{where}.raw_sha256")
        if raw_sha256 in unique_hashes:
            raise CalibrationValidationError(
                f"calibration raw trace manifest has duplicate raw_sha256 {raw_sha256}"
            )
        raw_trace_path = _calibration_evidence_file(
            raw_manifest_path.parent,
            raw_trace_relative,
            f"{where}.raw_trace",
        )
        observed_raw_hash = _calibration_sha256_file(raw_trace_path, f"{where}.raw_trace")
        if observed_raw_hash != raw_sha256:
            raise CalibrationValidationError(f"{where}.raw_sha256 does not match its trace file")
        observed_hashes[coordinate] = raw_sha256
        observed_paths.add(raw_trace_relative)
        unique_hashes.add(raw_sha256)
        raw_trace_paths.append(raw_trace_path)

    if observed_hashes != expected_hashes:
        missing = sorted(set(expected_hashes) - set(observed_hashes))
        extra = sorted(set(observed_hashes) - set(expected_hashes))
        mismatched = sorted(
            coordinate
            for coordinate in set(expected_hashes) & set(observed_hashes)
            if expected_hashes[coordinate] != observed_hashes[coordinate]
        )
        raise CalibrationValidationError(
            "calibration raw trace manifest does not exactly cover workload block hashes: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )

    analysis_script_relative = _calibration_relative_file(
        evidence["analysis_script"],
        "calibration.evidence.analysis_script",
    )
    if analysis_script_relative.suffix != ".py":
        raise CalibrationValidationError(
            "calibration.evidence.analysis_script must identify a .py file"
        )
    analysis_script_path = _calibration_evidence_file(
        evidence_root,
        analysis_script_relative,
        "calibration.evidence.analysis_script",
    )
    declared_script_hash = _calibration_sha256(
        evidence["analysis_script_sha256"],
        "calibration.evidence.analysis_script_sha256",
    )
    observed_script_hash = _calibration_sha256_file(
        analysis_script_path,
        "calibration analysis script",
    )
    if observed_script_hash != declared_script_hash:
        raise CalibrationValidationError(
            "calibration.evidence.analysis_script_sha256 does not match the analysis script"
        )
    return (raw_manifest_path, *raw_trace_paths, analysis_script_path)


def validate_calibration_manifest(
    payload: Any,
    *,
    manifest_path: Path | None = None,
) -> ValidatedCalibration:
    """Validate a passed AET calibration manifest against the closed v1 schema."""
    raw = _calibration_mapping(payload, "calibration")
    _calibration_closed_keys(
        raw,
        keys=_CALIBRATION_TOP_LEVEL_KEYS,
        where="calibration",
    )
    if raw["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationValidationError(
            f"calibration.schema_version must be exactly {CALIBRATION_SCHEMA_VERSION!r}"
        )
    if raw["status"] != "passed":
        raise CalibrationValidationError("calibration.status must be exactly 'passed'")
    calibration_id = _calibration_slug(raw["calibration_id"], "calibration.calibration_id")
    host_id = _calibration_slug(raw["host_id"], "calibration.host_id")
    execution_layer = raw["execution_layer"]
    if not isinstance(execution_layer, str) or execution_layer not in _EXECUTION_LAYERS:
        raise CalibrationValidationError(
            "calibration.execution_layer must be 'windows-native' or 'wsl2'"
        )
    created_at = _calibration_datetime(raw["created_at"], "calibration.created_at")

    hardware = _calibration_mapping(raw["hardware"], "calibration.hardware")
    _calibration_closed_keys(
        hardware,
        keys=_CALIBRATION_HARDWARE_KEYS,
        where="calibration.hardware",
    )
    _calibration_string(hardware["cpu"], "calibration.hardware.cpu")
    _calibration_integer(
        hardware["logical_cpu_count"],
        "calibration.hardware.logical_cpu_count",
        minimum=1,
    )
    _calibration_string(hardware["gpu"], "calibration.hardware.gpu")
    gpu_index = _calibration_integer(
        hardware["gpu_index"], "calibration.hardware.gpu_index", minimum=0
    )
    gpu_device_id_sha256 = _calibration_sha256(
        hardware["gpu_device_id_sha256"],
        "calibration.hardware.gpu_device_id_sha256",
    )
    _calibration_integer(hardware["ram_bytes"], "calibration.hardware.ram_bytes", minimum=1)
    _calibration_string(hardware["windows_build"], "calibration.hardware.windows_build")
    wsl_version = _calibration_optional_string(
        hardware["wsl_version"], "calibration.hardware.wsl_version"
    )
    wsl_kernel = _calibration_optional_string(
        hardware["wsl_kernel"], "calibration.hardware.wsl_kernel"
    )
    if execution_layer == "wsl2" and (wsl_version is None or wsl_kernel is None):
        raise CalibrationValidationError(
            "calibration.hardware.wsl_version and wsl_kernel are required for wsl2"
        )
    if execution_layer == "windows-native" and (wsl_version is not None or wsl_kernel is not None):
        raise CalibrationValidationError(
            "calibration.hardware.wsl_version and wsl_kernel must be null for windows-native"
        )
    _calibration_string(hardware["nvidia_driver"], "calibration.hardware.nvidia_driver")
    _calibration_string(hardware["cuda_version"], "calibration.hardware.cuda_version")

    wall_meter = _calibration_mapping(raw["wall_meter"], "calibration.wall_meter")
    _calibration_closed_keys(
        wall_meter,
        keys=_CALIBRATION_WALL_METER_KEYS,
        where="calibration.wall_meter",
    )
    if wall_meter["required"] is not True:
        raise CalibrationValidationError("calibration.wall_meter.required must be true")
    _calibration_string(wall_meter["model"], "calibration.wall_meter.model")
    _calibration_sha256(wall_meter["device_id_hash"], "calibration.wall_meter.device_id_hash")
    _calibration_string(wall_meter["firmware"], "calibration.wall_meter.firmware")
    meter_accuracy = _calibration_number(
        wall_meter["accuracy_percent"],
        "calibration.wall_meter.accuracy_percent",
    )
    _calibration_number(
        wall_meter["energy_resolution_j"],
        "calibration.wall_meter.energy_resolution_j",
    )
    sample_interval = _calibration_number(
        wall_meter["sample_interval_s"],
        "calibration.wall_meter.sample_interval_s",
    )
    data_interface = _calibration_string(
        wall_meter["data_interface"], "calibration.wall_meter.data_interface"
    ).lower()
    if data_interface not in {"api", "csv", "json"}:
        raise CalibrationValidationError(
            "calibration.wall_meter.data_interface must be 'api', 'csv' or 'json'"
        )
    _calibration_string(
        wall_meter["calibration_reference"],
        "calibration.wall_meter.calibration_reference",
    )

    counters = _calibration_mapping(raw["counter_backends"], "calibration.counter_backends")
    _calibration_closed_keys(
        counters,
        keys=_CALIBRATION_COUNTER_KEYS,
        where="calibration.counter_backends",
    )
    if counters["system_primary"] != "wall_meter":
        raise CalibrationValidationError(
            "calibration.counter_backends.system_primary must be 'wall_meter'"
        )
    if counters["gpu_diagnostic"] != "nvml":
        raise CalibrationValidationError(
            "calibration.counter_backends.gpu_diagnostic must be 'nvml'"
        )
    if counters["cpu_diagnostic"] is not None and counters["cpu_diagnostic"] != "rapl":
        raise CalibrationValidationError(
            "calibration.counter_backends.cpu_diagnostic must be null or 'rapl'"
        )
    if counters["strict_backend"] is not True:
        raise CalibrationValidationError("calibration.counter_backends.strict_backend must be true")
    pue = _calibration_number(counters["pue"], "calibration.counter_backends.pue")
    if pue != 1.0:
        raise CalibrationValidationError("calibration.counter_backends.pue must be exactly 1.0")
    if counters["report_embodied"] is not False:
        raise CalibrationValidationError(
            "calibration.counter_backends.report_embodied must be false"
        )

    timing = _calibration_mapping(raw["timing"], "calibration.timing")
    _calibration_closed_keys(
        timing,
        keys=_CALIBRATION_TIMING_KEYS,
        where="calibration.timing",
    )
    timestamp_alignment_error = _calibration_number(
        timing["timestamp_alignment_error_s"],
        "calibration.timing.timestamp_alignment_error_s",
        positive=False,
    )
    missing_sample_percent = _calibration_number(
        timing["missing_sample_percent"],
        "calibration.timing.missing_sample_percent",
        positive=False,
    )

    criteria = _calibration_mapping(raw["criteria"], "calibration.criteria")
    _calibration_closed_keys(
        criteria,
        keys=_CALIBRATION_CRITERIA_KEYS,
        where="calibration.criteria",
    )
    maximum_criteria = {
        "max_meter_accuracy_percent": _PROTOCOL_MAX_METER_ACCURACY_PERCENT,
        "max_sample_interval_s": _PROTOCOL_MAX_SAMPLE_INTERVAL_S,
        "max_timestamp_alignment_error_s": (_PROTOCOL_MAX_TIMESTAMP_ALIGNMENT_ERROR_S),
        "max_missing_sample_percent": _PROTOCOL_MAX_MISSING_SAMPLE_PERCENT,
        "max_repeat_cv_percent": _PROTOCOL_MAX_REPEAT_CV_PERCENT,
        "max_coverage_ratio_cv_percent": _PROTOCOL_MAX_COVERAGE_RATIO_CV_PERCENT,
    }
    validated_maxima: dict[str, float] = {}
    for field, protocol_maximum in maximum_criteria.items():
        value = _calibration_number(criteria[field], f"calibration.criteria.{field}")
        if value > protocol_maximum:
            raise CalibrationValidationError(
                f"calibration.criteria.{field} may not exceed {protocol_maximum}"
            )
        validated_maxima[field] = value
    minimum_duration = _calibration_number(
        criteria["minimum_block_duration_s"],
        "calibration.criteria.minimum_block_duration_s",
    )
    if minimum_duration < _PROTOCOL_MINIMUM_BLOCK_DURATION_S:
        raise CalibrationValidationError(
            "calibration.criteria.minimum_block_duration_s must be >= 120"
        )
    minimum_samples = _calibration_integer(
        criteria["minimum_samples_per_block"],
        "calibration.criteria.minimum_samples_per_block",
        minimum=_PROTOCOL_MINIMUM_SAMPLES_PER_BLOCK,
    )
    repetitions = _calibration_integer(
        criteria["repeats_per_workload_class"],
        "calibration.criteria.repeats_per_workload_class",
        minimum=_PROTOCOL_REPEATS_PER_WORKLOAD_CLASS,
    )
    if repetitions != _PROTOCOL_REPEATS_PER_WORKLOAD_CLASS:
        raise CalibrationValidationError(
            "calibration.criteria.repeats_per_workload_class must be exactly 5"
        )
    if meter_accuracy > validated_maxima["max_meter_accuracy_percent"]:
        raise CalibrationValidationError(
            "calibration.wall_meter.accuracy_percent exceeds its declared criterion"
        )
    if sample_interval > validated_maxima["max_sample_interval_s"]:
        raise CalibrationValidationError(
            "calibration.wall_meter.sample_interval_s exceeds its declared criterion"
        )
    if timestamp_alignment_error > validated_maxima["max_timestamp_alignment_error_s"]:
        raise CalibrationValidationError(
            "calibration.timing.timestamp_alignment_error_s exceeds its declared criterion"
        )
    if missing_sample_percent > validated_maxima["max_missing_sample_percent"]:
        raise CalibrationValidationError(
            "calibration.timing.missing_sample_percent exceeds its declared criterion"
        )

    workload_classes = _calibration_mapping(raw["workload_classes"], "calibration.workload_classes")
    _calibration_closed_keys(
        workload_classes,
        keys=_CALIBRATION_WORKLOAD_CLASSES,
        where="calibration.workload_classes",
    )
    expected_hashes: dict[tuple[str, int], str] = {}
    for workload_name in sorted(_CALIBRATION_WORKLOAD_CLASSES):
        records = workload_classes[workload_name]
        where = f"calibration.workload_classes.{workload_name}"
        if not isinstance(records, list) or len(records) != repetitions:
            raise CalibrationValidationError(f"{where} must contain exactly {repetitions} blocks")
        for index, record_value in enumerate(records):
            record_where = f"{where}[{index}]"
            record = _calibration_mapping(record_value, record_where)
            _calibration_closed_keys(
                record,
                keys=_CALIBRATION_BLOCK_KEYS,
                where=record_where,
            )
            duration = _calibration_number(record["duration_s"], f"{record_where}.duration_s")
            if duration < minimum_duration:
                raise CalibrationValidationError(
                    f"{record_where}.duration_s must be >= {minimum_duration}"
                )
            _calibration_integer(
                record["sample_count"],
                f"{record_where}.sample_count",
                minimum=minimum_samples,
            )
            raw_sha256 = _calibration_sha256(record["raw_sha256"], f"{record_where}.raw_sha256")
            if raw_sha256 in expected_hashes.values():
                raise CalibrationValidationError(
                    f"{record_where}.raw_sha256 duplicates another workload block hash"
                )
            expected_hashes[(workload_name, index)] = raw_sha256

    summary = _calibration_mapping(raw["summary"], "calibration.summary")
    _calibration_closed_keys(
        summary,
        keys=_CALIBRATION_SUMMARY_KEYS,
        where="calibration.summary",
    )
    repeat_cv = _calibration_number(
        summary["meter_repeat_cv_percent"],
        "calibration.summary.meter_repeat_cv_percent",
        positive=False,
    )
    if repeat_cv > validated_maxima["max_repeat_cv_percent"]:
        raise CalibrationValidationError(
            "calibration.summary.meter_repeat_cv_percent exceeds its declared criterion"
        )
    coverage_cv = _calibration_number(
        summary["coverage_ratio_cv_percent"],
        "calibration.summary.coverage_ratio_cv_percent",
        positive=False,
    )
    if coverage_cv > validated_maxima["max_coverage_ratio_cv_percent"]:
        raise CalibrationValidationError(
            "calibration.summary.coverage_ratio_cv_percent exceeds its declared criterion"
        )
    if summary["nvml_positive_and_monotonic"] is not True:
        raise CalibrationValidationError(
            "calibration.summary.nvml_positive_and_monotonic must be true"
        )
    _calibration_bool(summary["rapl_available"], "calibration.summary.rapl_available")
    if summary["prohibited_fallback_observed"] is not False:
        raise CalibrationValidationError(
            "calibration.summary.prohibited_fallback_observed must be false"
        )

    evidence = _calibration_mapping(raw["evidence"], "calibration.evidence")
    _calibration_closed_keys(
        evidence,
        keys=_CALIBRATION_EVIDENCE_KEYS,
        where="calibration.evidence",
    )
    evidence_files = _validate_calibration_evidence(
        evidence,
        calibration_path=manifest_path,
        expected_hashes=expected_hashes,
    )

    approval = _calibration_mapping(raw["approval"], "calibration.approval")
    _calibration_closed_keys(
        approval,
        keys=_CALIBRATION_APPROVAL_KEYS,
        where="calibration.approval",
    )
    passed_at = _calibration_datetime(approval["passed_at"], "calibration.approval.passed_at")
    if passed_at < created_at:
        raise CalibrationValidationError(
            "calibration.approval.passed_at may not precede calibration.created_at"
        )
    _calibration_string(approval["approved_by"], "calibration.approval.approved_by")
    _calibration_optional_string(approval["notes"], "calibration.approval.notes")

    return ValidatedCalibration(
        calibration_id=calibration_id,
        host_id=host_id,
        execution_layer=execution_layer,
        gpu_index=gpu_index,
        gpu_device_id_sha256=gpu_device_id_sha256,
        created_at=created_at,
        passed_at=passed_at,
        evidence_files=evidence_files,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "AET journal recipe validation needs PyYAML. Install the CLI report extra."
        ) from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RecipeValidationError(f"invalid YAML in {path}: {exc}") from exc
    return _mapping(value, "recipe")


def load_aet_journal_recipe(path: Path) -> AETJournalRecipe:
    """Load an AET smoke recipe using a closed, fail-closed schema."""
    raw = _load_yaml(Path(path))
    _reject_legacy_overrides(raw)
    _closed_keys(
        raw,
        allowed=_TOP_LEVEL_KEYS,
        required=_TOP_LEVEL_KEYS,
        where="recipe",
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise RecipeValidationError(f"recipe.schema_version must be exactly {SCHEMA_VERSION!r}")
    if raw["kind"] != RECIPE_KIND:
        raise RecipeValidationError(f"recipe.kind must be exactly {RECIPE_KIND!r}")
    name = _slug(raw["name"], "recipe.name")

    platform = _mapping(raw["platform"], "recipe.platform")
    _closed_keys(
        platform,
        allowed=_PLATFORM_KEYS,
        required=_PLATFORM_KEYS,
        where="recipe.platform",
    )
    host_id = _slug(platform["host_id"], "recipe.platform.host_id")
    execution_layer = platform["execution_layer"]
    if not isinstance(execution_layer, str) or execution_layer not in _EXECUTION_LAYERS:
        allowed_layers = ", ".join(sorted(_EXECUTION_LAYERS))
        raise RecipeValidationError(
            f"recipe.platform.execution_layer must be one of: {allowed_layers}"
        )
    gpu_index = _integer(
        platform["gpu_index"],
        "recipe.platform.gpu_index",
        minimum=0,
    )

    output_path = _safe_relative_path(raw["output_root"], "recipe.output_root")
    required_prefix = ("experiments", "aet-journal", "raw")
    if output_path.parts[: len(required_prefix)] != required_prefix:
        raise RecipeValidationError(
            "recipe.output_root must stay under experiments/aet-journal/raw/"
        )
    if len(output_path.parts) == len(required_prefix):
        raise RecipeValidationError("recipe.output_root must name a run collection")

    qualification_raw = _mapping(raw["qualification"], "recipe.qualification")
    _closed_keys(
        qualification_raw,
        allowed=_QUALIFICATION_KEYS,
        required=_QUALIFICATION_KEYS,
        where="recipe.qualification",
    )
    qualification = {
        "preflight_report": _json_artifact_path(
            qualification_raw["preflight_report"],
            "recipe.qualification.preflight_report",
        ),
        "calibration_manifest": _json_artifact_path(
            qualification_raw["calibration_manifest"],
            "recipe.qualification.calibration_manifest",
        ),
    }
    if len(set(qualification.values())) != len(qualification):
        raise RecipeValidationError("recipe.qualification paths must identify two different files")

    measurement = _mapping(raw["measurement"], "recipe.measurement")
    _closed_keys(
        measurement,
        allowed=_MEASUREMENT_KEYS,
        required=_MEASUREMENT_KEYS,
        where="recipe.measurement",
    )
    fallback = measurement["fallback"]
    if not isinstance(fallback, bool) or fallback:
        raise RecipeValidationError("recipe.measurement.fallback must be false")
    pue = measurement["pue"]
    if isinstance(pue, bool) or not isinstance(pue, (int, float)) or float(pue) != 1.0:
        raise RecipeValidationError("recipe.measurement.pue must be exactly 1.0")
    if measurement["primary_backend"] != "wall_meter":
        raise RecipeValidationError(
            "recipe.measurement.primary_backend must be 'wall_meter' for this Windows smoke"
        )
    diagnostic_backends = measurement["diagnostic_backends"]
    if not isinstance(diagnostic_backends, list) or not diagnostic_backends:
        raise RecipeValidationError(
            "recipe.measurement.diagnostic_backends must be a non-empty list"
        )
    if not all(isinstance(backend, str) for backend in diagnostic_backends):
        raise RecipeValidationError(
            "recipe.measurement.diagnostic_backends must contain only strings"
        )
    if len(set(diagnostic_backends)) != len(diagnostic_backends):
        raise RecipeValidationError("recipe.measurement.diagnostic_backends has duplicates")
    invalid_backends = sorted(set(diagnostic_backends) - _DIAGNOSTIC_BACKENDS)
    if invalid_backends:
        raise RecipeValidationError(
            "recipe.measurement.diagnostic_backends contains forbidden or unknown values: "
            + ", ".join(invalid_backends)
        )
    if "nvml" not in diagnostic_backends:
        raise RecipeValidationError("recipe.measurement.diagnostic_backends must include 'nvml'")
    minimum_duration = _positive_number(
        measurement["minimum_block_duration_s"],
        "recipe.measurement.minimum_block_duration_s",
    )
    if minimum_duration < 120.0:
        raise RecipeValidationError("recipe.measurement.minimum_block_duration_s must be >= 120")

    scope = _mapping(measurement["scope"], "recipe.measurement.scope")
    _closed_keys(scope, allowed=_STAGES, required=set(), where="recipe.measurement.scope")
    if not scope:
        raise RecipeValidationError("recipe.measurement.scope must not be empty")
    for stage, domains in scope.items():
        if not isinstance(domains, list) or not domains:
            raise RecipeValidationError(
                f"recipe.measurement.scope.{stage} must be a non-empty list"
            )
        if not all(isinstance(domain, str) for domain in domains):
            raise RecipeValidationError(
                f"recipe.measurement.scope.{stage} must contain only strings"
            )
        if len(set(domains)) != len(domains):
            raise RecipeValidationError(f"recipe.measurement.scope.{stage} has duplicate domains")
        invalid_domains = sorted(set(domains) - _SCOPE_DOMAINS)
        if invalid_domains:
            raise RecipeValidationError(
                f"recipe.measurement.scope.{stage} has unknown domains: "
                + ", ".join(invalid_domains)
            )
        if "whole_system_ac" not in domains:
            raise RecipeValidationError(
                f"recipe.measurement.scope.{stage} must include whole_system_ac"
            )

    raw_runs = raw["runs"]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise RecipeValidationError("recipe.runs must be a non-empty list")
    runs: list[JournalRun] = []
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    for index, raw_run in enumerate(raw_runs):
        where = f"recipe.runs[{index}]"
        run = _mapping(raw_run, where)
        _closed_keys(run, allowed=_RUN_KEYS, required=_RUN_KEYS, where=where)
        run_id = _slug(run["id"], f"{where}.id")
        if run_id in seen_ids:
            raise RecipeValidationError(f"duplicate run id: {run_id}")
        seen_ids.add(run_id)
        stage = run["stage"]
        if not isinstance(stage, str) or stage not in _STAGES:
            raise RecipeValidationError(
                f"{where}.stage must be one of: {', '.join(sorted(_STAGES))}"
            )
        if stage not in scope:
            raise RecipeValidationError(
                f"recipe.measurement.scope is missing the {stage!r} run stage"
            )
        problem = _slug(run["problem"], f"{where}.problem")
        policy = _slug(run["policy"], f"{where}.policy")
        output_subdir = _safe_relative_path(run["output_subdir"], f"{where}.output_subdir")
        output_key = output_subdir.as_posix()
        if output_key in seen_outputs:
            raise RecipeValidationError(f"duplicate run output_subdir: {output_key}")
        seen_outputs.add(output_key)
        max_walltime_s = _positive_number(run["max_walltime_s"], f"{where}.max_walltime_s")
        if max_walltime_s < minimum_duration:
            raise RecipeValidationError(
                f"{where}.max_walltime_s must be >= minimum_block_duration_s"
            )
        runs.append(
            JournalRun(
                run_id=run_id,
                stage=stage,
                problem=problem,
                size=_integer(run["size"], f"{where}.size", minimum=1),
                policy=policy,
                seed=_integer(run["seed"], f"{where}.seed", minimum=0),
                repetitions=_integer(run["repetitions"], f"{where}.repetitions", minimum=1),
                items_per_block=_integer(
                    run["items_per_block"], f"{where}.items_per_block", minimum=1
                ),
                max_walltime_s=max_walltime_s,
                output_subdir=output_key,
            )
        )

    return AETJournalRecipe(
        name=name,
        host_id=host_id,
        execution_layer=execution_layer,
        gpu_index=gpu_index,
        output_root=output_path.as_posix(),
        qualification=qualification,
        measurement=dict(measurement),
        runs=tuple(runs),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _current_git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _current_git_clean(excluded: tuple[Path, ...] = ()) -> bool:
    try:
        root_value = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return False
    root = Path(root_value).resolve()
    excluded_paths = {path.resolve() for path in excluded}
    entries = [entry for entry in status.split(b"\0") if entry]
    index = 0
    while index < len(entries):
        entry = entries[index]
        if len(entry) < 4:
            return False
        state = entry[:2].decode("ascii", errors="replace")
        relative = entry[3:].decode("utf-8", errors="surrogateescape")
        if "R" in state or "C" in state:
            return False
        if (root / relative).resolve() not in excluded_paths:
            return False
        index += 1
    return True


def _qualification_report(recipe: AETJournalRecipe) -> dict[str, Any]:
    preflight_path = Path(recipe.qualification["preflight_report"])
    calibration_path = Path(recipe.qualification["calibration_manifest"])
    preflight = _load_json_mapping(preflight_path)
    calibration = _load_json_mapping(calibration_path)
    preflight_mapping = preflight or {}
    preflight_backends = _dict_or_empty(preflight_mapping.get("backends"))
    preflight_nvml = _dict_or_empty(preflight_backends.get("nvml"))
    preflight_selected_device = _dict_or_empty(preflight_nvml.get("selected_device"))
    preflight_readiness = _dict_or_empty(preflight_mapping.get("readiness"))
    preflight_git = _dict_or_empty(preflight_mapping.get("git"))
    preflight_calibration = _dict_or_empty(preflight_mapping.get("calibration"))
    preflight_system = _dict_or_empty(preflight_mapping.get("system"))
    calibration_sha256: str | None = None
    calibration_error: str | None = None
    validated_calibration: ValidatedCalibration | None = None
    if calibration is not None:
        try:
            validated_calibration = validate_calibration_manifest(
                calibration,
                manifest_path=calibration_path,
            )
            calibration_sha256 = _sha256_file(calibration_path)
        except (CalibrationValidationError, OSError) as exc:
            calibration_error = str(exc)
    evidence_files = validated_calibration.evidence_files if validated_calibration else ()
    current_git_commit = _current_git_commit()
    current_git_clean = _current_git_clean((preflight_path, calibration_path, *evidence_files))
    git_commit_matches = bool(
        current_git_commit
        and isinstance(preflight_git, dict)
        and preflight_git.get("available") is True
        and preflight_git.get("sha") == current_git_commit
        and preflight_git.get("dirty") is False
        and current_git_clean
    )
    preflight_valid = bool(
        preflight
        and preflight.get("schema_version") == "aet-preflight/v1"
        and preflight.get("host_id") == recipe.host_id
        and preflight_system.get("execution_layer") == recipe.execution_layer
        and preflight_readiness.get("measured_smoke_ready") is True
        and preflight_readiness.get("confirmatory_primary_ready") is True
        and preflight_readiness.get("calibration_compatible") is True
        and preflight_readiness.get("git_clean") is True
        and git_commit_matches
        and preflight_nvml.get("available") is True
        and preflight_nvml.get("active_probe") is True
        and preflight_nvml.get("selected_index") == recipe.gpu_index
        and preflight_nvml.get("selection_valid") is True
        and preflight_selected_device.get("index") == recipe.gpu_index
        and preflight_selected_device.get("selected") is True
        and preflight_selected_device.get("measurement_mode_supported") is True
        and isinstance(preflight_selected_device.get("device_id_sha256"), str)
        and _SHA256_RE.fullmatch(preflight_selected_device["device_id_sha256"])
    )
    calibration_valid = bool(
        validated_calibration
        and validated_calibration.host_id == recipe.host_id
        and validated_calibration.execution_layer == recipe.execution_layer
        and validated_calibration.gpu_index == recipe.gpu_index
        and validated_calibration.gpu_device_id_sha256
        == preflight_selected_device.get("device_id_sha256")
    )
    preflight_calibration_valid = bool(
        preflight_calibration.get("passed") is True
        and preflight_calibration.get("schema_version") == CALIBRATION_SCHEMA_VERSION
        and preflight_calibration.get("host_id") == recipe.host_id
        and preflight_calibration.get("execution_layer") == recipe.execution_layer
        and preflight_calibration.get("gpu_index") == recipe.gpu_index
        and preflight_calibration.get("gpu_device_id_sha256")
        == preflight_selected_device.get("device_id_sha256")
    )
    linked = bool(
        preflight_valid
        and calibration_valid
        and preflight_calibration_valid
        and calibration_sha256 is not None
        and preflight_calibration.get("sha256") == calibration_sha256
    )
    return {
        "preflight_report": preflight_path.as_posix(),
        "preflight_valid": preflight_valid,
        "current_git_commit": current_git_commit,
        "current_git_clean": current_git_clean,
        "git_commit_matches": git_commit_matches,
        "selected_gpu_index": recipe.gpu_index,
        "calibration_manifest": calibration_path.as_posix(),
        "calibration_valid": calibration_valid,
        "calibration_validation_error": calibration_error,
        "preflight_calibration_valid": preflight_calibration_valid,
        "preflight_links_calibration_sha256": linked,
        "ready_to_execute": preflight_valid and calibration_valid and linked,
    }


def dry_run(path: Path) -> dict[str, Any]:
    """Validate and report bounds without executing jobs or writing outputs."""
    recipe = load_aet_journal_recipe(path)
    run_count = sum(run.repetitions for run in recipe.runs)
    maximum_walltime_s = sum(run.repetitions * run.max_walltime_s for run in recipe.runs)
    qualification = _qualification_report(recipe)
    report = {
        "status": "valid" if qualification["ready_to_execute"] else "valid_not_qualified",
        "schema_version": SCHEMA_VERSION,
        "name": recipe.name,
        "host_id": recipe.host_id,
        "execution_layer": recipe.execution_layer,
        "gpu_index": recipe.gpu_index,
        "output_root": recipe.output_root,
        "run_definitions": len(recipe.runs),
        "run_count": run_count,
        "maximum_single_run_walltime_s": max(run.max_walltime_s for run in recipe.runs),
        "maximum_walltime_s": maximum_walltime_s,
        "maximum_walltime_hours": maximum_walltime_s / 3600.0,
        "qualification": qualification,
        "ready_to_execute": qualification["ready_to_execute"],
        "writes_performed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report
