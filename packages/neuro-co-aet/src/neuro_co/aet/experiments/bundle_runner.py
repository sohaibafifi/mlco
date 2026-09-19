"""Fail-closed bundle runner for qualified AET instrumentation smoke measurements.

The caller supplies stage implementations that return in-memory artifacts.
The runner validates the
existing recipe, preflight, calibration, and Git qualification before creating
anything. It then writes a hidden sibling directory and promotes it with one
atomic rename only after every artifact and checksum has been verified.

No physical wall-meter or scientific solver adapter lives here. Bundles made
here are permanently marked non-scientific and ineligible for paper results.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from ctypes import CDLL, c_char_p, c_int, c_uint, get_errno
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Final

from neuro_co.aet.experiments.journal_recipe import (
    AETJournalRecipe,
    CalibrationValidationError,
    JournalRun,
    _qualification_report,
    load_aet_journal_recipe,
    validate_calibration_manifest,
)

BUNDLE_MANIFEST_SCHEMA: Final = "aet-bundle-manifest/v1"
REPORT_INPUT_MANIFEST_SCHEMA: Final = "aet-report-input-manifest/v1"
VALIDATION_SCHEMA: Final = "aet-validation/v1"
INSTANCE_MANIFEST_SCHEMA: Final = "aet-instance-manifest/v1"
TRAINING_DATASET_MANIFEST_SCHEMA: Final = "aet-training-dataset-manifest/v1"
NVML_DIAGNOSTIC_SCHEMA: Final = "aet-nvml-diagnostic/v1"
SMOKE_PURPOSE: Final = "instrumentation_smoke"

_REQUIRED_STAGES: Final = frozenset({"training", "inference", "baseline"})
_SAFE_COMPONENT: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ENERGY_FILENAMES: Final = {
    "training": "energy_train.json",
    "inference": "energy_eval.json",
    "baseline": "energy_baseline.json",
}
_ENERGY_ROLES: Final = {
    "training": "training",
    "inference": "inference",
    "baseline": "baseline",
}
_REPORT_RUN_BOUND_ROLES: Final = frozenset({"wall_trace", "nvml_trace", "validation_evidence"})
_ENERGY_UNITS: Final = {
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
_ENERGY_KEYS: Final = frozenset(
    {
        "schema_version",
        "units",
        "duration_s",
        "energy_j",
        "energy_gpu_j",
        "energy_cpu_j",
        "energy_dram_j",
        "co2_operational_kg",
        "co2_embodied_kg",
        "co2_total_kg",
        "avg_power_w",
        "items_processed",
        "throughput_items_per_s",
        "backend",
        "energy_domains",
        "measurement_scope",
        "hardware",
        "extra",
    }
)
_RUNNER_OWNED_ENERGY_METADATA: Final = frozenset(
    {
        "measurement_qualified_confirmatory",
        "measurement_qualified_instrumentation",
        "purpose",
        "scientific_use",
        "confirmatory_eligible",
        "fallback",
        "host_id",
        "calibration_sha256",
        "provenance_sha256",
        "artifact_sha256",
        "instance_manifest_sha256",
        "run_id",
        "stage",
        "repetition_index",
        "role",
        "execution_layer",
        "git_sha",
        "gpu_index",
        "gpu_device_id_sha256",
        "preflight_sha256",
        "training_dataset_manifest_sha256",
        "problem",
        "size",
        "size_key",
        "gap_to_reference_pct",
        "quality_feasible_confirmatory",
        "items_per_cycle",
        "workload_cycle_count",
    }
)
_VALIDATION_INPUT_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "complete",
        "independent",
        "validator",
        "checks",
        "metrics",
        "expected_records",
        "validated_records",
        "failure_count",
        "excluded_count",
        "gap_to_reference_pct",
        "quality_feasible_confirmatory",
    }
)
_VALIDATION_CHECK_KEYS: Final = frozenset({"name", "passed"})
_INSTANCE_KEYS: Final = frozenset(
    {
        "schema_version",
        "manifest_id",
        "status",
        "complete",
        "split",
        "problem",
        "size",
        "item_count",
        "entries",
    }
)
_TRAINING_DATASET_KEYS: Final = frozenset(
    {
        "schema_version",
        "dataset_id",
        "status",
        "complete",
        "split",
        "problem",
        "size",
        "item_count",
        "generation",
        "shards",
    }
)
_NVML_KEYS: Final = frozenset(
    {
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
)
_WALL_METER_KEYS: Final = frozenset(
    {
        "duration_s",
        "sample_interval_s",
        "sample_count",
        "dropped_sample_count",
        "trace_sha256",
        "calibration_id",
        "device_id_sha256",
        "started_at_utc",
        "ended_at_utc",
        "start_monotonic_s",
        "end_monotonic_s",
    }
)
_FINGERPRINT_KEYS: Final = frozenset({"path", "size_bytes", "sha256"})
_PROVENANCE_INPUT_ALLOWED_KEYS: Final = frozenset(
    {
        "timestamp_start",
        "timestamp_end",
        "python_version",
        "python_executable",
        "platform",
        "cpu",
        "memory_total_b",
        "gpus",
        "cuda_version",
        "environment_hash",
        "library_versions",
        "lockfiles",
        "seeds",
        "config",
        "config_sha256",
        "artifacts",
        "runtime_controls",
        "measurement",
        "extra",
        "hostname",
    }
)
_RUNTIME_CONTROL_KEYS: Final = frozenset(
    {
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
)
_SERVICE_BOUNDARY_KEYS: Final = frozenset({"mode", "initialization_reuse", "includes", "excludes"})
_WORKLOAD_AUDIT_KEYS: Final = frozenset(
    {
        "schema_version",
        "run_id",
        "stage",
        "repetition_index",
        "items_per_cycle",
        "workload_cycle_count",
        "items_processed",
        "measurement_duration_s",
        "idle_gap_count",
        "idle_gap_total_s",
    }
)


class AETBundleError(RuntimeError):
    """Base error for a bundle that cannot be safely planned or promoted."""


class AETBundleQualificationError(AETBundleError):
    """Raised before any write when execution qualification is incomplete."""


class AETStageContractError(AETBundleError):
    """Raised when an injected executor returns incomplete or invalid evidence."""


@dataclass(frozen=True)
class BundleQualification:
    """Validated identities that every stage artifact must bind to."""

    git_sha: str
    host_id: str
    execution_layer: str
    gpu_index: int
    gpu_device_id_sha256: str
    cpu: str
    logical_cpu_count: int
    ram_bytes: int
    windows_build: str
    wsl_version: str | None
    wsl_kernel: str | None
    nvidia_driver: str
    cuda_version: str
    calibration_id: str
    calibration_sha256: str
    wall_meter_device_id_sha256: str
    wall_meter_sample_interval_s: float
    maximum_timestamp_alignment_error_s: float
    maximum_missing_sample_percent: float
    minimum_samples_per_block: int
    preflight_path: Path
    preflight_sha256: str
    calibration_path: Path
    calibration_evidence: tuple[CalibrationEvidenceArtifact, ...]


@dataclass(frozen=True)
class CalibrationEvidenceArtifact:
    """One already-qualified calibration evidence file to preserve in the bundle."""

    role: str
    source: Path
    relative_path: PurePosixPath
    sha256: str


@dataclass(frozen=True)
class StageExecutionRequest:
    """Read-only request passed to one scientific stage executor.

    ``run.items_per_block`` is one workload cycle. Complete cycles are repeated
    without idle gaps until the registered minimum duration is reached.
    ``run.repetitions`` remains the number of independent measurement blocks.
    """

    recipe_name: str
    run: JournalRun
    execution_id: str
    repetition_index: int
    minimum_block_duration_s: float
    required_measurement_domains: tuple[str, ...]
    qualification: BundleQualification


@dataclass(frozen=True)
class StageArtifacts:
    """Stage records plus raw energy traces returned entirely in memory."""

    energy: Mapping[str, Any]
    provenance: Mapping[str, Any]
    validation: Mapping[str, Any]
    instance_manifest: Mapping[str, Any]
    log: str
    wall_trace: bytes
    nvml_trace: bytes | None
    workload_audit: Mapping[str, Any]


StageExecutor = Callable[[StageExecutionRequest], StageArtifacts]


@dataclass(frozen=True)
class BundlePlan:
    """Non-writing result from validation or dry-run planning."""

    recipe: AETJournalRecipe
    target: Path
    ready_to_execute: bool
    qualification_report: Mapping[str, Any]
    recipe_path: Path
    recipe_sha256: str
    preflight_sha256: str | None
    calibration_sha256: str | None
    writes_performed: bool = False


@dataclass(frozen=True)
class BundleResult:
    """Identity of an atomically promoted immutable bundle."""

    bundle_id: str
    path: Path
    bundle_manifest_sha256: str
    report_input_manifest: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AETStageContractError(f"{where} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise AETStageContractError(f"{where} must be finite and {qualifier}")
    return number


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AETStageContractError(f"{where} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AETStageContractError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AETStageContractError(f"{where} must be a non-empty string")
    return value.strip()


def _aware_datetime(value: Any, where: str) -> datetime:
    raw = _nonempty_string(value, where)
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AETStageContractError(f"{where} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AETStageContractError(f"{where} must include a timezone offset")
    return parsed


def _json_object(value: Mapping[str, Any], where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AETStageContractError(f"{where} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise AETStageContractError(f"{where} keys must be strings")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AETStageContractError(f"{where} must be finite JSON data") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover, guarded above
        raise AETStageContractError(f"{where} must encode one JSON object")
    return decoded


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _safe_component(value: str, where: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise AETBundleError(
            f"{where} must contain only lowercase letters, digits, '.', '_' and '-'"
        )
    if value in {".", ".."}:
        raise AETBundleError(f"{where} may not be '.' or '..'")
    return value


def _assert_no_symlink_ancestors(workspace_root: Path, path: Path) -> None:
    """Reject symlinks in every existing component under the repository root."""
    relative = path.relative_to(workspace_root)
    cursor = workspace_root
    for component in relative.parts:
        cursor = cursor / component
        if (cursor.exists() or cursor.is_symlink()) and cursor.is_symlink():
            raise AETBundleError(f"refusing symlinked bundle path component: {cursor}")


def _safe_target(workspace_root: Path, output_root: str, bundle_id: str) -> Path:
    root = workspace_root.resolve(strict=True)
    if root != Path.cwd().resolve(strict=True):
        raise AETBundleError("workspace_root must be the current repository working directory")
    lexical_output = root.joinpath(*PurePosixPath(output_root).parts)
    _assert_no_symlink_ancestors(root, lexical_output)
    resolved_output = lexical_output.resolve(strict=False)
    try:
        resolved_output.relative_to(root)
    except ValueError as exc:
        raise AETBundleError("recipe output_root resolves outside the repository") from exc
    target = resolved_output / _safe_component(bundle_id, "bundle_id")
    _assert_no_symlink_ancestors(root, target)
    return target


def _required_stage_contract(recipe: AETJournalRecipe) -> None:
    stages = {run.stage for run in recipe.runs}
    missing = sorted(_REQUIRED_STAGES - stages)
    extra = sorted(stages - _REQUIRED_STAGES)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("extra: " + ", ".join(extra))
        raise AETBundleError("bundle recipe stage set is invalid (" + "; ".join(details) + ")")
    seen_outputs: set[str] = set()
    for run in recipe.runs:
        output = _safe_stage_relative(run.output_subdir, f"run {run.run_id!r} output_subdir")
        normalized = output.as_posix()
        if normalized in seen_outputs:
            raise AETBundleError(f"duplicate run output_subdir: {normalized}")
        seen_outputs.add(normalized)


def _safe_evidence_relative(value: Any, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AETBundleQualificationError(f"{where} must be a portable relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AETBundleQualificationError(f"{where} must remain inside the bundle")
    return relative


def _safe_stage_relative(value: Any, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AETStageContractError(f"{where} must be a portable relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AETStageContractError(f"{where} must remain relative")
    return relative


def _calibration_evidence_artifacts(
    calibration: dict[str, Any],
    *,
    calibration_path: Path,
) -> tuple[CalibrationEvidenceArtifact, ...]:
    try:
        validated = validate_calibration_manifest(
            calibration,
            manifest_path=calibration_path,
        )
    except (CalibrationValidationError, OSError) as exc:
        raise AETBundleQualificationError(
            "calibration evidence changed after qualification"
        ) from exc
    evidence = calibration["evidence"]
    raw_manifest_relative = _safe_evidence_relative(
        evidence["raw_trace_manifest"],
        "calibration.evidence.raw_trace_manifest",
    )
    analysis_relative = _safe_evidence_relative(
        evidence["analysis_script"],
        "calibration.evidence.analysis_script",
    )
    raw_manifest_path = calibration_path.parent.joinpath(*raw_manifest_relative.parts)
    try:
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        raw_files = raw_manifest["traces"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AETBundleQualificationError(
            "qualified calibration trace manifest cannot be reread"
        ) from exc
    if not isinstance(raw_files, list):
        raise AETBundleQualificationError("calibration trace manifest files must be a list")
    trace_relatives = tuple(
        _safe_evidence_relative(
            entry.get("raw_trace") if isinstance(entry, dict) else None,
            f"calibration trace manifest traces[{index}].raw_trace",
        )
        for index, entry in enumerate(raw_files)
    )
    expected_paths = (
        raw_manifest_path.resolve(),
        *(
            raw_manifest_path.parent.joinpath(*relative.parts).resolve()
            for relative in trace_relatives
        ),
        calibration_path.parent.joinpath(*analysis_relative.parts).resolve(),
    )
    validated_paths = tuple(path.resolve() for path in validated.evidence_files)
    if expected_paths != validated_paths:
        raise AETBundleQualificationError(
            "calibration evidence inventory disagrees with strict validation"
        )
    destination_root = PurePosixPath("qualification")
    roles_and_relatives = (
        ("calibration_trace_manifest", destination_root / raw_manifest_relative),
        *(
            (
                "calibration_trace",
                destination_root / raw_manifest_relative.parent / relative,
            )
            for relative in trace_relatives
        ),
        ("calibration_analysis", destination_root / analysis_relative),
    )
    artifacts: list[CalibrationEvidenceArtifact] = []
    seen: set[PurePosixPath] = set()
    for (role, relative), source in zip(
        roles_and_relatives,
        validated_paths,
        strict=True,
    ):
        if relative in seen:
            raise AETBundleQualificationError("calibration evidence paths must be unique")
        seen.add(relative)
        artifacts.append(
            CalibrationEvidenceArtifact(
                role=role,
                source=source,
                relative_path=relative,
                sha256=_sha256_file(source),
            )
        )
    return tuple(artifacts)


def validate_aet_smoke_bundle(
    recipe_path: str | Path,
    *,
    bundle_id: str,
    workspace_root: str | Path | None = None,
    require_ready: bool = False,
) -> BundlePlan:
    """Validate a smoke-only bundle plan without modifying any path."""
    resolved_recipe_path = Path(recipe_path).resolve(strict=True)
    recipe_before = resolved_recipe_path.read_bytes()
    recipe = load_aet_journal_recipe(resolved_recipe_path)
    recipe_after = resolved_recipe_path.read_bytes()
    if recipe_before != recipe_after:
        raise AETBundleError("recipe changed during bundle validation")
    _required_stage_contract(recipe)
    root = Path.cwd() if workspace_root is None else Path(workspace_root)
    target = _safe_target(root, recipe.output_root, bundle_id)
    qualification = _qualification_report(recipe)
    ready = qualification.get("ready_to_execute") is True
    preflight_sha: str | None = None
    calibration_sha: str | None = None
    if ready:
        try:
            preflight_sha = _sha256_file(Path(str(qualification["preflight_report"])))
            calibration_sha = _sha256_file(Path(str(qualification["calibration_manifest"])))
        except (KeyError, OSError) as exc:
            raise AETBundleQualificationError(
                "qualified preflight/calibration disappeared after validation"
            ) from exc
    if require_ready and not ready:
        raise AETBundleQualificationError(
            "AET bundle execution requires passed recipe, preflight, calibration, and clean Git"
        )
    return BundlePlan(
        recipe=recipe,
        target=target,
        ready_to_execute=ready,
        qualification_report=dict(qualification),
        recipe_path=resolved_recipe_path,
        recipe_sha256=_sha256_bytes(recipe_before),
        preflight_sha256=preflight_sha,
        calibration_sha256=calibration_sha,
    )


def dry_run_aet_smoke_bundle(
    recipe_path: str | Path,
    *,
    bundle_id: str,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible, non-writing instrumentation-smoke plan."""
    plan = validate_aet_smoke_bundle(
        recipe_path,
        bundle_id=bundle_id,
        workspace_root=workspace_root,
    )
    return {
        "status": "valid" if plan.ready_to_execute else "valid_not_qualified",
        "purpose": SMOKE_PURPOSE,
        "scientific_use": False,
        "confirmatory_eligible": False,
        "bundle_id": bundle_id,
        "target": plan.target.as_posix(),
        "stages": [run.stage for run in plan.recipe.runs],
        "run_count": sum(run.repetitions for run in plan.recipe.runs),
        "ready_to_execute": plan.ready_to_execute,
        "qualification": dict(plan.qualification_report),
        "writes_performed": False,
    }


def _validate_preflight_snapshot(
    preflight: dict[str, Any],
    *,
    recipe: AETJournalRecipe,
    git_sha: str,
    calibration_sha256: str,
    gpu_device_id_sha256: str,
) -> None:
    """Validate the exact preflight bytes bound to the execution snapshot."""

    def mapping(value: Any, where: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise AETBundleQualificationError(f"{where} must be a mapping")
        return value

    if preflight.get("schema_version") != "aet-preflight/v1":
        raise AETBundleQualificationError("qualified preflight schema is invalid")
    if preflight.get("host_id") != recipe.host_id:
        raise AETBundleQualificationError("qualified preflight host identity is invalid")
    _aware_datetime(preflight.get("generated_at"), "preflight.generated_at")
    system = mapping(preflight.get("system"), "preflight.system")
    if system.get("execution_layer") != recipe.execution_layer:
        raise AETBundleQualificationError("qualified preflight execution layer is invalid")
    git = mapping(preflight.get("git"), "preflight.git")
    if (
        git.get("available") is not True
        or git.get("dirty") is not False
        or git.get("sha") != git_sha
    ):
        raise AETBundleQualificationError("qualified preflight Git snapshot is invalid")
    readiness = mapping(preflight.get("readiness"), "preflight.readiness")
    required_readiness = {
        "environment_ready",
        "git_clean",
        "gpu_component_ready",
        "calibration_compatible",
        "system_energy_ready",
        "measured_smoke_ready",
        "confirmatory_primary_ready",
    }
    if any(readiness.get(field) is not True for field in required_readiness):
        raise AETBundleQualificationError("qualified preflight readiness is incomplete")
    strict_tracker = mapping(preflight.get("strict_tracker"), "preflight.strict_tracker")
    if strict_tracker.get("importable") is not True or strict_tracker.get("supported") is not True:
        raise AETBundleQualificationError("qualified strict tracker is unavailable")
    backends = mapping(preflight.get("backends"), "preflight.backends")
    if (
        backends.get("codecarbon_allowed_primary") is not False
        or backends.get("tdp_allowed_primary") is not False
    ):
        raise AETBundleQualificationError("qualified preflight permits a prohibited backend")
    nvml = mapping(backends.get("nvml"), "preflight.backends.nvml")
    selected = mapping(nvml.get("selected_device"), "preflight.backends.nvml.selected_device")
    if (
        nvml.get("available") is not True
        or nvml.get("active_probe") is not True
        or nvml.get("selection_valid") is not True
        or nvml.get("selected_index") != recipe.gpu_index
        or selected.get("index") != recipe.gpu_index
        or selected.get("selected") is not True
        or selected.get("measurement_mode_supported") is not True
        or selected.get("device_id_sha256") != gpu_device_id_sha256
    ):
        raise AETBundleQualificationError("qualified preflight NVML selection is invalid")
    calibration = mapping(preflight.get("calibration"), "preflight.calibration")
    expected_calibration = {
        "passed": True,
        "schema_version": "aet-calibration/v1",
        "host_id": recipe.host_id,
        "execution_layer": recipe.execution_layer,
        "gpu_index": recipe.gpu_index,
        "gpu_device_id_sha256": gpu_device_id_sha256,
        "sha256": calibration_sha256,
    }
    if any(calibration.get(key) != value for key, value in expected_calibration.items()):
        raise AETBundleQualificationError("qualified preflight calibration binding is invalid")


def _load_qualification(plan: BundlePlan) -> BundleQualification:
    report = plan.qualification_report
    if report.get("ready_to_execute") is not True:
        raise AETBundleQualificationError("qualification changed before bundle execution")
    preflight_path = Path(str(report.get("preflight_report", "")))
    calibration_path = Path(str(report.get("calibration_manifest", "")))
    try:
        preflight_bytes = preflight_path.read_bytes()
        calibration_bytes = calibration_path.read_bytes()
        calibration = json.loads(calibration_bytes)
        preflight = json.loads(preflight_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise AETBundleQualificationError("cannot reread qualified preflight/calibration") from exc
    if not isinstance(calibration, dict) or not isinstance(preflight, dict):
        raise AETBundleQualificationError("qualified preflight/calibration must be JSON objects")
    calibration_evidence = _calibration_evidence_artifacts(
        calibration,
        calibration_path=calibration_path,
    )
    try:
        hardware = calibration["hardware"]
        wall_meter = calibration["wall_meter"]
        criteria = calibration["criteria"]
        git_sha = str(report["current_git_commit"])
        gpu_device_hash = str(hardware["gpu_device_id_sha256"])
        wall_device_hash = str(wall_meter["device_id_hash"])
        calibration_id = str(calibration["calibration_id"])
        calibration_sha = _sha256_bytes(calibration_bytes)
        preflight_sha = _sha256_bytes(preflight_bytes)
    except (KeyError, TypeError, ValueError) as exc:
        raise AETBundleQualificationError("qualified calibration identity is incomplete") from exc
    if plan.calibration_sha256 != calibration_sha or plan.preflight_sha256 != preflight_sha:
        raise AETBundleQualificationError("calibration changed after qualification")
    if report.get("git_commit_matches") is not True or _GIT_OBJECT_ID.fullmatch(git_sha) is None:
        raise AETBundleQualificationError("qualified Git commit identity is invalid")
    if _SHA256.fullmatch(gpu_device_hash) is None or _SHA256.fullmatch(wall_device_hash) is None:
        raise AETBundleQualificationError("qualified device identity hash is invalid")
    _validate_preflight_snapshot(
        preflight,
        recipe=plan.recipe,
        git_sha=git_sha,
        calibration_sha256=calibration_sha,
        gpu_device_id_sha256=gpu_device_hash,
    )
    return BundleQualification(
        git_sha=git_sha,
        host_id=plan.recipe.host_id,
        execution_layer=plan.recipe.execution_layer,
        gpu_index=plan.recipe.gpu_index,
        gpu_device_id_sha256=gpu_device_hash,
        cpu=str(hardware["cpu"]),
        logical_cpu_count=int(hardware["logical_cpu_count"]),
        ram_bytes=int(hardware["ram_bytes"]),
        windows_build=str(hardware["windows_build"]),
        wsl_version=(str(hardware["wsl_version"]) if hardware["wsl_version"] is not None else None),
        wsl_kernel=(str(hardware["wsl_kernel"]) if hardware["wsl_kernel"] is not None else None),
        nvidia_driver=str(hardware["nvidia_driver"]),
        cuda_version=str(hardware["cuda_version"]),
        calibration_id=calibration_id,
        calibration_sha256=calibration_sha,
        wall_meter_device_id_sha256=wall_device_hash,
        wall_meter_sample_interval_s=float(wall_meter["sample_interval_s"]),
        maximum_timestamp_alignment_error_s=float(criteria["max_timestamp_alignment_error_s"]),
        maximum_missing_sample_percent=float(criteria["max_missing_sample_percent"]),
        minimum_samples_per_block=int(criteria["minimum_samples_per_block"]),
        preflight_path=preflight_path,
        preflight_sha256=preflight_sha,
        calibration_path=calibration_path,
        calibration_evidence=calibration_evidence,
    )


def _close_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("extra " + ", ".join(extra))
        raise AETStageContractError(f"{where} has invalid keys ({'; '.join(parts)})")


def _validate_time_bounds(
    value: Mapping[str, Any],
    *,
    duration_s: float,
    tolerance_s: float,
    where: str,
) -> None:
    started = _aware_datetime(value["started_at_utc"], f"{where}.started_at_utc")
    ended = _aware_datetime(value["ended_at_utc"], f"{where}.ended_at_utc")
    if started.utcoffset() != timedelta(0) or ended.utcoffset() != timedelta(0):
        raise AETStageContractError(f"{where} timestamps must use UTC")
    observed = (ended - started).total_seconds()
    if observed <= 0.0 or not math.isclose(observed, duration_s, abs_tol=tolerance_s):
        raise AETStageContractError(f"{where} UTC bounds disagree with duration_s")


def _validate_wall_meter(
    value: Any,
    *,
    energy_duration_s: float,
    qualification: BundleQualification,
) -> None:
    if not isinstance(value, dict):
        raise AETStageContractError("energy.extra.wall_meter must be a mapping")
    _close_keys(value, _WALL_METER_KEYS, "energy.extra.wall_meter")
    duration = _finite_number(value["duration_s"], "wall_meter.duration_s", positive=True)
    if not math.isclose(
        duration,
        energy_duration_s,
        abs_tol=qualification.maximum_timestamp_alignment_error_s,
    ):
        raise AETStageContractError("wall-meter and energy durations disagree")
    sample_interval = _finite_number(
        value["sample_interval_s"], "wall_meter.sample_interval_s", positive=True
    )
    if not math.isclose(sample_interval, qualification.wall_meter_sample_interval_s):
        raise AETStageContractError("wall-meter sample interval differs from calibration")
    sample_count = _integer(value["sample_count"], "wall_meter.sample_count", minimum=1)
    dropped = _integer(value["dropped_sample_count"], "wall_meter.dropped_sample_count")
    if sample_count < qualification.minimum_samples_per_block:
        raise AETStageContractError("wall-meter sample count is below calibration minimum")
    missing_percent = 100.0 * dropped / (sample_count + dropped)
    if missing_percent > qualification.maximum_missing_sample_percent:
        raise AETStageContractError("wall-meter missing sample percentage exceeds calibration")
    expected_samples = math.floor(duration / sample_interval) + 1
    if abs((sample_count + dropped) - expected_samples) > 1:
        raise AETStageContractError(
            "wall-meter sample coverage disagrees with duration and sample interval"
        )
    _sha256(value["trace_sha256"], "wall_meter.trace_sha256")
    if value["calibration_id"] != qualification.calibration_id:
        raise AETStageContractError("wall-meter calibration_id mismatch")
    if value["device_id_sha256"] != qualification.wall_meter_device_id_sha256:
        raise AETStageContractError("wall-meter device identity mismatch")
    _validate_time_bounds(
        value,
        duration_s=duration,
        tolerance_s=qualification.maximum_timestamp_alignment_error_s,
        where="wall_meter",
    )
    start_monotonic = _finite_number(value["start_monotonic_s"], "wall_meter.start_monotonic_s")
    end_monotonic = _finite_number(value["end_monotonic_s"], "wall_meter.end_monotonic_s")
    if not math.isclose(
        end_monotonic - start_monotonic,
        duration,
        abs_tol=qualification.maximum_timestamp_alignment_error_s,
    ):
        raise AETStageContractError("wall-meter monotonic bounds disagree with duration_s")


def _validate_nvml_diagnostic(
    value: Any,
    *,
    energy_duration_s: float,
    qualification: BundleQualification,
) -> None:
    if not isinstance(value, dict):
        raise AETStageContractError("energy.extra.diagnostics.nvml must be a mapping")
    _close_keys(value, _NVML_KEYS, "energy.extra.diagnostics.nvml")
    if value["schema_version"] != NVML_DIAGNOSTIC_SCHEMA:
        raise AETStageContractError("NVML diagnostic schema is invalid")
    if value["source"] != "independent_concurrent_sampler":
        raise AETStageContractError("NVML evidence must come from a concurrent independent sampler")
    if value["gpu_index"] != qualification.gpu_index:
        raise AETStageContractError("NVML diagnostic GPU index mismatch")
    if value["device_id_sha256"] != qualification.gpu_device_id_sha256:
        raise AETStageContractError("NVML diagnostic device identity mismatch")
    mode = value["measurement_mode"]
    if mode not in {"nvml_total_energy_counter", "nvml_power_integration"}:
        raise AETStageContractError("NVML diagnostic measurement_mode is invalid")
    _finite_number(value["energy_j"], "nvml.energy_j")
    duration = _finite_number(value["duration_s"], "nvml.duration_s", positive=True)
    if not math.isclose(
        duration,
        energy_duration_s,
        abs_tol=qualification.maximum_timestamp_alignment_error_s,
    ):
        raise AETStageContractError("NVML and wall-meter durations disagree")
    _sha256(value["trace_sha256"], "nvml.trace_sha256")
    samples = _integer(value["sample_count"], "nvml.sample_count", minimum=2)
    dropped = _integer(value["dropped_sample_count"], "nvml.dropped_sample_count")
    interval = value["sampling_interval_s"]
    if mode == "nvml_total_energy_counter":
        if samples != 2 or interval is not None or dropped != 0:
            raise AETStageContractError(
                "NVML total-energy counter requires exactly two endpoints, null interval, "
                "and zero dropped samples"
            )
    else:
        sample_interval = _finite_number(interval, "nvml.sampling_interval_s", positive=True)
        if sample_interval > qualification.wall_meter_sample_interval_s:
            raise AETStageContractError("NVML sampling interval exceeds calibrated maximum")
        missing_percent = 100.0 * dropped / (samples + dropped)
        if missing_percent > qualification.maximum_missing_sample_percent:
            raise AETStageContractError("NVML missing sample percentage exceeds calibration")
        expected_samples = math.floor(duration / sample_interval) + 1
        if abs((samples + dropped) - expected_samples) > 1:
            raise AETStageContractError(
                "NVML sample coverage disagrees with duration and sample interval"
            )
    _validate_time_bounds(
        value,
        duration_s=duration,
        tolerance_s=qualification.maximum_timestamp_alignment_error_s,
        where="nvml",
    )


def _validate_energy(
    value: Mapping[str, Any],
    *,
    request: StageExecutionRequest,
) -> dict[str, Any]:
    energy = _json_object(value, "energy")
    _close_keys(energy, _ENERGY_KEYS, "energy")
    if energy["schema_version"] != "1.0" or energy["units"] != _ENERGY_UNITS:
        raise AETStageContractError("energy must use the exact canonical SI schema")
    duration = _finite_number(energy["duration_s"], "energy.duration_s", positive=True)
    if duration < request.minimum_block_duration_s:
        raise AETStageContractError("energy duration is below minimum_block_duration_s")
    if duration > request.run.max_walltime_s:
        raise AETStageContractError("energy duration exceeds the registered max_walltime_s")
    total = _finite_number(energy["energy_j"], "energy.energy_j", positive=True)
    components = sum(
        _finite_number(energy[key], f"energy.{key}")
        for key in ("energy_gpu_j", "energy_cpu_j", "energy_dram_j")
    )
    if components > total + max(1e-9, total * 1e-9):
        raise AETStageContractError("energy components exceed whole-system energy")
    operational = _finite_number(energy["co2_operational_kg"], "energy.co2_operational_kg")
    embodied = _finite_number(energy["co2_embodied_kg"], "energy.co2_embodied_kg")
    if embodied != 0.0:
        raise AETStageContractError("wall-meter primary record must not include embodied carbon")
    total_co2 = _finite_number(energy["co2_total_kg"], "energy.co2_total_kg")
    if not math.isclose(total_co2, operational + embodied, rel_tol=1e-9, abs_tol=1e-12):
        raise AETStageContractError("energy.co2_total_kg is inconsistent")
    items = _integer(energy["items_processed"], "energy.items_processed", minimum=1)
    if items < request.run.items_per_block or items % request.run.items_per_block != 0:
        raise AETStageContractError(
            "energy.items_processed must be a positive multiple of registered items_per_block"
        )
    throughput = _finite_number(energy["throughput_items_per_s"], "energy.throughput_items_per_s")
    if not math.isclose(throughput, items / duration, rel_tol=1e-9, abs_tol=1e-12):
        raise AETStageContractError("energy throughput is inconsistent")
    average_power = _finite_number(energy["avg_power_w"], "energy.avg_power_w")
    if not math.isclose(average_power, total / duration, rel_tol=1e-9, abs_tol=1e-12):
        raise AETStageContractError("energy average power is inconsistent")
    if energy["backend"] != "wall_meter":
        raise AETStageContractError("energy backend must be wall_meter")
    if energy["measurement_scope"] != "whole_system_ac":
        raise AETStageContractError("energy measurement_scope must be whole_system_ac")
    if energy["energy_domains"] != ["whole_system_ac"]:
        raise AETStageContractError("energy domains must be exactly whole_system_ac")
    hardware = energy["hardware"]
    if not isinstance(hardware, dict) or hardware.get("pue") != 1.0:
        raise AETStageContractError("energy hardware metadata must declare pue=1.0")
    extra = energy["extra"]
    if not isinstance(extra, dict):
        raise AETStageContractError("energy.extra must be a mapping")
    collision = sorted(_RUNNER_OWNED_ENERGY_METADATA.intersection(extra))
    if collision:
        raise AETStageContractError(
            "executor may not predeclare runner-owned energy metadata: " + ", ".join(collision)
        )
    if extra.get("allow_fallback") is not False or extra.get("tdp_fallback") is not False:
        raise AETStageContractError("energy must prove fallback was disabled and unused")
    if extra.get("run_status") != "completed" or extra.get("failure_type") is not None:
        raise AETStageContractError("energy record must come from a completed measurement")
    _validate_wall_meter(
        extra.get("wall_meter"),
        energy_duration_s=duration,
        qualification=request.qualification,
    )
    diagnostics = extra.get("diagnostics")
    if "gpu" in request.required_measurement_domains:
        if not isinstance(diagnostics, dict) or set(diagnostics) != {"nvml"}:
            raise AETStageContractError(
                "GPU-scoped energy must contain exactly one NVML diagnostic block"
            )
        _validate_nvml_diagnostic(
            diagnostics["nvml"],
            energy_duration_s=duration,
            qualification=request.qualification,
        )
    elif "diagnostics" in extra:
        raise AETStageContractError(
            "stage without registered GPU scope may not claim an NVML diagnostic"
        )
    return energy


def _validate_fingerprints(value: Any, where: str) -> None:
    if not isinstance(value, dict) or not value:
        raise AETStageContractError(f"{where} must be a non-empty mapping")
    for name, raw in value.items():
        _nonempty_string(name, f"{where} key")
        if not isinstance(raw, dict):
            raise AETStageContractError(f"{where}.{name} must be a mapping")
        _close_keys(raw, _FINGERPRINT_KEYS, f"{where}.{name}")
        _safe_stage_relative(raw["path"], f"{where}.{name}.path")
        _integer(raw["size_bytes"], f"{where}.{name}.size_bytes", minimum=1)
        _sha256(raw["sha256"], f"{where}.{name}.sha256")


def _validate_string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AETStageContractError(f"{where} must be a non-empty list")
    normalized = [_nonempty_string(item, f"{where} item") for item in value]
    if len(set(normalized)) != len(normalized):
        raise AETStageContractError(f"{where} must not contain duplicates")
    return normalized


def _validate_runtime_controls(value: Any, *, request: StageExecutionRequest) -> None:
    if not isinstance(value, dict):
        raise AETStageContractError("provenance.runtime_controls must be a mapping")
    _close_keys(value, _RUNTIME_CONTROL_KEYS, "provenance.runtime_controls")
    if value["schema_version"] != "aet-runtime-controls/v1":
        raise AETStageContractError("runtime controls schema is invalid")
    _nonempty_string(value["windows_build"], "runtime_controls.windows_build")
    if request.qualification.execution_layer == "wsl2":
        _nonempty_string(value["wsl_version"], "runtime_controls.wsl_version")
        _nonempty_string(value["wsl_kernel"], "runtime_controls.wsl_kernel")
        _integer(value["wsl_memory_limit_b"], "runtime_controls.wsl_memory_limit_b", minimum=1)
    elif any(
        value[field] is not None for field in ("wsl_version", "wsl_kernel", "wsl_memory_limit_b")
    ):
        raise AETStageContractError("native Windows runtime controls must have null WSL fields")
    logical = _integer(
        value["cpu_logical_processors"],
        "runtime_controls.cpu_logical_processors",
        minimum=1,
    )
    threads = _integer(value["cpu_threads"], "runtime_controls.cpu_threads", minimum=1)
    affinity = value["cpu_affinity"]
    if (
        not isinstance(affinity, list)
        or not affinity
        or any(isinstance(index, bool) or not isinstance(index, int) for index in affinity)
        or len(set(affinity)) != len(affinity)
        or any(index < 0 or index >= logical for index in affinity)
        or threads != len(affinity)
    ):
        raise AETStageContractError("runtime_controls.cpu_affinity is inconsistent")
    _nonempty_string(value["windows_power_plan"], "runtime_controls.windows_power_plan")
    _nonempty_string(value["cpu_power_mode"], "runtime_controls.cpu_power_mode")
    _nonempty_string(value["gpu_driver"], "runtime_controls.gpu_driver")
    _nonempty_string(value["cuda_version"], "runtime_controls.cuda_version")
    _finite_number(value["gpu_power_limit_w"], "runtime_controls.gpu_power_limit_w", positive=True)
    _finite_number(
        value["gpu_graphics_clock_mhz"],
        "runtime_controls.gpu_graphics_clock_mhz",
        positive=True,
    )
    _finite_number(
        value["gpu_memory_clock_mhz"],
        "runtime_controls.gpu_memory_clock_mhz",
        positive=True,
    )
    _nonempty_string(value["gpu_clock_policy"], "runtime_controls.gpu_clock_policy")
    if value["gpu_persistence_mode"] not in {"enabled", "disabled", "unsupported"}:
        raise AETStageContractError(
            "runtime_controls.gpu_persistence_mode must be enabled, disabled, or unsupported"
        )
    if value["warmup_completed"] is not True:
        raise AETStageContractError("runtime controls must prove completed warmup")
    _finite_number(
        value["warmup_duration_s"],
        "runtime_controls.warmup_duration_s",
        positive=True,
    )
    _nonempty_string(
        value["warmup_stability_criterion"],
        "runtime_controls.warmup_stability_criterion",
    )
    _sha256(value["warmup_trace_sha256"], "runtime_controls.warmup_trace_sha256")
    if value["background_load_status"] != "clear":
        raise AETStageContractError("runtime controls must report clear background load")
    _nonempty_string(value["background_load_check"], "runtime_controls.background_load_check")
    _sha256(
        value["background_load_trace_sha256"],
        "runtime_controls.background_load_trace_sha256",
    )
    boundary = value["service_boundary"]
    if not isinstance(boundary, dict):
        raise AETStageContractError("runtime_controls.service_boundary must be a mapping")
    _close_keys(boundary, _SERVICE_BOUNDARY_KEYS, "runtime_controls.service_boundary")
    allowed_modes = {"training"} if request.run.stage == "training" else {"warm", "cold"}
    if boundary["mode"] not in allowed_modes:
        raise AETStageContractError("service boundary mode does not match the stage")
    if not isinstance(boundary["initialization_reuse"], bool):
        raise AETStageContractError("service boundary initialization_reuse must be boolean")
    includes = _validate_string_list(boundary["includes"], "service_boundary.includes")
    excludes = _validate_string_list(boundary["excludes"], "service_boundary.excludes")
    if set(includes).intersection(excludes):
        raise AETStageContractError("service boundary includes and excludes must be disjoint")


def _validate_provenance(
    value: Mapping[str, Any],
    *,
    request: StageExecutionRequest,
) -> dict[str, Any]:
    provenance = _json_object(value, "provenance")
    unexpected = sorted(set(provenance) - _PROVENANCE_INPUT_ALLOWED_KEYS)
    if unexpected:
        raise AETStageContractError(
            "provenance contains unsupported fields: " + ", ".join(unexpected)
        )
    runner_owned = {
        "schema_version",
        "run_id",
        "host_id",
        "status",
        "failure_reason",
        "git_sha",
        "git_dirty",
        "dirty_patch_sha256",
        "execution_layer",
    }
    collision = sorted(runner_owned.intersection(provenance))
    if collision:
        raise AETStageContractError(
            "executor may not predeclare runner-owned provenance fields: " + ", ".join(collision)
        )
    gpus = provenance.get("gpus")
    if not isinstance(gpus, list) or not any(
        isinstance(gpu, dict)
        and gpu.get("index") == request.qualification.gpu_index
        and gpu.get("device_id_sha256") == request.qualification.gpu_device_id_sha256
        for gpu in gpus
    ):
        raise AETStageContractError("provenance does not bind the selected physical GPU")
    started = _aware_datetime(provenance.get("timestamp_start"), "provenance.timestamp_start")
    ended = _aware_datetime(provenance.get("timestamp_end"), "provenance.timestamp_end")
    if ended < started:
        raise AETStageContractError("provenance timestamp_end precedes timestamp_start")
    _nonempty_string(provenance.get("python_version"), "provenance.python_version")
    executable = _nonempty_string(
        provenance.get("python_executable"), "provenance.python_executable"
    )
    if "/" in executable or "\\" in executable or Path(executable).name != executable:
        raise AETStageContractError("provenance.python_executable must be a basename")
    _nonempty_string(provenance.get("platform"), "provenance.platform")
    cpu = _nonempty_string(provenance.get("cpu"), "provenance.cpu")
    if provenance.get("cuda_version") is not None:
        _nonempty_string(provenance["cuda_version"], "provenance.cuda_version")
    if "extra" in provenance and not isinstance(provenance["extra"], dict):
        raise AETStageContractError("provenance.extra must be a mapping")
    if provenance.get("hostname") is not None:
        raise AETStageContractError("provenance.hostname must be null for privacy")
    memory_total_b = _integer(
        provenance.get("memory_total_b"), "provenance.memory_total_b", minimum=1
    )
    libraries = provenance.get("library_versions")
    if not isinstance(libraries, dict) or not libraries:
        raise AETStageContractError("provenance.library_versions must be non-empty")
    for name, version in libraries.items():
        _nonempty_string(name, "provenance.library_versions key")
        _nonempty_string(version, f"provenance.library_versions.{name}")
    _validate_fingerprints(provenance.get("lockfiles"), "provenance.lockfiles")
    _validate_fingerprints(provenance.get("artifacts"), "provenance.artifacts")
    if provenance.get("seeds") != {"run": request.run.seed}:
        raise AETStageContractError("provenance.seeds must exactly bind the registered run seed")
    _validate_runtime_controls(provenance.get("runtime_controls"), request=request)
    runtime_controls = provenance["runtime_controls"]
    if (
        cpu != request.qualification.cpu
        or memory_total_b != request.qualification.ram_bytes
        or runtime_controls["cpu_logical_processors"] != request.qualification.logical_cpu_count
        or runtime_controls["windows_build"] != request.qualification.windows_build
        or runtime_controls["wsl_version"] != request.qualification.wsl_version
        or runtime_controls["wsl_kernel"] != request.qualification.wsl_kernel
        or runtime_controls["wsl_memory_limit_b"] != request.qualification.ram_bytes
        or runtime_controls["gpu_driver"] != request.qualification.nvidia_driver
        or runtime_controls["cuda_version"] != request.qualification.cuda_version
    ):
        raise AETStageContractError(
            "provenance runtime hardware does not match the qualified calibration snapshot"
        )
    if (
        provenance.get("cuda_version") is not None
        and provenance["cuda_version"] != runtime_controls["cuda_version"]
    ):
        raise AETStageContractError("provenance CUDA versions disagree")
    _sha256(provenance.get("environment_hash"), "provenance.environment_hash")
    config = provenance.get("config")
    if not isinstance(config, dict):
        raise AETStageContractError("provenance.config must be a mapping")
    config_digest = _sha256(provenance.get("config_sha256"), "provenance.config_sha256")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    if _sha256_bytes(encoded) != config_digest:
        raise AETStageContractError("provenance.config_sha256 does not match config")
    measurement = provenance.get("measurement")
    if not isinstance(measurement, dict) or set(measurement) != {"diagnostic_backend"}:
        raise AETStageContractError("provenance.measurement must be a mapping")
    expected_diagnostic = "nvml" if "gpu" in request.required_measurement_domains else None
    if measurement["diagnostic_backend"] != expected_diagnostic:
        raise AETStageContractError(
            "provenance measurement diagnostic backend disagrees with stage scope"
        )
    measurement_owned = {
        "primary_backend",
        "pue",
        "fallback",
        "preflight_sha256",
        "calibration_sha256",
        "gpu_index",
        "gpu_device_id_sha256",
        "training_dataset_manifest_sha256",
    }
    measurement_collision = sorted(measurement_owned.intersection(measurement))
    if measurement_collision:
        raise AETStageContractError(
            "executor may not predeclare runner-owned provenance measurement fields: "
            + ", ".join(measurement_collision)
        )
    return provenance


def _validate_validation(
    value: Mapping[str, Any],
    *,
    request: StageExecutionRequest,
) -> dict[str, Any]:
    validation = _json_object(value, "validation")
    _close_keys(validation, _VALIDATION_INPUT_KEYS, "validation")
    if validation["schema_version"] != VALIDATION_SCHEMA:
        raise AETStageContractError("validation schema is invalid")
    if (
        validation["status"] != "passed"
        or validation["complete"] is not True
        or validation["independent"] is not True
    ):
        raise AETStageContractError("validation must be independently passed")
    _nonempty_string(validation["validator"], "validation.validator")
    checks = validation["checks"]
    if not isinstance(checks, list) or not checks:
        raise AETStageContractError("validation.checks must be a non-empty list")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise AETStageContractError(f"validation.checks[{index}] must be a mapping")
        _close_keys(check, _VALIDATION_CHECK_KEYS, f"validation.checks[{index}]")
        _nonempty_string(check["name"], f"validation.checks[{index}].name")
        if check["passed"] is not True:
            raise AETStageContractError(f"validation.checks[{index}] did not pass")
    if not isinstance(validation["metrics"], dict):
        raise AETStageContractError("validation.metrics must be a mapping")
    expected = _integer(validation["expected_records"], "validation.expected_records", minimum=1)
    validated = _integer(validation["validated_records"], "validation.validated_records", minimum=0)
    failures = _integer(validation["failure_count"], "validation.failure_count")
    exclusions = _integer(validation["excluded_count"], "validation.excluded_count")
    if validated != expected or failures != 0 or exclusions != 0:
        raise AETStageContractError("validation contains partial, failed, or excluded records")
    gap = validation["gap_to_reference_pct"]
    quality = validation["quality_feasible_confirmatory"]
    if request.run.stage == "training":
        if gap is not None or quality is not None:
            raise AETStageContractError("training validation quality fields must be null")
    else:
        _finite_number(gap, "validation.gap_to_reference_pct")
        if quality is not None:
            raise AETStageContractError(
                "instrumentation-smoke validation may not assert confirmatory quality"
            )
    return validation


def _validate_instance_manifest(
    value: Mapping[str, Any],
    *,
    request: StageExecutionRequest,
) -> dict[str, Any]:
    manifest = _json_object(value, "instance_manifest")
    training = request.run.stage == "training"
    _close_keys(
        manifest,
        _TRAINING_DATASET_KEYS if training else _INSTANCE_KEYS,
        "instance_manifest",
    )
    if manifest["schema_version"] != (
        TRAINING_DATASET_MANIFEST_SCHEMA if training else INSTANCE_MANIFEST_SCHEMA
    ):
        raise AETStageContractError("stage dataset manifest schema is invalid")
    if manifest["status"] != "complete" or manifest["complete"] is not True:
        raise AETStageContractError("stage dataset manifest is incomplete")
    if manifest["split"] != ("training" if training else "confirmatory"):
        raise AETStageContractError("stage dataset manifest has wrong split")
    if manifest["problem"] != request.run.problem or manifest["size"] != request.run.size:
        raise AETStageContractError("instance manifest problem or size mismatch")
    item_count = _integer(manifest["item_count"], "instance_manifest.item_count", minimum=1)
    if training:
        _nonempty_string(manifest["dataset_id"], "instance_manifest.dataset_id")
        generation = manifest["generation"]
        if not isinstance(generation, dict) or set(generation) != {
            "method",
            "seed",
            "config_sha256",
        }:
            raise AETStageContractError("training dataset generation has an invalid schema")
        _nonempty_string(generation["method"], "training_dataset.generation.method")
        _integer(generation["seed"], "training_dataset.generation.seed")
        _sha256(generation["config_sha256"], "training_dataset.generation.config_sha256")
        shards = manifest["shards"]
        if not isinstance(shards, list) or not shards:
            raise AETStageContractError("training_dataset.shards must be a non-empty list")
        total = 0
        paths: set[str] = set()
        for index, shard in enumerate(shards):
            if not isinstance(shard, dict) or set(shard) != {"path", "sha256", "item_count"}:
                raise AETStageContractError(
                    f"training_dataset.shards[{index}] has an invalid schema"
                )
            relative = _safe_stage_relative(shard["path"], f"training_dataset.shards[{index}].path")
            if relative.as_posix() in paths:
                raise AETStageContractError("training_dataset contains duplicate shard paths")
            paths.add(relative.as_posix())
            _sha256(shard["sha256"], f"training_dataset.shards[{index}].sha256")
            total += _integer(
                shard["item_count"],
                f"training_dataset.shards[{index}].item_count",
                minimum=1,
            )
        if total != item_count:
            raise AETStageContractError("training dataset shard counts do not sum to item_count")
        return manifest
    _nonempty_string(manifest["manifest_id"], "instance_manifest.manifest_id")
    entries = manifest["entries"]
    if not isinstance(entries, list) or len(entries) != item_count:
        raise AETStageContractError("instance_manifest.entries must match item_count")
    identifiers: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"instance_id", "sha256"}:
            raise AETStageContractError(f"instance_manifest.entries[{index}] has an invalid schema")
        identifier = _nonempty_string(
            entry["instance_id"], f"instance_manifest.entries[{index}].instance_id"
        )
        if identifier in identifiers:
            raise AETStageContractError("instance_manifest contains duplicate instance IDs")
        identifiers.add(identifier)
        _sha256(entry["sha256"], f"instance_manifest.entries[{index}].sha256")
    return manifest


def _validate_workload_audit(
    value: Mapping[str, Any],
    *,
    request: StageExecutionRequest,
    energy: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _json_object(value, "workload_audit")
    _close_keys(audit, _WORKLOAD_AUDIT_KEYS, "workload_audit")
    if audit["schema_version"] != "aet-workload-audit/v1":
        raise AETStageContractError("workload audit schema is invalid")
    if (
        audit["run_id"] != request.execution_id
        or audit["stage"] != request.run.stage
        or audit["repetition_index"] != request.repetition_index
    ):
        raise AETStageContractError("workload audit run identity is invalid")
    items_per_cycle = _integer(
        audit["items_per_cycle"], "workload_audit.items_per_cycle", minimum=1
    )
    cycle_count = _integer(
        audit["workload_cycle_count"],
        "workload_audit.workload_cycle_count",
        minimum=1,
    )
    items_processed = _integer(
        audit["items_processed"], "workload_audit.items_processed", minimum=1
    )
    if (
        items_per_cycle != request.run.items_per_block
        or items_processed != energy["items_processed"]
        or items_processed != items_per_cycle * cycle_count
    ):
        raise AETStageContractError("workload audit cycle counters are inconsistent")
    duration = _finite_number(
        audit["measurement_duration_s"],
        "workload_audit.measurement_duration_s",
        positive=True,
    )
    if not math.isclose(
        duration,
        float(energy["duration_s"]),
        abs_tol=request.qualification.maximum_timestamp_alignment_error_s,
    ):
        raise AETStageContractError("workload audit duration disagrees with energy")
    if (
        _integer(audit["idle_gap_count"], "workload_audit.idle_gap_count") != 0
        or _finite_number(audit["idle_gap_total_s"], "workload_audit.idle_gap_total_s") != 0.0
    ):
        raise AETStageContractError("workload cycles must run without idle gaps")
    return audit


def _validated_stage_artifacts(
    artifacts: StageArtifacts,
    *,
    request: StageExecutionRequest,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    bytes,
    bytes,
    bytes | None,
]:
    if not isinstance(artifacts, StageArtifacts):
        raise AETStageContractError("stage executor must return StageArtifacts")
    energy = _validate_energy(artifacts.energy, request=request)
    provenance = _validate_provenance(artifacts.provenance, request=request)
    validation = _validate_validation(artifacts.validation, request=request)
    instance = _validate_instance_manifest(artifacts.instance_manifest, request=request)
    audit = _validate_workload_audit(
        artifacts.workload_audit,
        request=request,
        energy=energy,
    )
    wall_meter = energy["extra"]["wall_meter"]
    wall_started = _aware_datetime(wall_meter["started_at_utc"], "wall_meter.started_at_utc")
    wall_ended = _aware_datetime(wall_meter["ended_at_utc"], "wall_meter.ended_at_utc")
    provenance_started = _aware_datetime(
        provenance["timestamp_start"], "provenance.timestamp_start"
    )
    provenance_ended = _aware_datetime(provenance["timestamp_end"], "provenance.timestamp_end")
    tolerance = request.qualification.maximum_timestamp_alignment_error_s
    if (
        abs((provenance_started - wall_started).total_seconds()) > tolerance
        or abs((provenance_ended - wall_ended).total_seconds()) > tolerance
    ):
        raise AETStageContractError(
            "provenance timestamps do not align with the wall-meter measurement block"
        )
    if "gpu" in request.required_measurement_domains:
        nvml = energy["extra"]["diagnostics"]["nvml"]
        nvml_started = _aware_datetime(nvml["started_at_utc"], "nvml.started_at_utc")
        nvml_ended = _aware_datetime(nvml["ended_at_utc"], "nvml.ended_at_utc")
        if (
            abs((nvml_started - wall_started).total_seconds()) > tolerance
            or abs((nvml_ended - wall_ended).total_seconds()) > tolerance
        ):
            raise AETStageContractError(
                "NVML timestamps do not align with the wall-meter measurement block"
            )
    if not isinstance(artifacts.log, str) or not artifacts.log.strip():
        raise AETStageContractError("stage log must be a non-empty string")
    audit_line = "AET_WORKLOAD_AUDIT " + json.dumps(audit, sort_keys=True, separators=(",", ":"))
    log_bytes = (artifacts.log.rstrip("\n") + "\n" + audit_line + "\n").encode()
    log_sha256 = _sha256_bytes(log_bytes)
    for check in validation["checks"]:
        check["evidence_sha256"] = log_sha256
    if not isinstance(artifacts.wall_trace, bytes) or not artifacts.wall_trace:
        raise AETStageContractError("stage must return a non-empty raw wall-meter trace")
    expected_wall_hash = energy["extra"]["wall_meter"]["trace_sha256"]
    if _sha256_bytes(artifacts.wall_trace) != expected_wall_hash:
        raise AETStageContractError("raw wall-meter trace hash does not match energy metadata")
    gpu_scoped = "gpu" in request.required_measurement_domains
    if gpu_scoped:
        if not isinstance(artifacts.nvml_trace, bytes) or not artifacts.nvml_trace:
            raise AETStageContractError("GPU-scoped stage must return a non-empty raw NVML trace")
        expected_nvml_hash = energy["extra"]["diagnostics"]["nvml"]["trace_sha256"]
        if _sha256_bytes(artifacts.nvml_trace) != expected_nvml_hash:
            raise AETStageContractError("raw NVML trace hash does not match energy metadata")
    elif artifacts.nvml_trace is not None:
        raise AETStageContractError("stage without GPU scope may not return an NVML trace")
    return (
        energy,
        provenance,
        validation,
        instance,
        log_bytes,
        artifacts.wall_trace,
        artifacts.nvml_trace,
    )


def _with_provenance_links(
    provenance: dict[str, Any],
    *,
    request: StageExecutionRequest,
    training_dataset_manifest_sha256: str | None,
    items_processed: int,
) -> dict[str, Any]:
    linked = json.loads(json.dumps(provenance, allow_nan=False))
    linked.update(
        {
            "schema_version": "aet-provenance/v1",
            "run_id": request.execution_id,
            "host_id": request.qualification.host_id,
            "status": "completed",
            "failure_reason": None,
            "git_sha": request.qualification.git_sha,
            "git_dirty": False,
            "dirty_patch_sha256": None,
            "execution_layer": request.qualification.execution_layer,
        }
    )
    measurement = linked["measurement"]
    measurement.update(
        {
            "primary_backend": "wall_meter",
            "pue": 1.0,
            "fallback": False,
            "preflight_sha256": request.qualification.preflight_sha256,
            "calibration_sha256": request.qualification.calibration_sha256,
            "gpu_index": request.qualification.gpu_index,
            "gpu_device_id_sha256": request.qualification.gpu_device_id_sha256,
            "training_dataset_manifest_sha256": training_dataset_manifest_sha256,
        }
    )
    cycle_count = items_processed // request.run.items_per_block
    linked["config"]["measurement_workload"] = {
        "items_per_cycle": request.run.items_per_block,
        "workload_cycle_count": cycle_count,
        "items_processed": items_processed,
        "repetition_index": request.repetition_index,
    }
    linked["config_sha256"] = _sha256_bytes(
        json.dumps(
            linked["config"],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    return linked


def _with_validation_links(
    validation: dict[str, Any],
    *,
    request: StageExecutionRequest,
    provenance_sha256: str,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    linked = json.loads(json.dumps(validation, allow_nan=False))
    training = request.run.stage == "training"
    linked.update(
        {
            "run_id": request.execution_id,
            "role": request.run.stage,
            "host_id": request.qualification.host_id,
            "execution_layer": request.qualification.execution_layer,
            "git_sha": request.qualification.git_sha,
            "gpu_index": request.qualification.gpu_index,
            "gpu_device_id_sha256": request.qualification.gpu_device_id_sha256,
            "preflight_sha256": request.qualification.preflight_sha256,
            "calibration_sha256": request.qualification.calibration_sha256,
            "provenance_sha256": provenance_sha256,
            "instance_manifest_sha256": None if training else dataset_manifest_sha256,
            "training_dataset_manifest_sha256": (dataset_manifest_sha256 if training else None),
        }
    )
    return linked


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically rename a directory while refusing every existing target."""
    if sys.platform == "win32":
        os.rename(source, target)
        return
    libc = CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    function: Any
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise AETBundleError("renameat2 is unavailable; refusing unsafe bundle promotion")
        function.argtypes = [c_int, c_char_p, c_int, c_char_p, c_uint]
        function.restype = c_int
        result = function(-100, source_bytes, -100, target_bytes, 1)
    elif sys.platform == "darwin":
        function = getattr(libc, "renamex_np", None)
        if function is None:
            raise AETBundleError("renamex_np is unavailable; refusing unsafe bundle promotion")
        function.argtypes = [c_char_p, c_char_p, c_uint]
        function.restype = c_int
        result = function(source_bytes, target_bytes, 0x00000004)
    else:
        raise AETBundleError(
            f"atomic no-replace promotion is unsupported on platform {sys.platform!r}"
        )
    if result == 0:
        return
    error_number = get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _write_artifact(staging: Path, relative: PurePosixPath, payload: bytes) -> dict[str, Any]:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AETBundleError(f"refusing unsafe bundle artifact path: {relative}")
    path = staging.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AETBundleError(f"refusing duplicate or symlinked artifact path: {relative}")
    _atomic_write(path, payload)
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _with_energy_links(
    energy: dict[str, Any],
    *,
    request: StageExecutionRequest,
    provenance_sha256: str,
    validation_sha256: str,
    dataset_manifest_sha256: str,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    linked = json.loads(json.dumps(energy, allow_nan=False))
    extra = linked["extra"]
    extra.update(
        {
            "measurement_qualified_instrumentation": True,
            "purpose": SMOKE_PURPOSE,
            "scientific_use": False,
            "confirmatory_eligible": False,
            "fallback": False,
            "host_id": request.qualification.host_id,
            "execution_layer": request.qualification.execution_layer,
            "git_sha": request.qualification.git_sha,
            "gpu_index": request.qualification.gpu_index,
            "gpu_device_id_sha256": request.qualification.gpu_device_id_sha256,
            "preflight_sha256": request.qualification.preflight_sha256,
            "calibration_sha256": request.qualification.calibration_sha256,
            "provenance_sha256": provenance_sha256,
            "artifact_sha256": validation_sha256,
            "run_id": request.execution_id,
            "role": request.run.stage,
            "stage": request.run.stage,
            "repetition_index": request.repetition_index,
            "problem": request.run.problem,
            "items_per_cycle": request.run.items_per_block,
            "workload_cycle_count": (int(linked["items_processed"]) // request.run.items_per_block),
        }
    )
    if request.run.stage == "training":
        extra["training_dataset_manifest_sha256"] = dataset_manifest_sha256
        extra["size"] = request.run.size
    else:
        extra["instance_manifest_sha256"] = dataset_manifest_sha256
        extra["gap_to_reference_pct"] = validation["gap_to_reference_pct"]
        if request.run.stage == "inference":
            extra["size_key"] = request.run.size
        else:
            extra["size"] = request.run.size
    return linked


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise AETBundleError(f"bundle contains forbidden symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _verify_entries(root: Path, entries: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for entry in entries:
        relative = entry["path"]
        if relative in seen:
            raise AETBundleError(f"duplicate bundle manifest path: {relative}")
        seen.add(relative)
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise AETBundleError(f"bundle artifact is missing or unsafe: {relative}")
        if path.stat().st_size != entry["size_bytes"] or _sha256_file(path) != entry["sha256"]:
            raise AETBundleError(f"bundle artifact checksum mismatch: {relative}")


def _verify_report_entries(
    bundle_entries: list[dict[str, Any]],
    report_entries: list[dict[str, str]],
) -> None:
    bundle_by_path = {entry["path"]: entry for entry in bundle_entries}
    seen_paths: set[str] = set()
    hashes_by_role: dict[str, set[str]] = {}
    for entry in report_entries:
        role = entry["role"]
        expected_keys = {"role", "path", "sha256"}
        if role in _REPORT_RUN_BOUND_ROLES:
            expected_keys.add("run_id")
        if set(entry) != expected_keys:
            raise AETBundleError("report input entry has an invalid closed schema")
        path = entry["path"]
        if path in seen_paths:
            raise AETBundleError(f"duplicate report input path: {path}")
        seen_paths.add(path)
        bundled = bundle_by_path.get(path)
        if bundled is None or bundled["sha256"] != entry["sha256"]:
            raise AETBundleError(f"report input is not bound to the bundle manifest: {path}")
        role_hashes = hashes_by_role.setdefault(role, set())
        if entry["sha256"] in role_hashes:
            raise AETBundleError(f"duplicate report input content for role {role}: {path}")
        role_hashes.add(entry["sha256"])
        if role in _REPORT_RUN_BOUND_ROLES:
            _nonempty_string(entry["run_id"], f"report input {role} run_id")


def _revalidate_before_promotion(
    plan: BundlePlan,
    qualification: BundleQualification,
) -> None:
    if _sha256_file(plan.recipe_path) != plan.recipe_sha256:
        raise AETBundleQualificationError("recipe changed during stage execution")
    current = _qualification_report(plan.recipe)
    if (
        current.get("ready_to_execute") is not True
        or current.get("git_commit_matches") is not True
        or current.get("current_git_commit") != qualification.git_sha
    ):
        raise AETBundleQualificationError(
            "recipe, preflight, calibration, or Git qualification changed during execution"
        )
    if (
        _sha256_file(qualification.preflight_path) != qualification.preflight_sha256
        or _sha256_file(qualification.calibration_path) != qualification.calibration_sha256
    ):
        raise AETBundleQualificationError("preflight or calibration changed during stage execution")
    for evidence in qualification.calibration_evidence:
        if _sha256_file(evidence.source) != evidence.sha256:
            raise AETBundleQualificationError(
                f"qualified {evidence.role} changed during stage execution"
            )


def execute_aet_smoke_bundle(
    recipe_path: str | Path,
    *,
    bundle_id: str,
    executors: Mapping[str, StageExecutor],
    workspace_root: str | Path | None = None,
) -> BundleResult:
    """Run injected stages and atomically promote a verified smoke-only bundle.

    The function has no default scientific executors. An executor failure or
    contract violation deletes the hidden staging directory and never promotes
    a partial output.
    """
    plan = validate_aet_smoke_bundle(
        recipe_path,
        bundle_id=bundle_id,
        workspace_root=workspace_root,
        require_ready=True,
    )
    if set(executors) != _REQUIRED_STAGES:
        missing = sorted(_REQUIRED_STAGES - set(executors))
        extra = sorted(set(executors) - _REQUIRED_STAGES)
        raise AETBundleError(f"executor stage set mismatch; missing={missing}, extra={extra}")
    if any(not callable(executor) for executor in executors.values()):
        raise AETBundleError("every stage executor must be callable")
    qualification = _load_qualification(plan)
    _revalidate_before_promotion(plan, qualification)
    if plan.target.exists() or plan.target.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable AET bundle: {plan.target}")

    output_root = plan.target.parent
    _assert_no_symlink_ancestors(Path.cwd().resolve(), output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_ancestors(Path.cwd().resolve(), output_root)
    staging = output_root / f".{bundle_id}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o700)
    bundle_entries: list[dict[str, Any]] = []
    report_entries: list[dict[str, str]] = []
    stage_bindings: list[dict[str, Any]] = []
    shared_instance_payload: bytes | None = None
    shared_instance_entry: dict[str, Any] | None = None
    deployment_service_boundary: dict[str, Any] | None = None
    execution_ids: set[str] = set()
    try:
        recipe_bytes = plan.recipe_path.read_bytes()
        if _sha256_bytes(recipe_bytes) != plan.recipe_sha256:
            raise AETBundleQualificationError("recipe changed after validation")
        bundle_entries.append(_write_artifact(staging, PurePosixPath("recipe.yaml"), recipe_bytes))
        for role, source, relative in (
            (
                "preflight",
                qualification.preflight_path,
                PurePosixPath("qualification/preflight.json"),
            ),
            (
                "calibration",
                qualification.calibration_path,
                PurePosixPath("qualification/calibration.json"),
            ),
        ):
            payload = source.read_bytes()
            entry = _write_artifact(staging, relative, payload)
            expected = (
                qualification.preflight_sha256
                if role == "preflight"
                else qualification.calibration_sha256
            )
            if entry["sha256"] != expected:
                raise AETBundleQualificationError(f"qualified {role} changed during bundle write")
            bundle_entries.append(entry)
            report_entries.append({"role": role, "path": entry["path"], "sha256": entry["sha256"]})

        for evidence in qualification.calibration_evidence:
            payload = evidence.source.read_bytes()
            entry = _write_artifact(staging, evidence.relative_path, payload)
            if entry["sha256"] != evidence.sha256:
                raise AETBundleQualificationError(
                    f"qualified {evidence.role} changed during bundle write"
                )
            bundle_entries.append(entry)
            report_entries.append(
                {
                    "role": evidence.role,
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                }
            )

        for run in plan.recipe.runs:
            executor = executors[run.stage]
            for repetition_index in range(run.repetitions):
                execution_id = (
                    run.run_id
                    if run.repetitions == 1
                    else f"{run.run_id}-repeat-{repetition_index:03d}"
                )
                if execution_id in execution_ids:
                    raise AETBundleError(f"duplicate derived execution_id: {execution_id}")
                execution_ids.add(execution_id)
                request = StageExecutionRequest(
                    recipe_name=plan.recipe.name,
                    run=run,
                    execution_id=execution_id,
                    repetition_index=repetition_index,
                    minimum_block_duration_s=float(
                        plan.recipe.measurement["minimum_block_duration_s"]
                    ),
                    required_measurement_domains=tuple(plan.recipe.measurement["scope"][run.stage]),
                    qualification=qualification,
                )
                returned = executor(request)
                (
                    energy,
                    provenance,
                    validation,
                    instance,
                    log_bytes,
                    wall_trace_bytes,
                    nvml_trace_bytes,
                ) = _validated_stage_artifacts(returned, request=request)
                if run.stage in {"inference", "baseline"}:
                    service_boundary = provenance["runtime_controls"]["service_boundary"]
                    if deployment_service_boundary is None:
                        deployment_service_boundary = json.loads(
                            json.dumps(service_boundary, allow_nan=False)
                        )
                    elif service_boundary != deployment_service_boundary:
                        raise AETStageContractError(
                            "inference and baseline service boundaries must be identical"
                        )
                base = PurePosixPath(run.output_subdir)
                if run.repetitions > 1:
                    base /= f"repeat-{repetition_index:03d}"
                instance_payload = _json_bytes(instance)
                if run.stage == "training":
                    dataset_entry = _write_artifact(
                        staging,
                        base / "training_dataset_manifest.json",
                        instance_payload,
                    )
                    bundle_entries.append(dataset_entry)
                    report_entries.append(
                        {
                            "role": "training_dataset_manifest",
                            "path": dataset_entry["path"],
                            "sha256": dataset_entry["sha256"],
                        }
                    )
                elif shared_instance_entry is None:
                    shared_instance_payload = instance_payload
                    shared_instance_entry = _write_artifact(
                        staging,
                        PurePosixPath("instances/instance_manifest.json"),
                        instance_payload,
                    )
                    dataset_entry = shared_instance_entry
                    bundle_entries.append(dataset_entry)
                    report_entries.append(
                        {
                            "role": "instance_manifest",
                            "path": dataset_entry["path"],
                            "sha256": dataset_entry["sha256"],
                        }
                    )
                else:
                    if instance_payload != shared_instance_payload:
                        raise AETStageContractError(
                            "inference and baseline must return one byte-identical "
                            "instance manifest"
                        )
                    dataset_entry = shared_instance_entry
                if validation["expected_records"] != instance["item_count"]:
                    raise AETStageContractError(
                        "validation.expected_records must match its dataset manifest"
                    )
                linked_provenance = _with_provenance_links(
                    provenance,
                    request=request,
                    training_dataset_manifest_sha256=(
                        dataset_entry["sha256"] if run.stage == "training" else None
                    ),
                    items_processed=int(energy["items_processed"]),
                )
                provenance_entry = _write_artifact(
                    staging,
                    base / "provenance.json",
                    _json_bytes(linked_provenance),
                )
                linked_validation = _with_validation_links(
                    validation,
                    request=request,
                    provenance_sha256=provenance_entry["sha256"],
                    dataset_manifest_sha256=dataset_entry["sha256"],
                )
                validation_entry = _write_artifact(
                    staging,
                    base / "validation.json",
                    _json_bytes(linked_validation),
                )
                linked_energy = _with_energy_links(
                    energy,
                    request=request,
                    provenance_sha256=provenance_entry["sha256"],
                    validation_sha256=validation_entry["sha256"],
                    dataset_manifest_sha256=dataset_entry["sha256"],
                    validation=linked_validation,
                )
                energy_entry = _write_artifact(
                    staging,
                    base / _ENERGY_FILENAMES[run.stage],
                    _json_bytes(linked_energy),
                )
                log_entry = _write_artifact(staging, base / "run.log", log_bytes)
                wall_trace_entry = _write_artifact(
                    staging,
                    base / "wall-meter-trace.jsonl",
                    wall_trace_bytes,
                )
                if (
                    wall_trace_entry["sha256"]
                    != linked_energy["extra"]["wall_meter"]["trace_sha256"]
                ):
                    raise AETStageContractError(
                        "written wall-meter trace hash does not match energy metadata"
                    )
                nvml_trace_entry: dict[str, Any] | None = None
                if nvml_trace_bytes is not None:
                    nvml_trace_entry = _write_artifact(
                        staging,
                        base / "nvml-trace.jsonl",
                        nvml_trace_bytes,
                    )
                    if (
                        nvml_trace_entry["sha256"]
                        != linked_energy["extra"]["diagnostics"]["nvml"]["trace_sha256"]
                    ):
                        raise AETStageContractError(
                            "written NVML trace hash does not match energy metadata"
                        )
                bundle_entries.extend(
                    [
                        provenance_entry,
                        validation_entry,
                        energy_entry,
                        log_entry,
                        wall_trace_entry,
                    ]
                )
                if nvml_trace_entry is not None:
                    bundle_entries.append(nvml_trace_entry)
                for role, entry in (
                    ("provenance", provenance_entry),
                    ("validation", validation_entry),
                    (_ENERGY_ROLES[run.stage], energy_entry),
                ):
                    report_entries.append(
                        {"role": role, "path": entry["path"], "sha256": entry["sha256"]}
                    )
                report_entries.append(
                    {
                        "role": "wall_trace",
                        "run_id": request.execution_id,
                        "path": wall_trace_entry["path"],
                        "sha256": wall_trace_entry["sha256"],
                    }
                )
                report_entries.append(
                    {
                        "role": "validation_evidence",
                        "run_id": request.execution_id,
                        "path": log_entry["path"],
                        "sha256": log_entry["sha256"],
                    }
                )
                if nvml_trace_entry is not None:
                    report_entries.append(
                        {
                            "role": "nvml_trace",
                            "run_id": request.execution_id,
                            "path": nvml_trace_entry["path"],
                            "sha256": nvml_trace_entry["sha256"],
                        }
                    )
                stage_bindings.append(
                    {
                        "run_id": request.execution_id,
                        "stage": run.stage,
                        "dataset_manifest": dataset_entry["path"],
                        "energy": energy_entry["path"],
                        "provenance": provenance_entry["path"],
                        "validation": validation_entry["path"],
                        "log": log_entry["path"],
                        "validation_evidence": log_entry["path"],
                        "wall_trace": wall_trace_entry["path"],
                        "nvml_trace": (
                            nvml_trace_entry["path"] if nvml_trace_entry is not None else None
                        ),
                    }
                )

        if shared_instance_entry is None:
            raise AETStageContractError(
                "inference and baseline did not produce a shared instance manifest"
            )

        _revalidate_before_promotion(plan, qualification)
        _verify_report_entries(bundle_entries, report_entries)

        report_manifest = {
            "schema_version": REPORT_INPUT_MANIFEST_SCHEMA,
            "purpose": SMOKE_PURPOSE,
            "confirmatory_eligible": False,
            "files": sorted(report_entries, key=lambda entry: (entry["role"], entry["path"])),
        }
        report_entry = _write_artifact(
            staging,
            PurePosixPath("report-input-manifest.json"),
            _json_bytes(report_manifest),
        )
        bundle_entries.append(report_entry)
        _verify_entries(staging, bundle_entries)

        manifest = {
            "schema_version": BUNDLE_MANIFEST_SCHEMA,
            "bundle_id": bundle_id,
            "recipe_name": plan.recipe.name,
            "purpose": SMOKE_PURPOSE,
            "scientific_use": False,
            "confirmatory_eligible": False,
            "git_sha": qualification.git_sha,
            "host_id": qualification.host_id,
            "immutable": True,
            "manifest_excludes_itself": True,
            "stage_bindings": stage_bindings,
            "files": sorted(bundle_entries, key=lambda entry: entry["path"]),
        }
        manifest_bytes = _json_bytes(manifest)
        _write_artifact(staging, PurePosixPath("bundle-manifest.json"), manifest_bytes)
        _verify_entries(staging, bundle_entries)
        _freeze_tree(staging)
        if plan.target.exists() or plan.target.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite immutable AET bundle created during execution: "
                f"{plan.target}"
            )
        _rename_no_replace(staging, plan.target)
        return BundleResult(
            bundle_id=bundle_id,
            path=plan.target,
            bundle_manifest_sha256=_sha256_bytes(manifest_bytes),
            report_input_manifest=plan.target / "report-input-manifest.json",
        )
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                with suppress(OSError):
                    path.chmod(0o700 if path.is_dir() else 0o600)
            staging.chmod(0o700)
            shutil.rmtree(staging)
        raise
