"""Run the closed native-Windows CVRP50 deployment-energy pilot.

One process executes the ten preregistered blocks after one operator
attestation. Each complete block is promoted atomically, so a later launcher
run can resume the strict prefix after interruption without repeating it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import mean, median, stdev
from typing import Any

from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments import software_recipe as software_recipe
from neuro_co.aet.experiments import software_smoke_runner as software
from neuro_co.aet.experiments.deployment_energy_recipe import (
    CLASSIFICATION,
    DeploymentEnergyRecipe,
    EnergyBlock,
    expected_blocks,
    load_recipe,
    qualify_for_execution,
)
from neuro_co.aet.experiments.quality_recipe import EvaluationMode, load_aet_quality_recipe

RUN_STATE_SCHEMA = "aet-deployment-energy-run-state/v1"
CORPUS_SCHEMA = "aet-deployment-energy-corpus/v1"
BLOCK_SCHEMA = "aet-deployment-energy-block/v1"
SUMMARY_SCHEMA = "aet-deployment-energy-paired-summary/v1"
MANIFEST_SCHEMA = "aet-deployment-energy-manifest/v1"
COMPLETE_STATUS = "complete_software_exploratory"
INCOMPLETE_STATUS = "incomplete_software_exploratory"

ATTESTED_ENV = "AET_DEPLOYMENT_EXCLUSIVE_ATTESTED"
ATTESTED_AT_ENV = "AET_DEPLOYMENT_EXCLUSIVE_ATTESTED_AT"
ATTESTATION_SESSION_ENV = "AET_DEPLOYMENT_EXCLUSIVE_SESSION_ID"

_PARTIAL_ARTIFACT = re.compile(r"^\.[^/]+\.partial-[0-9a-f]{32}(?:\.npz)?$")
_PRESERVED_PREFLIGHT = re.compile(r"^preflight-([0-9a-f]{64})\.json$")


class DeploymentEnergyPilotError(RuntimeError):
    """Raised when the closed deployment-energy pilot cannot continue safely."""


class DeploymentEnergyQualificationError(DeploymentEnergyPilotError):
    """Raised before measured work when source or counter qualification fails."""


@dataclass(frozen=True, slots=True)
class DeploymentEnergyPilotResult:
    path: Path
    status: str
    completed_blocks: int
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class Corpus:
    coords: Any
    demands: Any
    capacity: float
    content_sha256: str
    file_sha256: str
    path: Path


def classification() -> dict[str, Any]:
    """Return a fresh copy of the permanent exploratory classification."""

    return dict(CLASSIFICATION)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentEnergyPilotError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, dict):
        raise DeploymentEnergyPilotError(f"JSON artifact is not an object: {path}")
    return value


def _relative(root: Path, value: str) -> Path:
    return root.joinpath(*PurePosixPath(value).parts)


def _safe_output(root: Path, value: str) -> Path:
    repository = root.resolve(strict=True)
    if repository != Path.cwd().resolve(strict=True):
        raise DeploymentEnergyPilotError("workspace_root must be the current Git checkout")
    output = _relative(repository, value)
    cursor = repository
    for component in output.relative_to(repository).parts:
        cursor /= component
        if (cursor.exists() or cursor.is_symlink()) and cursor.is_symlink():
            raise DeploymentEnergyPilotError(f"refusing symlinked output component: {cursor}")
    resolved = output.resolve(strict=False)
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise DeploymentEnergyPilotError("output root resolves outside the repository") from exc
    return resolved


def _exclusive_attestation(recipe: DeploymentEnergyRecipe) -> dict[str, Any]:
    if os.environ.get(ATTESTED_ENV) != "1":
        raise DeploymentEnergyQualificationError(
            "exclusive-use attestation is absent; use the Windows launcher"
        )
    raw_at = os.environ.get(ATTESTED_AT_ENV)
    session_id = os.environ.get(ATTESTATION_SESSION_ENV)
    if not raw_at or not session_id:
        raise DeploymentEnergyQualificationError("exclusive-use attestation is incomplete")
    try:
        attested_at = datetime.fromisoformat(raw_at.replace("Z", "+00:00"))
        uuid.UUID(session_id)
    except (ValueError, TypeError) as exc:
        raise DeploymentEnergyQualificationError("exclusive-use attestation is invalid") from exc
    if attested_at.tzinfo is None:
        raise DeploymentEnergyQualificationError("exclusive-use attestation needs a UTC offset")
    attested_at = attested_at.astimezone(UTC)
    age_s = (datetime.now(UTC) - attested_at).total_seconds()
    campaign_limit_s = min(
        recipe.campaign_attestation_max_age_s,
        recipe.maximum_campaign_walltime_s,
    )
    if age_s < -60 or age_s > campaign_limit_s:
        raise DeploymentEnergyQualificationError(
            "exclusive-use campaign expired; restart the Windows launcher"
        )
    return {
        "session_id": session_id,
        "attested_at": attested_at.isoformat(),
        "age_s_at_process_start": age_s,
        "statement": (
            "all other CPU and GPU workloads are stopped and exclusivity will be maintained "
            "until this launcher completes"
        ),
        "operator_supplied": True,
        "operator_attestation_is_authoritative": True,
        "gpu_process_lists_are_diagnostic_only": True,
        "software_process_gate_enforced": False,
    }


def _assert_attestation_active(recipe: DeploymentEnergyRecipe, attestation: dict[str, Any]) -> None:
    attested_at = datetime.fromisoformat(str(attestation["attested_at"]))
    age_s = (datetime.now(UTC) - attested_at).total_seconds()
    campaign_limit_s = min(
        recipe.campaign_attestation_max_age_s,
        recipe.maximum_campaign_walltime_s,
    )
    if age_s > campaign_limit_s:
        raise DeploymentEnergyQualificationError(
            "exclusive-use campaign exceeded its fixed validity window"
        )


def _validate_checksum_inventory(source_root: Path) -> dict[str, Any]:
    checksum_path = source_root / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DeploymentEnergyQualificationError(
            f"source checksum inventory is unreadable: {source_root}"
        ) from exc
    seen: set[str] = set()
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise DeploymentEnergyQualificationError("source checksum inventory is malformed")
        digest, relative = line[:64], line[66:]
        if relative in seen or not relative or "\\" in relative:
            raise DeploymentEnergyQualificationError("source checksum path is unsafe or repeated")
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise DeploymentEnergyQualificationError("source checksum path is unsafe")
        candidate = _relative(source_root, relative)
        if candidate.is_symlink() or not candidate.is_file() or _sha256_file(candidate) != digest:
            raise DeploymentEnergyQualificationError(
                f"source checksum verification failed: {relative}"
            )
        seen.add(relative)
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "run-state.json"}
    }
    if actual != seen:
        raise DeploymentEnergyQualificationError(
            "source bundle inventory has missing or unlisted files"
        )
    return {"entry_count": len(seen), "sha256": _sha256_file(checksum_path)}


def _validate_source_semantics(recipe: DeploymentEnergyRecipe, root: Path) -> dict[str, Any]:
    quality_root = _relative(root, recipe.quality_source.root)
    checkpoint_root = _relative(root, recipe.checkpoint_source.root)
    quality_inventory = _validate_checksum_inventory(quality_root)
    checkpoint_inventory = _validate_checksum_inventory(checkpoint_root)

    quality_manifest = _load_json(quality_root / "manifest.json")
    quality_state = _load_json(quality_root / "run-state.json")
    quality_assessment = _load_json(quality_root / "holdout-assessment.json")
    decision = quality_assessment.get("confirmatory_decision")
    if (
        quality_manifest.get("status") != recipe.quality_source.expected_status
        or quality_state.get("status") != recipe.quality_source.expected_status
        or quality_state.get("git_sha") != recipe.quality_source.source_git_sha
        or quality_state.get("source_snapshot", {}).get("sha256")
        != recipe.quality_source.source_snapshot_sha256
        or not isinstance(decision, dict)
        or decision.get("passed") is not True
        or decision.get("rule") != "neural_and_hgs10"
        or decision.get("hgs_budget") != 10
        or quality_manifest.get("dataset", {}).get("content_sha256")
        != "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf"
    ):
        raise DeploymentEnergyQualificationError(
            "the sealed seed2723 quality source no longer proves both fixed policies"
        )

    checkpoint_manifest = _load_json(checkpoint_root / "manifest.json")
    checkpoint_state = _load_json(checkpoint_root / "run-state.json")
    checkpoint_assessment = _load_json(checkpoint_root / "replication-assessment.json")
    if (
        checkpoint_manifest.get("status") != recipe.checkpoint_source.expected_status
        or checkpoint_state.get("status") != recipe.checkpoint_source.expected_status
        or checkpoint_state.get("git_sha") != recipe.checkpoint_source.source_git_sha
        or checkpoint_state.get("source_snapshot", {}).get("sha256")
        != recipe.checkpoint_source.source_snapshot_sha256
        or checkpoint_assessment.get("roles", {}).get("primary", {}).get("mode_id") != "pomo-50x8"
        or checkpoint_assessment.get("roles", {}).get("primary", {}).get("passed") is not True
    ):
        raise DeploymentEnergyQualificationError(
            "the sealed checkpoint source no longer proves the frozen POMO policy"
        )

    return {
        "quality_source": {
            "root": recipe.quality_source.root,
            "status": recipe.quality_source.expected_status,
            "manifest_sha256": _sha256_file(quality_root / "manifest.json"),
            "checksums_sha256": quality_inventory["sha256"],
            "checksum_entries": quality_inventory["entry_count"],
            "git_sha": recipe.quality_source.source_git_sha,
            "source_snapshot_sha256": recipe.quality_source.source_snapshot_sha256,
        },
        "checkpoint_source": {
            "root": recipe.checkpoint_source.root,
            "status": recipe.checkpoint_source.expected_status,
            "manifest_sha256": _sha256_file(checkpoint_root / "manifest.json"),
            "checksums_sha256": checkpoint_inventory["sha256"],
            "checksum_entries": checkpoint_inventory["entry_count"],
            "git_sha": recipe.checkpoint_source.source_git_sha,
            "source_snapshot_sha256": recipe.checkpoint_source.source_snapshot_sha256,
            "checkpoints": [asdict(item) for item in recipe.checkpoints],
        },
    }


def _runtime_identity(recipe: DeploymentEnergyRecipe) -> dict[str, Any]:
    import torch

    try:
        versions = {
            name: importlib.metadata.version(distribution)
            for name, distribution in {
                "torch": "torch",
                "numpy": "numpy",
                "pyvrp": "pyvrp",
                "codecarbon": "codecarbon",
                "pynvml": "nvidia-ml-py",
            }.items()
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise DeploymentEnergyQualificationError("required runtime distribution is absent") from exc
    return {
        "platform": sys.platform,
        "platform_release": platform.release(),
        "python": platform.python_version(),
        "versions": versions,
        "gpu_index": recipe.gpu_index,
        "gpu_name": torch.cuda.get_device_name(recipe.gpu_index),
        "torch_cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
    }


def _configure_runtime(recipe: DeploymentEnergyRecipe) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise DeploymentEnergyQualificationError(f"{name} must be exactly 1")
    import torch

    if not torch.cuda.is_available() or recipe.gpu_index >= torch.cuda.device_count():
        raise DeploymentEnergyQualificationError("the selected CUDA device is unavailable")
    torch.cuda.set_device(recipe.gpu_index)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(False)
    return _runtime_identity(recipe)


def _git_snapshot(root: Path, preflight: Path) -> dict[str, Any]:
    snapshot = software_recipe._current_git_snapshot((preflight,))
    if snapshot.get("available") is not True:
        raise DeploymentEnergyQualificationError("Git source identity is unavailable")
    return snapshot


def _prepare_output(
    recipe_path: Path,
    recipe: DeploymentEnergyRecipe,
    root: Path,
    *,
    resume: bool,
    runtime_identity: dict[str, Any],
    source_receipt: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output = _safe_output(root, recipe.output_root)
    recipe_bytes = recipe_path.read_bytes()
    recipe_sha = _sha256_bytes(recipe_bytes)
    lock_bytes = (root / "uv.lock").read_bytes()
    lock_sha = _sha256_bytes(lock_bytes)
    preflight = _relative(root, recipe.preflight_report)
    git = _git_snapshot(root, preflight)
    source_receipt_bytes = _json_bytes(source_receipt)
    source_receipt_sha256 = _sha256_bytes(source_receipt_bytes)
    state_path = output / "run-state.json"
    if output.exists():
        if not resume:
            raise DeploymentEnergyPilotError(f"output already exists; use --resume: {output}")
        state = _load_json(state_path)
        expected = {
            "schema_version": RUN_STATE_SCHEMA,
            "recipe_sha256": recipe_sha,
            "uv_lock_sha256": lock_sha,
            "git_sha": git.get("sha"),
            "worktree_fingerprint_sha256": git.get("worktree_fingerprint_sha256"),
            "runtime_identity": runtime_identity,
            "source_receipt": source_receipt,
            "source_receipt_sha256": source_receipt_sha256,
            "classification": classification(),
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise DeploymentEnergyPilotError("resume identity differs from initialized pilot")
        if _sha256_file(output / "recipe.yaml") != recipe_sha:
            raise DeploymentEnergyPilotError("frozen recipe changed")
        if _sha256_file(output / "environment" / "uv.lock") != lock_sha:
            raise DeploymentEnergyPilotError("frozen lockfile changed")
        source_receipt_path = output / "source-receipt.json"
        if (
            not source_receipt_path.is_file()
            or source_receipt_path.read_bytes() != source_receipt_bytes
            or _sha256_file(source_receipt_path) != source_receipt_sha256
        ):
            raise DeploymentEnergyPilotError("frozen source receipt changed")
        _validate_preserved_preflights(output, state)
        status = state.get("status")
        if status == INCOMPLETE_STATUS:
            _recover_interrupted_artifacts(recipe, output, state)
        elif status != COMPLETE_STATUS:
            raise DeploymentEnergyPilotError("resume state has an unknown status")
        return output, state

    state = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": INCOMPLETE_STATUS,
        "created_at": datetime.now(UTC).isoformat(),
        "recipe_sha256": recipe_sha,
        "uv_lock_sha256": lock_sha,
        "git_sha": git.get("sha"),
        "worktree_fingerprint_sha256": git.get("worktree_fingerprint_sha256"),
        "runtime_identity": runtime_identity,
        "source_receipt": source_receipt,
        "source_receipt_sha256": source_receipt_sha256,
        "classification": classification(),
        "dataset": None,
        "completed_blocks": [],
        "block_sha256": {},
        "block_attempts": [],
        "current_block": None,
        "sessions": [],
        "orphaned_preflight_evidence": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "environment").mkdir()
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "environment" / "uv.lock").write_bytes(lock_bytes)
        _atomic_write_bytes(staging / "source-receipt.json", source_receipt_bytes)
        _atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _validate_preserved_preflights(output: Path, state: dict[str, Any]) -> None:
    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise DeploymentEnergyPilotError("session history is malformed")
    for session in sessions:
        if not isinstance(session, dict):
            raise DeploymentEnergyPilotError("session history entry is malformed")
        evidence = session.get("preflight_evidence")
        if not isinstance(evidence, dict):
            raise DeploymentEnergyPilotError("session lacks preserved preflight evidence")
        relative = evidence.get("path")
        digest = evidence.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DeploymentEnergyPilotError("session preflight identity is malformed")
        path = _relative(output, relative)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise DeploymentEnergyPilotError("preserved session preflight changed")

    orphaned = state.get("orphaned_preflight_evidence", [])
    if not isinstance(orphaned, list):
        raise DeploymentEnergyPilotError("orphaned preflight history is malformed")
    for evidence in orphaned:
        if (
            not isinstance(evidence, dict)
            or evidence.get("reason") != "interrupted_before_session_state_promotion"
        ):
            raise DeploymentEnergyPilotError("orphaned preflight evidence is malformed")
        relative = evidence.get("path")
        digest = evidence.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DeploymentEnergyPilotError("orphaned preflight identity is malformed")
        path = _relative(output, relative)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise DeploymentEnergyPilotError("orphaned preflight evidence changed")


def _unique_preflight_evidence(state: dict[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for session in state.get("sessions", []):
        evidence = session["preflight_evidence"]
        identity = (str(evidence["path"]), str(evidence["sha256"]))
        if identity not in seen:
            seen.add(identity)
            records.append({"path": identity[0], "sha256": identity[1]})
    for evidence in state.get("orphaned_preflight_evidence", []):
        identity = (str(evidence["path"]), str(evidence["sha256"]))
        if identity not in seen:
            seen.add(identity)
            records.append({"path": identity[0], "sha256": identity[1]})
    records.sort(key=lambda item: (item["path"], item["sha256"]))
    return records


def _preflight_evidence_is_qualified(
    recipe: DeploymentEnergyRecipe,
    state: dict[str, Any],
    path: Path,
) -> bool:
    """Validate stable qualification facts without replaying volatile active probes."""

    report = _load_json(path)

    def mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def positive(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value > 0
        )

    system = mapping(report.get("system"))
    runtime = mapping(report.get("runtime"))
    git = mapping(report.get("git"))
    readiness = mapping(report.get("readiness"))
    strict_tracker = mapping(report.get("strict_tracker"))
    backends = mapping(report.get("backends"))
    emi = mapping(backends.get("windows_emi"))
    nvml = mapping(backends.get("nvml"))
    device = mapping(nvml.get("selected_device"))
    torch_cuda = mapping(backends.get("torch_cuda"))
    windows_build = system.get("windows_build")
    return bool(
        report.get("schema_version") == software_recipe.PREFLIGHT_SCHEMA_VERSION
        and report.get("host_id") == recipe.host_id
        and system.get("execution_layer") == "windows-native"
        and isinstance(windows_build, int)
        and not isinstance(windows_build, bool)
        and windows_build >= 22000
        and runtime.get("codecarbon") == software_recipe.SUPPORTED_CODECARBON_VERSION
        and git.get("available") is True
        and git.get("sha") == state.get("git_sha")
        and git.get("worktree_fingerprint_sha256") == state.get("worktree_fingerprint_sha256")
        and readiness.get("exploratory_software_ready") is True
        and strict_tracker.get("component_contract_supported") is True
        and emi.get("active_probe") is True
        and emi.get("available") is True
        and emi.get("platform_windows") is True
        and emi.get("backend_mode") == "windows_emi"
        and emi.get("interface_class") == "WindowsEMI"
        and emi.get("fallback_used") is False
        and emi.get("measurement_scope") == "cpu_package"
        and emi.get("ram_included") is False
        and emi.get("counter_positive") is True
        and positive(emi.get("sample_energy_j"))
        and nvml.get("active_probe") is True
        and nvml.get("available") is True
        and nvml.get("selection_valid") is True
        and nvml.get("selected_index") == recipe.gpu_index
        and device.get("selected") is True
        and device.get("index") == recipe.gpu_index
        and device.get("device_id_sha256") == recipe.gpu_device_id_sha256
        and device.get("measurement_mode") == "total_energy_counter"
        and device.get("measurement_mode_supported") is True
        and device.get("total_energy_counter_supported") is True
        and device.get("counter_monotonic") is True
        and device.get("counter_positive") is True
        and positive(device.get("sample_energy_delta_mj"))
        and torch_cuda.get("available") is True
        and torch_cuda.get("active_probe") is True
        and torch_cuda.get("selection_valid") is True
        and torch_cuda.get("identity_verified") is True
        and torch_cuda.get("visibility_remapped") is False
        and torch_cuda.get("requested_nvml_index") == recipe.gpu_index
        and torch_cuda.get("selected_cuda_index") == recipe.gpu_index
        and torch_cuda.get("compute_probe_passed") is True
    )


def _recover_interrupted_artifacts(
    recipe: DeploymentEnergyRecipe,
    output: Path,
    state: dict[str, Any],
) -> None:
    """Recover only runner-owned remnants while the state is still incomplete."""

    for path in sorted(output.rglob("*")):
        if not _PARTIAL_ARTIFACT.fullmatch(path.name):
            continue
        if path.is_symlink() or not path.is_file():
            raise DeploymentEnergyPilotError("interrupted partial artifact is not a regular file")
        path.unlink()

    orphaned = state.setdefault("orphaned_preflight_evidence", [])
    if not isinstance(orphaned, list):
        raise DeploymentEnergyPilotError("orphaned preflight history is malformed")
    known = {str(evidence["path"]) for evidence in _unique_preflight_evidence(state)}
    qualification = output / "qualification"
    changed = False
    if qualification.exists() or qualification.is_symlink():
        if qualification.is_symlink() or not qualification.is_dir():
            raise DeploymentEnergyPilotError("qualification evidence directory is unsafe")
        for path in sorted(qualification.iterdir()):
            relative = path.relative_to(output).as_posix()
            match = _PRESERVED_PREFLIGHT.fullmatch(path.name)
            if (
                match is None
                or path.is_symlink()
                or not path.is_file()
                or _sha256_file(path) != match.group(1)
            ):
                raise DeploymentEnergyPilotError("unrecognized qualification evidence artifact")
            if relative in known:
                continue
            if not _preflight_evidence_is_qualified(recipe, state, path):
                raise DeploymentEnergyPilotError("orphaned preflight evidence is not qualified")
            orphaned.append(
                {
                    "path": relative,
                    "sha256": match.group(1),
                    "reason": "interrupted_before_session_state_promotion",
                }
            )
            known.add(relative)
            changed = True
    if changed:
        orphaned.sort(key=lambda item: (str(item["path"]), str(item["sha256"])))
        _atomic_write_json(output / "run-state.json", state)
    _validate_preserved_preflights(output, state)


def _preserve_session_preflight(
    recipe: DeploymentEnergyRecipe,
    root: Path,
    output: Path,
) -> dict[str, str]:
    source = _relative(root, recipe.preflight_report)
    if source.is_symlink() or not source.is_file():
        raise DeploymentEnergyQualificationError("qualified preflight report is unavailable")
    payload = source.read_bytes()
    digest = _sha256_bytes(payload)
    relative = f"qualification/preflight-{digest}.json"
    destination = _relative(output, relative)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise DeploymentEnergyPilotError("content-addressed preflight evidence changed")
    else:
        _atomic_write_bytes(destination, payload)
    return {"path": relative, "sha256": digest}


def _content_sha256(coords: Any, demands: Any, capacity: float) -> str:
    """Use the established quality-corpus domain so prior hashes are comparable."""

    return quality._content_sha256(coords, demands, capacity)


def _expected_arrays(recipe: DeploymentEnergyRecipe) -> tuple[Any, Any]:
    import numpy as np
    import torch

    from neuro_co.core.factory import make_env

    spec = recipe.dataset
    env = make_env(
        spec.problem,
        size=spec.size,
        capacity=spec.capacity,
        max_demand=spec.max_demand,
    )
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    state = env.reset(spec.num_instances, generator=generator, device="cpu")
    return (
        np.ascontiguousarray(state.coords.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(state.demand.detach().cpu().numpy(), dtype=np.float32),
    )


def _prepare_corpus(
    recipe: DeploymentEnergyRecipe,
    output: Path,
    state: dict[str, Any],
    *,
    read_only: bool = False,
) -> Corpus:
    import numpy as np

    path = _relative(output, recipe.dataset.artifact)
    expected_coords, expected_demands = _expected_arrays(recipe)
    expected_content = _content_sha256(expected_coords, expected_demands, recipe.dataset.capacity)
    if expected_content in recipe.dataset.forbidden_content_sha256:
        raise DeploymentEnergyPilotError("fresh seed2724 corpus duplicates a prior corpus")
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                schema = str(archive["schema_version"].item())
                seed = int(archive["seed"].item())
                coords = np.ascontiguousarray(archive["coords"], dtype=np.float32)
                demands = np.ascontiguousarray(archive["demands"], dtype=np.float32)
                capacity = float(archive["capacity"].item())
                embedded = str(archive["content_sha256"].item())
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise DeploymentEnergyPilotError("seed2724 corpus is unreadable") from exc
        if (
            schema != CORPUS_SCHEMA
            or seed != recipe.dataset.seed
            or capacity != recipe.dataset.capacity
            or embedded != expected_content
            or not np.array_equal(coords, expected_coords)
            or not np.array_equal(demands, expected_demands)
        ):
            raise DeploymentEnergyPilotError("seed2724 corpus changed or is not deterministic")
    else:
        if read_only:
            raise DeploymentEnergyPilotError("completed seed2724 corpus is missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}.npz"
        try:
            np.savez_compressed(
                temporary,
                schema_version=np.asarray(CORPUS_SCHEMA),
                seed=np.asarray(recipe.dataset.seed, dtype=np.int64),
                coords=expected_coords,
                demands=expected_demands,
                capacity=np.asarray(recipe.dataset.capacity, dtype=np.float64),
                content_sha256=np.asarray(expected_content),
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        coords, demands, capacity = expected_coords, expected_demands, recipe.dataset.capacity
    corpus = Corpus(
        coords=coords,
        demands=demands,
        capacity=capacity,
        content_sha256=expected_content,
        file_sha256=_sha256_file(path),
        path=path,
    )
    record = {
        "schema_version": CORPUS_SCHEMA,
        "dataset_id": recipe.dataset.dataset_id,
        "seed": recipe.dataset.seed,
        "num_instances": recipe.dataset.num_instances,
        "content_sha256": corpus.content_sha256,
        "file_sha256": corpus.file_sha256,
        "path": recipe.dataset.artifact,
        "fresh_against_prior_content_hashes": True,
        "opened_after_durable_source_receipt": True,
    }
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists() and _load_json(manifest_path) != record:
        raise DeploymentEnergyPilotError("seed2724 corpus manifest changed")
    if not manifest_path.exists():
        if read_only:
            raise DeploymentEnergyPilotError("completed seed2724 corpus manifest is missing")
        _atomic_write_json(manifest_path, record)
    stored_dataset = state.get("dataset")
    if stored_dataset not in (None, record):
        raise DeploymentEnergyPilotError("durable seed2724 dataset identity changed")
    if stored_dataset is None:
        if read_only:
            raise DeploymentEnergyPilotError("completed seed2724 dataset anchor is missing")
        state["dataset"] = record
        _atomic_write_json(output / "run-state.json", state)
    return corpus


def _make_strict_tracker(label: str, recipe: DeploymentEnergyRecipe) -> Any:
    from neuro_co.aet import EnergyTracker

    return EnergyTracker(
        label,
        backend="hwcounters",
        pue=1.0,
        report_embodied=False,
        grid_intensity_g_per_kwh=0.0,
        allow_fallback=False,
        required_domains={"cpu", "gpu"},
        gpu_indices=[recipe.gpu_index],
        items=0,
    )


def _checked_energy(
    tracker: Any,
    recipe: DeploymentEnergyRecipe,
    *,
    items_processed: int,
) -> dict[str, Any]:
    payload = software._checked_energy(
        tracker,
        minimum_duration_s=recipe.minimum_block_duration_s,
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    if payload.get("items_processed") != items_processed:
        raise DeploymentEnergyPilotError("energy item count differs from completed work")
    for key in ("co2_operational_kg", "co2_embodied_kg", "co2_total_kg"):
        value = payload.get(key)
        if value not in (None, 0, 0.0):
            raise DeploymentEnergyPilotError("carbon output must be absent or exactly zero")
    cpu = float(payload["energy_cpu_j"])
    gpu = float(payload["energy_gpu_j"])
    observed = cpu + gpu
    if not math.isclose(float(payload["energy_j"]), observed, rel_tol=1e-12, abs_tol=1e-9):
        raise DeploymentEnergyPilotError("tracker total differs from CPU package plus GPU")
    payload.update(
        {
            "cpu_package_energy_j": cpu,
            "gpu_energy_j": gpu,
            "observed_component_energy_j": observed,
            "cpu_package_energy_j_per_instance": cpu / items_processed,
            "gpu_energy_j_per_instance": gpu / items_processed,
            "observed_component_energy_j_per_instance": observed / items_processed,
            "whole_system_energy": False,
            "cross_solver_energy_comparable": False,
            "carbon_accounting": "none",
        }
    )
    return payload


def _diagnostic_gpu_snapshot(gpu_index: int) -> dict[str, Any]:
    try:
        snapshot = software._gpu_process_snapshot(gpu_index)
        snapshot["diagnostic_only"] = True
        snapshot["can_block_execution"] = False
        return snapshot
    except Exception as exc:
        return {
            "diagnostic_only": True,
            "can_block_execution": False,
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _route_hash(
    routes: Any,
    validation: dict[str, Any],
    solver_metadata: Sequence[dict[str, Any]] | None = None,
) -> str:
    payload: dict[str, Any] = {"routes": routes, "validation": validation}
    if solver_metadata is not None:
        payload["solver_result_metadata"] = list(solver_metadata)
    return _sha256_bytes(_json_bytes(payload))


def _validated_hgs_results(
    results: Sequence[Any],
    *,
    expected_instances: int,
    base_seed: int,
    max_iterations: int,
    scaling_factor: int,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Validate every PyVRP result before accepting one corpus repetition."""

    if len(results) != expected_instances:
        raise DeploymentEnergyPilotError("HGS did not return one result per corpus instance")
    routes: list[Any] = []
    records: list[dict[str, Any]] = []
    for instance_index, result in enumerate(results):
        try:
            observed_instance = int(result.instance_index)
            effective_seed = int(result.seed)
            observed_iterations = int(result.max_iterations)
            observed_scaling = int(result.scaling_factor)
            integer_cost = result.integer_cost
            reported_cost = result.cost
            raw_routes = result.routes
        except (AttributeError, TypeError, ValueError) as exc:
            raise DeploymentEnergyPilotError("HGS result metadata is incomplete") from exc
        if (
            observed_instance != instance_index
            or effective_seed != base_seed + instance_index
            or observed_iterations != max_iterations
            or observed_scaling != scaling_factor
        ):
            raise DeploymentEnergyPilotError(
                "HGS result metadata disagrees with the fixed seed or budget"
            )
        if (
            isinstance(integer_cost, bool)
            or not isinstance(integer_cost, int)
            or isinstance(reported_cost, bool)
            or not isinstance(reported_cost, (int, float))
            or not math.isfinite(float(reported_cost))
            or not math.isclose(
                float(reported_cost),
                integer_cost / scaling_factor,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise DeploymentEnergyPilotError("HGS objective metadata is inconsistent")
        normalized_routes = tuple(
            tuple(int(customer) for customer in route) for route in raw_routes
        )
        routes.append(normalized_routes)
        records.append(
            {
                "instance_index": observed_instance,
                "effective_seed": effective_seed,
                "integer_cost": integer_cost,
                "reported_cost": float(reported_cost),
                "max_iterations": observed_iterations,
                "scaling_factor": observed_scaling,
            }
        )
    return tuple(routes), records


def _base_quality_recipe(recipe: DeploymentEnergyRecipe, root: Path) -> Any:
    source = _relative(root, recipe.checkpoint_source.root)
    return load_aet_quality_recipe(source / "base-recipe.yaml")


def _evaluation_mode(recipe: DeploymentEnergyRecipe) -> EvaluationMode:
    policy = recipe.neural_policy
    return EvaluationMode(
        mode_id=policy.mode_id,
        n_starts=policy.n_starts,
        augmentations=policy.augmentations,
        forced_first_actions=policy.forced_first_actions,
        batch_size=policy.batch_size,
        inference_precision=policy.inference_precision,
    )


def _execute_neural_block(
    recipe: DeploymentEnergyRecipe,
    block: EnergyBlock,
    *,
    root: Path,
    corpus: Corpus,
    attestation: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from neuro_co.core.factory import make_env

    base_recipe = _base_quality_recipe(recipe, root)
    mode = _evaluation_mode(recipe)
    checkpoint = recipe.checkpoints[block.round_index]
    checkpoint_path = _relative(_relative(root, recipe.checkpoint_source.root), checkpoint.path)
    if _sha256_file(checkpoint_path) != checkpoint.sha256:
        raise DeploymentEnergyPilotError("frozen neural checkpoint changed before measurement")
    checkpoint_payload = quality._load_torch_mapping(checkpoint_path)
    env = make_env(
        recipe.dataset.problem,
        size=recipe.dataset.size,
        capacity=recipe.dataset.capacity,
        max_demand=recipe.dataset.max_demand,
    )
    model = quality._make_mlco_am(base_recipe, env)
    if checkpoint_payload.get("model_identity") != quality._model_identity(base_recipe, model):
        raise DeploymentEnergyPilotError("checkpoint model identity changed")
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    device = torch.device(f"cuda:{recipe.gpu_index}")
    model.to(device=device, dtype=torch.float32).eval()

    def cycle(coords: Any = corpus.coords, demands: Any = corpus.demands) -> tuple[Any, ...]:
        routes: list[Any] = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for start in range(0, coords.shape[0], mode.batch_size):
                stop = min(start + mode.batch_size, coords.shape[0])
                state = quality._initial_state(env, coords[start:stop], demands[start:stop], device)
                routes.extend(
                    quality._multistart_routes(
                        model,
                        env,
                        state,
                        n_starts=mode.n_starts,
                        augmentations=mode.augmentations,
                        seed=block.evaluation_seed + start,
                    )
                )
        torch.cuda.synchronize(device)
        return tuple(routes)

    warmup_routes = cycle(corpus.coords[: mode.batch_size], corpus.demands[: mode.batch_size])
    warmup_validation = quality.validate_routes(
        corpus.coords[: mode.batch_size],
        corpus.demands[: mode.batch_size],
        corpus.capacity,
        warmup_routes,
    )
    if warmup_validation["complete"] is not True:
        raise DeploymentEnergyPilotError("neural warmup produced invalid routes")
    before = _diagnostic_gpu_snapshot(recipe.gpu_index)
    tracker = _make_strict_tracker(
        f"deployment-neural-r{block.round_index:02d}-seed{block.training_seed}", recipe
    )
    repetitions = 0
    first_routes: tuple[Any, ...] | None = None
    last_routes: tuple[Any, ...] | None = None
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    with tracker as active:
        measured_started = time.perf_counter()
        while True:
            current = cycle()
            first_routes = current if first_routes is None else first_routes
            last_routes = current
            repetitions += 1
            measured_elapsed = time.perf_counter() - measured_started
            if time.perf_counter() - wall_started > recipe.maximum_block_walltime_s:
                raise DeploymentEnergyPilotError("neural block exceeded maximum wall time")
            if measured_elapsed >= recipe.minimum_block_duration_s:
                break
        active.n_items = repetitions * recipe.dataset.num_instances
    ended_at = datetime.now(UTC)
    after = _diagnostic_gpu_snapshot(recipe.gpu_index)
    if first_routes is None or last_routes is None:
        raise DeploymentEnergyPilotError("neural block completed no corpus repetition")
    first_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, first_routes
    )
    last_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, last_routes
    )
    if first_validation["complete"] is not True or last_validation["complete"] is not True:
        raise DeploymentEnergyPilotError("neural measured block produced invalid routes")
    first_hash = _route_hash(first_routes, first_validation)
    last_hash = _route_hash(last_routes, last_validation)
    if first_hash != last_hash:
        raise DeploymentEnergyPilotError("neural first and last corpus outputs differ")
    instances = repetitions * recipe.dataset.num_instances
    energy = _checked_energy(tracker, recipe, items_processed=instances)
    return _block_payload(
        recipe,
        block,
        corpus=corpus,
        attestation=attestation,
        started_at=started_at,
        ended_at=ended_at,
        repetitions=repetitions,
        instances=instances,
        energy=energy,
        validation=first_validation,
        first_hash=first_hash,
        last_hash=last_hash,
        before=before,
        after=after,
        policy_details={
            "mode_id": mode.mode_id,
            "training_seed": block.training_seed,
            "evaluation_seed": block.evaluation_seed,
            "checkpoint_epoch": recipe.neural_policy.checkpoint_epoch,
            "checkpoint_path": checkpoint.path,
            "checkpoint_sha256": checkpoint.sha256,
            "n_starts": mode.n_starts,
            "augmentations": mode.augmentations,
            "batch_size": mode.batch_size,
            "inference_precision": mode.inference_precision,
        },
    )


def _execute_hgs_block(
    recipe: DeploymentEnergyRecipe,
    block: EnergyBlock,
    *,
    corpus: Corpus,
    attestation: dict[str, Any],
) -> dict[str, Any]:
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

    def cycle(
        coords: Any = corpus.coords,
        demands: Any = corpus.demands,
    ) -> Sequence[Any]:
        return solve_corpus_sequential(
            coords,
            demands,
            corpus.capacity,
            seed=block.hgs_seed,
            max_iterations=recipe.hgs_policy.max_iterations,
            scaling_factor=recipe.hgs_policy.scaling_factor,
            collect_stats=recipe.hgs_policy.collect_stats,
        )

    warmup_results = cycle(corpus.coords[:4], corpus.demands[:4])
    warmup_routes, _warmup_metadata = _validated_hgs_results(
        warmup_results,
        expected_instances=4,
        base_seed=block.hgs_seed,
        max_iterations=recipe.hgs_policy.max_iterations,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    warmup_validation = quality.validate_routes(
        corpus.coords[:4], corpus.demands[:4], corpus.capacity, warmup_routes
    )
    if warmup_validation["complete"] is not True:
        raise DeploymentEnergyPilotError("HGS warmup produced invalid routes")
    before = _diagnostic_gpu_snapshot(recipe.gpu_index)
    tracker = _make_strict_tracker(
        f"deployment-hgs-r{block.round_index:02d}-seed{block.hgs_seed}", recipe
    )
    repetitions = 0
    first_results: Sequence[Any] | None = None
    last_results: Sequence[Any] | None = None
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    with tracker as active:
        measured_started = time.perf_counter()
        while True:
            current_results = cycle()
            first_results = current_results if first_results is None else first_results
            last_results = current_results
            repetitions += 1
            measured_elapsed = time.perf_counter() - measured_started
            if time.perf_counter() - wall_started > recipe.maximum_block_walltime_s:
                raise DeploymentEnergyPilotError("HGS block exceeded maximum wall time")
            if measured_elapsed >= recipe.minimum_block_duration_s:
                break
        active.n_items = repetitions * recipe.dataset.num_instances
    ended_at = datetime.now(UTC)
    after = _diagnostic_gpu_snapshot(recipe.gpu_index)
    if first_results is None or last_results is None:
        raise DeploymentEnergyPilotError("HGS block completed no corpus repetition")
    first_routes, first_metadata = _validated_hgs_results(
        first_results,
        expected_instances=recipe.dataset.num_instances,
        base_seed=block.hgs_seed,
        max_iterations=recipe.hgs_policy.max_iterations,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    last_routes, last_metadata = _validated_hgs_results(
        last_results,
        expected_instances=recipe.dataset.num_instances,
        base_seed=block.hgs_seed,
        max_iterations=recipe.hgs_policy.max_iterations,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    first_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, first_routes
    )
    last_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, last_routes
    )
    if first_validation["complete"] is not True or last_validation["complete"] is not True:
        raise DeploymentEnergyPilotError("HGS measured block produced invalid routes")
    first_hash = _route_hash(first_routes, first_validation, first_metadata)
    last_hash = _route_hash(last_routes, last_validation, last_metadata)
    if first_hash != last_hash:
        raise DeploymentEnergyPilotError("HGS first and last corpus outputs differ")
    instances = repetitions * recipe.dataset.num_instances
    energy = _checked_energy(tracker, recipe, items_processed=instances)
    return _block_payload(
        recipe,
        block,
        corpus=corpus,
        attestation=attestation,
        started_at=started_at,
        ended_at=ended_at,
        repetitions=repetitions,
        instances=instances,
        energy=energy,
        validation=first_validation,
        first_hash=first_hash,
        last_hash=last_hash,
        before=before,
        after=after,
        solver_metadata=first_metadata,
        last_solver_metadata=last_metadata,
        policy_details={
            "solver": recipe.hgs_policy.solver,
            "base_seed": block.hgs_seed,
            "max_iterations": recipe.hgs_policy.max_iterations,
            "scaling_factor": recipe.hgs_policy.scaling_factor,
            "collect_stats": recipe.hgs_policy.collect_stats,
            "cpu_threads": recipe.hgs_policy.cpu_threads,
        },
    )


def _block_payload(
    recipe: DeploymentEnergyRecipe,
    block: EnergyBlock,
    *,
    corpus: Corpus,
    attestation: dict[str, Any],
    started_at: datetime,
    ended_at: datetime,
    repetitions: int,
    instances: int,
    energy: dict[str, Any],
    validation: dict[str, Any],
    first_hash: str,
    last_hash: str,
    before: dict[str, Any],
    after: dict[str, Any],
    policy_details: dict[str, Any],
    solver_metadata: list[dict[str, Any]] | None = None,
    last_solver_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    duration = float(energy["duration_s"])
    payload = {
        "schema_version": BLOCK_SCHEMA,
        "status": "complete",
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": block.policy,
        "dataset_content_sha256": corpus.content_sha256,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "attestation": attestation,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "repetitions": repetitions,
        "instances_per_repetition": recipe.dataset.num_instances,
        "instances_processed": instances,
        "duration_s": duration,
        "throughput_instances_per_s": instances / duration,
        "energy": energy,
        "validation": validation,
        "first_measured_output_sha256": first_hash,
        "last_measured_output_sha256": last_hash,
        "first_and_last_outputs_identical": True,
        "warmup_included_in_measurement": False,
        "validation_included_in_measurement": False,
        "serialization_included_in_measurement": False,
        "gpu_process_snapshot_before": before,
        "gpu_process_snapshot_after": after,
        "gpu_process_lists_are_diagnostic_only": True,
        "policy_details": policy_details,
    }
    if solver_metadata is not None:
        if last_solver_metadata is None:
            raise DeploymentEnergyPilotError("HGS last-cycle solver metadata is absent")
        payload["solver_result_metadata"] = solver_metadata
        payload["first_solver_result_metadata_sha256"] = _sha256_bytes(
            _json_bytes({"solver_result_metadata": solver_metadata})
        )
        payload["last_solver_result_metadata_sha256"] = _sha256_bytes(
            _json_bytes({"solver_result_metadata": last_solver_metadata})
        )
    return payload


def _validate_block(
    payload: dict[str, Any],
    recipe: DeploymentEnergyRecipe,
    block: EnergyBlock,
    corpus: Corpus,
    *,
    expected_session_id: str | None = None,
) -> None:
    expected = {
        "schema_version": BLOCK_SCHEMA,
        "status": "complete",
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": block.policy,
        "dataset_content_sha256": corpus.content_sha256,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "instances_per_repetition": recipe.dataset.num_instances,
        "first_and_last_outputs_identical": True,
        "gpu_process_lists_are_diagnostic_only": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise DeploymentEnergyPilotError(f"cached block identity changed: {block.relative_path}")
    if (
        expected_session_id is not None
        and payload.get("attestation", {}).get("session_id") != expected_session_id
    ):
        raise DeploymentEnergyPilotError("block attestation differs from its durable attempt")
    repetitions = payload.get("repetitions")
    instances = payload.get("instances_processed")
    duration = payload.get("duration_s")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or instances != repetitions * recipe.dataset.num_instances
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration < recipe.minimum_block_duration_s
        or payload.get("validation", {}).get("complete") is not True
    ):
        raise DeploymentEnergyPilotError(f"cached block metrics changed: {block.relative_path}")
    energy = payload.get("energy")
    if not isinstance(energy, dict):
        raise DeploymentEnergyPilotError("cached block has no energy mapping")
    for key in (
        "cpu_package_energy_j",
        "gpu_energy_j",
        "observed_component_energy_j",
        "cpu_package_energy_j_per_instance",
        "gpu_energy_j_per_instance",
        "observed_component_energy_j_per_instance",
    ):
        value = energy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise DeploymentEnergyPilotError(f"cached block energy changed: {key}")
    if energy.get("whole_system_energy") is not False or energy.get("carbon_accounting") != "none":
        raise DeploymentEnergyPilotError("cached block classification changed")
    if any(
        energy.get(key) not in (None, 0, 0.0)
        for key in ("co2_operational_kg", "co2_embodied_kg", "co2_total_kg")
    ):
        raise DeploymentEnergyPilotError("cached block contains nonzero carbon output")
    if block.policy == "hgs":
        records = payload.get("solver_result_metadata")
        if not isinstance(records, list) or len(records) != recipe.dataset.num_instances:
            raise DeploymentEnergyPilotError("cached HGS block lacks complete solver metadata")
        expected_metadata_sha = _sha256_bytes(_json_bytes({"solver_result_metadata": records}))
        if (
            payload.get("first_solver_result_metadata_sha256") != expected_metadata_sha
            or payload.get("last_solver_result_metadata_sha256") != expected_metadata_sha
        ):
            raise DeploymentEnergyPilotError("cached HGS solver metadata hash changed")
        for instance_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise DeploymentEnergyPilotError("cached HGS solver metadata is malformed")
            integer_cost = record.get("integer_cost")
            reported_cost = record.get("reported_cost")
            if (
                record.get("instance_index") != instance_index
                or record.get("effective_seed") != block.hgs_seed + instance_index
                or record.get("max_iterations") != recipe.hgs_policy.max_iterations
                or record.get("scaling_factor") != recipe.hgs_policy.scaling_factor
                or isinstance(integer_cost, bool)
                or not isinstance(integer_cost, int)
                or isinstance(reported_cost, bool)
                or not isinstance(reported_cost, (int, float))
                or not math.isfinite(float(reported_cost))
                or not math.isclose(
                    float(reported_cost),
                    integer_cost / recipe.hgs_policy.scaling_factor,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise DeploymentEnergyPilotError("cached HGS solver metadata changed")
        if payload.get("first_measured_output_sha256") != payload.get(
            "last_measured_output_sha256"
        ):
            raise DeploymentEnergyPilotError("cached HGS first and last outputs differ")


def _series_summary(values: list[float]) -> dict[str, Any]:
    n = len(values)
    average = mean(values)
    sample_sd = stdev(values) if n > 1 else 0.0
    half_width = 2.7764451051977987 * sample_sd / math.sqrt(n) if n == 5 else None
    return {
        "n": n,
        "mean": average,
        "median": median(values),
        "sample_standard_deviation": sample_sd,
        "minimum": min(values),
        "maximum": max(values),
        "descriptive_t_interval_95": (
            [average - half_width, average + half_width] if half_width is not None else None
        ),
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "inferential_use": False,
    }


def paired_directional_summary(block_payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired per-instance energy at both explicit accounting boundaries."""

    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for payload in block_payloads:
        round_index = int(payload["round"])
        policy = str(payload["policy"])
        if policy not in {"neural", "hgs"} or policy in by_round.setdefault(round_index, {}):
            raise DeploymentEnergyPilotError("paired summary received duplicate policy blocks")
        by_round[round_index][policy] = payload
    if set(by_round) != set(range(5)) or any(
        set(pair) != {"neural", "hgs"} for pair in by_round.values()
    ):
        raise DeploymentEnergyPilotError("paired summary requires five complete policy pairs")

    pairs: list[dict[str, Any]] = []
    same_host_deltas: list[float] = []
    conservative_deltas: list[float] = []
    for round_index in range(5):
        neural = by_round[round_index]["neural"]
        hgs = by_round[round_index]["hgs"]
        neural_energy = neural["energy"]
        hgs_energy = hgs["energy"]
        neural_total = float(neural_energy["observed_component_energy_j_per_instance"])
        hgs_total = float(hgs_energy["observed_component_energy_j_per_instance"])
        hgs_cpu = float(hgs_energy["cpu_package_energy_j_per_instance"])
        same_host = hgs_total - neural_total
        conservative = hgs_cpu - neural_total
        same_host_deltas.append(same_host)
        conservative_deltas.append(conservative)
        pairs.append(
            {
                "round": round_index,
                "neural": {
                    "cpu_package_j_per_instance": float(
                        neural_energy["cpu_package_energy_j_per_instance"]
                    ),
                    "gpu_j_per_instance": float(neural_energy["gpu_energy_j_per_instance"]),
                    "observed_components_j_per_instance": neural_total,
                    "duration_s": float(neural["duration_s"]),
                    "repetitions": int(neural["repetitions"]),
                    "instances": int(neural["instances_processed"]),
                    "throughput_instances_per_s": float(neural["throughput_instances_per_s"]),
                },
                "hgs": {
                    "cpu_package_j_per_instance": hgs_cpu,
                    "gpu_j_per_instance": float(hgs_energy["gpu_energy_j_per_instance"]),
                    "observed_components_j_per_instance": hgs_total,
                    "duration_s": float(hgs["duration_s"]),
                    "repetitions": int(hgs["repetitions"]),
                    "instances": int(hgs["instances_processed"]),
                    "throughput_instances_per_s": float(hgs["throughput_instances_per_s"]),
                },
                "same_host_observed_delta_hgs_minus_neural_j_per_instance": same_host,
                "conservative_delta_hgs_cpu_minus_neural_cpu_gpu_j_per_instance": conservative,
            }
        )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "classification": classification(),
        "interpretation": "variance_and_directional_signal_only",
        "paired_rounds": 5,
        "pairs": pairs,
        "same_host_observed_boundary": {
            "definition": "HGS CPU package plus idle GPU minus neural CPU package plus GPU",
            "delta_unit": "joule_per_instance",
            "positive_delta_interpretation": "lower observed component energy for neural",
            "summary": _series_summary(same_host_deltas),
        },
        "conservative_neural_boundary": {
            "definition": "HGS CPU package only minus neural CPU package plus GPU",
            "delta_unit": "joule_per_instance",
            "positive_delta_interpretation": "lower observed component energy for neural",
            "summary": _series_summary(conservative_deltas),
        },
        "idle_gpu_energy_during_hgs_is_explicit": True,
        "cross_solver_energy_comparable": False,
        "whole_system_energy": False,
        "confirmatory_eligible": False,
        "aet_computed": False,
        "carbon_accounting": "none",
    }


def _write_checksums(output: Path) -> str:
    records: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in {"SHA256SUMS", "run-state.json"} or ".partial-" in path.name:
            continue
        records.append(f"{_sha256_file(path)}  {relative}")
    _atomic_write_bytes(output / "SHA256SUMS", ("\n".join(records) + "\n").encode())
    return _sha256_file(output / "SHA256SUMS")


def _expected_output_files(
    recipe: DeploymentEnergyRecipe,
    state: dict[str, Any],
    *,
    include_checksums: bool,
) -> set[str]:
    corpus_path = PurePosixPath(recipe.dataset.artifact)
    expected = {
        "environment/uv.lock",
        "recipe.yaml",
        "source-receipt.json",
        "run-state.json",
        corpus_path.as_posix(),
        corpus_path.with_suffix(".manifest.json").as_posix(),
        "paired-summary.json",
        "manifest.json",
        *(block.relative_path for block in expected_blocks(recipe)),
    }
    expected.update(item["path"] for item in _unique_preflight_evidence(state))
    if include_checksums:
        expected.add("SHA256SUMS")
    return expected


def _assert_exact_output_inventory(
    recipe: DeploymentEnergyRecipe,
    output: Path,
    state: dict[str, Any],
    *,
    include_checksums: bool,
) -> None:
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected = _expected_output_files(
        recipe,
        state,
        include_checksums=include_checksums,
    )
    if actual != expected:
        raise DeploymentEnergyPilotError(
            "pilot output contains missing or unregistered files; refusing to seal it"
        )


def _validate_completed_output(
    recipe: DeploymentEnergyRecipe,
    output: Path,
    state: dict[str, Any],
    blocks: Sequence[dict[str, Any]],
) -> None:
    summary_path = output / "paired-summary.json"
    manifest_path = output / "manifest.json"
    checksums_path = output / "SHA256SUMS"
    if (
        state.get("status") != COMPLETE_STATUS
        or state.get("paired_summary_sha256") != _sha256_file(summary_path)
        or state.get("manifest_sha256") != _sha256_file(manifest_path)
        or state.get("checksums_sha256") != _sha256_file(checksums_path)
    ):
        raise DeploymentEnergyPilotError("completed pilot anchors changed")
    if _load_json(summary_path) != paired_directional_summary(blocks):
        raise DeploymentEnergyPilotError("completed paired summary changed")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != COMPLETE_STATUS
        or manifest.get("classification") != classification()
        or manifest.get("block_count") != len(expected_blocks(recipe))
        or manifest.get("paired_summary_sha256") != state["paired_summary_sha256"]
        or manifest.get("preflight_evidence") != _unique_preflight_evidence(state)
        or manifest.get("interrupted_preflight_evidence")
        != state.get("orphaned_preflight_evidence", [])
        or manifest.get("aet_was_computed") is not False
        or manifest.get("carbon_was_computed") is not False
    ):
        raise DeploymentEnergyPilotError("completed manifest changed")
    _assert_exact_output_inventory(recipe, output, state, include_checksums=True)
    expected_files: set[str] = set()
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise DeploymentEnergyPilotError("completed checksum inventory is malformed")
        digest, relative = line[:64], line[66:]
        path = _relative(output, relative)
        if relative in expected_files or not path.is_file() or _sha256_file(path) != digest:
            raise DeploymentEnergyPilotError("completed checksum inventory verification failed")
        expected_files.add(relative)
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.relative_to(output).as_posix() not in {"SHA256SUMS", "run-state.json"}
    }
    if expected_files != actual_files:
        raise DeploymentEnergyPilotError("completed checksum inventory is not exhaustive")


def _complete(
    recipe: DeploymentEnergyRecipe,
    output: Path,
    state: dict[str, Any],
    blocks: Sequence[dict[str, Any]],
) -> DeploymentEnergyPilotResult:
    summary = paired_directional_summary(blocks)
    _atomic_write_json(output / "paired-summary.json", summary)
    summary_sha = _sha256_file(output / "paired-summary.json")
    preflight_evidence = _unique_preflight_evidence(state)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "dataset": state["dataset"],
        "block_count": len(blocks),
        "paired_rounds": 5,
        "paired_summary_sha256": summary_sha,
        "source_receipt_sha256": _sha256_file(output / "source-receipt.json"),
        "preflight_evidence": preflight_evidence,
        "interrupted_preflight_evidence": state.get("orphaned_preflight_evidence", []),
        "quality_was_rerun": False,
        "training_was_run": False,
        "reference_was_generated": False,
        "aet_was_computed": False,
        "carbon_was_computed": False,
    }
    _atomic_write_json(output / "manifest.json", manifest)
    state["status"] = COMPLETE_STATUS
    state["completed_at"] = datetime.now(UTC).isoformat()
    state["paired_summary_sha256"] = summary_sha
    state["manifest_sha256"] = _sha256_file(output / "manifest.json")
    checksums_path = output / "SHA256SUMS"
    include_stale_checksums = checksums_path.exists() or checksums_path.is_symlink()
    if include_stale_checksums and (checksums_path.is_symlink() or not checksums_path.is_file()):
        raise DeploymentEnergyPilotError("stale checksum artifact is unsafe")
    _assert_exact_output_inventory(
        recipe,
        output,
        state,
        include_checksums=include_stale_checksums,
    )
    state["checksums_sha256"] = _write_checksums(output)
    _atomic_write_json(output / "run-state.json", state)
    manifest_path = output / "manifest.json"
    return DeploymentEnergyPilotResult(
        path=output,
        status=COMPLETE_STATUS,
        completed_blocks=len(blocks),
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
    )


def _load_completed_blocks(
    recipe: DeploymentEnergyRecipe,
    output: Path,
    state: dict[str, Any],
    corpus: Corpus,
) -> list[dict[str, Any]]:
    schedule = expected_blocks(recipe)
    completed = state.get("completed_blocks")
    hashes = state.get("block_sha256")
    if not isinstance(completed, list) or not isinstance(hashes, dict):
        raise DeploymentEnergyPilotError("durable block state is malformed")
    expected_prefix = [block.relative_path for block in schedule[: len(completed)]]
    if completed != expected_prefix or set(hashes) != set(completed):
        raise DeploymentEnergyPilotError("completed blocks are not the strict schedule prefix")
    attempts = state.get("block_attempts")
    current = state.get("current_block")
    if not isinstance(attempts, list) or (current is not None and not isinstance(current, dict)):
        raise DeploymentEnergyPilotError("durable block-attempt state is malformed")
    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise DeploymentEnergyPilotError("session history is malformed")
    known_session_ids = {
        session.get("session_id") for session in sessions if isinstance(session, dict)
    }
    if current is not None and (
        len(completed) >= len(schedule)
        or current.get("relative_path") != schedule[len(completed)].relative_path
    ):
        raise DeploymentEnergyPilotError("current block is not the next schedule item")
    payloads: list[dict[str, Any]] = []
    for index, relative in enumerate(completed):
        path = _relative(output, relative)
        if not path.is_file() or _sha256_file(path) != hashes[relative]:
            raise DeploymentEnergyPilotError(f"durable block changed: {relative}")
        completed_attempts = [
            attempt
            for attempt in attempts
            if isinstance(attempt, dict)
            and attempt.get("relative_path") == relative
            and attempt.get("artifact_sha256") == hashes[relative]
            and attempt.get("outcome") in {"complete", "complete_orphan_promotion_recovered"}
        ]
        if len(completed_attempts) != 1:
            raise DeploymentEnergyPilotError("completed block lacks one durable successful attempt")
        session_id = completed_attempts[0].get("session_id")
        if session_id not in known_session_ids:
            raise DeploymentEnergyPilotError("completed block attempt refers to an unknown session")
        payload = _load_json(path)
        _validate_block(
            payload,
            recipe,
            schedule[index],
            corpus,
            expected_session_id=str(session_id),
        )
        payloads.append(payload)
    return payloads


def run_deployment_energy_pilot(
    recipe_path: Path,
    workspace_root: Path | None = None,
    *,
    resume: bool = False,
) -> DeploymentEnergyPilotResult:
    """Execute or resume all remaining blocks in this one process."""

    root = (workspace_root or Path.cwd()).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    qualification = qualify_for_execution(recipe, root)
    if qualification["ready_to_execute"] is not True:
        raise DeploymentEnergyQualificationError("native Windows qualification did not pass")
    source_receipt = _validate_source_semantics(recipe, root)
    attestation = _exclusive_attestation(recipe)
    runtime_identity = _configure_runtime(recipe)
    output, state = _prepare_output(
        recipe_path,
        recipe,
        root,
        resume=resume,
        runtime_identity=runtime_identity,
        source_receipt=source_receipt,
    )
    corpus = _prepare_corpus(
        recipe,
        output,
        state,
        read_only=state.get("status") == COMPLETE_STATUS,
    )
    payloads = _load_completed_blocks(recipe, output, state, corpus)
    schedule = expected_blocks(recipe)
    if state.get("status") == COMPLETE_STATUS:
        if len(payloads) != len(schedule):
            raise DeploymentEnergyPilotError("complete state lacks all ten blocks")
        _validate_completed_output(recipe, output, state, payloads)
        manifest_path = output / "manifest.json"
        return DeploymentEnergyPilotResult(
            output,
            COMPLETE_STATUS,
            len(payloads),
            manifest_path,
            _sha256_file(manifest_path),
        )

    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise DeploymentEnergyPilotError("session history is malformed")
    preflight_evidence = _preserve_session_preflight(recipe, root, output)
    sessions.append(
        {
            **attestation,
            "preflight_evidence": preflight_evidence,
            "process_started_at": datetime.now(UTC).isoformat(),
            "resume": resume,
            "completed_block_count_at_start": len(state.get("completed_blocks", [])),
        }
    )
    _atomic_write_json(output / "run-state.json", state)

    executors: dict[str, Callable[..., dict[str, Any]]] = {
        "neural": _execute_neural_block,
        "hgs": _execute_hgs_block,
    }
    import torch

    for block in schedule[len(payloads) :]:
        _assert_attestation_active(recipe, attestation)
        torch.cuda.empty_cache()
        block_path = _relative(output, block.relative_path)
        if block_path.exists():
            current = state.get("current_block")
            if not isinstance(current, dict) or current.get("relative_path") != block.relative_path:
                raise DeploymentEnergyPilotError(
                    "unanchored block artifact exists without its durable attempt record"
                )
            attempt_index = int(current["attempt_index"])
            attempt = state["block_attempts"][attempt_index]
            orphan = _load_json(block_path)
            _validate_block(
                orphan,
                recipe,
                block,
                corpus,
                expected_session_id=str(attempt["session_id"]),
            )
            payload = orphan
            attempt["outcome"] = "complete_orphan_promotion_recovered"
        else:
            previous = state.get("current_block")
            if previous is not None:
                if (
                    not isinstance(previous, dict)
                    or previous.get("relative_path") != block.relative_path
                ):
                    raise DeploymentEnergyPilotError("interrupted attempt is not the next block")
                previous_index = int(previous["attempt_index"])
                state["block_attempts"][previous_index]["outcome"] = (
                    "interrupted_before_artifact_promotion"
                )
                state["block_attempts"][previous_index]["closed_at"] = datetime.now(UTC).isoformat()
            attempt_index = len(state["block_attempts"])
            attempt = {
                "attempt_index": attempt_index,
                "relative_path": block.relative_path,
                "round": block.round_index,
                "order_within_round": block.order_index,
                "policy": block.policy,
                "session_id": attestation["session_id"],
                "started_at": datetime.now(UTC).isoformat(),
                "outcome": "running",
            }
            state["block_attempts"].append(attempt)
            state["current_block"] = {
                "attempt_index": attempt_index,
                "relative_path": block.relative_path,
            }
            _atomic_write_json(output / "run-state.json", state)
            arguments: dict[str, Any] = {
                "recipe": recipe,
                "block": block,
                "corpus": corpus,
                "attestation": attestation,
            }
            if block.policy == "neural":
                arguments["root"] = root
            payload = executors[block.policy](**arguments)
            _assert_attestation_active(recipe, attestation)
            _validate_block(
                payload,
                recipe,
                block,
                corpus,
                expected_session_id=attestation["session_id"],
            )
            _atomic_write_json(block_path, payload)
        digest = _sha256_file(block_path)
        attempt["outcome"] = (
            attempt["outcome"]
            if attempt["outcome"] == "complete_orphan_promotion_recovered"
            else "complete"
        )
        attempt["completed_at"] = datetime.now(UTC).isoformat()
        attempt["artifact_sha256"] = digest
        state["completed_blocks"].append(block.relative_path)
        state["block_sha256"][block.relative_path] = digest
        state["current_block"] = None
        _atomic_write_json(output / "run-state.json", state)
        payloads.append(payload)
        print(
            json.dumps(
                {
                    "status": "block_complete",
                    "completed": len(payloads),
                    "total": len(schedule),
                    "round": block.round_index,
                    "policy": block.policy,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _assert_attestation_active(recipe, attestation)
    return _complete(recipe, output, state, payloads)


def dry_run(recipe_path: Path, workspace_root: Path | None = None) -> dict[str, Any]:
    """Delegate to the read-only recipe dry-run."""

    from neuro_co.aet.experiments.deployment_energy_recipe import dry_run as recipe_dry_run

    return recipe_dry_run(recipe_path, workspace_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_deployment_energy_pilot(args.recipe, resume=args.resume)
    except DeploymentEnergyPilotError as exc:
        print(f"deployment energy pilot failed: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(
            f"deployment energy pilot failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "classification": classification(),
                "path": str(result.path),
                "completed_blocks": result.completed_blocks,
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
