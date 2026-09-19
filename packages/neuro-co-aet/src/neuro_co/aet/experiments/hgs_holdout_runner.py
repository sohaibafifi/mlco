"""Execute the closed, quality-only CVRP50 HGS holdout profiles.

The runner opens seed 2722 only after the recipe, source snapshot, Git commit,
selection receipt, and replication receipt have been durably anchored.  It
evaluates five frozen POMO checkpoints, HGS budget 3 as the primary comparator,
and HGS budget 10 as a non-rescuing sensitivity.  It never imports or invokes
an energy tracker, performs carbon accounting, reads a preflight report, or
requests an exclusive-host attestation.  The confirmatory profile reuses the
same guarded runner with HGS-10 only on seed 2723.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from neuro_co.aet.experiments import hgs_frontier_runner as frontier
from neuro_co.aet.experiments import quality_replication_runner as replication
from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments.hgs_holdout_recipe import (
    CONFIRMATORY_POLICY_ARTIFACTS,
    PRIMARY_BUDGET,
    SELECTION_ARTIFACTS,
    SENSITIVITY_BUDGET,
    AETHGSHoldoutRecipe,
    ConfirmatoryPolicySource,
    SelectionSource,
    dry_run,
    load_aet_hgs_holdout_recipe,
    runtime_qualification,
)

RUN_STATE_SCHEMA = "aet-hgs-holdout-run-state/v1"
SELECTION_RECEIPT_SCHEMA = "aet-hgs-holdout-selection-source/v1"
POLICY_RECEIPT_SCHEMA = "aet-hgs10-confirmatory-policy-source/v1"
REPLICATION_RECEIPT_SCHEMA = "aet-hgs-holdout-replication-source/v1"
REFERENCE_LOCK_SCHEMA = "aet-hgs-holdout-reference-lock/v1"
NEURAL_EVALUATION_SCHEMA = "aet-hgs-holdout-neural-evaluation/v1"
HGS_CELL_SCHEMA = "aet-hgs-holdout-cell/v1"
ASSESSMENT_SCHEMA = "aet-hgs-holdout-assessment/v1"
MANIFEST_SCHEMA = "aet-hgs-holdout-manifest/v1"

INCOMPLETE_STATUS = "incomplete_holdout_pending"
INVALIDATED_STATUS = "invalidated_hgs_integrity_failure"
INVALIDATED_INPUT_STATUS = "invalidated_live_input_change"
COMPLETE_STATUSES = {
    "complete_primary_passed_sensitivity_passed",
    "complete_primary_passed_sensitivity_nonpass",
    "complete_primary_nonpass_sensitivity_passed",
    "complete_primary_nonpass_sensitivity_nonpass",
    "complete_confirmatory_passed",
    "complete_confirmatory_nonpass",
}

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_PYVRP_INCOMPLETE_SOLUTION = re.compile(
    r"^PyVRP did not return a complete feasible solution for instance ([0-9]+)$"
)


class HGSHoldoutError(RuntimeError):
    """Raised when a holdout artifact or execution invariant is invalid."""


class HGSHoldoutQualificationError(HGSHoldoutError):
    """Raised before opening or extending a holdout with unqualified inputs."""


class HGSHoldoutIntegrityError(HGSHoldoutError):
    """Raised when HGS violates its feasibility or result contract."""

    def __init__(self, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    path: Path
    status: str
    complete: bool
    completed_rounds: tuple[int, ...] = ()
    remaining_rounds: tuple[int, ...] = ()
    primary_passed: bool | None = None
    sensitivity_passed: bool | None = None
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    confirmatory_profile: bool = False


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    selection_receipt: dict[str, Any]
    replication_evidence: frontier.ReplicationEvidence


WorkUnitRevalidator = Callable[[str, str], None]


def _is_confirmatory_recipe(recipe: AETHGSHoldoutRecipe) -> bool:
    return recipe.confirmatory_profile


def _classification(
    recipe: AETHGSHoldoutRecipe | None = None,
    *,
    confirmatory: bool = False,
) -> dict[str, Any]:
    if confirmatory or (recipe is not None and _is_confirmatory_recipe(recipe)):
        return {
            "purpose": "confirmatory_quality_check",
            "scientific_use": True,
            "quality_evidence_role": "prospective_confirmatory_validation",
            "confirmatory_eligible": True,
            "energy_measurement": "none",
            "carbon_accounting": "none",
            "cross_solver_energy_comparable": False,
            "aet_eligible": False,
        }
    return {
        "purpose": "frozen_quality_holdout",
        "scientific_use": True,
        "quality_evidence_role": "internally_preregistered_holdout_validation",
        "confirmatory_eligible": False,
        "energy_measurement": "none",
        "carbon_accounting": "none",
        "cross_solver_energy_comparable": False,
        "aet_eligible": False,
    }


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    fields = getattr(value, "__slots__", ())
    if fields:
        return {name: getattr(value, name) for name in fields}
    raise HGSHoldoutError(f"cannot serialize {type(value).__name__} as a mapping")


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text_bytes(payload: bytes) -> tuple[bytes, str]:
    """Return platform-stable bytes while preserving binary files exactly."""

    if b"\x00" in payload:
        return payload, "binary_raw"
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload, "binary_raw"
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return normalized, "utf8_crlf_lf_canonical"


def _canonical_file_sha256(path: Path) -> tuple[str, str]:
    canonical, mode = _canonical_text_bytes(path.read_bytes())
    return _sha256_bytes(canonical), mode


def _relative(root: Path, value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise HGSHoldoutError(f"unsafe holdout artifact path: {value!r}")
    return root.joinpath(*path.parts)


def _assert_regular_under(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise HGSHoldoutQualificationError("source snapshot escaped the repository") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise HGSHoldoutQualificationError(
                f"source snapshot refuses symlink: {relative.as_posix()}"
            )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise HGSHoldoutQualificationError(
            f"source snapshot member escaped the repository: {relative.as_posix()}"
        ) from exc
    if not path.is_file():
        raise HGSHoldoutQualificationError(
            f"source snapshot member is not a file: {relative.as_posix()}"
        )
    return relative.as_posix()


def _canonical_source_snapshot(recipe: AETHGSHoldoutRecipe, root: Path) -> dict[str, Any]:
    """Fingerprint the declared source scope independent of CRLF or LF checkout style."""

    files_by_name: dict[str, Path] = {}
    for pattern in recipe.source_snapshot.include_patterns:
        matches = sorted(root.glob(pattern))
        regular = [path for path in matches if path.is_file() or path.is_symlink()]
        if not regular:
            raise HGSHoldoutQualificationError(
                f"holdout source snapshot pattern matched no file: {pattern}"
            )
        for path in regular:
            relative = _assert_regular_under(root, path)
            files_by_name[relative] = path
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for relative in sorted(files_by_name):
        digest, mode = _canonical_file_sha256(files_by_name[relative])
        files.append({"path": relative, "sha256": digest, "canonicalization": mode})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(mode.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": recipe.source_snapshot.schema_version,
        "canonicalization": "utf8_crlf_and_cr_to_lf_else_binary_raw",
        "scope": {
            "include_patterns": list(recipe.source_snapshot.include_patterns),
            "papers_and_experiment_outputs_excluded": True,
        },
        "file_count": len(files),
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


def _safe_source_root(repository_root: Path, relative: str, label: str) -> Path:
    path = _relative(repository_root, relative)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repository_root)
    except (OSError, ValueError) as exc:
        raise HGSHoldoutQualificationError(
            f"{label} is unavailable or escaped the repository"
        ) from exc
    if not resolved.is_dir():
        raise HGSHoldoutQualificationError(f"{label} is not a directory")
    return resolved


def _parse_checksum_inventory(path: Path, source_root: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise HGSHoldoutQualificationError("selection SHA256SUMS is unreadable") from exc
    inventory: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise HGSHoldoutQualificationError("selection SHA256SUMS contains an invalid line")
        digest, member = match.groups()
        member = member.replace("\\", "/")
        if member in inventory:
            raise HGSHoldoutQualificationError("selection SHA256SUMS contains a duplicate path")
        target = _relative(source_root, member)
        _assert_regular_under(source_root, target)
        if _sha256_file(target) != digest:
            raise HGSHoldoutQualificationError(f"selection checksum member changed: {member}")
        inventory[member] = digest
    if not inventory:
        raise HGSHoldoutQualificationError("selection SHA256SUMS is empty")
    return inventory


def _validate_selection_source(
    recipe: AETHGSHoldoutRecipe,
    repository_root: Path,
) -> dict[str, Any]:
    """Deeply validate the sealed selection bundle without old live-source coupling."""

    source = recipe.selection_source
    source_root = _safe_source_root(repository_root, source.root, "selection source")
    if not isinstance(source, SelectionSource):
        raise HGSHoldoutQualificationError("legacy selection source is missing")
    declared = {name: getattr(source, name) for name in SELECTION_ARTIFACTS}
    artifacts: list[dict[str, Any]] = []
    for name, artifact in declared.items():
        path = _relative(source_root, artifact.path)
        _assert_regular_under(source_root, path)
        observed = _sha256_file(path)
        if observed != artifact.sha256:
            raise HGSHoldoutQualificationError(
                f"selection source artifact changed: {artifact.path}"
            )
        artifacts.append({"name": name, "path": artifact.path, "sha256": observed})

    inventory = _parse_checksum_inventory(
        _relative(source_root, source.checksums.path), source_root
    )
    actual_files: set[str] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise HGSHoldoutQualificationError("selection source contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(source_root).as_posix())
    expected_files = set(inventory) | {source.checksums.path, source.run_state.path}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise HGSHoldoutQualificationError(
            f"selection source inventory changed; missing={missing}, extra={extra}"
        )
    for artifact in declared.values():
        if (
            artifact.path not in {source.checksums.path, source.run_state.path}
            and inventory.get(artifact.path) != artifact.sha256
        ):
            raise HGSHoldoutQualificationError(
                f"selection checksum inventory disagrees for {artifact.path}"
            )

    state = quality._load_json(_relative(source_root, source.run_state.path))
    assessment = quality._load_json(_relative(source_root, source.assessment.path))
    manifest = quality._load_json(_relative(source_root, source.manifest.path))
    neural_gate = quality._load_json(_relative(source_root, source.neural_quality_gate.path))
    reference_lock = quality._load_json(_relative(source_root, source.reference_lock.path))
    corpus_manifest = quality._load_json(
        _relative(source_root, source.selection_corpus_manifest.path)
    )
    if (
        state.get("status") != source.expected_status
        or assessment.get("status") != source.expected_status
        or manifest.get("status") != source.expected_status
        or state.get("selected_budget") != PRIMARY_BUDGET
        or assessment.get("selected_budget") != PRIMARY_BUDGET
        or PRIMARY_BUDGET not in assessment.get("passing_budgets", ())
        or SENSITIVITY_BUDGET not in assessment.get("passing_budgets", ())
        or assessment.get("selection_rule") != "smallest_passing_budget"
        or assessment.get("energy_used_for_selection") is not False
        or assessment.get("holdout_executed") is not False
        or assessment.get("aet_was_computed") is not False
        or neural_gate.get("status") != "pass"
        or neural_gate.get("passed") is not True
        or reference_lock.get("status") != "locked"
        or reference_lock.get("locked_before_candidate_evaluation") is not True
        or corpus_manifest.get("seed") != 2721
        or corpus_manifest.get("num_instances") != recipe.dataset.holdout.num_instances
        or corpus_manifest.get("content_sha256") != source.selection_corpus_content_sha256
    ):
        raise HGSHoldoutQualificationError(
            "selection source semantics do not authorize the frozen holdout"
        )
    if (
        state.get("recipe_sha256") != source.recipe.sha256
        or state.get("manifest_sha256") != source.manifest.sha256
        or state.get("checksums_sha256") != source.checksums.sha256
        or state.get("reference_lock_sha256") != source.reference_lock.sha256
        or state.get("neural_quality_gate_sha256") != source.neural_quality_gate.sha256
        or state.get("git_sha") != source.source_git_sha
        or state.get("source_snapshot", {}).get("sha256") != source.source_snapshot_sha256
    ):
        raise HGSHoldoutQualificationError("selection source provenance relationships changed")
    manifest_assessment = manifest.get("selection_assessment")
    if (
        not isinstance(manifest_assessment, dict)
        or manifest_assessment.get("sha256") != source.assessment.sha256
        or not _strict_json_equal(manifest_assessment.get("payload"), assessment)
    ):
        raise HGSHoldoutQualificationError("selection manifest assessment binding changed")
    return {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "validation": "closed_inventory_sha256_and_semantic_relationships",
        "copied_into_holdout": False,
        "root": source.root,
        "status": source.expected_status,
        "git_sha": source.source_git_sha,
        "source_snapshot_sha256": source.source_snapshot_sha256,
        "selected_budget": PRIMARY_BUDGET,
        "sensitivity_budget": SENSITIVITY_BUDGET,
        "selection_corpus_content_sha256": source.selection_corpus_content_sha256,
        "artifacts": sorted(artifacts, key=lambda item: item["name"]),
        "checksummed_member_count": len(inventory),
        "all_budgets_evaluated": assessment.get("all_budgets_evaluated"),
        "passing_budgets": assessment.get("passing_budgets"),
        "energy_used_for_selection": False,
        "holdout_executed": False,
        "aet_was_computed": False,
    }


def _validate_confirmatory_policy_source(
    recipe: AETHGSHoldoutRecipe,
    repository_root: Path,
) -> dict[str, Any]:
    source = recipe.selection_source
    if not isinstance(source, ConfirmatoryPolicySource):
        raise HGSHoldoutQualificationError("confirmatory policy source is missing")
    source_root = _safe_source_root(repository_root, source.root, "policy source")
    declared = {name: getattr(source, name) for name in CONFIRMATORY_POLICY_ARTIFACTS}
    artifacts: list[dict[str, Any]] = []
    for name, artifact in declared.items():
        path = _relative(source_root, artifact.path)
        _assert_regular_under(source_root, path)
        observed = _sha256_file(path)
        if observed != artifact.sha256:
            raise HGSHoldoutQualificationError(
                f"confirmatory policy source artifact changed: {artifact.path}"
            )
        artifacts.append({"name": name, "path": artifact.path, "sha256": observed})

    inventory = _parse_checksum_inventory(
        _relative(source_root, source.checksums.path), source_root
    )
    actual_files: set[str] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise HGSHoldoutQualificationError("confirmatory policy source contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(source_root).as_posix())
    expected_files = set(inventory) | {source.checksums.path, source.run_state.path}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise HGSHoldoutQualificationError(
            f"confirmatory policy source inventory changed; missing={missing}, extra={extra}"
        )
    for artifact in declared.values():
        if (
            artifact.path not in {source.checksums.path, source.run_state.path}
            and inventory.get(artifact.path) != artifact.sha256
        ):
            raise HGSHoldoutQualificationError(
                f"confirmatory policy checksum inventory disagrees for {artifact.path}"
            )

    state = quality._load_json(_relative(source_root, source.run_state.path))
    assessment = quality._load_json(_relative(source_root, source.assessment.path))
    manifest = quality._load_json(_relative(source_root, source.manifest.path))
    reference_lock = quality._load_json(_relative(source_root, source.reference_lock.path))
    corpus_manifest = quality._load_json(_relative(source_root, source.prior_corpus_manifest.path))
    neural = assessment.get("policies", {}).get("neural", {})
    hgs10 = assessment.get("policies", {}).get("hgs_b10", {})
    decision = assessment.get("sensitivity_decision", {})
    if (
        state.get("status") != source.expected_status
        or assessment.get("status") != source.expected_status
        or manifest.get("status") != source.expected_status
        or neural.get("passed") is not True
        or hgs10.get("passed") is not True
        or decision.get("budget") != source.confirmed_budget
        or decision.get("neural_passed") is not True
        or decision.get("hgs_sensitivity_passed") is not True
        or decision.get("joint_passed") is not True
        or assessment.get("energy_measurement") != "none"
        or assessment.get("aet_was_computed") is not False
        or reference_lock.get("status") != "locked"
        or reference_lock.get("locked_before_policy_evaluation") is not True
        or corpus_manifest.get("seed") != 2722
        or corpus_manifest.get("num_instances") != recipe.dataset.holdout.num_instances
        or corpus_manifest.get("content_sha256") != source.prior_corpus_content_sha256
    ):
        raise HGSHoldoutQualificationError(
            "seed-2722 policy source does not authorize HGS-10 confirmation"
        )
    if (
        state.get("manifest_sha256") != source.manifest.sha256
        or state.get("checksums_sha256") != source.checksums.sha256
        or state.get("assessment_sha256") != source.assessment.sha256
        or state.get("reference_lock_sha256") != source.reference_lock.sha256
        or state.get("git_sha") != source.source_git_sha
        or state.get("source_snapshot", {}).get("sha256") != source.source_snapshot_sha256
    ):
        raise HGSHoldoutQualificationError(
            "confirmatory policy source provenance relationships changed"
        )
    manifest_assessment = manifest.get("holdout_assessment")
    if (
        not isinstance(manifest_assessment, dict)
        or manifest_assessment.get("sha256") != source.assessment.sha256
        or not _strict_json_equal(manifest_assessment.get("payload"), assessment)
    ):
        raise HGSHoldoutQualificationError(
            "confirmatory policy manifest assessment binding changed"
        )
    return {
        "schema_version": POLICY_RECEIPT_SCHEMA,
        "validation": "closed_inventory_sha256_and_semantic_relationships",
        "copied_into_confirmatory_holdout": False,
        "root": source.root,
        "status": source.expected_status,
        "git_sha": source.source_git_sha,
        "source_snapshot_sha256": source.source_snapshot_sha256,
        "confirmed_budget": source.confirmed_budget,
        "prior_corpus_content_sha256": source.prior_corpus_content_sha256,
        "artifacts": sorted(artifacts, key=lambda item: item["name"]),
        "checksummed_member_count": len(inventory),
        "neural_passed": True,
        "hgs10_passed": True,
        "joint_passed": True,
        "energy_measurement": "none",
        "aet_was_computed": False,
    }


def _validate_sources(recipe: AETHGSHoldoutRecipe, root: Path) -> SourceEvidence:
    selection_receipt = (
        _validate_confirmatory_policy_source(recipe, root)
        if _is_confirmatory_recipe(recipe)
        else _validate_selection_source(recipe, root)
    )
    try:
        replication_evidence = frontier._validate_replication_source(recipe, root)
    except frontier.HGSFrontierError as exc:
        raise HGSHoldoutQualificationError(
            "sealed quality-replication source failed deep validation"
        ) from exc
    if replication_evidence.receipt.get("status") != "complete_replication_passed":
        raise HGSHoldoutQualificationError("replication source did not pass")
    source_replication = recipe.selection_source.replication_receipt
    selection_root = _safe_source_root(
        root,
        recipe.selection_source.root,
        "policy source" if _is_confirmatory_recipe(recipe) else "selection source",
    )
    selected_receipt_path = _relative(selection_root, source_replication.path)
    selected_receipt = quality._load_json(selected_receipt_path)
    if not _strict_json_equal(selected_receipt, replication_evidence.receipt):
        raise HGSHoldoutQualificationError(
            "selection and holdout disagree about the quality-replication source"
        )
    return SourceEvidence(selection_receipt, replication_evidence)


def _holdout_quality_recipe(recipe: AETHGSHoldoutRecipe, base_recipe: Any) -> Any:
    holdout_spec = replace(
        base_recipe.dataset.holdout,
        split_id=recipe.dataset.holdout.split_id,
        num_instances=recipe.dataset.holdout.num_instances,
        seed=recipe.dataset.holdout.seed,
        artifact=recipe.dataset.holdout.artifact,
    )
    dataset = replace(
        base_recipe.dataset,
        problem=recipe.dataset.problem,
        size=recipe.dataset.size,
        capacity=recipe.dataset.capacity,
        max_demand=recipe.dataset.max_demand,
        selection=holdout_spec,
        holdout=holdout_spec,
    )
    return replace(
        base_recipe,
        dataset=dataset,
        reference=recipe.reference,
        host_id=recipe.host_id,
        accelerator_label=recipe.expected_accelerator_label,
        gpu_index=recipe.gpu_index,
    )


def _runtime_identity(recipe: AETHGSHoldoutRecipe, quality_recipe: Any) -> dict[str, Any]:
    try:
        import ortools
        import pyvrp
        import torch
    except Exception as exc:
        raise HGSHoldoutQualificationError("holdout runtime imports are unavailable") from exc
    if not torch.cuda.is_available() or recipe.gpu_index >= torch.cuda.device_count():
        raise HGSHoldoutQualificationError("holdout CUDA device is unavailable")
    name = str(torch.cuda.get_device_name(recipe.gpu_index))
    if name != recipe.expected_accelerator_label:
        raise HGSHoldoutQualificationError("holdout GPU identity changed")
    properties = torch.cuda.get_device_properties(recipe.gpu_index)
    return {
        "schema_version": "aet-hgs-holdout-runtime-identity/v1",
        "platform": sys.platform,
        "host_id": recipe.host_id,
        "python": sys.version.split()[0],
        "gpu_index": recipe.gpu_index,
        "gpu_name": name,
        "gpu_total_memory_b": int(properties.total_memory),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "libraries": {
            "numpy": _runtime_version(np.__version__),
            # torch.__version__ is a TorchVersion subclass, not a plain str.
            # Casting prevents a strict JSON type mismatch after state reload.
            "torch": _runtime_version(torch.__version__),
            "pyvrp": _runtime_version(getattr(pyvrp, "__version__", None)),
            "ortools": _runtime_version(getattr(ortools, "__version__", None)),
        },
        "mode": _mapping(
            next(
                mode
                for mode in quality_recipe.evaluation.modes
                if mode.mode_id == recipe.neural_policy.mode_id
            )
        ),
        "energy_measurement": "none",
        "preflight_used": False,
        "exclusive_attestation_used": False,
    }


def _runtime_version(value: Any) -> str | None:
    """Normalize package versions to the JSON type restored on resume."""

    if value is None:
        return None
    return str(value)


def _stable_runtime_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return strict v1 identity fields, excluding volatile Windows VRAM."""

    libraries = identity.get("libraries")
    capability = identity.get("gpu_compute_capability")
    mode = identity.get("mode")
    if (
        identity.get("schema_version") != "aet-hgs-holdout-runtime-identity/v1"
        or not isinstance(libraries, Mapping)
        or set(libraries) != {"numpy", "torch", "pyvrp", "ortools"}
        or any(not isinstance(libraries.get(name), str) for name in ("numpy", "torch", "ortools"))
        or not (libraries.get("pyvrp") is None or isinstance(libraries.get("pyvrp"), str))
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or not isinstance(mode, Mapping)
    ):
        raise HGSHoldoutQualificationError("holdout v1 runtime identity is malformed")
    return {
        "schema_version": identity.get("schema_version"),
        "platform": identity.get("platform"),
        "host_id": identity.get("host_id"),
        "python": identity.get("python"),
        "gpu_index": identity.get("gpu_index"),
        "gpu_name": identity.get("gpu_name"),
        "gpu_compute_capability": list(capability),
        "libraries": {
            "numpy": libraries["numpy"],
            "torch": libraries["torch"],
            "pyvrp": libraries["pyvrp"],
            "ortools": libraries["ortools"],
        },
        "mode": dict(mode),
        "energy_measurement": identity.get("energy_measurement"),
        "preflight_used": identity.get("preflight_used"),
        "exclusive_attestation_used": identity.get("exclusive_attestation_used"),
    }


def _git_snapshot(root: Path) -> dict[str, Any]:
    snapshot = quality._git_snapshot(root)
    sha = snapshot.get("sha")
    if not isinstance(sha, str) or not sha:
        raise HGSHoldoutQualificationError("Git commit is unavailable")
    return snapshot


def _selection_receipt_path(output: Path) -> Path:
    return output / "provenance" / "hgs-selection-receipt.json"


def _replication_receipt_path(output: Path) -> Path:
    return output / "provenance" / "quality-replication-receipt.json"


def _canonical_input_hash(payload: bytes) -> str:
    canonical, _mode = _canonical_text_bytes(payload)
    return _sha256_bytes(canonical)


def _prepare_output(
    recipe: AETHGSHoldoutRecipe,
    *,
    recipe_bytes: bytes,
    root: Path,
    evidence: SourceEvidence,
    runtime_identity: dict[str, Any],
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    output = quality._safe_output_target(root, recipe.output_root)
    recipe_sha256 = _canonical_input_hash(recipe_bytes)
    try:
        uv_lock_bytes = (root / "uv.lock").read_bytes()
    except OSError as exc:
        raise HGSHoldoutQualificationError("uv.lock is unavailable") from exc
    uv_lock_sha256 = _canonical_input_hash(uv_lock_bytes)
    git = _git_snapshot(root)
    source_snapshot = _canonical_source_snapshot(recipe, root)
    selection_receipt_sha256 = _sha256_bytes(quality._json_bytes(evidence.selection_receipt))
    replication_receipt_sha256 = _sha256_bytes(
        quality._json_bytes(evidence.replication_evidence.receipt)
    )
    schedule = [
        {
            "round": index,
            "seed": seed,
            "budget_order": list(recipe.hgs_policies.budget_orders[index]),
        }
        for index, seed in enumerate(recipe.hgs_policies.seeds)
    ]

    if output.exists():
        if not resume:
            raise HGSHoldoutError(f"holdout output already exists; use --resume: {output}")
        state_path = output / "run-state.json"
        if not state_path.is_file():
            raise HGSHoldoutQualificationError(
                "holdout output exists without its pre-opening run-state anchor"
            )
        state = quality._load_json(state_path)
        if state.get("status") in {INVALIDATED_STATUS, INVALIDATED_INPUT_STATUS}:
            raise HGSHoldoutError("holdout was permanently invalidated; use a new protocol")
        expected = {
            "schema_version": RUN_STATE_SCHEMA,
            "recipe_sha256": recipe_sha256,
            "uv_lock_sha256": uv_lock_sha256,
            "classification": _classification(recipe),
            "round_schedule": schedule,
            "selection_source_receipt_sha256": selection_receipt_sha256,
            "replication_source_receipt_sha256": replication_receipt_sha256,
        }
        if any(not _strict_json_equal(state.get(key), value) for key, value in expected.items()):
            raise HGSHoldoutQualificationError(
                "existing holdout was created from different frozen inputs"
            )
        if state.get("git_sha") != git.get("sha"):
            raise HGSHoldoutQualificationError("holdout Git commit changed since opening")
        if state.get("source_snapshot", {}).get("sha256") != source_snapshot.get("sha256"):
            raise HGSHoldoutQualificationError(
                "holdout canonical source snapshot changed since opening"
            )
        stored_runtime_identity = state.get("runtime_identity")
        if not isinstance(stored_runtime_identity, Mapping) or not _strict_json_equal(
            _stable_runtime_identity(stored_runtime_identity),
            _stable_runtime_identity(runtime_identity),
        ):
            raise HGSHoldoutQualificationError("holdout runtime identity changed")
        frozen = (
            (output / "recipe.yaml", recipe_sha256),
            (output / "environment" / "uv.lock", uv_lock_sha256),
        )
        for path, digest in frozen:
            if not path.is_file() or _canonical_file_sha256(path)[0] != digest:
                raise HGSHoldoutQualificationError(
                    f"frozen holdout input changed or disappeared: {path}"
                )
        if not _strict_json_equal(
            quality._load_json(_selection_receipt_path(output)),
            evidence.selection_receipt,
        ):
            raise HGSHoldoutQualificationError("selection source receipt changed")
        if not _strict_json_equal(
            quality._load_json(_replication_receipt_path(output)),
            evidence.replication_evidence.receipt,
        ):
            raise HGSHoldoutQualificationError("replication source receipt changed")
        if state.get("status") not in COMPLETE_STATUSES:
            quality._record_partial_cleanup(output, state)
        return output, state

    created_at = datetime.now(UTC).isoformat()
    state: dict[str, Any] = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": "initialized_before_holdout_open",
        "created_at": created_at,
        "recipe_sha256": recipe_sha256,
        "uv_lock_sha256": uv_lock_sha256,
        "git_sha": git.get("sha"),
        "git": git,
        "source_snapshot": source_snapshot,
        "runtime_identity": runtime_identity,
        "classification": _classification(recipe),
        "selection_source_receipt_sha256": selection_receipt_sha256,
        "replication_source_receipt_sha256": replication_receipt_sha256,
        "round_schedule": schedule,
        "holdout_opening_authorized_at": created_at,
        "holdout_opened_at": None,
        "holdout_file_sha256": None,
        "holdout_content_sha256": None,
        "reference_lock_sha256": None,
        "reference_locked_before_policy_evaluation": False,
        "neural_completed_seeds": [],
        "neural_artifact_sha256": {},
        "completed_cells": [],
        "completed_rounds": [],
        "remaining_rounds": list(range(len(recipe.hgs_policies.seeds))),
        "invocations": [],
        "resume_cleanups": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    if staging.exists():
        raise HGSHoldoutError("holdout initialization staging path already exists")
    try:
        (staging / "environment").mkdir(parents=True)
        (staging / "provenance").mkdir(parents=True)
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "environment" / "uv.lock").write_bytes(uv_lock_bytes)
        quality._atomic_write_json(
            staging / "provenance" / "hgs-selection-receipt.json",
            evidence.selection_receipt,
        )
        quality._atomic_write_json(
            staging / "provenance" / "quality-replication-receipt.json",
            evidence.replication_evidence.receipt,
        )
        quality._atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _forbidden_content_hashes(recipe: AETHGSHoldoutRecipe) -> tuple[str, ...]:
    return (
        recipe.dataset.forbidden_selection_content_sha256,
        *recipe.dataset.additional_forbidden_content_sha256,
    )


def _prepare_holdout_corpus(
    recipe: AETHGSHoldoutRecipe,
    quality_recipe: Any,
    output: Path,
    state: dict[str, Any],
) -> quality.Corpus:
    if not isinstance(state.get("holdout_opening_authorized_at"), str):
        raise HGSHoldoutError("holdout generation lacks a durable pre-opening authorization")
    spec = recipe.dataset.holdout
    quality_spec = quality_recipe.dataset.holdout
    path = _relative(output, spec.artifact)
    if path.exists():
        corpus = quality._load_corpus(quality_recipe, "holdout", quality_spec, path)
    else:
        if state.get("holdout_opened_at") is not None:
            raise HGSHoldoutError("anchored holdout corpus disappeared")
        corpus = quality._generate_corpus(quality_recipe, "holdout", quality_spec, path)
    if corpus.content_sha256 in _forbidden_content_hashes(recipe):
        raise HGSHoldoutError("holdout content unexpectedly matches a previously opened corpus")
    file_hash = corpus.file_sha256
    content_hash = corpus.content_sha256
    if state.get("holdout_opened_at") is None:
        state["holdout_opened_at"] = datetime.now(UTC).isoformat()
        state["holdout_file_sha256"] = file_hash
        state["holdout_content_sha256"] = content_hash
        state["status"] = INCOMPLETE_STATUS
        quality._atomic_write_json(output / "run-state.json", state)
    elif (
        state.get("holdout_file_sha256") != file_hash
        or state.get("holdout_content_sha256") != content_hash
    ):
        raise HGSHoldoutError("anchored holdout corpus changed")
    return corpus


def _source_binding(state: Mapping[str, Any]) -> dict[str, str]:
    """Return the immutable source identity embedded in every reusable work unit."""

    snapshot = state.get("source_snapshot")
    values = {
        "git_sha": state.get("git_sha"),
        "source_snapshot_sha256": (
            snapshot.get("sha256") if isinstance(snapshot, Mapping) else None
        ),
        "recipe_sha256": state.get("recipe_sha256"),
        "uv_lock_sha256": state.get("uv_lock_sha256"),
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise HGSHoldoutError("holdout source binding is incomplete")
    return {key: str(value) for key, value in values.items()}


def _validate_reference_candidate(
    payload: dict[str, Any],
    *,
    recipe: AETHGSHoldoutRecipe,
    state: Mapping[str, Any],
    corpus: quality.Corpus,
    path: Path,
    family: str,
    solver: str,
    seed: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version": quality.REFERENCE_CANDIDATE_SCHEMA,
        "dataset_content_sha256": corpus.content_sha256,
        "family": family,
        "solver": solver,
        "seed": seed,
        "settings": settings,
        "source_binding": _source_binding(state),
    }
    if any(not _strict_json_equal(payload.get(key), value) for key, value in expected.items()):
        raise HGSHoldoutError(f"holdout reference candidate changed: {path}")
    try:
        quality._revalidate_stored_routes(payload, corpus, str(path))
        quality._validate_candidate_result_metadata(
            payload,
            corpus,
            base_seed=seed,
            label=str(path),
        )
    except Exception as exc:
        raise HGSHoldoutError(f"holdout reference candidate is invalid: {path}") from exc
    records = payload.get("solver_results")
    if not isinstance(records, list):
        raise HGSHoldoutError(f"holdout reference candidate inventory changed: {path}")
    if family == "hybrid-genetic-search":
        if any(
            record.get("max_iterations") != recipe.reference.hgs.max_iterations
            or record.get("search_status") != "feasible_complete"
            or record.get("scaling_factor") != 1_000_000
            for record in records
            if isinstance(record, dict)
        ) or any(not isinstance(record, dict) for record in records):
            raise HGSHoldoutError(f"holdout HGS reference metadata changed: {path}")
    else:
        if any(
            record.get("limit_kind") != "solutions"
            or record.get("solution_limit") != recipe.reference.ortools.solution_limit
            or record.get("max_runtime_s") is not None
            or not isinstance(record.get("search_status"), str)
            for record in records
            if isinstance(record, dict)
        ) or any(not isinstance(record, dict) for record in records):
            raise HGSHoldoutError(f"holdout OR-Tools reference metadata changed: {path}")
    return payload


def _reference_candidate(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    family: str,
    solver: str,
    seed: int,
    settings: dict[str, Any],
    filename: str,
    revalidate: WorkUnitRevalidator | None,
    allow_create: bool = True,
) -> dict[str, Any]:
    path = output / "reference" / corpus.split / "candidates" / filename
    work_unit = f"reference_{solver}_seed_{seed}"
    if path.exists():
        if revalidate is not None:
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="before_cache_reuse",
            )
        payload = _validate_reference_candidate(
            quality._load_json(path),
            recipe=recipe,
            state=state,
            corpus=corpus,
            path=path,
            family=family,
            solver=solver,
            seed=seed,
            settings=settings,
        )
        if revalidate is not None:
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="after_cache_reuse",
                promoted_path=path,
            )
        return payload
    if revalidate is None or not allow_create:
        raise HGSHoldoutError(f"completed holdout reference candidate is missing: {path}")

    _guard_work_unit_inputs(
        revalidate,
        output=output,
        state=state,
        work_unit=work_unit,
        phase="before_compute",
    )
    started = time.perf_counter()
    if family == "hybrid-genetic-search":
        from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

        results = solve_corpus_sequential(
            corpus.coords,
            corpus.demands,
            corpus.capacity,
            seed=seed,
            max_iterations=recipe.reference.hgs.max_iterations,
            collect_stats=False,
        )
    else:
        from neuro_co.problems.cvrp.ortools import solve_corpus_sequential

        results = solve_corpus_sequential(
            corpus.coords,
            corpus.demands,
            corpus.capacity,
            seed=seed,
            solution_limit=recipe.reference.ortools.solution_limit,
            scaling_factor=recipe.reference.ortools.scaling_factor,
        )
    payload = quality._candidate_payload(
        corpus=corpus,
        family=family,
        solver=solver,
        seed=seed,
        settings=settings,
        results=results,
        elapsed_s=time.perf_counter() - started,
    )
    payload["source_binding"] = _source_binding(state)
    _promote_work_unit_json(
        path,
        payload,
        output=output,
        state=state,
        work_unit=work_unit,
        revalidate=revalidate,
    )
    return _validate_reference_candidate(
        quality._load_json(path),
        recipe=recipe,
        state=state,
        corpus=corpus,
        path=path,
        family=family,
        solver=solver,
        seed=seed,
        settings=settings,
    )


def _prepare_reference_candidates(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    revalidate: WorkUnitRevalidator | None,
    allow_create: bool = True,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seed in recipe.reference.hgs.seeds:
        candidates.append(
            _reference_candidate(
                recipe,
                output=output,
                state=state,
                corpus=corpus,
                family="hybrid-genetic-search",
                solver=recipe.reference.hgs.solver,
                seed=seed,
                settings={"max_iterations": recipe.reference.hgs.max_iterations},
                filename=f"pyvrp-hgs-seed{seed}.json",
                revalidate=revalidate,
                allow_create=allow_create,
            )
        )
    ortools_seed = recipe.reference.ortools.seed
    candidates.append(
        _reference_candidate(
            recipe,
            output=output,
            state=state,
            corpus=corpus,
            family="ortools-routing",
            solver=recipe.reference.ortools.solver,
            seed=ortools_seed,
            settings={
                "solution_limit": recipe.reference.ortools.solution_limit,
                "scaling_factor": recipe.reference.ortools.scaling_factor,
                "search": "parallel-cheapest-insertion-plus-guided-local-search",
                "num_search_workers": 1,
                "timing_sensitivity": False,
            },
            filename=f"ortools-routing-gls-seed{ortools_seed}.json",
            revalidate=revalidate,
            allow_create=allow_create,
        )
    )
    return candidates


def _reference_payload(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: Mapping[str, Any],
    corpus: quality.Corpus,
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    candidate_artifacts: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["solver"] == recipe.reference.hgs.solver:
            filename = f"pyvrp-hgs-seed{candidate['seed']}.json"
        elif candidate["solver"] == recipe.reference.ortools.solver:
            filename = f"ortools-routing-gls-seed{candidate['seed']}.json"
        else:
            raise HGSHoldoutError("unexpected solver in holdout reference candidates")
        candidate_path = output / "reference" / corpus.split / "candidates" / filename
        candidate_artifacts.append(
            {
                "path": candidate_path.relative_to(output).as_posix(),
                "sha256": quality._sha256_file(candidate_path),
                "family": candidate["family"],
                "solver": candidate["solver"],
                "base_seed": candidate["seed"],
                "settings": candidate["settings"],
            }
        )
    candidate_costs = [candidate["validation"]["costs"] for candidate in candidates]
    routes: list[Any] = []
    costs: list[float] = []
    sources: list[dict[str, Any]] = []
    for instance_index in range(corpus.coords.shape[0]):
        best_index = min(
            range(len(candidates)),
            key=lambda index: (float(candidate_costs[index][instance_index]), index),
        )
        candidate = candidates[best_index]
        solver_result = candidate["solver_results"][instance_index]
        routes.append(candidate["routes"][instance_index])
        costs.append(float(candidate_costs[best_index][instance_index]))
        sources.append(
            {
                "family": candidate["family"],
                "solver": candidate["solver"],
                "base_seed": candidate["seed"],
                "effective_seed": solver_result["effective_seed"],
                "integer_cost": solver_result["integer_cost"],
                "reported_cost": solver_result["reported_cost"],
                "candidate_sha256": candidate_artifacts[best_index]["sha256"],
            }
        )
    validation = quality.validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        routes,
    )
    quality._assert_valid(validation, f"combined {corpus.split} reference")
    if not np.allclose(validation["costs"], costs, rtol=0.0, atol=1e-10):
        raise HGSHoldoutError("combined holdout reference costs changed")
    source_counts: dict[str, int] = {}
    for source in sources:
        key = f"{source['solver']}:seed{source['base_seed']}"
        source_counts[key] = source_counts.get(key, 0) + 1
    return {
        "schema_version": quality.REFERENCE_SCHEMA,
        "status": "complete",
        "split": corpus.split,
        "dataset_content_sha256": corpus.content_sha256,
        "policy": recipe.reference.policy,
        "independent_solver_families": sorted(
            {str(candidate["family"]) for candidate in candidates}
        ),
        "candidate_count": len(candidates),
        "candidate_artifacts": candidate_artifacts,
        "source_counts": source_counts,
        "sources": sources,
        "routes": routes,
        "costs": costs,
        "mean_cost": float(np.mean(costs)),
        "validation": validation,
        "future_timed_baseline_is_separate": True,
        "energy_measurement": "none",
        "source_binding": _source_binding(state),
    }


def _validate_reference(
    payload: dict[str, Any],
    expected: dict[str, Any],
    *,
    corpus: quality.Corpus,
    path: Path,
) -> dict[str, Any]:
    if not _strict_json_equal(payload, expected):
        raise HGSHoldoutError(f"locked holdout reference changed: {path}")
    validation = quality.validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        payload.get("routes", ()),
    )
    quality._assert_valid(validation, f"locked holdout reference {path}")
    if not _strict_json_equal(payload.get("validation"), validation):
        raise HGSHoldoutError("locked holdout reference validation changed")
    return payload


def _reference_lock_fields(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: Mapping[str, Any],
    corpus: quality.Corpus,
    reference: Mapping[str, Any],
    reference_path: Path,
) -> dict[str, Any]:
    candidates = reference.get("candidate_artifacts")
    if not isinstance(candidates, list):
        raise HGSHoldoutError("holdout reference candidate inventory is invalid")
    return {
        "schema_version": REFERENCE_LOCK_SCHEMA,
        "status": "locked",
        "locked_before_policy_evaluation": True,
        "policy": recipe.reference.policy,
        "dataset": {
            "path": corpus.path.relative_to(output).as_posix(),
            "file_sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
        },
        "reference": {
            "path": "reference/holdout/reference.json",
            "sha256": quality._sha256_file(reference_path),
        },
        "candidate_artifacts": candidates,
        "energy_measurement": "none",
        "source_binding": _source_binding(state),
    }


def _validate_reference_lock(
    payload: dict[str, Any],
    *,
    expected_fields: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    expected_keys = {*expected_fields, "locked_at"}
    locked_at = payload.get("locked_at")
    if set(payload) != expected_keys or not isinstance(locked_at, str):
        raise HGSHoldoutError(f"holdout reference lock schema changed: {path}")
    try:
        parsed_locked_at = datetime.fromisoformat(locked_at)
    except ValueError as exc:
        raise HGSHoldoutError(f"holdout reference lock timestamp changed: {path}") from exc
    if parsed_locked_at.tzinfo is None or any(
        not _strict_json_equal(payload.get(key), value) for key, value in expected_fields.items()
    ):
        raise HGSHoldoutError(f"holdout reference lock relationships changed: {path}")
    return payload


def _prepare_reference(
    recipe: AETHGSHoldoutRecipe,
    quality_recipe: Any,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    *,
    revalidate: WorkUnitRevalidator | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del quality_recipe
    lock_path = output / "reference" / "reference-lock.json"
    reference_path = output / "reference" / "holdout" / "reference.json"
    anchored = state.get("reference_lock_sha256")
    recovering_unanchored_lock = anchored is None and lock_path.exists()
    candidates = _prepare_reference_candidates(
        recipe,
        output=output,
        state=state,
        corpus=corpus,
        revalidate=revalidate,
        allow_create=not recovering_unanchored_lock,
    )
    expected_reference = _reference_payload(
        recipe,
        output=output,
        state=state,
        corpus=corpus,
        candidates=candidates,
    )
    if anchored is None:
        if revalidate is None:
            raise HGSHoldoutError("completed holdout reference lock is missing")
        if (output / "neural").exists() or (output / "hgs").exists():
            raise HGSHoldoutError("policy artifacts exist without a frozen holdout reference")
        work_unit = "reference_combination"
        if reference_path.exists():
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="before_cache_reuse",
            )
            reference = _validate_reference(
                quality._load_json(reference_path),
                expected_reference,
                corpus=corpus,
                path=reference_path,
            )
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="after_cache_reuse",
                promoted_path=reference_path,
            )
        else:
            if recovering_unanchored_lock:
                raise HGSHoldoutError(
                    "unanchored reference lock survived without its referenced artifact"
                )
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="before_compute",
            )
            _promote_work_unit_json(
                reference_path,
                expected_reference,
                output=output,
                state=state,
                work_unit=work_unit,
                revalidate=revalidate,
            )
            reference = _validate_reference(
                quality._load_json(reference_path),
                expected_reference,
                corpus=corpus,
                path=reference_path,
            )
        lock_fields = _reference_lock_fields(
            recipe,
            output=output,
            state=state,
            corpus=corpus,
            reference=reference,
            reference_path=reference_path,
        )
        if lock_path.exists():
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit="reference_lock",
                phase="before_cache_reuse",
            )
            try:
                stored_lock = quality._load_json(lock_path)
            except Exception as exc:
                raise HGSHoldoutError("unanchored holdout reference lock is unreadable") from exc
            lock = _validate_reference_lock(
                stored_lock,
                expected_fields=lock_fields,
                path=lock_path,
            )
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit="reference_lock",
                phase="after_cache_reuse",
                promoted_path=lock_path,
            )
            state.setdefault("resume_cleanups", []).append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "action": "anchored_atomic_reference_lock_after_interrupted_state_update",
                    "sha256": quality._sha256_file(lock_path),
                }
            )
        else:
            lock = {
                **lock_fields,
                "locked_at": datetime.now(UTC).isoformat(),
            }
            _promote_work_unit_json(
                lock_path,
                lock,
                output=output,
                state=state,
                work_unit="reference_lock",
                revalidate=revalidate,
            )
        state["reference_lock_sha256"] = quality._sha256_file(lock_path)
        state["reference_locked_before_policy_evaluation"] = True
        quality._atomic_write_json(output / "run-state.json", state)
        return reference, lock

    if not isinstance(anchored, str) or not quality._file_matches_sha256(lock_path, anchored):
        raise HGSHoldoutError("anchored holdout reference lock changed")
    lock_fields = _reference_lock_fields(
        recipe,
        output=output,
        state=state,
        corpus=corpus,
        reference=expected_reference,
        reference_path=reference_path,
    )
    try:
        stored_lock = quality._load_json(lock_path)
    except Exception as exc:
        raise HGSHoldoutError("anchored holdout reference lock is unreadable") from exc
    lock = _validate_reference_lock(
        stored_lock,
        expected_fields=lock_fields,
        path=lock_path,
    )
    reference_entry = lock["reference"]
    if revalidate is not None:
        _guard_work_unit_inputs(
            revalidate,
            output=output,
            state=state,
            work_unit="reference_combination",
            phase="before_cache_reuse",
        )
    reference = _validate_reference(
        quality._load_json(reference_path),
        expected_reference,
        corpus=corpus,
        path=reference_path,
    )
    if revalidate is not None:
        _guard_work_unit_inputs(
            revalidate,
            output=output,
            state=state,
            work_unit="reference_combination",
            phase="after_cache_reuse",
            promoted_path=reference_path,
        )
    if quality._sha256_file(reference_path) != reference_entry["sha256"]:
        raise HGSHoldoutError("anchored holdout reference semantics changed")
    if not _strict_json_equal(
        lock.get("candidate_artifacts"), reference.get("candidate_artifacts")
    ):
        raise HGSHoldoutError("anchored reference candidate inventory changed")
    return reference, lock


def _neural_path(seed: int) -> str:
    return f"neural/pomo-50x8-seed-{seed:03d}.json"


def _neural_evaluation_seed(recipe: AETHGSHoldoutRecipe, training_seed: int) -> int:
    pairs = dict(
        zip(
            recipe.neural_policy.training_seeds,
            recipe.neural_policy.evaluation_seeds,
            strict=True,
        )
    )
    try:
        return int(pairs[training_seed])
    except KeyError as exc:
        raise HGSHoldoutError(
            f"missing neural evaluation seed for training seed {training_seed}"
        ) from exc


def _validate_neural_evaluation(
    payload: dict[str, Any],
    *,
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    evidence: frontier.ReplicationEvidence,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    state: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    checkpoint = evidence.checkpoints[seed]
    expected = {
        "schema_version": NEURAL_EVALUATION_SCHEMA,
        "classification": {
            "purpose": _classification(recipe)["purpose"],
            "policy_role": (
                "neural_confirmatory" if _is_confirmatory_recipe(recipe) else "neural_primary"
            ),
            "energy_measurement": "none",
            "timing_scientific_use": False,
        },
        "split": "holdout",
        "training_seed": seed,
        "checkpoint_epoch": recipe.neural_policy.checkpoint_epoch,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "mode": _mapping(evidence.primary_mode),
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "selection_source_receipt_sha256": quality._sha256_file(_selection_receipt_path(output)),
        "replication_source_receipt_sha256": quality._sha256_file(
            _replication_receipt_path(output)
        ),
        "evaluation_seed": _neural_evaluation_seed(recipe, seed),
        "energy_measurement": "none",
        "source_binding": _source_binding(state),
    }
    if any(not _strict_json_equal(payload.get(key), value) for key, value in expected.items()):
        raise HGSHoldoutError(f"cached neural holdout evaluation changed for seed {seed}")
    validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, payload.get("routes", ())
    )
    if not _strict_json_equal(payload.get("validation"), validation):
        raise HGSHoldoutError(f"stored neural route validation changed for seed {seed}")
    summary = quality._gap_summary(validation["costs"], reference["costs"])
    if not _strict_json_equal(payload.get("quality"), summary):
        raise HGSHoldoutError(f"stored neural quality changed for seed {seed}")
    return payload


def _prepare_neural_evaluations(
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    state: dict[str, Any],
    evidence: frontier.ReplicationEvidence,
    quality_recipe: Any,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    *,
    revalidate: WorkUnitRevalidator | None = None,
) -> dict[int, dict[str, Any]]:
    completed = state.get("neural_completed_seeds")
    if not isinstance(completed, list):
        raise HGSHoldoutError("neural completion state is invalid")
    expected_seeds = tuple(recipe.neural_policy.training_seeds)
    if tuple(completed) != expected_seeds[: len(completed)] or len(set(completed)) != len(
        completed
    ):
        raise HGSHoldoutError("neural completion state is not the scheduled seed prefix")
    artifact_sha256 = state.get("neural_artifact_sha256")
    if not isinstance(artifact_sha256, dict):
        raise HGSHoldoutError("neural artifact hash state is invalid")
    if set(artifact_sha256) != {str(seed) for seed in completed}:
        raise HGSHoldoutError("neural completion and artifact hash state disagree")
    evaluations: dict[int, dict[str, Any]] = {}
    for seed in expected_seeds:
        path = output / _neural_path(seed)
        work_unit = f"neural_seed_{seed}"
        if path.exists():
            if revalidate is not None:
                _guard_work_unit_inputs(
                    revalidate,
                    output=output,
                    state=state,
                    work_unit=work_unit,
                    phase="before_cache_reuse",
                )
            try:
                payload = quality._load_json(path)
            except Exception as exc:
                raise HGSHoldoutError(
                    f"neural holdout artifact is unreadable or corrupt for seed {seed}"
                ) from exc
        else:
            if seed in completed:
                raise HGSHoldoutError(
                    f"declared-complete neural holdout artifact is missing for seed {seed}"
                )
            if revalidate is None:
                raise HGSHoldoutError(
                    f"completed holdout neural artifact is missing for seed {seed}"
                )
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="before_compute",
            )
            checkpoint = evidence.checkpoints[seed]
            checkpoint_path = _relative(evidence.root, checkpoint["path"])
            seed_recipe = replication._seed_recipe(
                quality_recipe, evidence.replication_recipe, seed
            )
            routes = quality._evaluate_routes(
                seed_recipe,
                checkpoint_path,
                corpus,
                evidence.primary_mode,
                eval_seed=_neural_evaluation_seed(recipe, seed),
            )
            validation = quality.validate_routes(
                corpus.coords, corpus.demands, corpus.capacity, routes
            )
            payload = {
                "schema_version": NEURAL_EVALUATION_SCHEMA,
                "classification": {
                    "purpose": _classification(recipe)["purpose"],
                    "policy_role": (
                        "neural_confirmatory"
                        if _is_confirmatory_recipe(recipe)
                        else "neural_primary"
                    ),
                    "energy_measurement": "none",
                    "timing_scientific_use": False,
                },
                "split": "holdout",
                "training_seed": seed,
                "checkpoint_epoch": recipe.neural_policy.checkpoint_epoch,
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "mode": _mapping(evidence.primary_mode),
                "dataset_content_sha256": corpus.content_sha256,
                "reference_sha256": reference_sha256,
                "selection_source_receipt_sha256": quality._sha256_file(
                    _selection_receipt_path(output)
                ),
                "replication_source_receipt_sha256": quality._sha256_file(
                    _replication_receipt_path(output)
                ),
                "evaluation_seed": _neural_evaluation_seed(recipe, seed),
                "energy_measurement": "none",
                "source_binding": _source_binding(state),
                "routes": routes,
                "validation": validation,
                "quality": quality._gap_summary(validation["costs"], reference["costs"]),
            }
            _promote_work_unit_json(
                path,
                payload,
                output=output,
                state=state,
                work_unit=work_unit,
                revalidate=revalidate,
            )
        evaluations[seed] = _validate_neural_evaluation(
            payload,
            recipe=recipe,
            output=output,
            evidence=evidence,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            state=state,
            seed=seed,
        )
        digest = quality._sha256_file(path)
        if seed in completed:
            if artifact_sha256.get(str(seed)) != digest:
                raise HGSHoldoutError(
                    f"declared-complete neural holdout artifact changed for seed {seed}"
                )
        else:
            if str(seed) in artifact_sha256:
                raise HGSHoldoutError("undeclared neural artifact has a pre-existing state hash")
            completed.append(seed)
            artifact_sha256[str(seed)] = digest
            quality._atomic_write_json(output / "run-state.json", state)
        if revalidate is not None:
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="after_cache_validation",
                promoted_path=path,
            )
    if tuple(completed) != expected_seeds:
        raise HGSHoldoutError("all five frozen neural checkpoints must be evaluated")
    return evaluations


def _cell_relative(round_index: int, budget: int, seed: int) -> str:
    return f"hgs/round-{round_index:02d}/budget-{budget:03d}-seed-{seed}.json"


def _expected_cells(
    recipe: AETHGSHoldoutRecipe,
) -> tuple[tuple[int, int, int, int, str], ...]:
    records: list[tuple[int, int, int, int, str]] = []
    for round_index, seed in enumerate(recipe.hgs_policies.seeds):
        order = recipe.hgs_policies.budget_orders[round_index]
        for order_index, budget in enumerate(order):
            records.append(
                (
                    round_index,
                    order_index,
                    seed,
                    budget,
                    _cell_relative(round_index, budget, seed),
                )
            )
    return tuple(records)


def _solver_records(results: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "instance_index": int(frontier._attribute(result, "instance_index")),
            "effective_seed": int(frontier._attribute(result, "seed")),
            "integer_cost": int(frontier._attribute(result, "integer_cost")),
            "cost": float(frontier._attribute(result, "cost")),
            "max_iterations": int(frontier._attribute(result, "max_iterations")),
            "scaling_factor": int(frontier._attribute(result, "scaling_factor")),
        }
        for result in results
    ]


def _execute_hgs_quality_cell(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: Mapping[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    round_index: int,
    order_index: int,
    seed: int,
    budget: int,
    invocation_number: int,
) -> dict[str, Any]:
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

    started_at = datetime.now(UTC).isoformat()
    try:
        results = solve_corpus_sequential(
            corpus.coords,
            corpus.demands,
            corpus.capacity,
            seed=seed,
            max_iterations=budget,
            scaling_factor=recipe.hgs_policies.scaling_factor,
            collect_stats=recipe.hgs_policies.collect_stats,
        )
    except RuntimeError as exc:
        match = _PYVRP_INCOMPLETE_SOLUTION.fullmatch(str(exc))
        if match is None:
            raise
        raise HGSHoldoutIntegrityError(
            "PyVRP did not return a complete feasible holdout solution",
            details={
                "round": round_index,
                "budget": budget,
                "seed": seed,
                "instance_index": int(match.group(1)),
                "pyvrp_error": str(exc),
            },
        ) from exc
    try:
        routes = frontier._validate_solver_results(results, corpus=corpus, seed=seed, budget=budget)
    except (frontier.HGSFrontierError, TypeError, ValueError, OverflowError) as exc:
        raise HGSHoldoutIntegrityError(
            "HGS holdout result metadata, cardinality, routes, or cost violated its contract",
            details={
                "round": round_index,
                "budget": budget,
                "seed": seed,
                "validation_error_type": type(exc).__name__,
                "validation_error": str(exc),
            },
        ) from exc
    validation = quality.validate_routes(corpus.coords, corpus.demands, corpus.capacity, routes)
    if validation.get("complete") is not True:
        raise HGSHoldoutIntegrityError(
            "HGS returned an invalid route on the frozen holdout",
            details={
                "round": round_index,
                "budget": budget,
                "seed": seed,
                "validation": validation,
            },
        )
    records = _solver_records(results)
    role = (
        "confirmatory"
        if _is_confirmatory_recipe(recipe)
        else "primary"
        if budget == recipe.hgs_policies.primary_budget
        else "sensitivity"
    )
    return {
        "schema_version": HGS_CELL_SCHEMA,
        "classification": {
            "purpose": _classification(recipe)["purpose"],
            "policy_role": role,
            "energy_measurement": "none",
            "timing_scientific_use": False,
        },
        "status": "complete",
        "split": "holdout",
        "round": round_index,
        "order_within_round": order_index,
        "base_seed": seed,
        "budget": budget,
        "budget_kind": recipe.hgs_policies.budget_kind,
        "solver": recipe.hgs_policies.solver,
        "scaling_factor": recipe.hgs_policies.scaling_factor,
        "collect_stats": recipe.hgs_policies.collect_stats,
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "recipe_sha256": _canonical_file_sha256(output / "recipe.yaml")[0],
        "selection_source_receipt_sha256": quality._sha256_file(_selection_receipt_path(output)),
        "replication_source_receipt_sha256": quality._sha256_file(
            _replication_receipt_path(output)
        ),
        "source_binding": _source_binding(state),
        "invocation_number": invocation_number,
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "timing_scientific_use": False,
        "energy_measurement": "none",
        "carbon_accounting": "none",
        "items_processed": int(corpus.coords.shape[0]),
        "routes": routes,
        "solver_results": records,
        "validation": validation,
        "quality": quality._gap_summary(validation["costs"], reference["costs"]),
    }


def _validate_hgs_cell(
    payload: dict[str, Any],
    *,
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    state: Mapping[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    round_index: int,
    order_index: int,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    role = (
        "confirmatory"
        if _is_confirmatory_recipe(recipe)
        else "primary"
        if budget == recipe.hgs_policies.primary_budget
        else "sensitivity"
    )
    expected = {
        "schema_version": HGS_CELL_SCHEMA,
        "classification": {
            "purpose": _classification(recipe)["purpose"],
            "policy_role": role,
            "energy_measurement": "none",
            "timing_scientific_use": False,
        },
        "status": "complete",
        "split": "holdout",
        "round": round_index,
        "order_within_round": order_index,
        "base_seed": seed,
        "budget": budget,
        "budget_kind": recipe.hgs_policies.budget_kind,
        "solver": recipe.hgs_policies.solver,
        "scaling_factor": recipe.hgs_policies.scaling_factor,
        "collect_stats": recipe.hgs_policies.collect_stats,
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "recipe_sha256": _canonical_file_sha256(output / "recipe.yaml")[0],
        "selection_source_receipt_sha256": quality._sha256_file(_selection_receipt_path(output)),
        "replication_source_receipt_sha256": quality._sha256_file(
            _replication_receipt_path(output)
        ),
        "source_binding": _source_binding(state),
        "timing_scientific_use": False,
        "energy_measurement": "none",
        "carbon_accounting": "none",
        "items_processed": int(corpus.coords.shape[0]),
    }
    if any(not _strict_json_equal(payload.get(key), value) for key, value in expected.items()):
        raise HGSHoldoutError(
            f"cached HGS holdout cell changed for round {round_index}, budget {budget}"
        )
    invocation_number = payload.get("invocation_number")
    if (
        isinstance(invocation_number, bool)
        or not isinstance(invocation_number, int)
        or invocation_number < 1
        or not isinstance(payload.get("started_at"), str)
        or not isinstance(payload.get("ended_at"), str)
    ):
        raise HGSHoldoutError("cached HGS holdout invocation metadata changed")
    validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, payload.get("routes", ())
    )
    quality._assert_valid(validation, f"cached HGS holdout b{budget} seed {seed}")
    if not _strict_json_equal(payload.get("validation"), validation):
        raise HGSHoldoutError("stored HGS holdout route validation changed")
    summary = quality._gap_summary(validation["costs"], reference["costs"])
    if not _strict_json_equal(payload.get("quality"), summary):
        raise HGSHoldoutError("stored HGS holdout quality changed")
    records = payload.get("solver_results")
    if not isinstance(records, list) or len(records) != int(corpus.coords.shape[0]):
        raise HGSHoldoutError("stored HGS holdout solver-result inventory changed")
    for instance, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "instance_index",
                "effective_seed",
                "integer_cost",
                "cost",
                "max_iterations",
                "scaling_factor",
            }
            or record.get("instance_index") != instance
            or record.get("effective_seed") != seed + instance
            or record.get("max_iterations") != budget
            or record.get("scaling_factor") != recipe.hgs_policies.scaling_factor
        ):
            raise HGSHoldoutError("stored HGS holdout solver metadata changed")
        integer_cost = record.get("integer_cost")
        reported_cost = record.get("cost")
        if (
            isinstance(integer_cost, bool)
            or not isinstance(integer_cost, int)
            or isinstance(reported_cost, bool)
            or not isinstance(reported_cost, (int, float))
            or not math.isfinite(float(reported_cost))
            or not math.isclose(
                float(reported_cost),
                integer_cost / recipe.hgs_policies.scaling_factor,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise HGSHoldoutError("stored HGS holdout objective metadata changed")
    forbidden = {"energy", "energy_j", "energy_cpu_j", "energy_gpu_j", "co2", "carbon"}
    if forbidden & set(payload):
        raise HGSHoldoutError("quality-only HGS cell contains forbidden energy or carbon fields")
    return payload


def _load_cells(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    revalidate: WorkUnitRevalidator | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    expected = {
        relative: (round_index, order_index, seed, budget)
        for round_index, order_index, seed, budget, relative in _expected_cells(recipe)
    }
    hgs_root = output / "hgs"
    if hgs_root.exists():
        actual = {
            path.relative_to(output).as_posix()
            for path in hgs_root.rglob("*.json")
            if path.is_file()
        }
        unknown = sorted(actual - set(expected))
        if unknown:
            raise HGSHoldoutError(f"unexpected HGS holdout cells: {unknown}")
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for relative, (round_index, order_index, seed, budget) in expected.items():
        path = output / relative
        if not path.exists():
            continue
        work_unit = f"hgs_round_{round_index:02d}_budget_{budget:03d}_seed_{seed}"
        if revalidate is not None:
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="before_cache_reuse",
            )
        payload = _validate_hgs_cell(
            quality._load_json(path),
            recipe=recipe,
            output=output,
            state=state,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            round_index=round_index,
            order_index=order_index,
            seed=seed,
            budget=budget,
        )
        if revalidate is not None:
            _guard_work_unit_inputs(
                revalidate,
                output=output,
                state=state,
                work_unit=work_unit,
                phase="after_cache_reuse",
                promoted_path=path,
            )
        cells[(round_index, budget)] = payload
    return cells


def _progress(
    recipe: AETHGSHoldoutRecipe,
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> tuple[list[str], list[int], list[int]]:
    paths: list[str] = []
    completed_rounds: list[int] = []
    remaining: list[int] = []
    incomplete_seen = False
    for round_index, seed in enumerate(recipe.hgs_policies.seeds):
        order = recipe.hgs_policies.budget_orders[round_index]
        present = [budget for budget in order if (round_index, budget) in cells]
        if present != list(order[: len(present)]):
            raise HGSHoldoutError("HGS holdout cell sequence contains a non-prefix partial round")
        if incomplete_seen and present:
            raise HGSHoldoutError("HGS holdout cells exist after the first incomplete round")
        paths.extend(_cell_relative(round_index, budget, seed) for budget in present)
        if len(present) == len(order):
            completed_rounds.append(round_index)
        else:
            incomplete_seen = True
            remaining.append(round_index)
    return paths, completed_rounds, remaining


def _reconcile_progress(
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    state: dict[str, Any],
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> tuple[list[int], list[int]]:
    paths, completed, remaining = _progress(recipe, cells)
    changed = (
        state.get("completed_cells") != paths
        or state.get("completed_rounds") != completed
        or state.get("remaining_rounds") != remaining
    )
    state["completed_cells"] = paths
    state["completed_rounds"] = completed
    state["remaining_rounds"] = remaining
    if changed:
        state.setdefault("resume_cleanups", []).append(
            {
                "at": datetime.now(UTC).isoformat(),
                "action": "reconciled_atomic_cell_files_with_run_state",
            }
        )
        quality._atomic_write_json(output / "run-state.json", state)
    return completed, remaining


def _start_invocation(
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    state: dict[str, Any],
) -> int:
    invocations = state.get("invocations")
    if not isinstance(invocations, list):
        raise HGSHoldoutError("holdout invocation history is invalid")
    _validate_invocation_history(recipe, state, require_closed=False)
    if invocations and invocations[-1].get("ended_at") is None:
        active = invocations[-1]
        expected_cell_count = len(recipe.hgs_policies.budget_orders[0])
        if len(active["completed_cells_added"]) == expected_cell_count:
            raise HGSHoldoutError(
                "a recovered complete HGS round must be closed before another invocation"
            )
        active.setdefault("resume_events", []).append(
            {
                "at": datetime.now(UTC).isoformat(),
                "reason": "continue_interrupted_paired_round",
            }
        )
        quality._atomic_write_json(output / "run-state.json", state)
        return int(active["invocation_number"])
    number = len(invocations) + 1
    invocations.append(
        {
            "invocation_number": number,
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": None,
            "outcome": "running",
            "completed_cells_added": [],
            "energy_measurement": "none",
            "preflight_used": False,
            "exclusive_attestation_used": False,
        }
    )
    quality._atomic_write_json(output / "run-state.json", state)
    return number


def _finish_invocation(
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    state: dict[str, Any],
    outcome: str,
) -> None:
    _validate_invocation_history(recipe, state, require_closed=False)
    invocations = state.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise HGSHoldoutError("holdout invocation history is empty")
    active = invocations[-1]
    if active.get("ended_at") is not None:
        raise HGSHoldoutError("holdout invocation is already closed")
    active["ended_at"] = datetime.now(UTC).isoformat()
    active["outcome"] = outcome
    quality._atomic_write_json(output / "run-state.json", state)
    _validate_invocation_history(recipe, state, require_closed=True)


def _close_recovered_complete_invocation(
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    state: dict[str, Any],
) -> bool:
    """Close one fully committed open round without starting later work."""

    _validate_invocation_history(recipe, state, require_closed=False)
    invocations = state.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        return False
    active = invocations[-1]
    if active.get("ended_at") is not None:
        return False
    records = active.get("completed_cells_added")
    expected_cell_count = len(recipe.hgs_policies.budget_orders[0])
    if not isinstance(records, list) or len(records) != expected_cell_count:
        return False
    _finish_invocation(
        recipe,
        output,
        state,
        outcome="recovered_after_interruption",
    )
    return True


def _complete_one_round(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    cells: dict[tuple[int, int], dict[str, Any]],
    invocation_number: int,
    revalidate: WorkUnitRevalidator | None = None,
) -> int:
    _paths, _completed, remaining = _progress(recipe, cells)
    if not remaining:
        raise HGSHoldoutError("no HGS holdout round remains")
    round_index = remaining[0]
    seed = recipe.hgs_policies.seeds[round_index]
    order = recipe.hgs_policies.budget_orders[round_index]
    for order_index, budget in enumerate(order):
        key = (round_index, budget)
        if key in cells:
            continue
        if revalidate is None:

            def no_op_revalidation(_work_unit: str, _phase: str) -> None:
                return None

            unit_revalidate = no_op_revalidation
        else:
            unit_revalidate = revalidate
        work_unit = f"hgs_round_{round_index:02d}_budget_{budget:03d}_seed_{seed}"
        _guard_work_unit_inputs(
            unit_revalidate,
            output=output,
            state=state,
            work_unit=work_unit,
            phase="before_compute",
        )
        payload = _execute_hgs_quality_cell(
            recipe,
            output=output,
            state=state,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            round_index=round_index,
            order_index=order_index,
            seed=seed,
            budget=budget,
            invocation_number=invocation_number,
        )
        path = output / _cell_relative(round_index, budget, seed)
        _promote_work_unit_json(
            path,
            payload,
            output=output,
            state=state,
            work_unit=work_unit,
            revalidate=unit_revalidate,
        )
        cells[key] = _validate_hgs_cell(
            quality._load_json(path),
            recipe=recipe,
            output=output,
            state=state,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            round_index=round_index,
            order_index=order_index,
            seed=seed,
            budget=budget,
        )
        state["invocations"][-1]["completed_cells_added"].append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": quality._sha256_file(path),
            }
        )
        _reconcile_progress(recipe, output, state, cells)
        quality._atomic_write_json(output / "run-state.json", state)
        _guard_work_unit_inputs(
            unit_revalidate,
            output=output,
            state=state,
            work_unit=work_unit,
            phase="after_state_promotion",
            promoted_path=path,
        )
    return round_index


def _invalidate_integrity(
    output: Path,
    state: dict[str, Any],
    error: HGSHoldoutIntegrityError,
) -> None:
    state["status"] = INVALIDATED_STATUS
    state["invalidated_at"] = datetime.now(UTC).isoformat()
    state["integrity_failure"] = {
        "message": str(error),
        "details": error.details,
    }
    invocations = state.get("invocations")
    if isinstance(invocations, list) and invocations and invocations[-1].get("ended_at") is None:
        invocations[-1]["ended_at"] = state["invalidated_at"]
        invocations[-1]["outcome"] = "hgs_integrity_failure"
    quality._atomic_write_json(output / "run-state.json", state)


def _crossed_bootstrap_policy_samples(
    neural: np.ndarray,
    hgs_by_budget: Mapping[int, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return crossed-bootstrap distributions for the mean and empirical Q95.

    Every replicate shares one instance draw across all policies.  The neural
    checkpoint draw is independent of the HGS seed draw, while both HGS
    policies share the same HGS draw when more than one budget is present.
    This preserves paired-budget comparisons without pretending that neural
    training seeds and HGS seeds are paired experimental units.
    """

    neural = np.asarray(neural, dtype=np.float64)
    hgs = {
        int(budget): np.asarray(matrix, dtype=np.float64)
        for budget, matrix in hgs_by_budget.items()
    }
    if neural.ndim != 2 or neural.shape[0] < 2 or neural.shape[1] < 1:
        raise HGSHoldoutError("neural holdout matrix must be two-dimensional")
    if not hgs:
        raise HGSHoldoutError("holdout bootstrap requires at least one HGS policy")
    hgs_shapes = {matrix.shape for matrix in hgs.values()}
    if len(hgs_shapes) != 1:
        raise HGSHoldoutError("HGS holdout matrices must share one shape")
    hgs_shape = next(iter(hgs_shapes))
    if len(hgs_shape) != 2 or hgs_shape[0] < 2 or hgs_shape[1] != neural.shape[1]:
        raise HGSHoldoutError("neural and HGS matrices must share the instance axis")
    if replicates < 1:
        raise HGSHoldoutError("bootstrap replicate count must be positive")
    if not np.isfinite(neural).all() or any(
        not np.isfinite(matrix).all() for matrix in hgs.values()
    ):
        raise HGSHoldoutError("bootstrap matrices must contain only finite gaps")

    policy_ids = ("neural", *(f"hgs_b{budget}" for budget in sorted(hgs)))
    samples = {
        policy_id: {
            "mean": np.empty(replicates, dtype=np.float64),
            "empirical_q95": np.empty(replicates, dtype=np.float64),
        }
        for policy_id in policy_ids
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    for replicate_index in range(replicates):
        instance_indices = rng.integers(0, neural.shape[1], size=neural.shape[1])
        neural_seed_indices = rng.integers(0, neural.shape[0], size=neural.shape[0])
        hgs_seed_indices = rng.integers(0, hgs_shape[0], size=hgs_shape[0])
        neural_selected = neural[neural_seed_indices[:, None], instance_indices[None, :]]
        samples["neural"]["mean"][replicate_index] = neural_selected.mean()
        samples["neural"]["empirical_q95"][replicate_index] = np.quantile(
            neural_selected, 0.95, method="linear"
        )
        for budget, matrix in hgs.items():
            selected = matrix[hgs_seed_indices[:, None], instance_indices[None, :]]
            policy_samples = samples[f"hgs_b{budget}"]
            policy_samples["mean"][replicate_index] = selected.mean()
            policy_samples["empirical_q95"][replicate_index] = np.quantile(
                selected, 0.95, method="linear"
            )
    return samples


def _summarize_policy(
    matrix: np.ndarray,
    *,
    policy_id: str,
    policy_role: str,
    invalid_instances: int,
    threshold_pct: float,
    t_critical_value: float,
    t_degrees_of_freedom: int,
    bootstrap_mean_samples: np.ndarray,
    bootstrap_q95_samples: np.ndarray,
    bootstrap_quantile: float,
) -> dict[str, Any]:
    """Apply every preregistered quality condition to one policy matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)
    mean_samples = np.asarray(bootstrap_mean_samples, dtype=np.float64)
    q95_samples = np.asarray(bootstrap_q95_samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise HGSHoldoutError(f"{policy_id} gap matrix must be two-dimensional")
    if (
        mean_samples.ndim != 1
        or q95_samples.ndim != 1
        or mean_samples.size < 1
        or mean_samples.shape != q95_samples.shape
    ):
        raise HGSHoldoutError(f"{policy_id} bootstrap samples are invalid")
    if isinstance(invalid_instances, bool) or invalid_instances < 0:
        raise HGSHoldoutError(f"{policy_id} invalid-instance count is invalid")

    finite = bool(
        np.isfinite(matrix).all()
        and np.isfinite(mean_samples).all()
        and np.isfinite(q95_samples).all()
    )
    seed_means: np.ndarray | None = None
    seed_q95: np.ndarray | None = None
    mean: float | None = None
    sample_std: float | None = None
    pooled_q95: float | None = None
    t_ucb: float | None = None
    bootstrap_mean_estimate: float | None = None
    bootstrap_mean_ucb: float | None = None
    bootstrap_q95_estimate: float | None = None
    bootstrap_q95_ucb: float | None = None
    minimum_gap: float | None = None
    maximum_gap: float | None = None
    threshold_count: int | None = None
    threshold_rate: float | None = None
    if finite:
        computed_seed_means = matrix.mean(axis=1)
        computed_seed_q95 = np.quantile(matrix, 0.95, axis=1, method="linear")
        seed_means = computed_seed_means
        seed_q95 = computed_seed_q95
        mean = float(computed_seed_means.mean())
        sample_std = float(computed_seed_means.std(ddof=1))
        pooled_q95 = float(np.quantile(matrix, 0.95, method="linear"))
        t_ucb = float(mean + t_critical_value * sample_std / math.sqrt(matrix.shape[0]))
        bootstrap_mean_estimate = float(mean_samples.mean())
        bootstrap_mean_ucb = float(np.quantile(mean_samples, bootstrap_quantile, method="linear"))
        bootstrap_q95_estimate = float(q95_samples.mean())
        bootstrap_q95_ucb = float(np.quantile(q95_samples, bootstrap_quantile, method="linear"))
        minimum_gap = float(matrix.min())
        maximum_gap = float(matrix.max())
        threshold_count = int(np.count_nonzero(matrix > threshold_pct))
        threshold_rate = float(threshold_count / matrix.size)

    conditions = {
        "finite": finite,
        "zero_invalid_instances": invalid_instances == 0,
        "overall_mean_strictly_below_threshold": (mean is not None and mean < threshold_pct),
        "every_seed_mean_strictly_below_threshold": (
            seed_means is not None and bool(np.all(seed_means < threshold_pct))
        ),
        "every_seed_empirical_q95_strictly_below_threshold": (
            seed_q95 is not None and bool(np.all(seed_q95 < threshold_pct))
        ),
        "pooled_empirical_q95_strictly_below_threshold": (
            pooled_q95 is not None and pooled_q95 < threshold_pct
        ),
        "t_ucb_strictly_below_threshold": (t_ucb is not None and t_ucb < threshold_pct),
        "bootstrap_mean_ucb_strictly_below_threshold": (
            bootstrap_mean_ucb is not None and bootstrap_mean_ucb < threshold_pct
        ),
        "bootstrap_q95_ucb_strictly_below_threshold": (
            bootstrap_q95_ucb is not None and bootstrap_q95_ucb < threshold_pct
        ),
    }
    common_conditions = (
        conditions["finite"],
        conditions["zero_invalid_instances"],
    )
    mean_gate_passed = all(
        (
            *common_conditions,
            conditions["overall_mean_strictly_below_threshold"],
            conditions["every_seed_mean_strictly_below_threshold"],
            conditions["t_ucb_strictly_below_threshold"],
            conditions["bootstrap_mean_ucb_strictly_below_threshold"],
        )
    )
    tail_gate_passed = all(
        (
            *common_conditions,
            conditions["every_seed_empirical_q95_strictly_below_threshold"],
            conditions["pooled_empirical_q95_strictly_below_threshold"],
            conditions["bootstrap_q95_ucb_strictly_below_threshold"],
        )
    )
    passed = bool(mean_gate_passed and tail_gate_passed)
    if not conditions["finite"] or not conditions["zero_invalid_instances"]:
        classification = "invalid"
    elif passed:
        classification = "pass"
    elif mean_gate_passed:
        classification = "average_quality_feasible_tail_quality_infeasible"
    elif tail_gate_passed:
        classification = "average_quality_infeasible_tail_quality_feasible"
    else:
        classification = "average_and_tail_quality_infeasible"

    per_seed_means = seed_means.tolist() if seed_means is not None else None
    per_seed_q95 = seed_q95.tolist() if seed_q95 is not None else None
    return {
        "policy_id": policy_id,
        "policy_role": policy_role,
        "classification": classification,
        "passed": passed,
        "finite": finite,
        "invalid_instances": invalid_instances,
        "gap_matrix_shape": list(matrix.shape),
        "gap_matrix_pct": matrix.tolist() if finite else None,
        "per_seed_mean_gap_pct": per_seed_means,
        "per_seed_empirical_q95_gap_pct": per_seed_q95,
        "mean_of_seed_means_gap_pct": mean,
        "sample_std_of_seed_means_gap_pct": sample_std,
        "maximum_seed_mean_gap_pct": (float(seed_means.max()) if seed_means is not None else None),
        "maximum_seed_empirical_q95_gap_pct": (
            float(seed_q95.max()) if seed_q95 is not None else None
        ),
        "pooled_empirical_q95_gap_pct": pooled_q95,
        "t_ucb_95_gap_pct": t_ucb,
        "bootstrap_mean_estimate_gap_pct": bootstrap_mean_estimate,
        "bootstrap_mean_ucb_95_gap_pct": bootstrap_mean_ucb,
        "bootstrap_q95_estimate_gap_pct": bootstrap_q95_estimate,
        "bootstrap_q95_ucb_95_gap_pct": bootstrap_q95_ucb,
        "threshold_pct": threshold_pct,
        "strict_threshold": True,
        "t_critical_value": t_critical_value,
        "t_degrees_of_freedom": t_degrees_of_freedom,
        "bootstrap_quantile": bootstrap_quantile,
        "gate_conditions": conditions,
        "mean_gate_passed": mean_gate_passed,
        "tail_gate_passed": tail_gate_passed,
        "all_required_conditions_satisfied": passed,
        "diagnostics_not_used_for_gate": {
            "minimum_instance_gap_pct": minimum_gap,
            "maximum_instance_gap_pct": maximum_gap,
            "strict_exceedance_counts": {
                "gt_5_pct": (int(np.count_nonzero(matrix > 5.0)) if finite else None),
                "gt_10_pct": (int(np.count_nonzero(matrix > 10.0)) if finite else None),
                "gt_20_pct": (int(np.count_nonzero(matrix > 20.0)) if finite else None),
            },
            "strict_exceedance_fractions": {
                "gt_5_pct": (
                    float(np.count_nonzero(matrix > 5.0) / matrix.size) if finite else None
                ),
                "gt_10_pct": (
                    float(np.count_nonzero(matrix > 10.0) / matrix.size) if finite else None
                ),
                "gt_20_pct": (
                    float(np.count_nonzero(matrix > 20.0) / matrix.size) if finite else None
                ),
            },
            "instances_strictly_above_gate_threshold": threshold_count,
            "fraction_strictly_above_gate_threshold": threshold_rate,
        },
    }


def _invalid_instances(payloads: Sequence[dict[str, Any]]) -> int:
    total = 0
    for payload in payloads:
        validation = payload.get("validation")
        if not isinstance(validation, dict):
            raise HGSHoldoutError("policy evaluation lacks route validation")
        value = validation.get("invalid_instance_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HGSHoldoutError("policy invalid-instance count changed")
        total += value
    return total


def _assessment_status(
    primary_passed: bool,
    sensitivity_passed: bool = False,
    *,
    recipe: AETHGSHoldoutRecipe | None = None,
) -> str:
    if recipe is not None and _is_confirmatory_recipe(recipe):
        return "complete_confirmatory_passed" if primary_passed else "complete_confirmatory_nonpass"
    primary = "passed" if primary_passed else "nonpass"
    sensitivity = "passed" if sensitivity_passed else "nonpass"
    status = f"complete_primary_{primary}_sensitivity_{sensitivity}"
    if status not in COMPLETE_STATUSES:
        raise HGSHoldoutError("computed an unknown holdout completion status")
    return status


def _build_assessment(
    recipe: AETHGSHoldoutRecipe,
    neural_evaluations: Mapping[int, dict[str, Any]],
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic holdout decision from the complete policy grid."""

    neural_seeds = tuple(recipe.neural_policy.training_seeds)
    if tuple(sorted(neural_evaluations)) != neural_seeds:
        raise HGSHoldoutError("holdout assessment requires all five neural seeds")
    budgets = tuple(
        dict.fromkeys(budget for order in recipe.hgs_policies.budget_orders for budget in order)
    )
    expected_cells = {
        (round_index, budget)
        for round_index in range(len(recipe.hgs_policies.seeds))
        for budget in budgets
    }
    if set(cells) != expected_cells:
        raise HGSHoldoutError("holdout assessment requires the complete HGS grid")

    neural_payloads = [neural_evaluations[seed] for seed in neural_seeds]
    neural = np.asarray(
        [payload["quality"]["gaps_pct"] for payload in neural_payloads],
        dtype=np.float64,
    )
    hgs_payloads: dict[int, list[dict[str, Any]]] = {}
    hgs: dict[int, np.ndarray] = {}
    for budget in budgets:
        payloads = [
            cells[(round_index, budget)] for round_index in range(len(recipe.hgs_policies.seeds))
        ]
        hgs_payloads[budget] = payloads
        hgs[budget] = np.asarray(
            [payload["quality"]["gaps_pct"] for payload in payloads],
            dtype=np.float64,
        )
    expected_instances = recipe.dataset.holdout.num_instances
    if neural.shape != (len(neural_seeds), expected_instances):
        raise HGSHoldoutError("neural holdout gap matrix shape changed")
    expected_hgs_shape = (len(recipe.hgs_policies.seeds), expected_instances)
    if any(matrix.shape != expected_hgs_shape for matrix in hgs.values()):
        raise HGSHoldoutError("HGS holdout gap matrix shape changed")

    samples = _crossed_bootstrap_policy_samples(
        neural,
        hgs,
        replicates=recipe.bootstrap.replicates,
        seed=recipe.bootstrap.seed,
    )
    threshold = float(recipe.gate.maximum_mean_gap_pct)
    neural_summary = _summarize_policy(
        neural,
        policy_id="neural_pomo_epoch40_seeds2_6",
        policy_role=(
            "neural_confirmatory" if _is_confirmatory_recipe(recipe) else "neural_primary"
        ),
        invalid_instances=_invalid_instances(neural_payloads),
        threshold_pct=threshold,
        t_critical_value=recipe.bootstrap.neural_t_critical_value,
        t_degrees_of_freedom=recipe.bootstrap.neural_t_degrees_of_freedom,
        bootstrap_mean_samples=samples["neural"]["mean"],
        bootstrap_q95_samples=samples["neural"]["empirical_q95"],
        bootstrap_quantile=recipe.bootstrap.confidence_level,
    )
    primary_budget = recipe.hgs_policies.primary_budget
    sensitivity_budget = recipe.hgs_policies.sensitivity_budget
    primary_hgs_summary = _summarize_policy(
        hgs[primary_budget],
        policy_id=f"hgs_b{primary_budget}",
        policy_role=("hgs_confirmatory" if _is_confirmatory_recipe(recipe) else "hgs_primary"),
        invalid_instances=_invalid_instances(hgs_payloads[primary_budget]),
        threshold_pct=threshold,
        t_critical_value=recipe.bootstrap.hgs_t_critical_value,
        t_degrees_of_freedom=recipe.bootstrap.hgs_t_degrees_of_freedom,
        bootstrap_mean_samples=samples[f"hgs_b{primary_budget}"]["mean"],
        bootstrap_q95_samples=samples[f"hgs_b{primary_budget}"]["empirical_q95"],
        bootstrap_quantile=recipe.bootstrap.confidence_level,
    )
    primary_passed = bool(neural_summary["passed"] and primary_hgs_summary["passed"])
    if _is_confirmatory_recipe(recipe):
        status = _assessment_status(primary_passed, recipe=recipe)
        return {
            "schema_version": ASSESSMENT_SCHEMA,
            "status": status,
            "classification": _classification(recipe),
            "split": "holdout",
            "holdout_seed": recipe.dataset.holdout.seed,
            "instances": recipe.dataset.holdout.num_instances,
            "all_preregistered_policies_evaluated": True,
            "policy_source_confirmed_budget": primary_budget,
            "quality_method": {
                "metric": recipe.gate.metric,
                "comparison": recipe.gate.comparison,
                "threshold_pct": threshold,
                "strict_threshold": True,
                "bootstrap": _mapping(recipe.bootstrap),
                "shared_instance_resampling_across_all_policies": True,
                "independent_neural_and_hgs_seed_resampling": True,
                "maximum_gap_and_threshold_rate_are_diagnostics_only": True,
            },
            "policies": {
                "neural": neural_summary,
                f"hgs_b{primary_budget}": primary_hgs_summary,
            },
            "confirmatory_decision": {
                "rule": recipe.gate.primary_joint_rule,
                "neural_passed": neural_summary["passed"],
                "hgs_budget": primary_budget,
                "hgs_passed": primary_hgs_summary["passed"],
                "passed": primary_passed,
            },
            "primary_passed": primary_passed,
            "energy_measurement": "none",
            "carbon_accounting": "none",
            "timing_scientific_use": False,
            "cross_solver_energy_comparable": False,
            "aet_was_computed": False,
        }
    if sensitivity_budget is None:
        raise HGSHoldoutError("legacy holdout sensitivity budget is missing")
    sensitivity_hgs_summary = _summarize_policy(
        hgs[sensitivity_budget],
        policy_id=f"hgs_b{sensitivity_budget}",
        policy_role="hgs_sensitivity",
        invalid_instances=_invalid_instances(hgs_payloads[sensitivity_budget]),
        threshold_pct=threshold,
        t_critical_value=recipe.bootstrap.hgs_t_critical_value,
        t_degrees_of_freedom=recipe.bootstrap.hgs_t_degrees_of_freedom,
        bootstrap_mean_samples=samples[f"hgs_b{sensitivity_budget}"]["mean"],
        bootstrap_q95_samples=samples[f"hgs_b{sensitivity_budget}"]["empirical_q95"],
        bootstrap_quantile=recipe.bootstrap.confidence_level,
    )
    sensitivity_passed = bool(neural_summary["passed"] and sensitivity_hgs_summary["passed"])
    status = _assessment_status(primary_passed, sensitivity_passed)
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": status,
        "classification": _classification(recipe),
        "split": "holdout",
        "holdout_seed": recipe.dataset.holdout.seed,
        "instances": recipe.dataset.holdout.num_instances,
        "all_preregistered_policies_evaluated": True,
        "selection_source_selected_budget": PRIMARY_BUDGET,
        "quality_method": {
            "metric": recipe.gate.metric,
            "comparison": recipe.gate.comparison,
            "threshold_pct": threshold,
            "strict_threshold": True,
            "bootstrap": _mapping(recipe.bootstrap),
            "shared_instance_resampling_across_all_policies": True,
            "independent_neural_and_hgs_seed_resampling": True,
            "shared_hgs_seed_resampling_between_b3_and_b10": True,
            "maximum_gap_and_threshold_rate_are_diagnostics_only": True,
        },
        "policies": {
            "neural": neural_summary,
            f"hgs_b{primary_budget}": primary_hgs_summary,
            f"hgs_b{sensitivity_budget}": sensitivity_hgs_summary,
        },
        "primary_decision": {
            "rule": recipe.gate.primary_joint_rule,
            "neural_passed": neural_summary["passed"],
            "hgs_primary_budget": primary_budget,
            "hgs_primary_passed": primary_hgs_summary["passed"],
            "passed": primary_passed,
        },
        "sensitivity_decision": {
            "budget": sensitivity_budget,
            "neural_passed": neural_summary["passed"],
            "hgs_sensitivity_passed": sensitivity_hgs_summary["passed"],
            "joint_passed": sensitivity_passed,
            "can_rescue_primary": False,
            "rescued_primary": False,
        },
        "primary_passed": primary_passed,
        "sensitivity_passed": sensitivity_passed,
        "energy_measurement": "none",
        "carbon_accounting": "none",
        "timing_scientific_use": False,
        "cross_solver_energy_comparable": False,
        "aet_was_computed": False,
    }


def _cell_inventory_entry(
    output: Path,
    cell: dict[str, Any],
) -> dict[str, Any]:
    relative = _cell_relative(int(cell["round"]), int(cell["budget"]), int(cell["base_seed"]))
    return {
        "path": relative,
        "sha256": quality._sha256_file(output / relative),
    }


def _reconcile_cell_invocation_links(
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    state: dict[str, Any],
    cells: Mapping[tuple[int, int], dict[str, Any]],
    *,
    allow_recovery: bool = True,
) -> None:
    """Recover a cell committed just before an interrupted state update."""

    invocations = state.get("invocations")
    if not isinstance(invocations, list):
        raise HGSHoldoutError("holdout invocation history is invalid")
    by_number: dict[int, dict[str, Any]] = {}
    for expected_number, invocation in enumerate(invocations, start=1):
        if (
            not isinstance(invocation, dict)
            or invocation.get("invocation_number") != expected_number
            or not isinstance(invocation.get("completed_cells_added"), list)
        ):
            raise HGSHoldoutError("holdout invocation history changed")
        by_number[expected_number] = invocation

    changed = False
    cell_by_path: dict[str, dict[str, Any]] = {}
    for cell in cells.values():
        entry = _cell_inventory_entry(output, cell)
        path = entry["path"]
        if path in cell_by_path:
            raise HGSHoldoutError("duplicate HGS cell path")
        cell_by_path[path] = cell
        invocation_number = cell.get("invocation_number")
        if isinstance(invocation_number, bool) or invocation_number not in by_number:
            raise HGSHoldoutError("HGS cell references an unknown invocation")
        invocation = by_number[int(invocation_number)]
        records = invocation["completed_cells_added"]
        matching = [record for record in records if record.get("path") == path]
        if len(matching) > 1:
            raise HGSHoldoutError("HGS cell has duplicate invocation records")
        if matching:
            if not _strict_json_equal(matching[0], entry):
                raise HGSHoldoutError("HGS cell invocation checksum changed")
        else:
            if not allow_recovery:
                raise HGSHoldoutError("completed holdout cell lacks its sealed invocation link")
            records.append(entry)
            invocation.setdefault("recovered_cell_links", []).append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "path": path,
                    "reason": "atomic_cell_survived_interrupted_state_update",
                }
            )
            changed = True

    for invocation in invocations:
        seen: set[str] = set()
        for record in invocation["completed_cells_added"]:
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise HGSHoldoutError("holdout invocation cell record changed")
            path = record.get("path")
            digest = record.get("sha256")
            if (
                not isinstance(path, str)
                or path in seen
                or path not in cell_by_path
                or not isinstance(digest, str)
                or _HEX_SHA256.fullmatch(digest) is None
                or quality._sha256_file(output / path) != digest
            ):
                raise HGSHoldoutError("holdout invocation cell inventory changed")
            if cell_by_path[path].get("invocation_number") != invocation["invocation_number"]:
                raise HGSHoldoutError("HGS cell invocation ownership changed")
            seen.add(path)
    _validate_invocation_history(recipe, state, require_closed=False)
    if changed:
        quality._atomic_write_json(output / "run-state.json", state)


def _validate_invocation_history(
    recipe: AETHGSHoldoutRecipe,
    state: Mapping[str, Any],
    *,
    require_closed: bool,
) -> None:
    invocations = state.get("invocations")
    if not isinstance(invocations, list):
        raise HGSHoldoutError("holdout invocation history is invalid")
    if require_closed and not invocations:
        raise HGSHoldoutError("holdout invocation history is empty")
    expected_by_round = {
        round_index: [
            _cell_relative(round_index, budget, seed)
            for budget in recipe.hgs_policies.budget_orders[round_index]
        ]
        for round_index, seed in enumerate(recipe.hgs_policies.seeds)
    }
    round_by_path = {
        path: round_index for round_index, paths in expected_by_round.items() for path in paths
    }
    scheduled_paths = [
        path
        for round_index in range(len(recipe.hgs_policies.seeds))
        for path in expected_by_round[round_index]
    ]
    open_count = 0
    referenced_paths: set[str] = set()
    attributed_paths: list[str] = []
    allowed_outcomes = {
        "running",
        "recovered_after_interruption",
        "paired_seed_round_completed",
        "hgs_seed_round_completed",
        "recovered_complete_without_new_hgs",
    }
    for number, invocation in enumerate(invocations, start=1):
        if not isinstance(invocation, dict) or invocation.get("invocation_number") != number:
            raise HGSHoldoutError("holdout invocation numbering changed")
        if not isinstance(invocation.get("started_at"), str):
            raise HGSHoldoutError("holdout invocation start time changed")
        if invocation.get("outcome") not in allowed_outcomes:
            raise HGSHoldoutError("holdout invocation outcome changed")
        if invocation.get("ended_at") is None:
            open_count += 1
            if invocation.get("outcome") != "running" or number != len(invocations):
                raise HGSHoldoutError("only the final holdout invocation may be open")
        elif not isinstance(invocation.get("ended_at"), str):
            raise HGSHoldoutError("holdout invocation end time changed")
        if (
            invocation.get("energy_measurement") != "none"
            or invocation.get("preflight_used") is not False
            or invocation.get("exclusive_attestation_used") is not False
            or not isinstance(invocation.get("completed_cells_added"), list)
        ):
            raise HGSHoldoutError("holdout invocation quality-only controls changed")
        records = invocation["completed_cells_added"]
        maximum_records = max(len(paths) for paths in expected_by_round.values())
        if len(records) > maximum_records:
            raise HGSHoldoutError("one holdout invocation may add at most one HGS round")
        paths: list[str] = []
        for record in records:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or not isinstance(record.get("path"), str)
                or record["path"] not in round_by_path
                or not isinstance(record.get("sha256"), str)
                or _HEX_SHA256.fullmatch(record["sha256"]) is None
            ):
                raise HGSHoldoutError("holdout invocation cell evidence changed")
            paths.append(record["path"])
        if len(set(paths)) != len(paths) or referenced_paths.intersection(paths):
            raise HGSHoldoutError("HGS cell attribution is duplicated across invocations")
        referenced_paths.update(paths)
        attributed_paths.extend(paths)
        if paths:
            rounds = {round_by_path[path] for path in paths}
            if len(rounds) != 1:
                raise HGSHoldoutError("one holdout invocation spans multiple HGS rounds")
            round_index = next(iter(rounds))
            if paths != expected_by_round[round_index][: len(paths)]:
                raise HGSHoldoutError("holdout invocation cells are not the scheduled round prefix")
    if attributed_paths != scheduled_paths[: len(attributed_paths)]:
        raise HGSHoldoutError("holdout invocation history is not the global scheduled cell prefix")
    if require_closed and open_count:
        raise HGSHoldoutError("completed holdout contains an open invocation")
    if open_count > 1:
        raise HGSHoldoutError("holdout contains multiple open invocations")


def _reference_inventory(reference: dict[str, Any]) -> set[str]:
    candidates = reference.get("candidate_artifacts")
    if not isinstance(candidates, list):
        raise HGSHoldoutError("holdout reference candidate inventory is invalid")
    paths = {"reference/holdout/reference.json", "reference/reference-lock.json"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
            raise HGSHoldoutError("holdout reference candidate entry is invalid")
        paths.add(str(candidate["path"]))
    return paths


def _expected_artifact_inventory(
    recipe: AETHGSHoldoutRecipe,
    reference: dict[str, Any],
) -> set[str]:
    corpus_relative = recipe.dataset.holdout.artifact
    expected = {
        "recipe.yaml",
        "environment/uv.lock",
        "provenance/hgs-selection-receipt.json",
        "provenance/quality-replication-receipt.json",
        corpus_relative,
        Path(corpus_relative).with_suffix(".manifest.json").as_posix(),
        "holdout-assessment.json",
        "manifest.json",
        "SHA256SUMS",
        "run-state.json",
        *_reference_inventory(reference),
    }
    expected.update(_neural_path(seed) for seed in recipe.neural_policy.training_seeds)
    expected.update(relative for *_prefix, relative in _expected_cells(recipe))
    return expected


def _assert_exact_artifact_inventory(
    recipe: AETHGSHoldoutRecipe,
    output: Path,
    reference: dict[str, Any],
) -> None:
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected = _expected_artifact_inventory(recipe, reference)
    if actual != expected:
        raise HGSHoldoutError(
            "holdout artifact inventory changed; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _manifest_payload(
    recipe: AETHGSHoldoutRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_lock: dict[str, Any],
    neural_evaluations: Mapping[int, dict[str, Any]],
    cells: Mapping[tuple[int, int], dict[str, Any]],
    assessment: dict[str, Any],
    ended_at: str,
) -> dict[str, Any]:
    invocations = state.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise HGSHoldoutError("holdout manifest requires invocation history")
    assessment_path = output / "holdout-assessment.json"
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": assessment["status"],
        "classification": _classification(recipe),
        "initialized_at": state["created_at"],
        "holdout_opening_authorized_at": state["holdout_opening_authorized_at"],
        "holdout_opened_at": state["holdout_opened_at"],
        "ended_at": ended_at,
        "recipe": {
            "path": "recipe.yaml",
            "sha256": state["recipe_sha256"],
            "sha256_basis": "utf8_crlf_and_cr_to_lf_canonical",
            "name": recipe.name,
        },
        "source": {
            "git_sha": state["git_sha"],
            "git_at_initialization": state["git"],
            "uv_lock_sha256": state["uv_lock_sha256"],
            "uv_lock_sha256_basis": "utf8_crlf_and_cr_to_lf_canonical",
            "canonical_source_snapshot": state["source_snapshot"],
        },
        "runtime_identity_at_opening": state["runtime_identity"],
        "execution_invocations": invocations,
        "quality_only_controls": {
            "energy_measurement": "none",
            "carbon_accounting": "none",
            "preflight_used": False,
            "exclusive_attestation_used": False,
            "timing_scientific_use": False,
            "maximum_newly_completed_rounds_per_invocation": 1,
        },
        "source_bundles": {
            ("policy" if _is_confirmatory_recipe(recipe) else "selection"): {
                "path": "provenance/hgs-selection-receipt.json",
                "sha256": quality._sha256_file(_selection_receipt_path(output)),
                "validated_in_place": True,
                "copied_into_holdout": False,
            },
            "replication": {
                "path": "provenance/quality-replication-receipt.json",
                "sha256": quality._sha256_file(_replication_receipt_path(output)),
                "validated_in_place": True,
                "copied_into_holdout": False,
            },
        },
        "dataset": {
            "split": "holdout",
            "path": corpus.path.relative_to(output).as_posix(),
            "file_sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
            "instances": int(corpus.coords.shape[0]),
            "seed": recipe.dataset.holdout.seed,
            **(
                {"forbidden_prior_content_sha256": list(_forbidden_content_hashes(recipe))}
                if _is_confirmatory_recipe(recipe)
                else {
                    "selection_content_sha256_forbidden": (
                        recipe.dataset.forbidden_selection_content_sha256
                    )
                }
            ),
            "opened_only_after_durable_source_anchor": True,
        },
        "reference_lock": {
            "path": "reference/reference-lock.json",
            "sha256": state["reference_lock_sha256"],
            "payload": reference_lock,
            "reference_sha256": quality._sha256_file(
                output / "reference" / "holdout" / "reference.json"
            ),
            "reference_mean_cost": reference["mean_cost"],
            "fresh_for_holdout": True,
            "locked_before_all_policy_evaluation": True,
        },
        "neural_evaluations": [
            {
                "training_seed": seed,
                "evaluation_seed": neural_evaluations[seed]["evaluation_seed"],
                "path": _neural_path(seed),
                "sha256": quality._sha256_file(output / _neural_path(seed)),
                "mean_gap_pct": neural_evaluations[seed]["quality"]["mean_gap_pct"],
                "energy_measured": False,
            }
            for seed in sorted(neural_evaluations)
        ],
        "hgs_round_schedule": state["round_schedule"],
        "hgs_cells": [
            {
                "round": round_index,
                "base_seed": cell["base_seed"],
                "budget": budget,
                "policy_role": cell["classification"]["policy_role"],
                "path": _cell_relative(round_index, budget, cell["base_seed"]),
                "sha256": quality._sha256_file(
                    output / _cell_relative(round_index, budget, cell["base_seed"])
                ),
                "mean_gap_pct": cell["quality"]["mean_gap_pct"],
                "invocation_number": cell["invocation_number"],
                "energy_measured": False,
                "timing_scientific_use": False,
            }
            for (round_index, budget), cell in sorted(cells.items())
        ],
        "holdout_assessment": {
            "path": "holdout-assessment.json",
            "sha256": quality._sha256_file(assessment_path),
            "payload": assessment,
        },
        "decision": (
            {
                "hgs_budget": recipe.hgs_policies.primary_budget,
                "neural_passed": assessment["policies"]["neural"]["passed"],
                "hgs_passed": assessment["policies"][f"hgs_b{recipe.hgs_policies.primary_budget}"][
                    "passed"
                ],
                "passed": assessment["primary_passed"],
                "rule": recipe.gate.primary_joint_rule,
            }
            if _is_confirmatory_recipe(recipe)
            else {
                "primary_budget": recipe.hgs_policies.primary_budget,
                "sensitivity_budget": recipe.hgs_policies.sensitivity_budget,
                "primary_passed": assessment["primary_passed"],
                "sensitivity_passed": assessment["sensitivity_passed"],
                "sensitivity_can_rescue_primary": False,
            }
        ),
        "cross_solver_energy_comparable": False,
        "aet_was_computed": False,
    }


def _revalidate_live_inputs(
    *,
    recipe: AETHGSHoldoutRecipe,
    source_recipe: Path,
    recipe_bytes: bytes,
    root: Path,
    output: Path,
    state: dict[str, Any],
) -> None:
    try:
        current_recipe = source_recipe.read_bytes()
        current_uv_lock = (root / "uv.lock").read_bytes()
    except OSError as exc:
        raise HGSHoldoutQualificationError("holdout frozen inputs became unavailable") from exc
    frozen_recipe = output / "recipe.yaml"
    frozen_uv_lock = output / "environment" / "uv.lock"
    if (
        _canonical_input_hash(current_recipe) != _canonical_input_hash(recipe_bytes)
        or _canonical_input_hash(current_recipe) != state.get("recipe_sha256")
        or not frozen_recipe.is_file()
        or _canonical_file_sha256(frozen_recipe)[0] != state.get("recipe_sha256")
        or _canonical_input_hash(current_uv_lock) != state.get("uv_lock_sha256")
        or not frozen_uv_lock.is_file()
        or _canonical_file_sha256(frozen_uv_lock)[0] != state.get("uv_lock_sha256")
    ):
        raise HGSHoldoutQualificationError("holdout frozen input changed during execution")
    if _canonical_source_snapshot(recipe, root).get("sha256") != state.get(
        "source_snapshot", {}
    ).get("sha256"):
        raise HGSHoldoutQualificationError(
            "holdout canonical source snapshot changed during execution"
        )
    if _git_snapshot(root).get("sha") != state.get("git_sha"):
        raise HGSHoldoutQualificationError("holdout Git commit changed during execution")


def _revalidate_sources(
    recipe: AETHGSHoldoutRecipe,
    root: Path,
    expected: SourceEvidence,
) -> None:
    current = _validate_sources(recipe, root)
    if not _strict_json_equal(current.selection_receipt, expected.selection_receipt):
        raise HGSHoldoutQualificationError("HGS selection source changed during holdout execution")
    if not _strict_json_equal(
        current.replication_evidence.receipt,
        expected.replication_evidence.receipt,
    ):
        raise HGSHoldoutQualificationError(
            "quality-replication source changed during holdout execution"
        )


def _revalidate_work_unit_inputs(
    *,
    recipe: AETHGSHoldoutRecipe,
    source_recipe: Path,
    recipe_bytes: bytes,
    root: Path,
    output: Path,
    state: dict[str, Any],
    evidence: SourceEvidence,
    work_unit: str,
    phase: str,
) -> None:
    """Revalidate every anchored input at one work-unit boundary."""

    if not work_unit or not phase:
        raise HGSHoldoutError("work-unit revalidation requires a unit and phase")
    _revalidate_live_inputs(
        recipe=recipe,
        source_recipe=source_recipe,
        recipe_bytes=recipe_bytes,
        root=root,
        output=output,
        state=state,
    )
    _revalidate_sources(recipe, root, evidence)


def _quarantine_promoted_artifact(path: Path) -> dict[str, Any]:
    """Remove a compromised final path, retaining only a cleanup-safe quarantine."""

    result: dict[str, Any] = {
        "path": path.as_posix(),
        "existed": path.exists(),
        "final_path_removed": not path.exists(),
        "quarantine_path": None,
        "quarantine_removed": True,
    }
    if not path.exists():
        return result
    quarantine = path.parent / f".{path.name}.partial-invalidated-{uuid.uuid4().hex}"
    try:
        os.replace(path, quarantine)
        result["quarantine_path"] = quarantine.as_posix()
    except OSError:
        with suppress(OSError):
            path.unlink()
    if quarantine.exists():
        try:
            quarantine.unlink()
        except OSError:
            result["quarantine_removed"] = False
    result["final_path_removed"] = not path.exists()
    return result


def _invalidate_live_input_change(
    *,
    output: Path,
    state: dict[str, Any],
    work_unit: str,
    phase: str,
    error: Exception,
    promoted_path: Path | None,
) -> None:
    invalidated_at = datetime.now(UTC).isoformat()
    quarantine = _quarantine_promoted_artifact(promoted_path) if promoted_path is not None else None
    state["status"] = INVALIDATED_INPUT_STATUS
    state["invalidated_at"] = invalidated_at
    state["live_input_failure"] = {
        "schema_version": "aet-hgs-holdout-live-input-failure/v1",
        "work_unit": work_unit,
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "promoted_artifact": quarantine,
        "resume_allowed": False,
        "recovery": "use a new preregistered output_root after diagnosing the input change",
    }
    invocations = state.get("invocations")
    if isinstance(invocations, list) and invocations:
        active = invocations[-1]
        if isinstance(active, dict) and active.get("ended_at") is None:
            active["ended_at"] = invalidated_at
            active["outcome"] = "input_changed_during_work_unit"
    quality._atomic_write_json(output / "run-state.json", state)


def _guard_work_unit_inputs(
    revalidate: WorkUnitRevalidator,
    *,
    output: Path,
    state: dict[str, Any],
    work_unit: str,
    phase: str,
    promoted_path: Path | None = None,
) -> None:
    try:
        revalidate(work_unit, phase)
    except Exception as exc:
        _invalidate_live_input_change(
            output=output,
            state=state,
            work_unit=work_unit,
            phase=phase,
            error=exc,
            promoted_path=promoted_path,
        )
        raise HGSHoldoutQualificationError(
            f"live input changed during {work_unit} ({phase}); "
            "the holdout output is permanently invalidated"
        ) from exc


def _promote_work_unit_json(
    path: Path,
    payload: dict[str, Any],
    *,
    output: Path,
    state: dict[str, Any],
    work_unit: str,
    revalidate: WorkUnitRevalidator,
) -> None:
    """Atomically promote JSON only between successful input checks."""

    if path.exists():
        raise HGSHoldoutError(f"work-unit promotion target already exists: {path}")
    _guard_work_unit_inputs(
        revalidate,
        output=output,
        state=state,
        work_unit=work_unit,
        phase="before_artifact_promotion",
    )
    quality._atomic_write_json(path, payload)
    _guard_work_unit_inputs(
        revalidate,
        output=output,
        state=state,
        work_unit=work_unit,
        phase="after_artifact_promotion",
        promoted_path=path,
    )


def _validate_completed_run_state(
    recipe: AETHGSHoldoutRecipe,
    state: dict[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "created_at",
        "recipe_sha256",
        "uv_lock_sha256",
        "git_sha",
        "git",
        "source_snapshot",
        "runtime_identity",
        "classification",
        "selection_source_receipt_sha256",
        "replication_source_receipt_sha256",
        "round_schedule",
        "holdout_opening_authorized_at",
        "holdout_opened_at",
        "holdout_file_sha256",
        "holdout_content_sha256",
        "reference_lock_sha256",
        "reference_locked_before_policy_evaluation",
        "neural_completed_seeds",
        "neural_artifact_sha256",
        "completed_cells",
        "completed_rounds",
        "remaining_rounds",
        "invocations",
        "resume_cleanups",
        "completed_at",
        "primary_passed",
        "assessment_sha256",
        "manifest_sha256",
        "checksums_sha256",
    }
    if not _is_confirmatory_recipe(recipe):
        expected_keys.add("sensitivity_passed")
    if set(state) != expected_keys:
        raise HGSHoldoutError("completed holdout run-state schema changed")
    if (
        state.get("schema_version") != RUN_STATE_SCHEMA
        or state.get("status") not in COMPLETE_STATUSES
        or not _strict_json_equal(state.get("classification"), _classification(recipe))
        or state.get("reference_locked_before_policy_evaluation") is not True
        or state.get("neural_completed_seeds") != list(recipe.neural_policy.training_seeds)
    ):
        raise HGSHoldoutError("completed holdout run-state invariants changed")
    neural_hashes = state.get("neural_artifact_sha256")
    if (
        not isinstance(neural_hashes, dict)
        or set(neural_hashes) != {str(seed) for seed in recipe.neural_policy.training_seeds}
        or any(
            not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None
            for digest in neural_hashes.values()
        )
    ):
        raise HGSHoldoutError("completed holdout neural artifact hashes changed")
    for field in (
        "recipe_sha256",
        "uv_lock_sha256",
        "selection_source_receipt_sha256",
        "replication_source_receipt_sha256",
        "holdout_file_sha256",
        "holdout_content_sha256",
        "reference_lock_sha256",
        "assessment_sha256",
        "manifest_sha256",
        "checksums_sha256",
    ):
        value = state.get(field)
        if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
            raise HGSHoldoutError(f"completed holdout has invalid {field}")
    for field in (
        "created_at",
        "holdout_opening_authorized_at",
        "holdout_opened_at",
        "completed_at",
    ):
        if not isinstance(state.get(field), str):
            raise HGSHoldoutError(f"completed holdout has invalid {field}")
    schedule = [
        {
            "round": index,
            "seed": seed,
            "budget_order": list(recipe.hgs_policies.budget_orders[index]),
        }
        for index, seed in enumerate(recipe.hgs_policies.seeds)
    ]
    all_paths = [relative for *_prefix, relative in _expected_cells(recipe)]
    all_rounds = list(range(len(recipe.hgs_policies.seeds)))
    decision_valid = (
        state.get("status") == _assessment_status(bool(state.get("primary_passed")), recipe=recipe)
        if _is_confirmatory_recipe(recipe)
        else isinstance(state.get("sensitivity_passed"), bool)
        and state.get("status")
        == _assessment_status(
            bool(state.get("primary_passed")),
            bool(state.get("sensitivity_passed")),
        )
    )
    if (
        not _strict_json_equal(state.get("round_schedule"), schedule)
        or state.get("completed_cells") != all_paths
        or state.get("completed_rounds") != all_rounds
        or state.get("remaining_rounds") != []
        or not isinstance(state.get("primary_passed"), bool)
        or not decision_valid
    ):
        raise HGSHoldoutError("completed holdout progress or decision changed")
    _validate_invocation_history(recipe, state, require_closed=True)


def _completed_frozen_input_check(
    *,
    source_recipe: Path,
    recipe_bytes: bytes,
    output: Path,
    state: dict[str, Any],
) -> None:
    """Validate a completed bundle without CUDA or live-source qualification."""

    try:
        current_recipe = source_recipe.read_bytes()
    except OSError as exc:
        raise HGSHoldoutError("completed holdout recipe became unavailable") from exc
    frozen_recipe = output / "recipe.yaml"
    frozen_uv_lock = output / "environment" / "uv.lock"
    if (
        _canonical_input_hash(current_recipe) != _canonical_input_hash(recipe_bytes)
        or _canonical_input_hash(current_recipe) != state.get("recipe_sha256")
        or not frozen_recipe.is_file()
        or _canonical_file_sha256(frozen_recipe)[0] != state.get("recipe_sha256")
        or not frozen_uv_lock.is_file()
        or _canonical_file_sha256(frozen_uv_lock)[0] != state.get("uv_lock_sha256")
    ):
        raise HGSHoldoutError("completed holdout frozen inputs changed")


def _validate_completed_bundle(
    recipe: AETHGSHoldoutRecipe,
    quality_recipe: Any,
    evidence: SourceEvidence,
    *,
    source_recipe: Path,
    recipe_bytes: bytes,
    output: Path,
    state: dict[str, Any],
) -> HoldoutResult:
    _validate_completed_run_state(recipe, state)
    _completed_frozen_input_check(
        source_recipe=source_recipe,
        recipe_bytes=recipe_bytes,
        output=output,
        state=state,
    )
    try:
        quality._verify_checksums(output, expected_sha256=state["checksums_sha256"])
    except Exception as exc:
        raise HGSHoldoutError("completed holdout checksums changed") from exc
    if quality._stale_partial_artifacts(output):
        raise HGSHoldoutError("completed holdout contains atomic partial artifacts")
    if not _strict_json_equal(
        quality._load_json(_selection_receipt_path(output)),
        evidence.selection_receipt,
    ) or quality._sha256_file(_selection_receipt_path(output)) != state.get(
        "selection_source_receipt_sha256"
    ):
        raise HGSHoldoutError("completed holdout selection receipt changed")
    if not _strict_json_equal(
        quality._load_json(_replication_receipt_path(output)),
        evidence.replication_evidence.receipt,
    ) or quality._sha256_file(_replication_receipt_path(output)) != state.get(
        "replication_source_receipt_sha256"
    ):
        raise HGSHoldoutError("completed holdout replication receipt changed")

    corpus = quality._load_corpus(
        quality_recipe,
        "holdout",
        quality_recipe.dataset.holdout,
        _relative(output, recipe.dataset.holdout.artifact),
    )
    if (
        corpus.file_sha256 != state["holdout_file_sha256"]
        or corpus.content_sha256 != state["holdout_content_sha256"]
        or corpus.content_sha256 in _forbidden_content_hashes(recipe)
    ):
        raise HGSHoldoutError("completed holdout corpus anchor changed")
    reference, reference_lock = _prepare_reference(recipe, quality_recipe, output, state, corpus)
    reference_sha256 = quality._sha256_file(output / "reference" / "holdout" / "reference.json")
    neural_evaluations = _prepare_neural_evaluations(
        recipe,
        output,
        state,
        evidence.replication_evidence,
        quality_recipe,
        corpus,
        reference,
        reference_sha256,
    )
    cells = _load_cells(
        recipe,
        output=output,
        state=state,
        corpus=corpus,
        reference=reference,
        reference_sha256=reference_sha256,
    )
    _reconcile_cell_invocation_links(
        recipe,
        output,
        state,
        cells,
        allow_recovery=False,
    )
    _validate_invocation_history(recipe, state, require_closed=True)
    assessment = quality._load_json(output / "holdout-assessment.json")
    recomputed = _build_assessment(recipe, neural_evaluations, cells)
    sensitivity_matches = (
        "sensitivity_passed" not in assessment and "sensitivity_passed" not in state
        if _is_confirmatory_recipe(recipe)
        else assessment.get("sensitivity_passed") == state.get("sensitivity_passed")
    )
    if (
        not _strict_json_equal(assessment, recomputed)
        or quality._sha256_file(output / "holdout-assessment.json") != state["assessment_sha256"]
        or assessment.get("status") != state["status"]
        or assessment.get("primary_passed") != state["primary_passed"]
        or not sensitivity_matches
    ):
        raise HGSHoldoutError("completed holdout assessment changed")
    _assert_exact_artifact_inventory(recipe, output, reference)

    manifest_path = output / "manifest.json"
    if quality._sha256_file(manifest_path) != state["manifest_sha256"]:
        raise HGSHoldoutError("completed holdout manifest hash changed")
    manifest = quality._load_json(manifest_path)
    expected_manifest = _manifest_payload(
        recipe,
        output=output,
        state=state,
        corpus=corpus,
        reference=reference,
        reference_lock=reference_lock,
        neural_evaluations=neural_evaluations,
        cells=cells,
        assessment=assessment,
        ended_at=str(manifest.get("ended_at")),
    )
    if (
        not _strict_json_equal(manifest, expected_manifest)
        or manifest.get("ended_at") != state["completed_at"]
    ):
        raise HGSHoldoutError("completed holdout manifest relationships changed")
    return HoldoutResult(
        path=output,
        status=state["status"],
        complete=True,
        completed_rounds=tuple(state["completed_rounds"]),
        remaining_rounds=(),
        primary_passed=state["primary_passed"],
        sensitivity_passed=state.get("sensitivity_passed"),
        manifest_path=manifest_path,
        manifest_sha256=state["manifest_sha256"],
        confirmatory_profile=_is_confirmatory_recipe(recipe),
    )


def _finalize(
    recipe: AETHGSHoldoutRecipe,
    quality_recipe: Any,
    evidence: SourceEvidence,
    *,
    root: Path,
    source_recipe: Path,
    recipe_bytes: bytes,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_lock: dict[str, Any],
    neural_evaluations: Mapping[int, dict[str, Any]],
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> HoldoutResult:
    _validate_invocation_history(recipe, state, require_closed=True)
    _reconcile_cell_invocation_links(recipe, output, state, cells)
    assessment = _build_assessment(recipe, neural_evaluations, cells)
    assessment_path = output / "holdout-assessment.json"
    quality._atomic_write_json(assessment_path, assessment)
    if not _strict_json_equal(
        quality._load_json(assessment_path),
        _build_assessment(recipe, neural_evaluations, cells),
    ):
        raise HGSHoldoutError("holdout assessment changed after atomic write")

    _revalidate_live_inputs(
        recipe=recipe,
        source_recipe=source_recipe,
        recipe_bytes=recipe_bytes,
        root=root,
        output=output,
        state=state,
    )
    _revalidate_sources(recipe, root, evidence)
    all_paths, all_rounds, remaining = _progress(recipe, cells)
    if remaining:
        raise HGSHoldoutError("cannot finalize an incomplete holdout grid")
    ended_at = datetime.now(UTC).isoformat()
    candidate_state = {
        **state,
        "status": assessment["status"],
        "completed_at": ended_at,
        "completed_cells": all_paths,
        "completed_rounds": all_rounds,
        "remaining_rounds": [],
        "primary_passed": assessment["primary_passed"],
        **(
            {}
            if _is_confirmatory_recipe(recipe)
            else {"sensitivity_passed": assessment["sensitivity_passed"]}
        ),
        "assessment_sha256": quality._sha256_file(assessment_path),
    }
    manifest = _manifest_payload(
        recipe,
        output=output,
        state=candidate_state,
        corpus=corpus,
        reference=reference,
        reference_lock=reference_lock,
        neural_evaluations=neural_evaluations,
        cells=cells,
        assessment=assessment,
        ended_at=ended_at,
    )
    manifest_path = output / "manifest.json"
    quality._atomic_write_json(manifest_path, manifest)
    checksum_path = quality._write_checksums(output)
    candidate_state["manifest_sha256"] = quality._sha256_file(manifest_path)
    candidate_state["checksums_sha256"] = quality._sha256_file(checksum_path)
    result = _validate_completed_bundle(
        recipe,
        quality_recipe,
        evidence,
        source_recipe=source_recipe,
        recipe_bytes=recipe_bytes,
        output=output,
        state=candidate_state,
    )
    quality._atomic_write_json(output / "run-state.json", candidate_state)
    state.clear()
    state.update(candidate_state)
    return result


def execute_hgs_holdout(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    resume: bool = False,
) -> HoldoutResult:
    """Execute or resume at most one complete paired HGS seed round."""

    try:
        root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
        source_recipe = Path(recipe_path).resolve(strict=True)
    except OSError as exc:
        raise HGSHoldoutQualificationError(
            f"holdout workspace or recipe is unavailable: {exc}"
        ) from exc
    if root != Path.cwd().resolve(strict=True):
        raise HGSHoldoutQualificationError("workspace_root must be the current repository")
    try:
        source_recipe.relative_to(root)
    except ValueError as exc:
        raise HGSHoldoutQualificationError(
            "holdout recipe must be stored inside the repository"
        ) from exc
    try:
        recipe_bytes = source_recipe.read_bytes()
        recipe = load_aet_hgs_holdout_recipe(source_recipe)
    except Exception as exc:
        raise HGSHoldoutQualificationError(f"holdout recipe is invalid: {exc}") from exc
    if source_recipe.read_bytes() != recipe_bytes:
        raise HGSHoldoutQualificationError("holdout recipe changed while it was being loaded")

    output = quality._safe_output_target(root, recipe.output_root)
    if output.exists() and not (output / "run-state.json").is_file():
        raise HGSHoldoutQualificationError(
            "holdout output exists without its durable pre-opening state anchor"
        )
    if (output / "run-state.json").is_file():
        existing_state = quality._load_json(output / "run-state.json")
        if existing_state.get("status") in {INVALIDATED_STATUS, INVALIDATED_INPUT_STATUS}:
            raise HGSHoldoutError(
                "holdout was permanently invalidated by an integrity or input failure; "
                "diagnose it and use a new preregistered output_root"
            )
        if existing_state.get("status") in COMPLETE_STATUSES:
            evidence = _validate_sources(recipe, root)
            quality_recipe = _holdout_quality_recipe(
                recipe, evidence.replication_evidence.base_recipe
            )
            return _validate_completed_bundle(
                recipe,
                quality_recipe,
                evidence,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                output=output,
                state=existing_state,
            )
        if existing_state.get("status") not in {
            "initialized_before_holdout_open",
            INCOMPLETE_STATUS,
        }:
            raise HGSHoldoutError("holdout run-state has an unknown active status")

    qualification = runtime_qualification(recipe, repository_root=root)
    if qualification.get("ready_to_execute") is not True:
        raise HGSHoldoutQualificationError(
            "native-Windows holdout qualification is incomplete; no holdout work ran"
        )
    evidence = _validate_sources(recipe, root)
    quality_recipe = _holdout_quality_recipe(recipe, evidence.replication_evidence.base_recipe)
    runtime_identity = _runtime_identity(recipe, quality_recipe)

    with quality._output_lock(output):
        if source_recipe.read_bytes() != recipe_bytes:
            raise HGSHoldoutQualificationError("holdout recipe changed before initialization")
        output, state = _prepare_output(
            recipe,
            recipe_bytes=recipe_bytes,
            root=root,
            evidence=evidence,
            runtime_identity=runtime_identity,
            resume=resume,
        )
        if state.get("status") in COMPLETE_STATUSES:
            return _validate_completed_bundle(
                recipe,
                quality_recipe,
                evidence,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                output=output,
                state=state,
            )

        _revalidate_live_inputs(
            recipe=recipe,
            source_recipe=source_recipe,
            recipe_bytes=recipe_bytes,
            root=root,
            output=output,
            state=state,
        )
        _revalidate_sources(recipe, root, evidence)
        if not isinstance(state.get("invocations"), list):
            raise HGSHoldoutError("holdout invocation history is invalid")
        if state["invocations"]:
            _validate_invocation_history(recipe, state, require_closed=False)

        def revalidate_work_unit(work_unit: str, phase: str) -> None:
            _revalidate_work_unit_inputs(
                recipe=recipe,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                root=root,
                output=output,
                state=state,
                evidence=evidence,
                work_unit=work_unit,
                phase=phase,
            )

        corpus = _prepare_holdout_corpus(recipe, quality_recipe, output, state)
        reference, reference_lock = _prepare_reference(
            recipe,
            quality_recipe,
            output,
            state,
            corpus,
            revalidate=revalidate_work_unit,
        )
        reference_sha256 = quality._sha256_file(output / "reference" / "holdout" / "reference.json")
        neural_evaluations = _prepare_neural_evaluations(
            recipe,
            output,
            state,
            evidence.replication_evidence,
            quality_recipe,
            corpus,
            reference,
            reference_sha256,
            revalidate=revalidate_work_unit,
        )
        cells = _load_cells(
            recipe,
            output=output,
            state=state,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            revalidate=revalidate_work_unit,
        )
        _reconcile_cell_invocation_links(recipe, output, state, cells)
        completed_rounds, remaining = _reconcile_progress(recipe, output, state, cells)
        recovered_complete_round = _close_recovered_complete_invocation(
            recipe,
            output,
            state,
        )
        if recovered_complete_round and remaining:
            state["status"] = INCOMPLETE_STATUS
            quality._atomic_write_json(output / "run-state.json", state)
            return HoldoutResult(
                path=output,
                status=INCOMPLETE_STATUS,
                complete=False,
                completed_rounds=tuple(completed_rounds),
                remaining_rounds=tuple(remaining),
                confirmatory_profile=_is_confirmatory_recipe(recipe),
            )
        if not remaining:
            return _finalize(
                recipe,
                quality_recipe,
                evidence,
                root=root,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                output=output,
                state=state,
                corpus=corpus,
                reference=reference,
                reference_lock=reference_lock,
                neural_evaluations=neural_evaluations,
                cells=cells,
            )

        invocation_number = _start_invocation(recipe, output, state)
        try:
            _complete_one_round(
                recipe,
                output=output,
                state=state,
                corpus=corpus,
                reference=reference,
                reference_sha256=reference_sha256,
                cells=cells,
                invocation_number=invocation_number,
                revalidate=revalidate_work_unit,
            )
        except HGSHoldoutIntegrityError as exc:
            _invalidate_integrity(output, state, exc)
            raise HGSHoldoutError(
                "HGS integrity failure invalidated this holdout output; "
                "a new preregistered output_root is required"
            ) from exc
        _finish_invocation(
            recipe,
            output,
            state,
            outcome=(
                "hgs_seed_round_completed"
                if _is_confirmatory_recipe(recipe)
                else "paired_seed_round_completed"
            ),
        )
        _reconcile_cell_invocation_links(recipe, output, state, cells)
        _revalidate_live_inputs(
            recipe=recipe,
            source_recipe=source_recipe,
            recipe_bytes=recipe_bytes,
            root=root,
            output=output,
            state=state,
        )
        _revalidate_sources(recipe, root, evidence)
        completed_rounds, remaining = _reconcile_progress(recipe, output, state, cells)
        if not remaining:
            return _finalize(
                recipe,
                quality_recipe,
                evidence,
                root=root,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                output=output,
                state=state,
                corpus=corpus,
                reference=reference,
                reference_lock=reference_lock,
                neural_evaluations=neural_evaluations,
                cells=cells,
            )
        state["status"] = INCOMPLETE_STATUS
        quality._atomic_write_json(output / "run-state.json", state)
        return HoldoutResult(
            path=output,
            status=INCOMPLETE_STATUS,
            complete=False,
            completed_rounds=tuple(completed_rounds),
            remaining_rounds=tuple(remaining),
            confirmatory_profile=_is_confirmatory_recipe(recipe),
        )


def _result_payload(result: HoldoutResult) -> dict[str, Any]:
    decisions = (
        {"confirmatory_passed": result.primary_passed}
        if result.confirmatory_profile
        else {
            "primary_passed": result.primary_passed,
            "sensitivity_passed": result.sensitivity_passed,
        }
    )
    return {
        "status": result.status,
        "path": result.path.as_posix(),
        "complete": result.complete,
        "completed_rounds": list(result.completed_rounds),
        "remaining_rounds": list(result.remaining_rounds),
        **decisions,
        "manifest": result.manifest_path.as_posix() if result.manifest_path else None,
        "manifest_sha256": result.manifest_sha256,
        **_classification(confirmatory=result.confirmatory_profile),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = execute_hgs_holdout(args.recipe, resume=args.resume)
    except (
        HGSHoldoutError,
        frontier.HGSFrontierError,
        quality.QualityPilotError,
        replication.QualityReplicationError,
    ) as exc:
        parser.exit(2, f"HGS holdout failed: {exc}\n")
    print(json.dumps(_result_payload(result), indent=2, sort_keys=True), flush=True)
    if not result.complete:
        return 4
    if result.primary_passed is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HGSHoldoutError",
    "HGSHoldoutIntegrityError",
    "HGSHoldoutQualificationError",
    "HoldoutResult",
    "dry_run",
    "execute_hgs_holdout",
    "load_aet_hgs_holdout_recipe",
    "main",
    "runtime_qualification",
]
