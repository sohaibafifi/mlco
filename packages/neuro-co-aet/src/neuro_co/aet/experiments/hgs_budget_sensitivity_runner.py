"""Measure the fixed HGS budget sensitivity grid on native Windows.

This sidecar consumes the finalized seed-2723 batch frontier and its sealed
quality reference.  It measures HGS only.  It never trains a model, loads a
checkpoint, executes neural inference, probes neural batch capacity, or
creates a reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any

from neuro_co.aet.experiments import deployment_energy_runner as energy_runner
from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments import software_recipe
from neuro_co.aet.experiments.hgs_budget_sensitivity_recipe import (
    CLASSIFICATION,
    HGSBudgetSensitivityRecipe,
    SensitivityBlock,
    dry_run,
    expected_blocks,
    load_recipe,
    qualify_for_execution,
)

RUN_STATE_SCHEMA = "aet-hgs-budget-sensitivity-run-state/v1"
BLOCK_SCHEMA = "aet-hgs-budget-sensitivity-block/v1"
SUMMARY_SCHEMA = "aet-hgs-budget-sensitivity-summary/v1"
MANIFEST_SCHEMA = "aet-hgs-budget-sensitivity-manifest/v1"
SOURCE_RECEIPT_SCHEMA = "aet-hgs-budget-sensitivity-source-receipt/v1"
INCOMPLETE_STATUS = "incomplete"
COMPLETE_STATUS = "complete"

ATTESTED_ENV = "AET_HGS_BUDGET_EXCLUSIVE_ATTESTED"
ATTESTED_AT_ENV = "AET_HGS_BUDGET_EXCLUSIVE_ATTESTED_AT"
ATTESTATION_SESSION_ENV = "AET_HGS_BUDGET_EXCLUSIVE_SESSION_ID"


class HGSBudgetSensitivityError(RuntimeError):
    """Raised when the sidecar cannot continue without weakening its contract."""


class HGSBudgetSensitivityQualificationError(HGSBudgetSensitivityError):
    """Raised before measurement when the host or immutable inputs are invalid."""


@dataclass(frozen=True, slots=True)
class HGSBudgetSensitivityResult:
    path: Path
    status: str
    completed_blocks: int
    manifest_path: Path
    manifest_sha256: str
    combined_surface_eligible: bool


@dataclass(frozen=True, slots=True)
class SourceContext:
    receipt: dict[str, Any]
    corpus: energy_runner.Corpus
    reference_costs: tuple[float, ...]
    reference_sha256: str
    old_hgs10_blocks: tuple[dict[str, Any], ...]


@contextmanager
def _campaign_lock(output: Path):
    """Hold one cross-process lock for the whole read-measure-seal transaction."""

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.runner.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise HGSBudgetSensitivityError(
                f"another HGS budget-sensitivity process is active for {output}"
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _classification() -> dict[str, Any]:
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
        raise HGSBudgetSensitivityError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, dict):
        raise HGSBudgetSensitivityError(f"JSON artifact is not an object: {path}")
    return value


def _relative(root: Path, value: str) -> Path:
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise HGSBudgetSensitivityError(f"unsafe relative path: {value!r}")
    return root.joinpath(*parsed.parts)


def _safe_output(root: Path, value: str) -> Path:
    repository = root.resolve(strict=True)
    if repository != Path.cwd().resolve(strict=True):
        raise HGSBudgetSensitivityError("workspace_root must be the current Git checkout")
    output = _relative(repository, value)
    cursor = repository
    for component in output.relative_to(repository).parts:
        cursor /= component
        if (cursor.exists() or cursor.is_symlink()) and cursor.is_symlink():
            raise HGSBudgetSensitivityError(f"refusing symlinked output component: {cursor}")
    try:
        output.resolve(strict=False).relative_to(repository)
    except ValueError as exc:
        raise HGSBudgetSensitivityError("output root resolves outside the repository") from exc
    return output


def _artifact_value(artifact: Any, name: str) -> Any:
    if isinstance(artifact, Mapping):
        return artifact.get(name)
    return getattr(artifact, name, None)


def _validate_declared_artifacts(
    source_root: Path, artifacts: Sequence[Any]
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for artifact in artifacts:
        relative = _artifact_value(artifact, "path")
        expected_sha = _artifact_value(artifact, "sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise HGSBudgetSensitivityQualificationError("source artifact identity is malformed")
        path = _relative(source_root, relative)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_sha:
            raise HGSBudgetSensitivityQualificationError(
                f"immutable source artifact changed: {relative}"
            )
        records.append({"path": relative, "sha256": expected_sha})
    return records


def _validate_checksum_inventory(source_root: Path) -> dict[str, Any]:
    try:
        return energy_runner._validate_checksum_inventory(source_root)
    except Exception as exc:
        raise HGSBudgetSensitivityQualificationError(
            f"source checksum inventory failed: {source_root}"
        ) from exc


def _finite_costs(value: Any, *, count: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise HGSBudgetSensitivityQualificationError(f"{label} has the wrong cost cardinality")
    costs: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
        ):
            raise HGSBudgetSensitivityQualificationError(f"{label} contains an invalid cost")
        costs.append(float(item))
    return tuple(costs)


def _load_exact_corpus(
    recipe: HGSBudgetSensitivityRecipe,
    quality_root: Path,
    frontier_root: Path,
) -> energy_runner.Corpus:
    import numpy as np

    source_path = _relative(quality_root, recipe.dataset.artifact)
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            coords = np.ascontiguousarray(archive["coords"], dtype=np.float32)
            demands = np.ascontiguousarray(archive["demands"], dtype=np.float32)
            capacity = float(archive["capacity"].item())
            seed = int(archive["seed"].item())
            embedded = str(archive["content_sha256"].item())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HGSBudgetSensitivityQualificationError(
            "sealed seed2723 corpus is unreadable"
        ) from exc
    content_sha = quality._content_sha256(coords, demands, capacity)
    if (
        seed != recipe.dataset.seed
        or coords.shape[0] != recipe.dataset.num_instances
        or capacity != recipe.dataset.capacity
        or embedded != recipe.dataset.content_sha256
        or content_sha != recipe.dataset.content_sha256
    ):
        raise HGSBudgetSensitivityQualificationError("sealed seed2723 corpus identity changed")
    frontier_manifest = _load_json(frontier_root / "manifest.json")
    frontier_dataset = frontier_manifest.get("dataset")
    if not isinstance(frontier_dataset, dict):
        raise HGSBudgetSensitivityQualificationError("frontier dataset receipt is missing")
    frontier_path_value = frontier_dataset.get("path")
    if not isinstance(frontier_path_value, str):
        raise HGSBudgetSensitivityQualificationError("frontier corpus path is malformed")
    frontier_path = _relative(frontier_root, frontier_path_value)
    file_sha = _sha256_file(source_path)
    if (
        frontier_dataset.get("content_sha256") != content_sha
        or frontier_dataset.get("file_sha256") != file_sha
        or frontier_dataset.get("instances") != recipe.dataset.num_instances
        or frontier_dataset.get("seed") != recipe.dataset.seed
        or frontier_path.is_symlink()
        or not frontier_path.is_file()
        or _sha256_file(frontier_path) != file_sha
    ):
        raise HGSBudgetSensitivityQualificationError(
            "frontier and quality sources do not bind the exact same corpus"
        )
    return energy_runner.Corpus(
        coords=coords,
        demands=demands,
        capacity=capacity,
        content_sha256=content_sha,
        file_sha256=file_sha,
        path=source_path,
    )


def _old_hgs10_blocks(
    recipe: HGSBudgetSensitivityRecipe,
    frontier_root: Path,
    corpus: energy_runner.Corpus,
) -> tuple[dict[str, Any], ...]:
    candidates = sorted((frontier_root / "blocks").glob("round-*/order-*-hgs.json"))
    if len(candidates) != len(recipe.hgs_policy.seeds):
        raise HGSBudgetSensitivityQualificationError(
            "finalized frontier does not contain exactly five HGS-10 blocks"
        )
    by_round: dict[int, dict[str, Any]] = {}
    for path in candidates:
        payload = _load_json(path)
        round_index = payload.get("round")
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise HGSBudgetSensitivityQualificationError("frontier HGS round is malformed")
        if not 0 <= round_index < len(recipe.hgs_policy.seeds):
            raise HGSBudgetSensitivityQualificationError("frontier HGS round is out of range")
        expected_seed = recipe.hgs_policy.seeds[round_index]
        records = payload.get("solver_result_metadata")
        validation = payload.get("validation")
        energy = payload.get("energy")
        if (
            payload.get("schema_version") != "aet-batch-frontier-block/v1"
            or payload.get("status") != "complete"
            or payload.get("policy") != "hgs"
            or payload.get("hgs_seed") != expected_seed
            or payload.get("dataset_content_sha256") != corpus.content_sha256
            or payload.get("first_and_last_outputs_identical") is not True
            or payload.get("first_measured_output_sha256")
            != payload.get("last_measured_output_sha256")
            or not isinstance(validation, dict)
            or validation.get("complete") is not True
            or not isinstance(energy, dict)
            or not isinstance(records, list)
            or len(records) != recipe.dataset.num_instances
        ):
            raise HGSBudgetSensitivityQualificationError("frontier HGS-10 block is invalid")
        _finite_costs(
            validation.get("costs"), count=recipe.dataset.num_instances, label="frontier HGS-10"
        )
        for instance_index, record in enumerate(records):
            integer_cost = record.get("integer_cost") if isinstance(record, dict) else None
            reported_cost = record.get("reported_cost") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("instance_index") != instance_index
                or record.get("effective_seed") != expected_seed + instance_index
                or record.get("max_iterations") != recipe.drift_bridge.remeasurement_budget
                or record.get("scaling_factor") != recipe.hgs_policy.scaling_factor
                or isinstance(integer_cost, bool)
                or not isinstance(integer_cost, int)
                or isinstance(reported_cost, bool)
                or not isinstance(reported_cost, (int, float))
                or not math.isclose(
                    float(reported_cost),
                    integer_cost / recipe.hgs_policy.scaling_factor,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise HGSBudgetSensitivityQualificationError(
                    "frontier HGS-10 solver metadata changed"
                )
        cpu = energy.get("cpu_package_energy_j_per_instance")
        if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or float(cpu) <= 0:
            raise HGSBudgetSensitivityQualificationError("frontier HGS-10 energy is invalid")
        if round_index in by_round:
            raise HGSBudgetSensitivityQualificationError("duplicate frontier HGS-10 round")
        by_round[round_index] = payload
    if set(by_round) != set(range(5)):
        raise HGSBudgetSensitivityQualificationError("frontier HGS-10 rounds are incomplete")
    return tuple(by_round[index] for index in range(5))


def _validate_sources(
    recipe: HGSBudgetSensitivityRecipe,
    root: Path,
) -> SourceContext:
    frontier_root = _relative(root, recipe.frontier_source.root)
    quality_root = _relative(root, recipe.quality_source.root)
    if any(path.is_symlink() or not path.is_dir() for path in (frontier_root, quality_root)):
        raise HGSBudgetSensitivityQualificationError("immutable source root is unavailable")
    frontier_artifacts = _validate_declared_artifacts(
        frontier_root, recipe.frontier_source.artifacts
    )
    quality_artifacts = _validate_declared_artifacts(quality_root, recipe.quality_source.artifacts)
    frontier_inventory = _validate_checksum_inventory(frontier_root)
    quality_inventory = _validate_checksum_inventory(quality_root)

    frontier_manifest = _load_json(frontier_root / "manifest.json")
    frontier_state = _load_json(frontier_root / "run-state.json")
    frontier_summary = _load_json(frontier_root / "batch-frontier-summary.json")
    if (
        frontier_manifest.get("schema_version") != recipe.frontier_source.manifest_schema
        or frontier_manifest.get("status") != recipe.frontier_source.expected_status
        or frontier_state.get("status") != recipe.frontier_source.expected_status
        or frontier_summary.get("schema_version") != "aet-batch-frontier-summary/v1"
        or frontier_summary.get("status") != recipe.frontier_source.expected_status
    ):
        raise HGSBudgetSensitivityQualificationError("batch frontier is not finalized")

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
        or decision.get("hgs_budget") != recipe.drift_bridge.remeasurement_budget
        or quality_assessment.get("policies", {}).get("hgs_b10", {}).get("passed") is not True
        or quality_assessment.get("quality_method", {}).get("threshold_pct")
        != recipe.quality_gate.threshold_pct
        or quality_manifest.get("dataset", {}).get("content_sha256")
        != recipe.dataset.content_sha256
    ):
        raise HGSBudgetSensitivityQualificationError(
            "seed2723 quality source no longer proves the frozen HGS-10 policy"
        )

    corpus = _load_exact_corpus(recipe, quality_root, frontier_root)
    lock_path = quality_root / "reference" / "reference-lock.json"
    lock = _load_json(lock_path)
    reference_record = lock.get("reference")
    if (
        lock.get("status") != "locked"
        or lock.get("locked_before_policy_evaluation") is not True
        or lock.get("dataset", {}).get("content_sha256") != corpus.content_sha256
        or not isinstance(reference_record, dict)
        or lock.get("source_binding", {}).get("git_sha") != recipe.quality_source.source_git_sha
        or lock.get("source_binding", {}).get("source_snapshot_sha256")
        != recipe.quality_source.source_snapshot_sha256
    ):
        raise HGSBudgetSensitivityQualificationError("sealed reference lock changed")
    reference_relative = reference_record.get("path")
    reference_sha = reference_record.get("sha256")
    if not isinstance(reference_relative, str) or not isinstance(reference_sha, str):
        raise HGSBudgetSensitivityQualificationError("sealed reference identity is malformed")
    reference_path = _relative(quality_root, reference_relative)
    if (
        reference_path.is_symlink()
        or not reference_path.is_file()
        or _sha256_file(reference_path) != reference_sha
    ):
        raise HGSBudgetSensitivityQualificationError("sealed reference changed")
    reference = _load_json(reference_path)
    if (
        reference.get("status") != "complete"
        or reference.get("dataset_content_sha256") != corpus.content_sha256
        or reference.get("validation", {}).get("complete") is not True
    ):
        raise HGSBudgetSensitivityQualificationError("sealed reference is invalid")
    reference_costs = _finite_costs(
        reference.get("costs"), count=recipe.dataset.num_instances, label="sealed reference"
    )
    if tuple(reference.get("validation", {}).get("costs", ())) != reference_costs:
        raise HGSBudgetSensitivityQualificationError(
            "sealed reference costs disagree with their route validation"
        )
    old_blocks = _old_hgs10_blocks(recipe, frontier_root, corpus)

    frontier_hashes = {item["path"]: item["sha256"] for item in frontier_artifacts}
    receipt = {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "source_gates_passed": True,
        "batch_frontier": {
            "root": recipe.frontier_source.root,
            "status": recipe.frontier_source.expected_status,
            "manifest_sha256": frontier_hashes["manifest.json"],
            "batch_frontier_summary_sha256": frontier_hashes["batch-frontier-summary.json"],
            "checksums_sha256": frontier_hashes["SHA256SUMS"],
            "checksum_entries": frontier_inventory["entry_count"],
            "artifacts": frontier_artifacts,
            "old_hgs10_blocks": [
                {
                    "round": index,
                    "hgs_seed": recipe.hgs_policy.seeds[index],
                    "path": candidates.relative_to(frontier_root).as_posix(),
                    "sha256": _sha256_file(candidates),
                }
                for index, candidates in enumerate(
                    sorted((frontier_root / "blocks").glob("round-*/order-*-hgs.json"))
                )
            ],
        },
        "quality_source": {
            "root": recipe.quality_source.root,
            "status": recipe.quality_source.expected_status,
            "git_sha": recipe.quality_source.source_git_sha,
            "source_snapshot_sha256": recipe.quality_source.source_snapshot_sha256,
            "checksums_sha256": _sha256_file(quality_root / "SHA256SUMS"),
            "checksum_entries": quality_inventory["entry_count"],
            "artifacts": quality_artifacts,
            "reference_path": reference_relative,
            "reference_sha256": reference_sha,
            "reference_lock_sha256": _sha256_file(lock_path),
        },
        "dataset": {
            "id": recipe.dataset.dataset_id,
            "seed": recipe.dataset.seed,
            "instances": recipe.dataset.num_instances,
            "content_sha256": corpus.content_sha256,
            "file_sha256": corpus.file_sha256,
            "quality_source_path": recipe.dataset.artifact,
            "reference_path": reference_relative,
            "reference_sha256": reference_sha,
            "reference_lock_sha256": _sha256_file(lock_path),
        },
    }
    return SourceContext(receipt, corpus, reference_costs, reference_sha, old_blocks)


def _exclusive_attestation(recipe: HGSBudgetSensitivityRecipe) -> dict[str, Any]:
    if os.environ.get(ATTESTED_ENV) != "1":
        raise HGSBudgetSensitivityQualificationError(
            "exclusive-use attestation is absent; use the Windows launcher"
        )
    raw_at = os.environ.get(ATTESTED_AT_ENV)
    session_id = os.environ.get(ATTESTATION_SESSION_ENV)
    if not raw_at or not session_id:
        raise HGSBudgetSensitivityQualificationError("exclusive-use attestation is incomplete")
    try:
        attested_at = datetime.fromisoformat(raw_at.replace("Z", "+00:00")).astimezone(UTC)
        uuid.UUID(session_id)
    except (TypeError, ValueError) as exc:
        raise HGSBudgetSensitivityQualificationError(
            "exclusive-use attestation is invalid"
        ) from exc
    age_s = (datetime.now(UTC) - attested_at).total_seconds()
    limit = min(
        recipe.campaign_attestation_max_age_s,
        recipe.maximum_campaign_walltime_s,
    )
    if age_s < -60 or age_s > limit:
        raise HGSBudgetSensitivityQualificationError("exclusive-use campaign has expired")
    return {
        "session_id": session_id,
        "attested_at": attested_at.isoformat(),
        "age_s_at_process_start": age_s,
        "operator_supplied": True,
        "operator_attestation_is_authoritative": True,
        "gpu_process_lists_are_diagnostic_only": True,
    }


def _assert_attestation_active(
    recipe: HGSBudgetSensitivityRecipe, attestation: Mapping[str, Any]
) -> None:
    attested_at = datetime.fromisoformat(str(attestation["attested_at"]))
    age_s = (datetime.now(UTC) - attested_at).total_seconds()
    if age_s > min(recipe.campaign_attestation_max_age_s, recipe.maximum_campaign_walltime_s):
        raise HGSBudgetSensitivityQualificationError(
            "exclusive-use campaign exceeded its fixed validity window"
        )


def _runtime_identity(recipe: HGSBudgetSensitivityRecipe) -> dict[str, Any]:
    return energy_runner._configure_runtime(recipe)  # type: ignore[arg-type]


def _git_snapshot(root: Path, preflight: Path) -> dict[str, Any]:
    snapshot = software_recipe._current_git_snapshot((preflight,))
    if (
        snapshot.get("available") is not True
        or not isinstance(snapshot.get("sha"), str)
        or not isinstance(snapshot.get("worktree_fingerprint_sha256"), str)
    ):
        raise HGSBudgetSensitivityQualificationError("Git source identity is unavailable")
    return snapshot


def _prepare_output(
    recipe_path: Path,
    recipe: HGSBudgetSensitivityRecipe,
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
    source_bytes = _json_bytes(source_receipt)
    source_sha = _sha256_bytes(source_bytes)
    preflight = _relative(root, recipe.preflight_report)
    git = _git_snapshot(root, preflight)
    if output.exists():
        if not resume:
            raise HGSBudgetSensitivityError(f"output already exists; use --resume: {output}")
        state = _load_json(output / "run-state.json")
        expected = {
            "schema_version": RUN_STATE_SCHEMA,
            "recipe_sha256": recipe_sha,
            "uv_lock_sha256": lock_sha,
            "git_sha": git["sha"],
            "worktree_fingerprint_sha256": git["worktree_fingerprint_sha256"],
            "runtime_identity": runtime_identity,
            "source_receipt_sha256": source_sha,
            "classification": _classification(),
            "schedule": [asdict(block) for block in expected_blocks(recipe)],
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise HGSBudgetSensitivityError("resume identity differs from initialized sidecar")
        for relative, digest in (
            ("recipe.yaml", recipe_sha),
            ("environment/uv.lock", lock_sha),
            ("source-receipt.json", source_sha),
        ):
            path = output / relative
            if not path.is_file() or _sha256_file(path) != digest:
                raise HGSBudgetSensitivityError(f"frozen sidecar artifact changed: {relative}")
        if state.get("status") != INCOMPLETE_STATUS:
            raise HGSBudgetSensitivityError("resume state has an unknown status")
        try:
            energy_runner._recover_interrupted_artifacts(
                recipe,  # type: ignore[arg-type]
                output,
                state,
            )
        except Exception as exc:
            raise HGSBudgetSensitivityError("interrupted sidecar recovery failed") from exc
        _validate_session_evidence(output, state)
        return output, state

    state = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": INCOMPLETE_STATUS,
        "created_at": datetime.now(UTC).isoformat(),
        "recipe_sha256": recipe_sha,
        "uv_lock_sha256": lock_sha,
        "git_sha": git["sha"],
        "worktree_fingerprint_sha256": git["worktree_fingerprint_sha256"],
        "runtime_identity": runtime_identity,
        "source_receipt_sha256": source_sha,
        "classification": _classification(),
        "schedule": [asdict(block) for block in expected_blocks(recipe)],
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
        (staging / "source-receipt.json").write_bytes(source_bytes)
        _atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _preserve_preflight(
    recipe: HGSBudgetSensitivityRecipe, root: Path, output: Path
) -> dict[str, str]:
    source = _relative(root, recipe.preflight_report)
    if source.is_symlink() or not source.is_file():
        raise HGSBudgetSensitivityQualificationError("qualified preflight report is unavailable")
    payload = source.read_bytes()
    digest = _sha256_bytes(payload)
    relative = f"qualification/preflight-{digest}.json"
    destination = _relative(output, relative)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise HGSBudgetSensitivityError("content-addressed preflight evidence changed")
    else:
        _atomic_write_bytes(destination, payload)
    return {"path": relative, "sha256": digest}


def _validate_session_evidence(output: Path, state: Mapping[str, Any]) -> None:
    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise HGSBudgetSensitivityError("session history is malformed")
    for session in sessions:
        if not isinstance(session, dict):
            raise HGSBudgetSensitivityError("session history entry is malformed")
        evidence = session.get("preflight_evidence")
        if not isinstance(evidence, dict):
            raise HGSBudgetSensitivityError("session lacks preserved preflight evidence")
        relative = evidence.get("path")
        digest = evidence.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise HGSBudgetSensitivityError("session preflight identity is malformed")
        path = _relative(output, relative)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise HGSBudgetSensitivityError("preserved session preflight changed")
    orphaned = state.get("orphaned_preflight_evidence", [])
    if not isinstance(orphaned, list):
        raise HGSBudgetSensitivityError("orphaned preflight evidence is malformed")
    for evidence in orphaned:
        if (
            not isinstance(evidence, dict)
            or evidence.get("reason") != "interrupted_before_session_state_promotion"
        ):
            raise HGSBudgetSensitivityError("orphaned preflight evidence is malformed")
        relative = evidence.get("path")
        digest = evidence.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise HGSBudgetSensitivityError("orphaned preflight identity is malformed")
        path = _relative(output, relative)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise HGSBudgetSensitivityError("orphaned preflight evidence changed")


def _semantic_cost_sha256(costs: Sequence[float]) -> str:
    return _sha256_bytes(_json_bytes({"costs": [float(value) for value in costs]}))


def _routes_sha256(routes: Any) -> str:
    return _sha256_bytes(_json_bytes({"routes": routes}))


def _execute_block(
    recipe: HGSBudgetSensitivityRecipe,
    block: SensitivityBlock,
    *,
    corpus: energy_runner.Corpus,
    reference_sha256: str,
    attestation: dict[str, Any],
) -> dict[str, Any]:
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

    def cycle(coords: Any = corpus.coords, demands: Any = corpus.demands) -> Sequence[Any]:
        return solve_corpus_sequential(
            coords,
            demands,
            corpus.capacity,
            seed=block.hgs_seed,
            max_iterations=block.budget,
            scaling_factor=recipe.hgs_policy.scaling_factor,
            collect_stats=recipe.hgs_policy.collect_stats,
        )

    warmup = cycle(corpus.coords[:4], corpus.demands[:4])
    warmup_routes, _ = energy_runner._validated_hgs_results(
        warmup,
        expected_instances=4,
        base_seed=block.hgs_seed,
        max_iterations=block.budget,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    if (
        quality.validate_routes(
            corpus.coords[:4], corpus.demands[:4], corpus.capacity, warmup_routes
        ).get("complete")
        is not True
    ):
        raise HGSBudgetSensitivityError("HGS warmup produced invalid routes")

    before = energy_runner._diagnostic_gpu_snapshot(recipe.gpu_index)
    tracker = energy_runner._make_strict_tracker(
        f"hgs-budget-b{block.budget:03d}-r{block.round_index:02d}-seed{block.hgs_seed}",
        recipe,  # type: ignore[arg-type]
    )
    repetitions = 0
    first_results: Sequence[Any] | None = None
    last_results: Sequence[Any] | None = None
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    with tracker as active:
        measured_started = time.perf_counter()
        while True:
            results = cycle()
            first_results = results if first_results is None else first_results
            last_results = results
            repetitions += 1
            if time.perf_counter() - wall_started > recipe.maximum_block_walltime_s:
                raise HGSBudgetSensitivityError("HGS sensitivity block exceeded maximum wall time")
            if time.perf_counter() - measured_started >= recipe.minimum_block_duration_s:
                break
        active.n_items = repetitions * recipe.dataset.num_instances
    ended_at = datetime.now(UTC)
    after = energy_runner._diagnostic_gpu_snapshot(recipe.gpu_index)
    if first_results is None or last_results is None:
        raise HGSBudgetSensitivityError("HGS sensitivity block completed no corpus repetition")

    first_routes, first_metadata = energy_runner._validated_hgs_results(
        first_results,
        expected_instances=recipe.dataset.num_instances,
        base_seed=block.hgs_seed,
        max_iterations=block.budget,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    last_routes, last_metadata = energy_runner._validated_hgs_results(
        last_results,
        expected_instances=recipe.dataset.num_instances,
        base_seed=block.hgs_seed,
        max_iterations=block.budget,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    first_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, first_routes
    )
    last_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, last_routes
    )
    if first_validation.get("complete") is not True or last_validation.get("complete") is not True:
        raise HGSBudgetSensitivityError("HGS measured block produced invalid routes")
    first_costs = _finite_costs(
        first_validation.get("costs"),
        count=recipe.dataset.num_instances,
        label="first measured HGS output",
    )
    last_costs = _finite_costs(
        last_validation.get("costs"),
        count=recipe.dataset.num_instances,
        label="last measured HGS output",
    )
    first_semantic = _semantic_cost_sha256(first_costs)
    last_semantic = _semantic_cost_sha256(last_costs)
    if first_costs != last_costs or first_semantic != last_semantic:
        raise HGSBudgetSensitivityError("HGS first and last semantic costs differ")
    instances = repetitions * recipe.dataset.num_instances
    measured_energy = energy_runner._checked_energy(
        tracker,
        recipe,  # type: ignore[arg-type]
        items_processed=instances,
    )
    first_route_hash = _routes_sha256(first_routes)
    last_route_hash = _routes_sha256(last_routes)
    return {
        "schema_version": BLOCK_SCHEMA,
        "status": "complete",
        "classification": _classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "budget": block.budget,
        "hgs_seed": block.hgs_seed,
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "attestation": attestation,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "repetitions": repetitions,
        "instances_per_repetition": recipe.dataset.num_instances,
        "instances_processed": instances,
        "duration_s": float(measured_energy["duration_s"]),
        "throughput_instances_per_s": instances / float(measured_energy["duration_s"]),
        "energy": measured_energy,
        "routes": first_routes,
        "last_routes": last_routes,
        "solver_result_metadata": first_metadata,
        "last_solver_result_metadata": last_metadata,
        "validation": first_validation,
        "last_validation": last_validation,
        "first_measured_semantic_cost_sha256": first_semantic,
        "last_measured_semantic_cost_sha256": last_semantic,
        "first_and_last_semantic_costs_identical": True,
        "first_measured_route_sha256": first_route_hash,
        "last_measured_route_sha256": last_route_hash,
        "first_and_last_routes_identical_diagnostic": first_route_hash == last_route_hash,
        "warmup_included_in_measurement": False,
        "validation_included_in_measurement": False,
        "serialization_included_in_measurement": False,
        "gpu_process_snapshot_before": before,
        "gpu_process_snapshot_after": after,
        "gpu_process_lists_are_diagnostic_only": True,
        "policy_details": {
            "solver": recipe.hgs_policy.solver,
            "budget_kind": "max_iterations",
            "max_iterations": block.budget,
            "base_seed": block.hgs_seed,
            "scaling_factor": recipe.hgs_policy.scaling_factor,
            "collect_stats": recipe.hgs_policy.collect_stats,
            "cpu_threads": recipe.hgs_policy.cpu_threads,
        },
    }


def _validate_block(
    payload: dict[str, Any],
    recipe: HGSBudgetSensitivityRecipe,
    block: SensitivityBlock,
    corpus: energy_runner.Corpus,
    reference_sha256: str,
    *,
    expected_session_id: str | None = None,
) -> None:
    expected = {
        "schema_version": BLOCK_SCHEMA,
        "status": "complete",
        "classification": _classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "budget": block.budget,
        "hgs_seed": block.hgs_seed,
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "instances_per_repetition": recipe.dataset.num_instances,
        "first_and_last_semantic_costs_identical": True,
        "gpu_process_lists_are_diagnostic_only": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise HGSBudgetSensitivityError(f"cached block identity changed: {block.relative_path}")
    if (
        expected_session_id is not None
        and payload.get("attestation", {}).get("session_id") != expected_session_id
    ):
        raise HGSBudgetSensitivityError("cached block attestation differs from its attempt")
    repetitions = payload.get("repetitions")
    instances = payload.get("instances_processed")
    duration = payload.get("duration_s")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or isinstance(instances, bool)
        or not isinstance(instances, int)
        or instances != repetitions * recipe.dataset.num_instances
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < recipe.minimum_block_duration_s
    ):
        raise HGSBudgetSensitivityError("cached block work metrics changed")
    instance_count = int(instances)
    measured_duration = float(duration)
    routes = payload.get("routes")
    last_routes = payload.get("last_routes")
    if not isinstance(routes, (list, tuple)) or not isinstance(last_routes, (list, tuple)):
        raise HGSBudgetSensitivityError("cached block routes are missing")
    first_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, routes
    )
    last_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, last_routes
    )
    if first_validation != payload.get("validation") or last_validation != payload.get(
        "last_validation"
    ):
        raise HGSBudgetSensitivityError("cached block route validation changed")
    first_costs = _finite_costs(
        first_validation.get("costs"), count=recipe.dataset.num_instances, label="cached HGS output"
    )
    last_costs = _finite_costs(
        last_validation.get("costs"),
        count=recipe.dataset.num_instances,
        label="cached final HGS output",
    )
    first_semantic = _semantic_cost_sha256(first_costs)
    last_semantic = _semantic_cost_sha256(last_costs)
    if (
        first_costs != last_costs
        or first_semantic != last_semantic
        or payload.get("first_measured_semantic_cost_sha256") != first_semantic
        or payload.get("last_measured_semantic_cost_sha256") != last_semantic
        or payload.get("first_measured_route_sha256") != _routes_sha256(routes)
        or payload.get("last_measured_route_sha256") != _routes_sha256(last_routes)
    ):
        raise HGSBudgetSensitivityError("cached block semantic identity changed")
    records = payload.get("solver_result_metadata")
    last_records = payload.get("last_solver_result_metadata")
    for candidate in (records, last_records):
        if not isinstance(candidate, list) or len(candidate) != recipe.dataset.num_instances:
            raise HGSBudgetSensitivityError("cached block solver metadata is incomplete")
        for instance_index, record in enumerate(candidate):
            integer_cost = record.get("integer_cost") if isinstance(record, dict) else None
            reported_cost = record.get("reported_cost") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("instance_index") != instance_index
                or record.get("effective_seed") != block.hgs_seed + instance_index
                or record.get("max_iterations") != block.budget
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
                raise HGSBudgetSensitivityError("cached block solver metadata changed")
    measured_energy = payload.get("energy")
    if not isinstance(measured_energy, dict):
        raise HGSBudgetSensitivityError("cached block energy is missing")
    for key in (
        "cpu_package_energy_j",
        "gpu_energy_j",
        "observed_component_energy_j",
        "cpu_package_energy_j_per_instance",
        "gpu_energy_j_per_instance",
        "observed_component_energy_j_per_instance",
    ):
        value = measured_energy.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise HGSBudgetSensitivityError(f"cached block energy changed: {key}")
    cpu_total = float(measured_energy["cpu_package_energy_j"])
    gpu_total = float(measured_energy["gpu_energy_j"])
    observed_total = float(measured_energy["observed_component_energy_j"])
    cpu_per_instance = float(measured_energy["cpu_package_energy_j_per_instance"])
    gpu_per_instance = float(measured_energy["gpu_energy_j_per_instance"])
    observed_per_instance = float(measured_energy["observed_component_energy_j_per_instance"])
    throughput = payload.get("throughput_instances_per_s")
    if (
        measured_energy.get("backend") != "hwcounters"
        or measured_energy.get("items_processed") != instance_count
        or not math.isclose(
            float(measured_energy.get("duration_s", math.nan)),
            measured_duration,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not math.isclose(observed_total, cpu_total + gpu_total, rel_tol=1e-12, abs_tol=1e-9)
        or not math.isclose(
            float(measured_energy.get("energy_j", math.nan)),
            observed_total,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not math.isclose(
            cpu_per_instance, cpu_total / instance_count, rel_tol=1e-12, abs_tol=1e-12
        )
        or not math.isclose(
            gpu_per_instance, gpu_total / instance_count, rel_tol=1e-12, abs_tol=1e-12
        )
        or not math.isclose(
            observed_per_instance,
            observed_total / instance_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or isinstance(throughput, bool)
        or not isinstance(throughput, (int, float))
        or not math.isclose(
            float(throughput),
            instance_count / measured_duration,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(measured_energy.get("throughput_items_per_s", math.nan)),
            instance_count / measured_duration,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or measured_energy.get("extra", {}).get("allow_fallback") is not False
        or measured_energy.get("extra", {}).get("tdp_fallback") is not False
        or set(measured_energy.get("energy_domains", ())) != {"cpu", "gpu"}
        or any(
            measured_energy.get(key) not in (None, 0, 0.0)
            for key in ("co2_operational_kg", "co2_embodied_kg", "co2_total_kg")
        )
    ):
        raise HGSBudgetSensitivityError("cached block energy arithmetic changed")
    if (
        measured_energy.get("whole_system_energy") is not False
        or measured_energy.get("carbon_accounting") != "none"
        or measured_energy.get("cross_solver_energy_comparable") is not False
    ):
        raise HGSBudgetSensitivityError("cached block energy boundary changed")


def _load_completed_blocks(
    recipe: HGSBudgetSensitivityRecipe,
    output: Path,
    state: dict[str, Any],
    source: SourceContext,
) -> list[dict[str, Any]]:
    _validate_session_evidence(output, state)
    schedule = expected_blocks(recipe)
    completed = state.get("completed_blocks")
    hashes = state.get("block_sha256")
    attempts = state.get("block_attempts")
    sessions = state.get("sessions")
    if (
        not isinstance(completed, list)
        or not isinstance(hashes, dict)
        or not isinstance(attempts, list)
        or not isinstance(sessions, list)
    ):
        raise HGSBudgetSensitivityError("durable block state is malformed")
    expected_prefix = [block.relative_path for block in schedule[: len(completed)]]
    if completed != expected_prefix or set(hashes) != set(completed):
        raise HGSBudgetSensitivityError("completed blocks are not the strict schedule prefix")
    known_sessions = {
        session.get("session_id") for session in sessions if isinstance(session, dict)
    }
    current = state.get("current_block")
    if current is not None and (
        not isinstance(current, dict)
        or len(completed) >= len(schedule)
        or current.get("relative_path") != schedule[len(completed)].relative_path
    ):
        raise HGSBudgetSensitivityError("current block is not the next schedule item")
    if current is not None:
        _validated_current_attempt(state, schedule[len(completed)])
    payloads: list[dict[str, Any]] = []
    for index, relative in enumerate(completed):
        path = _relative(output, relative)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != hashes[relative]:
            raise HGSBudgetSensitivityError(f"durable block changed: {relative}")
        successful = [
            attempt
            for attempt in attempts
            if isinstance(attempt, dict)
            and attempt.get("relative_path") == relative
            and attempt.get("artifact_sha256") == hashes[relative]
            and attempt.get("outcome") in {"complete", "complete_orphan_promotion_recovered"}
        ]
        if len(successful) != 1 or successful[0].get("session_id") not in known_sessions:
            raise HGSBudgetSensitivityError("completed block lacks one valid durable attempt")
        payload = _load_json(path)
        _validate_block(
            payload,
            recipe,
            schedule[index],
            source.corpus,
            source.reference_sha256,
            expected_session_id=str(successful[0]["session_id"]),
        )
        payloads.append(payload)
    return payloads


def _validated_current_attempt(state: Mapping[str, Any], block: SensitivityBlock) -> dict[str, Any]:
    current = state.get("current_block")
    attempts = state.get("block_attempts")
    sessions = state.get("sessions")
    if (
        not isinstance(current, dict)
        or not isinstance(attempts, list)
        or not isinstance(sessions, list)
    ):
        raise HGSBudgetSensitivityError("current block attempt state is malformed")
    attempt_index = current.get("attempt_index")
    if (
        isinstance(attempt_index, bool)
        or not isinstance(attempt_index, int)
        or not 0 <= attempt_index < len(attempts)
    ):
        raise HGSBudgetSensitivityError("current block attempt index is invalid")
    attempt = attempts[attempt_index]
    if not isinstance(attempt, dict):
        raise HGSBudgetSensitivityError("current block attempt record is malformed")
    expected = {
        "attempt_index": attempt_index,
        "relative_path": block.relative_path,
        "round": block.round_index,
        "order_within_round": block.order_index,
        "budget": block.budget,
        "hgs_seed": block.hgs_seed,
        "outcome": "running",
    }
    if current.get("relative_path") != block.relative_path or any(
        attempt.get(key) != value for key, value in expected.items()
    ):
        raise HGSBudgetSensitivityError("current block attempt identity changed")
    known_sessions = {
        session.get("session_id") for session in sessions if isinstance(session, dict)
    }
    if attempt.get("session_id") not in known_sessions:
        raise HGSBudgetSensitivityError("current block attempt refers to an unknown session")
    return attempt


def _bootstrap_policy_samples(
    matrices: Mapping[int, Any], *, replicates: int, seed: int
) -> dict[int, tuple[Any, Any]]:
    import numpy as np

    arrays = {budget: np.asarray(matrix, dtype=np.float64) for budget, matrix in matrices.items()}
    if not arrays or len({array.shape for array in arrays.values()}) != 1:
        raise HGSBudgetSensitivityError("quality matrices do not share one shape")
    shape = next(iter(arrays.values())).shape
    if len(shape) != 2 or shape[0] != 5 or shape[1] < 1 or replicates < 1:
        raise HGSBudgetSensitivityError("quality matrix or bootstrap dimensions changed")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise HGSBudgetSensitivityError("quality matrices contain non-finite gaps")
    samples = {
        budget: (
            np.empty(replicates, dtype=np.float64),
            np.empty(replicates, dtype=np.float64),
        )
        for budget in arrays
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    for replicate in range(replicates):
        instances = rng.integers(0, shape[1], size=shape[1])
        seeds = rng.integers(0, shape[0], size=shape[0])
        for budget, matrix in arrays.items():
            selected = matrix[seeds[:, None], instances[None, :]]
            samples[budget][0][replicate] = selected.mean()
            samples[budget][1][replicate] = np.quantile(selected, 0.95, method="linear")
    return samples


def _quality_summary(
    matrix: Any,
    *,
    budget: int,
    invalid_instances: int,
    recipe: HGSBudgetSensitivityRecipe,
    mean_samples: Any,
    q95_samples: Any,
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(matrix, dtype=np.float64)
    mean_draws = np.asarray(mean_samples, dtype=np.float64)
    q95_draws = np.asarray(q95_samples, dtype=np.float64)
    gate = recipe.quality_gate
    finite = bool(
        np.isfinite(values).all() and np.isfinite(mean_draws).all() and np.isfinite(q95_draws).all()
    )
    threshold = float(gate.threshold_pct)
    seed_means = values.mean(axis=1) if finite else None
    seed_q95 = np.quantile(values, 0.95, axis=1, method="linear") if finite else None
    overall_mean = float(seed_means.mean()) if seed_means is not None else None
    sample_std = float(seed_means.std(ddof=1)) if seed_means is not None else None
    pooled_q95 = float(np.quantile(values, 0.95, method="linear")) if finite else None
    t_ucb = (
        float(overall_mean + gate.t_critical_value * sample_std / math.sqrt(values.shape[0]))
        if overall_mean is not None and sample_std is not None
        else None
    )
    bootstrap_mean_estimate = float(mean_draws.mean()) if finite else None
    bootstrap_mean_ucb = (
        float(np.quantile(mean_draws, gate.bootstrap_quantile, method="linear")) if finite else None
    )
    bootstrap_q95_estimate = float(q95_draws.mean()) if finite else None
    bootstrap_q95_ucb = (
        float(np.quantile(q95_draws, gate.bootstrap_quantile, method="linear")) if finite else None
    )
    conditions = {
        "finite": finite,
        "zero_invalid_instances": invalid_instances == gate.maximum_invalid_instances,
        "overall_mean_strictly_below_threshold": (
            overall_mean is not None and overall_mean < threshold
        ),
        "every_seed_mean_strictly_below_threshold": (
            seed_means is not None and bool(np.all(seed_means < threshold))
        ),
        "every_seed_empirical_q95_strictly_below_threshold": (
            seed_q95 is not None and bool(np.all(seed_q95 < threshold))
        ),
        "pooled_empirical_q95_strictly_below_threshold": (
            pooled_q95 is not None and pooled_q95 < threshold
        ),
        "t_ucb_strictly_below_threshold": t_ucb is not None and t_ucb < threshold,
        "bootstrap_mean_ucb_strictly_below_threshold": (
            bootstrap_mean_ucb is not None and bootstrap_mean_ucb < threshold
        ),
        "bootstrap_q95_ucb_strictly_below_threshold": (
            bootstrap_q95_ucb is not None and bootstrap_q95_ucb < threshold
        ),
    }
    mean_gate = all(
        conditions[key]
        for key in (
            "finite",
            "zero_invalid_instances",
            "overall_mean_strictly_below_threshold",
            "every_seed_mean_strictly_below_threshold",
            "t_ucb_strictly_below_threshold",
            "bootstrap_mean_ucb_strictly_below_threshold",
        )
    )
    tail_gate = all(
        conditions[key]
        for key in (
            "finite",
            "zero_invalid_instances",
            "every_seed_empirical_q95_strictly_below_threshold",
            "pooled_empirical_q95_strictly_below_threshold",
            "bootstrap_q95_ucb_strictly_below_threshold",
        )
    )
    passed = bool(mean_gate and tail_gate)
    if not conditions["finite"] or not conditions["zero_invalid_instances"]:
        classification = "invalid"
    elif passed:
        classification = "pass"
    elif mean_gate:
        classification = "average_quality_feasible_tail_quality_infeasible"
    elif tail_gate:
        classification = "average_quality_infeasible_tail_quality_feasible"
    else:
        classification = "average_and_tail_quality_infeasible"
    return {
        "policy_id": f"hgs_b{budget}",
        "policy_role": "budget_sensitivity",
        "classification": classification,
        "passed": passed,
        "finite": finite,
        "invalid_instances": invalid_instances,
        "gap_matrix_shape": list(values.shape),
        "gap_matrix_pct": values.tolist() if finite else None,
        "per_seed_mean_gap_pct": seed_means.tolist() if seed_means is not None else None,
        "per_seed_empirical_q95_gap_pct": seed_q95.tolist() if seed_q95 is not None else None,
        "mean_of_seed_means_gap_pct": overall_mean,
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
        "threshold_pct": threshold,
        "strict_threshold": True,
        "t_critical_value": gate.t_critical_value,
        "t_degrees_of_freedom": gate.t_degrees_of_freedom,
        "bootstrap_quantile": gate.bootstrap_quantile,
        "gate_conditions": conditions,
        "mean_gate_passed": mean_gate,
        "tail_gate_passed": tail_gate,
        "all_required_conditions_satisfied": passed,
        "diagnostics_not_used_for_gate": {
            "minimum_instance_gap_pct": float(values.min()) if finite else None,
            "maximum_instance_gap_pct": float(values.max()) if finite else None,
            "instances_strictly_above_gate_threshold": (
                int(np.count_nonzero(values > threshold)) if finite else None
            ),
            "fraction_strictly_above_gate_threshold": (
                float(np.count_nonzero(values > threshold) / values.size) if finite else None
            ),
        },
    }


def _build_summary(
    recipe: HGSBudgetSensitivityRecipe,
    source: SourceContext,
    blocks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    expected = expected_blocks(recipe)
    if len(blocks) != len(expected):
        raise HGSBudgetSensitivityError("summary requires all twenty measured blocks")
    by_budget_round: dict[int, dict[int, dict[str, Any]]] = {
        budget: {} for budget in recipe.hgs_policy.budgets
    }
    matrices: dict[int, Any] = {}
    invalid: dict[int, int] = {}
    for payload in blocks:
        budget = int(payload["budget"])
        round_index = int(payload["round"])
        if round_index in by_budget_round[budget]:
            raise HGSBudgetSensitivityError("summary received a duplicate budget-round block")
        by_budget_round[budget][round_index] = payload
    for budget, by_round in by_budget_round.items():
        if set(by_round) != set(range(5)):
            raise HGSBudgetSensitivityError(f"budget {budget} lacks five measured rounds")
        gaps: list[list[float]] = []
        invalid[budget] = 0
        for round_index in range(5):
            validation = by_round[round_index]["validation"]
            gap = quality._gap_summary(validation["costs"], source.reference_costs)
            gaps.append([float(value) for value in gap["gaps_pct"]])
            invalid[budget] += int(validation["invalid_instance_count"])
        matrices[budget] = np.asarray(gaps, dtype=np.float64)
    samples = _bootstrap_policy_samples(
        matrices,
        replicates=recipe.quality_gate.bootstrap_replicates,
        seed=recipe.quality_gate.bootstrap_seed,
    )

    budgets: dict[str, Any] = {}
    all_quality_passed = True
    for budget in recipe.hgs_policy.budgets:
        by_round = by_budget_round[budget]
        gate = _quality_summary(
            matrices[budget],
            budget=budget,
            invalid_instances=invalid[budget],
            recipe=recipe,
            mean_samples=samples[budget][0],
            q95_samples=samples[budget][1],
        )
        all_quality_passed = all_quality_passed and bool(gate["passed"])
        rounds: list[dict[str, Any]] = []
        cpu_values: list[float] = []
        observed_values: list[float] = []
        for round_index in range(5):
            payload = by_round[round_index]
            measured_energy = payload["energy"]
            cpu = float(measured_energy["cpu_package_energy_j_per_instance"])
            observed = float(measured_energy["observed_component_energy_j_per_instance"])
            cpu_values.append(cpu)
            observed_values.append(observed)
            rounds.append(
                {
                    "round": round_index,
                    "hgs_seed": int(payload["hgs_seed"]),
                    "cpu_package_energy_j_per_instance": cpu,
                    "observed_component_energy_j_per_instance": observed,
                    "gpu_energy_j_per_instance": float(
                        measured_energy["gpu_energy_j_per_instance"]
                    ),
                    "throughput_instances_per_s": float(payload["throughput_instances_per_s"]),
                    "semantic_cost_sha256": payload["first_measured_semantic_cost_sha256"],
                    "route_sha256_diagnostic": payload["first_measured_route_sha256"],
                }
            )
        budgets[str(budget)] = {
            "budget": budget,
            "quality_status": "feasible" if gate["passed"] else "infeasible",
            "quality_gate": {
                "passed": bool(gate["passed"]),
                "status": "quality_passed" if gate["passed"] else "quality_nonpass",
                "threshold_pct": recipe.quality_gate.threshold_pct,
                "reference_sha256": source.reference_sha256,
                "bootstrap_replicates": recipe.quality_gate.bootstrap_replicates,
                "bootstrap_seed": recipe.quality_gate.bootstrap_seed,
                "bootstrap_quantile": recipe.quality_gate.bootstrap_quantile,
                "summary": gate,
            },
            "rounds": rounds,
            "energy_summary": {
                "cpu_package_j_per_instance": energy_runner._series_summary(cpu_values),
                "observed_components_j_per_instance": energy_runner._series_summary(
                    observed_values
                ),
                "whole_system_energy": False,
                "carbon_accounting": "none",
            },
        }

    budget10 = by_budget_round[recipe.drift_bridge.remeasurement_budget]
    old_cpu = [
        float(payload["energy"]["cpu_package_energy_j_per_instance"])
        for payload in source.old_hgs10_blocks
    ]
    new_cpu = [
        float(budget10[index]["energy"]["cpu_package_energy_j_per_instance"]) for index in range(5)
    ]
    semantic_rounds: list[dict[str, Any]] = []
    semantic_compatible = True
    route_hash_diagnostic = True
    for round_index in range(5):
        old = source.old_hgs10_blocks[round_index]
        new = budget10[round_index]
        old_costs = np.asarray(old["validation"]["costs"], dtype=np.float64)
        new_costs = np.asarray(new["validation"]["costs"], dtype=np.float64)
        equal = bool(np.array_equal(old_costs, new_costs))
        semantic_compatible = semantic_compatible and equal
        old_route_hash = str(old["first_measured_output_sha256"])
        new_legacy_hash = energy_runner._route_hash(
            new["routes"], new["validation"], new["solver_result_metadata"]
        )
        route_equal = old_route_hash == new_legacy_hash
        route_hash_diagnostic = route_hash_diagnostic and route_equal
        semantic_rounds.append(
            {
                "round": round_index,
                "hgs_seed": recipe.hgs_policy.seeds[round_index],
                "exact_per_instance_costs_equal": equal,
                "maximum_absolute_cost_difference": float(np.max(np.abs(old_costs - new_costs))),
                "old_semantic_cost_sha256": _semantic_cost_sha256(old_costs.tolist()),
                "new_semantic_cost_sha256": new["first_measured_semantic_cost_sha256"],
                "old_route_output_sha256": old_route_hash,
                "new_legacy_route_output_sha256": new_legacy_hash,
                "route_hashes_equal_diagnostic": route_equal,
            }
        )
    old_mean = mean(old_cpu)
    new_mean = mean(new_cpu)
    relative_difference = abs(new_mean - old_mean) / old_mean
    energy_compatible = (
        relative_difference <= recipe.drift_bridge.max_relative_mean_cpu_energy_difference
    )
    drift_passed = bool(semantic_compatible and energy_compatible)
    drift = {
        "budget": recipe.drift_bridge.remeasurement_budget,
        "old_mean_cpu_package_energy_j_per_instance": old_mean,
        "new_mean_cpu_package_energy_j_per_instance": new_mean,
        "relative_mean_difference": relative_difference,
        "maximum_relative_mean_difference": (
            recipe.drift_bridge.max_relative_mean_cpu_energy_difference
        ),
        "exact_per_instance_cost_reproducibility_required": (
            recipe.drift_bridge.require_exact_per_instance_cost_reproducibility
        ),
        "semantic_cost_compatible": semantic_compatible,
        "energy_compatible": energy_compatible,
        "route_hashes_all_equal_diagnostic": route_hash_diagnostic,
        "rounds": semantic_rounds,
        "passed": drift_passed,
    }
    source_binding = {
        "source_gates_passed": source.receipt["source_gates_passed"],
        "batch_frontier_manifest_sha256": source.receipt["batch_frontier"]["manifest_sha256"],
        "batch_frontier_summary_sha256": source.receipt["batch_frontier"][
            "batch_frontier_summary_sha256"
        ],
        "batch_frontier_checksums_sha256": source.receipt["batch_frontier"]["checksums_sha256"],
        "quality_manifest_sha256": next(
            item["sha256"]
            for item in source.receipt["quality_source"]["artifacts"]
            if item["path"] == "manifest.json"
        ),
        "reference_sha256": source.reference_sha256,
        "dataset_content_sha256": source.corpus.content_sha256,
    }
    combined = bool(source_binding["source_gates_passed"] and all_quality_passed and drift_passed)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": _classification(),
        "source_binding": source_binding,
        "budgets": budgets,
        "quality_method": {
            "metric": "mean_gap_to_locked_reference_pct",
            "threshold_pct": recipe.quality_gate.threshold_pct,
            "strict_mean_and_q95_holdout_logic": True,
            "shared_instance_and_hgs_seed_resampling_across_budgets": True,
            "bootstrap_method": recipe.quality_gate.bootstrap_method,
            "bootstrap_replicates": recipe.quality_gate.bootstrap_replicates,
            "bootstrap_seed": recipe.quality_gate.bootstrap_seed,
            "bootstrap_quantile": recipe.quality_gate.bootstrap_quantile,
        },
        "hgs10_drift_bridge": drift,
        "all_budget_quality_gates_passed": all_quality_passed,
        "combined_surface_eligible": combined,
        "training_was_run": False,
        "checkpoint_was_loaded": False,
        "neural_inference_was_run": False,
        "capacity_probe_was_run": False,
        "reference_was_generated": False,
        "source_bundle_was_mutated": False,
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


def _expected_inventory(
    recipe: HGSBudgetSensitivityRecipe,
    state: Mapping[str, Any],
    *,
    include_checksums: bool,
) -> set[str]:
    result = {
        "recipe.yaml",
        "environment/uv.lock",
        "source-receipt.json",
        "run-state.json",
        "hgs-budget-sensitivity-summary.json",
        "manifest.json",
        *(block.relative_path for block in expected_blocks(recipe)),
    }
    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise HGSBudgetSensitivityError("session history is malformed")
    for session in sessions:
        if not isinstance(session, dict) or not isinstance(session.get("preflight_evidence"), dict):
            raise HGSBudgetSensitivityError("session preflight evidence is malformed")
        result.add(str(session["preflight_evidence"]["path"]))
    orphaned = state.get("orphaned_preflight_evidence", [])
    if not isinstance(orphaned, list):
        raise HGSBudgetSensitivityError("orphaned preflight evidence is malformed")
    for evidence in orphaned:
        if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
            raise HGSBudgetSensitivityError("orphaned preflight evidence is malformed")
        result.add(evidence["path"])
    if include_checksums:
        result.add("SHA256SUMS")
    return result


def _assert_inventory(
    recipe: HGSBudgetSensitivityRecipe,
    output: Path,
    state: Mapping[str, Any],
    *,
    include_checksums: bool,
) -> None:
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != _expected_inventory(recipe, state, include_checksums=include_checksums):
        raise HGSBudgetSensitivityError("output contains missing or unregistered artifacts")


def _complete(
    recipe: HGSBudgetSensitivityRecipe,
    output: Path,
    state: dict[str, Any],
    source: SourceContext,
    blocks: Sequence[dict[str, Any]],
) -> HGSBudgetSensitivityResult:
    summary = _build_summary(recipe, source, blocks)
    summary_path = output / "hgs-budget-sensitivity-summary.json"
    _atomic_write_json(summary_path, summary)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": _classification(),
        "block_count": len(blocks),
        "budgets": list(recipe.hgs_policy.budgets),
        "rounds": 5,
        "dataset": source.receipt["dataset"],
        "source_receipt_sha256": _sha256_file(output / "source-receipt.json"),
        "summary_sha256": _sha256_file(summary_path),
        "source_binding": summary["source_binding"],
        "preflight_evidence": energy_runner._unique_preflight_evidence(state),
        "interrupted_preflight_evidence": state.get("orphaned_preflight_evidence", []),
        "combined_surface_eligible": summary["combined_surface_eligible"],
        "measurement_scope": "hgs_only",
        "training_was_run": False,
        "checkpoint_was_loaded": False,
        "neural_inference_was_run": False,
        "capacity_probe_was_run": False,
        "reference_was_generated": False,
        "source_bundle_was_mutated": False,
    }
    manifest_path = output / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    state["status"] = COMPLETE_STATUS
    state["completed_at"] = datetime.now(UTC).isoformat()
    state["summary_sha256"] = _sha256_file(summary_path)
    state["manifest_sha256"] = _sha256_file(manifest_path)
    stale_checksums = (output / "SHA256SUMS").exists()
    _assert_inventory(recipe, output, state, include_checksums=stale_checksums)
    state["checksums_sha256"] = _write_checksums(output)
    _atomic_write_json(output / "run-state.json", state)
    return HGSBudgetSensitivityResult(
        output,
        COMPLETE_STATUS,
        len(blocks),
        manifest_path,
        _sha256_file(manifest_path),
        bool(summary["combined_surface_eligible"]),
    )


def _validate_complete(
    recipe_path: Path,
    recipe: HGSBudgetSensitivityRecipe,
    root: Path,
    output: Path,
    source: SourceContext,
) -> HGSBudgetSensitivityResult:
    state = _load_json(output / "run-state.json")
    recipe_sha = _sha256_file(recipe_path)
    lock_sha = _sha256_file(root / "uv.lock")
    source_bytes = _json_bytes(source.receipt)
    if (
        state.get("schema_version") != RUN_STATE_SCHEMA
        or state.get("status") != COMPLETE_STATUS
        or state.get("recipe_sha256") != recipe_sha
        or state.get("uv_lock_sha256") != lock_sha
        or state.get("source_receipt_sha256") != _sha256_bytes(source_bytes)
        or state.get("schedule") != [asdict(block) for block in expected_blocks(recipe)]
        or _sha256_file(output / "recipe.yaml") != recipe_sha
        or _sha256_file(output / "environment" / "uv.lock") != lock_sha
        or (output / "source-receipt.json").read_bytes() != source_bytes
    ):
        raise HGSBudgetSensitivityError("completed sidecar identity changed")
    blocks = _load_completed_blocks(recipe, output, state, source)
    if len(blocks) != len(expected_blocks(recipe)) or state.get("current_block") is not None:
        raise HGSBudgetSensitivityError("complete state lacks all twenty blocks")
    expected_summary = _build_summary(recipe, source, blocks)
    summary_path = output / "hgs-budget-sensitivity-summary.json"
    manifest_path = output / "manifest.json"
    checksums_path = output / "SHA256SUMS"
    if (
        _load_json(summary_path) != expected_summary
        or state.get("summary_sha256") != _sha256_file(summary_path)
        or state.get("manifest_sha256") != _sha256_file(manifest_path)
        or state.get("checksums_sha256") != _sha256_file(checksums_path)
    ):
        raise HGSBudgetSensitivityError("completed sidecar anchors changed")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != COMPLETE_STATUS
        or manifest.get("summary_sha256") != state["summary_sha256"]
        or manifest.get("preflight_evidence") != energy_runner._unique_preflight_evidence(state)
        or manifest.get("interrupted_preflight_evidence")
        != state.get("orphaned_preflight_evidence", [])
        or manifest.get("combined_surface_eligible")
        is not expected_summary["combined_surface_eligible"]
    ):
        raise HGSBudgetSensitivityError("completed sidecar manifest changed")
    _assert_inventory(recipe, output, state, include_checksums=True)
    listed: set[str] = set()
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise HGSBudgetSensitivityError("checksum inventory is malformed")
        digest, relative = line[:64], line[66:]
        path = _relative(output, relative)
        if relative in listed or not path.is_file() or _sha256_file(path) != digest:
            raise HGSBudgetSensitivityError("checksum inventory verification failed")
        listed.add(relative)
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.relative_to(output).as_posix() not in {"SHA256SUMS", "run-state.json"}
    }
    if listed != actual:
        raise HGSBudgetSensitivityError("checksum inventory is not exhaustive")
    return HGSBudgetSensitivityResult(
        output,
        COMPLETE_STATUS,
        len(blocks),
        manifest_path,
        _sha256_file(manifest_path),
        bool(expected_summary["combined_surface_eligible"]),
    )


def run_hgs_budget_sensitivity(
    recipe_path: Path,
    workspace_root: Path | None = None,
    *,
    resume: bool = False,
) -> HGSBudgetSensitivityResult:
    """Execute all remaining HGS blocks under one attested process."""

    root = (workspace_root or Path.cwd()).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    output = _safe_output(root, recipe.output_root)
    source = _validate_sources(recipe, root)
    with _campaign_lock(output):
        return _run_hgs_budget_sensitivity_locked(
            recipe_path,
            recipe,
            root,
            output,
            source,
            resume=resume,
        )


def _run_hgs_budget_sensitivity_locked(
    recipe_path: Path,
    recipe: HGSBudgetSensitivityRecipe,
    root: Path,
    output: Path,
    source: SourceContext,
    *,
    resume: bool,
) -> HGSBudgetSensitivityResult:
    if output.exists():
        state = _load_json(output / "run-state.json")
        if state.get("status") == COMPLETE_STATUS:
            return _validate_complete(recipe_path, recipe, root, output, source)
        if not resume:
            raise HGSBudgetSensitivityError(f"output already exists; use --resume: {output}")

    qualification = qualify_for_execution(recipe, root)
    if qualification.get("ready_to_execute") is not True:
        raise HGSBudgetSensitivityQualificationError(
            "native-Windows EMI/NVML preflight is not qualified"
        )
    attestation = _exclusive_attestation(recipe)
    runtime_identity = _runtime_identity(recipe)
    output, state = _prepare_output(
        recipe_path,
        recipe,
        root,
        resume=resume,
        runtime_identity=runtime_identity,
        source_receipt=source.receipt,
    )
    if state.get("status") != INCOMPLETE_STATUS:
        raise HGSBudgetSensitivityError("sidecar has an unknown incomplete status")
    payloads = _load_completed_blocks(recipe, output, state, source)
    schedule = expected_blocks(recipe)

    evidence = _preserve_preflight(recipe, root, output)
    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise HGSBudgetSensitivityError("session history is malformed")
    sessions.append(
        {
            **attestation,
            "preflight_evidence": evidence,
            "process_started_at": datetime.now(UTC).isoformat(),
            "resume": resume,
            "completed_block_count_at_start": len(payloads),
        }
    )
    _atomic_write_json(output / "run-state.json", state)

    for block in schedule[len(payloads) :]:
        _assert_attestation_active(recipe, attestation)
        block_path = _relative(output, block.relative_path)
        if block_path.exists():
            attempt = _validated_current_attempt(state, block)
            payload = _load_json(block_path)
            _validate_block(
                payload,
                recipe,
                block,
                source.corpus,
                source.reference_sha256,
                expected_session_id=str(attempt["session_id"]),
            )
            attempt["outcome"] = "complete_orphan_promotion_recovered"
        else:
            previous = state.get("current_block")
            if previous is not None:
                old_attempt = _validated_current_attempt(state, block)
                old_attempt["outcome"] = "interrupted_before_artifact_promotion"
                old_attempt["closed_at"] = datetime.now(UTC).isoformat()
            attempt_index = len(state["block_attempts"])
            attempt = {
                "attempt_index": attempt_index,
                "relative_path": block.relative_path,
                "round": block.round_index,
                "order_within_round": block.order_index,
                "budget": block.budget,
                "hgs_seed": block.hgs_seed,
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
            payload = _execute_block(
                recipe,
                block,
                corpus=source.corpus,
                reference_sha256=source.reference_sha256,
                attestation=attestation,
            )
            _assert_attestation_active(recipe, attestation)
            _validate_block(
                payload,
                recipe,
                block,
                source.corpus,
                source.reference_sha256,
                expected_session_id=attestation["session_id"],
            )
            _atomic_write_json(block_path, payload)
        digest = _sha256_file(block_path)
        if attempt["outcome"] != "complete_orphan_promotion_recovered":
            attempt["outcome"] = "complete"
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
                    "budget": block.budget,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _assert_attestation_active(recipe, attestation)
    return _complete(recipe, output, state, source, payloads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.estimate_only:
        try:
            dry_run(args.recipe)
        except Exception as exc:
            print(f"HGS budget sensitivity estimate failed: {exc}", file=sys.stderr, flush=True)
            return 2
        return 0
    try:
        result = run_hgs_budget_sensitivity(args.recipe, resume=args.resume)
    except HGSBudgetSensitivityError as exc:
        print(f"HGS budget sensitivity failed: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(
            f"HGS budget sensitivity failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "path": str(result.path),
                "completed_blocks": result.completed_blocks,
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "combined_surface_eligible": result.combined_surface_eligible,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
