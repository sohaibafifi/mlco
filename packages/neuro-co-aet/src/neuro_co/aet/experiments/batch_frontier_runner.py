"""Run the native-Windows AM/GNN batch frontier against fixed HGS-10.

The capacity probe is deliberately outside every energy tracker.  It resolves
one feasible powers-of-two prefix per architecture before the first measured
block.  Five rounds then share exactly one HGS-10 block across every feasible
AM/GNN batch cell.  Every promoted block is a durable strict schedule prefix,
so the same command resumes after interruption without repeating completed
energy blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import mean, median, stdev
from typing import Any

from neuro_co.aet.experiments import deployment_energy_runner as base
from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments import software_recipe as software_recipe
from neuro_co.aet.experiments.batch_frontier_recipe import (
    ARCHITECTURES,
    CLASSIFICATION,
    TRAINING_SEEDS,
    BatchFrontierRecipe,
    FrontierBlock,
    expected_blocks,
    load_recipe,
    qualify_for_execution,
)

RUN_STATE_SCHEMA = "aet-batch-frontier-run-state/v1"
CAPACITY_SCHEMA = "aet-batch-frontier-capacity-probe/v1"
BLOCK_SCHEMA = "aet-batch-frontier-block/v1"
SUMMARY_SCHEMA = "aet-batch-frontier-summary/v1"
MANIFEST_SCHEMA = "aet-batch-frontier-manifest/v1"
COMPLETE_STATUS = "complete"
INCOMPLETE_STATUS = "incomplete"

ATTESTED_ENV = "AET_EXPERIMENT_EXCLUSIVE_ATTESTED"
ATTESTED_AT_ENV = "AET_EXPERIMENT_EXCLUSIVE_ATTESTED_AT"
ATTESTATION_SESSION_ENV = "AET_EXPERIMENT_EXCLUSIVE_SESSION_ID"


class BatchFrontierError(RuntimeError):
    """Raised when the batch-frontier campaign cannot continue safely."""


class BatchFrontierQualificationError(BatchFrontierError):
    """Raised before measurement when a fixed qualification does not hold."""


@dataclass(frozen=True, slots=True)
class BatchFrontierResult:
    path: Path
    status: str
    completed_blocks: int
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenQualityGate:
    threshold_pct: float
    maximum_invalid_instances: int
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_quantile: float
    t_critical_value: float
    t_degrees_of_freedom: int


def classification() -> dict[str, Any]:
    return dict(CLASSIFICATION)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    )


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
        raise BatchFrontierError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, dict):
        raise BatchFrontierError(f"JSON artifact is not an object: {path}")
    return value


def _relative(root: Path, value: str) -> Path:
    return root.joinpath(*PurePosixPath(value).parts)


def _safe_relative(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BatchFrontierQualificationError(f"{where} is not a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise BatchFrontierQualificationError(f"{where} is not a safe relative path")
    return parsed.as_posix()


def _safe_output(root: Path, value: str) -> Path:
    repository = root.resolve(strict=True)
    if repository != Path.cwd().resolve(strict=True):
        raise BatchFrontierError("workspace_root must be the current Git checkout")
    output = _relative(repository, value)
    cursor = repository
    for component in output.relative_to(repository).parts:
        cursor /= component
        if (cursor.exists() or cursor.is_symlink()) and cursor.is_symlink():
            raise BatchFrontierError(f"refusing symlinked output component: {cursor}")
    resolved = output.resolve(strict=False)
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise BatchFrontierError("output root resolves outside the repository") from exc
    return resolved


def _training_root(recipe: BatchFrontierRecipe, root: Path, override: Path | None) -> Path:
    candidate = override if override is not None else _relative(root, recipe.training_source.root)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BatchFrontierQualificationError(
            "training source must be inside the repository"
        ) from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise BatchFrontierQualificationError("training source is not a regular directory")
    return resolved


def _checkpoint_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    direct = manifest.get("checkpoint_entries")
    if isinstance(direct, list):
        entries = direct
    else:
        entries = []
        architectures = manifest.get("architectures")
        if not isinstance(architectures, dict):
            raise BatchFrontierQualificationError("training manifest lacks architectures")
        for architecture in ARCHITECTURES:
            record = architectures.get(architecture)
            if not isinstance(record, dict) or not isinstance(
                record.get("checkpoint_entries"), list
            ):
                raise BatchFrontierQualificationError(
                    f"training manifest lacks {architecture} checkpoint entries"
                )
            entries.extend(record["checkpoint_entries"])
    if any(not isinstance(entry, dict) for entry in entries):
        raise BatchFrontierQualificationError("training checkpoint entries are malformed")
    return entries


def _architecture_quality_receipts(
    manifest: dict[str, Any], source: Path
) -> dict[str, dict[str, Any]]:
    architectures = manifest.get("architectures")
    if not isinstance(architectures, dict) or set(architectures) != set(ARCHITECTURES):
        raise BatchFrontierQualificationError(
            "training manifest must contain exactly the AM and GNN architectures"
        )
    receipts: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        record = architectures[architecture]
        if not isinstance(record, dict):
            raise BatchFrontierQualificationError(
                f"training quality record for {architecture} is malformed"
            )
        frontier_eligible = record.get("frontier_eligible")
        if not isinstance(frontier_eligible, bool):
            raise BatchFrontierQualificationError(
                f"training quality eligibility for {architecture} is missing"
            )
        expected_status = "feasible" if frontier_eligible else "infeasible"
        quality_status = record.get("quality_status", expected_status)
        if quality_status != expected_status:
            raise BatchFrontierQualificationError(
                f"training quality status for {architecture} is inconsistent"
            )
        gate = record.get("quality_gate")
        if not isinstance(gate, dict) or set(gate) != {"passed", "status", "path", "sha256"}:
            raise BatchFrontierQualificationError(
                f"training quality gate for {architecture} is malformed"
            )
        expected_gate_status = "quality_passed" if frontier_eligible else "quality_nonpass"
        if (
            gate.get("passed") is not frontier_eligible
            or gate.get("status") != expected_gate_status
            or record.get("energy_frontier_measurement_required") is not True
        ):
            raise BatchFrontierQualificationError(
                f"training quality gate for {architecture} is inconsistent"
            )
        gate_path = _safe_relative(gate.get("path"), f"{architecture} quality gate path")
        gate_sha = _sha_field(gate.get("sha256"), f"{architecture} quality gate sha256")
        artifact = _relative(source, gate_path)
        if artifact.is_symlink() or not artifact.is_file() or _sha256_file(artifact) != gate_sha:
            raise BatchFrontierQualificationError(
                f"training quality gate artifact changed for {architecture}"
            )
        gate_artifact = _load_json(artifact)
        gate_summary = gate_artifact.get("summary")
        if (
            gate_artifact.get("architecture") != architecture
            or gate_artifact.get("status") != expected_gate_status
            or gate_artifact.get("frontier_eligible") is not frontier_eligible
            or not isinstance(gate_summary, dict)
            or gate_summary.get("passed") is not frontier_eligible
        ):
            raise BatchFrontierQualificationError(
                f"training quality gate contents for {architecture} are inconsistent"
            )
        receipts[architecture] = {
            "quality_status": quality_status,
            "frontier_eligible": frontier_eligible,
            "quality_gate": {
                "passed": frontier_eligible,
                "status": expected_gate_status,
                "path": gate_path,
                "sha256": gate_sha,
            },
            "energy_frontier_measurement_required": True,
        }
    return receipts


def _sha_field(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BatchFrontierQualificationError(f"{where} is not a lowercase SHA-256")
    return value


def _expected_model_configuration(architecture: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "backbone": architecture,
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 8,
    }
    if architecture == "am":
        return common
    return {
        **common,
        "encoder": "GNNEncoder",
        "normalization": "batch",
        "prenorm": True,
        "dropout": 0.1,
        "sparsify": True,
        "k_sparse": 10,
        "edge_features": True,
        "rbf_k": 16,
        "fourier_feats": 1,
        "residual": True,
        "decoder": "mlco-pointer-decoder",
    }


def _validate_training_model_metadata(
    architecture: str,
    identity: dict[str, Any],
    configuration: dict[str, Any],
    backend: dict[str, Any],
) -> None:
    if configuration != _expected_model_configuration(architecture):
        raise BatchFrontierQualificationError(
            f"checkpoint {architecture} model configuration changed"
        )
    if (
        identity.get("architecture") != architecture
        or identity.get("factory_backbone") != architecture
        or identity.get("model_configuration") != configuration
        or identity.get("backend_identity") != backend
    ):
        raise BatchFrontierQualificationError("checkpoint model identity is inconsistent")
    decoder_class = backend.get("decoder_class")
    if not isinstance(decoder_class, str) or not decoder_class.endswith(".PointerDecoder"):
        raise BatchFrontierQualificationError("checkpoint does not use the common pointer decoder")
    if architecture == "am":
        if backend.get("layer_backend") != "torch":
            raise BatchFrontierQualificationError("AM checkpoint backend identity changed")
        return
    expected_attributes = {
        "in_dim": 3,
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 8,
        "k_sparse": 10,
        "dropout": 0.1,
        "rbf_k": 16,
        "fourier_feats": 1,
        "residual": True,
    }
    if backend.get("encoder_attributes") != expected_attributes:
        raise BatchFrontierQualificationError("GNN checkpoint encoder attributes changed")
    try:
        runtime_pyg = importlib.metadata.version("torch-geometric")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BatchFrontierQualificationError("torch-geometric runtime is unavailable") from exc
    if (
        backend.get("encoder_class") != "neuro_co.core.models.encoders.gnn.GNNEncoder"
        or backend.get("layer_backend") != "torch_geometric.TransformerConv"
        or backend.get("edge_index_backend") != "torch_cdist_topk"
        or backend.get("torch_geometric_version") != runtime_pyg
    ):
        raise BatchFrontierQualificationError("GNN checkpoint backend identity changed")


def _resolve_training_source(
    recipe: BatchFrontierRecipe,
    root: Path,
    override: Path | None,
) -> tuple[Path, dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    source = _training_root(recipe, root, override)
    manifest_path = source / recipe.training_source.manifest_path
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != recipe.training_source.manifest_schema
        or manifest.get("status") != recipe.training_source.expected_status
    ):
        raise BatchFrontierQualificationError("training source is not a completed measured bundle")
    architecture_quality = _architecture_quality_receipts(manifest, source)
    entries = _checkpoint_entries(manifest)
    expected_keys = {
        (architecture, seed) for architecture in ARCHITECTURES for seed in TRAINING_SEEDS
    }
    resolved: dict[tuple[str, int], dict[str, Any]] = {}
    models: list[dict[str, Any]] = []
    for raw in entries:
        architecture = raw.get("architecture")
        seed = raw.get("training_seed")
        if not isinstance(architecture, str) or not isinstance(seed, int) or isinstance(seed, bool):
            raise BatchFrontierQualificationError("training source has invalid checkpoint keys")
        key = (architecture, seed)
        if key not in expected_keys or key in resolved:
            raise BatchFrontierQualificationError("training source has unexpected checkpoint keys")
        relative = _safe_relative(raw.get("path"), "checkpoint path")
        checkpoint = _relative(source, relative)
        checkpoint_sha = _sha_field(raw.get("checkpoint_sha256"), "checkpoint_sha256")
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or _sha256_file(checkpoint) != checkpoint_sha
        ):
            raise BatchFrontierQualificationError(f"training checkpoint changed: {relative}")
        model_state_sha = _sha_field(raw.get("model_state_sha256"), "model_state_sha256")
        identity = raw.get("model_identity")
        configuration = raw.get("model_configuration")
        backend = raw.get("backend_identity")
        if (
            not isinstance(identity, dict)
            or not isinstance(configuration, dict)
            or not isinstance(backend, dict)
        ):
            raise BatchFrontierQualificationError("checkpoint model identity is incomplete")
        identity_sha = _sha_field(raw.get("model_identity_sha256"), "model_identity_sha256")
        identity_payload = dict(identity)
        embedded_identity_sha = identity_payload.pop("model_identity_sha256", None)
        if (
            embedded_identity_sha != identity_sha
            or _canonical_sha256(identity_payload) != identity_sha
        ):
            raise BatchFrontierQualificationError("checkpoint model identity hash changed")
        _validate_training_model_metadata(str(architecture), identity, configuration, backend)
        checkpoint_payload = quality._load_torch_mapping(checkpoint)
        if (
            checkpoint_payload.get("architecture") != architecture
            or checkpoint_payload.get("training_seed") != seed
            or checkpoint_payload.get("completed_epochs") != recipe.training_source.checkpoint_epoch
            or checkpoint_payload.get("model_configuration") != configuration
            or checkpoint_payload.get("model_identity") != identity
        ):
            raise BatchFrontierQualificationError("checkpoint embedded model identity changed")
        from neuro_co.aet.experiments.training_debt_runner import model_state_sha256

        if model_state_sha256(checkpoint_payload.get("model", {})) != model_state_sha:
            raise BatchFrontierQualificationError("checkpoint model-state hash changed")
        record = {
            "architecture": architecture,
            "seed": seed,
            "training_seed": seed,
            "path": relative,
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": model_state_sha,
            "model_identity": identity,
            "model_identity_sha256": identity_sha,
            "model_configuration": configuration,
            "backend_identity": backend,
        }
        resolved[key] = record
        models.append({key: value for key, value in record.items() if key != "checkpoint_path"})
    if set(resolved) != expected_keys:
        raise BatchFrontierQualificationError(
            "training source does not contain ten AM/GNN checkpoints"
        )

    base_recipe_path = _safe_relative(manifest.get("base_recipe_path"), "base_recipe_path")
    base_recipe = _relative(source, base_recipe_path)
    base_recipe_sha = _sha_field(manifest.get("base_recipe_sha256"), "base_recipe_sha256")
    if (
        base_recipe.is_symlink()
        or not base_recipe.is_file()
        or _sha256_file(base_recipe) != base_recipe_sha
    ):
        raise BatchFrontierQualificationError("training base recipe changed")
    receipt = {
        "root": source.relative_to(root).as_posix(),
        "training_manifest_path": recipe.training_source.manifest_path,
        "training_manifest_sha256": _sha256_file(manifest_path),
        "training_manifest_schema": manifest["schema_version"],
        "training_status": manifest["status"],
        "base_recipe_path": base_recipe_path,
        "base_recipe_sha256": base_recipe_sha,
        "base_recipe_semantic_sha256": manifest.get("base_recipe_semantic_sha256"),
        "architecture_config_sha256": manifest.get("architecture_config_sha256"),
        "architecture_quality": architecture_quality,
        "models": sorted(models, key=lambda item: (item["architecture"], item["training_seed"])),
    }
    return source, receipt, resolved


def _quality_source_receipt(recipe: BatchFrontierRecipe, root: Path) -> dict[str, Any]:
    source = _relative(root, recipe.quality_source.root)
    manifest = _load_json(source / "manifest.json")
    state = _load_json(source / "run-state.json")
    assessment = _load_json(source / "holdout-assessment.json")
    decision = assessment.get("confirmatory_decision")
    if (
        manifest.get("status") != recipe.quality_source.expected_status
        or state.get("status") != recipe.quality_source.expected_status
        or state.get("git_sha") != recipe.quality_source.source_git_sha
        or state.get("source_snapshot", {}).get("sha256")
        != recipe.quality_source.source_snapshot_sha256
        or not isinstance(decision, dict)
        or decision.get("passed") is not True
        or decision.get("rule") != "neural_and_hgs10"
        or decision.get("hgs_budget") != 10
    ):
        raise BatchFrontierQualificationError("seed2723 no longer proves the fixed HGS-10 policy")
    return {
        "root": recipe.quality_source.root,
        "manifest_sha256": _sha256_file(source / "manifest.json"),
        "dataset_content_sha256": (
            "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf"
        ),
        "source_git_sha": recipe.quality_source.source_git_sha,
        "source_snapshot_sha256": recipe.quality_source.source_snapshot_sha256,
    }


def _quality_gate_context(
    recipe: BatchFrontierRecipe,
    root: Path,
    source_receipt: dict[str, Any],
    training_source: Path | None,
) -> dict[str, Any]:
    """Load the frozen training gate and sealed reference without inference."""

    import yaml

    training = source_receipt.get("training")
    if not isinstance(training, dict):
        raise BatchFrontierQualificationError("source receipt lacks the training source")
    source = _training_root(recipe, root, training_source)
    if training.get("root") != source.relative_to(root).as_posix():
        raise BatchFrontierQualificationError("training source root differs from its receipt")
    manifest_path = source / recipe.training_source.manifest_path
    manifest_sha = _sha_field(training.get("training_manifest_sha256"), "training manifest sha256")
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or _sha256_file(manifest_path) != manifest_sha
    ):
        raise BatchFrontierQualificationError("frozen training manifest changed")
    manifest = _load_json(manifest_path)
    training_recipe_sha = _sha_field(manifest.get("recipe_sha256"), "frozen training recipe sha256")
    training_recipe_path = source / "recipe.yaml"
    if (
        training_recipe_path.is_symlink()
        or not training_recipe_path.is_file()
        or _sha256_file(training_recipe_path) != training_recipe_sha
    ):
        raise BatchFrontierQualificationError("frozen training recipe changed")
    try:
        frozen = yaml.safe_load(training_recipe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BatchFrontierQualificationError("frozen training recipe is unreadable") from exc
    if not isinstance(frozen, dict):
        raise BatchFrontierQualificationError("frozen training recipe is malformed")
    evaluation = frozen.get("evaluation")
    raw_gate = frozen.get("quality_gate")
    expected_gate_keys = {
        "threshold_pct",
        "maximum_invalid_instances",
        "bootstrap_replicates",
        "bootstrap_seed",
        "bootstrap_quantile",
        "t_critical_value",
        "t_degrees_of_freedom",
    }
    if (
        not isinstance(evaluation, dict)
        or type(evaluation.get("batch_size")) is not int
        or evaluation["batch_size"] < 1
        or not isinstance(raw_gate, dict)
        or set(raw_gate) != expected_gate_keys
    ):
        raise BatchFrontierQualificationError("frozen training quality gate is malformed")

    def positive_number(value: Any, where: str) -> float:
        if type(value) is not float or not math.isfinite(float(value)) or float(value) <= 0:
            raise BatchFrontierQualificationError(f"{where} is invalid")
        return float(value)

    maximum_invalid = raw_gate["maximum_invalid_instances"]
    bootstrap_replicates = raw_gate["bootstrap_replicates"]
    bootstrap_seed = raw_gate["bootstrap_seed"]
    degrees_of_freedom = raw_gate["t_degrees_of_freedom"]
    bootstrap_quantile = raw_gate["bootstrap_quantile"]
    if (
        type(maximum_invalid) is not int
        or maximum_invalid < 0
        or type(bootstrap_replicates) is not int
        or bootstrap_replicates < 1
        or type(bootstrap_seed) is not int
        or bootstrap_seed < 0
        or type(degrees_of_freedom) is not int
        or degrees_of_freedom < 1
        or type(bootstrap_quantile) is not float
        or not 0 < float(bootstrap_quantile) < 1
    ):
        raise BatchFrontierQualificationError("frozen training quality-gate values are invalid")
    gate = FrozenQualityGate(
        threshold_pct=positive_number(raw_gate["threshold_pct"], "quality threshold"),
        maximum_invalid_instances=maximum_invalid,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_quantile=float(bootstrap_quantile),
        t_critical_value=positive_number(raw_gate["t_critical_value"], "t critical value"),
        t_degrees_of_freedom=degrees_of_freedom,
    )
    if gate.maximum_invalid_instances != 0:
        raise BatchFrontierQualificationError("frozen quality gate invalid-instance rule changed")

    quality_root = _relative(root, recipe.quality_source.root)
    for artifact in recipe.quality_source.artifacts:
        path = _relative(quality_root, artifact.path)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != artifact.sha256:
            raise BatchFrontierQualificationError(
                f"sealed quality-source artifact changed: {artifact.path}"
            )
    lock_path = quality_root / "reference" / "reference-lock.json"
    lock = _load_json(lock_path)
    reference_entry = lock.get("reference")
    if not isinstance(reference_entry, dict):
        raise BatchFrontierQualificationError("sealed reference lock is malformed")
    reference_relative = _safe_relative(reference_entry.get("path"), "sealed reference path")
    reference_sha = _sha_field(reference_entry.get("sha256"), "sealed reference sha256")
    reference_path = _relative(quality_root, reference_relative)
    if (
        reference_path.is_symlink()
        or not reference_path.is_file()
        or _sha256_file(reference_path) != reference_sha
    ):
        raise BatchFrontierQualificationError("sealed quality reference changed")
    reference = _load_json(reference_path)
    costs = reference.get("costs")
    if (
        reference.get("status") != "complete"
        or reference.get("dataset_content_sha256")
        != source_receipt.get("quality", {}).get("dataset_content_sha256")
        or not isinstance(costs, list)
        or len(costs) != recipe.dataset.num_instances
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in costs
        )
    ):
        raise BatchFrontierQualificationError("sealed quality-reference costs are invalid")
    return {
        "reference_costs": [float(value) for value in costs],
        "reference_sha256": reference_sha,
        "quality_gate": gate,
        "provenance": {
            "training_recipe_path": "recipe.yaml",
            "training_recipe_sha256": training_recipe_sha,
            "training_gate_anchor_batch_size": evaluation["batch_size"],
            "reference_path": reference_relative,
            "reference_sha256": reference_sha,
        },
    }


def _verify_source_receipt(
    recipe: BatchFrontierRecipe,
    root: Path,
    source_receipt: dict[str, Any],
    training_source: Path | None,
) -> dict[str, Any]:
    """Verify both source trees and return their frozen quality context."""

    _, current_training, _ = _resolve_training_source(recipe, root, training_source)
    current_quality = _quality_source_receipt(recipe, root)
    if source_receipt.get("training") != current_training:
        raise BatchFrontierQualificationError("training source differs from its frozen receipt")
    if source_receipt.get("quality") != current_quality:
        raise BatchFrontierQualificationError("quality source differs from its frozen receipt")
    return _quality_gate_context(recipe, root, source_receipt, training_source)


def _exclusive_attestation(recipe: BatchFrontierRecipe) -> dict[str, Any]:
    if os.environ.get(ATTESTED_ENV) != "1":
        raise BatchFrontierQualificationError("exclusive-use attestation is absent")
    raw_at = os.environ.get(ATTESTED_AT_ENV)
    session_id = os.environ.get(ATTESTATION_SESSION_ENV)
    if not raw_at or not session_id:
        raise BatchFrontierQualificationError("exclusive-use attestation is incomplete")
    try:
        attested_at = datetime.fromisoformat(raw_at.replace("Z", "+00:00")).astimezone(UTC)
        uuid.UUID(session_id)
    except (TypeError, ValueError) as exc:
        raise BatchFrontierQualificationError("exclusive-use attestation is invalid") from exc
    age_s = (datetime.now(UTC) - attested_at).total_seconds()
    if age_s < -60 or age_s > recipe.campaign_attestation_max_age_s:
        raise BatchFrontierQualificationError("exclusive-use campaign expired")
    return {
        "session_id": session_id,
        "attested_at": attested_at.isoformat(),
        "age_s_at_frontier_start": age_s,
        "operator_supplied": True,
        "operator_attestation_is_authoritative": True,
        "gpu_process_lists_are_diagnostic_only": True,
    }


def _assert_campaign_active(
    recipe: BatchFrontierRecipe,
    attestation: dict[str, Any],
    process_started: float,
) -> None:
    age = (
        datetime.now(UTC) - datetime.fromisoformat(str(attestation["attested_at"]))
    ).total_seconds()
    if age > recipe.campaign_attestation_max_age_s:
        raise BatchFrontierQualificationError("exclusive-use campaign exceeded 36 hours")
    if time.perf_counter() - process_started > recipe.maximum_campaign_walltime_s:
        raise BatchFrontierError("batch-frontier process exceeded 36 hours")


def _runtime_identity(recipe: BatchFrontierRecipe) -> dict[str, Any]:
    import torch
    import torch_geometric

    versions = {
        label: importlib.metadata.version(distribution)
        for label, distribution in {
            "torch": "torch",
            "torch_geometric": "torch-geometric",
            "numpy": "numpy",
            "pyvrp": "pyvrp",
            "codecarbon": "codecarbon",
            "pynvml": "nvidia-ml-py",
        }.items()
    }
    if versions["torch_geometric"] != torch_geometric.__version__:
        raise BatchFrontierQualificationError("torch-geometric runtime identity is inconsistent")
    return {
        "platform": sys.platform,
        "platform_release": platform.release(),
        "python": platform.python_version(),
        "versions": versions,
        "gpu_index": recipe.gpu_index,
        "gpu_name": torch.cuda.get_device_name(recipe.gpu_index),
        "torch_cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "gnn_layer_backend": "torch_geometric.TransformerConv",
        "gnn_edge_index_backend": "torch_cdist_topk",
    }


def _configure_runtime(recipe: BatchFrontierRecipe) -> dict[str, Any]:
    if sys.platform != "win32":
        raise BatchFrontierQualificationError(
            "batch-frontier qualification requires native Windows, not WSL"
        )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise BatchFrontierQualificationError(f"{name} must be exactly 1")
    import torch

    if not torch.cuda.is_available() or recipe.gpu_index >= torch.cuda.device_count():
        raise BatchFrontierQualificationError("the selected CUDA device is unavailable")
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
        raise BatchFrontierQualificationError("Git source identity is unavailable")
    return snapshot


def _prepare_output(
    recipe_path: Path,
    recipe: BatchFrontierRecipe,
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
    git = _git_snapshot(root, _relative(root, recipe.preflight_report))
    receipt_bytes = _json_bytes(source_receipt)
    identity = {
        "schema_version": RUN_STATE_SCHEMA,
        "recipe_sha256": recipe_sha,
        "uv_lock_sha256": lock_sha,
        "git_sha": git.get("sha"),
        "worktree_fingerprint_sha256": git.get("worktree_fingerprint_sha256"),
        "runtime_identity": runtime_identity,
        "source_receipt": source_receipt,
        "source_receipt_sha256": _sha256_bytes(receipt_bytes),
        "classification": classification(),
    }
    state_path = output / "run-state.json"
    if output.exists():
        if not resume:
            raise BatchFrontierError(f"output already exists; use --resume: {output}")
        state = _load_json(state_path)
        if any(state.get(key) != value for key, value in identity.items()):
            raise BatchFrontierError("resume identity differs from initialized batch frontier")
        if (output / "recipe.yaml").read_bytes() != recipe_bytes:
            raise BatchFrontierError("frozen recipe changed")
        if (output / "environment" / "uv.lock").read_bytes() != lock_bytes:
            raise BatchFrontierError("frozen lockfile changed")
        if (output / "source-receipt.json").read_bytes() != receipt_bytes:
            raise BatchFrontierError("frozen source receipt changed")
        if state.get("status") not in {INCOMPLETE_STATUS, COMPLETE_STATUS}:
            raise BatchFrontierError("resume state has an unknown status")
        return output, state

    state = {
        **identity,
        "status": INCOMPLETE_STATUS,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": None,
        "capacity_probe_sha256": None,
        "feasible_batches": None,
        "schedule": None,
        "completed_blocks": [],
        "block_sha256": {},
        "block_attempts": [],
        "current_block": None,
        "sessions": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "environment").mkdir()
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "environment" / "uv.lock").write_bytes(lock_bytes)
        _atomic_write_bytes(staging / "source-receipt.json", receipt_bytes)
        _atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _preserve_preflight(recipe: BatchFrontierRecipe, root: Path, output: Path) -> dict[str, str]:
    source = _relative(root, recipe.preflight_report)
    if source.is_symlink() or not source.is_file():
        raise BatchFrontierQualificationError("qualified preflight report is unavailable")
    payload = source.read_bytes()
    digest = _sha256_bytes(payload)
    relative = f"qualification/preflight-{digest}.json"
    destination = _relative(output, relative)
    if destination.exists() and destination.read_bytes() != payload:
        raise BatchFrontierError("content-addressed preflight changed")
    if not destination.exists():
        _atomic_write_bytes(destination, payload)
    return {"path": relative, "sha256": digest}


def _prepare_corpus(
    recipe: BatchFrontierRecipe,
    root: Path,
    output: Path,
    state: dict[str, Any],
) -> base.Corpus:
    import numpy as np

    source = _relative(_relative(root, recipe.quality_source.root), recipe.dataset.artifact)
    destination = _relative(output, recipe.dataset.artifact)
    if source.is_symlink() or not source.is_file():
        raise BatchFrontierQualificationError("sealed seed2723 corpus is unavailable")
    source_bytes = source.read_bytes()
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != source_bytes:
            raise BatchFrontierError("copied seed2723 corpus changed")
    else:
        _atomic_write_bytes(destination, source_bytes)
    try:
        with np.load(destination, allow_pickle=False) as archive:
            coords = archive["coords"]
            demands = archive["demands"]
            capacity = float(archive["capacity"].item())
            embedded = str(archive["content_sha256"].item())
    except Exception as exc:
        raise BatchFrontierError("seed2723 corpus is unreadable") from exc
    content = base._content_sha256(coords, demands, capacity)
    expected = "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf"
    if (
        coords.shape != (recipe.dataset.num_instances, recipe.dataset.size + 1, 2)
        or demands.shape != (recipe.dataset.num_instances, recipe.dataset.size + 1)
        or not math.isclose(capacity, recipe.dataset.capacity, rel_tol=0.0, abs_tol=0.0)
        or embedded != expected
        or content != expected
    ):
        raise BatchFrontierError("seed2723 corpus identity changed")
    record = {
        "dataset_id": recipe.dataset.dataset_id,
        "seed": recipe.dataset.seed,
        "instances": recipe.dataset.num_instances,
        "content_sha256": content,
        "file_sha256": _sha256_file(destination),
        "path": recipe.dataset.artifact,
    }
    if state.get("dataset") not in (None, record):
        raise BatchFrontierError("durable dataset identity changed")
    state["dataset"] = record
    return base.Corpus(coords, demands, capacity, content, record["file_sha256"], destination)


def _state_structure_sha256(model: Any) -> str:
    structure = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in model.state_dict().items()
    ]
    return _canonical_sha256(structure)


def _build_model(
    recipe: BatchFrontierRecipe,
    entry: dict[str, Any],
    *,
    load_checkpoint: bool,
) -> tuple[Any, Any]:
    import torch

    from neuro_co.core.factory import make_env, make_model

    env = make_env(
        recipe.dataset.problem,
        size=recipe.dataset.size,
        capacity=recipe.dataset.capacity,
        max_demand=recipe.dataset.max_demand,
    )
    architecture = str(entry["architecture"])
    model = make_model(
        env,
        backbone=architecture,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
    )
    if architecture == "gnn":
        from torch_geometric.nn import TransformerConv

        from neuro_co.core.models.encoders.gnn import GNNEncoder

        encoder = model.encoder
        expected = entry.get("model_configuration", {})
        attributes = {
            "in_dim": encoder.in_dim,
            "hidden_dim": encoder.hidden_dim,
            "num_layers": encoder.num_layers,
            "num_heads": encoder.num_heads,
            "k_sparse": encoder.k_sparse,
            "dropout": encoder.dropout,
            "rbf_k": encoder.rbf_k,
            "fourier_feats": encoder.fourier_feats,
            "residual": encoder.residual,
        }
        expected_attributes = {
            "in_dim": 3,
            "hidden_dim": expected.get("hidden_dim"),
            "num_layers": expected.get("num_layers"),
            "num_heads": expected.get("num_heads"),
            "k_sparse": expected.get("k_sparse"),
            "dropout": expected.get("dropout"),
            "rbf_k": expected.get("rbf_k"),
            "fourier_feats": expected.get("fourier_feats"),
            "residual": expected.get("residual"),
        }
        if (
            not isinstance(encoder, GNNEncoder)
            or attributes != expected_attributes
            or len(encoder.blocks) != 3
            or any(not isinstance(block.gnn, TransformerConv) for block in encoder.blocks)
        ):
            raise BatchFrontierQualificationError("GNN runtime structure changed")
    identity = entry.get("model_identity", {})
    if load_checkpoint and (
        identity.get("parameter_count")
        != sum(parameter.numel() for parameter in model.parameters())
        or identity.get("state_dict_structure_sha256") != _state_structure_sha256(model)
        or identity.get("factory_backbone", architecture) != architecture
    ):
        raise BatchFrontierQualificationError("checkpoint model structure changed")
    if load_checkpoint:
        payload = quality._load_torch_mapping(entry["checkpoint_path"])
        if "model_identity" in payload and payload["model_identity"] != identity:
            raise BatchFrontierQualificationError("checkpoint embedded model identity changed")
        state_dict = payload.get("model")
        if not isinstance(state_dict, dict):
            raise BatchFrontierQualificationError("checkpoint lacks a model state")
        model.load_state_dict(state_dict, strict=True)
    device = torch.device(f"cuda:{recipe.gpu_index}")
    model.to(device=device, dtype=torch.float32).eval()
    return env, model


def _random_entry(architecture: str) -> dict[str, Any]:
    gnn = {
        "encoder": "GNNEncoder",
        "normalization": "batch",
        "prenorm": True,
        "k_sparse": 10,
        "dropout": 0.1,
        "sparsify": True,
        "edge_features": True,
        "rbf_k": 16,
        "fourier_feats": 1,
        "residual": True,
        "decoder": "mlco-pointer-decoder",
    }
    return {
        "architecture": architecture,
        "model_configuration": {
            "backbone": architecture,
            "hidden_dim": 128,
            "num_layers": 3,
            "num_heads": 8,
            **(gnn if architecture == "gnn" else {}),
        },
        "model_identity": {},
    }


def _cycle(
    recipe: BatchFrontierRecipe,
    corpus: base.Corpus,
    env: Any,
    model: Any,
    *,
    batch_size: int,
    evaluation_seed: int,
    limit: int | None = None,
) -> tuple[Any, ...]:
    import torch

    stop_at = corpus.coords.shape[0] if limit is None else limit
    routes: list[Any] = []
    device = torch.device(f"cuda:{recipe.gpu_index}")
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, stop_at, batch_size):
            stop = min(start + batch_size, stop_at)
            state = quality._initial_state(
                env, corpus.coords[start:stop], corpus.demands[start:stop], device
            )
            routes.extend(
                quality._multistart_routes(
                    model,
                    env,
                    state,
                    n_starts=recipe.neural_policy.n_starts,
                    augmentations=recipe.neural_policy.augmentations,
                    seed=evaluation_seed + start,
                )
            )
    torch.cuda.synchronize(device)
    return tuple(routes)


def _capacity_probe(
    recipe: BatchFrontierRecipe,
    corpus: base.Corpus,
    entries: dict[tuple[str, int], dict[str, Any]] | None,
    *,
    load_checkpoints: bool,
) -> dict[str, Any]:
    import torch

    architectures: dict[str, Any] = {}
    device = torch.device(f"cuda:{recipe.gpu_index}")
    properties = torch.cuda.get_device_properties(device)
    for architecture in ARCHITECTURES:
        torch.manual_seed(2700 + ARCHITECTURES.index(architecture))
        entry = (
            entries[(architecture, recipe.capacity_probe.checkpoint_seed)]
            if entries is not None
            else _random_entry(architecture)
        )
        env, model = _build_model(recipe, entry, load_checkpoint=load_checkpoints)
        records: list[dict[str, Any]] = []
        feasible: list[int] = []
        failed = False
        for batch_size in recipe.neural_policy.batch_candidates:
            if failed:
                records.append(
                    {
                        "batch_size": batch_size,
                        "status": "not_probed_after_first_infeasible",
                    }
                )
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = int(torch.cuda.memory_allocated(device))
            reserved_before = int(torch.cuda.memory_reserved(device))
            started = time.perf_counter()
            try:
                routes = _cycle(
                    recipe,
                    corpus,
                    env,
                    model,
                    batch_size=batch_size,
                    evaluation_seed=recipe.neural_policy.evaluation_seeds[0],
                    limit=batch_size,
                )
                validation = quality.validate_routes(
                    corpus.coords[:batch_size],
                    corpus.demands[:batch_size],
                    corpus.capacity,
                    routes,
                )
                if validation.get("complete") is not True:
                    raise BatchFrontierError("capacity probe produced invalid routes")
                feasible.append(batch_size)
                records.append(
                    {
                        "batch_size": batch_size,
                        "status": "feasible",
                        "elapsed_s": time.perf_counter() - started,
                        "peak_memory_allocated_b": int(torch.cuda.max_memory_allocated(device)),
                        "peak_memory_reserved_b": int(torch.cuda.max_memory_reserved(device)),
                        "memory_allocated_before_b": allocated_before,
                        "memory_reserved_before_b": reserved_before,
                        "device_total_memory_b": int(properties.total_memory),
                        "route_output_sha256": base._route_hash(routes, validation),
                        "mean_cost": float(validation["mean_cost"]),
                    }
                )
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                is_oom = (
                    isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                )
                if not is_oom:
                    raise
                failed = True
                records.append(
                    {
                        "batch_size": batch_size,
                        "status": "infeasible_cuda_oom",
                        "elapsed_s": time.perf_counter() - started,
                        "peak_memory_allocated_b": int(torch.cuda.max_memory_allocated(device)),
                        "peak_memory_reserved_b": int(torch.cuda.max_memory_reserved(device)),
                        "device_total_memory_b": int(properties.total_memory),
                        "exception_type": type(exc).__name__,
                    }
                )
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        if not feasible:
            raise BatchFrontierQualificationError(f"{architecture} has no feasible neural batch")
        architectures[architecture] = {
            "feasible_prefix": feasible,
            "maximum_feasible_batch_size": feasible[-1],
            "records": records,
        }
    return {
        "schema_version": CAPACITY_SCHEMA,
        "status": "complete",
        "measured_energy": False,
        "checkpoint_backed": load_checkpoints,
        "gpu_index": recipe.gpu_index,
        "batch_candidates": list(recipe.neural_policy.batch_candidates),
        "architectures": architectures,
    }


def estimate_only(
    recipe_path: Path,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Run a pre-attestation CUDA capacity and timing estimate without writes."""

    root = (workspace_root or Path.cwd()).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    _configure_runtime(recipe)
    dummy_state: dict[str, Any] = {"dataset": None}
    source = _relative(_relative(root, recipe.quality_source.root), recipe.dataset.artifact)
    import numpy as np

    with np.load(source, allow_pickle=False) as archive:
        corpus = base.Corpus(
            archive["coords"],
            archive["demands"],
            float(archive["capacity"].item()),
            str(archive["content_sha256"].item()),
            _sha256_file(source),
            source,
        )
    del dummy_state
    probe = _capacity_probe(recipe, corpus, None, load_checkpoints=False)
    feasible = {
        architecture: tuple(record["feasible_prefix"])
        for architecture, record in probe["architectures"].items()
    }
    schedule = expected_blocks(recipe, feasible)
    full_capacity = {
        architecture: tuple(recipe.neural_policy.batch_candidates) for architecture in ARCHITECTURES
    }
    full_schedule = expected_blocks(recipe, full_capacity)
    result = {
        "schema_version": "aet-batch-frontier-estimate/v1",
        "status": "complete",
        "writes_performed": False,
        "energy_measured": False,
        "execution_layer": "windows-native",
        "architectures": list(ARCHITECTURES),
        "capacity_probe": probe,
        "resolved_feasible_batches": {
            architecture: list(values) for architecture, values in feasible.items()
        },
        "resolved_neural_cells_per_round": sum(len(values) for values in feasible.values()),
        "resolved_block_count": len(schedule),
        "resolved_hgs_block_count": recipe.paired_rounds,
        "minimum_measured_walltime_s": len(schedule) * recipe.minimum_block_duration_s,
        "minimum_measured_walltime_hours": len(schedule) * recipe.minimum_block_duration_s / 3600,
        "full_capacity_block_count": len(full_schedule),
        "full_capacity_minimum_measured_walltime_s": (
            len(full_schedule) * recipe.minimum_block_duration_s
        ),
    }
    return result


def _execute_neural_block(
    recipe: BatchFrontierRecipe,
    block: FrontierBlock,
    *,
    corpus: base.Corpus,
    entry: dict[str, Any],
    attestation: dict[str, Any],
) -> dict[str, Any]:
    import torch

    if block.architecture is None or block.batch_size is None:
        raise BatchFrontierError("neural block lacks architecture or batch size")
    env, model = _build_model(recipe, entry, load_checkpoint=True)
    warmup_limit = min(block.batch_size, recipe.dataset.num_instances)
    warmup = _cycle(
        recipe,
        corpus,
        env,
        model,
        batch_size=block.batch_size,
        evaluation_seed=block.evaluation_seed,
        limit=warmup_limit,
    )
    warmup_validation = quality.validate_routes(
        corpus.coords[:warmup_limit],
        corpus.demands[:warmup_limit],
        corpus.capacity,
        warmup,
    )
    if warmup_validation.get("complete") is not True:
        raise BatchFrontierError("neural warmup produced invalid routes")
    before = base._diagnostic_gpu_snapshot(recipe.gpu_index)
    tracker = base._make_strict_tracker(
        f"batch-frontier-{block.architecture}-b{block.batch_size}-r{block.round_index}",
        recipe,  # type: ignore[arg-type]
    )
    repetitions = 0
    first_routes: tuple[Any, ...] | None = None
    last_routes: tuple[Any, ...] | None = None
    wall_started = time.perf_counter()
    with tracker as active:
        measured_started = time.perf_counter()
        while True:
            current = _cycle(
                recipe,
                corpus,
                env,
                model,
                batch_size=block.batch_size,
                evaluation_seed=block.evaluation_seed,
            )
            first_routes = current if first_routes is None else first_routes
            last_routes = current
            repetitions += 1
            elapsed = time.perf_counter() - measured_started
            if time.perf_counter() - wall_started > recipe.maximum_block_walltime_s:
                raise BatchFrontierError("neural block exceeded maximum wall time")
            if elapsed >= recipe.minimum_block_duration_s:
                break
        active.n_items = repetitions * recipe.dataset.num_instances
    after = base._diagnostic_gpu_snapshot(recipe.gpu_index)
    if first_routes is None or last_routes is None:
        raise BatchFrontierError("neural block completed no corpus repetition")
    first_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, first_routes
    )
    last_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, last_routes
    )
    first_hash = base._route_hash(first_routes, first_validation)
    last_hash = base._route_hash(last_routes, last_validation)
    if (
        first_validation.get("complete") is not True
        or last_validation.get("complete") is not True
        or first_hash != last_hash
    ):
        raise BatchFrontierError("neural block output is invalid or inconsistent")
    instances = repetitions * recipe.dataset.num_instances
    energy = base._checked_energy(tracker, recipe, items_processed=instances)  # type: ignore[arg-type]
    duration = float(energy["duration_s"])
    del model
    torch.cuda.empty_cache()
    return {
        "schema_version": BLOCK_SCHEMA,
        "status": "complete",
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": "neural",
        "architecture": block.architecture,
        "batch_size": block.batch_size,
        "training_seed": block.training_seed,
        "evaluation_seed": block.evaluation_seed,
        "dataset_content_sha256": corpus.content_sha256,
        "attestation": attestation,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "repetitions": repetitions,
        "instances_per_repetition": recipe.dataset.num_instances,
        "instances_processed": instances,
        "duration_s": duration,
        "throughput_instances_per_s": instances / duration,
        "energy": energy,
        "validation": first_validation,
        "first_measured_output_sha256": first_hash,
        "last_measured_output_sha256": last_hash,
        "first_and_last_outputs_identical": True,
        "checkpoint": {
            "path": entry["path"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "model_state_sha256": entry["model_state_sha256"],
            "model_identity_sha256": entry["model_identity_sha256"],
        },
        "gpu_process_snapshot_before": before,
        "gpu_process_snapshot_after": after,
        "gpu_process_lists_are_diagnostic_only": True,
    }


def _execute_hgs_block(
    recipe: BatchFrontierRecipe,
    block: FrontierBlock,
    *,
    corpus: base.Corpus,
    attestation: dict[str, Any],
) -> dict[str, Any]:
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

    def cycle(coords: Any = corpus.coords, demands: Any = corpus.demands) -> Sequence[Any]:
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
    warmup_routes, _ = base._validated_hgs_results(
        warmup_results,
        expected_instances=4,
        base_seed=block.hgs_seed,
        max_iterations=recipe.hgs_policy.max_iterations,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    if (
        quality.validate_routes(
            corpus.coords[:4], corpus.demands[:4], corpus.capacity, warmup_routes
        ).get("complete")
        is not True
    ):
        raise BatchFrontierError("HGS warmup produced invalid routes")
    before = base._diagnostic_gpu_snapshot(recipe.gpu_index)
    tracker = base._make_strict_tracker(
        f"batch-frontier-hgs-r{block.round_index}",
        recipe,  # type: ignore[arg-type]
    )
    repetitions = 0
    first_results: Sequence[Any] | None = None
    last_results: Sequence[Any] | None = None
    wall_started = time.perf_counter()
    with tracker as active:
        measured_started = time.perf_counter()
        while True:
            current = cycle()
            first_results = current if first_results is None else first_results
            last_results = current
            repetitions += 1
            elapsed = time.perf_counter() - measured_started
            if time.perf_counter() - wall_started > recipe.maximum_block_walltime_s:
                raise BatchFrontierError("HGS block exceeded maximum wall time")
            if elapsed >= recipe.minimum_block_duration_s:
                break
        active.n_items = repetitions * recipe.dataset.num_instances
    after = base._diagnostic_gpu_snapshot(recipe.gpu_index)
    if first_results is None or last_results is None:
        raise BatchFrontierError("HGS block completed no corpus repetition")
    first_routes, first_metadata = base._validated_hgs_results(
        first_results,
        expected_instances=recipe.dataset.num_instances,
        base_seed=block.hgs_seed,
        max_iterations=recipe.hgs_policy.max_iterations,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    last_routes, last_metadata = base._validated_hgs_results(
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
    first_hash = base._route_hash(first_routes, first_validation, first_metadata)
    last_hash = base._route_hash(last_routes, last_validation, last_metadata)
    if (
        first_validation.get("complete") is not True
        or last_validation.get("complete") is not True
        or first_hash != last_hash
    ):
        raise BatchFrontierError("HGS block output is invalid or inconsistent")
    instances = repetitions * recipe.dataset.num_instances
    energy = base._checked_energy(tracker, recipe, items_processed=instances)  # type: ignore[arg-type]
    duration = float(energy["duration_s"])
    return {
        "schema_version": BLOCK_SCHEMA,
        "status": "complete",
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": "hgs",
        "architecture": None,
        "batch_size": None,
        "training_seed": block.training_seed,
        "hgs_seed": block.hgs_seed,
        "dataset_content_sha256": corpus.content_sha256,
        "attestation": attestation,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "repetitions": repetitions,
        "instances_per_repetition": recipe.dataset.num_instances,
        "instances_processed": instances,
        "duration_s": duration,
        "throughput_instances_per_s": instances / duration,
        "energy": energy,
        "validation": first_validation,
        "first_measured_output_sha256": first_hash,
        "last_measured_output_sha256": last_hash,
        "first_and_last_outputs_identical": True,
        "solver_result_metadata": first_metadata,
        "first_solver_result_metadata_sha256": _sha256_bytes(
            _json_bytes({"solver_result_metadata": first_metadata})
        ),
        "last_solver_result_metadata_sha256": _sha256_bytes(
            _json_bytes({"solver_result_metadata": last_metadata})
        ),
        "gpu_process_snapshot_before": before,
        "gpu_process_snapshot_after": after,
        "gpu_process_lists_are_diagnostic_only": True,
    }


def _validate_block(
    payload: dict[str, Any],
    recipe: BatchFrontierRecipe,
    block: FrontierBlock,
    corpus: base.Corpus,
) -> None:
    expected = {
        "schema_version": BLOCK_SCHEMA,
        "status": "complete",
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": block.policy,
        "architecture": block.architecture,
        "batch_size": block.batch_size,
        "training_seed": block.training_seed,
        "dataset_content_sha256": corpus.content_sha256,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "instances_per_repetition": recipe.dataset.num_instances,
        "first_and_last_outputs_identical": True,
    }
    if block.policy == "neural":
        expected["evaluation_seed"] = block.evaluation_seed
    else:
        expected["hgs_seed"] = block.hgs_seed
    if any(payload.get(key) != value for key, value in expected.items()):
        raise BatchFrontierError(f"cached block identity changed: {block.relative_path}")
    repetitions = payload.get("repetitions")
    duration = payload.get("duration_s")
    validation = payload.get("validation")
    costs = validation.get("costs") if isinstance(validation, dict) else None
    first_hash = payload.get("first_measured_output_sha256")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or payload.get("instances_processed") != repetitions * recipe.dataset.num_instances
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) < recipe.minimum_block_duration_s
        or not isinstance(validation, dict)
        or validation.get("complete") is not True
        or not isinstance(costs, list)
        or len(costs) != recipe.dataset.num_instances
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in costs
        )
        or not isinstance(first_hash, str)
        or len(first_hash) != 64
        or any(character not in "0123456789abcdef" for character in first_hash)
        or first_hash != payload.get("last_measured_output_sha256")
    ):
        raise BatchFrontierError(f"cached block metrics changed: {block.relative_path}")
    energy = payload.get("energy")
    if not isinstance(energy, dict):
        raise BatchFrontierError("cached block lacks energy")
    for key in (
        "cpu_package_energy_j_per_instance",
        "gpu_energy_j_per_instance",
        "observed_component_energy_j_per_instance",
    ):
        value = energy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise BatchFrontierError(f"cached block energy changed: {key}")


def _series_summary(values: list[float]) -> dict[str, Any]:
    average = mean(values)
    sample_sd = stdev(values) if len(values) > 1 else 0.0
    half_width = 2.7764451051977987 * sample_sd / math.sqrt(5) if len(values) == 5 else None
    return {
        "n": len(values),
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


def _cell_quality_gate(
    architecture: str,
    batch_size: int,
    neural_by_round: dict[int, dict[str, Any]],
    architecture_quality: dict[str, Any],
    quality_context: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen training gate independently to one measured batch cell."""

    import numpy as np

    from neuro_co.aet.experiments import hgs_holdout_runner as holdout
    from neuro_co.aet.experiments import training_debt_runner as training

    gate = quality_context.get("quality_gate")
    reference_costs = quality_context.get("reference_costs")
    reference_sha = quality_context.get("reference_sha256")
    if gate is None or not isinstance(reference_costs, list) or not isinstance(reference_sha, str):
        raise BatchFrontierError("cell quality context is incomplete")
    gaps: list[list[float]] = []
    invalid_instances = 0
    for round_index in range(5):
        validation = neural_by_round[round_index].get("validation")
        if not isinstance(validation, dict):
            raise BatchFrontierError("neural block lacks its stored route validation")
        metrics = quality._gap_summary(validation.get("costs"), reference_costs)
        round_gaps = metrics.get("gaps_pct")
        if not isinstance(round_gaps, list) or len(round_gaps) != len(reference_costs):
            raise BatchFrontierError("neural block quality vector is malformed")
        gaps.append([float(value) for value in round_gaps])
        invalid = validation.get("invalid_instance_count")
        if isinstance(invalid, bool) or not isinstance(invalid, int) or invalid < 0:
            raise BatchFrontierError("neural block invalid-instance count is malformed")
        invalid_instances += invalid
    matrix = np.asarray(gaps, dtype=np.float64)
    mean_samples, q95_samples = training._bootstrap_samples(
        matrix,
        replicates=gate.bootstrap_replicates,
        seed=gate.bootstrap_seed + (0 if architecture == "am" else 1),
    )
    measured_summary = holdout._summarize_policy(
        matrix,
        policy_id=f"{architecture}_batch_{batch_size}_seeds2_6",
        policy_role="batch_frontier_cell_quality_qualification",
        invalid_instances=invalid_instances,
        threshold_pct=gate.threshold_pct,
        t_critical_value=gate.t_critical_value,
        t_degrees_of_freedom=gate.t_degrees_of_freedom,
        bootstrap_mean_samples=mean_samples,
        bootstrap_q95_samples=q95_samples,
        bootstrap_quantile=gate.bootstrap_quantile,
    )
    training_passed = architecture_quality.get("frontier_eligible") is True
    measured_passed = measured_summary.get("passed") is True
    passed = training_passed and measured_passed
    joint_classification = (
        measured_summary.get("classification")
        if training_passed
        else "architecture_training_gate_ineligible_no_post_hoc_rescue"
    )
    summary = {
        **measured_summary,
        "passed": passed,
        "classification": joint_classification,
        "all_required_conditions_satisfied": passed,
        "architecture_training_gate_passed": training_passed,
        "measured_output_gate_passed": measured_passed,
        "measured_output_classification": measured_summary.get("classification"),
        "joint_eligibility_rule": "architecture_training_gate_and_measured_batch_gate",
        "post_hoc_rescue_allowed": False,
    }
    return {
        "quality_status": "feasible" if passed else "infeasible",
        "frontier_eligible": passed,
        "quality_gate": {
            "passed": passed,
            "status": "quality_passed" if passed else "quality_nonpass",
            "threshold_pct": gate.threshold_pct,
            "summary": summary,
            "measured_output_summary": measured_summary,
            "reference_sha256": reference_sha,
            "bootstrap_replicates": gate.bootstrap_replicates,
            "bootstrap_seed": gate.bootstrap_seed + (0 if architecture == "am" else 1),
            "bootstrap_quantile": gate.bootstrap_quantile,
            "training_gate_anchor_batch_size": quality_context.get("provenance", {}).get(
                "training_gate_anchor_batch_size"
            ),
        },
    }


def batch_frontier_summary(
    recipe: BatchFrontierRecipe,
    source_receipt: dict[str, Any],
    capacity: dict[str, Any],
    payloads: Sequence[dict[str, Any]],
    *,
    quality_context: dict[str, Any],
) -> dict[str, Any]:
    hgs_payloads = [payload for payload in payloads if payload["policy"] == "hgs"]
    hgs_by_round = {int(payload["round"]): payload for payload in hgs_payloads}
    if len(hgs_payloads) != 5 or set(hgs_by_round) != set(range(5)):
        raise BatchFrontierError("summary requires one HGS block in every round")
    architectures: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        quality_receipt = source_receipt["training"]["architecture_quality"][architecture]
        training_quality_status = quality_receipt["quality_status"]
        if training_quality_status not in {"feasible", "infeasible"}:
            raise BatchFrontierError(
                f"training quality status for {architecture} is not classifiable"
            )
        feasible = capacity["architectures"][architecture]["feasible_prefix"]
        batches: list[dict[str, Any]] = []
        for batch_size in feasible:
            neural_by_round = {
                int(payload["round"]): payload
                for payload in payloads
                if payload["policy"] == "neural"
                and payload["architecture"] == architecture
                and payload["batch_size"] == batch_size
            }
            if set(neural_by_round) != set(range(5)):
                raise BatchFrontierError("summary requires five neural blocks per feasible cell")
            cell_quality = _cell_quality_gate(
                architecture,
                batch_size,
                neural_by_round,
                quality_receipt,
                quality_context,
            )
            pairs: list[dict[str, Any]] = []
            primary: list[float] = []
            same_host: list[float] = []
            route_hashes_by_round: dict[int, str] = {}
            for round_index in range(5):
                neural = neural_by_round[round_index]
                hgs = hgs_by_round[round_index]
                neural_energy = neural["energy"]
                hgs_energy = hgs["energy"]
                neural_total = float(neural_energy["observed_component_energy_j_per_instance"])
                hgs_cpu = float(hgs_energy["cpu_package_energy_j_per_instance"])
                hgs_total = float(hgs_energy["observed_component_energy_j_per_instance"])
                delta = hgs_cpu - neural_total
                same_host_delta = hgs_total - neural_total
                primary.append(delta)
                same_host.append(same_host_delta)
                route_hashes_by_round[round_index] = str(neural["first_measured_output_sha256"])
                pairs.append(
                    {
                        "round": round_index,
                        "training_seed": neural["training_seed"],
                        "neural_observed_components_j_per_instance": neural_total,
                        "hgs_cpu_package_j_per_instance": hgs_cpu,
                        "hgs_observed_components_j_per_instance": hgs_total,
                        "delta_hgs_minus_neural_j_per_instance": delta,
                        "same_host_observed_delta_hgs_minus_neural_j_per_instance": same_host_delta,
                        "neural_throughput_instances_per_s": neural["throughput_instances_per_s"],
                        "hgs_throughput_instances_per_s": hgs["throughput_instances_per_s"],
                        "neural_output_sha256": neural["first_measured_output_sha256"],
                    }
                )
            batches.append(
                {
                    "batch_size": batch_size,
                    **cell_quality,
                    "pairs": pairs,
                    "delta_hgs_minus_neural_j_per_instance": _series_summary(primary),
                    "same_host_observed_delta_hgs_minus_neural_j_per_instance": _series_summary(
                        same_host
                    ),
                    "output_hashes_by_round": route_hashes_by_round,
                    "output_repeat_consistency": "first_last_identical",
                }
            )
        by_round: list[dict[str, Any]] = []
        for round_index in range(5):
            hashes_by_batch = {
                str(batch["batch_size"]): batch["output_hashes_by_round"][round_index]
                for batch in batches
            }
            by_round.append(
                {
                    "round": round_index,
                    "all_batch_hashes_equal": len(set(hashes_by_batch.values())) == 1,
                    "distinct_hash_count": len(set(hashes_by_batch.values())),
                    "hashes_by_batch_size": hashes_by_batch,
                }
            )
        all_equal = all(item["all_batch_hashes_equal"] for item in by_round)
        for batch in batches:
            batch["route_hash_diagnostic"] = {
                "first_last_exact_repeat_required": True,
                "cross_batch_hashes_used_as_quality_gate": False,
                "all_batch_hashes_equal_in_every_round": all_equal,
                "by_round": [
                    {
                        "round": item["round"],
                        "matches_all_batches": item["all_batch_hashes_equal"],
                    }
                    for item in by_round
                ],
            }
        architectures[architecture] = {
            "capacity": capacity["architectures"][architecture],
            "quality_status": training_quality_status,
            "frontier_eligible": quality_receipt["frontier_eligible"],
            "quality_gate": quality_receipt["quality_gate"],
            "cross_batch_route_hash_diagnostic": {
                "used_as_gate": False,
                "all_batch_hashes_equal_in_every_round": all_equal,
                "by_round": by_round,
            },
            "batches": batches,
        }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "training_source": {
            "training_manifest_sha256": source_receipt["training"]["training_manifest_sha256"],
            "models": source_receipt["training"]["models"],
            "architecture_quality": source_receipt["training"]["architecture_quality"],
        },
        "cell_quality_gate_provenance": quality_context.get("provenance"),
        "primary_delta_definition": (
            "HGS CPU-package energy minus neural CPU-package-plus-GPU energy per instance"
        ),
        "primary_delta_positive_interpretation": "lower observed component energy for neural",
        "architectures": architectures,
        "hgs_reference": {
            "policy": "pyvrp-hgs",
            "max_iterations": recipe.hgs_policy.max_iterations,
            "shared_once_per_round": True,
            "rounds": 5,
        },
        "whole_system_energy": False,
        "carbon_accounting": "none",
        "aet_computed": False,
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


def _complete(
    recipe: BatchFrontierRecipe,
    output: Path,
    state: dict[str, Any],
    capacity: dict[str, Any],
    payloads: list[dict[str, Any]],
    quality_context: dict[str, Any],
    *,
    finalization: dict[str, Any] | None = None,
) -> BatchFrontierResult:
    summary = batch_frontier_summary(
        recipe,
        state["source_receipt"],
        capacity,
        payloads,
        quality_context=quality_context,
    )
    _atomic_write_json(output / "batch-frontier-summary.json", summary)
    summary_sha = _sha256_file(output / "batch-frontier-summary.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "dataset": state["dataset"],
        "capacity_probe_sha256": state["capacity_probe_sha256"],
        "resolved_block_count": len(payloads),
        "completed_block_count": len(payloads),
        "paired_rounds": 5,
        "hgs_blocks": 5,
        "batch_frontier_summary_sha256": summary_sha,
        "source_receipt_sha256": state["source_receipt_sha256"],
        "runtime_identity": state["runtime_identity"],
        "torch_geometric_version": state["runtime_identity"]["versions"]["torch_geometric"],
        "edge_index_backend": "torch_cdist_topk",
        "training_was_run_by_frontier": False,
        "capacity_probe_energy_measured": False,
        "finalization": finalization,
        "aet_was_computed": False,
        "carbon_was_computed": False,
    }
    _atomic_write_json(output / "manifest.json", manifest)
    state["status"] = COMPLETE_STATUS
    if finalization is not None:
        state["finalization"] = finalization
    state["completed_at"] = datetime.now(UTC).isoformat()
    state["batch_frontier_summary_sha256"] = summary_sha
    state["manifest_sha256"] = _sha256_file(output / "manifest.json")
    state["checksums_sha256"] = _write_checksums(output)
    _atomic_write_json(output / "run-state.json", state)
    return BatchFrontierResult(
        output,
        COMPLETE_STATUS,
        len(payloads),
        output / "manifest.json",
        state["manifest_sha256"],
    )


def _validate_existing_corpus(
    recipe: BatchFrontierRecipe,
    output: Path,
    state: dict[str, Any],
) -> base.Corpus:
    """Load an already copied corpus without creating or repairing artifacts."""

    import numpy as np

    path = _relative(output, recipe.dataset.artifact)
    if path.is_symlink() or not path.is_file():
        raise BatchFrontierError("completed-prefix corpus is missing")
    try:
        with np.load(path, allow_pickle=False) as archive:
            coords = archive["coords"]
            demands = archive["demands"]
            capacity = float(archive["capacity"].item())
            embedded = str(archive["content_sha256"].item())
    except Exception as exc:
        raise BatchFrontierError("completed-prefix corpus is unreadable") from exc
    content = base._content_sha256(coords, demands, capacity)
    expected = "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf"
    record = {
        "dataset_id": recipe.dataset.dataset_id,
        "seed": recipe.dataset.seed,
        "instances": recipe.dataset.num_instances,
        "content_sha256": content,
        "file_sha256": _sha256_file(path),
        "path": recipe.dataset.artifact,
    }
    if (
        coords.shape != (recipe.dataset.num_instances, recipe.dataset.size + 1, 2)
        or demands.shape != (recipe.dataset.num_instances, recipe.dataset.size + 1)
        or not math.isclose(capacity, recipe.dataset.capacity, rel_tol=0.0, abs_tol=0.0)
        or embedded != expected
        or content != expected
        or state.get("dataset") != record
    ):
        raise BatchFrontierError("completed-prefix corpus identity changed")
    return base.Corpus(coords, demands, capacity, content, record["file_sha256"], path)


def _validate_existing_capacity_and_schedule(
    recipe: BatchFrontierRecipe,
    output: Path,
    state: dict[str, Any],
) -> tuple[dict[str, Any], tuple[FrontierBlock, ...]]:
    capacity_path = output / "capacity-probe.json"
    capacity_sha = state.get("capacity_probe_sha256")
    if (
        not isinstance(capacity_sha, str)
        or capacity_path.is_symlink()
        or not capacity_path.is_file()
        or _sha256_file(capacity_path) != capacity_sha
    ):
        raise BatchFrontierError("durable capacity probe changed")
    capacity = _load_json(capacity_path)
    architectures = capacity.get("architectures")
    if (
        capacity.get("schema_version") != CAPACITY_SCHEMA
        or capacity.get("status") != COMPLETE_STATUS
        or capacity.get("measured_energy") is not False
        or capacity.get("checkpoint_backed") is not True
        or capacity.get("gpu_index") != recipe.gpu_index
        or capacity.get("batch_candidates") != list(recipe.neural_policy.batch_candidates)
        or not isinstance(architectures, dict)
        or set(architectures) != set(ARCHITECTURES)
    ):
        raise BatchFrontierError("durable capacity probe is malformed")
    feasible: dict[str, tuple[int, ...]] = {}
    for architecture in ARCHITECTURES:
        record = architectures[architecture]
        values = record.get("feasible_prefix") if isinstance(record, dict) else None
        if (
            not isinstance(values, list)
            or not values
            or values != list(recipe.neural_policy.batch_candidates[: len(values)])
            or record.get("maximum_feasible_batch_size") != values[-1]
            or not isinstance(record.get("records"), list)
        ):
            raise BatchFrontierError(f"durable {architecture} capacity prefix is malformed")
        feasible[architecture] = tuple(values)
    if state.get("feasible_batches") != {
        architecture: list(values) for architecture, values in feasible.items()
    }:
        raise BatchFrontierError("durable feasible-batch identity changed")
    schedule = expected_blocks(recipe, feasible)
    if state.get("schedule") != [asdict(block) for block in schedule]:
        raise BatchFrontierError("durable resolved schedule changed")
    return capacity, schedule


def _validate_full_block_prefix(
    recipe: BatchFrontierRecipe,
    output: Path,
    state: dict[str, Any],
    schedule: Sequence[FrontierBlock],
    corpus: base.Corpus,
) -> list[dict[str, Any]]:
    completed = state.get("completed_blocks")
    hashes = state.get("block_sha256")
    expected = [block.relative_path for block in schedule]
    if state.get("current_block") is not None:
        raise BatchFrontierError("cannot finalize while a block attempt is active")
    if completed != expected or not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise BatchFrontierError("finalization requires the complete strict schedule prefix")
    payloads: list[dict[str, Any]] = []
    for block in schedule:
        path = _relative(output, block.relative_path)
        digest = hashes.get(block.relative_path)
        if (
            not isinstance(digest, str)
            or path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != digest
        ):
            raise BatchFrontierError(f"durable block changed: {block.relative_path}")
        payload = _load_json(path)
        _validate_block(payload, recipe, block, corpus)
        payloads.append(payload)
    return payloads


def _validate_complete_anchors(output: Path, state: dict[str, Any]) -> BatchFrontierResult:
    manifest = output / "manifest.json"
    summary = output / "batch-frontier-summary.json"
    checksums = output / "SHA256SUMS"
    if (
        not manifest.is_file()
        or not summary.is_file()
        or not checksums.is_file()
        or _sha256_file(manifest) != state.get("manifest_sha256")
        or _sha256_file(summary) != state.get("batch_frontier_summary_sha256")
        or _sha256_file(checksums) != state.get("checksums_sha256")
    ):
        raise BatchFrontierError("completed batch-frontier anchors changed")
    return BatchFrontierResult(
        output,
        COMPLETE_STATUS,
        len(state["completed_blocks"]),
        manifest,
        _sha256_file(manifest),
    )


def finalize_existing(
    recipe_path: Path,
    workspace_root: Path | None = None,
    *,
    training_source: Path | None = None,
) -> BatchFrontierResult:
    """Finalize a fully measured prefix without probing or measuring anything."""

    root = (workspace_root or Path.cwd()).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    output = _safe_output(root, recipe.output_root)
    state_path = output / "run-state.json"
    if not state_path.is_file():
        raise BatchFrontierError("existing batch-frontier run state is unavailable")
    state = _load_json(state_path)
    if state.get("schema_version") != RUN_STATE_SCHEMA:
        raise BatchFrontierError("existing batch-frontier run state schema changed")

    recipe_bytes = recipe_path.read_bytes()
    frozen_recipe = output / "recipe.yaml"
    lock_bytes = (root / "uv.lock").read_bytes()
    frozen_lock = output / "environment" / "uv.lock"
    if (
        state.get("recipe_sha256") != _sha256_bytes(recipe_bytes)
        or not frozen_recipe.is_file()
        or frozen_recipe.read_bytes() != recipe_bytes
    ):
        raise BatchFrontierError("frozen recipe changed")
    if (
        state.get("uv_lock_sha256") != _sha256_bytes(lock_bytes)
        or not frozen_lock.is_file()
        or frozen_lock.read_bytes() != lock_bytes
    ):
        raise BatchFrontierError("frozen lockfile changed")
    source_path = output / "source-receipt.json"
    if not source_path.is_file():
        raise BatchFrontierError("frozen source receipt is unavailable")
    source_receipt = _load_json(source_path)
    source_bytes = source_path.read_bytes()
    if (
        source_receipt != state.get("source_receipt")
        or source_bytes != _json_bytes(source_receipt)
        or _sha256_bytes(source_bytes) != state.get("source_receipt_sha256")
    ):
        raise BatchFrontierError("frozen source receipt changed")

    corpus = _validate_existing_corpus(recipe, output, state)
    capacity, schedule = _validate_existing_capacity_and_schedule(recipe, output, state)
    payloads = _validate_full_block_prefix(recipe, output, state, schedule, corpus)
    quality_context = _verify_source_receipt(
        recipe,
        root,
        source_receipt,
        training_source,
    )
    if state.get("status") == COMPLETE_STATUS:
        return _validate_complete_anchors(output, state)
    if state.get("status") != INCOMPLETE_STATUS:
        raise BatchFrontierError("existing batch-frontier state is not finalizable")

    current_git = software_recipe._current_git_snapshot(())
    if current_git.get("available") is not True:
        raise BatchFrontierError("finalizer Git identity is unavailable")
    finalization = {
        "mode": "finalize_existing_without_measurement",
        "finalized_at": datetime.now(UTC).isoformat(),
        "measurement_git_sha": state.get("git_sha"),
        "measurement_worktree_fingerprint_sha256": state.get("worktree_fingerprint_sha256"),
        "finalizer_git_sha": current_git.get("sha"),
        "finalizer_worktree_fingerprint_sha256": current_git.get("worktree_fingerprint_sha256"),
        "finalizer_worktree_dirty": current_git.get("dirty"),
        "source_commit_changed_after_measurement": current_git.get("sha") != state.get("git_sha"),
        "measurement_performed": False,
        "capacity_probe_performed": False,
        "inference_performed": False,
        "training_performed": False,
    }
    return _complete(
        recipe,
        output,
        state,
        capacity,
        payloads,
        quality_context,
        finalization=finalization,
    )


def run_batch_frontier(
    recipe_path: Path,
    workspace_root: Path | None = None,
    *,
    resume: bool = False,
    training_source: Path | None = None,
) -> BatchFrontierResult:
    root = (workspace_root or Path.cwd()).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    qualification = qualify_for_execution(recipe, root, training_source)
    if qualification["ready_to_execute"] is not True:
        raise BatchFrontierQualificationError("native Windows or source qualification failed")
    training_root, training_receipt, entries = _resolve_training_source(
        recipe, root, training_source
    )
    del training_root
    source_receipt = {
        "quality": _quality_source_receipt(recipe, root),
        "training": training_receipt,
    }
    quality_context = _quality_gate_context(recipe, root, source_receipt, training_source)
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
    corpus = _prepare_corpus(recipe, root, output, state)
    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise BatchFrontierError("session history is malformed")
    if state.get("status") != COMPLETE_STATUS:
        sessions.append(
            {
                **attestation,
                "preflight_evidence": _preserve_preflight(recipe, root, output),
                "process_started_at": datetime.now(UTC).isoformat(),
                "resume": resume,
                "completed_block_count_at_start": len(state.get("completed_blocks", [])),
            }
        )
        _atomic_write_json(output / "run-state.json", state)
    process_started = time.perf_counter()

    capacity_path = output / "capacity-probe.json"
    if state.get("capacity_probe_sha256") is None:
        capacity = _capacity_probe(recipe, corpus, entries, load_checkpoints=True)
        _atomic_write_json(capacity_path, capacity)
        state["capacity_probe_sha256"] = _sha256_file(capacity_path)
        state["feasible_batches"] = {
            architecture: record["feasible_prefix"]
            for architecture, record in capacity["architectures"].items()
        }
        schedule = expected_blocks(
            recipe,
            {key: tuple(value) for key, value in state["feasible_batches"].items()},
        )
        state["schedule"] = [asdict(block) for block in schedule]
        _atomic_write_json(output / "run-state.json", state)
    else:
        if (
            not capacity_path.is_file()
            or _sha256_file(capacity_path) != state["capacity_probe_sha256"]
        ):
            raise BatchFrontierError("durable capacity probe changed")
        capacity = _load_json(capacity_path)
        schedule = expected_blocks(
            recipe,
            {key: tuple(value) for key, value in (state.get("feasible_batches") or {}).items()},
        )
        if state.get("schedule") != [asdict(block) for block in schedule]:
            raise BatchFrontierError("durable resolved schedule changed")

    completed = state.get("completed_blocks")
    hashes = state.get("block_sha256")
    if not isinstance(completed, list) or not isinstance(hashes, dict):
        raise BatchFrontierError("durable block prefix is malformed")
    expected_prefix = [block.relative_path for block in schedule[: len(completed)]]
    if completed != expected_prefix or set(hashes) != set(completed):
        raise BatchFrontierError("completed blocks are not the strict schedule prefix")
    payloads: list[dict[str, Any]] = []
    for block, relative in zip(schedule, completed, strict=False):
        path = _relative(output, relative)
        if not path.is_file() or _sha256_file(path) != hashes[relative]:
            raise BatchFrontierError(f"durable block changed: {relative}")
        payload = _load_json(path)
        _validate_block(payload, recipe, block, corpus)
        payloads.append(payload)
    if state.get("status") == COMPLETE_STATUS:
        if len(payloads) != len(schedule):
            raise BatchFrontierError("complete state lacks all resolved blocks")
        manifest = output / "manifest.json"
        summary = output / "batch-frontier-summary.json"
        checksums = output / "SHA256SUMS"
        if (
            not manifest.is_file()
            or not summary.is_file()
            or not checksums.is_file()
            or _sha256_file(manifest) != state.get("manifest_sha256")
            or _sha256_file(summary) != state.get("batch_frontier_summary_sha256")
            or _sha256_file(checksums) != state.get("checksums_sha256")
        ):
            raise BatchFrontierError("completed batch-frontier anchors changed")
        return BatchFrontierResult(
            output, COMPLETE_STATUS, len(payloads), manifest, _sha256_file(manifest)
        )

    import torch

    for block in schedule[len(payloads) :]:
        _assert_campaign_active(recipe, attestation, process_started)
        torch.cuda.empty_cache()
        path = _relative(output, block.relative_path)
        if path.exists():
            current = state.get("current_block")
            if not isinstance(current, dict) or current.get("relative_path") != block.relative_path:
                raise BatchFrontierError("unanchored block artifact exists")
            payload = _load_json(path)
            _validate_block(payload, recipe, block, corpus)
            digest = _sha256_file(path)
            prior_attempt = state["block_attempts"][int(current["attempt_index"])]
            prior_attempt["outcome"] = "complete_orphan_promotion_recovered"
            prior_attempt["completed_at"] = datetime.now(UTC).isoformat()
            prior_attempt["artifact_sha256"] = digest
            state["completed_blocks"].append(block.relative_path)
            state["block_sha256"][block.relative_path] = digest
            state["current_block"] = None
            _atomic_write_json(output / "run-state.json", state)
            payloads.append(payload)
            continue
        previous = state.get("current_block")
        if previous is not None:
            if (
                not isinstance(previous, dict)
                or previous.get("relative_path") != block.relative_path
            ):
                raise BatchFrontierError("interrupted attempt is not the next schedule block")
            prior_attempt = state["block_attempts"][int(previous["attempt_index"])]
            prior_attempt["outcome"] = "interrupted_before_artifact_promotion"
            prior_attempt["closed_at"] = datetime.now(UTC).isoformat()
        attempt = {
            "attempt_index": len(state["block_attempts"]),
            "relative_path": block.relative_path,
            "round": block.round_index,
            "order_within_round": block.order_index,
            "policy": block.policy,
            "architecture": block.architecture,
            "batch_size": block.batch_size,
            "session_id": attestation["session_id"],
            "started_at": datetime.now(UTC).isoformat(),
            "outcome": "running",
        }
        state["block_attempts"].append(attempt)
        state["current_block"] = {
            "attempt_index": attempt["attempt_index"],
            "relative_path": block.relative_path,
        }
        _atomic_write_json(output / "run-state.json", state)
        if block.policy == "hgs":
            payload = _execute_hgs_block(recipe, block, corpus=corpus, attestation=attestation)
        else:
            payload = _execute_neural_block(
                recipe,
                block,
                corpus=corpus,
                entry=entries[(str(block.architecture), block.training_seed)],
                attestation=attestation,
            )
        _assert_campaign_active(recipe, attestation, process_started)
        _validate_block(payload, recipe, block, corpus)
        _atomic_write_json(path, payload)
        digest = _sha256_file(path)
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
                    "policy": block.policy,
                    "architecture": block.architecture,
                    "batch_size": block.batch_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return _complete(recipe, output, state, capacity, payloads, quality_context)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--training-source", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-output", type=Path)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.estimate_output is not None and not args.estimate_only:
            raise BatchFrontierQualificationError("--estimate-output requires --estimate-only")
        if args.finalize_existing and (args.estimate_only or args.resume):
            raise BatchFrontierQualificationError(
                "--finalize-existing cannot be combined with --estimate-only or --resume"
            )
        if args.estimate_only:
            estimate = estimate_only(args.recipe)
            if args.estimate_output is not None:
                root = Path.cwd().resolve(strict=True)
                destination = args.estimate_output.resolve(strict=False)
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise BatchFrontierQualificationError(
                        "--estimate-output must remain inside the repository"
                    ) from exc
                _atomic_write_json(destination, estimate)
            print(json.dumps(estimate, indent=2, sort_keys=True), flush=True)
            return 0
        if args.finalize_existing:
            result = finalize_existing(
                args.recipe,
                training_source=args.training_source,
            )
        else:
            result = run_batch_frontier(
                args.recipe,
                resume=args.resume,
                training_source=args.training_source,
            )
    except BatchFrontierError as exc:
        print(f"batch frontier failed: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"batch frontier failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
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
