"""Resumable software-only HGS quality and CPU-energy frontier.

This runner is intentionally limited to the exploratory selection phase.  It
validates the sealed quality-replication bundle in place, evaluates the frozen
neural policy on a fresh selection corpus, and then measures a predeclared HGS
budget grid with Windows EMI.  It never opens a holdout, computes an AET, or
uses observed energy to choose the HGS budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from neuro_co.aet.experiments import quality_replication_runner as replication
from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments import software_smoke_runner as software
from neuro_co.aet.experiments.hgs_frontier_recipe import (
    load_aet_hgs_frontier_recipe,
    runtime_qualification,
)
from neuro_co.aet.experiments.quality_recipe import load_aet_quality_recipe
from neuro_co.aet.experiments.quality_replication_recipe import (
    load_aet_quality_replication_recipe,
)

RUN_STATE_SCHEMA = "aet-hgs-frontier-run-state/v1"
SOURCE_RECEIPT_SCHEMA = "aet-hgs-frontier-replication-source/v1"
REFERENCE_LOCK_SCHEMA = "aet-hgs-frontier-reference-lock/v1"
NEURAL_EVALUATION_SCHEMA = "aet-hgs-frontier-neural-evaluation/v1"
NEURAL_GATE_SCHEMA = "aet-hgs-frontier-neural-quality-gate/v1"
HGS_CELL_SCHEMA = "aet-hgs-frontier-cell/v1"
ASSESSMENT_SCHEMA = "aet-hgs-frontier-assessment/v1"
MANIFEST_SCHEMA = "aet-hgs-frontier-manifest/v1"

INCOMPLETE_STATUS = "incomplete_selection_pending"
COMPLETE_STATUS = "complete_selection_passed"
NO_ELIGIBLE_STATUS = "complete_selection_no_eligible_budget"
NEURAL_NONPASS_STATUS = "complete_selection_neural_policy_nonpass"
INVALIDATED_STATUS = "invalidated_hgs_integrity_failure"
COMPLETE_STATUSES = {COMPLETE_STATUS, NO_ELIGIBLE_STATUS, NEURAL_NONPASS_STATUS}

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PYVRP_INCOMPLETE_SOLUTION = re.compile(
    r"^PyVRP did not return a complete feasible solution for instance ([0-9]+)$"
)


class HGSFrontierError(RuntimeError):
    """Raised when a frontier artifact or execution invariant is invalid."""


class HGSFrontierQualificationError(HGSFrontierError):
    """Raised before measured work when the frozen environment is not ready."""


class HGSFrontierIntegrityError(HGSFrontierError):
    """Raised when HGS violates the solver-integrity contract."""

    def __init__(self, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


@dataclass(frozen=True, slots=True)
class FrontierResult:
    path: Path
    status: str
    complete: bool
    selected_budget: int | None = None
    completed_rounds: tuple[int, ...] = ()
    remaining_rounds: tuple[int, ...] = ()
    manifest_path: Path | None = None
    manifest_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ReplicationEvidence:
    root: Path
    receipt: dict[str, Any]
    replication_recipe: Any
    base_recipe: Any
    checkpoints: dict[int, dict[str, Any]]
    primary_mode: Any


def _classification() -> dict[str, Any]:
    return {
        "purpose": "software_exploratory",
        "scientific_use": False,
        "confirmatory_eligible": False,
        "whole_system_energy": False,
        "cross_solver_energy_comparable": False,
        "aet_eligible": False,
        "holdout_executed": False,
    }


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


_MISSING = object()


def _attribute(value: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    raise HGSFrontierError(f"frontier recipe lacks required field {names[0]!r}")


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    fields = getattr(value, "__slots__", ())
    if fields:
        return {name: getattr(value, name) for name in fields}
    raise HGSFrontierError(f"cannot serialize {type(value).__name__} as a mapping")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise HGSFrontierError(f"{label} is not a lowercase SHA-256")
    return value


def _relative(root: Path, value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise HGSFrontierError(f"unsafe frontier artifact path: {value!r}")
    return root.joinpath(*path.parts)


def _budgets(recipe: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in _attribute(recipe.candidates, "budgets"))


def _candidate_seeds(recipe: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in _attribute(recipe.candidates, "seeds"))


def _round_schedule(recipe: Any) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    budgets = _budgets(recipe)
    seeds = _candidate_seeds(recipe)
    generator_name = _attribute(recipe.round_schedule, "generator")
    strategy = _attribute(recipe.round_schedule, "strategy")
    unit = _attribute(recipe.round_schedule, "unit")
    if (
        generator_name != "numpy-pcg64"
        or strategy != "randomized_balanced_cyclic_latin_square"
        or unit != "paired_seed_round"
    ):
        raise HGSFrontierError("unsupported frozen HGS round schedule")
    orders = tuple(
        tuple(int(value) for value in row)
        for row in _attribute(recipe.round_schedule, "budget_orders")
    )
    if len(orders) != len(seeds) or any(
        len(order) != len(budgets) or set(order) != set(budgets) for order in orders
    ):
        raise HGSFrontierError("frozen HGS round table is not a complete budget grid")
    position_counts = {
        budget: [
            sum(order[position] == budget for order in orders) for position in range(len(budgets))
        ]
        for budget in budgets
    }
    lower = len(orders) // len(budgets)
    upper = math.ceil(len(orders) / len(budgets))
    if any(count not in {lower, upper} for counts in position_counts.values() for count in counts):
        raise HGSFrontierError("frozen HGS round table is not position balanced")
    return tuple((round_index, seed, orders[round_index]) for round_index, seed in enumerate(seeds))


def _cell_relative(round_index: int, budget: int, seed: int) -> str:
    return f"hgs/round-{round_index:02d}/budget-{budget:03d}-seed-{seed}.json"


def _attestation_sha256(attestation: Mapping[str, Any]) -> str:
    return quality._sha256_bytes(quality._json_bytes(dict(attestation)))


def _neural_relative(seed: int) -> str:
    return f"neural/pomo-50x8-seed-{seed:03d}.json"


def _neural_pairs(recipe: Any) -> tuple[tuple[int, int], ...]:
    prerequisite = recipe.neural_prerequisite
    pairs = zip(
        prerequisite.training_seeds,
        prerequisite.evaluation_seeds,
        strict=True,
    )
    result = tuple((int(training), int(evaluation)) for training, evaluation in pairs)
    if not result:
        raise HGSFrontierError("neural prerequisite has no frozen seed pairs")
    return result


def _neural_evaluation_seed(recipe: Any, training_seed: int) -> int:
    matches = [
        evaluation_seed for seed, evaluation_seed in _neural_pairs(recipe) if seed == training_seed
    ]
    if len(matches) != 1:
        raise HGSFrontierError(
            f"neural prerequisite lacks one evaluation seed for training seed {training_seed}"
        )
    return matches[0]


def _selection_spec(recipe: Any) -> Any:
    return recipe.dataset.selection


def _assert_snapshot_path(root: Path, path: Path) -> str:
    """Return a safe repository-relative path with no symlink component."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise HGSFrontierQualificationError(
            "frontier source snapshot escaped the repository"
        ) from exc
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise HGSFrontierQualificationError(
                f"frontier source snapshot refuses symlink: {relative.as_posix()}"
            )
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HGSFrontierQualificationError(
            f"frontier source snapshot escaped the repository: {relative.as_posix()}"
        ) from exc
    if not path.is_file():
        raise HGSFrontierQualificationError(
            f"frontier source snapshot member is not a file: {relative.as_posix()}"
        )
    return relative.as_posix()


def _source_snapshot(recipe: Any, root: Path) -> dict[str, Any]:
    """Fingerprint exactly the closed source patterns declared by the recipe."""

    config = recipe.source_snapshot
    patterns = tuple(str(value) for value in config.include_patterns)
    if not patterns:
        raise HGSFrontierQualificationError("frontier source snapshot has no patterns")
    relative_paths: set[str] = set()
    for pattern in patterns:
        pure = PurePosixPath(pattern)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or "\\" in pattern
        ):
            raise HGSFrontierQualificationError(f"unsafe frontier source pattern: {pattern!r}")
        matched: list[str] = []
        try:
            candidates = root.glob(pattern)
            for candidate in candidates:
                if candidate.is_file() or candidate.is_symlink():
                    matched.append(_assert_snapshot_path(root, candidate))
        except OSError as exc:
            raise HGSFrontierQualificationError(
                f"cannot expand frontier source pattern {pattern!r}"
            ) from exc
        if not matched:
            raise HGSFrontierQualificationError(
                f"frontier source pattern matched no files: {pattern!r}"
            )
        relative_paths.update(matched)

    if any(
        path == "papers"
        or path.startswith("papers/")
        or path == "experiments"
        or path.startswith("experiments/")
        for path in relative_paths
    ):
        raise HGSFrontierQualificationError(
            "frontier source snapshot unexpectedly includes papers or experiments"
        )
    digest = hashlib.sha256()
    digest.update((str(config.schema_version) + "\0").encode("utf-8"))
    files: list[dict[str, str]] = []
    for relative in sorted(relative_paths):
        path = _relative(root, relative)
        file_sha256 = quality._sha256_file(path)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha256))
        files.append({"path": relative, "sha256": file_sha256})
    return {
        "schema_version": str(config.schema_version),
        "scope": {
            "include_patterns": list(patterns),
            "papers_and_experiment_outputs_excluded": True,
        },
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _selection_quality_recipe(recipe: Any, base_recipe: Any) -> Any:
    dataset = replace(
        base_recipe.dataset,
        problem=_attribute(recipe.dataset, "problem", default=base_recipe.dataset.problem),
        size=int(_attribute(recipe.dataset, "size", default=base_recipe.dataset.size)),
        capacity=float(
            _attribute(recipe.dataset, "capacity", default=base_recipe.dataset.capacity)
        ),
        max_demand=int(
            _attribute(recipe.dataset, "max_demand", default=base_recipe.dataset.max_demand)
        ),
        selection=_selection_spec(recipe),
        holdout=_selection_spec(recipe),
    )
    return replace(base_recipe, dataset=dataset, reference=recipe.reference)


def _crossed_statistics(
    neural: np.ndarray,
    hgs_by_budget: Mapping[int, np.ndarray],
    *,
    replicates: int,
    seed: int,
    quantile: float,
) -> tuple[dict[int, float], dict[int, float]]:
    if neural.ndim != 2 or neural.shape[0] < 1 or neural.shape[1] < 1:
        raise HGSFrontierError("neural gap matrix must be two-dimensional and non-empty")
    if not hgs_by_budget:
        raise HGSFrontierError("HGS gap matrices are empty")
    shapes = {matrix.shape for matrix in hgs_by_budget.values()}
    if len(shapes) != 1:
        raise HGSFrontierError("HGS gap matrices must share one shape")
    hgs_shape = next(iter(shapes))
    if len(hgs_shape) != 2 or hgs_shape[1] != neural.shape[1]:
        raise HGSFrontierError("neural and HGS gap matrices must share the instance axis")
    if not np.isfinite(neural).all() or any(
        not np.isfinite(matrix).all() for matrix in hgs_by_budget.values()
    ):
        raise HGSFrontierError("bootstrap gap matrices contain non-finite values")
    rng = np.random.Generator(np.random.PCG64(seed))
    absolute = {budget: np.empty(replicates, dtype=np.float64) for budget in hgs_by_budget}
    differences = {budget: np.empty(replicates, dtype=np.float64) for budget in hgs_by_budget}
    for replicate in range(replicates):
        hgs_indices = rng.integers(0, hgs_shape[0], size=hgs_shape[0])
        neural_indices = rng.integers(0, neural.shape[0], size=neural.shape[0])
        instance_indices = rng.integers(0, neural.shape[1], size=neural.shape[1])
        neural_mean = float(neural[neural_indices[:, None], instance_indices[None, :]].mean())
        for budget, matrix in hgs_by_budget.items():
            hgs_mean = float(matrix[hgs_indices[:, None], instance_indices[None, :]].mean())
            absolute[budget][replicate] = hgs_mean
            differences[budget][replicate] = neural_mean - hgs_mean
    return (
        {
            budget: float(np.quantile(values, quantile, method="linear"))
            for budget, values in absolute.items()
        },
        {
            budget: float(np.quantile(values, quantile, method="linear"))
            for budget, values in differences.items()
        },
    )


def _checked_hgs_energy(
    tracker: Any,
    *,
    recipe: Any,
    items_processed: int,
) -> dict[str, Any]:
    payload = software._checked_energy(
        tracker,
        minimum_duration_s=float(recipe.measurement.minimum_block_duration_s),
        gpu_index=int(recipe.gpu_index),
        gpu_device_id_sha256=str(
            _attribute(
                recipe,
                "gpu_device_id_sha256",
                default=_attribute(
                    _attribute(recipe, "platform", default={}),
                    "gpu_device_id_sha256",
                    default="",
                ),
            )
        ),
    )
    if payload.get("items_processed") != items_processed:
        raise HGSFrontierError("energy items_processed disagrees with completed HGS work")
    cpu_energy = payload.get("energy_cpu_j")
    if (
        isinstance(cpu_energy, bool)
        or not isinstance(cpu_energy, (int, float))
        or not math.isfinite(float(cpu_energy))
        or float(cpu_energy) <= 0.0
    ):
        raise HGSFrontierError("Windows EMI CPU package energy must be positive")
    payload["primary_estimand"] = recipe.measurement.primary_estimand
    payload["cpu_package_energy_j_per_valid_instance"] = float(cpu_energy) / items_processed
    payload["energy_j_role"] = recipe.measurement.energy_j_role
    payload["gpu_energy_role"] = recipe.measurement.gpu_energy_role
    return payload


def _t_ucb(matrix: np.ndarray, critical: float) -> tuple[list[float], float, float]:
    seed_means = matrix.mean(axis=1)
    mean = float(seed_means.mean())
    standard_deviation = float(seed_means.std(ddof=1))
    upper = mean + critical * standard_deviation / math.sqrt(seed_means.size)
    return seed_means.tolist(), standard_deviation, float(upper)


def _path_hash_pairs(value: Any) -> list[tuple[str, str]]:
    """Return every closed recipe record that binds an artifact path to a hash."""

    pairs: list[tuple[str, str]] = []

    def visit(current: Any) -> None:
        if is_dataclass(current) and not isinstance(current, type):
            current = asdict(current)
        if isinstance(current, Mapping):
            if "path" in current and "sha256" in current:
                path = current["path"]
                digest = current["sha256"]
                if not isinstance(path, str):
                    raise HGSFrontierError("replication source artifact path is invalid")
                pairs.append((path, _sha256(digest, f"replication source {path}")))
            for nested in current.values():
                visit(nested)
        elif isinstance(current, (list, tuple)):
            for nested in current:
                visit(nested)

    visit(value)
    return pairs


def _checkpoint_records(replication_source: Any) -> tuple[Any, ...]:
    records = _attribute(replication_source, "checkpoints", "checkpoint_artifacts")
    if not isinstance(records, (list, tuple)):
        raise HGSFrontierError("replication source checkpoint inventory is invalid")
    return tuple(records)


def _validate_replication_source(recipe: Any, root: Path) -> ReplicationEvidence:
    source = recipe.replication_source
    source_root = _relative(root, str(source.root)).resolve(strict=True)
    try:
        source_root.relative_to(root)
    except ValueError as exc:
        raise HGSFrontierQualificationError(
            "quality-replication source resolves outside the repository"
        ) from exc
    frozen_replication_recipe = source_root / "recipe.yaml"
    frozen_base_recipe = source_root / "base-recipe.yaml"
    try:
        replication_recipe = load_aet_quality_replication_recipe(frozen_replication_recipe)
        base_recipe = load_aet_quality_recipe(frozen_base_recipe)
        state = quality._load_json(source_root / "run-state.json")
        validated = replication._validate_completed_bundle(
            replication_recipe,
            base_recipe,
            output=source_root,
            state=state,
        )
    except (
        OSError,
        ValueError,
        quality.QualityPilotError,
        replication.QualityReplicationError,
    ) as exc:
        raise HGSFrontierQualificationError(
            "sealed quality-replication bundle failed in-place validation"
        ) from exc

    expected_status = str(_attribute(source, "status", "expected_status"))
    if validated.status != expected_status or expected_status != "complete_replication_passed":
        raise HGSFrontierQualificationError(
            "quality-replication source did not pass its primary gate"
        )
    declared_pairs = _path_hash_pairs(source)
    if not declared_pairs:
        raise HGSFrontierQualificationError(
            "quality-replication source declares no hashed artifacts"
        )
    for relative, expected in declared_pairs:
        path = _relative(source_root, relative)
        if not quality._file_matches_sha256(path, expected):
            raise HGSFrontierQualificationError(
                f"quality-replication source artifact changed: {relative}"
            )

    expected_git_sha = str(_attribute(source, "source_git_sha", "git_sha"))
    if state.get("git_sha") != expected_git_sha:
        raise HGSFrontierQualificationError("quality-replication source Git commit changed")
    expected_source_sha = str(_attribute(source, "source_snapshot_sha256", default=""))
    if not expected_source_sha:
        source_snapshot = _attribute(source, "source_snapshot", default={})
        expected_source_sha = str(_attribute(source_snapshot, "sha256", default=""))
    if (
        expected_source_sha
        and state.get("source_snapshot", {}).get("sha256") != expected_source_sha
    ):
        raise HGSFrontierQualificationError("quality-replication source snapshot changed")

    manifest = quality._load_json(source_root / "manifest.json")
    assessment = quality._load_json(source_root / "replication-assessment.json")
    if (
        manifest.get("status") != expected_status
        or assessment.get("status") != expected_status
        or assessment.get("roles", {}).get("primary", {}).get("classification") != "pass"
    ):
        raise HGSFrontierQualificationError(
            "quality-replication source semantics no longer authorize a frontier"
        )

    checkpoint_epoch = int(_attribute(source, "checkpoint_epoch"))
    primary_mode_id = str(_attribute(source, "mode", "mode_id", "primary_mode_id"))
    prerequisite = recipe.neural_prerequisite
    if checkpoint_epoch != int(prerequisite.checkpoint_epoch) or primary_mode_id != str(
        prerequisite.mode_id
    ):
        raise HGSFrontierQualificationError(
            "quality-replication source policy is not frozen to POMO-50x8 epoch 40"
        )
    modes = [
        mode for mode in replication_recipe.evaluation.modes if mode.mode_id == primary_mode_id
    ]
    if len(modes) != 1:
        raise HGSFrontierQualificationError(
            "quality-replication source lacks the frozen primary policy"
        )

    checkpoints: dict[int, dict[str, Any]] = {}
    for record in _checkpoint_records(source):
        seed = int(_attribute(record, "seed", "training_seed"))
        relative = str(_attribute(record, "path"))
        digest = _sha256(_attribute(record, "sha256"), f"seed {seed} checkpoint")
        path = _relative(source_root, relative)
        if not quality._file_matches_sha256(path, digest):
            raise HGSFrontierQualificationError(
                f"quality-replication checkpoint changed for seed {seed}"
            )
        checkpoints[seed] = {
            "seed": seed,
            "epoch": checkpoint_epoch,
            "path": relative,
            "sha256": digest,
        }
    if tuple(sorted(checkpoints)) != tuple(prerequisite.training_seeds):
        raise HGSFrontierQualificationError(
            "quality-replication source must provide checkpoints for seeds 2 through 6"
        )

    receipt = {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "validation": "in_place_deep_semantic_and_sha256",
        "copied_into_frontier": False,
        "root": str(source.root),
        "status": expected_status,
        "git_sha": expected_git_sha,
        "source_snapshot_sha256": state.get("source_snapshot", {}).get("sha256"),
        "manifest_sha256": quality._sha256_file(source_root / "manifest.json"),
        "checksums_sha256": quality._sha256_file(source_root / "SHA256SUMS"),
        "assessment_sha256": quality._sha256_file(source_root / "replication-assessment.json"),
        "run_state_sha256": quality._sha256_file(source_root / "run-state.json"),
        "reference_lock_sha256": quality._sha256_file(
            source_root / "reference" / "reference-lock.json"
        ),
        "checkpoint_epoch": checkpoint_epoch,
        "mode_id": primary_mode_id,
        "checkpoints": [checkpoints[seed] for seed in sorted(checkpoints)],
        "declared_artifacts": [
            {"path": relative, "sha256": digest} for relative, digest in declared_pairs
        ],
    }
    return ReplicationEvidence(
        root=source_root,
        receipt=receipt,
        replication_recipe=replication_recipe,
        base_recipe=base_recipe,
        checkpoints=checkpoints,
        primary_mode=modes[0],
    )


def _source_receipt_path(output: Path) -> Path:
    return output / "provenance" / "quality-replication-receipt.json"


def _prepare_output(
    *,
    recipe: Any,
    recipe_bytes: bytes,
    root: Path,
    evidence: ReplicationEvidence,
    runtime_identity: dict[str, Any],
    runtime_controls: dict[str, Any],
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    output = quality._safe_output_target(root, str(recipe.output_root))
    recipe_sha256 = quality._sha256_bytes(recipe_bytes)
    try:
        lock_bytes = (root / "uv.lock").read_bytes()
    except OSError as exc:
        raise HGSFrontierQualificationError("uv.lock is unavailable") from exc
    lock_sha256 = quality._sha256_bytes(lock_bytes)
    git = quality._git_snapshot(root)
    source_snapshot = _source_snapshot(recipe, root)
    schedule = [
        {"round": index, "seed": seed, "budget_order": list(order)}
        for index, seed, order in _round_schedule(recipe)
    ]
    receipt_sha256 = quality._sha256_bytes(quality._json_bytes(evidence.receipt))

    if output.exists():
        if not resume:
            raise HGSFrontierError(f"output already exists; use --resume: {output}")
        state = quality._load_json(output / "run-state.json")
        if state.get("status") == INVALIDATED_STATUS:
            raise HGSFrontierError(
                "frontier output was invalidated by an HGS integrity failure; "
                "diagnose it and use a new output_root"
            )
        expected = {
            "schema_version": RUN_STATE_SCHEMA,
            "recipe_sha256": recipe_sha256,
            "uv_lock_sha256": lock_sha256,
            "classification": _classification(),
            "round_schedule": schedule,
            "replication_source_receipt_sha256": receipt_sha256,
        }
        if any(not _strict_json_equal(state.get(key), value) for key, value in expected.items()):
            raise HGSFrontierQualificationError(
                "existing frontier was created from different frozen inputs"
            )
        if state.get("git_sha") != git.get("sha"):
            raise HGSFrontierQualificationError("frontier Git commit changed since initialization")
        if state.get("source_snapshot", {}).get("sha256") != source_snapshot.get("sha256"):
            raise HGSFrontierQualificationError(
                "frontier source snapshot changed since initialization"
            )
        if not _strict_json_equal(state.get("runtime_identity"), runtime_identity):
            raise HGSFrontierQualificationError(
                "frontier runtime identity changed since initialization"
            )
        if not _strict_json_equal(state.get("runtime_controls"), runtime_controls):
            raise HGSFrontierQualificationError(
                "frontier runtime controls changed since initialization"
            )
        frozen = (
            (output / "recipe.yaml", recipe_sha256),
            (output / "environment" / "uv.lock", lock_sha256),
            (_source_receipt_path(output), receipt_sha256),
        )
        for path, digest in frozen:
            if not quality._file_matches_sha256(path, digest):
                raise HGSFrontierQualificationError(
                    f"frozen frontier input changed or disappeared: {path}"
                )
        if not _strict_json_equal(
            quality._load_json(_source_receipt_path(output)), evidence.receipt
        ):
            raise HGSFrontierQualificationError("quality-replication source receipt changed")
        if state.get("status") not in COMPLETE_STATUSES:
            quality._record_partial_cleanup(output, state)
        return output, state

    state: dict[str, Any] = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": "initialized",
        "created_at": datetime.now(UTC).isoformat(),
        "recipe_sha256": recipe_sha256,
        "uv_lock_sha256": lock_sha256,
        "git_sha": git.get("sha"),
        "git": git,
        "source_snapshot": source_snapshot,
        "runtime_identity": runtime_identity,
        "runtime_controls": runtime_controls,
        "classification": _classification(),
        "round_schedule": schedule,
        "replication_source_receipt_sha256": receipt_sha256,
        "reference_lock_sha256": None,
        "reference_locked_before_candidate_evaluation": False,
        "neural_completed_seeds": [],
        "neural_quality_gate_sha256": None,
        "completed_cells": [],
        "completed_rounds": [],
        "remaining_rounds": list(range(len(_candidate_seeds(recipe)))),
        "invocations": [],
        "resume_cleanups": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    if staging.exists():
        raise HGSFrontierError("frontier initialization staging path already exists")
    try:
        (staging / "environment").mkdir(parents=True)
        (staging / "provenance").mkdir(parents=True)
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "environment" / "uv.lock").write_bytes(lock_bytes)
        quality._atomic_write_json(
            staging / "provenance" / "quality-replication-receipt.json",
            evidence.receipt,
        )
        quality._atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _prepare_corpus_and_reference(
    recipe: Any,
    quality_recipe: Any,
    output: Path,
    state: dict[str, Any],
) -> tuple[quality.Corpus, dict[str, Any], dict[str, Any]]:
    spec = _selection_spec(recipe)
    corpus_path = _relative(output, str(spec.artifact))
    if corpus_path.exists():
        corpus = quality._load_corpus(quality_recipe, "selection", spec, corpus_path)
    else:
        corpus = quality._generate_corpus(quality_recipe, "selection", spec, corpus_path)

    lock_path = output / "reference" / "reference-lock.json"
    reference_path = output / "reference" / "selection" / "reference.json"
    anchored = state.get("reference_lock_sha256")
    if anchored is None:
        if (output / "neural").exists() or (output / "hgs").exists():
            raise HGSFrontierError("candidate artifacts exist without a frozen selection reference")
        reference = quality._build_reference(quality_recipe, corpus, output)
        candidates = reference.get("candidate_artifacts")
        if not isinstance(candidates, list):
            raise HGSFrontierError("selection reference candidate inventory is invalid")
        lock = {
            "schema_version": REFERENCE_LOCK_SCHEMA,
            "status": "locked",
            "locked_at": datetime.now(UTC).isoformat(),
            "locked_before_candidate_evaluation": True,
            "policy": recipe.reference.policy,
            "dataset": {
                "path": corpus.path.relative_to(output).as_posix(),
                "file_sha256": corpus.file_sha256,
                "content_sha256": corpus.content_sha256,
            },
            "reference": {
                "path": "reference/selection/reference.json",
                "sha256": quality._sha256_file(reference_path),
            },
            "candidate_artifacts": candidates,
        }
        quality._atomic_write_json(lock_path, lock)
        state["reference_lock_sha256"] = quality._sha256_file(lock_path)
        state["reference_locked_before_candidate_evaluation"] = True
        quality._atomic_write_json(output / "run-state.json", state)
        return corpus, reference, lock

    if not isinstance(anchored, str) or not quality._file_matches_sha256(lock_path, anchored):
        raise HGSFrontierError("anchored selection reference lock changed")
    lock = quality._load_json(lock_path)
    dataset_entry = lock.get("dataset")
    reference_entry = lock.get("reference")
    if (
        lock.get("schema_version") != REFERENCE_LOCK_SCHEMA
        or lock.get("status") != "locked"
        or lock.get("locked_before_candidate_evaluation") is not True
        or lock.get("policy") != recipe.reference.policy
        or not isinstance(dataset_entry, dict)
        or dataset_entry.get("path") != corpus.path.relative_to(output).as_posix()
        or dataset_entry.get("file_sha256") != corpus.file_sha256
        or dataset_entry.get("content_sha256") != corpus.content_sha256
        or not isinstance(reference_entry, dict)
        or reference_entry.get("path") != "reference/selection/reference.json"
        or not quality._file_matches_sha256(reference_path, str(reference_entry.get("sha256", "")))
    ):
        raise HGSFrontierError("anchored selection reference metadata changed")
    reference = quality._build_reference(quality_recipe, corpus, output)
    if quality._sha256_file(reference_path) != reference_entry["sha256"]:
        raise HGSFrontierError("anchored selection reference semantics changed")
    if not _strict_json_equal(
        lock.get("candidate_artifacts"), reference.get("candidate_artifacts")
    ):
        raise HGSFrontierError("anchored reference candidate inventory changed")
    return corpus, reference, lock


def _validate_neural_evaluation(
    payload: dict[str, Any],
    *,
    recipe: Any,
    output: Path,
    evidence: ReplicationEvidence,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    seed: int,
) -> dict[str, Any]:
    checkpoint = evidence.checkpoints[seed]
    expected = {
        "schema_version": NEURAL_EVALUATION_SCHEMA,
        "classification": {
            "purpose": "frontier_quality_control",
            "energy_measurement": "none",
            "included_in_hgs_energy_frontier": False,
        },
        "training_seed": seed,
        "checkpoint_epoch": int(recipe.neural_prerequisite.checkpoint_epoch),
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "mode": _mapping(evidence.primary_mode),
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "source_receipt_sha256": quality._sha256_file(_source_receipt_path(output)),
        "evaluation_seed": _neural_evaluation_seed(recipe, seed),
    }
    if any(not _strict_json_equal(payload.get(key), value) for key, value in expected.items()):
        raise HGSFrontierError(f"cached neural selection evaluation changed for seed {seed}")
    validation = quality.validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        payload.get("routes", ()),
    )
    if not _strict_json_equal(payload.get("validation"), validation):
        raise HGSFrontierError(f"stored neural route validation changed for seed {seed}")
    recomputed = quality._gap_summary(validation["costs"], reference["costs"])
    if not _strict_json_equal(payload.get("quality"), recomputed):
        raise HGSFrontierError(f"stored neural quality changed for seed {seed}")
    return payload


def _prepare_neural_evaluations(
    recipe: Any,
    output: Path,
    state: dict[str, Any],
    evidence: ReplicationEvidence,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[int, dict[str, Any]]:
    evaluations: dict[int, dict[str, Any]] = {}
    completed = state.get("neural_completed_seeds")
    if not isinstance(completed, list):
        raise HGSFrontierError("neural completion state is invalid")
    for seed, evaluation_seed in _neural_pairs(recipe):
        path = output / _neural_relative(seed)
        if path.exists():
            payload = quality._load_json(path)
        else:
            checkpoint = evidence.checkpoints[seed]
            checkpoint_path = _relative(evidence.root, checkpoint["path"])
            seed_recipe = replication._seed_recipe(
                evidence.base_recipe,
                evidence.replication_recipe,
                seed,
            )
            routes = quality._evaluate_routes(
                seed_recipe,
                checkpoint_path,
                corpus,
                evidence.primary_mode,
                eval_seed=evaluation_seed,
            )
            validation = quality.validate_routes(
                corpus.coords,
                corpus.demands,
                corpus.capacity,
                routes,
            )
            payload = {
                "schema_version": NEURAL_EVALUATION_SCHEMA,
                "classification": {
                    "purpose": "frontier_quality_control",
                    "energy_measurement": "none",
                    "included_in_hgs_energy_frontier": False,
                },
                "training_seed": seed,
                "checkpoint_epoch": int(recipe.neural_prerequisite.checkpoint_epoch),
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "mode": _mapping(evidence.primary_mode),
                "dataset_content_sha256": corpus.content_sha256,
                "reference_sha256": reference_sha256,
                "source_receipt_sha256": quality._sha256_file(_source_receipt_path(output)),
                "evaluation_seed": evaluation_seed,
                "routes": routes,
                "validation": validation,
                "quality": quality._gap_summary(validation["costs"], reference["costs"]),
            }
            quality._atomic_write_json(path, payload)
        evaluations[seed] = _validate_neural_evaluation(
            payload,
            recipe=recipe,
            output=output,
            evidence=evidence,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            seed=seed,
        )
        if seed not in completed:
            completed.append(seed)
            completed.sort()
            quality._atomic_write_json(output / "run-state.json", state)
    if tuple(completed) != tuple(recipe.neural_prerequisite.training_seeds):
        raise HGSFrontierError("all five frozen neural checkpoints must be evaluated first")
    return evaluations


def _neural_quality_gate(
    recipe: Any,
    evaluations: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    seeds = tuple(sorted(evaluations))
    prerequisite = recipe.neural_prerequisite
    if seeds != tuple(prerequisite.training_seeds):
        raise HGSFrontierError("neural gate requires frozen training seeds 2 through 6")
    matrix = np.asarray(
        [evaluations[seed]["quality"]["gaps_pct"] for seed in seeds],
        dtype=np.float64,
    )
    invalid = sum(int(evaluations[seed]["validation"]["invalid_instance_count"]) for seed in seeds)
    summaries = replication._summarize_gap_matrices(
        {prerequisite.mode_id: matrix},
        invalid_instances={prerequisite.mode_id: invalid},
        threshold_pct=float(prerequisite.maximum_mean_gap_pct),
        bootstrap_replicates=int(prerequisite.bootstrap_replicates),
        bootstrap_seed=int(prerequisite.bootstrap_seed),
        bootstrap_quantile=float(prerequisite.bootstrap_confidence_level),
        t_critical=float(prerequisite.seed_t_critical_value),
    )
    summary = summaries[prerequisite.mode_id]
    passed = summary["classification"] == "pass"
    return {
        "schema_version": NEURAL_GATE_SCHEMA,
        "status": "pass" if passed else "nonpass",
        "passed": passed,
        "mode_id": prerequisite.mode_id,
        "checkpoint_epoch": int(prerequisite.checkpoint_epoch),
        "training_seeds": list(seeds),
        "evaluation_seeds": list(prerequisite.evaluation_seeds),
        "summary": summary,
        "method": {
            "metric": prerequisite.metric,
            "comparison": prerequisite.comparison,
            "threshold_pct": float(prerequisite.maximum_mean_gap_pct),
            "require_each_seed_below_threshold": bool(
                prerequisite.require_each_seed_below_threshold
            ),
            "require_mean_below_threshold": True,
            "require_t_ucb_below_threshold": bool(prerequisite.require_t_ucb_below_threshold),
            "require_bootstrap_ucb_below_threshold": bool(
                prerequisite.require_bootstrap_ucb_below_threshold
            ),
            "t_ucb": "one-sided Student t upper confidence bound across five training seeds",
            "t_degrees_of_freedom": int(prerequisite.seed_t_degrees_of_freedom),
            "t_critical": float(prerequisite.seed_t_critical_value),
            "crossed_bootstrap": {
                "method": prerequisite.bootstrap_method,
                "generator": prerequisite.bootstrap_generator,
                "seed": int(prerequisite.bootstrap_seed),
                "replicates": int(prerequisite.bootstrap_replicates),
                "one_sided_quantile": float(prerequisite.bootstrap_confidence_level),
                "resampled_axes": ["training_seed", "instance"],
            },
        },
        "energy_measured": False,
        "required_before_hgs_measurement": bool(prerequisite.require_pass_before_hgs_measurement),
        "stop_without_hgs_on_nonpass": bool(prerequisite.stop_without_hgs_on_nonpass),
    }


def _prepare_neural_quality_gate(
    recipe: Any,
    output: Path,
    state: dict[str, Any],
    evaluations: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    expected = _neural_quality_gate(recipe, evaluations)
    path = output / "neural-quality-gate.json"
    anchored = state.get("neural_quality_gate_sha256")
    if path.exists():
        actual = quality._load_json(path)
        if not _strict_json_equal(actual, expected):
            raise HGSFrontierError("cached neural quality gate changed")
    else:
        if anchored is not None or (output / "hgs").exists():
            raise HGSFrontierError("HGS artifacts exist without the neural gate")
        quality._atomic_write_json(path, expected)
        actual = expected
    digest = quality._sha256_file(path)
    if anchored is None:
        state["neural_quality_gate_sha256"] = digest
        quality._atomic_write_json(output / "run-state.json", state)
    elif anchored != digest:
        raise HGSFrontierError("anchored neural quality gate changed")
    if actual["passed"] is not True and (output / "hgs").exists():
        raise HGSFrontierError("HGS artifacts exist after a non-passing neural gate")
    return actual


def _validate_solver_results(
    results: Sequence[Any],
    *,
    corpus: quality.Corpus,
    seed: int,
    budget: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if len(results) != int(corpus.coords.shape[0]):
        raise HGSFrontierError("HGS did not return one result per selection instance")
    routes: list[tuple[tuple[int, ...], ...]] = []
    for instance, result in enumerate(results):
        if (
            int(_attribute(result, "instance_index")) != instance
            or int(_attribute(result, "seed")) != seed + instance
            or int(_attribute(result, "max_iterations")) != budget
            or int(_attribute(result, "scaling_factor"))
            != int(_attribute(corpus, "scaling_factor", default=1_000_000))
        ):
            raise HGSFrontierError("HGS result metadata disagrees with budget or seed")
        integer_cost = _attribute(result, "integer_cost")
        reported_cost = _attribute(result, "cost")
        if (
            isinstance(integer_cost, bool)
            or not isinstance(integer_cost, int)
            or isinstance(reported_cost, bool)
            or not isinstance(reported_cost, (int, float))
            or not math.isfinite(float(reported_cost))
            or not math.isclose(
                float(reported_cost),
                integer_cost / int(_attribute(result, "scaling_factor")),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise HGSFrontierError("HGS reported objective metadata is inconsistent")
        raw_routes = _attribute(result, "routes")
        routes.append(tuple(tuple(int(customer) for customer in route) for route in raw_routes))
    return tuple(routes)


class _StoredReading:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._payload, allow_nan=False))


class _StoredTracker:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.reading = _StoredReading(payload)


def _validate_energy_payload(
    payload: dict[str, Any],
    *,
    recipe: Any,
    items_processed: int,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    base = json.loads(json.dumps(payload, allow_nan=False))
    audit = base.pop("gpu_process_audit", None)
    for field in (
        "primary_estimand",
        "cpu_package_energy_j_per_valid_instance",
        "energy_j_role",
        "gpu_energy_role",
    ):
        base.pop(field, None)
    expected = _checked_hgs_energy(
        _StoredTracker(base),
        recipe=recipe,
        items_processed=items_processed,
    )
    expected["gpu_process_audit"] = software._gpu_process_audit(before, after)
    if audit is None or not _strict_json_equal(payload, expected):
        raise HGSFrontierError("stored HGS energy contract changed")


def _solver_result_records(results: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "instance_index": int(_attribute(result, "instance_index")),
            "effective_seed": int(_attribute(result, "seed")),
            "integer_cost": int(_attribute(result, "integer_cost")),
            "reported_cost": float(_attribute(result, "cost")),
            "max_iterations": int(_attribute(result, "max_iterations")),
            "scaling_factor": int(_attribute(result, "scaling_factor")),
        }
        for result in results
    ]


def _execute_hgs_cell(
    recipe: Any,
    *,
    output: Path,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    round_index: int,
    order_index: int,
    seed: int,
    budget: int,
    invocation_binding: Mapping[str, Any],
) -> dict[str, Any]:
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

    def solve_checked(
        coords: Any,
        demands: Any,
        *,
        phase: str,
    ) -> Sequence[Any]:
        try:
            return solve_corpus_sequential(
                coords,
                demands,
                corpus.capacity,
                seed=seed,
                max_iterations=budget,
                scaling_factor=int(recipe.candidates.scaling_factor),
                collect_stats=bool(recipe.candidates.collect_stats),
            )
        except RuntimeError as exc:
            match = _PYVRP_INCOMPLETE_SOLUTION.fullmatch(str(exc))
            if match is None:
                raise
            raise HGSFrontierIntegrityError(
                "PyVRP did not return a complete feasible HGS solution; "
                "the frontier output is invalidated",
                details={
                    "round": round_index,
                    "budget": budget,
                    "seed": seed,
                    "phase": phase,
                    "instance_index": int(match.group(1)),
                    "pyvrp_error": str(exc),
                },
            ) from exc

    def validate_results_checked(
        results: Sequence[Any],
        checked_corpus: quality.Corpus,
        *,
        phase: str,
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        try:
            return _validate_solver_results(
                results,
                corpus=checked_corpus,
                seed=seed,
                budget=budget,
            )
        except (HGSFrontierError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, HGSFrontierIntegrityError):
                raise
            raise HGSFrontierIntegrityError(
                "HGS result metadata, cardinality, routes, or cost violated its contract; "
                "the frontier output is invalidated",
                details={
                    "round": round_index,
                    "budget": budget,
                    "seed": seed,
                    "phase": phase,
                    "validation_error_type": type(exc).__name__,
                    "validation_error": str(exc),
                },
            ) from exc

    warmup_count = int(recipe.candidates.warmup_instances)
    warmup = solve_checked(
        corpus.coords[:warmup_count],
        corpus.demands[:warmup_count],
        phase="warmup",
    )
    warmup_corpus = replace(
        corpus,
        coords=corpus.coords[:warmup_count],
        demands=corpus.demands[:warmup_count],
    )
    validate_results_checked(
        warmup,
        warmup_corpus,
        phase="warmup_result_contract",
    )
    items_per_cycle = int(corpus.coords.shape[0])
    before = software._gpu_process_snapshot(int(recipe.gpu_index))
    tracker = software._make_tracker(
        f"hgs-frontier-r{round_index:02d}-b{budget}-s{seed}",
        int(recipe.gpu_index),
    )
    cycles = 0
    first_results: Sequence[Any] | None = None
    last_results: Sequence[Any] | None = None
    started_at = datetime.now(UTC)
    with tracker as active_tracker:
        measured_started = time.perf_counter()
        while True:
            current_results = solve_checked(
                corpus.coords,
                corpus.demands,
                phase="measured_block",
            )
            if first_results is None:
                first_results = current_results
            last_results = current_results
            cycles += 1
            if time.perf_counter() - measured_started >= float(
                recipe.measurement.minimum_block_duration_s
            ):
                break
        active_tracker.n_items = cycles * items_per_cycle
    ended_at = datetime.now(UTC)
    after = software._gpu_process_snapshot(int(recipe.gpu_index))
    items_processed = cycles * items_per_cycle
    energy = _checked_hgs_energy(
        tracker,
        recipe=recipe,
        items_processed=items_processed,
    )
    energy["gpu_process_audit"] = software._gpu_process_audit(before, after)
    if first_results is None or last_results is None:
        raise HGSFrontierError("HGS measured block produced no complete corpus cycle")
    routes = validate_results_checked(
        first_results,
        corpus,
        phase="first_measured_cycle_result_contract",
    )
    validation = quality.validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        routes,
    )
    last_routes = validate_results_checked(
        last_results,
        corpus,
        phase="last_measured_cycle_result_contract",
    )
    last_validation = quality.validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        last_routes,
    )
    if validation["complete"] is not True or last_validation["complete"] is not True:
        raise HGSFrontierIntegrityError(
            "HGS returned invalid CVRP routes; the frontier output is invalidated",
            details={
                "round": round_index,
                "budget": budget,
                "seed": seed,
                "first_cycle_validation": validation,
                "last_cycle_validation": last_validation,
            },
        )
    solver_records = _solver_result_records(first_results)
    last_records = _solver_result_records(last_results)
    first_output_sha256 = quality._sha256_bytes(
        quality._json_bytes(
            {
                "routes": routes,
                "solver_results": solver_records,
                "validation": validation,
            }
        )
    )
    last_output_sha256 = quality._sha256_bytes(
        quality._json_bytes(
            {
                "routes": last_routes,
                "solver_results": last_records,
                "validation": last_validation,
            }
        )
    )
    if last_output_sha256 != first_output_sha256:
        raise HGSFrontierIntegrityError(
            "first and last measured HGS cycles produced different routes or costs",
            details={
                "round": round_index,
                "budget": budget,
                "seed": seed,
                "first_measured_cycle_output_sha256": first_output_sha256,
                "last_measured_cycle_output_sha256": last_output_sha256,
            },
        )
    quality_summary = quality._gap_summary(validation["costs"], reference["costs"])
    return {
        "schema_version": HGS_CELL_SCHEMA,
        "classification": {
            "purpose": "hgs_frontier_selection",
            "scientific_use": False,
            "confirmatory_eligible": False,
            "cross_solver_energy_comparable": False,
            "aet_eligible": False,
        },
        "status": "complete",
        "round": round_index,
        "order_within_round": order_index,
        "base_seed": seed,
        "budget": budget,
        "budget_kind": recipe.candidates.budget_kind,
        "solver": recipe.candidates.solver,
        "scaling_factor": int(recipe.candidates.scaling_factor),
        "collect_stats": bool(recipe.candidates.collect_stats),
        "warmup_instances": warmup_count,
        "warmup_included_in_measurement": False,
        "validation_included_in_measurement": False,
        "serialization_included_in_measurement": False,
        "cycle_integrity_hashing_included_in_measurement": False,
        "measured_cycle_output_retention": "first_and_last_cycles_constant_memory",
        "intermediate_measured_cycle_outputs_retained": False,
        "intermediate_measured_cycle_outputs_verified": False,
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "recipe_sha256": quality._sha256_file(output / "recipe.yaml"),
        "source_receipt_sha256": quality._sha256_file(_source_receipt_path(output)),
        "invocation_number": invocation_binding["invocation_number"],
        "preflight_evidence_sha256": invocation_binding["preflight_evidence_sha256"],
        "exclusive_use_attestation_sha256": invocation_binding["exclusive_use_attestation_sha256"],
        "exclusive_use_attested_at": invocation_binding["exclusive_use_attested_at"],
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "cycles": cycles,
        "first_measured_cycle_output_sha256": first_output_sha256,
        "last_measured_cycle_output_sha256": last_output_sha256,
        "first_and_last_measured_outputs_identical": True,
        "measured_instances_attempted": items_processed,
        "items_processed": items_processed,
        "valid_instances_processed": items_processed,
        "launch_gpu_process_snapshot": before,
        "final_gpu_process_snapshot": after,
        "energy": energy,
        "routes": routes,
        "solver_results": solver_records,
        "validation": validation,
        "quality": quality_summary,
    }


def _validate_hgs_cell(
    payload: dict[str, Any],
    *,
    recipe: Any,
    output: Path,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    round_index: int,
    order_index: int,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    expected = {
        "schema_version": HGS_CELL_SCHEMA,
        "classification": {
            "purpose": "hgs_frontier_selection",
            "scientific_use": False,
            "confirmatory_eligible": False,
            "cross_solver_energy_comparable": False,
            "aet_eligible": False,
        },
        "status": "complete",
        "round": round_index,
        "order_within_round": order_index,
        "base_seed": seed,
        "budget": budget,
        "budget_kind": recipe.candidates.budget_kind,
        "solver": recipe.candidates.solver,
        "scaling_factor": int(recipe.candidates.scaling_factor),
        "collect_stats": bool(recipe.candidates.collect_stats),
        "warmup_instances": int(recipe.candidates.warmup_instances),
        "warmup_included_in_measurement": False,
        "validation_included_in_measurement": False,
        "serialization_included_in_measurement": False,
        "cycle_integrity_hashing_included_in_measurement": False,
        "measured_cycle_output_retention": "first_and_last_cycles_constant_memory",
        "intermediate_measured_cycle_outputs_retained": False,
        "intermediate_measured_cycle_outputs_verified": False,
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "recipe_sha256": quality._sha256_file(output / "recipe.yaml"),
        "source_receipt_sha256": quality._sha256_file(_source_receipt_path(output)),
    }
    if any(not _strict_json_equal(payload.get(field), value) for field, value in expected.items()):
        raise HGSFrontierError(f"cached HGS cell changed for round {round_index}, budget {budget}")
    invocation_number = payload.get("invocation_number")
    attested_at = payload.get("exclusive_use_attested_at")
    if (
        isinstance(invocation_number, bool)
        or not isinstance(invocation_number, int)
        or invocation_number < 1
        or not isinstance(attested_at, str)
    ):
        raise HGSFrontierError("cached HGS invocation binding changed")
    for field in (
        "preflight_evidence_sha256",
        "exclusive_use_attestation_sha256",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
            raise HGSFrontierError("cached HGS invocation binding changed")
    cycles = payload.get("cycles")
    items = payload.get("items_processed")
    measured_attempted = payload.get("measured_instances_attempted")
    expected_items_per_cycle = int(corpus.coords.shape[0])
    if (
        isinstance(cycles, bool)
        or not isinstance(cycles, int)
        or cycles < 1
        or isinstance(items, bool)
        or not isinstance(items, int)
        or isinstance(measured_attempted, bool)
        or not isinstance(measured_attempted, int)
        or measured_attempted != cycles * expected_items_per_cycle
        or items != measured_attempted
        or payload.get("valid_instances_processed") != items
    ):
        raise HGSFrontierError("cached HGS work-unit accounting changed")
    first_sha256 = payload.get("first_measured_cycle_output_sha256")
    last_sha256 = payload.get("last_measured_cycle_output_sha256")
    if (
        not isinstance(first_sha256, str)
        or _HEX_SHA256.fullmatch(first_sha256) is None
        or not isinstance(last_sha256, str)
        or last_sha256 != first_sha256
        or payload.get("first_and_last_measured_outputs_identical") is not True
    ):
        raise HGSFrontierError("cached HGS measured-endpoint evidence changed")
    validation = quality.validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        payload.get("routes", ()),
    )
    quality._assert_valid(validation, f"cached HGS budget {budget} seed {seed}")
    if not _strict_json_equal(payload.get("validation"), validation):
        raise HGSFrontierError("stored HGS route validation changed")
    quality_summary = quality._gap_summary(validation["costs"], reference["costs"])
    if not _strict_json_equal(payload.get("quality"), quality_summary):
        raise HGSFrontierError("stored HGS quality metrics changed")
    records = payload.get("solver_results")
    if not isinstance(records, list) or len(records) != expected_items_per_cycle:
        raise HGSFrontierError("stored HGS solver-result inventory changed")
    for instance, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("instance_index") != instance
            or record.get("effective_seed") != seed + instance
            or record.get("max_iterations") != budget
            or record.get("scaling_factor") != int(recipe.candidates.scaling_factor)
        ):
            raise HGSFrontierError("stored HGS solver-result metadata changed")
    recomputed_first_sha256 = quality._sha256_bytes(
        quality._json_bytes(
            {
                "routes": payload.get("routes"),
                "solver_results": records,
                "validation": validation,
            }
        )
    )
    if first_sha256 != recomputed_first_sha256:
        raise HGSFrontierError("stored first measured HGS cycle hash changed")
    before = payload.get("launch_gpu_process_snapshot")
    after = payload.get("final_gpu_process_snapshot")
    energy = payload.get("energy")
    if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(energy, dict):
        raise HGSFrontierError("stored HGS measurement metadata changed")
    _validate_energy_payload(
        energy,
        recipe=recipe,
        items_processed=items,
        before=before,
        after=after,
    )
    return payload


def _build_assessment(
    recipe: Any,
    neural_evaluations: Mapping[int, dict[str, Any]],
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    budgets = _budgets(recipe)
    seeds = _candidate_seeds(recipe)
    schedule = _round_schedule(recipe)
    if tuple(sorted(neural_evaluations)) != tuple(recipe.neural_prerequisite.training_seeds):
        raise HGSFrontierError("frontier assessment requires five neural evaluations")
    neural_gate = _neural_quality_gate(recipe, neural_evaluations)
    if neural_gate["passed"] is not True:
        if cells:
            raise HGSFrontierError(
                "HGS cells may not exist after a non-passing neural prerequisite"
            )
        return {
            "schema_version": ASSESSMENT_SCHEMA,
            "status": NEURAL_NONPASS_STATUS,
            "classification": _classification(),
            "selection_split": "selection",
            "holdout_executed": False,
            "all_budgets_evaluated": False,
            "budget_order": list(budgets),
            "candidate_seeds": list(seeds),
            "neural_policy": neural_gate,
            "quality_method": {
                "metric": recipe.gate.metric,
                "comparison": recipe.gate.comparison,
                "threshold_pct": float(recipe.gate.maximum_mean_gap_pct),
                "bootstrap": _mapping(recipe.bootstrap),
            },
            "budget_points": {},
            "passing_budgets": [],
            "selected_budget": None,
            "selection_rule": recipe.selection.rule,
            "selection_stopped_before_hgs": True,
            "stop_reason": "frozen_neural_policy_failed_preregistered_quality_gate",
            "energy_used_for_selection": False,
            "primary_energy_estimand": recipe.measurement.primary_estimand,
            "cross_solver_energy_comparable": False,
            "aet_was_computed": False,
        }
    expected_cells = {
        (round_index, budget) for round_index, _seed, order in schedule for budget in order
    }
    if set(cells) != expected_cells:
        raise HGSFrontierError("frontier assessment requires the complete budget-seed grid")

    neural = np.asarray(
        [neural_evaluations[seed]["quality"]["gaps_pct"] for seed in sorted(neural_evaluations)],
        dtype=np.float64,
    )
    hgs_matrices: dict[int, np.ndarray] = {}
    for budget in budgets:
        hgs_matrices[budget] = np.asarray(
            [
                cells[(round_index, budget)]["quality"]["gaps_pct"]
                for round_index in range(len(seeds))
            ],
            dtype=np.float64,
        )
    bootstrap_ucb, neural_minus_hgs_ucb = _crossed_statistics(
        neural,
        hgs_matrices,
        replicates=int(recipe.bootstrap.replicates),
        seed=int(recipe.bootstrap.seed),
        quantile=float(recipe.bootstrap.confidence_level),
    )
    threshold = float(recipe.gate.maximum_mean_gap_pct)
    budget_points: dict[str, dict[str, Any]] = {}
    passing: list[int] = []
    for budget in budgets:
        matrix = hgs_matrices[budget]
        finite = bool(np.isfinite(matrix).all())
        seed_means, sample_std, t_upper = _t_ucb(
            matrix, float(recipe.bootstrap.seed_t_critical_value)
        )
        overall_mean = float(matrix.mean())
        invalid = sum(
            int(cells[(round_index, budget)]["validation"]["invalid_instance_count"])
            for round_index in range(len(seeds))
        )
        classification = replication._mode_classification(
            finite=finite,
            invalid_instances=invalid,
            mean_gap_pct=overall_mean if finite else None,
            maximum_seed_mean_gap_pct=max(seed_means) if finite else None,
            t_ucb_pct=t_upper if finite else None,
            bootstrap_ucb_pct=bootstrap_ucb[budget] if finite else None,
            threshold_pct=threshold,
        )
        if classification == "pass":
            passing.append(budget)
        cell_payloads = [cells[(round_index, budget)] for round_index in range(len(seeds))]
        measured_cells = [cell for cell in cell_payloads if cell["status"] == "complete"]
        energy_measurement_complete = len(measured_cells) == len(cell_payloads)
        total_cpu_energy = (
            sum(float(cell["energy"]["energy_cpu_j"]) for cell in measured_cells)
            if energy_measurement_complete
            else None
        )
        total_items = (
            sum(int(cell["valid_instances_processed"]) for cell in measured_cells)
            if energy_measurement_complete
            else None
        )
        cpu_per_instance = (
            total_cpu_energy / total_items
            if total_cpu_energy is not None and total_items is not None
            else None
        )
        budget_points[str(budget)] = {
            "classification": classification,
            "finite": finite,
            "invalid_instances": invalid,
            "gap_matrix_shape": list(matrix.shape),
            "gap_matrix_pct": matrix.tolist(),
            "per_seed_mean_gap_pct": seed_means,
            "mean_of_seed_means_gap_pct": overall_mean,
            "sample_std_of_seed_means_gap_pct": sample_std,
            "maximum_seed_mean_gap_pct": max(seed_means),
            "t_ucb_95_pct": t_upper,
            "bootstrap_ucb_95_pct": bootstrap_ucb[budget],
            "neural_minus_hgs_bootstrap_ucb_95_pct": neural_minus_hgs_ucb[budget],
            "maximum_mean_gap_pct": threshold,
            "strict_threshold": True,
            "cpu_package_energy_j": total_cpu_energy,
            "valid_instances_processed": total_items,
            "cpu_package_energy_j_per_valid_instance": cpu_per_instance,
            "per_block_cpu_package_energy_j_per_valid_instance": [
                (
                    float(cell["energy"]["cpu_package_energy_j_per_valid_instance"])
                    if cell["status"] == "complete"
                    else None
                )
                for cell in cell_payloads
            ],
            "energy_measurement_complete": energy_measurement_complete,
            "energy_not_measured_rounds": [
                int(cell["round"]) for cell in cell_payloads if cell["status"] != "complete"
            ],
            "energy_j_role": "diagnostic_only",
            "gpu_energy_role": "diagnostic_only",
            "energy_used_for_selection": False,
        }
    selected = min(passing) if passing else None
    status = COMPLETE_STATUS if selected is not None else NO_ELIGIBLE_STATUS
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": status,
        "classification": _classification(),
        "selection_split": "selection",
        "holdout_executed": False,
        "all_budgets_evaluated": True,
        "budget_order": list(budgets),
        "candidate_seeds": list(seeds),
        "neural_policy": neural_gate,
        "quality_method": {
            "metric": recipe.gate.metric,
            "comparison": recipe.gate.comparison,
            "threshold_pct": threshold,
            "bootstrap": _mapping(recipe.bootstrap),
            "unit_of_replication": "hgs_seed_round",
        },
        "budget_points": budget_points,
        "passing_budgets": passing,
        "selected_budget": selected,
        "selection_rule": recipe.selection.rule,
        "energy_used_for_selection": False,
        "primary_energy_estimand": recipe.measurement.primary_estimand,
        "cross_solver_energy_comparable": False,
        "aet_was_computed": False,
    }


def _expected_cell_order(recipe: Any) -> tuple[tuple[int, int, int, int, str], ...]:
    return tuple(
        (
            round_index,
            order_index,
            seed,
            budget,
            _cell_relative(round_index, budget, seed),
        )
        for round_index, seed, order in _round_schedule(recipe)
        for order_index, budget in enumerate(order)
    )


def _assert_hgs_cell_file_inventory(recipe: Any, output: Path) -> None:
    hgs_root = output / "hgs"
    if not hgs_root.exists():
        return
    if hgs_root.is_symlink() or not hgs_root.is_dir():
        raise HGSFrontierError("frontier HGS artifact root is unsafe")
    expected = {relative for *_prefix, relative in _expected_cell_order(recipe)}
    actual: set[str] = set()
    for path in hgs_root.rglob("*"):
        if path.is_symlink():
            raise HGSFrontierError("frontier HGS artifact inventory contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(output).as_posix())
    extra = actual - expected
    if extra:
        raise HGSFrontierError(
            f"frontier HGS artifact inventory contains unexpected files: {sorted(extra)}"
        )


def _invocation_binding(invocation: Any, number: int) -> dict[str, Any]:
    if not isinstance(invocation, dict) or invocation.get("invocation_number") != number:
        raise HGSFrontierError("frontier invocation numbering changed")
    evidence = invocation.get("preflight_evidence")
    attestation = invocation.get("exclusive_use_attestation")
    completed_cells = invocation.get("completed_cells_added")
    if (
        not isinstance(evidence, dict)
        or evidence.get("path") != f"qualification/preflight-invocation-{number:02d}.json"
        or not isinstance(evidence.get("sha256"), str)
        or _HEX_SHA256.fullmatch(evidence["sha256"]) is None
        or not isinstance(attestation, dict)
        or attestation.get("attested") is not True
        or not isinstance(attestation.get("attested_at"), str)
        or not isinstance(completed_cells, list)
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or _HEX_SHA256.fullmatch(item["sha256"]) is None
            for item in completed_cells
        )
        or len({item["path"] for item in completed_cells}) != len(completed_cells)
        or not isinstance(invocation.get("started_at"), str)
        or not isinstance(invocation.get("qualification"), dict)
        or not isinstance(invocation.get("runtime_identity"), dict)
    ):
        raise HGSFrontierError("frontier invocation provenance changed")
    attestation_sha256 = _attestation_sha256(attestation)
    if invocation.get("exclusive_use_attestation_sha256") != attestation_sha256:
        raise HGSFrontierError("frontier invocation attestation hash changed")
    ended_at = invocation.get("ended_at")
    outcome = invocation.get("outcome")
    end_time_basis = invocation.get("end_time_basis")
    if ended_at is None:
        if outcome != "running" or end_time_basis is not None:
            raise HGSFrontierError("active frontier invocation state changed")
    elif (
        not isinstance(ended_at, str)
        or not isinstance(outcome, str)
        or outcome == "running"
        or end_time_basis not in {"runner_observed", "resume_detection"}
    ):
        raise HGSFrontierError("closed frontier invocation state changed")
    return {
        "invocation_number": number,
        "preflight_evidence_sha256": evidence["sha256"],
        "exclusive_use_attestation_sha256": attestation_sha256,
        "exclusive_use_attested_at": attestation["attested_at"],
    }


def _invocation_cell_paths(invocation: Mapping[str, Any]) -> list[str]:
    return [item["path"] for item in invocation["completed_cells_added"]]


def _validate_invocation_history(
    recipe: Any,
    state: Mapping[str, Any],
    *,
    require_closed: bool,
) -> dict[int, dict[str, Any]]:
    invocations = state.get("invocations")
    if not isinstance(invocations, list):
        raise HGSFrontierError("frontier invocation history is invalid")
    expected_cell_rounds = {
        relative: round_index
        for round_index, _order, _seed, _budget, relative in _expected_cell_order(recipe)
    }
    expected_cell_paths = set(expected_cell_rounds)
    bindings: dict[int, dict[str, Any]] = {}
    referenced_cells: set[str] = set()
    active_numbers: list[int] = []
    for number, invocation in enumerate(invocations, start=1):
        binding = _invocation_binding(invocation, number)
        bindings[number] = binding
        completed_cells = _invocation_cell_paths(invocation)
        if any(path not in expected_cell_paths for path in completed_cells):
            raise HGSFrontierError("frontier invocation references an unexpected HGS cell")
        if len({expected_cell_rounds[path] for path in completed_cells}) > 1:
            raise HGSFrontierError("one frontier invocation spans more than one HGS round")
        overlap = referenced_cells.intersection(completed_cells)
        if overlap:
            raise HGSFrontierError("an HGS cell is attributed to multiple invocations")
        referenced_cells.update(completed_cells)
        if invocation.get("ended_at") is None:
            active_numbers.append(number)
    if active_numbers and active_numbers != [len(invocations)]:
        raise HGSFrontierError("only the latest frontier invocation may remain active")
    if require_closed and active_numbers:
        raise HGSFrontierError("completed frontier contains an active invocation")
    return bindings


def _recover_interrupted_invocation(
    recipe: Any,
    output: Path,
    state: dict[str, Any],
) -> None:
    bindings = _validate_invocation_history(recipe, state, require_closed=False)
    invocations = state["invocations"]
    if not invocations:
        return
    _assert_hgs_cell_file_inventory(recipe, output)
    expected_order = [relative for *_prefix, relative in _expected_cell_order(recipe)]
    linked_evidence: dict[int, list[dict[str, str]]] = {number: [] for number in bindings}
    for relative in expected_order:
        path = output / relative
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise HGSFrontierError(f"unsafe HGS cell encountered during resume: {relative}")
        payload = quality._load_json(path)
        number = payload.get("invocation_number")
        if isinstance(number, bool) or not isinstance(number, int) or number not in bindings:
            raise HGSFrontierError("HGS cell is not linked to a recorded invocation")
        linked_evidence[number].append({"path": relative, "sha256": quality._sha256_file(path)})

    changed = False
    for number, invocation in enumerate(invocations, start=1):
        declared = invocation["completed_cells_added"]
        actual = linked_evidence[number]
        actual_paths = {item["path"] for item in actual}
        if any(item["path"] not in actual_paths or item not in actual for item in declared):
            raise HGSFrontierError("frontier invocation claims a missing or relinked HGS cell")
        if declared != actual:
            invocation["completed_cells_added"] = actual
            changed = True
    active = invocations[-1]
    if active.get("ended_at") is None:
        recovered_at = datetime.now(UTC).isoformat()
        active["ended_at"] = recovered_at
        active["outcome"] = "interrupted_before_resume"
        active["end_time_basis"] = "resume_detection"
        changed = True
    if changed:
        quality._atomic_write_json(output / "run-state.json", state)
    _validate_invocation_history(recipe, state, require_closed=True)


def _active_invocation_binding(recipe: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    bindings = _validate_invocation_history(recipe, state, require_closed=False)
    invocations = state.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise HGSFrontierError("HGS measurement requires a recorded invocation")
    number = len(invocations)
    if invocations[-1].get("ended_at") is not None:
        raise HGSFrontierError("HGS measurement requires an active invocation")
    return bindings[number]


def _finish_invocation(
    recipe: Any,
    output: Path,
    state: dict[str, Any],
    *,
    outcome: str,
) -> None:
    if not outcome or outcome == "running":
        raise HGSFrontierError("frontier invocation outcome is invalid")
    _active_invocation_binding(recipe, state)
    invocation = state["invocations"][-1]
    invocation["ended_at"] = datetime.now(UTC).isoformat()
    invocation["outcome"] = outcome
    invocation["end_time_basis"] = "runner_observed"
    quality._atomic_write_json(output / "run-state.json", state)
    _validate_invocation_history(recipe, state, require_closed=True)


def _invalidate_hgs_integrity(
    recipe: Any,
    output: Path,
    state: dict[str, Any],
    failure: HGSFrontierIntegrityError,
) -> None:
    invalidated_at = datetime.now(UTC).isoformat()
    state["status"] = INVALIDATED_STATUS
    state["integrity_failure"] = {
        "schema_version": "aet-hgs-frontier-integrity-failure/v1",
        "invalidated_at": invalidated_at,
        "reason": str(failure),
        "details": failure.details,
        "resume_allowed": False,
        "recovery": "use a new output_root after diagnosing the HGS integrity failure",
    }
    _finish_invocation(
        recipe,
        output,
        state,
        outcome="hgs_integrity_failure",
    )


def _validate_cell_invocation_links(
    recipe: Any,
    output: Path,
    cells: Mapping[tuple[int, int], dict[str, Any]],
    state: Mapping[str, Any],
    *,
    require_closed: bool,
) -> None:
    bindings = _validate_invocation_history(recipe, state, require_closed=require_closed)
    invocations = state["invocations"]
    referenced: dict[str, tuple[int, str]] = {}
    for number, invocation in enumerate(invocations, start=1):
        for evidence in invocation["completed_cells_added"]:
            referenced[evidence["path"]] = (number, evidence["sha256"])
    actual_paths: set[str] = set()
    for (round_index, budget), cell in cells.items():
        seed = int(cell["base_seed"])
        relative = _cell_relative(round_index, budget, seed)
        actual_paths.add(relative)
        number = cell.get("invocation_number")
        if isinstance(number, bool) or not isinstance(number, int) or number not in bindings:
            raise HGSFrontierError("HGS cell is linked to an unknown invocation")
        binding = bindings[number]
        if any(not _strict_json_equal(cell.get(field), value) for field, value in binding.items()):
            raise HGSFrontierError("HGS cell invocation evidence changed")
        expected_evidence = (number, quality._sha256_file(output / relative))
        if referenced.get(relative) != expected_evidence:
            raise HGSFrontierError("HGS cell invocation inventory changed")
    if set(referenced) != actual_paths:
        raise HGSFrontierError("frontier invocation cell inventory changed")


def _load_neural_evaluations(
    recipe: Any,
    output: Path,
    evidence: ReplicationEvidence,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[int, dict[str, Any]]:
    evaluations: dict[int, dict[str, Any]] = {}
    for seed in sorted(evidence.checkpoints):
        path = output / _neural_relative(seed)
        if not path.is_file():
            raise HGSFrontierError(f"neural evaluation is missing for seed {seed}")
        evaluations[seed] = _validate_neural_evaluation(
            quality._load_json(path),
            recipe=recipe,
            output=output,
            evidence=evidence,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            seed=seed,
        )
    return evaluations


def _load_cells(
    recipe: Any,
    *,
    output: Path,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    _assert_hgs_cell_file_inventory(recipe, output)
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for round_index, order_index, seed, budget, relative in _expected_cell_order(recipe):
        path = output / relative
        if not path.exists():
            continue
        if not path.is_file():
            raise HGSFrontierError(f"HGS cell path is not a file: {relative}")
        cells[(round_index, budget)] = _validate_hgs_cell(
            quality._load_json(path),
            recipe=recipe,
            output=output,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            round_index=round_index,
            order_index=order_index,
            seed=seed,
            budget=budget,
        )
    return cells


def _cell_progress(
    recipe: Any,
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> tuple[list[str], list[int], list[int]]:
    schedule = _round_schedule(recipe)
    completed_paths: list[str] = []
    completed_rounds: list[int] = []
    incomplete_seen = False
    for round_index, seed, order in schedule:
        present = [budget for budget in order if (round_index, budget) in cells]
        if present != list(order[: len(present)]):
            raise HGSFrontierError(f"HGS cells are not a scheduled prefix in round {round_index}")
        if incomplete_seen and present:
            raise HGSFrontierError("HGS cells exist after the first incomplete round")
        completed_paths.extend(_cell_relative(round_index, budget, seed) for budget in present)
        if len(present) == len(order):
            completed_rounds.append(round_index)
        else:
            incomplete_seen = True
    remaining = [index for index, _seed, _order in schedule if index not in completed_rounds]
    return completed_paths, completed_rounds, remaining


def _reconcile_progress(
    recipe: Any,
    output: Path,
    state: dict[str, Any],
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> tuple[list[int], list[int]]:
    completed_paths, completed_rounds, remaining = _cell_progress(recipe, cells)
    declared_cells = state.get("completed_cells")
    declared_rounds = state.get("completed_rounds")
    if not isinstance(declared_cells, list) or not isinstance(declared_rounds, list):
        raise HGSFrontierError("frontier progress state is invalid")
    if not set(declared_cells).issubset(completed_paths):
        raise HGSFrontierError("run-state claims an HGS cell that is missing")
    if not set(declared_rounds).issubset(completed_rounds):
        raise HGSFrontierError("run-state claims an HGS round that is incomplete")
    if declared_cells != completed_paths or declared_rounds != completed_rounds:
        state["completed_cells"] = completed_paths
        state["completed_rounds"] = completed_rounds
        state["remaining_rounds"] = remaining
        quality._atomic_write_json(output / "run-state.json", state)
    return completed_rounds, remaining


def _complete_one_round(
    recipe: Any,
    *,
    root: Path,
    source_recipe: Path,
    recipe_bytes: bytes,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    cells: dict[tuple[int, int], dict[str, Any]],
) -> int | None:
    gate_path = output / "neural-quality-gate.json"
    anchored_gate = state.get("neural_quality_gate_sha256")
    if (
        not isinstance(anchored_gate, str)
        or not quality._file_matches_sha256(gate_path, anchored_gate)
        or quality._load_json(gate_path).get("passed") is not True
    ):
        raise HGSFrontierError("HGS measurement requires the anchored passing neural gate")
    invocation_binding = _active_invocation_binding(recipe, state)
    _validate_cell_invocation_links(recipe, output, cells, state, require_closed=False)
    _completed, remaining = _reconcile_progress(recipe, output, state, cells)
    if not remaining:
        return None
    target_round = remaining[0]
    round_index, seed, order = _round_schedule(recipe)[target_round]
    for order_index, budget in enumerate(order):
        key = (round_index, budget)
        if key in cells:
            continue
        _revalidate_live_inputs(
            recipe=recipe,
            source_recipe=source_recipe,
            recipe_bytes=recipe_bytes,
            root=root,
            output=output,
            state=state,
        )
        payload = _execute_hgs_cell(
            recipe,
            output=output,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            round_index=round_index,
            order_index=order_index,
            seed=seed,
            budget=budget,
            invocation_binding=invocation_binding,
        )
        relative = _cell_relative(round_index, budget, seed)
        quality._atomic_write_json(output / relative, payload)
        cells[key] = _validate_hgs_cell(
            quality._load_json(output / relative),
            recipe=recipe,
            output=output,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            round_index=round_index,
            order_index=order_index,
            seed=seed,
            budget=budget,
        )
        completed_paths, completed_rounds, remaining_rounds = _cell_progress(recipe, cells)
        state["status"] = INCOMPLETE_STATUS
        state["completed_cells"] = completed_paths
        state["completed_rounds"] = completed_rounds
        state["remaining_rounds"] = remaining_rounds
        invocation = state["invocations"][invocation_binding["invocation_number"] - 1]
        if relative in _invocation_cell_paths(invocation):
            raise HGSFrontierError("new HGS cell was already attributed to this invocation")
        invocation["completed_cells_added"].append(
            {"path": relative, "sha256": quality._sha256_file(output / relative)}
        )
        quality._atomic_write_json(output / "run-state.json", state)
        _validate_cell_invocation_links(recipe, output, cells, state, require_closed=False)
    completed_rounds, remaining = _reconcile_progress(recipe, output, state, cells)
    if target_round not in completed_rounds:
        raise HGSFrontierError("frontier invocation did not finish its selected round")
    state["status"] = INCOMPLETE_STATUS
    state["remaining_rounds"] = remaining
    quality._atomic_write_json(output / "run-state.json", state)
    return target_round


def _reference_inventory(reference: dict[str, Any]) -> set[str]:
    candidates = reference.get("candidate_artifacts")
    if not isinstance(candidates, list):
        raise HGSFrontierError("reference candidate inventory is invalid")
    paths = {"reference/selection/reference.json", "reference/reference-lock.json"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
            raise HGSFrontierError("reference candidate inventory entry is invalid")
        paths.add(str(candidate["path"]))
    return paths


def _expected_artifact_inventory(
    recipe: Any,
    reference: dict[str, Any],
    status: str,
    state: dict[str, Any],
) -> set[str]:
    corpus_relative = str(_selection_spec(recipe).artifact)
    expected = {
        "recipe.yaml",
        "environment/uv.lock",
        "provenance/quality-replication-receipt.json",
        corpus_relative,
        Path(corpus_relative).with_suffix(".manifest.json").as_posix(),
        "neural-quality-gate.json",
        "selection-assessment.json",
        "manifest.json",
        "SHA256SUMS",
        "run-state.json",
        *_reference_inventory(reference),
    }
    expected.update(_neural_relative(seed) for seed in recipe.neural_prerequisite.training_seeds)
    invocations = state.get("invocations")
    if not isinstance(invocations, list):
        raise HGSFrontierError("frontier invocation inventory is invalid")
    for invocation in invocations:
        evidence = invocation.get("preflight_evidence") if isinstance(invocation, dict) else None
        if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
            raise HGSFrontierError("frontier invocation lacks preflight evidence")
        expected.add(evidence["path"])
    if status != NEURAL_NONPASS_STATUS:
        expected.update(relative for *_prefix, relative in _expected_cell_order(recipe))
    return expected


def _assert_exact_artifact_inventory(
    recipe: Any,
    output: Path,
    reference: dict[str, Any],
    status: str,
    state: dict[str, Any],
) -> None:
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected = _expected_artifact_inventory(recipe, reference, status, state)
    if actual != expected:
        raise HGSFrontierError(
            "frontier artifact inventory changed; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _manifest_payload(
    recipe: Any,
    *,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_lock: dict[str, Any],
    neural_evaluations: Mapping[int, dict[str, Any]],
    neural_gate: dict[str, Any],
    cells: Mapping[tuple[int, int], dict[str, Any]],
    assessment: dict[str, Any],
    ended_at: str,
) -> dict[str, Any]:
    invocations = state.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise HGSFrontierError("frontier manifest requires an invocation history")
    final_invocation = invocations[-1]
    assessment_path = output / "selection-assessment.json"
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": assessment["status"],
        "classification": _classification(),
        "initialized_at": state["created_at"],
        "first_invocation_started_at": invocations[0]["started_at"],
        "final_invocation_started_at": final_invocation["started_at"],
        "ended_at": ended_at,
        "recipe": {
            "path": "recipe.yaml",
            "sha256": state["recipe_sha256"],
            "name": recipe.name,
        },
        "source": {
            "git_sha": state["git_sha"],
            "git_at_initialization": state["git"],
            "uv_lock_sha256": state["uv_lock_sha256"],
            "scoped_source_snapshot": state["source_snapshot"],
        },
        "runtime_controls": state["runtime_controls"],
        "execution_qualification": final_invocation["qualification"],
        "exclusive_use_attestation": final_invocation["exclusive_use_attestation"],
        "execution_invocations": invocations,
        "preflight_evidence": [invocation["preflight_evidence"] for invocation in invocations],
        "run_state_history": {
            "created_at": state["created_at"],
            "resume_cleanups": state.get("resume_cleanups", []),
        },
        "replication_source": {
            "path": "provenance/quality-replication-receipt.json",
            "sha256": quality._sha256_file(_source_receipt_path(output)),
            "copied_into_frontier": False,
            "validated_in_place": True,
        },
        "dataset": {
            "split": "selection",
            "path": corpus.path.relative_to(output).as_posix(),
            "file_sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
            "instances": int(corpus.coords.shape[0]),
            "fresh_seed": int(_selection_spec(recipe).seed),
        },
        "reference_lock": {
            "path": "reference/reference-lock.json",
            "sha256": state["reference_lock_sha256"],
            "payload": reference_lock,
            "reference_sha256": quality._sha256_file(
                output / "reference" / "selection" / "reference.json"
            ),
            "reference_mean_cost": reference["mean_cost"],
        },
        "neural_quality_gate": {
            "path": "neural-quality-gate.json",
            "sha256": quality._sha256_file(output / "neural-quality-gate.json"),
            "payload": neural_gate,
            "required_before_hgs_measurement": True,
        },
        "neural_evaluations": [
            {
                "training_seed": seed,
                "path": _neural_relative(seed),
                "sha256": quality._sha256_file(output / _neural_relative(seed)),
                "mean_gap_pct": neural_evaluations[seed]["quality"]["mean_gap_pct"],
                "energy_measured": False,
            }
            for seed in sorted(neural_evaluations)
        ],
        "round_schedule": state["round_schedule"],
        "hgs_cells": [
            {
                "round": round_index,
                "base_seed": cell["base_seed"],
                "budget": budget,
                "path": _cell_relative(round_index, budget, cell["base_seed"]),
                "sha256": quality._sha256_file(
                    output / _cell_relative(round_index, budget, cell["base_seed"])
                ),
                "status": cell["status"],
                "mean_gap_pct": cell["quality"]["mean_gap_pct"],
                "cpu_package_energy_j_per_valid_instance": cell["energy"].get(
                    "cpu_package_energy_j_per_valid_instance"
                ),
                "energy_j": cell["energy"].get("energy_j"),
                "energy_gpu_j": cell["energy"].get("energy_gpu_j"),
                "energy_j_role": "diagnostic_only",
                "gpu_energy_role": "diagnostic_only",
                "invocation_number": cell["invocation_number"],
                "preflight_evidence_sha256": cell["preflight_evidence_sha256"],
                "exclusive_use_attestation_sha256": cell["exclusive_use_attestation_sha256"],
            }
            for (round_index, budget), cell in sorted(cells.items())
        ],
        "selection_assessment": {
            "path": "selection-assessment.json",
            "sha256": quality._sha256_file(assessment_path),
            "payload": assessment,
        },
        "selection_rule": {
            "rule": recipe.selection.rule,
            "energy_used": False,
            "selected_budget": assessment["selected_budget"],
        },
        "primary_energy_estimand": recipe.measurement.primary_estimand,
        "total_and_gpu_energy_are_diagnostic_only": True,
        "holdout_executed": False,
        "cross_solver_energy_comparable": False,
        "aet_was_computed": False,
    }


def _revalidate_live_inputs(
    *,
    recipe: Any,
    source_recipe: Path,
    recipe_bytes: bytes,
    root: Path,
    output: Path,
    state: dict[str, Any],
) -> None:
    try:
        current_recipe = source_recipe.read_bytes()
    except OSError as exc:
        raise HGSFrontierQualificationError("frontier recipe became unavailable") from exc
    if (
        current_recipe != recipe_bytes
        or not quality._file_matches_sha256(output / "recipe.yaml", state["recipe_sha256"])
        or not quality._file_matches_sha256(
            output / "environment" / "uv.lock", state["uv_lock_sha256"]
        )
        or not quality._file_matches_sha256(root / "uv.lock", state["uv_lock_sha256"])
    ):
        raise HGSFrontierQualificationError("frontier frozen input changed during execution")
    if _source_snapshot(recipe, root)["sha256"] != state["source_snapshot"]["sha256"]:
        raise HGSFrontierQualificationError("frontier runtime source changed during execution")
    if quality._git_snapshot(root)["sha"] != state["git_sha"]:
        raise HGSFrontierQualificationError("frontier Git commit changed during execution")


def _validate_completed_run_state(recipe: Any, state: dict[str, Any]) -> None:
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
        "runtime_controls",
        "classification",
        "round_schedule",
        "replication_source_receipt_sha256",
        "reference_lock_sha256",
        "reference_locked_before_candidate_evaluation",
        "neural_completed_seeds",
        "neural_quality_gate_sha256",
        "completed_cells",
        "completed_rounds",
        "remaining_rounds",
        "invocations",
        "resume_cleanups",
        "completed_at",
        "selected_budget",
        "manifest_sha256",
        "checksums_sha256",
    }
    if set(state) != expected_keys:
        raise HGSFrontierError("completed frontier run-state schema changed")
    if (
        state.get("schema_version") != RUN_STATE_SCHEMA
        or state.get("status") not in COMPLETE_STATUSES
        or not _strict_json_equal(state.get("classification"), _classification())
        or state.get("reference_locked_before_candidate_evaluation") is not True
        or state.get("neural_completed_seeds") != list(recipe.neural_prerequisite.training_seeds)
    ):
        raise HGSFrontierError("completed frontier run-state invariants changed")
    for field in (
        "recipe_sha256",
        "uv_lock_sha256",
        "replication_source_receipt_sha256",
        "reference_lock_sha256",
        "neural_quality_gate_sha256",
        "manifest_sha256",
        "checksums_sha256",
    ):
        value = state.get(field)
        if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
            raise HGSFrontierError(f"completed frontier has invalid {field}")
    if not isinstance(state.get("created_at"), str) or not isinstance(
        state.get("completed_at"), str
    ):
        raise HGSFrontierError("completed frontier timestamps changed")
    expected_schedule = [
        {"round": index, "seed": seed, "budget_order": list(order)}
        for index, seed, order in _round_schedule(recipe)
    ]
    if not _strict_json_equal(state.get("round_schedule"), expected_schedule):
        raise HGSFrontierError("completed frontier round schedule changed")
    invocations = state.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise HGSFrontierError("completed frontier invocation history changed")
    _validate_invocation_history(recipe, state, require_closed=True)
    all_paths = [relative for *_prefix, relative in _expected_cell_order(recipe)]
    all_rounds = list(range(len(_candidate_seeds(recipe))))
    if state["status"] == NEURAL_NONPASS_STATUS:
        expected_progress = ([], [], all_rounds, None)
    else:
        expected_progress = (all_paths, all_rounds, [], state.get("selected_budget"))
    actual_progress = (
        state.get("completed_cells"),
        state.get("completed_rounds"),
        state.get("remaining_rounds"),
        state.get("selected_budget"),
    )
    if not _strict_json_equal(actual_progress, expected_progress):
        raise HGSFrontierError("completed frontier progress changed")
    if state["status"] == COMPLETE_STATUS:
        selected = state.get("selected_budget")
        if isinstance(selected, bool) or selected not in _budgets(recipe):
            raise HGSFrontierError("completed frontier selected budget changed")
    elif state.get("selected_budget") is not None:
        raise HGSFrontierError("non-passing frontier unexpectedly selected a budget")


def _validate_completed_bundle(
    recipe: Any,
    quality_recipe: Any,
    evidence: ReplicationEvidence,
    *,
    root: Path,
    source_recipe: Path,
    recipe_bytes: bytes,
    output: Path,
    state: dict[str, Any],
) -> FrontierResult:
    _validate_completed_run_state(recipe, state)
    _revalidate_live_inputs(
        recipe=recipe,
        source_recipe=source_recipe,
        recipe_bytes=recipe_bytes,
        root=root,
        output=output,
        state=state,
    )
    quality._verify_checksums(output, expected_sha256=state["checksums_sha256"])
    if quality._stale_partial_artifacts(output):
        raise HGSFrontierError("completed frontier contains atomic partial artifacts")
    if not _strict_json_equal(quality._load_json(_source_receipt_path(output)), evidence.receipt):
        raise HGSFrontierError("completed frontier replication receipt changed")
    _validate_preflight_evidence(recipe, output, state)

    corpus, reference, reference_lock = _prepare_corpus_and_reference(
        recipe, quality_recipe, output, state
    )
    reference_sha256 = quality._sha256_file(output / "reference" / "selection" / "reference.json")
    evaluations = _load_neural_evaluations(
        recipe, output, evidence, corpus, reference, reference_sha256
    )
    neural_gate = quality._load_json(output / "neural-quality-gate.json")
    if (
        not _strict_json_equal(neural_gate, _neural_quality_gate(recipe, evaluations))
        or quality._sha256_file(output / "neural-quality-gate.json")
        != state["neural_quality_gate_sha256"]
    ):
        raise HGSFrontierError("completed frontier neural quality gate changed")
    cells = _load_cells(
        recipe,
        output=output,
        corpus=corpus,
        reference=reference,
        reference_sha256=reference_sha256,
    )
    _validate_cell_invocation_links(recipe, output, cells, state, require_closed=True)
    if state["status"] == NEURAL_NONPASS_STATUS and cells:
        raise HGSFrontierError("HGS was measured after the neural prerequisite failed")
    assessment = quality._load_json(output / "selection-assessment.json")
    recomputed = _build_assessment(recipe, evaluations, cells)
    if not _strict_json_equal(assessment, recomputed):
        raise HGSFrontierError("completed frontier assessment changed")
    if assessment["status"] != state["status"]:
        raise HGSFrontierError("completed frontier status disagrees with assessment")
    _assert_exact_artifact_inventory(recipe, output, reference, state["status"], state)

    manifest_path = output / "manifest.json"
    if quality._sha256_file(manifest_path) != state["manifest_sha256"]:
        raise HGSFrontierError("completed frontier manifest hash changed")
    manifest = quality._load_json(manifest_path)
    expected_manifest = _manifest_payload(
        recipe,
        output=output,
        state=state,
        corpus=corpus,
        reference=reference,
        reference_lock=reference_lock,
        neural_evaluations=evaluations,
        neural_gate=neural_gate,
        cells=cells,
        assessment=assessment,
        ended_at=str(manifest.get("ended_at")),
    )
    if (
        not _strict_json_equal(manifest, expected_manifest)
        or manifest.get("status") != state["status"]
        or manifest.get("ended_at") != state["completed_at"]
    ):
        raise HGSFrontierError("completed frontier manifest relationships changed")
    return FrontierResult(
        path=output,
        status=state["status"],
        complete=True,
        selected_budget=state["selected_budget"],
        completed_rounds=tuple(state["completed_rounds"]),
        remaining_rounds=tuple(state["remaining_rounds"]),
        manifest_path=manifest_path,
        manifest_sha256=state["manifest_sha256"],
    )


def _finalize(
    recipe: Any,
    quality_recipe: Any,
    evidence: ReplicationEvidence,
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
    neural_gate: dict[str, Any],
    cells: Mapping[tuple[int, int], dict[str, Any]],
) -> FrontierResult:
    _validate_invocation_history(recipe, state, require_closed=True)
    _validate_cell_invocation_links(recipe, output, cells, state, require_closed=True)
    assessment = _build_assessment(recipe, neural_evaluations, cells)
    assessment_path = output / "selection-assessment.json"
    quality._atomic_write_json(assessment_path, assessment)
    if not _strict_json_equal(
        quality._load_json(assessment_path),
        _build_assessment(recipe, neural_evaluations, cells),
    ):
        raise HGSFrontierError("frontier assessment changed after atomic write")

    _revalidate_live_inputs(
        recipe=recipe,
        source_recipe=source_recipe,
        recipe_bytes=recipe_bytes,
        root=root,
        output=output,
        state=state,
    )
    current_evidence = _validate_replication_source(recipe, root)
    if not _strict_json_equal(current_evidence.receipt, evidence.receipt):
        raise HGSFrontierQualificationError(
            "quality-replication source changed during frontier execution"
        )

    all_paths, all_rounds, remaining = _cell_progress(recipe, cells)
    if assessment["status"] == NEURAL_NONPASS_STATUS:
        all_paths = []
        all_rounds = []
        remaining = list(range(len(_candidate_seeds(recipe))))
    ended_at = datetime.now(UTC).isoformat()
    candidate_state = {
        **state,
        "status": assessment["status"],
        "completed_at": ended_at,
        "completed_cells": all_paths,
        "completed_rounds": all_rounds,
        "remaining_rounds": remaining,
        "selected_budget": assessment["selected_budget"],
        "resume_cleanups": list(state.get("resume_cleanups", [])),
    }
    candidate_state.pop("resume_reason", None)
    manifest = _manifest_payload(
        recipe,
        output=output,
        state=candidate_state,
        corpus=corpus,
        reference=reference,
        reference_lock=reference_lock,
        neural_evaluations=neural_evaluations,
        neural_gate=neural_gate,
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
        root=root,
        source_recipe=source_recipe,
        recipe_bytes=recipe_bytes,
        output=output,
        state=candidate_state,
    )
    quality._atomic_write_json(output / "run-state.json", candidate_state)
    state.clear()
    state.update(candidate_state)
    return result


def _record_invocation(
    output: Path,
    state: dict[str, Any],
    *,
    recipe: Any,
    root: Path,
    qualification: dict[str, Any],
    runtime_identity: dict[str, Any],
    attestation: dict[str, Any],
) -> dict[str, Any]:
    invocations = state.get("invocations")
    if not isinstance(invocations, list):
        raise HGSFrontierError("frontier invocation history is invalid")
    _validate_preflight_evidence(recipe, output, state)
    _validate_invocation_history(recipe, state, require_closed=False)
    _cleanup_unreferenced_preflight_evidence(output, state)
    _recover_interrupted_invocation(recipe, output, state)
    preflight_evidence = _capture_preflight_evidence(
        recipe,
        root=root,
        output=output,
        invocation_number=len(invocations) + 1,
        qualification=qualification,
    )
    number = len(invocations) + 1
    invocation = {
        "invocation_number": number,
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": None,
        "outcome": "running",
        "end_time_basis": None,
        "runtime_identity": runtime_identity,
        "qualification": qualification,
        "exclusive_use_attestation": attestation,
        "exclusive_use_attestation_sha256": _attestation_sha256(attestation),
        "preflight_evidence": preflight_evidence,
        "completed_cells_added": [],
    }
    invocations.append(invocation)
    quality._atomic_write_json(output / "run-state.json", state)
    return _invocation_binding(invocation, number)


def _cleanup_unreferenced_preflight_evidence(
    output: Path,
    state: dict[str, Any],
) -> None:
    invocations = state.get("invocations")
    cleanups = state.get("resume_cleanups")
    if not isinstance(invocations, list) or not isinstance(cleanups, list):
        raise HGSFrontierError("frontier recovery history is invalid")
    referenced = {
        invocation["preflight_evidence"]["path"]
        for invocation in invocations
        if isinstance(invocation, dict)
        and isinstance(invocation.get("preflight_evidence"), dict)
        and isinstance(invocation["preflight_evidence"].get("path"), str)
    }
    qualification_dir = output / "qualification"
    if not qualification_dir.exists():
        return
    orphaned: list[dict[str, str]] = []
    for path in sorted(qualification_dir.glob("preflight-invocation-*.json")):
        relative = path.relative_to(output).as_posix()
        if relative in referenced:
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or not re.fullmatch(r"preflight-invocation-[0-9]{2,}\.json", path.name)
        ):
            raise HGSFrontierError("unsafe unreferenced preflight evidence path")
        digest = quality._sha256_file(path)
        path.unlink()
        orphaned.append({"path": relative, "sha256_before_removal": digest})
    if orphaned:
        cleanups.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "kind": "unreferenced_atomic_preflight_capture",
                "removed": orphaned,
            }
        )
        quality._atomic_write_json(output / "run-state.json", state)


def _validate_preflight_payload(recipe: Any, payload: dict[str, Any]) -> None:
    system = payload.get("system")
    runtime = payload.get("runtime")
    backends = payload.get("backends")
    if (
        not isinstance(system, dict)
        or not isinstance(runtime, dict)
        or not isinstance(backends, dict)
    ):
        raise HGSFrontierQualificationError("frontier preflight structure is incomplete")
    emi = backends.get("windows_emi")
    nvml = backends.get("nvml")
    selected = nvml.get("selected_device") if isinstance(nvml, dict) else None
    if (
        payload.get("schema_version") != "aet-preflight/v1"
        or payload.get("host_id") != recipe.host_id
        or system.get("execution_layer") != "windows-native"
        or system.get("cpu_label") != recipe.cpu_label
        or not isinstance(system.get("windows_build"), int)
        or system["windows_build"] < 22000
        or runtime.get("codecarbon") != "3.3.1"
        or not isinstance(emi, dict)
        or emi.get("available") is not True
        or emi.get("backend_mode") != "windows_emi"
        or emi.get("measurement_scope") != "cpu_package"
        or emi.get("fallback_used") is not False
        or not isinstance(selected, dict)
        or selected.get("index") != int(recipe.gpu_index)
        or selected.get("device_id_sha256") != recipe.gpu_device_id_sha256
        or selected.get("measurement_mode") != "total_energy_counter"
        or selected.get("total_energy_counter_supported") is not True
    ):
        raise HGSFrontierQualificationError(
            "frontier preflight hardware or counter evidence changed"
        )


def _capture_preflight_evidence(
    recipe: Any,
    *,
    root: Path,
    output: Path,
    invocation_number: int,
    qualification: dict[str, Any],
) -> dict[str, Any]:
    source = _relative(root, str(recipe.preflight_report)).resolve(strict=True)
    try:
        source.relative_to(root)
        before = source.read_bytes()
        payload = json.loads(before.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HGSFrontierQualificationError(
            "frontier preflight evidence is unavailable or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise HGSFrontierQualificationError("frontier preflight evidence is not an object")
    _validate_preflight_payload(recipe, payload)
    fresh_qualification = runtime_qualification(recipe, repository_root=root)
    if not _strict_json_equal(fresh_qualification, qualification):
        raise HGSFrontierQualificationError(
            "frontier qualification changed before preflight capture"
        )
    summary = qualification.get("preflight")
    if (
        not isinstance(summary, dict)
        or summary.get("ready_to_execute") is not True
        or summary.get("declared_cpu_label_matches") is not True
        or summary.get("preflight_cpu_label") != recipe.cpu_label
        or summary.get("cpu_label_source") != recipe.cpu_label_source
        or summary.get("cpu_identity_independently_verified")
        is not recipe.cpu_identity_independently_verified
    ):
        raise HGSFrontierQualificationError(
            "frontier preflight evidence disagrees with qualification"
        )
    if source.read_bytes() != before:
        raise HGSFrontierQualificationError("frontier preflight changed while captured")
    relative = f"qualification/preflight-invocation-{invocation_number:02d}.json"
    destination = output / relative
    if destination.exists() or destination.is_symlink():
        raise HGSFrontierError(f"refusing to overwrite preflight evidence: {relative}")
    quality._atomic_write_bytes(destination, before)
    digest = quality._sha256_bytes(before)
    if not quality._file_matches_sha256(destination, digest):
        raise HGSFrontierError("captured frontier preflight hash changed")
    return {
        "path": relative,
        "sha256": digest,
        "source_path": str(recipe.preflight_report),
        "schema_version": payload["schema_version"],
        "host_id": payload["host_id"],
        "cpu_label": payload["system"]["cpu_label"],
        "cpu_label_source": recipe.cpu_label_source,
        "cpu_identity_independently_verified": (recipe.cpu_identity_independently_verified),
        "windows_build": payload["system"]["windows_build"],
        "gpu_index": int(recipe.gpu_index),
        "gpu_device_id_sha256": recipe.gpu_device_id_sha256,
        "cpu_backend": "windows_emi",
        "gpu_backend": "nvml_total_energy_counter",
    }


def _validate_preflight_evidence(recipe: Any, output: Path, state: dict[str, Any]) -> None:
    invocations = state.get("invocations")
    if not isinstance(invocations, list):
        raise HGSFrontierError("frontier preflight invocation history is invalid")
    seen: set[str] = set()
    for number, invocation in enumerate(invocations, start=1):
        evidence = invocation.get("preflight_evidence") if isinstance(invocation, dict) else None
        expected_path = f"qualification/preflight-invocation-{number:02d}.json"
        if (
            not isinstance(evidence, dict)
            or evidence.get("path") != expected_path
            or expected_path in seen
            or not isinstance(evidence.get("sha256"), str)
            or not quality._file_matches_sha256(output / expected_path, evidence["sha256"])
        ):
            raise HGSFrontierError("frontier preflight evidence inventory changed")
        payload = quality._load_json(output / expected_path)
        _validate_preflight_payload(recipe, payload)
        expected = {
            "path": expected_path,
            "sha256": quality._sha256_file(output / expected_path),
            "source_path": str(recipe.preflight_report),
            "schema_version": payload["schema_version"],
            "host_id": payload["host_id"],
            "cpu_label": payload["system"]["cpu_label"],
            "cpu_label_source": recipe.cpu_label_source,
            "cpu_identity_independently_verified": (recipe.cpu_identity_independently_verified),
            "windows_build": payload["system"]["windows_build"],
            "gpu_index": int(recipe.gpu_index),
            "gpu_device_id_sha256": recipe.gpu_device_id_sha256,
            "cpu_backend": "windows_emi",
            "gpu_backend": "nvml_total_energy_counter",
        }
        if not _strict_json_equal(evidence, expected):
            raise HGSFrontierError("frontier preflight evidence metadata changed")
        seen.add(expected_path)


def _runtime_controls(recipe: Any) -> dict[str, Any]:
    configured = software._configure_runtime()
    return {
        **configured,
        "tracker_backend": recipe.measurement.tracker_backend,
        "required_domains": list(recipe.measurement.required_domains),
        "cpu_primary_backend": recipe.measurement.cpu_primary_backend,
        "gpu_backend": recipe.measurement.gpu_backend,
        "fallback_allowed": False,
        "minimum_block_duration_s": float(recipe.measurement.minimum_block_duration_s),
        "repeat_full_corpus": True,
        "primary_estimand": recipe.measurement.primary_estimand,
        "energy_j_role": "diagnostic_only",
        "gpu_energy_role": "diagnostic_only",
        "hgs_cpu_threads": 1,
    }


def _runtime_identity(
    recipe: Any,
    quality_recipe: Any,
    qualification: dict[str, Any],
) -> dict[str, Any]:
    """Build the quality identity from live CUDA, not the reduced preflight view."""

    import torch

    if qualification.get("ready_to_execute") is not True or not torch.cuda.is_available():
        raise HGSFrontierQualificationError(
            "frontier runtime identity requires the qualified CUDA device"
        )
    device_count = int(torch.cuda.device_count())
    gpu_index = int(recipe.gpu_index)
    if gpu_index < 0 or gpu_index >= device_count:
        raise HGSFrontierQualificationError("frontier selected GPU index is unavailable at launch")
    selected_name = str(torch.cuda.get_device_name(gpu_index))
    if selected_name != str(recipe.expected_accelerator_label):
        raise HGSFrontierQualificationError(
            "frontier live GPU name disagrees with the closed recipe"
        )
    identity = quality._runtime_identity(
        quality_recipe,
        {
            "cuda_device_count": device_count,
            "selected_gpu_name": selected_name,
        },
    )
    identity["qualification_shape"] = "live_cuda_plus_closed_frontier_preflight"
    preflight = qualification.get("preflight")
    expected_cpu = str(recipe.cpu_label)
    if (
        not isinstance(preflight, dict)
        or preflight.get("declared_cpu_label_matches") is not True
        or preflight.get("preflight_cpu_label") != expected_cpu
        or preflight.get("cpu_label_source") != recipe.cpu_label_source
        or preflight.get("cpu_identity_independently_verified")
        is not recipe.cpu_identity_independently_verified
    ):
        raise HGSFrontierQualificationError(
            "frontier preflight CPU label disagrees with the closed recipe"
        )
    identity["declared_cpu_label"] = expected_cpu
    identity["preflight_cpu_label"] = expected_cpu
    identity["cpu_label_source"] = recipe.cpu_label_source
    identity["cpu_identity_independently_verified"] = recipe.cpu_identity_independently_verified
    return identity


def execute_hgs_frontier(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    resume: bool = False,
) -> FrontierResult:
    """Execute or resume at most one complete paired HGS seed round."""

    try:
        root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
        source_recipe = Path(recipe_path).resolve(strict=True)
    except OSError as exc:
        raise HGSFrontierQualificationError(
            f"frontier workspace or recipe is unavailable: {exc}"
        ) from exc
    if root != Path.cwd().resolve(strict=True):
        raise HGSFrontierQualificationError("workspace_root must be the current repository")
    try:
        source_recipe.relative_to(root)
    except ValueError as exc:
        raise HGSFrontierQualificationError(
            "frontier recipe must be stored inside the repository"
        ) from exc
    try:
        recipe_bytes = source_recipe.read_bytes()
        recipe = load_aet_hgs_frontier_recipe(source_recipe)
    except Exception as exc:
        raise HGSFrontierQualificationError(f"frontier recipe is invalid: {exc}") from exc
    if source_recipe.read_bytes() != recipe_bytes:
        raise HGSFrontierQualificationError("frontier recipe changed while it was being loaded")

    output = quality._safe_output_target(root, str(recipe.output_root))
    if (output / "run-state.json").is_file():
        existing_state = quality._load_json(output / "run-state.json")
        if existing_state.get("status") == INVALIDATED_STATUS:
            raise HGSFrontierError(
                "frontier output was invalidated by an HGS integrity failure; "
                "diagnose it and use a new output_root"
            )
        if existing_state.get("status") in COMPLETE_STATUSES:
            evidence = _validate_replication_source(recipe, root)
            quality_recipe = _selection_quality_recipe(recipe, evidence.base_recipe)
            return _validate_completed_bundle(
                recipe,
                quality_recipe,
                evidence,
                root=root,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                output=output,
                state=existing_state,
            )

    qualification = runtime_qualification(recipe, repository_root=root)
    if qualification.get("ready_to_execute") is not True:
        raise HGSFrontierQualificationError(
            "native-Windows frontier qualification is incomplete; no measured work ran"
        )
    try:
        attestation = software._exclusive_use_attestation()
        runtime_controls = _runtime_controls(recipe)
    except (software.SoftwareSmokeError, software.SoftwareSmokeQualificationError) as exc:
        raise HGSFrontierQualificationError(str(exc)) from exc
    evidence = _validate_replication_source(recipe, root)
    quality_recipe = _selection_quality_recipe(recipe, evidence.base_recipe)
    try:
        runtime_identity = _runtime_identity(recipe, quality_recipe, qualification)
    except Exception as exc:
        raise HGSFrontierQualificationError(
            f"frontier runtime identity is unavailable: {exc}"
        ) from exc

    with quality._output_lock(output):
        if source_recipe.read_bytes() != recipe_bytes:
            raise HGSFrontierQualificationError("frontier recipe changed before initialization")
        output, state = _prepare_output(
            recipe=recipe,
            recipe_bytes=recipe_bytes,
            root=root,
            evidence=evidence,
            runtime_identity=runtime_identity,
            runtime_controls=runtime_controls,
            resume=resume,
        )
        if state.get("status") in COMPLETE_STATUSES:
            return _validate_completed_bundle(
                recipe,
                quality_recipe,
                evidence,
                root=root,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                output=output,
                state=state,
            )
        _record_invocation(
            output,
            state,
            recipe=recipe,
            root=root,
            qualification=qualification,
            runtime_identity=runtime_identity,
            attestation=attestation,
        )
        corpus, reference, reference_lock = _prepare_corpus_and_reference(
            recipe, quality_recipe, output, state
        )
        reference_sha256 = quality._sha256_file(
            output / "reference" / "selection" / "reference.json"
        )
        neural_evaluations = _prepare_neural_evaluations(
            recipe,
            output,
            state,
            evidence,
            corpus,
            reference,
            reference_sha256,
        )
        neural_gate = _prepare_neural_quality_gate(recipe, output, state, neural_evaluations)
        if neural_gate["passed"] is not True:
            _finish_invocation(
                recipe,
                output,
                state,
                outcome="neural_prerequisite_nonpass",
            )
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
                neural_gate=neural_gate,
                cells={},
            )

        cells = _load_cells(
            recipe,
            output=output,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
        )
        _validate_cell_invocation_links(
            recipe,
            output,
            cells,
            state,
            require_closed=False,
        )
        _reconcile_progress(recipe, output, state, cells)
        if len(cells) == len(_expected_cell_order(recipe)):
            _finish_invocation(
                recipe,
                output,
                state,
                outcome="recovered_complete_without_new_measurement",
            )
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
                neural_gate=neural_gate,
                cells=cells,
            )
        try:
            _complete_one_round(
                recipe,
                root=root,
                source_recipe=source_recipe,
                recipe_bytes=recipe_bytes,
                output=output,
                state=state,
                corpus=corpus,
                reference=reference,
                reference_sha256=reference_sha256,
                cells=cells,
            )
        except HGSFrontierIntegrityError as exc:
            _invalidate_hgs_integrity(recipe, output, state, exc)
            raise HGSFrontierError(
                "HGS integrity failure invalidated this frontier output; "
                "a new output_root is required"
            ) from exc
        _finish_invocation(
            recipe,
            output,
            state,
            outcome="paired_seed_round_completed",
        )
        _revalidate_live_inputs(
            recipe=recipe,
            source_recipe=source_recipe,
            recipe_bytes=recipe_bytes,
            root=root,
            output=output,
            state=state,
        )
        if len(cells) == len(_expected_cell_order(recipe)):
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
                neural_gate=neural_gate,
                cells=cells,
            )
        completed_rounds, remaining = _reconcile_progress(recipe, output, state, cells)
        state["status"] = INCOMPLETE_STATUS
        state["remaining_rounds"] = remaining
        quality._atomic_write_json(output / "run-state.json", state)
        return FrontierResult(
            path=output,
            status=INCOMPLETE_STATUS,
            complete=False,
            completed_rounds=tuple(completed_rounds),
            remaining_rounds=tuple(remaining),
        )


def _result_payload(result: FrontierResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "path": result.path.as_posix(),
        "complete": result.complete,
        "selected_budget": result.selected_budget,
        "completed_rounds": list(result.completed_rounds),
        "remaining_rounds": list(result.remaining_rounds),
        "manifest": result.manifest_path.as_posix() if result.manifest_path else None,
        "manifest_sha256": result.manifest_sha256,
        **_classification(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = execute_hgs_frontier(args.recipe, resume=args.resume)
    except HGSFrontierError as exc:
        parser.exit(2, f"HGS frontier failed: {exc}\n")
    print(json.dumps(_result_payload(result), indent=2, sort_keys=True), flush=True)
    if not result.complete:
        return 4
    if result.selected_budget is None:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrontierResult",
    "HGSFrontierError",
    "HGSFrontierQualificationError",
    "execute_hgs_frontier",
    "main",
]
