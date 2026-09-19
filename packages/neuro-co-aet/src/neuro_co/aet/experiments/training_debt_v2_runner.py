"""Train AM and GNN for 100 epochs and select checkpoints prospectively.

Each architecture/seed pair is an indivisible resumable cell.  Training energy
and checkpoint-selection energy are measured in separate direct-counter
intervals.  Selection evaluates epochs 90 through 100 with greedy 1x1 decoding
on the frozen development split.  This runner does not load a final holdout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import statistics
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments import software_smoke_runner as software_runner
from neuro_co.aet.experiments import training_debt_runner as v1
from neuro_co.aet.experiments.training_debt_v2_recipe import (
    CLASSIFICATION,
    Architecture,
    TrainingDebtV2Recipe,
    TrainingDebtV2RecipeValidationError,
    load_aet_training_debt_v2_recipe,
    runtime_qualification,
)

RUN_STATE_SCHEMA = "aet-training-debt-v2-run-state/v1"
TRAINING_ENERGY_SCHEMA = "aet-training-debt-v2-training-energy/v1"
SELECTION_ENERGY_SCHEMA = "aet-training-debt-v2-selection-energy/v1"
EVALUATION_SCHEMA = "aet-training-debt-v2-selection-evaluation/v1"
SELECTION_SCHEMA = "aet-training-debt-v2-checkpoint-selection/v1"
SEED_RESULT_SCHEMA = "aet-training-debt-v2-seed-result/v1"
SUMMARY_SCHEMA = "aet-training-debt-v2-summary/v1"
MANIFEST_SCHEMA = "aet-training-debt-v2-manifest/v1"
SOURCE_RECEIPT_SCHEMA = "aet-training-debt-v2-source-receipt/v1"
ESTIMATE_SCHEMA = "aet-training-debt-v2-estimate/v1"
COMPLETE_STATUS = "complete"
INCOMPLETE_STATUS = "incomplete_training_cells_pending"


class TrainingDebtV2Error(RuntimeError):
    """Raised when the measured v2 campaign cannot continue safely."""


class TrainingDebtV2QualificationError(TrainingDebtV2Error):
    """Raised before a measured v2 cell starts."""


@dataclass(frozen=True, slots=True)
class TrainingDebtV2Result:
    path: Path
    status: str
    complete: bool
    completed_cells: tuple[str, ...]
    remaining_cells: tuple[str, ...]
    manifest_path: Path | None = None
    manifest_sha256: str | None = None


def _json_bytes(value: Any) -> bytes:
    return v1._json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return v1._sha256_bytes(value)


def _sha256_file(path: Path) -> str:
    return v1._sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    return v1._load_json(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    v1._atomic_write_json(path, value)


def _relative(root: Path, value: str) -> Path:
    return root.joinpath(*PurePosixPath(value).parts)


def _canonical_sha256(value: Any) -> str:
    return v1._canonical_sha256(value)


def architecture_recipe_sha256(recipe: TrainingDebtV2Recipe, architecture: Architecture) -> str:
    return v1.architecture_recipe_sha256(recipe, architecture)  # type: ignore[arg-type]


def model_state_sha256(state: Mapping[str, Any]) -> str:
    return v1.model_state_sha256(state)


def _cell_id(architecture: str, seed: int) -> str:
    return f"{architecture}:seed-{seed:03d}"


def _cell_relative(architecture: str, seed: int) -> str:
    return f"architectures/{architecture}/seeds/seed-{seed:03d}"


def _expected_cells(
    recipe: TrainingDebtV2Recipe,
) -> tuple[tuple[Architecture, int], ...]:
    return tuple(
        (architecture, seed)
        for architecture in recipe.architectures
        for seed in recipe.training.seeds
    )


def _load_selection_source(
    recipe: TrainingDebtV2Recipe, root: Path
) -> tuple[quality.Corpus, dict[str, Any], str]:
    source = _relative(root, recipe.selection_source_root)
    manifest = _load_json(source / "manifest.json")
    if manifest.get("status") != recipe.selection_source_status:
        raise TrainingDebtV2QualificationError("selection source status changed")
    corpus_path = _relative(source, recipe.corpus.path)
    reference_path = _relative(source, recipe.reference.path)
    if _sha256_file(corpus_path) != recipe.corpus.sha256:
        raise TrainingDebtV2QualificationError("development corpus changed")
    if _sha256_file(reference_path) != recipe.reference.sha256:
        raise TrainingDebtV2QualificationError("development reference changed")
    try:
        with np.load(corpus_path, allow_pickle=False) as archive:
            coords = np.ascontiguousarray(archive["coords"], dtype=np.float32)
            demands = np.ascontiguousarray(archive["demands"], dtype=np.float32)
            capacity = float(archive["capacity"].item())
            content_sha256 = str(archive["content_sha256"].item())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise TrainingDebtV2QualificationError("development corpus cannot be loaded") from exc
    if (
        coords.shape != (512, 51, 2)
        or demands.shape != (512, 51)
        or capacity != 40.0
        or content_sha256 != recipe.corpus_content_sha256
    ):
        raise TrainingDebtV2QualificationError("development corpus identity changed")
    reference = _load_json(reference_path)
    costs = reference.get("costs")
    if reference.get("split") != "development" or not isinstance(costs, list) or len(costs) != 512:
        raise TrainingDebtV2QualificationError("development reference vector changed")
    return (
        quality.Corpus(
            split="development",
            split_id="cvrp50-development-seed2711",
            coords=coords,
            demands=demands,
            capacity=capacity,
            content_sha256=content_sha256,
            file_sha256=recipe.corpus.sha256,
            path=corpus_path,
        ),
        reference,
        recipe.reference.sha256,
    )


def _evaluate_checkpoint_greedy(
    recipe: TrainingDebtV2Recipe,
    architecture: Architecture,
    *,
    seed: int,
    epoch: int,
    checkpoint_path: Path,
    expected_identity: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    import torch

    from neuro_co.core.factory import make_env, make_model

    checkpoint = quality._load_torch_mapping(checkpoint_path)
    if (
        checkpoint.get("completed_epochs") != epoch
        or checkpoint.get("training_seed") != seed
        or checkpoint.get("architecture") != architecture.architecture_id
    ):
        raise TrainingDebtV2Error("candidate checkpoint metadata changed")
    env = make_env("cvrp", size=50, capacity=40.0, max_demand=9)
    model = make_model(
        env,
        backbone=architecture.backbone,
        hidden_dim=architecture.hidden_dim,
        num_layers=architecture.num_layers,
        num_heads=architecture.num_heads,
    )
    identity = v1._model_identity(recipe, architecture, model)  # type: ignore[arg-type]
    if identity != expected_identity or checkpoint.get("model_identity") != identity:
        raise TrainingDebtV2Error("candidate checkpoint model identity changed")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(f"cuda:{recipe.gpu_index}")
    model.to(device=device, dtype=torch.float32).eval()
    routes: list[Any] = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, corpus.coords.shape[0], recipe.selection.batch_size):
            stop = min(start + recipe.selection.batch_size, corpus.coords.shape[0])
            state = quality._initial_state(
                env, corpus.coords[start:stop], corpus.demands[start:stop], device
            )
            routes.extend(quality._greedy_routes(model, env, state))
    torch.cuda.synchronize(recipe.gpu_index)
    elapsed = time.perf_counter() - started
    validation = quality.validate_routes(corpus.coords, corpus.demands, corpus.capacity, routes)
    metrics = quality._gap_summary(validation["costs"], reference["costs"])
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "classification": {
            "purpose": "prospective_checkpoint_selection",
            "split": "development",
            "final_holdout_used": False,
            "included_in_selection_energy": True,
        },
        "architecture": architecture.architecture_id,
        "training_seed": seed,
        "epoch": epoch,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_state_sha256": model_state_sha256(checkpoint["model"]),
        "model_identity_sha256": identity["model_identity_sha256"],
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "mode": asdict(recipe.selection),
        "elapsed_s": elapsed,
        "routes": routes,
        "validation": validation,
        "quality": metrics,
    }


def _make_tracker(label: str, recipe: TrainingDebtV2Recipe) -> Any:
    return v1._make_tracker(label, recipe)  # type: ignore[arg-type]


def _checked_component_energy(
    tracker: Any,
    recipe: TrainingDebtV2Recipe,
    *,
    items_processed: int,
    minimum_duration_s: float,
    schema_version: str,
    phase: str,
) -> dict[str, Any]:
    try:
        payload = software_runner._checked_energy(
            tracker,
            minimum_duration_s=minimum_duration_s,
            gpu_index=recipe.gpu_index,
            gpu_device_id_sha256=recipe.gpu_device_id_sha256,
        )
    except Exception as exc:
        raise TrainingDebtV2Error(f"strict {phase} energy record is invalid: {exc}") from exc
    if payload.get("items_processed") != items_processed:
        raise TrainingDebtV2Error(f"{phase} energy item count differs from workload")
    for key in ("co2_operational_kg", "co2_embodied_kg", "co2_total_kg"):
        if payload.get(key) not in (None, 0, 0.0):
            raise TrainingDebtV2Error(f"{phase} energy record contains carbon output")
    cpu = float(payload["energy_cpu_j"])
    gpu = float(payload["energy_gpu_j"])
    observed = cpu + gpu
    if not math.isclose(float(payload["energy_j"]), observed, rel_tol=1e-12, abs_tol=1e-9):
        raise TrainingDebtV2Error(f"{phase} tracker total differs from CPU plus GPU")
    return {
        "schema_version": schema_version,
        "status": "complete",
        "phase": phase,
        "classification": CLASSIFICATION,
        "cpu_package_energy_j": cpu,
        "gpu_energy_j": gpu,
        "observed_component_energy_j": observed,
        "duration_s": float(payload["duration_s"]),
        "items_processed": items_processed,
        "whole_system_energy": False,
        "carbon_accounting": "none",
        "raw_tracker_reading": payload,
    }


def _choose_checkpoint_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise TrainingDebtV2Error("checkpoint selection has no candidates")
    for candidate in candidates:
        epoch = candidate.get("epoch")
        gap = candidate.get("mean_gap_pct")
        invalid = candidate.get("invalid_instances")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or isinstance(gap, bool)
            or not isinstance(gap, (int, float))
            or not math.isfinite(float(gap))
            or invalid != 0
        ):
            raise TrainingDebtV2Error("checkpoint candidate is invalid")
    return min(candidates, key=lambda item: (float(item["mean_gap_pct"]), int(item["epoch"])))


def _select_checkpoint(
    recipe: TrainingDebtV2Recipe,
    architecture: Architecture,
    *,
    seed: int,
    staging: Path,
    training_result: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records_by_epoch = {int(item["epoch"]): item for item in training_result["checkpoint_records"]}
    tracker = _make_tracker(
        f"checkpoint-selection-{architecture.architecture_id}-seed{seed}", recipe
    )
    evaluations: list[dict[str, Any]] = []
    with tracker as active:
        for epoch in recipe.selection.candidate_epochs:
            record = records_by_epoch.get(epoch)
            if record is None:
                raise TrainingDebtV2Error(f"selection checkpoint epoch {epoch} is missing")
            evaluations.append(
                _evaluate_checkpoint_greedy(
                    recipe,
                    architecture,
                    seed=seed,
                    epoch=epoch,
                    checkpoint_path=staging / str(record["path"]),
                    expected_identity=training_result["model_identity"],
                    corpus=corpus,
                    reference=reference,
                    reference_sha256=reference_sha256,
                )
            )
        active.n_items = len(evaluations) * int(corpus.coords.shape[0])
    energy = _checked_component_energy(
        tracker,
        recipe,
        items_processed=len(evaluations) * int(corpus.coords.shape[0]),
        minimum_duration_s=recipe.minimum_selection_duration_s,
        schema_version=SELECTION_ENERGY_SCHEMA,
        phase="checkpoint_selection",
    )
    candidates = [
        {
            "epoch": int(item["epoch"]),
            "mean_gap_pct": float(item["quality"]["mean_gap_pct"]),
            "invalid_instances": int(item["validation"]["invalid_instance_count"]),
            "checkpoint_sha256": str(item["checkpoint_sha256"]),
            "model_state_sha256": str(item["model_state_sha256"]),
            "evaluation_path": f"selection/evaluations/epoch-{int(item['epoch']):03d}.json",
        }
        for item in evaluations
    ]
    selected = _choose_checkpoint_candidate(candidates)
    for evaluation in evaluations:
        path = staging / "selection" / "evaluations" / f"epoch-{evaluation['epoch']:03d}.json"
        _atomic_write_json(path, evaluation)
    energy.update(
        {
            "architecture": architecture.architecture_id,
            "training_seed": seed,
            "candidate_checkpoint_count": len(evaluations),
            "development_instances_per_checkpoint": int(corpus.coords.shape[0]),
            "selection_energy_is_predeployment_debt": True,
            "minimum_duration_s": recipe.minimum_selection_duration_s,
            "measurement_quality": "direct_component_counters_short_block",
        }
    )
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "status": "selected",
        "architecture": architecture.architecture_id,
        "training_seed": seed,
        "source_split": "development",
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "final_holdout_used": False,
        "mode": asdict(recipe.selection),
        "criterion": recipe.selection.criterion,
        "tie_break": recipe.selection.tie_break,
        "candidates": candidates,
        "selected_epoch": selected["epoch"],
        "selected_mean_gap_pct": selected["mean_gap_pct"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_model_state_sha256": selected["model_state_sha256"],
        "selection_energy_path": "selection/energy.json",
        "selection_energy_included_in_predeployment_debt": True,
        "deployment_holdout_required": True,
    }
    _atomic_write_json(staging / "selection" / "energy.json", energy)
    _atomic_write_json(staging / "selection" / "checkpoint-selection.json", selection)
    return selection, energy


def _source_receipt(recipe: TrainingDebtV2Recipe, root: Path) -> dict[str, Any]:
    source = _relative(root, recipe.selection_source_root)
    base = _relative(root, recipe.base_recipe.path)
    if _sha256_file(base) != recipe.base_recipe.sha256:
        raise TrainingDebtV2QualificationError("base recipe changed")
    artifacts: list[dict[str, str]] = []
    for artifact in (
        *recipe.selection_source_artifacts,
        recipe.corpus,
        recipe.reference,
        recipe.reference_lock,
    ):
        path = _relative(source, artifact.path)
        if _sha256_file(path) != artifact.sha256:
            raise TrainingDebtV2QualificationError(f"selection source changed: {artifact.path}")
        record = {"path": artifact.path, "sha256": artifact.sha256}
        if record not in artifacts:
            artifacts.append(record)
    manifest = _load_json(source / "manifest.json")
    if manifest.get("status") != recipe.selection_source_status:
        raise TrainingDebtV2QualificationError("selection source manifest status changed")
    return {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "base_recipe": {
            "source_path": recipe.base_recipe.path,
            "sha256": recipe.base_recipe.sha256,
        },
        "selection_source": {
            "root": recipe.selection_source_root,
            "expected_status": recipe.selection_source_status,
            "split": "development",
            "artifacts": artifacts,
            "corpus_content_sha256": recipe.corpus_content_sha256,
            "final_holdout_used": False,
        },
        "architecture_recipes": {
            item.architecture_id: {
                "sha256": architecture_recipe_sha256(recipe, item),
                "training": asdict(recipe.training),
                "selection": asdict(recipe.selection),
                "model_configuration": item.model_configuration,
            }
            for item in recipe.architectures
        },
    }


def _git_snapshot(root: Path, preflight: Path) -> dict[str, Any]:
    try:
        return v1._git_snapshot(root, preflight)
    except v1.TrainingDebtQualificationError as exc:
        raise TrainingDebtV2QualificationError(str(exc)) from exc


def _prepare_output(
    recipe_path: Path,
    recipe: TrainingDebtV2Recipe,
    root: Path,
    *,
    resume: bool,
    runtime_identity: dict[str, Any],
    source_receipt: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output = quality._safe_output_target(root, recipe.output_root)
    recipe_bytes = recipe_path.read_bytes()
    recipe_sha = _sha256_bytes(recipe_bytes)
    lock_bytes = (root / "uv.lock").read_bytes()
    lock_sha = _sha256_bytes(lock_bytes)
    base_bytes = _relative(root, recipe.base_recipe.path).read_bytes()
    preflight = _relative(root, recipe.preflight_report)
    git = _git_snapshot(root, preflight)
    source_bytes = _json_bytes(source_receipt)
    source_sha = _sha256_bytes(source_bytes)
    state_path = output / "run-state.json"
    expected = {
        "schema_version": RUN_STATE_SCHEMA,
        "recipe_sha256": recipe_sha,
        "uv_lock_sha256": lock_sha,
        "base_recipe_sha256": recipe.base_recipe.sha256,
        "git_sha": git.get("sha"),
        "worktree_fingerprint_sha256": git.get("worktree_fingerprint_sha256"),
        "runtime_identity": runtime_identity,
        "source_receipt_sha256": source_sha,
        "classification": CLASSIFICATION,
    }
    if output.exists():
        if not resume:
            raise TrainingDebtV2Error(f"output already exists; use --resume: {output}")
        state = _load_json(state_path)
        if any(state.get(key) != value for key, value in expected.items()):
            raise TrainingDebtV2Error("resume identity differs from initialized v2 campaign")
        for relative, digest in (
            ("recipe.yaml", recipe_sha),
            ("environment/uv.lock", lock_sha),
            ("base-recipe.yaml", recipe.base_recipe.sha256),
            ("source-receipt.json", source_sha),
        ):
            if _sha256_file(output / relative) != digest:
                raise TrainingDebtV2Error(f"frozen v2 campaign artifact changed: {relative}")
        return output, state

    state = {
        **expected,
        "status": INCOMPLETE_STATUS,
        "created_at": datetime.now(UTC).isoformat(),
        "completed_cells": [],
        "remaining_cells": [
            _cell_id(architecture.architecture_id, seed)
            for architecture, seed in _expected_cells(recipe)
        ],
        "invocations": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "environment").mkdir()
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "base-recipe.yaml").write_bytes(base_bytes)
        (staging / "environment" / "uv.lock").write_bytes(lock_bytes)
        (staging / "source-receipt.json").write_bytes(source_bytes)
        _atomic_write_json(
            staging / "attempt-ledger.json",
            {
                "schema_version": v1.ATTEMPT_LEDGER_SCHEMA,
                "append_only_semantics": True,
                "events": [],
            },
        )
        _atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _preserve_preflight(recipe: TrainingDebtV2Recipe, root: Path, output: Path) -> dict[str, str]:
    source = _relative(root, recipe.preflight_report)
    digest = _sha256_file(source)
    relative = f"qualification/preflight-{digest}.json"
    target = output / relative
    if target.exists():
        if _sha256_file(target) != digest:
            raise TrainingDebtV2Error("preserved preflight changed")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        quality._atomic_write_bytes(target, source.read_bytes())
    return {"path": relative, "sha256": digest}


def _recover_staging(output: Path) -> None:
    staging = output / "staging"
    if not staging.exists():
        return
    if staging.is_symlink() or not staging.is_dir():
        raise TrainingDebtV2Error("v2 training staging path is unsafe")
    for architecture_dir in staging.iterdir():
        if (
            architecture_dir.name not in {"am", "gnn"}
            or architecture_dir.is_symlink()
            or not architecture_dir.is_dir()
        ):
            raise TrainingDebtV2Error("v2 training staging contains an unknown artifact")
        for seed_dir in architecture_dir.iterdir():
            prefix, separator, raw_seed = seed_dir.name.partition("-")
            if (
                prefix != "seed"
                or separator != "-"
                or len(raw_seed) != 3
                or not raw_seed.isdecimal()
                or seed_dir.is_symlink()
                or not seed_dir.is_dir()
            ):
                raise TrainingDebtV2Error("v2 training staging contains an unsafe seed")
            seed = int(raw_seed)
            measured: dict[str, Any] = {}
            for phase, relative in (
                ("training", "training-energy.json"),
                ("checkpoint_selection", "selection/energy.json"),
            ):
                path = seed_dir / relative
                if path.is_file():
                    with contextlib.suppress(quality.QualityPilotError):
                        payload = _load_json(path)
                        observed = payload.get("observed_component_energy_j")
                        if (
                            not isinstance(observed, bool)
                            and isinstance(observed, (int, float))
                            and math.isfinite(float(observed))
                            and float(observed) > 0.0
                        ):
                            measured[phase] = payload
            ledger = v1._attempt_ledger(output)
            attempt_id, is_open = v1._latest_attempt_for_cell(
                ledger, architecture=architecture_dir.name, seed=seed
            )
            if attempt_id is None:
                attempt_id = str(uuid.uuid4())
            known_energy = sum(
                float(item.get("observed_component_energy_j", 0.0)) for item in measured.values()
            )
            v1._append_attempt_event(
                output,
                {
                    "event": (
                        "interrupted_attempt_recovered" if is_open else "attempt_staging_discarded"
                    ),
                    "attempt_id": attempt_id,
                    "architecture": architecture_dir.name,
                    "training_seed": seed,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "measured_phase_energy": measured,
                    "known_excluded_component_energy_j": (known_energy if measured else None),
                    "energy_status": (
                        "known_components_excluded_from_primary" if measured else "unknown"
                    ),
                    "primary_recipe_debt_eligible": False,
                    "reason": "partial cell cannot be combined across attempts",
                },
            )
            shutil.rmtree(seed_dir)
        architecture_dir.rmdir()
    staging.rmdir()


def _bind_training_energy_record(staging: Path, training_result: dict[str, Any]) -> dict[str, Any]:
    """Point the v2 training result at its actual seed-root energy record."""

    result_path = staging / "training" / "result.json"
    energy_path = staging / "training-energy.json"
    if not result_path.is_file() or not energy_path.is_file():
        raise TrainingDebtV2Error("training result or v2 training energy record is missing")
    if _load_json(result_path) != training_result:
        raise TrainingDebtV2Error("training result changed before energy binding")
    if training_result.get("energy_record") != "../energy.json":
        raise TrainingDebtV2Error("delegated training result energy path changed")
    bound = {**training_result, "energy_record": "../training-energy.json"}
    _atomic_write_json(result_path, bound)
    resolved_energy = (result_path.parent / str(bound["energy_record"])).resolve(strict=True)
    if resolved_energy != energy_path.resolve(strict=True):
        raise TrainingDebtV2Error("v2 training result does not resolve to its energy record")
    return bound


def _load_bound_training_result(seed_root: Path, expected_sha256: str) -> dict[str, Any]:
    result_path = seed_root / "training" / "result.json"
    energy_path = seed_root / "training-energy.json"
    if _sha256_file(result_path) != expected_sha256:
        raise TrainingDebtV2Error("completed training result changed")
    result = _load_json(result_path)
    if (
        result.get("schema_version") != v1.TRAINING_RESULT_SCHEMA
        or result.get("status") != COMPLETE_STATUS
        or result.get("energy_record") != "../training-energy.json"
    ):
        raise TrainingDebtV2Error("completed training result energy binding changed")
    try:
        resolved_energy = (result_path.parent / str(result["energy_record"])).resolve(strict=True)
    except OSError as exc:
        raise TrainingDebtV2Error("completed training energy record is unavailable") from exc
    if resolved_energy != energy_path.resolve(strict=True):
        raise TrainingDebtV2Error("completed training result points to the wrong energy record")
    return result


def _selected_checkpoint_entry_from_seed_root(
    recipe: TrainingDebtV2Recipe,
    architecture: Architecture,
    seed: int,
    seed_root: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    selected_epoch = int(selection["selected_epoch"])
    checkpoint_relative = (
        f"{_cell_relative(architecture.architecture_id, seed)}/"
        f"training/checkpoints/epoch-{selected_epoch:03d}.pt"
    )
    checkpoint_path = seed_root / "training" / "checkpoints" / f"epoch-{selected_epoch:03d}.pt"
    checkpoint = quality._load_torch_mapping(checkpoint_path)
    identity = checkpoint.get("model_identity")
    if not isinstance(identity, dict):
        raise TrainingDebtV2Error("selected checkpoint model identity is missing")
    state_sha = model_state_sha256(checkpoint.get("model", {}))
    if (
        _sha256_file(checkpoint_path) != selection["selected_checkpoint_sha256"]
        or state_sha != selection["selected_model_state_sha256"]
    ):
        raise TrainingDebtV2Error("selected checkpoint differs from selection decision")
    return {
        "architecture": architecture.architecture_id,
        "training_seed": seed,
        "selected_epoch": selected_epoch,
        "path": checkpoint_relative,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_state_sha256": state_sha,
        "model_identity": identity,
        "model_identity_sha256": identity.get("model_identity_sha256"),
        "model_configuration": architecture.model_configuration,
        "backend_identity": identity.get("backend_identity"),
        "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
        "training_result_path": (
            f"{_cell_relative(architecture.architecture_id, seed)}/training/result.json"
        ),
        "seed_result_path": f"{_cell_relative(architecture.architecture_id, seed)}/seed-result.json",
        "training_energy_path": (
            f"{_cell_relative(architecture.architecture_id, seed)}/training-energy.json"
        ),
        "selection_path": (
            f"{_cell_relative(architecture.architecture_id, seed)}/"
            "selection/checkpoint-selection.json"
        ),
        "selection_energy_path": (
            f"{_cell_relative(architecture.architecture_id, seed)}/selection/energy.json"
        ),
        "selection_evaluation_path": (
            f"{_cell_relative(architecture.architecture_id, seed)}/selection/evaluations/"
            f"epoch-{selected_epoch:03d}.json"
        ),
    }


def _selected_checkpoint_entry(
    recipe: TrainingDebtV2Recipe,
    architecture: Architecture,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    seed_root = output / _cell_relative(architecture.architecture_id, seed)
    selection = _load_json(seed_root / "selection" / "checkpoint-selection.json")
    return _selected_checkpoint_entry_from_seed_root(
        recipe, architecture, seed, seed_root, selection
    )


def _load_completed_cell(
    recipe: TrainingDebtV2Recipe,
    architecture: Architecture,
    output: Path,
    seed: int,
    *,
    corpus: quality.Corpus,
    reference: dict[str, Any],
) -> dict[str, Any]:
    root = output / _cell_relative(architecture.architecture_id, seed)
    stored = _load_json(root / "seed-result.json")
    entry = _selected_checkpoint_entry(recipe, architecture, output, seed)
    training_energy = _load_json(root / "training-energy.json")
    selection_energy = _load_json(root / "selection" / "energy.json")
    selection = _load_json(root / "selection" / "checkpoint-selection.json")
    training_result_sha256 = stored.get("training_result_sha256")
    if not isinstance(training_result_sha256, str):
        raise TrainingDebtV2Error("completed training result digest is missing")
    _load_bound_training_result(root, training_result_sha256)
    if (
        stored.get("schema_version") != SEED_RESULT_SCHEMA
        or stored.get("status") != COMPLETE_STATUS
        or stored.get("architecture") != architecture.architecture_id
        or stored.get("training_seed") != seed
        or stored.get("checkpoint") != entry
        or stored.get("training_energy") != training_energy
        or stored.get("selection_energy") != selection_energy
        or stored.get("selection_sha256")
        != _sha256_file(root / "selection" / "checkpoint-selection.json")
    ):
        raise TrainingDebtV2Error(
            f"completed v2 cell changed: {_cell_id(architecture.architecture_id, seed)}"
        )
    if (
        training_energy.get("schema_version") != TRAINING_ENERGY_SCHEMA
        or selection_energy.get("schema_version") != SELECTION_ENERGY_SCHEMA
        or training_energy.get("observed_component_energy_j", 0) <= 0
        or selection_energy.get("observed_component_energy_j", 0) <= 0
        or training_energy.get("duration_s", 0) < recipe.minimum_training_duration_s
        or selection_energy.get("duration_s", 0) < recipe.minimum_selection_duration_s
    ):
        raise TrainingDebtV2Error("completed v2 energy record changed")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(
        recipe.selection.candidate_epochs
    ):
        raise TrainingDebtV2Error("completed checkpoint selection changed")
    for candidate in candidates:
        epoch = int(candidate["epoch"])
        evaluation_path = root / "selection" / "evaluations" / f"epoch-{epoch:03d}.json"
        evaluation = _load_json(evaluation_path)
        validation = quality.validate_routes(
            corpus.coords, corpus.demands, corpus.capacity, evaluation.get("routes", ())
        )
        metrics = quality._gap_summary(validation["costs"], reference["costs"])
        if evaluation.get("validation") != validation or evaluation.get("quality") != metrics:
            raise TrainingDebtV2Error("stored development evaluation changed")
        if (
            candidate.get("mean_gap_pct") != metrics["mean_gap_pct"]
            or candidate.get("invalid_instances") != validation["invalid_instance_count"]
        ):
            raise TrainingDebtV2Error("stored checkpoint candidate summary changed")
    selected = _choose_checkpoint_candidate(candidates)
    if (
        selection.get("selected_epoch") != selected["epoch"]
        or selection.get("selected_mean_gap_pct") != selected["mean_gap_pct"]
        or selection.get("final_holdout_used") is not False
    ):
        raise TrainingDebtV2Error("stored checkpoint selection rule changed")
    expected_debt = float(training_energy["observed_component_energy_j"]) + float(
        selection_energy["observed_component_energy_j"]
    )
    if not math.isclose(
        float(stored.get("predeployment_energy_debt_j", -1)),
        expected_debt,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise TrainingDebtV2Error("stored predeployment energy debt changed")
    return stored


def _completed_prefix(
    recipe: TrainingDebtV2Recipe,
    output: Path,
    *,
    corpus: quality.Corpus,
    reference: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    missing_seen = False
    for architecture, seed in _expected_cells(recipe):
        cell_id = _cell_id(architecture.architecture_id, seed)
        path = output / _cell_relative(architecture.architecture_id, seed) / "seed-result.json"
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise TrainingDebtV2Error("completed v2 cells are not an ordered prefix")
        completed[cell_id] = _load_completed_cell(
            recipe,
            architecture,
            output,
            seed,
            corpus=corpus,
            reference=reference,
        )
    return completed


def _execute_cell(
    recipe: TrainingDebtV2Recipe,
    architecture: Architecture,
    *,
    output: Path,
    state: dict[str, Any],
    seed: int,
    attestation: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    cell_id = _cell_id(architecture.architecture_id, seed)
    staging = output / "staging" / architecture.architecture_id / f"seed-{seed:03d}"
    final = output / _cell_relative(architecture.architecture_id, seed)
    if staging.exists() or final.exists():
        raise TrainingDebtV2Error(f"fresh v2 training target already exists: {cell_id}")
    attempt_id = str(uuid.uuid4())
    v1._append_attempt_event(
        output,
        {
            "event": "attempt_started",
            "attempt_id": attempt_id,
            "architecture": architecture.architecture_id,
            "training_seed": seed,
            "recorded_at": datetime.now(UTC).isoformat(),
            "attestation_session_id": attestation["session_id"],
            "energy_status": "unknown",
            "primary_recipe_debt_eligible": False,
        },
    )
    try:
        staging.mkdir(parents=True)
        tracker = _make_tracker(
            f"training-debt-v2-{architecture.architecture_id}-seed{seed}", recipe
        )
        started_at = datetime.now(UTC)
        wall_started = time.perf_counter()
        with tracker as active:
            training_result = v1._train_seed_workload(
                recipe,  # type: ignore[arg-type]
                architecture,
                seed=seed,
                staging=staging,
                recipe_sha256=state["recipe_sha256"],
                git_sha=state["git_sha"],
            )
            active.n_items = recipe.training.epochs * recipe.training.instances_per_epoch
        training_ended_at = datetime.now(UTC)
        training_elapsed = time.perf_counter() - wall_started
        if training_elapsed > recipe.maximum_seed_walltime_s:
            raise TrainingDebtV2Error(f"{cell_id} exceeded maximum measured seed walltime")
        training_energy = _checked_component_energy(
            tracker,
            recipe,
            items_processed=recipe.training.epochs * recipe.training.instances_per_epoch,
            minimum_duration_s=recipe.minimum_training_duration_s,
            schema_version=TRAINING_ENERGY_SCHEMA,
            phase="training",
        )
        training_energy.update(
            {
                "architecture": architecture.architecture_id,
                "training_seed": seed,
                "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
                "started_at": started_at.isoformat(),
                "ended_at": training_ended_at.isoformat(),
                "attestation": attestation,
                "energy_per_training_instance_j": (
                    float(training_energy["observed_component_energy_j"])
                    / (recipe.training.epochs * recipe.training.instances_per_epoch)
                ),
                "measured_boundary": (
                    "model_initialization_through_final_checkpoint_serialization"
                ),
            }
        )
        _atomic_write_json(staging / "training-energy.json", training_energy)
        training_result = _bind_training_energy_record(staging, training_result)
        selection, selection_energy = _select_checkpoint(
            recipe,
            architecture,
            seed=seed,
            staging=staging,
            training_result=training_result,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
        )
        entry = _selected_checkpoint_entry_from_seed_root(
            recipe, architecture, seed, staging, selection
        )
        predeployment_debt = float(training_energy["observed_component_energy_j"]) + float(
            selection_energy["observed_component_energy_j"]
        )
        seed_result = {
            "schema_version": SEED_RESULT_SCHEMA,
            "status": COMPLETE_STATUS,
            "architecture": architecture.architecture_id,
            "training_seed": seed,
            "checkpoint": entry,
            "training_energy": training_energy,
            "selection_energy": selection_energy,
            "predeployment_energy_debt_j": predeployment_debt,
            "predeployment_energy_debt_includes": ["training", "checkpoint_selection"],
            "training_result_sha256": _sha256_file(staging / "training" / "result.json"),
            "selection_sha256": _sha256_file(staging / "selection" / "checkpoint-selection.json"),
            "selection": {
                "source_split": "development",
                "selected_epoch": selection["selected_epoch"],
                "selected_mean_gap_pct": selection["selected_mean_gap_pct"],
                "criterion": selection["criterion"],
                "tie_break": selection["tie_break"],
                "final_holdout_used": False,
            },
            "quality_status": "not_evaluated_on_final_holdout",
            "deployment_holdout_required": True,
        }
        _atomic_write_json(staging / "seed-result.json", seed_result)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        completed = _load_completed_cell(
            recipe,
            architecture,
            output,
            seed,
            corpus=corpus,
            reference=reference,
        )
        v1._append_attempt_event(
            output,
            {
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "architecture": architecture.architecture_id,
                "training_seed": seed,
                "recorded_at": datetime.now(UTC).isoformat(),
                "energy_status": "training_and_selection_complete",
                "predeployment_energy_debt_j": predeployment_debt,
                "primary_recipe_debt_eligible": True,
            },
        )
        return completed
    except BaseException as exc:
        v1._append_attempt_event(
            output,
            {
                "event": "attempt_failed",
                "attempt_id": attempt_id,
                "architecture": architecture.architecture_id,
                "training_seed": seed,
                "recorded_at": datetime.now(UTC).isoformat(),
                "energy_status": "excluded_incomplete_cell",
                "primary_recipe_debt_eligible": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def _energy_statistics(values: Sequence[float]) -> dict[str, Any]:
    return v1._energy_statistics(values)


def _architecture_summary(
    recipe: TrainingDebtV2Recipe,
    architecture: Architecture,
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    analyzer_seeds: list[dict[str, Any]] = []
    for result in results:
        training = result["training_energy"]
        selection_energy = result["selection_energy"]
        checkpoint = result["checkpoint"]
        selection = result["selection"]
        total = float(result["predeployment_energy_debt_j"])
        record = {
            "training_seed": result["training_seed"],
            "training_energy_j": training["observed_component_energy_j"],
            "checkpoint_selection_energy_j": selection_energy["observed_component_energy_j"],
            "predeployment_energy_debt_j": total,
            "training_cpu_package_energy_j": training["cpu_package_energy_j"],
            "training_gpu_energy_j": training["gpu_energy_j"],
            "selection_cpu_package_energy_j": selection_energy["cpu_package_energy_j"],
            "selection_gpu_energy_j": selection_energy["gpu_energy_j"],
            "training_duration_s": training["duration_s"],
            "selection_duration_s": selection_energy["duration_s"],
            "selected_epoch": checkpoint["selected_epoch"],
            "selected_mean_gap_pct": selection["selected_mean_gap_pct"],
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "model_state_sha256": checkpoint["model_state_sha256"],
            "model_identity_sha256": checkpoint["model_identity_sha256"],
        }
        per_seed.append(record)
        analyzer_seeds.append(
            {
                "seed": result["training_seed"],
                "energy_j": total,
                "training_energy_j": training["observed_component_energy_j"],
                "checkpoint_selection_energy_j": selection_energy["observed_component_energy_j"],
                "selected_epoch": checkpoint["selected_epoch"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "model_state_sha256": checkpoint["model_state_sha256"],
                "model_identity_sha256": checkpoint["model_identity_sha256"],
            }
        )
    training_values = [float(item["training_energy_j"]) for item in per_seed]
    selection_values = [float(item["checkpoint_selection_energy_j"]) for item in per_seed]
    debt_values = [float(item["predeployment_energy_debt_j"]) for item in per_seed]
    model_identity = results[0]["checkpoint"]["model_identity"]
    if any(item["checkpoint"]["model_identity"] != model_identity for item in results[1:]):
        raise TrainingDebtV2Error("model identity differs across v2 training seeds")
    return {
        "status": COMPLETE_STATUS,
        "architecture": architecture.architecture_id,
        "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
        "model_configuration": architecture.model_configuration,
        "model_identity": model_identity,
        "model_identity_sha256": model_identity["model_identity_sha256"],
        "backend_identity": model_identity["backend_identity"],
        "seed_count": len(results),
        "training_seeds": [int(item["training_seed"]) for item in per_seed],
        "recipe_training_energy_j_mean": statistics.mean(debt_values),
        "recipe_training_energy_j_mean_includes_checkpoint_selection": True,
        "selection_energy_included": True,
        "training_only_energy_j_mean": statistics.mean(training_values),
        "checkpoint_selection_energy_j_mean": statistics.mean(selection_values),
        "energy_debt": {
            "definition": (
                "mean training plus prospective checkpoint-selection component energy "
                "for one independently trained model under the frozen recipe"
            ),
            "included_components": ["training", "checkpoint_selection"],
            "predeployment_energy_debt_j": _energy_statistics(debt_values),
            "training_energy_j": _energy_statistics(training_values),
            "checkpoint_selection_energy_j": _energy_statistics(selection_values),
            "whole_system_energy": False,
        },
        "selection_debt": {
            "status": "measured",
            "included_in_primary": True,
            "observed_component_energy_j": _energy_statistics(selection_values),
            "measurement_quality": "direct_component_counters_short_block",
        },
        "study_cost_measured_five_seeds_j_sum": sum(debt_values),
        "study_cost_is_not_single_model_training_debt": True,
        "per_seed": per_seed,
        "seeds": analyzer_seeds,
        "checkpoint_selection": {
            "status": "complete",
            "source_split": "development",
            "mode_id": recipe.selection.mode_id,
            "candidate_epochs": list(recipe.selection.candidate_epochs),
            "criterion": recipe.selection.criterion,
            "tie_break": recipe.selection.tie_break,
            "selected_epochs": [int(item["selected_epoch"]) for item in per_seed],
            "final_holdout_used": False,
        },
        "quality_status": "not_evaluated_on_final_holdout",
        "frontier_eligible": False,
        "deployment_holdout_required": True,
        "energy_frontier_measurement_required": True,
    }


def _attempt_audit(output: Path) -> dict[str, Any]:
    try:
        summary = v1._attempt_ledger_summary(output)
        ledger = v1._attempt_ledger(output)
    except v1.TrainingDebtError as exc:
        raise TrainingDebtV2Error(str(exc)) from exc
    excluded_events = {
        "attempt_failed",
        "interrupted_attempt_recovered",
        "attempt_staging_discarded",
    }
    attempts: dict[str, dict[str, Any]] = {}
    for event in ledger["events"]:
        if not isinstance(event, dict) or event.get("event") not in excluded_events:
            continue
        attempt_id = event.get("attempt_id")
        architecture = event.get("architecture")
        if not isinstance(attempt_id, str) or architecture not in {"am", "gnn"}:
            raise TrainingDebtV2Error("excluded attempt ledger event is malformed")
        attempt = attempts.setdefault(
            attempt_id, {"architecture": architecture, "known_values": []}
        )
        if attempt["architecture"] != architecture:
            raise TrainingDebtV2Error("attempt architecture changed in the ledger")
        value = event.get("known_excluded_component_energy_j")
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise TrainingDebtV2Error("known excluded attempt energy is invalid")
        attempt["known_values"].append(float(value))

    known_by_architecture = {"am": 0.0, "gnn": 0.0}
    known_count_by_architecture = {"am": 0, "gnn": 0}
    unknown_count_by_architecture = {"am": 0, "gnn": 0}
    for attempt in attempts.values():
        architecture = str(attempt["architecture"])
        values = attempt["known_values"]
        if values:
            if any(
                not math.isclose(value, values[0], rel_tol=1e-12, abs_tol=1e-9)
                for value in values[1:]
            ):
                raise TrainingDebtV2Error("excluded attempt energy changed in the ledger")
            known_by_architecture[architecture] += float(values[0])
            known_count_by_architecture[architecture] += 1
        else:
            unknown_count_by_architecture[architecture] += 1
    known_excluded = sum(known_by_architecture.values())
    return {
        **summary,
        "known_excluded_component_energy_j": known_excluded,
        "known_excluded_component_energy_j_by_architecture": known_by_architecture,
        "known_excluded_attempt_count": sum(known_count_by_architecture.values()),
        "known_excluded_attempt_count_by_architecture": known_count_by_architecture,
        "unknown_excluded_attempt_count": sum(unknown_count_by_architecture.values()),
        "unknown_excluded_attempt_count_by_architecture": unknown_count_by_architecture,
        "unknown_excluded_energy_is_not_imputed": True,
        "known_interrupted_energy_is_reported_but_not_in_primary_recipe_debt": True,
    }


def _write_checksums(output: Path) -> str:
    return v1._write_checksums(output)


def _validate_checksums(output: Path) -> None:
    try:
        v1._validate_checksums(output)
    except v1.TrainingDebtError as exc:
        raise TrainingDebtV2Error(str(exc)) from exc


def _validate_completed_output(output: Path, state: dict[str, Any]) -> None:
    summary_path = output / "training-energy-summary.json"
    manifest_path = output / "manifest.json"
    checksums_path = output / "SHA256SUMS"
    if (
        state.get("status") != COMPLETE_STATUS
        or state.get("summary_sha256") != _sha256_file(summary_path)
        or state.get("manifest_sha256") != _sha256_file(manifest_path)
        or state.get("checksums_sha256") != _sha256_file(checksums_path)
    ):
        raise TrainingDebtV2Error("completed v2 anchors changed")
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    entries = manifest.get("checkpoint_entries")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != COMPLETE_STATUS
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != COMPLETE_STATUS
        or manifest.get("training_energy_summary_sha256") != state["summary_sha256"]
        or entries != summary.get("checkpoint_entries")
        or not isinstance(entries, list)
        or len(entries) != 10
        or manifest.get("final_holdout_used") is not False
        or manifest.get("carbon_was_computed") is not False
        or manifest.get("aet_was_computed") is not False
    ):
        raise TrainingDebtV2Error("completed v2 manifest changed")
    _validate_checksums(output)


def _reconcile_attempt_ledger_v2(output: Path, completed: Mapping[str, dict[str, Any]]) -> None:
    payload = v1._attempt_ledger(output)
    for cell_id, result in completed.items():
        architecture = result.get("architecture")
        seed = result.get("training_seed")
        if not isinstance(architecture, str) or not isinstance(seed, int):
            raise TrainingDebtV2Error(f"completed v2 cell metadata is malformed: {cell_id}")
        attempt_id, is_open = v1._latest_attempt_for_cell(
            payload, architecture=architecture, seed=seed
        )
        if attempt_id is not None and is_open:
            v1._append_attempt_event(
                output,
                {
                    "event": "completed_attempt_recovered",
                    "attempt_id": attempt_id,
                    "architecture": architecture,
                    "training_seed": seed,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "energy_status": "training_and_selection_complete",
                    "predeployment_energy_debt_j": result["predeployment_energy_debt_j"],
                    "primary_recipe_debt_eligible": True,
                },
            )


def _finalize(
    recipe: TrainingDebtV2Recipe,
    output: Path,
    state: dict[str, Any],
    completed: Mapping[str, dict[str, Any]],
) -> TrainingDebtV2Result:
    _reconcile_attempt_ledger_v2(output, completed)
    attempt_audit = _attempt_audit(output)
    architecture_summaries: dict[str, dict[str, Any]] = {}
    checkpoint_entries: list[dict[str, Any]] = []
    for architecture in recipe.architectures:
        results = [
            completed[_cell_id(architecture.architecture_id, seed)]
            for seed in recipe.training.seeds
        ]
        architecture_summaries[architecture.architecture_id] = _architecture_summary(
            recipe, architecture, results
        )
        checkpoint_entries.extend(result["checkpoint"] for result in results)
    if len(checkpoint_entries) != 10:
        raise TrainingDebtV2Error("v2 manifest must expose exactly ten selected checkpoints")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": CLASSIFICATION,
        "seed_count_per_architecture": len(recipe.training.seeds),
        "measured_training_run_count": len(completed),
        "training_debt_definition": (
            "architecture-specific mean training plus prospective checkpoint-selection "
            "component energy for one independently trained model"
        ),
        "recipe_training_energy_j_mean_includes_checkpoint_selection": True,
        "architectures": architecture_summaries,
        "checkpoint_entries": checkpoint_entries,
        "attempt_audit": attempt_audit,
        "selection_source": {
            "split": "development",
            "content_sha256": recipe.corpus_content_sha256,
            "reference_sha256": recipe.reference.sha256,
            "final_holdout_used": False,
        },
        "study_debt": {
            "status": "measured_for_this_campaign_only",
            "definition": "sum of all five training plus selection cells per architecture",
            "included_in_single_model_recipe_debt": False,
        },
        "whole_system_energy": False,
        "carbon_accounting": "none",
        "aet_computed": False,
    }
    _atomic_write_json(output / "training-energy-summary.json", summary)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": CLASSIFICATION,
        "recipe_sha256": state["recipe_sha256"],
        "base_recipe_path": "base-recipe.yaml",
        "base_recipe_sha256": state["base_recipe_sha256"],
        "source_receipt_path": "source-receipt.json",
        "source_receipt_sha256": state["source_receipt_sha256"],
        "git_sha": state["git_sha"],
        "runtime_identity": state["runtime_identity"],
        "architectures": architecture_summaries,
        "checkpoint_entries": checkpoint_entries,
        "attempt_audit": attempt_audit,
        "training_energy_summary_path": "training-energy-summary.json",
        "training_energy_summary_sha256": _sha256_file(output / "training-energy-summary.json"),
        "completed_cells": list(completed),
        "preflight_evidence": state["preflight_evidence"],
        "selection_source_split": "development",
        "final_holdout_used": False,
        "deployment_holdout_required": True,
        "whole_system_energy": False,
        "carbon_was_computed": False,
        "aet_was_computed": False,
    }
    _atomic_write_json(output / "manifest.json", manifest)
    checksums_sha = _write_checksums(output)
    state.update(
        {
            "status": COMPLETE_STATUS,
            "completed_at": datetime.now(UTC).isoformat(),
            "completed_cells": list(completed),
            "remaining_cells": [],
            "summary_sha256": manifest["training_energy_summary_sha256"],
            "manifest_sha256": _sha256_file(output / "manifest.json"),
            "checksums_sha256": checksums_sha,
        }
    )
    if state["invocations"] and state["invocations"][-1].get("ended_at") is None:
        state["invocations"][-1]["ended_at"] = state["completed_at"]
        state["invocations"][-1]["outcome"] = COMPLETE_STATUS
    _atomic_write_json(output / "run-state.json", state)
    return TrainingDebtV2Result(
        path=output,
        status=COMPLETE_STATUS,
        complete=True,
        completed_cells=tuple(completed),
        remaining_cells=(),
        manifest_path=output / "manifest.json",
        manifest_sha256=state["manifest_sha256"],
    )


def execute_training_debt_v2(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    resume: bool = True,
) -> TrainingDebtV2Result:
    try:
        root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
        source_recipe = Path(recipe_path).resolve(strict=True)
    except OSError as exc:
        raise TrainingDebtV2QualificationError(f"workspace or recipe unavailable: {exc}") from exc
    if root != Path.cwd().resolve(strict=True):
        raise TrainingDebtV2QualificationError("workspace_root must be the current repository")
    try:
        source_recipe.relative_to(root)
    except ValueError as exc:
        raise TrainingDebtV2QualificationError("recipe must be inside the repository") from exc
    try:
        recipe = load_aet_training_debt_v2_recipe(source_recipe)
    except TrainingDebtV2RecipeValidationError as exc:
        raise TrainingDebtV2QualificationError(
            f"training-debt-v2 recipe is invalid: {exc}"
        ) from exc
    qualification = runtime_qualification(
        recipe, repository_root=root, active_architecture_probe=True
    )
    if not qualification["ready_to_execute"]:
        raise TrainingDebtV2QualificationError("; ".join(qualification["errors"]))
    try:
        attestation = v1._attestation(recipe)  # type: ignore[arg-type]
        runtime_identity = v1._configure_runtime(recipe)  # type: ignore[arg-type]
    except v1.TrainingDebtQualificationError as exc:
        raise TrainingDebtV2QualificationError(str(exc)) from exc
    source_receipt = _source_receipt(recipe, root)
    corpus, reference, reference_sha = _load_selection_source(recipe, root)
    campaign_started = time.perf_counter()
    output_target = quality._safe_output_target(root, recipe.output_root)
    with quality._output_lock(output_target):
        output, state = _prepare_output(
            source_recipe,
            recipe,
            root,
            resume=resume if output_target.exists() else False,
            runtime_identity=runtime_identity,
            source_receipt=source_receipt,
        )
        if state.get("status") == COMPLETE_STATUS:
            completed = _completed_prefix(recipe, output, corpus=corpus, reference=reference)
            _validate_completed_output(output, state)
            return TrainingDebtV2Result(
                path=output,
                status=COMPLETE_STATUS,
                complete=True,
                completed_cells=tuple(completed),
                remaining_cells=(),
                manifest_path=output / "manifest.json",
                manifest_sha256=_sha256_file(output / "manifest.json"),
            )
        if state.get("status") != INCOMPLETE_STATUS:
            raise TrainingDebtV2Error("v2 run state has an unknown status")
        _recover_staging(output)
        preflight = _preserve_preflight(recipe, root, output)
        state.setdefault("preflight_evidence", [])
        if preflight not in state["preflight_evidence"]:
            state["preflight_evidence"].append(preflight)
        state["invocations"].append(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "attestation": attestation,
                "preflight_evidence": preflight,
                "ended_at": None,
                "outcome": None,
            }
        )
        _atomic_write_json(output / "run-state.json", state)
        completed = _completed_prefix(recipe, output, corpus=corpus, reference=reference)
        _reconcile_attempt_ledger_v2(output, completed)
        expected_ids = [
            _cell_id(architecture.architecture_id, seed)
            for architecture, seed in _expected_cells(recipe)
        ]
        for architecture, seed in _expected_cells(recipe):
            cell_id = _cell_id(architecture.architecture_id, seed)
            if cell_id in completed:
                continue
            if time.perf_counter() - campaign_started > recipe.maximum_campaign_walltime_s:
                raise TrainingDebtV2Error("v2 campaign exceeded its maximum walltime")
            completed[cell_id] = _execute_cell(
                recipe,
                architecture,
                output=output,
                state=state,
                seed=seed,
                attestation=attestation,
                corpus=corpus,
                reference=reference,
                reference_sha256=reference_sha,
            )
            state["completed_cells"] = list(completed)
            state["remaining_cells"] = [item for item in expected_ids if item not in completed]
            _atomic_write_json(output / "run-state.json", state)
        return _finalize(recipe, output, state, completed)


def estimate_training_debt_v2(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run unmeasured CUDA probes and estimate remaining 100-epoch time."""

    try:
        root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
        source_recipe = Path(recipe_path).resolve(strict=True)
        source_recipe.relative_to(root)
    except (OSError, ValueError) as exc:
        raise TrainingDebtV2QualificationError(f"workspace or recipe unavailable: {exc}") from exc
    try:
        recipe = load_aet_training_debt_v2_recipe(source_recipe)
    except TrainingDebtV2RecipeValidationError as exc:
        raise TrainingDebtV2QualificationError(
            f"training-debt-v2 recipe is invalid: {exc}"
        ) from exc
    qualification = runtime_qualification(
        recipe, repository_root=root, active_architecture_probe=True
    )
    if not qualification["ready_to_execute"]:
        raise TrainingDebtV2QualificationError("; ".join(qualification["errors"]))
    output = quality._safe_output_target(root, recipe.output_root)
    present: set[str] = set()
    missing_seen = False
    for architecture, seed in _expected_cells(recipe):
        cell_id = _cell_id(architecture.architecture_id, seed)
        path = output / _cell_relative(architecture.architecture_id, seed) / "seed-result.json"
        if path.is_file():
            if missing_seen:
                raise TrainingDebtV2Error("completed v2 cells are not an ordered prefix")
            present.add(cell_id)
        else:
            missing_seen = True
    checks = qualification["architecture_checks"]
    estimates: dict[str, dict[str, Any]] = {}
    remaining_point = 0.0
    remaining_lower = 0.0
    remaining_upper = 0.0
    for architecture in recipe.architectures:
        completed_count = sum(
            _cell_id(architecture.architecture_id, seed) in present
            for seed in recipe.training.seeds
        )
        remaining = len(recipe.training.seeds) - completed_count
        point = float(
            checks[architecture.architecture_id]["estimated_seed_walltime_s_from_am_anchor"]
        )
        if architecture.architecture_id == "am":
            lower, upper = 3506.0 * 2.5, 3553.0 * 2.5
            point = 3526.0 * 2.5
            basis = "observed 40-epoch AM host range scaled linearly to 100 epochs"
        else:
            lower, upper = point * 0.75, point * 1.5
            basis = "CUDA GNN/AM step ratio applied to the scaled AM host anchor"
        # Greedy selection is short relative to training.  It is reported as a
        # range rather than hidden inside the training-step extrapolation.
        selection_point = 30.0
        selection_lower = 5.0
        selection_upper = 180.0
        estimates[architecture.architecture_id] = {
            "completed_seed_count": completed_count,
            "remaining_seed_count": remaining,
            "seconds_per_training_seed_point": point,
            "seconds_per_training_seed_lower": lower,
            "seconds_per_training_seed_upper": upper,
            "seconds_per_checkpoint_selection_point": selection_point,
            "seconds_per_checkpoint_selection_lower": selection_lower,
            "seconds_per_checkpoint_selection_upper": selection_upper,
            "seconds_per_complete_cell_point": point + selection_point,
            "seconds_per_complete_cell_lower": lower + selection_lower,
            "seconds_per_complete_cell_upper": upper + selection_upper,
            "estimation_basis": basis,
            "median_optimizer_step_s": checks[architecture.architecture_id][
                "median_optimizer_step_s"
            ],
            "parameter_count": checks[architecture.architecture_id]["parameter_count"],
            "backend_metadata": checks[architecture.architecture_id]["backend_metadata"],
        }
        remaining_point += remaining * (point + selection_point)
        remaining_lower += remaining * (lower + selection_lower)
        remaining_upper += remaining * (upper + selection_upper)
    return {
        "schema_version": ESTIMATE_SCHEMA,
        "status": "ready",
        "generated_at": datetime.now(UTC).isoformat(),
        "classification": {
            "energy_measurement": "none",
            "included_in_training_debt": False,
            "purpose": "pre_attestation_eta_and_architecture_qualification",
        },
        "cuda_probe": {
            "real_forward_backward": True,
            "warmup_steps_per_architecture": 1,
            "timed_steps_per_architecture": recipe.eta_calibration_steps,
            "selected_gpu_index": recipe.gpu_index,
            "architecture_qualification": checks,
        },
        "architectures": estimates,
        "completed_cells": sorted(present),
        "remaining_complete_cell_walltime_s_point": remaining_point,
        "remaining_complete_cell_walltime_s_lower": remaining_lower,
        "remaining_complete_cell_walltime_s_upper": remaining_upper,
        "maximum_campaign_walltime_s": recipe.maximum_campaign_walltime_s,
        "selection_source_split": "development",
        "final_holdout_used": False,
        "estimate_is_not_an_energy_measurement": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--no-resume", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.estimate_output is not None and not args.estimate_only:
            raise TrainingDebtV2QualificationError("--estimate-output requires --estimate-only")
        if args.estimate_only:
            estimate = estimate_training_debt_v2(args.recipe)
            if args.estimate_output is not None:
                root = Path.cwd().resolve(strict=True)
                destination = args.estimate_output.resolve(strict=False)
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise TrainingDebtV2QualificationError(
                        "--estimate-output must remain inside the repository"
                    ) from exc
                _atomic_write_json(destination, estimate)
            print(json.dumps(estimate, indent=2, sort_keys=True))
            return 0
        result = execute_training_debt_v2(args.recipe, resume=args.resume or not args.no_resume)
    except TrainingDebtV2QualificationError as exc:
        print(f"training debt v2 not qualified: {exc}", file=sys.stderr, flush=True)
        return 2
    except (TrainingDebtV2Error, quality.QualityPilotError) as exc:
        print(f"training debt v2 failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "complete": result.complete,
                "path": result.path.as_posix(),
                "completed_cells": list(result.completed_cells),
                "remaining_cells": list(result.remaining_cells),
                "manifest": result.manifest_path.as_posix() if result.manifest_path else None,
                "manifest_sha256": result.manifest_sha256,
                "classification": CLASSIFICATION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TrainingDebtV2Error",
    "TrainingDebtV2QualificationError",
    "TrainingDebtV2Result",
    "architecture_recipe_sha256",
    "estimate_training_debt_v2",
    "execute_training_debt_v2",
    "model_state_sha256",
]
